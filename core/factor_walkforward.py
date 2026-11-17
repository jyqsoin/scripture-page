# -*- coding: utf-8 -*-
"""
样本外分段回测（V1.5）。

把检验区间切成若干等长测试段，每段只使用「该段开始之前」的数据筛选因子、
再用筛选结果在该段内调仓回测，最后把各段净值首尾相接，得到一条完全样本外的净值曲线。

流程（每个测试段）：
    1. 训练窗口 = [段开始 - train_months, 段开始)，只在此窗口算 IC/显著性/相关性；
    2. judge_factors(use_validation=False) 判定方向，deduplicate 去冗余；
    3. 用入选因子在测试段内调用 run_factor_backtest 调仓回测；
    4. 记录入选因子、方向、区间绩效，拼接净值。

防前视约束：
- 因子筛选只使用测试段开始之前的截面与未来收益，绝不使用测试段内部或之后的数据；
- 回测持仓收益严格从调仓日收盘到下一调仓日收盘；
- 分段边界处丢弃 1 个调仓周期的跨段收益，基准同口径处理，相对比较不受影响。

仅研究用途，不构成投资建议。
"""

import sys
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.factor_engine import FACTOR_REGISTRY
from core.factor_analysis import get_rebalance_dates
from core.factor_screen import (
    prepare_panels, build_cross_sections, compute_details, judge_factors,
    mean_correlation,
)
from core.factor_process import deduplicate
from core.factor_backtest import run_factor_backtest

DISCLAIMER = "仅研究用途，不构成投资建议"


# ---------------------------------------------------------------------------
# 日期工具
# ---------------------------------------------------------------------------

def add_months(date_str, months):
    """在 YYYY-MM-DD 上加减整月（日期取 28 日以内，避免月末溢出）。"""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    total = d.year * 12 + (d.month - 1) + int(months)
    year, month = divmod(total, 12)
    month += 1
    day = min(d.day, 28)
    return datetime(year, month, day).strftime("%Y-%m-%d")


def plan_segments(start_date, end_date, test_months=6):
    """把 [start_date, end_date] 切成等长测试段，返回 [(start, end), ...]（左闭右开）。
    末尾不足半个测试段的碎片会并入上一段，避免出现无法回测的区间。"""
    segments = []
    current = start_date
    while current < end_date:
        nxt = add_months(current, test_months)
        if nxt > end_date:
            nxt = end_date
        segments.append([current, nxt])
        current = nxt
    if len(segments) >= 2:
        tail_start = datetime.strptime(segments[-1][0], "%Y-%m-%d")
        tail_end = datetime.strptime(segments[-1][1], "%Y-%m-%d")
        if (tail_end - tail_start).days < 30 * test_months // 2:
            segments[-2][1] = segments[-1][1]
            segments.pop()
    return [tuple(s) for s in segments]


# ---------------------------------------------------------------------------
# 净值与绩效
# ---------------------------------------------------------------------------

def nav_metrics(nav, benchmark_nav, turnovers=None):
    """由净值序列计算绩效指标。"""
    if nav is None or len(nav) < 2:
        return {}
    nav = pd.Series(nav).astype(float)
    bench = pd.Series(benchmark_nav).astype(float)
    returns = nav.pct_change().dropna()
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    periods = len(nav)
    annual = (1 + total) ** (52.0 / periods) - 1 if periods > 0 else 0.0
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * math.sqrt(52)) if std > 0 else 0.0
    max_drawdown = float(abs((nav / nav.cummax() - 1).min()))
    bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1) if len(bench) else 0.0
    return {
        "total_return": total,
        "annual_return": float(annual),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "periods": periods,
        "benchmark_total_return": bench_total,
        "excess_return": total - bench_total,
    }


# ---------------------------------------------------------------------------
# 分段走查
# ---------------------------------------------------------------------------

def run_walkforward(codes=None, factor_names=None, start_date=None, end_date=None,
                    freq="W", forward_n=5, n_layers=5, top_n=30, method="equal",
                    train_months=24, test_months=6, min_abs_ic=0.02, min_abs_t=2.0,
                    corr_threshold=0.7, ic_lookback=52, fq="qfq", progress_cb=None):
    """
    样本外分段走查回测。

    参数：
        codes: 股票池，默认本地全部有 qfq 数据的股票
        factor_names: 候选因子，默认全部已注册因子
        start_date / end_date: 样本外检验区间
        freq: 调仓频率 "W" / "M"
        forward_n: 筛选时使用的未来收益天数
        n_layers: 分层数
        top_n: 每期持仓数
        method: 合成方法 "equal"（按筛选方向）或 "ic"（滚动 IC 加权）
        train_months: 每段开始前的训练窗口长度（月）
        test_months: 每个测试段长度（月）
        min_abs_ic / min_abs_t: 因子入选门槛
        corr_threshold: 冗余阈值
        ic_lookback: method="ic" 时的滚动 IC 窗口

    返回 dict：
        segments: 每段的训练区间、入选因子、方向、绩效
        nav / benchmark_nav: 拼接后的样本外净值
        metrics: 样本外整体绩效
        meta: 参数与区间
    """
    factor_names = factor_names or list(FACTOR_REGISTRY.keys())
    if codes is None:
        codes = [r["code"] for r in db.query_all(
            "SELECT DISTINCT code FROM daily_bars WHERE fq=? ORDER BY code", (fq,))]
    if not start_date or not end_date:
        raise ValueError("必须同时指定 start_date 与 end_date")

    panels, data_map = prepare_panels(codes, factor_names, fq=fq)
    usable = list(panels.keys())
    if not usable:
        raise ValueError("无可用日线数据")
    all_dates = get_rebalance_dates(usable, freq=freq, fq=fq)
    plan = plan_segments(start_date, end_date, test_months)

    segments = []
    chained = {}
    chained_bench = {}
    turnovers = []
    base_nav = 1.0
    base_bench = 1.0

    for idx, (seg_start, seg_end) in enumerate(plan, 1):
        if progress_cb:
            progress_cb(idx, len(plan), "分段 %s~%s" % (seg_start, seg_end))

        train_start = add_months(seg_start, -train_months)
        train_dates = [d for d in all_dates if train_start <= d < seg_start]
        seg_dates = [d for d in all_dates if seg_start <= d < seg_end]
        record = {
            "start": seg_start, "end": seg_end,
            "train_start": train_start,
            "train_periods": len(train_dates),
            "test_periods": len(seg_dates),
            "factors": [], "directions": {}, "status": "ok", "reason": "",
            "metrics": {},
        }

        # 训练窗口或测试段过短 → 跳过
        if len(train_dates) < 20:
            record.update(status="skipped", reason="训练窗口调仓日不足（%d）" % len(train_dates))
            segments.append(record)
            continue
        if len(seg_dates) < 3:
            record.update(status="skipped", reason="测试段调仓日不足（%d）" % len(seg_dates))
            segments.append(record)
            continue

        # 1) 仅用训练窗口筛选因子
        train_frames = build_cross_sections(panels, train_dates)
        details = compute_details(train_frames, factor_names, forward_n=forward_n,
                                  n_layers=n_layers)
        decisions = judge_factors(details, min_abs_ic=min_abs_ic, min_abs_t=min_abs_t,
                                  use_validation=False)
        corr = mean_correlation(train_frames, factor_names)
        candidates = [n for n in factor_names
                      if decisions[n]["status"] in ("keep", "reverse")]
        scores = {n: details[n]["train"]["t_stat"] for n in candidates}
        cand_directions = {n: decisions[n]["direction"] for n in candidates}
        sub_corr = corr.loc[candidates, candidates] if candidates else corr
        selected, _ = deduplicate(sub_corr, scores, directions=cand_directions,
                                  threshold=corr_threshold)

        selected = [n for n in factor_names if n in selected]
        directions = {n: decisions[n]["direction"] for n in selected}
        record["factors"] = selected
        record["directions"] = directions
        record["universe"] = len(usable)

        if not selected:
            record.update(status="empty", reason="训练窗口无因子通过筛选")
            segments.append(record)
            continue

        # 2) 用入选因子在测试段内回测
        seg_end_inclusive = seg_dates[-1]
        result = run_factor_backtest(
            codes=usable, factor_names=selected,
            start_date=seg_start, end_date=seg_end_inclusive,
            freq=freq, top_n=top_n, method=method,
            directions=directions if method == "equal" else None,
            ic_lookback=ic_lookback, fq=fq,
        )
        nav = result["nav"]
        bench = result["benchmark_nav"]
        turnover = float(np.mean(result["turnover"])) if len(result["turnover"]) else 0.0
        turnovers.append(turnover)

        for date, value in nav.items():
            chained[date] = base_nav * float(value)
        for date, value in bench.items():
            chained_bench[date] = base_bench * float(value)
        base_nav = chained[nav.index[-1]]
        base_bench = chained_bench[bench.index[-1]]

        record["metrics"] = result["metrics"]
        record["turnover"] = turnover
        record["nav_end"] = base_nav
        segments.append(record)

    if not chained:
        raise ValueError("所有分段都被跳过，无法形成样本外净值")

    nav_series = pd.Series(chained).sort_index()
    bench_series = pd.Series(chained_bench).sort_index()
    metrics = nav_metrics(nav_series, bench_series, turnovers)

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": len(usable),
            "candidates": len(factor_names),
            "start": start_date,
            "end": end_date,
            "freq": freq,
            "top_n": top_n,
            "method": method,
            "train_months": train_months,
            "test_months": test_months,
            "min_abs_ic": min_abs_ic,
            "min_abs_t": min_abs_t,
            "corr_threshold": corr_threshold,
            "disclaimer": DISCLAIMER,
        },
        "segments": segments,
        "nav": nav_series,
        "benchmark_nav": bench_series,
        "metrics": metrics,
    }


def print_report(result):
    """打印分段走查报告。"""
    meta = result["meta"]
    print("\n=== 样本外分段回测（V1.5）===")
    print("股票池 %d 只 | 候选因子 %d 个 | %s ~ %s | %s 频 | top%d | %s 合成 | 训练 %d 月 / 测试 %d 月" % (
        meta["codes"], meta["candidates"], meta["start"], meta["end"], meta["freq"],
        meta["top_n"], meta["method"], meta["train_months"], meta["test_months"]))
    print("-" * 112)
    print("%-21s %8s %8s %-30s %8s %7s %8s %8s" % (
        "测试区间", "训练期数", "测试期数", "入选因子(方向)", "年化", "夏普", "回撤", "超额"))
    print("-" * 112)
    for seg in result["segments"]:
        label = "%s~%s" % (seg["start"], seg["end"])
        factors = ", ".join("%s(%+d)" % (n, seg["directions"].get(n, 1))
                            for n in seg["factors"]) or "-"
        if seg["status"] != "ok":
            print("%-21s %8d %8d %-30s %s" % (
                label, seg["train_periods"], seg["test_periods"], factors, seg["reason"]))
            continue
        m = seg["metrics"]
        print("%-21s %8d %8d %-30s %7.2f%% %7.3f %7.2f%% %7.2f%%" % (
            label, seg["train_periods"], seg["test_periods"], factors,
            m["annual_return"] * 100, m["sharpe"], m["max_drawdown"] * 100,
            m["excess_return"] * 100))
    print("-" * 112)
    m = result["metrics"]
    print("样本外整体：总收益 %.2f%% | 年化 %.2f%% | 夏普 %.3f | 最大回撤 %.2f%% | 胜率 %.2f%% | 换手 %.2f%% | 基准 %.2f%% | 超额 %.2f%%" % (
        m["total_return"] * 100, m["annual_return"] * 100, m["sharpe"],
        m["max_drawdown"] * 100, m["win_rate"] * 100, m["avg_turnover"] * 100,
        m["benchmark_total_return"] * 100, m["excess_return"] * 100))


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()

    rows = db.query_all("SELECT DISTINCT code FROM daily_bars WHERE fq='qfq' ORDER BY code LIMIT 50")
    codes = [r["code"] for r in rows]
    print("测试股票数:", len(codes))

    names = ["ma5", "atr_pct", "volatility_20", "bias_20", "return_20",
             "price_position_60", "ret_1", "volume_trend_20"]

    # 1. 分段切分
    plan = plan_segments("2025-06-01", "2026-11-13", test_months=6)
    print("分段:", plan)
    assert plan[0][0] == "2025-06-01" and plan[-1][1] == "2026-11-13"
    for a, b in zip(plan, plan[1:]):
        assert a[1] == b[0], "分段必须连续且不重叠"
    assert add_months("2024-01-31", 1) == "2024-02-28"

    # 2. 走查回测
    result = run_walkforward(
        codes=codes, factor_names=names, start_date="2025-06-01", end_date="2026-11-13",
        freq="W", top_n=15, method="equal", train_months=18, test_months=6,
        min_abs_ic=0.02, min_abs_t=1.5,
    )
    print_report(result)

    # 3. 一致性校验：每个分段的入选因子只依赖该段之前的训练窗口
    for seg in result["segments"]:
        if seg["status"] != "ok":
            continue
        assert seg["train_start"] < seg["start"]
        assert seg["factors"], "分段入选因子不应为空"
        for name in seg["factors"]:
            assert seg["directions"][name] in (1, -1)

    # 4. 净值连续性：拼接后单调索引且无重复日期
    nav = result["nav"]
    assert nav.index.is_monotonic_increasing
    assert len(nav.index) == len(set(nav.index))
    assert nav.iloc[0] > 0 and result["benchmark_nav"].iloc[0] > 0
    assert set(result["metrics"].keys()) >= {"annual_return", "sharpe", "excess_return"}

    print("\nfactor_walkforward.py 自测通过；%s" % DISCLAIMER)
