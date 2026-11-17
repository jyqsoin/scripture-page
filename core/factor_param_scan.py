# -*- coding: utf-8 -*-
"""Step I1 参数敏感性复核（V1.5-I1 补充）。

背景：I1 已在「一致率 0.7 / 4 子窗 / 测试段 6 月 / top30 / 周频」下得到
「方向稳定但仍不稳健」的结论（5 段样本外全线负超额，稳定性门槛造成空仓）。
本脚本只做参数扫描复核，不改动 factor_screen / factor_backtest 的任何逻辑：
- 固定 V1.4 的 24 因子池；
- 扫描 一致率阈值(0.5/0.6/0.7) × 子窗数(3/4) × 测试段长度(12/24 月)
       × top_n(20/50) × 调仓频率(周/双周/月)；
- 每个组合按「只用训练窗口筛选 + 测试段回测」的样本外流程执行；
- 输出入选因子数、空仓段数、各段超额/夏普/换手、方向翻转次数；
- 判定：放宽组合（一致率 0.5 + top50 + 月频）若 ≥2 段超额为正且无空仓段
       → 「参数过严」并给出建议参数；否则 → 「因子源失效」建议转 Smart Beta。

防前视约束：
- 因子方向、稳定性、冗余剔除只使用测试段开始之前的训练窗口；
- 回测持仓收益严格从调仓日收盘到下一调仓日收盘；
- 空仓段（训练窗口无因子入选）不建仓，基准同步不计入链式净值。

仅研究用途，不构成投资建议。
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.factor_engine import FACTOR_REGISTRY
from core.factor_screen import (
    prepare_panels, build_cross_sections, compute_details, judge_factors,
    mean_correlation, compute_direction_stability, SCREEN_DIR,
)
from core.factor_process import deduplicate
from core.factor_backtest import (
    run_factor_backtest, PERIODS_PER_YEAR, _rebalance_dates,
)
from core.factor_walkforward import add_months, plan_segments

DISCLAIMER = "仅研究用途，不构成投资建议"

# 固定检验区间与训练窗口（沿用 I1）
START, END = "2024-05-01", "2026-11-13"
TRAIN_MONTHS = 24
FORWARD_N = 5
MIN_ABS_IC, MIN_ABS_T = 0.02, 2.0
CORR_THRESHOLD = 0.7

# 扫描网格
CONSISTENCY_GRID = (0.5, 0.6, 0.7)
WINDOW_GRID = (3, 4)
TEST_MONTH_GRID = (12, 24)
TOP_N_GRID = (20, 50)
FREQ_GRID = ("W", "2W", "M")

# 判定「参数过严」时使用的放宽组合
RELAX_CONSISTENCY, RELAX_TOP_N, RELAX_FREQ = 0.5, 50, "M"


# ---------------------------------------------------------------------------
# 单段筛选与回测
# ---------------------------------------------------------------------------

def screen_segment(panels, factors, train_dates, min_consistency, n_windows):
    """只用训练窗口筛选因子，返回 (入选因子, 方向 dict, 一致率 dict)。"""
    train_frames = build_cross_sections(panels, train_dates)
    details = compute_details(train_frames, factors, forward_n=FORWARD_N,
                              n_layers=5, include_decay=False,
                              include_layers=False)
    stability = compute_direction_stability(train_frames, factors,
                                            n_windows=n_windows,
                                            forward_n=FORWARD_N)
    decisions = judge_factors(details, min_abs_ic=MIN_ABS_IC,
                              min_abs_t=MIN_ABS_T, use_validation=False,
                              stability=stability,
                              min_consistency=min_consistency)
    corr = mean_correlation(train_frames, factors)
    candidates = [n for n in factors
                  if decisions[n]["status"] in ("keep", "reverse")]
    scores = {n: details[n]["train"]["t_stat"] for n in candidates}
    directions = {n: decisions[n]["direction"] for n in candidates}
    if candidates:
        selected, _ = deduplicate(corr.loc[candidates, candidates], scores,
                                  directions=directions,
                                  threshold=CORR_THRESHOLD)
    else:
        selected = []
    selected = [n for n in factors if n in selected]
    return selected, {n: decisions[n]["direction"] for n in selected}


def run_combo(panels, factors, usable, consistency, n_windows, test_months,
              top_n, freq, backtest_cache):
    """跑一个参数组合，返回该组合的分段结果与汇总。"""
    all_dates = _rebalance_dates(usable, freq=freq, fq="qfq")
    plan = plan_segments(START, END, test_months)
    per_year = PERIODS_PER_YEAR[freq]
    segments = []
    chained, chained_bench = {}, {}
    base_nav = base_bench = 1.0
    for seg_start, seg_end in plan:
        train_start = add_months(seg_start, -TRAIN_MONTHS)
        train_dates = [d for d in all_dates if train_start <= d < seg_start]
        seg_dates = [d for d in all_dates if seg_start <= d < seg_end]
        rec = {"start": seg_start, "end": seg_end,
               "train_periods": len(train_dates),
               "test_periods": len(seg_dates),
               "factors": [], "directions": {}, "status": "ok",
               "reason": "", "excess": None, "sharpe": None,
               "turnover": None, "annual": None}
        if len(train_dates) < 20:
            rec.update(status="skipped", reason="训练窗口调仓日不足（%d）" % len(train_dates))
            segments.append(rec)
            continue
        if len(seg_dates) < 3:
            rec.update(status="skipped", reason="测试段调仓日不足（%d）" % len(seg_dates))
            segments.append(rec)
            continue

        selected, directions = screen_segment(panels, factors, train_dates,
                                              consistency, n_windows)
        rec["factors"] = selected
        rec["directions"] = directions
        if not selected:
            rec.update(status="empty", reason="训练窗口无因子通过门槛")
            segments.append(rec)
            continue

        # 回测结果缓存：同一 (频率, top_n, 区间, 因子+方向) 只算一次
        key = (freq, top_n, seg_start, seg_end, tuple(selected),
               tuple(sorted(directions.items())))
        if key in backtest_cache:
            metrics = backtest_cache[key]
        else:
            res = run_factor_backtest(
                codes=usable, factor_names=selected, start_date=seg_start,
                end_date=seg_dates[-1], freq=freq, top_n=top_n,
                method="equal", directions=directions, fq="qfq")
            metrics = res["metrics"]
            backtest_cache[key] = metrics
            # 缓存净值，便于链式拼接
            backtest_cache[key + ("nav",)] = res["nav"]
            backtest_cache[key + ("bench",)] = res["benchmark_nav"]

        nav = backtest_cache[key + ("nav",)]
        bench = backtest_cache[key + ("bench",)]
        for d, v in nav.items():
            chained[d] = base_nav * float(v)
        for d, v in bench.items():
            chained_bench[d] = base_bench * float(v)
        base_nav = chained[nav.index[-1]]
        base_bench = chained_bench[bench.index[-1]]

        rec["excess"] = float(metrics["excess_return"])
        rec["sharpe"] = float(metrics["sharpe"])
        rec["turnover"] = float(metrics.get("total_turnover") or 0.0)
        rec["annual"] = float(metrics["annual_return"])
        segments.append(rec)

    ok = [s for s in segments if s["status"] == "ok"]
    empty = [s for s in segments if s["status"] == "empty"]
    skipped = [s for s in segments if s["status"] == "skipped"]

    # 拼接净值后按该频率周期数重算整体绩效
    if chained:
        nav_s = pd.Series(chained).sort_index()
        bench_s = pd.Series(chained_bench).sort_index()
        rets = nav_s.pct_change().dropna()
        total = float(nav_s.iloc[-1] / nav_s.iloc[0] - 1)
        annual = (1 + total) ** (per_year / max(len(nav_s), 1)) - 1
        std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
        sharpe = float(rets.mean() / std * math.sqrt(per_year)) if std > 0 else 0.0
        bench_total = float(bench_s.iloc[-1] / bench_s.iloc[0] - 1)
    else:
        total = annual = sharpe = bench_total = 0.0

    # 方向翻转次数：相邻有效段的「代表方向」符号变化（跳过空仓段）
    seq = [sum(s["directions"].values()) for s in ok if s["factors"]]
    seq = [1 if v > 0 else (-1 if v < 0 else 0) for v in seq]
    flips = sum(1 for a, b in zip(seq, seq[1:]) if a and b and a != b)

    return {
        "min_consistency": consistency,
        "n_windows": n_windows,
        "test_months": test_months,
        "top_n": top_n,
        "freq": freq,
        "segments_total": len(segments),
        "segments_ok": len(ok),
        "empty_segments": len(empty),
        "skipped_segments": len(skipped),
        "selected_counts": [len(s["factors"]) for s in ok],
        "excess_list": [s["excess"] for s in ok],
        "sharpe_list": [s["sharpe"] for s in ok],
        "turnover_total": float(sum(s["turnover"] for s in ok)),
        "pos_excess_segments": sum(1 for s in ok if (s["excess"] or 0) > 0),
        "annual_return": float(annual),
        "sharpe": float(sharpe),
        "total_return": float(total),
        "benchmark_total": float(bench_total),
        "excess_return": float(total - bench_total),
        "flip_count": int(flips),
        "direction_sequence": seq,
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# 全网格扫描与报告
# ---------------------------------------------------------------------------

def scan(codes=None, factors=None, start_date=START, end_date=END,
         consistency_grid=CONSISTENCY_GRID, window_grid=WINDOW_GRID,
         test_month_grid=TEST_MONTH_GRID, top_n_grid=TOP_N_GRID,
         freq_grid=FREQ_GRID, log=print):
    """执行参数网格扫描，返回 (组合结果列表, 元信息)。"""
    global START, END
    START, END = start_date, end_date
    factors = factors or list(FACTOR_REGISTRY.keys())
    if codes is None:
        codes = [r["code"] for r in db.query_all(
            "SELECT DISTINCT code FROM daily_bars WHERE fq='qfq' ORDER BY code LIMIT 300")]
    panels, _ = prepare_panels(codes, factors, fq="qfq")
    usable = list(panels.keys())
    if not usable:
        raise ValueError("无可用日线数据")

    # 同一 (频率, 一致率, 子窗数, 测试段长度) 的分段筛选只做一次
    screen_cache = {}
    backtest_cache = {}
    results = []
    total = (len(consistency_grid) * len(window_grid) * len(test_month_grid)
             * len(top_n_grid) * len(freq_grid))
    done = 0
    for consistency in consistency_grid:
        for n_windows in window_grid:
            for test_months in test_month_grid:
                for top_n in top_n_grid:
                    for freq in freq_grid:
                        done += 1
                        t0 = time.time()
                        item = run_combo(panels, factors, usable, consistency,
                                         n_windows, test_months, top_n, freq,
                                         backtest_cache)
                        item["elapsed_sec"] = round(time.time() - t0, 2)
                        results.append(item)
                        log("[%2d/%d] 一致率%.1f 窗%d 测试%2d月 top%-2d %-2s | "
                            "有效段%d 空仓%d 正超额%d 翻转%d | 超额%+.2f%%" % (
                                done, total, consistency, n_windows, test_months,
                                top_n, freq, item["segments_ok"],
                                item["empty_segments"],
                                item["pos_excess_segments"], item["flip_count"],
                                item["excess_return"] * 100))
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "codes": len(usable),
        "factors": len(factors),
        "start": start_date,
        "end": end_date,
        "train_months": TRAIN_MONTHS,
        "grid": {
            "min_consistency": list(consistency_grid),
            "n_windows": list(window_grid),
            "test_months": list(test_month_grid),
            "top_n": list(top_n_grid),
            "freq": list(freq_grid),
        },
        "disclaimer": DISCLAIMER,
    }
    return results, meta


def judge(results):
    """按放宽组合判定「参数过严」还是「因子源失效」。"""
    relaxed = [r for r in results
               if r["min_consistency"] == RELAX_CONSISTENCY
               and r["top_n"] == RELAX_TOP_N and r["freq"] == RELAX_FREQ]
    passed = [r for r in relaxed
              if r["pos_excess_segments"] >= 2 and r["empty_segments"] == 0]
    if passed:
        best = max(passed, key=lambda r: r["sharpe"])
        return {
            "verdict": "参数过严",
            "detail": "放宽组合存在达标参数：一致率 %.1f / 子窗 %d / 测试段 %d 月 / top%d / %s" % (
                best["min_consistency"], best["n_windows"],
                best["test_months"], best["top_n"], best["freq"]),
            "suggest": {
                "min_consistency": best["min_consistency"],
                "n_windows": best["n_windows"],
                "test_months": best["test_months"],
                "top_n": best["top_n"],
                "freq": best["freq"],
            },
        }
    return {
        "verdict": "因子源失效",
        "detail": "放宽组合（一致率 %.1f / top%d / %s）仍无法做到 ≥2 段超额为正且无空仓段" % (
            RELAX_CONSISTENCY, RELAX_TOP_N, RELAX_FREQ),
        "suggest": {"next_step": "转 I2-alt：评估 Smart Beta 固定暴露或引入基本面因子"},
    }


def save_reports(results, meta, path=None):
    """落盘复核报告 CSV + JSON，返回 (csv_path, json_path)。"""
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = SCREEN_DIR / ("i1_param_scan_%s_%s.csv" % (
            meta["codes"], meta["end"].replace("-", "")))
    csv_path = Path(path)
    rows = []
    max_seg = max((len(r["excess_list"]) for r in results), default=0)
    for r in results:
        row = {
            "min_consistency": r["min_consistency"],
            "n_windows": r["n_windows"],
            "test_months": r["test_months"],
            "top_n": r["top_n"],
            "freq": r["freq"],
            "segments_total": r["segments_total"],
            "segments_ok": r["segments_ok"],
            "empty_segments": r["empty_segments"],
            "selected_counts": "/".join(str(c) for c in r["selected_counts"]),
            "pos_excess_segments": r["pos_excess_segments"],
        }
        for i in range(max_seg):
            row["excess_seg%d" % (i + 1)] = (r["excess_list"][i]
                                             if i < len(r["excess_list"]) else None)
            row["sharpe_seg%d" % (i + 1)] = (r["sharpe_list"][i]
                                             if i < len(r["sharpe_list"]) else None)
        row.update({
            "annual_return": r["annual_return"],
            "sharpe": r["sharpe"],
            "excess_return": r["excess_return"],
            "turnover_total": r["turnover_total"],
            "flip_count": r["flip_count"],
            "direction_sequence": "/".join("%+d" % d for d in r["direction_sequence"]),
            "elapsed_sec": r["elapsed_sec"],
        })
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    verdict = judge(results)
    payload = {"meta": meta, "verdict": verdict, "items": results}
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    default=str), encoding="utf-8")
    return csv_path, json_path


def print_summary(results, verdict, top=12):
    """控制台摘要：重点看空仓与正超额段数，再列代表组合。"""
    frame = pd.DataFrame([{
        "consistency": r["min_consistency"], "windows": r["n_windows"],
        "test_m": r["test_months"], "top_n": r["top_n"], "freq": r["freq"],
        "ok": r["segments_ok"], "empty": r["empty_segments"],
        "pos": r["pos_excess_segments"], "flip": r["flip_count"],
        "sharpe": round(r["sharpe"], 3),
        "excess": round(r["excess_return"] * 100, 2),
        "turnover": round(r["turnover_total"], 2),
    } for r in results])
    print("\n=== 参数扫描摘要（共 %d 组）===" % len(results))
    print("空仓段为 0 的组合: %d 组" % int((frame["empty"] == 0).sum()))
    print("≥2 段正超额且无空仓: %d 组" % int(((frame["pos"] >= 2)
                                              & (frame["empty"] == 0)).sum()))
    print("\n代表组合（按夏普降序，仅列非空仓组）:")
    show = frame[frame["empty"] == 0].sort_values("sharpe", ascending=False).head(top)
    print(show.to_string(index=False) if len(show) else "  （无）")
    print("\n判定: %s" % verdict["verdict"])
    print("说明: %s" % verdict["detail"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="I1 参数敏感性复核")
    parser.add_argument("--codes", type=int, default=300, help="股票池数量")
    parser.add_argument("--selftest", action="store_true", help="小网格快速自测")
    args = parser.parse_args(argv)
    db.init_db()
    codes = [r["code"] for r in db.query_all(
        "SELECT DISTINCT code FROM daily_bars WHERE fq='qfq' ORDER BY code LIMIT ?",
        (args.codes,))]
    print(DISCLAIMER)
    print("股票池 %d 只 | 区间 %s ~ %s | 训练 %d 月" % (
        len(codes), START, END, TRAIN_MONTHS))

    if args.selftest:
        results, meta = scan(codes=codes[:60], consistency_grid=(0.5, 0.7),
                             window_grid=(3,), test_month_grid=(6,),
                             top_n_grid=(10,), freq_grid=("W",),
                             start_date="2025-06-01", end_date="2026-11-13")
        assert len(results) == 2, "组合数应为 2"
        for r in results:
            assert r["segments_total"] >= 1
            assert r["flip_count"] >= 0
            assert isinstance(r["excess_return"], float)
        verdict = judge(results)
        print_summary(results, verdict)
        print("\nfactor_param_scan.py 自测通过；%s" % DISCLAIMER)
        return

    t0 = time.time()
    results, meta = scan(codes=codes)
    csv_path, json_path = save_reports(results, meta)
    verdict = judge(results)
    print_summary(results, verdict)
    print("\n总耗时: %.1f 秒" % (time.time() - t0))
    print("复核 CSV: %s" % csv_path)
    print("复核 JSON: %s" % json_path)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
