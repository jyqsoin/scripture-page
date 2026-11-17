# -*- coding: utf-8 -*-
"""Smart Beta 弱段亏损归因（Step I2-alt-8）。

针对 I2-alt-6 分段验证中全线亏损的 S4（2025-11-28 ~ 2026-05-29），拆解：
1. 逐期收益：双因子、三因子等权、五因子与池内等权基准的差异；
2. 因子贡献：低价、低波、动量、低 PE、低 PB 五个单因子子组合的同期表现；
3. 风格分层：把池内股票按价格、波动率、动量、PE、PB 各分 5 组，
   检验“低价/低波/高动量/低估值”方向是否仍占优；
4. 个股拖累：五因子每期持仓相对基准的贡献，找出主要负贡献股票。

本地 stock_info 暂无行业字段，因此不做行业归因。
不修改既有回测核心；不接券商、不下单、不上传云端。
仅研究用途，不构成投资建议。
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.factor_analysis import load_all_daily, get_rebalance_dates
from core.smart_beta import (
    DISCLAIMER, DEFAULT_PCT, DEFAULT_VOL_WINDOW,
    DEFAULT_MOMENTUM_WINDOW, DEFAULT_MOMENTUM_SKIP,
    build_exposures, build_three_factor_exposures,
)
from core.smart_beta_backtest import (
    SMART_BETA_DIR, period_returns, equal_weight_return, performance_metrics,
)
from core.smart_beta_3factor_backtest import weighted_return
from core.smart_beta_fundamental_backtest import build_fundamental_composite
from core.valuation_data import load_valuation

S4_START = "2025-11-28"
S4_END = "2026-05-29"
SLEEVES = ("low_price", "low_vol", "momentum", "low_pe", "low_pb")
SLEEVE_LABELS = {
    "low_price": "低价", "low_vol": "低波", "momentum": "动量",
    "low_pe": "低PE", "low_pb": "低PB",
}
STYLE_DIMENSIONS = (
    ("price", "价格", "ascending"),      # Q1 最低价
    ("volatility", "波动率", "ascending"),  # Q1 最低波
    ("momentum", "动量", "ascending"),      # Q1 最弱动量
    ("pe_ttm", "PE", "ascending"),          # Q1 最低 PE
    ("pb_mrq", "PB", "ascending"),          # Q1 最低 PB
)


def bucket_returns(scores, returns, buckets=5):
    """按因子分数分位（Q1 最低分）统计下期等权收益。"""
    frame = pd.DataFrame({"score": scores, "ret": returns}).dropna()
    if len(frame) < buckets * 2:
        return []
    try:
        labels = pd.qcut(frame["score"], buckets, labels=False, duplicates="drop")
    except ValueError:
        return []
    rows = []
    for bucket in sorted(labels.dropna().unique()):
        subset = frame[labels == bucket]
        if subset.empty:
            continue
        rows.append({"bucket": int(bucket) + 1, "count": int(len(subset)),
                     "return": float(subset["ret"].mean())})
    return rows


def contribution_table(returns, weights, benchmark):
    """个股贡献 = 有效权重 × (个股收益 - 基准)；权重在有效样本内归一化。"""
    valid = returns.reindex(weights.index).notna() & (weights > 0)
    if not valid.any():
        return pd.DataFrame(columns=["weight", "return", "contribution"])
    weights = weights[valid] / weights[valid].sum()
    values = returns.reindex(weights.index)
    return pd.DataFrame({
        "weight": weights, "return": values,
        "contribution": weights * (values - benchmark),
    })


def judge_style(style_frame):
    """按各维度 Q1 与 Q5 的累计收益判断因子方向是否仍成立。"""
    verdicts = []
    for key, label, _ in STYLE_DIMENSIONS:
        frame = style_frame[style_frame["dimension"] == key]
        if frame.empty:
            continue
        pivot = frame.groupby(["bucket"])["return"].apply(
            lambda values: float(np.prod(1 + values) - 1))
        if 1 not in pivot.index or pivot.index.max() not in pivot.index:
            continue
        low_end, high_end = float(pivot.loc[1]), float(pivot.loc[pivot.index.max()])
        if key == "momentum":
            effective = high_end > low_end   # 高动量应占优
            favourable = "高动量组"
        elif key in ("pe_ttm", "pb_mrq"):
            effective = low_end > high_end   # 低估值应占优
            favourable = "低估值组"
        else:
            effective = low_end > high_end   # 低价、低波应占优
            favourable = "低分组"
        verdicts.append({
            "dimension": key, "label": label,
            "q1_return": low_end, "q5_return": high_end,
            "spread_q1_minus_q5": low_end - high_end,
            "effective": bool(effective), "favourable": favourable,
        })
    return verdicts


def run_attribution(
        codes=None, start_date=S4_START, end_date=S4_END,
        pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, fq="qfq", buckets=5):
    """逐期复现 S4 持仓并拆解亏损来源。"""
    if codes is None:
        codes = [row["code"] for row in db.query_all(
            "SELECT DISTINCT code FROM daily_bars WHERE fq=? ORDER BY code", (fq,))]
    data_map = load_all_daily(codes, fq=fq)
    if not data_map:
        raise ValueError("无可用本地日线数据")
    valuation_map = {code: load_valuation(code) for code in data_map}
    missing = [code for code, frame in valuation_map.items() if frame.empty]
    if missing:
        raise ValueError("以下股票缺少本地估值数据：%s" % ",".join(missing[:10]))

    dates = get_rebalance_dates(list(data_map), freq="M", fq=fq)
    dates = [date for date in dates if start_date <= date <= end_date]
    if len(dates) < 2:
        raise ValueError("调仓日不足，至少需要两个自然月末交易日")
    names = {row["code"]: row["name"] for row in db.query_all(
        "SELECT code, name FROM stock_info")}

    period_rows, contribution_rows, style_rows = [], [], []
    nav_map = {name: [1.0] for name in ("dual", "three_equal", "five_factor", "benchmark")}
    sleeve_nav = {name: [1.0] for name in SLEEVES}

    for i in range(len(dates) - 1):
        trade_date, next_date = dates[i], dates[i + 1]
        returns = period_returns(data_map, trade_date, next_date)
        dual = build_exposures(
            data_map, trade_date, pct=pct, vol_window=vol_window)
        three = build_three_factor_exposures(
            data_map, trade_date, pct=pct, vol_window=vol_window,
            momentum_window=momentum_window, momentum_skip=momentum_skip,
            mode="equal")
        five = build_fundamental_composite(
            data_map, valuation_map, trade_date, True, True, pct, vol_window,
            momentum_window, momentum_skip)

        benchmark = equal_weight_return(returns, list(data_map))
        dual_return = 0.5 * weighted_return(returns, pd.Series(
            1.0 / len(dual["low_price"]), index=dual["low_price"]))
        dual_return += 0.5 * weighted_return(returns, pd.Series(
            1.0 / len(dual["low_vol"]), index=dual["low_vol"]))
        five_return = weighted_return(returns, five["weights"])
        portfolio_returns = {
            "dual": dual_return,
            "three_equal": weighted_return(returns, three["weights"]),
            "five_factor": five_return, "benchmark": benchmark,
        }
        sleeve_returns = {}
        for key in SLEEVES:
            members = five["sleeves"][key]
            sleeve_returns[key] = equal_weight_return(returns, members)
            sleeve_nav[key].append(sleeve_nav[key][-1] * (1 + sleeve_returns[key]))
        for name in nav_map:
            nav_map[name].append(nav_map[name][-1] * (1 + portfolio_returns[name]))

        contributions = contribution_table(returns, five["weights"], benchmark)
        contributions["date"] = trade_date
        contributions["next_date"] = next_date
        contributions["code"] = contributions.index
        contributions["name"] = [names.get(code, code) for code in contributions.index]
        contributions["pe_ttm"] = [five["scores"]["pe_ttm"].get(code)
                                   for code in contributions.index]
        contributions["pb_mrq"] = [five["scores"]["pb_mrq"].get(code)
                                   for code in contributions.index]
        contributions["momentum"] = [five["scores"]["momentum"].get(code)
                                     for code in contributions.index]
        contribution_rows.append(contributions.reset_index(drop=True))

        for key, label, _ in STYLE_DIMENSIONS:
            if key not in five["scores"]:
                continue
            for row in bucket_returns(five["scores"][key], returns, buckets):
                row.update({"date": trade_date, "dimension": key, "label": label})
                style_rows.append(row)

        sleeve_frame = pd.Series(sleeve_returns)
        period_rows.append({
            "date": trade_date, "next_date": next_date,
            "dual_return": dual_return, "three_equal_return": portfolio_returns["three_equal"],
            "five_factor_return": five_return, "benchmark_return": benchmark,
            "five_excess": five_return - benchmark,
            "low_price_return": sleeve_returns["low_price"],
            "low_vol_return": sleeve_returns["low_vol"],
            "momentum_return": sleeve_returns["momentum"],
            "low_pe_return": sleeve_returns["low_pe"],
            "low_pb_return": sleeve_returns["low_pb"],
            "best_sleeve": SLEEVE_LABELS[sleeve_frame.idxmax()],
            "worst_sleeve": SLEEVE_LABELS[sleeve_frame.idxmin()],
            "five_holdings": len(five["weights"]),
            "intersection_count": five["counts"]["intersection"],
        })

    periods = pd.DataFrame(period_rows)
    contributions = pd.concat(contribution_rows, ignore_index=True)
    style = pd.DataFrame(style_rows)
    nav = pd.DataFrame(nav_map, index=dates)
    nav.index.name = "date"
    metrics = {name: performance_metrics(nav[name]) for name in (
        "dual", "three_equal", "five_factor", "benchmark")}

    # 五因子全段负贡献前 5 只股票（按累计贡献排序）
    ranked = (contributions.groupby(["code", "name"], as_index=False)
              .agg(times=("date", "count"), weight=("weight", "mean"),
                   contribution=("contribution", "sum"))
              .sort_values("contribution"))
    style_verdicts = judge_style(style)
    ineffective = [item["label"] for item in style_verdicts if not item["effective"]]
    if len(ineffective) >= 3:
        diagnosis = "多因子同时失效，符合风格切换特征"
    elif ineffective:
        diagnosis = "部分因子失效，需结合个股集中度判断"
    else:
        diagnosis = "因子方向仍有效，亏损主要来自组合结构与个股集中"

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": len(data_map), "start": dates[0], "end": dates[-1],
            "pct": pct, "vol_window": vol_window, "buckets": buckets,
            "momentum_window": momentum_window, "momentum_skip": momentum_skip,
            "freq": "M", "fq": fq,
            "industry_note": "本地 stock_info 行业字段为空，本次不做行业归因",
            "bucket_note": "Q1 为该维度分数最低组；价格/波动率/PE/PB 越低越好，动量越高越好",
            "disclaimer": DISCLAIMER,
        },
        "periods": periods, "contributions": contributions, "style": style,
        "nav": nav, "metrics": metrics,
        "worst_stocks": ranked.head(8), "best_stocks": ranked.tail(5).iloc[::-1],
        "style_verdicts": style_verdicts, "diagnosis": diagnosis,
        "sleeve_compound": {key: float(sleeve_nav[key][-1] / sleeve_nav[key][0] - 1)
                            for key in SLEEVES},
    }


def save_attribution_reports(result, directory=None):
    """保存逐期、个股贡献、风格分层 CSV 与摘要 JSON。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    stem = "smart_beta_s4_attr_%s_%s" % (
        meta["start"].replace("-", ""), meta["end"].replace("-", ""))
    paths = {}
    for key, frame in (("periods", result["periods"]),
                       ("contributions", result["contributions"]),
                       ("style", result["style"]),
                       ("stocks", pd.concat([
                           result["worst_stocks"].assign(group="拖累"),
                           result["best_stocks"].assign(group="贡献")]))):
        path = directory / ("%s_%s.csv" % (stem, key))
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    summary_path = directory / (stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "meta": meta, "metrics": result["metrics"],
        "diagnosis": result["diagnosis"],
        "sleeve_compound": result["sleeve_compound"],
        "style_verdicts": result["style_verdicts"],
        "periods": result["periods"].to_dict("records"),
        "worst_stocks": result["worst_stocks"].to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def print_attribution_report(result):
    """打印逐期收益、因子表现、风格方向与主要拖累股票。"""
    meta = result["meta"]
    print("\n=== Smart Beta 弱段亏损归因（%s ~ %s）===" % (
        meta["start"], meta["end"]))
    print("股票池 %d 只 | 各前 %.0f%% | 每 %d 个自然月 %d 期" % (
        meta["codes"], meta["pct"] * 100, 6, len(result["periods"])))
    print("\n--- 逐期收益 ---")
    print("%-12s %9s %9s %9s %9s %9s %8s" % (
        "调仓日", "双因子", "三因子", "五因子", "基准", "五因子超额", "最差因子"))
    for _, row in result["periods"].iterrows():
        print("%-12s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8s" % (
            row["date"], row["dual_return"] * 100, row["three_equal_return"] * 100,
            row["five_factor_return"] * 100, row["benchmark_return"] * 100,
            row["five_excess"] * 100, row["worst_sleeve"]))

    print("\n--- 单因子子组合（全段复利）---")
    for key in SLEEVES:
        print("%-8s %8.2f%%" % (SLEEVE_LABELS[key],
                                result["sleeve_compound"][key] * 100))
    print("%-8s %8.2f%%" % ("池内基准",
                            result["metrics"]["benchmark"]["total_return"] * 100))

    print("\n--- 风格分层（Q1 最低分，Q5 最高分）---")
    print("%-8s %10s %10s %10s %s" % ("维度", "Q1 累计", "Q5 累计", "Q1-Q5", "方向"))
    for item in result["style_verdicts"]:
        print("%-8s %9.2f%% %9.2f%% %9.2f%% %s" % (
            item["label"], item["q1_return"] * 100, item["q5_return"] * 100,
            item["spread_q1_minus_q5"] * 100,
            "有效" if item["effective"] else "失效"))

    print("\n--- 主要拖累股票（五因子，累计贡献）---")
    print("%-8s %-10s %6s %8s %10s" % ("代码", "名称", "期数", "平均权重", "累计贡献"))
    for _, row in result["worst_stocks"].iterrows():
        print("%-8s %-10s %6d %7.2f%% %9.2f%%" % (
            row["code"], row["name"], row["times"], row["weight"] * 100,
            row["contribution"] * 100))
    print("\n归因结论：%s" % result["diagnosis"])


def synthetic_attribution_test():
    """验证分位收益、贡献可加性与风格判定。"""
    scores = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    returns = pd.Series([0.10, 0.09, 0.08, 0.07, 0.06,
                         0.05, 0.04, 0.03, 0.02, 0.01])
    buckets = bucket_returns(scores, returns, 5)
    assert len(buckets) == 5
    assert buckets[0]["count"] == 2 and buckets[0]["bucket"] == 1
    assert buckets[0]["return"] > buckets[-1]["return"], "Q1 应跑赢 Q5"

    weights = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    values = pd.Series({"a": 0.10, "b": -0.05, "c": 0.02, "d": 0.50})
    benchmark = 0.01
    table = contribution_table(values, weights, benchmark)
    assert abs(table["weight"].sum() - 1) < 1e-12
    assert abs(table["contribution"].sum() - (0.5 * 0.10 + 0.3 * -0.05 + 0.2 * 0.02 - 0.01)) < 1e-12
    # 缺行情的股票自动从有效权重中剔除并重新归一化
    partial = contribution_table(pd.Series({"a": 0.2, "b": np.nan}), weights, 0.0)
    assert abs(float(partial["weight"].iloc[0]) - 1.0) < 1e-12
    assert partial.shape[0] == 1

    style = pd.DataFrame([
        {"dimension": "price", "bucket": 1, "return": 0.05},
        {"dimension": "price", "bucket": 5, "return": 0.10},
        {"dimension": "momentum", "bucket": 1, "return": 0.02},
        {"dimension": "momentum", "bucket": 5, "return": 0.08},
    ])
    verdicts = {item["dimension"]: item for item in judge_style(style)}
    assert verdicts["price"]["effective"] is False, "高价组更强应判低价失效"
    assert verdicts["momentum"]["effective"] is True, "高动量组更强应判动量有效"
    print("合成数据：分位分组、贡献可加、缺失行情归一化与风格方向判定均通过")


if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()
    synthetic_attribution_test()
    result = run_attribution()
    print_attribution_report(result)

    periods = result["periods"]
    assert len(periods) == 6, len(periods)
    assert "five_excess" in periods
    # 每期个股贡献之和必须等于五因子相对基准的超额
    for date, frame in result["contributions"].groupby("date"):
        excess = float(periods.loc[periods["date"] == date, "five_excess"].iloc[0])
        assert abs(frame["contribution"].sum() - excess) < 1e-12, date
    # 归因复现的分段结果应与 I2-alt-6 的 S4 五因子一致
    five_total = result["metrics"]["five_factor"]["total_return"]
    assert abs(five_total - (-0.0383)) < 0.001, five_total
    assert result["style"]["bucket"].nunique() >= 4
    paths = save_attribution_reports(result)
    assert len(pd.read_csv(paths["periods"])) == 6
    assert len(pd.read_csv(paths["stocks"])) == 13
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["diagnosis"] == result["diagnosis"]
    print("\n报告已保存：")
    for path in paths.values():
        print(path)
    print("\n联调清单：1 逐期复现 → 2 单因子表现 → 3 风格分层 → 4 个股贡献 → 5 结论与落盘，均已完成")
    print("smart_beta_attribution.py 自测通过；%s" % DISCLAIMER)
