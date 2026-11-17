# -*- coding: utf-8 -*-
"""并集规则修复与重跑验证（Step I2-alt-9）。

I2-alt-8 归因结论：四/五因子在交集不足、走“并集按名次之和取前 30”时，
纯动量股因其余四维名次垫底被系统性淘汰，S4 期间动量权重长期为 0，
组合退化为纯深度价值暴露。

本步只做一件事：给并集分支加“动量最低权重”下限（执行值 10%）：
- 强制动量前 20% 子组合至少入选 round(30 × 10%) = 3 只；
- 若动量持仓权重合计不足 10%，按比例上调到 10%，其余因子等比例下调；
- 交集分支与既有默认调用完全不受影响（momentum_floor 默认 None）。

10% 由 I2-alt-9 四档敏感性（10/15/20/30%）选出：全期回撤 10.76% 不超基准
11.95%，余量 1.19pp，S4 亏损收敛，且 S5 年化保留 17.86%。

验证范围：全期 2024-05-31 ~ 2026-11-13 逐期回测 + 每 6 个自然月分段验证，
修复前后同数据、同调仓日、同判定口径直接对比。

不接券商、不下单、不上传云端；仅研究用途，不构成投资建议。
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
    SMART_BETA_DIR, period_returns, equal_weight_return,
    performance_metrics, holding_turnover,
)
from core.smart_beta_3factor_backtest import weighted_return, weighted_turnover
from core.smart_beta_fundamental_backtest import build_fundamental_composite
from core.smart_beta_walkforward import assign_segments, log_share
from core.valuation_data import load_valuation

MOMENTUM_FLOOR = 0.10
DOMINANT_SHARE = 0.60
PORTFOLIOS = ("dual", "three_equal", "four_old", "four_floor",
              "five_old", "five_floor")
_FLOOR_TEXT = "%d%%" % int(round(MOMENTUM_FLOOR * 100))
PORTFOLIO_LABELS = {
    "dual": "双因子50:50",
    "three_equal": "三因子等权33/33/34",
    "four_old": "四因子·旧并集规则",
    "four_floor": "四因子·动量≥" + _FLOOR_TEXT,
    "five_old": "五因子·旧并集规则",
    "five_floor": "五因子·动量≥" + _FLOOR_TEXT,
    "benchmark": "池内等权基准",
}


def _composite(data_map, valuation_map, trade_date, include_pb, pct,
               vol_window, momentum_window, momentum_skip, floor):
    """统一构造四/五因子组合，floor 为 None 时与既有行为完全一致。"""
    return build_fundamental_composite(
        data_map, valuation_map, trade_date, True, include_pb, pct, vol_window,
        momentum_window, momentum_skip, momentum_floor=floor)


def _segment_label(index, frame):
    return "S%d %s~%s" % (index + 1, frame.iloc[0]["date"],
                          frame.iloc[-1]["next_date"])


def run_union_fix(
        codes=None, start_date="2024-05-31", end_date="2026-11-13",
        pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, fq="qfq",
        momentum_floor=MOMENTUM_FLOOR, months_per_segment=6):
    """同一批调仓日上同时回测旧规则与修复规则，逐段独立统计。"""
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
    seg_of = assign_segments(dates[:-1], start_date, months_per_segment)

    nav_map = {name: [1.0] for name in PORTFOLIOS + ("benchmark",)}
    turnover_map = {name: [] for name in PORTFOLIOS}
    previous = {name: pd.Series(dtype=float)
                for name in PORTFOLIOS if name != "dual"}
    previous_dual = {"low_price": [], "low_vol": []}
    seg_nav, seg_turnover = {}, {}
    period_rows = []

    for i in range(len(dates) - 1):
        trade_date, next_date = dates[i], dates[i + 1]
        segment = seg_of[trade_date]
        seg_nav.setdefault(segment, {name: [1.0]
                                     for name in PORTFOLIOS + ("benchmark",)})
        seg_turnover.setdefault(segment, {name: [] for name in PORTFOLIOS})

        returns = period_returns(data_map, trade_date, next_date)
        dual = build_exposures(
            data_map, trade_date, pct=pct, vol_window=vol_window)
        three = build_three_factor_exposures(
            data_map, trade_date, pct=pct, vol_window=vol_window,
            momentum_window=momentum_window, momentum_skip=momentum_skip,
            mode="equal")
        four_old = _composite(data_map, valuation_map, trade_date, False,
                              pct, vol_window, momentum_window, momentum_skip, None)
        four_floor = _composite(data_map, valuation_map, trade_date, False,
                                pct, vol_window, momentum_window,
                                momentum_skip, momentum_floor)
        five_old = _composite(data_map, valuation_map, trade_date, True,
                              pct, vol_window, momentum_window, momentum_skip, None)
        five_floor = _composite(data_map, valuation_map, trade_date, True,
                                pct, vol_window, momentum_window,
                                momentum_skip, momentum_floor)

        low_ret = weighted_return(returns, pd.Series(
            1.0 / len(dual["low_price"]), index=dual["low_price"]))
        vol_ret = weighted_return(returns, pd.Series(
            1.0 / len(dual["low_vol"]), index=dual["low_vol"]))
        dual_return = 0.5 * low_ret + 0.5 * vol_ret
        dual_turnover = 0.5 * holding_turnover(
            previous_dual["low_price"], dual["low_price"])
        dual_turnover += 0.5 * holding_turnover(
            previous_dual["low_vol"], dual["low_vol"])
        previous_dual = {"low_price": dual["low_price"], "low_vol": dual["low_vol"]}

        portfolio_returns = {"dual": dual_return}
        turnovers = {"dual": dual_turnover}
        for name, exposure in (("three_equal", three), ("four_old", four_old),
                               ("four_floor", four_floor), ("five_old", five_old),
                               ("five_floor", five_floor)):
            weights = exposure["weights"]
            portfolio_returns[name] = weighted_return(returns, weights)
            turnovers[name] = weighted_turnover(previous[name], weights)
            previous[name] = weights.copy()

        benchmark_return = equal_weight_return(returns, list(data_map))
        for name in PORTFOLIOS:
            nav_map[name].append(nav_map[name][-1] * (1 + portfolio_returns[name]))
            seg_nav[segment][name].append(
                seg_nav[segment][name][-1] * (1 + portfolio_returns[name]))
            seg_turnover[segment][name].append(turnovers[name])
        nav_map["benchmark"].append(
            nav_map["benchmark"][-1] * (1 + benchmark_return))
        seg_nav[segment]["benchmark"].append(
            seg_nav[segment]["benchmark"][-1] * (1 + benchmark_return))

        row = {"date": trade_date, "next_date": next_date, "segment": segment,
               "benchmark_return": benchmark_return}
        for name in PORTFOLIOS:
            row[name + "_return"] = portfolio_returns[name]
        for prefix, exposure in (("four_old", four_old), ("four_floor", four_floor),
                                 ("five_old", five_old), ("five_floor", five_floor)):
            row[prefix + "_rule"] = exposure["selection_rule"]
            row[prefix + "_intersection"] = exposure["counts"]["intersection"]
            row[prefix + "_selected"] = exposure["counts"]["selected"]
            row[prefix + "_momentum_count"] = exposure["momentum_selected"]
            row[prefix + "_momentum_weight"] = exposure["momentum_weight"]
        period_rows.append(row)

    periods = pd.DataFrame(period_rows)
    labels = {segment: _segment_label(segment, frame)
              for segment, frame in periods.groupby("segment")}
    periods["segment_label"] = periods["segment"].map(labels)

    full = []
    full_nav = pd.DataFrame(nav_map, index=dates)
    full_metrics = {}
    for name in PORTFOLIOS:
        full_metrics[name] = performance_metrics(
            full_nav[name], turnover_map[name])
    benchmark_metrics = performance_metrics(full_nav["benchmark"])
    for name in PORTFOLIOS:
        metrics = full_metrics[name]
        beat_cash = metrics["annual_return"] > 0
        controlled = metrics["max_drawdown"] <= benchmark_metrics["max_drawdown"]
        full.append({
            "portfolio": name, "label": PORTFOLIO_LABELS[name],
            "total_return": metrics["total_return"],
            "annual_return": metrics["annual_return"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "total_turnover": metrics["total_turnover"],
            "benchmark_annual_return": benchmark_metrics["annual_return"],
            "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
            "excess_return": metrics["total_return"] - benchmark_metrics["total_return"],
            "beat_cash": bool(beat_cash),
            "drawdown_not_above_benchmark": bool(controlled),
            "passed": bool(beat_cash and controlled),
        })
    full = pd.DataFrame(full)

    segment_rows, nav_payload = [], {}
    for segment in sorted(seg_nav):
        frame = periods[periods["segment"] == segment]
        index = [frame.iloc[0]["date"]] + frame["next_date"].tolist()
        label = labels[segment]
        segment_benchmark = performance_metrics(
            pd.Series(seg_nav[segment]["benchmark"], index=index))
        nav_payload[label] = {
            "periods": len(frame),
            "nav": {name: {date: round(float(value), 8)
                           for date, value in zip(index, seg_nav[segment][name])}
                    for name in PORTFOLIOS + ("benchmark",)},
        }
        for name in PORTFOLIOS:
            metrics = performance_metrics(
                pd.Series(seg_nav[segment][name], index=index),
                seg_turnover[segment][name])
            beat_cash = metrics["annual_return"] > 0
            controlled = metrics["max_drawdown"] <= segment_benchmark["max_drawdown"]
            segment_rows.append({
                "segment": segment, "segment_label": label,
                "portfolio": name, "label": PORTFOLIO_LABELS[name],
                "periods": metrics["periods"],
                "total_return": metrics["total_return"],
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "total_turnover": metrics["total_turnover"],
                "benchmark_total_return": segment_benchmark["total_return"],
                "benchmark_max_drawdown": segment_benchmark["max_drawdown"],
                "excess_return": metrics["total_return"]
                - segment_benchmark["total_return"],
                "beat_cash": bool(beat_cash),
                "drawdown_not_above_benchmark": bool(controlled),
                "passed": bool(beat_cash and controlled),
            })
    segments = pd.DataFrame(segment_rows)
    stability = stability_table(segments)
    s4 = s4_comparison(segments)

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": len(data_map), "start": dates[0], "end": dates[-1],
            "pct": pct, "vol_window": vol_window,
            "momentum_window": momentum_window, "momentum_skip": momentum_skip,
            "momentum_floor": momentum_floor, "months_per_segment": months_per_segment,
            "freq": "M", "fq": fq,
            "fix_note": "仅并集分支生效：强制动量入选 round(30×下限) 只，"
                        "且动量权重合计不低于下限，其余因子等比例下调",
            "old_note": "old 后缀为 I2-alt-8 之前的并集规则，代码路径相同且 floor=None",
            "verdict_note": "年化收益大于0且最大回撤不超过同期池内等权基准",
            "stability_note": "全部段通过且无单段对数收益占比超过%.0f%%记为稳定"
                              % (DOMINANT_SHARE * 100),
            "disclaimer": DISCLAIMER,
        },
        "full": full, "segments": segments, "stability": stability,
        "s4": s4, "periods": periods, "full_nav": full_nav, "nav": nav_payload,
        "benchmark_metrics": benchmark_metrics,
    }


def stability_table(segments, portfolios=PORTFOLIOS, labels=None):
    """按组合汇总跨段稳定性，口径与 I2-alt-6 一致。"""
    labels = labels or PORTFOLIO_LABELS
    rows = []
    for name in portfolios:
        subset = segments[segments["portfolio"] == name]
        if subset.empty:
            continue
        returns = subset["total_return"].tolist()
        shares = log_share(returns)
        max_share = float(np.nanmax(shares)) if np.isfinite(shares).any() else np.nan
        dominant = (subset.iloc[int(np.nanargmax(shares))]["segment_label"]
                    if max_share == max_share else "")
        passed = int(subset["passed"].sum())
        if float(np.log1p(returns).sum()) <= 0:
            verdict = "全期亏损"
        elif passed < len(subset):
            verdict = "不稳定"
        elif max_share > DOMINANT_SHARE:
            verdict = "单段主导"
        else:
            verdict = "稳定"
        rows.append({
            "portfolio": name, "label": labels[name],
            "segments": len(subset), "passed_segments": passed,
            "compound_return": float(np.prod([1 + value for value in returns]) - 1),
            "annual_min": float(subset["annual_return"].min()),
            "annual_max": float(subset["annual_return"].max()),
            "annual_std": float(subset["annual_return"].std(ddof=1))
            if len(subset) > 1 else 0.0,
            "worst_drawdown": float(subset["max_drawdown"].max()),
            "max_segment_log_share": max_share,
            "dominant_segment": dominant,
            "negative_segments": int((subset["total_return"] < 0).sum()),
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def s4_comparison(segments):
    """只取 S4 段，修复前后成对对比。"""
    rows = []
    for old, new in (("four_old", "four_floor"), ("five_old", "five_floor")):
        old_row = segments[(segments["portfolio"] == old)
                           & segments["segment_label"].str.startswith("S4")].iloc[0]
        new_row = segments[(segments["portfolio"] == new)
                           & segments["segment_label"].str.startswith("S4")].iloc[0]
        rows.append({
            "pair": "%s → %s" % (old, new),
            "old_total_return": old_row["total_return"],
            "new_total_return": new_row["total_return"],
            "delta_total_return": new_row["total_return"] - old_row["total_return"],
            "old_annual_return": old_row["annual_return"],
            "new_annual_return": new_row["annual_return"],
            "delta_annual_return": new_row["annual_return"] - old_row["annual_return"],
            "old_max_drawdown": old_row["max_drawdown"],
            "new_max_drawdown": new_row["max_drawdown"],
            "delta_max_drawdown": new_row["max_drawdown"] - old_row["max_drawdown"],
            "old_excess_return": old_row["excess_return"],
            "new_excess_return": new_row["excess_return"],
            "delta_excess_return": new_row["excess_return"] - old_row["excess_return"],
            "old_passed": bool(old_row["passed"]),
            "new_passed": bool(new_row["passed"]),
        })
    return pd.DataFrame(rows)


def save_union_fix_reports(result, directory=None):
    """保存全期、逐期、分段、稳定性、S4 对比与净值 JSON。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    stem = "smart_beta_union_fix_%s_%s_%s" % (
        meta["start"].replace("-", ""), meta["end"].replace("-", ""),
        meta["codes"])
    paths = {}
    for key in ("full", "periods", "segments", "stability", "s4"):
        path = directory / ("%s_%s.csv" % (stem, key))
        result[key].to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    nav_path = directory / (stem + "_nav.json")
    nav_path.write_text(json.dumps({
        "meta": meta,
        "full_nav": {column: {str(date): round(float(value), 8)
                              for date, value in result["full_nav"][column].items()}
                     for column in result["full_nav"].columns},
        "segment_nav": result["nav"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["nav"] = nav_path
    summary_path = directory / (stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "meta": meta,
        "full": result["full"].to_dict("records"),
        "stability": result["stability"].to_dict("records"),
        "s4": result["s4"].to_dict("records"),
        "benchmark_metrics": result["benchmark_metrics"],
        "periods": result["periods"].to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def print_union_fix_report(result):
    """打印全期、S4 与分段对比，重点看亏损收敛与判定是否维持。"""
    meta = result["meta"]
    print("\n=== 并集规则修复与重跑验证 ===")
    print("股票池 %d 只 | %s ~ %s | 各前 %.0f%% | 动量下限 %.0f%%" % (
        meta["codes"], meta["start"], meta["end"], meta["pct"] * 100,
        meta["momentum_floor"] * 100))

    print("\n--- 全期绩效（基准年化 %.2f%%，回撤 %.2f%%）---" % (
        result["benchmark_metrics"]["annual_return"] * 100,
        result["benchmark_metrics"]["max_drawdown"] * 100))
    print("%-22s %9s %9s %8s %9s %9s %8s" % (
        "组合", "总收益", "年化", "夏普", "最大回撤", "超额", "判定"))
    for _, row in result["full"].iterrows():
        print("%-22s %8.2f%% %8.2f%% %8.3f %8.2f%% %8.2f%% %8s" % (
            row["label"], row["total_return"] * 100, row["annual_return"] * 100,
            row["sharpe"], row["max_drawdown"] * 100,
            row["excess_return"] * 100, "通过" if row["passed"] else "未通过"))

    print("\n--- S4 修复前后 ---")
    print("%-24s %9s %9s %9s %9s %9s" % (
        "对比", "旧总收益", "新总收益", "改善pp", "旧回撤", "新回撤"))
    for _, row in result["s4"].iterrows():
        print("%-24s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%%" % (
            row["pair"], row["old_total_return"] * 100,
            row["new_total_return"] * 100, row["delta_total_return"] * 100,
            row["old_max_drawdown"] * 100, row["new_max_drawdown"] * 100))

    print("\n--- 分段绩效（四/五因子修复前后）---")
    focus = ("four_old", "four_floor", "five_old", "five_floor")
    for label, frame in result["segments"].groupby("segment_label", sort=True):
        benchmark = frame.iloc[0]
        print("\n%s | 基准总收益 %.2f%% 回撤 %.2f%%" % (
            label, benchmark["benchmark_total_return"] * 100,
            benchmark["benchmark_max_drawdown"] * 100))
        print("%-22s %9s %9s %9s %8s %6s" % (
            "组合", "总收益", "年化", "最大回撤", "超额", "判定"))
        for _, row in frame[frame["portfolio"].isin(focus)].iterrows():
            print("%-22s %8.2f%% %8.2f%% %8.2f%% %7.2f%% %6s" % (
                row["label"], row["total_return"] * 100,
                row["annual_return"] * 100, row["max_drawdown"] * 100,
                row["excess_return"] * 100, "通过" if row["passed"] else "未通过"))

    print("\n--- 跨段稳定性 ---")
    print("%-22s %4s %5s %9s %9s %9s %8s %6s %s" % (
        "组合", "段数", "通过", "年化最低", "年化最高", "最差回撤", "最大占比", "负段", "结论"))
    for _, row in result["stability"].iterrows():
        share = "—" if not np.isfinite(row["max_segment_log_share"]) else "%.0f%%" % (
            row["max_segment_log_share"] * 100)
        print("%-22s %4d %5d %8.2f%% %8.2f%% %8.2f%% %8s %6d %s" % (
            row["label"], row["segments"], row["passed_segments"],
            row["annual_min"] * 100, row["annual_max"] * 100,
            row["worst_drawdown"] * 100, share, row["negative_segments"],
            row["verdict"]))


def _synthetic_market(codes=60, days=400):
    """构造“动量股价格高、波动大、估值高”的合成市场，复现动量被挤出。"""
    dates = pd.bdate_range(end="2026-05-29", periods=days).strftime(
        "%Y-%m-%d").tolist()
    rng = np.random.default_rng(7)
    data_map, valuation_map = {}, {}
    for index in range(codes):
        code = "%06d" % (600000 + index)
        if index < 12:      # 强动量组：高价格、高波动、高估值
            steps = np.concatenate([[0.0], rng.normal(0.012, 0.030, days - 1)])
            close = 5.0 * np.exp(np.cumsum(steps))
            pe, pb = 200.0, 15.0
        else:               # 价值组：低价格、低波动、低估值
            steps = np.concatenate([[0.0], rng.normal(0.0001, 0.004, days - 1)])
            close = (5.0 + index * 0.15) * np.exp(np.cumsum(steps))
            # 让 PE/PB 排序与价格排序相反，避免四个价值子组合高度重叠，
            # 从而保证并集里价值股数量足够多、旧规则能把动量股完全挤出
            pe = 6.0 + (59 - index) * 0.08
            pb = 0.6 + (59 - index) * 0.02
        data_map[code] = pd.DataFrame({"date": dates, "close": close})
        valuation_map[code] = pd.DataFrame({
            "date": [dates[0], dates[-1]], "code": [code, code],
            "pe_ttm": [pe, pe], "pb_mrq": [pb, pb],
            "tradestatus": [1, 1], "is_st": [0, 0],
        })
    return data_map, valuation_map, dates[-1]


def synthetic_union_fix_test():
    """合成数据验证：旧规则动量权重为 0，新规则只数与仓位下限均生效。"""
    data_map, valuation_map, trade_date = _synthetic_market()
    old = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, True)
    assert old["selection_rule"] == "union_top", old["selection_rule"]
    assert old["momentum_selected"] == 0, "旧规则应把动量股全部挤出"
    assert abs(old["momentum_weight"]) < 1e-12

    new = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, True, momentum_floor=0.20)
    assert new["momentum_slots"] == 6, new["momentum_slots"]
    assert new["momentum_selected"] >= 6, new["momentum_selected"]
    assert abs(new["momentum_weight"] - 0.20) < 1e-9, new["momentum_weight"]
    assert len(new["selected"]) == len(old["selected"]) == 30
    assert abs(float(new["weights"].sum()) - 1) < 1e-12
    rest = new["weights"][~new["weights"].index.isin(new["sleeves"]["momentum"])]
    assert abs(float(rest.sum()) - 0.80) < 1e-9
    assert set(old["selected"]) & set(new["selected"]), "调仓应保留原有价值暴露"

    # 默认参数必须与既有行为逐值一致，保证旧调用方零影响
    legacy = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, True)
    assert legacy["weights"].equals(old["weights"])
    try:
        build_fundamental_composite(
            data_map, valuation_map, trade_date, True, True, momentum_floor=1.2)
    except ValueError:
        pass
    else:
        raise AssertionError("非法 momentum_floor 应报错")
    print("合成数据：旧规则动量 0 只/0%%，新规则动量 6 只/20%%，默认调用不变，均通过")


SENS_FLOORS = (0.10, 0.15, 0.20, 0.30)


def _floor_tag(floor):
    """把下限转成列名后缀：None 记 old，0.15 记 f15。"""
    return "old" if floor is None else "f%02d" % int(round(floor * 100))


def _sensitivity_names(floors=SENS_FLOORS):
    names, labels = [], {}
    for floor in (None,) + tuple(floors):
        tag = _floor_tag(floor)
        suffix = "旧规则" if floor is None else "动量≥%d%%" % int(round(floor * 100))
        for factor, factor_label in (("four", "四因子"), ("five", "五因子")):
            name = "%s_%s" % (factor, tag)
            names.append(name)
            labels[name] = factor_label + "·" + suffix
    return names, labels


def run_floor_sensitivity(
        codes=None, start_date="2024-05-31", end_date="2026-11-13",
        floors=SENS_FLOORS, pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, fq="qfq", months_per_segment=6):
    """同一批调仓日上横向遍历动量下限，旧规则与各档同数据直接可比。"""
    floors = tuple(floors)
    if any(not 0 < floor < 1 for floor in floors):
        raise ValueError("动量下限必须在 (0, 1) 内")
    variants = (None,) + floors
    names, labels = _sensitivity_names(floors)
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
    seg_of = assign_segments(dates[:-1], start_date, months_per_segment)

    nav_map = {name: [1.0] for name in names + ["benchmark"]}
    turnover_map = {name: [] for name in names}
    previous = {name: pd.Series(dtype=float) for name in names}
    seg_nav, seg_turnover = {}, {}
    period_rows = []

    for i in range(len(dates) - 1):
        trade_date, next_date = dates[i], dates[i + 1]
        segment = seg_of[trade_date]
        seg_nav.setdefault(segment, {name: [1.0] for name in names + ["benchmark"]})
        seg_turnover.setdefault(segment, {name: [] for name in names})
        row = {"date": trade_date, "next_date": next_date, "segment": segment}

        returns = period_returns(data_map, trade_date, next_date)
        benchmark_return = equal_weight_return(returns, list(data_map))
        row["benchmark_return"] = benchmark_return
        for floor in variants:
            tag = _floor_tag(floor)
            four = _composite(data_map, valuation_map, trade_date, False, pct,
                              vol_window, momentum_window, momentum_skip, floor)
            five = _composite(data_map, valuation_map, trade_date, True, pct,
                              vol_window, momentum_window, momentum_skip, floor)
            for name, exposure in (("four_" + tag, four), ("five_" + tag, five)):
                weights = exposure["weights"]
                value = weighted_return(returns, weights)
                turnover = weighted_turnover(previous[name], weights)
                previous[name] = weights.copy()
                nav_map[name].append(nav_map[name][-1] * (1 + value))
                seg_nav[segment][name].append(
                    seg_nav[segment][name][-1] * (1 + value))
                seg_turnover[segment][name].append(turnover)
                row[name + "_return"] = value
                row[name + "_momentum_weight"] = exposure["momentum_weight"]
                row[name + "_rule"] = exposure["selection_rule"]
        nav_map["benchmark"].append(nav_map["benchmark"][-1] * (1 + benchmark_return))
        seg_nav[segment]["benchmark"].append(
            seg_nav[segment]["benchmark"][-1] * (1 + benchmark_return))
        period_rows.append(row)

    periods = pd.DataFrame(period_rows)
    labels_by_segment = {segment: _segment_label(segment, frame)
                         for segment, frame in periods.groupby("segment")}
    periods["segment_label"] = periods["segment"].map(labels_by_segment)

    full_nav = pd.DataFrame(nav_map, index=dates)
    benchmark_metrics = performance_metrics(full_nav["benchmark"])
    full = []
    for name in names:
        metrics = performance_metrics(full_nav[name], turnover_map[name])
        beat_cash = metrics["annual_return"] > 0
        controlled = metrics["max_drawdown"] <= benchmark_metrics["max_drawdown"]
        full.append({
            "portfolio": name, "label": labels[name],
            "total_return": metrics["total_return"],
            "annual_return": metrics["annual_return"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "total_turnover": metrics["total_turnover"],
            "benchmark_annual_return": benchmark_metrics["annual_return"],
            "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
            "excess_return": metrics["total_return"] - benchmark_metrics["total_return"],
            "beat_cash": bool(beat_cash),
            "drawdown_not_above_benchmark": bool(controlled),
            "passed": bool(beat_cash and controlled),
        })
    full = pd.DataFrame(full)

    segment_rows, nav_payload = [], {}
    for segment in sorted(seg_nav):
        frame = periods[periods["segment"] == segment]
        index = [frame.iloc[0]["date"]] + frame["next_date"].tolist()
        label = labels_by_segment[segment]
        segment_benchmark = performance_metrics(
            pd.Series(seg_nav[segment]["benchmark"], index=index))
        nav_payload[label] = {
            "periods": len(frame),
            "nav": {name: {date: round(float(value), 8)
                           for date, value in zip(index, seg_nav[segment][name])}
                    for name in names + ["benchmark"]},
        }
        for name in names:
            metrics = performance_metrics(
                pd.Series(seg_nav[segment][name], index=index),
                seg_turnover[segment][name])
            beat_cash = metrics["annual_return"] > 0
            controlled = metrics["max_drawdown"] <= segment_benchmark["max_drawdown"]
            segment_rows.append({
                "segment": segment, "segment_label": label,
                "portfolio": name, "label": labels[name],
                "periods": metrics["periods"],
                "total_return": metrics["total_return"],
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "total_turnover": metrics["total_turnover"],
                "benchmark_total_return": segment_benchmark["total_return"],
                "benchmark_max_drawdown": segment_benchmark["max_drawdown"],
                "excess_return": metrics["total_return"]
                - segment_benchmark["total_return"],
                "beat_cash": bool(beat_cash),
                "drawdown_not_above_benchmark": bool(controlled),
                "passed": bool(beat_cash and controlled),
            })
    segments = pd.DataFrame(segment_rows)
    stability = stability_table(segments, names, labels)
    result = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": len(data_map), "start": dates[0], "end": dates[-1],
            "pct": pct, "vol_window": vol_window,
            "momentum_window": momentum_window, "momentum_skip": momentum_skip,
            "floors": list(floors), "months_per_segment": months_per_segment,
            "freq": "M", "fq": fq,
            "sensitivity_note": "各档同数据、同调仓日，仅动量下限不同；"
                                "old 为不加下限的旧并集规则",
            "verdict_note": "全期或分段：年化收益大于0且最大回撤不超过同期池内等权基准",
            "safe_note": "安全下限定义：全期回撤不超过基准且 S4 总收益不小于0",
            "disclaimer": DISCLAIMER,
        },
        "full": full, "segments": segments, "stability": stability,
        "periods": periods, "full_nav": full_nav, "nav": nav_payload,
        "benchmark_metrics": benchmark_metrics,
        "floors": floors, "names": names, "labels": labels,
    }
    result["sensitivity"] = floor_sensitivity_table(result)
    return result


def _segment_metrics(frame, prefix):
    """取某一段的绩效；样本不足时返回 NaN，避免敏感性表报错。"""
    matched = frame[frame["segment_label"].str.startswith(prefix)]
    if matched.empty:
        return {"total_return": np.nan, "annual_return": np.nan,
                "max_drawdown": np.nan, "passed": False}
    row = matched.iloc[0]
    return {"total_return": row["total_return"],
            "annual_return": row["annual_return"],
            "max_drawdown": row["max_drawdown"],
            "passed": bool(row["passed"])}


def floor_sensitivity_table(result):
    """汇总四档下限的全期、S4、S5 表现与平均动量权重。"""
    full = result["full"].set_index("portfolio")
    segments = result["segments"]
    periods = result["periods"]
    benchmark_dd = result["benchmark_metrics"]["max_drawdown"]
    rows = []
    for factor in ("four", "five"):
        for floor in (None,) + tuple(result["floors"]):
            tag = _floor_tag(floor)
            name = "%s_%s" % (factor, tag)
            frame = segments[segments["portfolio"] == name]
            s4 = _segment_metrics(frame, "S4")
            s5 = _segment_metrics(frame, "S5")
            rows.append({
                "factor": factor,
                "variant": "旧规则" if floor is None else "%d%%" % int(round(floor * 100)),
                "floor": 0.0 if floor is None else floor,
                "annual_return": full.loc[name, "annual_return"],
                "total_return": full.loc[name, "total_return"],
                "max_drawdown": full.loc[name, "max_drawdown"],
                "drawdown_ok": bool(full.loc[name, "max_drawdown"] <= benchmark_dd),
                "full_passed": bool(full.loc[name, "passed"]),
                "s4_total_return": s4["total_return"],
                "s4_max_drawdown": s4["max_drawdown"],
                "s4_passed": s4["passed"],
                "s5_annual_return": s5["annual_return"],
                "s5_total_return": s5["total_return"],
                "s5_max_drawdown": s5["max_drawdown"],
                "avg_momentum_weight": float(periods[name + "_momentum_weight"].mean()),
            })
    frame = pd.DataFrame(rows)
    frame["safe"] = frame["drawdown_ok"] & (frame["s4_total_return"] >= 0)
    return frame


def recommend_floor(sensitivity):
    """返回“全期回撤不超基准且 S4 转正”的最大下限，没有则返回 None。"""
    candidates = sensitivity[(sensitivity["factor"] == "five")
                             & sensitivity["safe"] & (sensitivity["floor"] > 0)]
    if candidates.empty:
        return None
    return float(candidates["floor"].max())


def save_sensitivity_reports(result, directory=None):
    """保存四档敏感性报告：全期、分段、稳定性、敏感性表与净值 JSON。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    stem = "smart_beta_floor_sensitivity_%s_%s_%s" % (
        meta["start"].replace("-", ""), meta["end"].replace("-", ""),
        meta["codes"])
    paths = {}
    for key in ("full", "segments", "stability", "sensitivity"):
        path = directory / ("%s_%s.csv" % (stem, key))
        result[key].to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    nav_path = directory / (stem + "_nav.json")
    nav_path.write_text(json.dumps({
        "meta": meta,
        "full_nav": {column: {str(date): round(float(value), 8)
                              for date, value in result["full_nav"][column].items()}
                     for column in result["full_nav"].columns},
        "segment_nav": result["nav"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["nav"] = nav_path
    summary_path = directory / (stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "meta": meta,
        "sensitivity": result["sensitivity"].to_dict("records"),
        "full": result["full"].to_dict("records"),
        "stability": result["stability"].to_dict("records"),
        "recommended_floor": recommend_floor(result["sensitivity"]),
        "benchmark_metrics": result["benchmark_metrics"],
        "periods": result["periods"].to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def print_sensitivity_report(result):
    """打印敏感性主表，重点看全期回撤约束与 S4/S5 的跷跷板。"""
    meta = result["meta"]
    benchmark = result["benchmark_metrics"]
    table = result["sensitivity"]
    print("\n=== 动量下限敏感性（%s ~ %s，%d 只）===" % (
        meta["start"], meta["end"], meta["codes"]))
    print("基准：年化 %.2f%% | 最大回撤 %.2f%% | 安全门槛 = 回撤≤%.2f%% 且 S4≥0" % (
        benchmark["annual_return"] * 100, benchmark["max_drawdown"] * 100,
        benchmark["max_drawdown"] * 100))
    for factor in ("four", "five"):
        print("\n--- %s ---" % ("四因子" if factor == "four" else "五因子"))
        print("%-8s %9s %9s %9s %9s %9s %9s %8s %6s" % (
            "下限", "全期年化", "全期回撤", "全期判定", "S4收益", "S5年化",
            "S5回撤", "动量权重", "安全"))
        for _, row in table[table["factor"] == factor].iterrows():
            print("%-8s %8.2f%% %8.2f%% %9s %8.2f%% %8.2f%% %8.2f%% %7.2f%% %6s" % (
                row["variant"], row["annual_return"] * 100,
                row["max_drawdown"] * 100,
                "通过" if row["full_passed"] else "未通过",
                row["s4_total_return"] * 100, row["s5_annual_return"] * 100,
                row["s5_max_drawdown"] * 100, row["avg_momentum_weight"] * 100,
                "是" if row["safe"] else "否"))
    best = recommend_floor(table)
    if best is None:
        print("\n结论：四档下限均不满足“全期回撤≤基准 且 S4 转正”，无安全下限")
    else:
        print("\n结论：满足条件（五因子口径）的最大安全下限为 %d%%" % int(best * 100))


def synthetic_floor_grid_test():
    """合成验证四档下限均生效、只数与权重达标、默认调用逐值不变。"""
    data_map, valuation_map, trade_date = _synthetic_market()
    baseline = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, True)
    legacy = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, True)
    assert legacy["weights"].equals(baseline["weights"])
    assert legacy["momentum_weight"] < 1e-12
    expected = ((0.10, 3), (0.15, 4), (0.20, 6), (0.30, 9))
    weights = []
    for floor, slots in expected:
        result = build_fundamental_composite(
            data_map, valuation_map, trade_date, True, True,
            momentum_floor=floor)
        assert result["momentum_slots"] == slots, (floor, result["momentum_slots"])
        assert result["momentum_selected"] >= slots
        assert abs(result["momentum_weight"] - floor) < 1e-9, (floor, result["momentum_weight"])
        assert abs(float(result["weights"].sum()) - 1) < 1e-12
        assert len(result["selected"]) == 30
        weights.append(result["momentum_weight"])
    assert weights == sorted(weights), "下限越高，动量权重不应下降"
    print("合成数据网格：10%%/15%%/20%%/30%% → 3/4/6/9 只，动量权重分别为 "
          "10%%/15%%/20%%/30%%，默认调用不变，均通过")



def _run_union_fix_main():
    """I2-alt-9 原入口：20% 下限的修复前后对比。"""
    print(DISCLAIMER)
    synthetic_union_fix_test()
    result = run_union_fix(momentum_floor=MOMENTUM_FLOOR, months_per_segment=6)
    print_union_fix_report(result)

    periods = result["periods"]
    assert len(periods) == 30, len(periods)
    assert result["segments"]["segment"].nunique() == 5
    assert result["segments"]["portfolio"].nunique() == len(PORTFOLIOS)
    # 修复后的每个并集期，动量权重都必须达到下限
    five_rows = periods[periods["five_floor_rule"] == "union_top"]
    four_rows = periods[periods["four_floor_rule"] == "union_top"]
    assert len(five_rows) > 0
    assert (five_rows["five_floor_momentum_weight"] >= MOMENTUM_FLOOR - 1e-9).all()
    assert (four_rows["four_floor_momentum_weight"] >= MOMENTUM_FLOOR - 1e-9).all()
    # 旧规则在 S4 的动量权重几乎为 0（I2-alt-8 归因：6 期中 5 期为 0）
    s4 = periods[periods["segment_label"].str.startswith("S4")]
    assert (s4["five_old_momentum_weight"].abs() < 1e-12).sum() >= 5
    assert float(s4["five_old_momentum_weight"].sum()) < 0.05
    # S4 五因子亏损必须收敛
    five_old_s4 = float(s4["five_old_return"].add(1).prod() - 1)
    five_new_s4 = float(s4["five_floor_return"].add(1).prod() - 1)
    print("\nS4 五因子复利：旧 %.2f%% → 新 %.2f%%" % (
        five_old_s4 * 100, five_new_s4 * 100))
    assert five_new_s4 > five_old_s4
    # 分段复利必须与全期净值一致
    for name in PORTFOLIOS:
        compounded = float(np.prod(
            1 + result["segments"].loc[result["segments"]["portfolio"] == name,
                                       "total_return"]) - 1)
        total = float(result["full"].loc[result["full"]["portfolio"] == name,
                                         "total_return"].iloc[0])
        assert abs(compounded - total) < 1e-9, name

    paths = save_union_fix_reports(result)
    assert len(pd.read_csv(paths["full"])) == len(PORTFOLIOS)
    assert len(pd.read_csv(paths["segments"])) == len(PORTFOLIOS) * 5
    assert len(pd.read_csv(paths["s4"])) == 2
    payload = json.loads(paths["nav"].read_text(encoding="utf-8"))
    assert len(payload["segment_nav"]) == 5
    print("\n报告已保存：")
    for path in paths.values():
        print(path)
    print("\n联调清单：1 动量下限生效 → 2 全期重跑 → 3 五段重跑 → 4 S4 对比 → 5 落盘，均已完成")
    print("smart_beta_union_fix.py 自测通过；%s" % DISCLAIMER)


def _run_sensitivity_main():
    """I2-alt-9 敏感性入口：10%/15%/20%/30% 四档全期 + 五段。"""
    print(DISCLAIMER)
    synthetic_floor_grid_test()
    result = run_floor_sensitivity(floors=SENS_FLOORS, months_per_segment=6)
    print_sensitivity_report(result)

    periods = result["periods"]
    assert len(periods) == 30, len(periods)
    assert result["segments"]["segment"].nunique() == 5
    assert result["segments"]["portfolio"].nunique() == len(result["names"]) == 10
    # 每一档的并集期都必须达到自身下限
    for floor in result["floors"]:
        tag = _floor_tag(floor)
        rows = periods[periods["five_%s_rule" % tag] == "union_top"]
        assert len(rows) > 0
        assert (rows["five_%s_momentum_weight" % tag] >= floor - 1e-9).all(), floor
    # 旧规则结果必须与 I2-alt-9 完全一致
    old_five = float(result["full"].loc[
        result["full"]["portfolio"] == "five_old", "total_return"].iloc[0])
    assert abs(old_five - 0.4096875035104999) < 1e-9, old_five
    # 分段复利与全期净值一致
    for name in result["names"]:
        compounded = float(np.prod(
            1 + result["segments"].loc[result["segments"]["portfolio"] == name,
                                       "total_return"]) - 1)
        total = float(result["full"].loc[result["full"]["portfolio"] == name,
                                         "total_return"].iloc[0])
        assert abs(compounded - total) < 1e-9, name
    # 下限越高，平均动量权重不应下降
    five = result["sensitivity"][result["sensitivity"]["factor"] == "five"].sort_values("floor")
    assert list(five["avg_momentum_weight"]) == sorted(five["avg_momentum_weight"])

    paths = save_sensitivity_reports(result)
    assert len(pd.read_csv(paths["sensitivity"])) == 10
    assert len(pd.read_csv(paths["segments"])) == 50
    payload = json.loads(paths["nav"].read_text(encoding="utf-8"))
    assert len(payload["segment_nav"]) == 5
    print("\n报告已保存：")
    for path in paths.values():
        print(path)
    print("\n联调清单：1 四档参数生效 → 2 全期重跑 → 3 五段重跑 → 4 安全下限判定 → 5 落盘，均已完成")
    print("smart_beta_union_fix.py 敏感性自测通过；%s" % DISCLAIMER)


if __name__ == "__main__":
    db.init_db()
    if len(sys.argv) > 1 and sys.argv[1] == "sensitivity":
        _run_sensitivity_main()
    else:
        _run_union_fix_main()
