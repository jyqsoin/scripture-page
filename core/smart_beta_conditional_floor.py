# -*- coding: utf-8 -*-
"""五段条件化动量下限 + 四因子口径核对/优化（Step I2-alt-13）。

任务 A：在五因子“并集分支动量最低权重 10%”基础上增加条件开关：
- 条件口径：调仓日只使用此前已实现的横截面动量 IC（Spearman 排名相关）；
- 近 window 期 IC 均值严格大于 0 时启用 10% 下限，否则退回旧并集规则；
- 历史不足 window 期时按“启用”处理（warmup_on=True），与主基线对齐；
- window 取 3 / 6 两档，与“全时启用”“始终不启用”同数据对比五段通过率与 S5 回撤。

任务 B：核对“低 PE/PB 与四/五因子对比”下四因子的实际口径：
- 界面/固化基线中 momentum_floor 只传给五因子，四因子始终为旧并集规则（回撤 10.72%）；
- I2-alt-9 敏感性表对四因子也加了 10% 下限（回撤 12.45%）；
- 本脚本在同一批调仓日上同时复现两者，并测试四因子的其他约束变体
  （5% 下限、仅保只数不调权重、并集扩至 40 只），目标全期回撤 ≤ 基准 11.95%。

不改动任何既有核心模块；五因子主基线（全时 10% 下限）只做回归核对，不参与调参。
防未来函数：IC 只用已实现的历史区间收益；调仓日只使用当日及之前的行情/估值。
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
    DEFAULT_MOMENTUM_WINDOW, DEFAULT_MOMENTUM_SKIP, momentum_scores,
)
from core.smart_beta_backtest import (
    SMART_BETA_DIR, period_returns, equal_weight_return,
    performance_metrics, holding_turnover,
)
from core.smart_beta_3factor_backtest import weighted_return, weighted_turnover
from core.smart_beta_fundamental_backtest import (
    build_fundamental_composite, _sleeve_weights,
)
from core.smart_beta_union_fix import stability_table
from core.smart_beta_walkforward import assign_segments
from core.valuation_data import load_valuation

COND_FLOOR = 0.10
IC_WINDOWS = (3, 6)
DOMINANT_SHARE = 0.60

# I2-alt-9 / I2-alt-10 已落盘报告值（同数据、同调仓日），用于 1e-9 回归核对
REFERENCE = {
    "five_floor": {
        "total_return": 0.39115649186590207,
        "annual_return": 0.14117012781891436,
        "max_drawdown": 0.10759067992734539,
    },
    "five_old": {
        "total_return": 0.40968750351049990,
        "annual_return": 0.14722641983967820,
        "max_drawdown": 0.10241368053467859,
    },
    "four_old": {
        "total_return": 0.39038722042089624,
        "annual_return": 0.14091767160060176,
        "max_drawdown": 0.10723814585322256,
    },
    "four_floor10": {
        "total_return": 0.34835728480588823,
        "annual_return": 0.12699501317242978,
        "max_drawdown": 0.12452410162395744,
    },
    "benchmark": {
        "total_return": 0.46556539015863050,
        "annual_return": 0.16520430407382602,
        "max_drawdown": 0.11952704787716517,
    },
}


def _window_tag(window):
    return "w%d" % int(window)


def _conditional_names(windows=IC_WINDOWS):
    return tuple("five_cond_%s" % _window_tag(window) for window in windows)


def _labels(windows=IC_WINDOWS):
    labels = {
        "five_always": "五因子·全时10%下限（主基线）",
        "five_never": "五因子·不启用下限（旧并集）",
        "four_old": "四因子·不做并集修复",
        "four_floor05": "四因子·动量≥5%（完整修复）",
        "four_floor10": "四因子·动量≥10%（完整修复）",
        "four_floor10_slots": "四因子·动量≥10%·仅保只数",
        "four_top40": "四因子·旧规则·并集扩至40只",
        "benchmark": "池内等权基准",
    }
    for window in windows:
        labels["five_cond_%s" % _window_tag(window)] = (
            "五因子·条件化：近%d期IC>0" % int(window))
    return labels


def decide_floor(ic_history, window, floor=COND_FLOOR, warmup_on=True):
    """按“近 window 期已实现 IC 均值 > 0”决定是否启用动量下限。

    ic_history 只包含当前调仓日之前已经实现（区间收益已知）的 IC，
    因此不会使用当期或未来信息。历史不足时按 warmup_on 回退。
    """
    if window < 1:
        raise ValueError("window 必须为正整数")
    finite = [value for value in ic_history if np.isfinite(value)]
    if len(finite) < window:
        return floor if warmup_on else None
    recent = np.asarray(finite[-window:], dtype=float)
    return floor if float(recent.mean()) > 0 else None


def _composite(data_map, valuation_map, trade_date, include_pb, pct,
               vol_window, momentum_window, momentum_skip, floor,
               union_top_n=30):
    """统一构造四/五因子组合；floor=None 与既有核心模块默认行为一致。"""
    return build_fundamental_composite(
        data_map, valuation_map, trade_date, True, include_pb, pct, vol_window,
        momentum_window, momentum_skip, union_top_n=union_top_n,
        momentum_floor=floor)


def _momentum_weight(weights, momentum_set):
    """按权重合计的动量暴露；用于校验下限是否真正生效。"""
    if not momentum_set:
        return 0.0
    mask = weights.index.isin(momentum_set)
    return float(weights[mask].sum())


def _segment_label(index, frame):
    return "S%d %s~%s" % (index + 1, frame.iloc[0]["date"],
                          frame.iloc[-1]["next_date"])


def _segment_row(segments, name, prefix):
    """取指定组合某一段的绩效；缺失时返回 NaN 占位。"""
    matched = segments[(segments["portfolio"] == name)
                       & segments["segment_label"].str.startswith(prefix)]
    if matched.empty:
        return {"total_return": np.nan, "annual_return": np.nan,
                "max_drawdown": np.nan, "passed": False}
    row = matched.iloc[0]
    return {"total_return": float(row["total_return"]),
            "annual_return": float(row["annual_return"]),
            "max_drawdown": float(row["max_drawdown"]),
            "passed": bool(row["passed"])}


def run_conditional_analysis(
        codes=None, start_date="2024-05-31", end_date="2026-11-13",
        pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, fq="qfq", floor=COND_FLOOR,
        ic_windows=IC_WINDOWS, months_per_segment=6):
    """同一批调仓日上同时回测五因子条件化与四因子约束变体。"""
    windows = tuple(int(window) for window in ic_windows)
    if any(window < 1 for window in windows):
        raise ValueError("IC 滚动窗口必须为正整数")
    if not 0 < floor < 1:
        raise ValueError("动量下限必须在 (0, 1) 内")

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

    labels = _labels(windows)
    cond_names = _conditional_names(windows)
    five_names = ("five_always", "five_never") + cond_names
    four_names = ("four_old", "four_floor05", "four_floor10",
                  "four_floor10_slots", "four_top40")
    names = list(five_names + four_names)
    nav_map = {name: [1.0] for name in names + ["benchmark",]}
    turnover_map = {name: [] for name in names}
    previous = {name: pd.Series(dtype=float) for name in names}
    seg_nav, seg_turnover = {}, {}
    ic_history, period_rows = [], []

    for i in range(len(dates) - 1):
        trade_date, next_date = dates[i], dates[i + 1]
        segment = seg_of[trade_date]
        seg_nav.setdefault(segment, {name: [1.0]
                                     for name in names + ["benchmark",]})
        seg_turnover.setdefault(segment, {name: [] for name in names})

        returns = period_returns(data_map, trade_date, next_date)
        benchmark_return = equal_weight_return(returns, list(data_map))
        momentum = momentum_scores(
            data_map, trade_date, momentum_window, momentum_skip)
        ic_frame = pd.DataFrame({
            "momentum": momentum,
            "return": returns.reindex(momentum.index),
        }).dropna()
        ic = (float(ic_frame["momentum"].corr(ic_frame["return"],
                                              method="spearman"))
              if len(ic_frame) >= 5 else np.nan)

        # 条件开关只使用 ic_history 中此前已实现的 IC，当前期 IC 尚未入列
        decisions = {window: decide_floor(ic_history, window, floor)
                     for window in windows}

        five_always = _composite(
            data_map, valuation_map, trade_date, True, pct, vol_window,
            momentum_window, momentum_skip, floor)
        five_never = _composite(
            data_map, valuation_map, trade_date, True, pct, vol_window,
            momentum_window, momentum_skip, None)
        four_old = _composite(
            data_map, valuation_map, trade_date, False, pct, vol_window,
            momentum_window, momentum_skip, None)
        four_floor05 = _composite(
            data_map, valuation_map, trade_date, False, pct, vol_window,
            momentum_window, momentum_skip, 0.05)
        four_floor10 = _composite(
            data_map, valuation_map, trade_date, False, pct, vol_window,
            momentum_window, momentum_skip, floor)
        four_top40 = _composite(
            data_map, valuation_map, trade_date, False, pct, vol_window,
            momentum_window, momentum_skip, None, union_top_n=40)
        # 仅保留“强制动量入选只数”，不做权重上调：用同一入选名单重算原始份额
        slots_weights = _sleeve_weights(
            four_floor10["sleeves"], four_floor10["allocations"],
            four_floor10["selected"])

        weights_map = {
            "five_always": (five_always["weights"],
                            set(five_always["sleeves"]["momentum"])),
            "five_never": (five_never["weights"],
                           set(five_never["sleeves"]["momentum"])),
            "four_old": (four_old["weights"],
                         set(four_old["sleeves"]["momentum"])),
            "four_floor05": (four_floor05["weights"],
                             set(four_floor05["sleeves"]["momentum"])),
            "four_floor10": (four_floor10["weights"],
                             set(four_floor10["sleeves"]["momentum"])),
            "four_floor10_slots": (slots_weights,
                                   set(four_floor10["sleeves"]["momentum"])),
            "four_top40": (four_top40["weights"],
                           set(four_top40["sleeves"]["momentum"])),
        }
        for window in windows:
            chosen = five_always if decisions[window] else five_never
            weights_map["five_cond_%s" % _window_tag(window)] = (
                chosen["weights"], set(chosen["sleeves"]["momentum"]))

        row = {
            "date": trade_date, "next_date": next_date, "segment": segment,
            "benchmark_return": benchmark_return, "momentum_ic": ic,
        }
        for name in names:
            weights, momentum_set = weights_map[name]
            value = weighted_return(returns, weights)
            turnover = weighted_turnover(previous[name], weights)
            previous[name] = weights.copy()
            nav_map[name].append(nav_map[name][-1] * (1 + value))
            seg_nav[segment][name].append(
                seg_nav[segment][name][-1] * (1 + value))
            seg_turnover[segment][name].append(turnover)
            turnover_map[name].append(turnover)
            row[name + "_return"] = value
            row[name + "_momentum_weight"] = _momentum_weight(
                weights, momentum_set)
        for window in windows:
            name = "five_cond_%s" % _window_tag(window)
            row[name + "_floor_on"] = decisions[window] is not None
            row[name + "_floor"] = (decisions[window]
                                    if decisions[window] is not None else 0.0)

        for prefix, exposure in (
                ("four_old", four_old), ("four_floor05", four_floor05),
                ("four_floor10", four_floor10), ("four_top40", four_top40),
                ("five_always", five_always), ("five_never", five_never)):
            row[prefix + "_rule"] = exposure["selection_rule"]
            row[prefix + "_intersection"] = exposure["counts"]["intersection"]
            row[prefix + "_selected"] = exposure["counts"]["selected"]
            row[prefix + "_momentum_selected"] = exposure["momentum_selected"]

        nav_map["benchmark"].append(
            nav_map["benchmark"][-1] * (1 + benchmark_return))
        seg_nav[segment]["benchmark"].append(
            seg_nav[segment]["benchmark"][-1] * (1 + benchmark_return))
        period_rows.append(row)
        # 当前期 IC 已实现，下一期决策才允许使用
        ic_history.append(ic)

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
            "excess_return": metrics["total_return"]
            - benchmark_metrics["total_return"],
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
                    for name in names + ["benchmark",]},
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
            "momentum_floor": floor, "ic_windows": list(windows),
            "months_per_segment": months_per_segment, "freq": "M", "fq": fq,
            "conditional_rule": "近N期已实现动量IC（Spearman）均值>0时启用"
                                "下限，否则退回旧并集规则；历史不足N期按启用处理",
            "no_future_note": "当期IC在区间收益实现后才计入历史，下一期决策才使用",
            "cost_note": "与 I2-alt-9/10 口径一致，收益未扣交易成本",
            "verdict_note": "年化收益大于0且最大回撤不超过同期池内等权基准",
            "stability_note": "全部段通过且无单段对数收益占比超过%.0f%%记为稳定"
                              % (DOMINANT_SHARE * 100),
            "reference": REFERENCE,
            "disclaimer": DISCLAIMER,
        },
        "full": full, "segments": segments, "stability": stability,
        "periods": periods, "full_nav": full_nav, "nav": nav_payload,
        "benchmark_metrics": benchmark_metrics,
        "five_names": five_names, "four_names": four_names,
        "labels": labels, "ic_windows": windows,
    }
    result["conditional_summary"] = conditional_summary(result)
    result["four_reconciliation"] = four_reconciliation(result)
    result["four_variants"] = four_variants(result)
    return result


def conditional_summary(result):
    """五因子条件化 vs 全时/不启用：全期、五段通过率与 S5 回撤。"""
    full = result["full"].set_index("portfolio")
    segments = result["segments"]
    periods = result["periods"]
    benchmark_dd = result["benchmark_metrics"]["max_drawdown"]
    rows = []
    for name in result["five_names"]:
        subset = segments[segments["portfolio"] == name]
        s1 = _segment_row(segments, name, "S1")
        s5 = _segment_row(segments, name, "S5")
        row = {
            "portfolio": name, "label": result["labels"][name],
            "annual_return": float(full.loc[name, "annual_return"]),
            "total_return": float(full.loc[name, "total_return"]),
            "max_drawdown": float(full.loc[name, "max_drawdown"]),
            "sharpe": float(full.loc[name, "sharpe"]),
            "drawdown_not_above_benchmark": bool(
                full.loc[name, "max_drawdown"] <= benchmark_dd),
            "full_passed": bool(full.loc[name, "passed"]),
            "passed_segments": int(subset["passed"].sum()),
            "pass_rate": float(subset["passed"].sum() / len(subset)),
            "s1_total_return": s1["total_return"],
            "s5_total_return": s5["total_return"],
            "s5_annual_return": s5["annual_return"],
            "s5_max_drawdown": s5["max_drawdown"],
            "avg_momentum_weight": float(
                periods[name + "_momentum_weight"].mean()),
        }
        if name.startswith("five_cond_"):
            row["floor_on_periods"] = int(periods[name + "_floor_on"].sum())
            row["floor_on_ratio"] = float(periods[name + "_floor_on"].mean())
        else:
            row["floor_on_periods"] = len(periods) if name == "five_always" else 0
            row["floor_on_ratio"] = 1.0 if name == "five_always" else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    # 显式给出“条件化 vs 全时启用”的差异，正值为变差
    base = summary[summary["portfolio"] == "five_always"].iloc[0]
    summary["delta_annual_vs_always_pp"] = (
        summary["annual_return"] - base["annual_return"]) * 100
    summary["delta_dd_vs_always_pp"] = (
        summary["max_drawdown"] - base["max_drawdown"]) * 100
    summary["delta_s5_annual_vs_always_pp"] = (
        summary["s5_annual_return"] - base["s5_annual_return"]) * 100
    summary["delta_s5_dd_vs_always_pp"] = (
        summary["s5_max_drawdown"] - base["s5_max_drawdown"]) * 100
    summary["delta_passed_segments_vs_always"] = (
        summary["passed_segments"] - int(base["passed_segments"]))
    return summary


def four_reconciliation(result):
    """四因子 10.72% 与敏感性 12.45% 的口径核对表。"""
    full = result["full"].set_index("portfolio")
    periods = result["periods"]
    benchmark_dd = result["benchmark_metrics"]["max_drawdown"]
    benchmark_total = result["benchmark_metrics"]["total_return"]
    specs = (
        ("界面/固化基线·四因子（floor=None）", "four_old",
         "smart_beta_fundamental_backtest.run_fundamental_comparison / "
         "smart_beta_baseline", None, False,
         "momentum_floor 只传给五因子，四因子始终走旧并集规则"),
        ("I2-alt-9 敏感性表·四因子@10%", "four_floor10",
         "smart_beta_union_fix.run_floor_sensitivity", result["meta"]["momentum_floor"],
         True, "敏感性脚本对四因子也加了 10% 下限，故回撤破基准"),
        ("固化基线·五因子@10%（主基线）", "five_always",
         "smart_beta_baseline / smart_beta_cost_backtest", result["meta"]["momentum_floor"],
         True, "五因子主基线，本次只核对不调整"),
    )
    rows = []
    for item, name, source, floor, union_fix, note in specs:
        row = full.loc[name]
        rule_union = int((periods[name + "_rule"] == "union_top").sum())
        rule_intersection = int((periods[name + "_rule"] == "intersection").sum())
        rows.append({
            "item": item, "source": source, "portfolio": name,
            "momentum_floor": np.nan if floor is None else floor,
            "union_fix": bool(union_fix),
            "total_return": float(row["total_return"]),
            "annual_return": float(row["annual_return"]),
            "sharpe": float(row["sharpe"]),
            "max_drawdown": float(row["max_drawdown"]),
            "benchmark_max_drawdown": benchmark_dd,
            "dd_vs_benchmark_pp": float(
                (row["max_drawdown"] - benchmark_dd) * 100),
            "excess_return": float(row["total_return"] - benchmark_total),
            "passed": bool(row["passed"]),
            "union_periods": rule_union,
            "intersection_periods": rule_intersection,
            "avg_momentum_weight": float(
                periods[name + "_momentum_weight"].mean()),
            "note": note,
        })
    # 显式给出两版四因子回撤差，便于核对 12.45% 与 10.72%
    old_dd = float(full.loc["four_old", "max_drawdown"])
    floor_dd = float(full.loc["four_floor10", "max_drawdown"])
    rows.append({
        "item": "差异·四因子@10% − 四因子floor=None", "source": "本脚本复算",
        "portfolio": "four_floor10 - four_old", "momentum_floor": np.nan,
        "union_fix": True, "total_return": np.nan, "annual_return": np.nan,
        "sharpe": np.nan, "max_drawdown": floor_dd - old_dd,
        "benchmark_max_drawdown": benchmark_dd,
        "dd_vs_benchmark_pp": float((floor_dd - benchmark_dd) * 100),
        "excess_return": np.nan, "passed": False,
        "union_periods": np.nan, "intersection_periods": np.nan,
        "avg_momentum_weight": np.nan,
        "note": "回撤差 %.2fpp，来源仅为四因子是否叠加动量下限"
                % ((floor_dd - old_dd) * 100),
    })
    return pd.DataFrame(rows)


def four_variants(result):
    """四因子约束变体对比：目标全期回撤不超过基准。"""
    full = result["full"].set_index("portfolio")
    periods = result["periods"]
    benchmark_dd = result["benchmark_metrics"]["max_drawdown"]
    old_dd = float(full.loc["four_old", "max_drawdown"])
    old_annual = float(full.loc["four_old", "annual_return"])
    specs = (
        ("four_old", "不做并集修复（旧规则）", None, 30, "none"),
        ("four_floor05", "动量≥5%·完整修复", 0.05, 30, "floor_weight"),
        ("four_floor10", "动量≥10%·完整修复", 0.10, 30, "floor_weight"),
        ("four_floor10_slots", "动量≥10%·仅保只数", 0.10, 30, "floor_slots"),
        ("four_top40", "旧规则·并集扩至40只", None, 40, "union_40"),
    )
    rows = []
    for name, label, floor, top_n, fix_mode in specs:
        row = full.loc[name]
        # 仅保只数变体复用四因子@10% 的入选名单与选择规则
        rule_name = "four_floor10" if name == "four_floor10_slots" else name
        rule_union = int((periods[rule_name + "_rule"] == "union_top").sum())
        rule_intersection = int(
            (periods[rule_name + "_rule"] == "intersection").sum())
        rows.append({
            "variant": name, "label": label,
            "momentum_floor": np.nan if floor is None else floor,
            "union_top_n": top_n, "fix_mode": fix_mode,
            "total_return": float(row["total_return"]),
            "annual_return": float(row["annual_return"]),
            "sharpe": float(row["sharpe"]),
            "max_drawdown": float(row["max_drawdown"]),
            "benchmark_max_drawdown": benchmark_dd,
            "dd_vs_benchmark_pp": float(
                (row["max_drawdown"] - benchmark_dd) * 100),
            "meets_target": bool(row["max_drawdown"] <= benchmark_dd),
            "passed": bool(row["passed"]),
            "avg_momentum_weight": float(
                periods[name + "_momentum_weight"].mean()),
            "union_periods": rule_union,
            "intersection_periods": rule_intersection,
            "delta_dd_vs_old_pp": float((row["max_drawdown"] - old_dd) * 100),
            "delta_annual_vs_old_pp": float(
                (row["annual_return"] - old_annual) * 100),
        })
    return pd.DataFrame(rows)


def save_conditional_reports(result, directory=None):
    """保存条件化五段、四因子口径核对与优化对比 CSV/JSON。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    stem = "smart_beta_conditional_floor_%s_%s_%s" % (
        meta["start"].replace("-", ""), meta["end"].replace("-", ""),
        meta["codes"])
    paths = {}
    for key in ("conditional_summary", "four_reconciliation", "four_variants"):
        path = directory / ("%s_%s.csv" % (stem, key))
        result[key].to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    segment_path = directory / ("%s_segments.csv" % stem)
    result["segments"].to_csv(segment_path, index=False, encoding="utf-8-sig")
    paths["segments"] = segment_path
    period_path = directory / ("%s_periods.csv" % stem)
    result["periods"].to_csv(period_path, index=False, encoding="utf-8-sig")
    paths["periods"] = period_path
    stability_path = directory / ("%s_stability.csv" % stem)
    result["stability"].to_csv(stability_path, index=False, encoding="utf-8-sig")
    paths["stability"] = stability_path

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
        "conditional_summary": result["conditional_summary"].to_dict("records"),
        "four_reconciliation": result["four_reconciliation"].to_dict("records"),
        "four_variants": result["four_variants"].to_dict("records"),
        "stability": result["stability"].to_dict("records"),
        "benchmark_metrics": result["benchmark_metrics"],
        "periods": result["periods"].to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def print_conditional_report(result):
    """打印任务 A：IC 概况、条件化汇总与五段明细。"""
    meta = result["meta"]
    benchmark = result["benchmark_metrics"]
    periods = result["periods"]
    ic = periods["momentum_ic"].astype(float)
    print("\n=== 任务A：五因子条件化动量下限（%s ~ %s，%d 只）===" % (
        meta["start"], meta["end"], meta["codes"]))
    print("基准：年化 %.2f%% | 最大回撤 %.2f%% | 动量下限 %.0f%%" % (
        benchmark["annual_return"] * 100, benchmark["max_drawdown"] * 100,
        meta["momentum_floor"] * 100))
    print("已实现动量IC：均值 %.4f | >0 期数 %d/%d | 最低 %.4f | 最高 %.4f" % (
        float(ic.mean()), int((ic > 0).sum()), len(ic),
        float(ic.min()), float(ic.max())))

    summary = result["conditional_summary"]
    print("\n%-26s %9s %9s %8s %6s %9s %9s %9s %8s" % (
        "组合", "全期年化", "全期回撤", "全期判定", "通过段", "S5总收益",
        "S5年化", "S5回撤", "开启下限"))
    for _, row in summary.iterrows():
        print("%-26s %8.2f%% %8.2f%% %8s %5d/5 %8.2f%% %8.2f%% %8.2f%% %5d/%d" % (
            row["label"], row["annual_return"] * 100,
            row["max_drawdown"] * 100,
            "通过" if row["full_passed"] else "未通过",
            row["passed_segments"], row["s5_total_return"] * 100,
            row["s5_annual_return"] * 100, row["s5_max_drawdown"] * 100,
            row["floor_on_periods"], len(periods)))

    print("\n--- 条件化 vs 全时启用差异（百分点，正值为变差）---")
    print("%-26s %10s %10s %10s %10s %8s" % (
        "组合", "Δ全期年化", "Δ全期回撤", "ΔS5年化", "ΔS5回撤", "Δ通过段"))
    for _, row in summary.iterrows():
        print("%-26s %+9.2f %+9.2f %+9.2f %+9.2f %+7d" % (
            row["label"], row["delta_annual_vs_always_pp"],
            row["delta_dd_vs_always_pp"], row["delta_s5_annual_vs_always_pp"],
            row["delta_s5_dd_vs_always_pp"],
            row["delta_passed_segments_vs_always"]))

    print("\n--- 五段通过明细（✓通过 / ✗未通过）---")
    print("%-26s %s" % ("组合", "  ".join(
        ["S1", "S2", "S3", "S4", "S5"])))
    for name in result["five_names"]:
        frame = result["segments"][result["segments"]["portfolio"] == name]
        marks = ["✓" if bool(row["passed"]) else "✗" for _, row in frame.iterrows()]
        dds = ["%.1f%%" % (row["max_drawdown"] * 100) for _, row in frame.iterrows()]
        print("%-26s %s" % (
            result["labels"][name],
            "  ".join("%s %s" % (mark, dd) for mark, dd in zip(marks, dds))))

    print("\n--- 逐期 IC 与条件化决策（w3/w6）---")
    print("%-12s %8s %8s %6s %8s %6s" % (
        "调仓日", "动量IC", "w3下限", "w3开", "w6下限", "w6开"))
    for _, row in periods.iterrows():
        print("%-12s %8.3f %8s %6s %8s %6s" % (
            row["date"], row["momentum_ic"],
            "10%" if row["five_cond_w3_floor"] else "—",
            "是" if row["five_cond_w3_floor_on"] else "否",
            "10%" if row["five_cond_w6_floor"] else "—",
            "是" if row["five_cond_w6_floor_on"] else "否"))


def print_four_report(result):
    """打印任务 B：口径核对与四因子约束优化对比。"""
    benchmark = result["benchmark_metrics"]
    print("\n=== 任务B：四因子口径核对 ===")
    print("基准：年化 %.2f%% | 最大回撤 %.2f%%" % (
        benchmark["annual_return"] * 100, benchmark["max_drawdown"] * 100))
    print("%-40s %9s %9s %9s %8s %6s %6s" % (
        "口径", "总收益", "年化", "最大回撤", "超基准pp", "并集期", "判定"))
    for _, row in result["four_reconciliation"].iterrows():
        if not np.isfinite(row["total_return"]):
            print("%-40s %9s %9s %8.2f%% %7.2f" % (
                row["item"], "—", "—", row["max_drawdown"] * 100,
                row["dd_vs_benchmark_pp"]))
            continue
        print("%-40s %8.2f%% %8.2f%% %8.2f%% %7.2f %6d %6s" % (
            row["item"], row["total_return"] * 100,
            row["annual_return"] * 100, row["max_drawdown"] * 100,
            row["dd_vs_benchmark_pp"], row["union_periods"],
            "通过" if row["passed"] else "未通过"))

    print("\n=== 四因子约束优化对比（目标：全期回撤 ≤ %.2f%%）===" % (
        benchmark["max_drawdown"] * 100))
    print("%-26s %9s %9s %9s %9s %8s %8s %6s" % (
        "变体", "总收益", "年化", "最大回撤", "较旧规则", "平均动量", "并集期", "达标"))
    for _, row in result["four_variants"].iterrows():
        print("%-26s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %7.2f%% %8d %6s" % (
            row["label"], row["total_return"] * 100,
            row["annual_return"] * 100, row["max_drawdown"] * 100,
            row["delta_dd_vs_old_pp"], row["avg_momentum_weight"] * 100,
            row["union_periods"], "是" if row["meets_target"] else "否"))


def _synthetic_market(codes=60, days=400):
    """构造“动量股价格高、波动大、估值高”的合成市场（复用 I2-alt-9）。"""
    dates = pd.bdate_range(end="2026-05-29", periods=days).strftime(
        "%Y-%m-%d").tolist()
    rng = np.random.default_rng(7)
    data_map, valuation_map = {}, {}
    for index in range(codes):
        code = "%06d" % (600000 + index)
        if index < 12:
            steps = np.concatenate([[0.0], rng.normal(0.012, 0.030, days - 1)])
            close = 5.0 * np.exp(np.cumsum(steps))
            pe, pb = 200.0, 15.0
        else:
            steps = np.concatenate([[0.0], rng.normal(0.0001, 0.004, days - 1)])
            close = (5.0 + index * 0.15) * np.exp(np.cumsum(steps))
            pe = 6.0 + (59 - index) * 0.08
            pb = 0.6 + (59 - index) * 0.02
        data_map[code] = pd.DataFrame({"date": dates, "close": close})
        valuation_map[code] = pd.DataFrame({
            "date": [dates[0], dates[-1]], "code": [code, code],
            "pe_ttm": [pe, pe], "pb_mrq": [pb, pb],
            "tradestatus": [1, 1], "is_st": [0, 0],
        })
    return data_map, valuation_map, dates[-1]


def synthetic_conditional_test():
    """合成测试条件开关边界：历史不足、正/负 IC、严格大于 0。"""
    assert decide_floor([], 3) == COND_FLOOR
    assert decide_floor([], 3, warmup_on=False) is None
    assert decide_floor([0.10, 0.20], 3) == COND_FLOOR
    assert decide_floor([0.10, 0.20, 0.05], 3) == COND_FLOOR
    assert decide_floor([-0.10, -0.20, -0.05], 3) is None
    # 均值恰好为 0 时不启用（要求严格大于 0）
    assert decide_floor([0.10, -0.10, 0.0], 3) is None
    assert decide_floor([0.30, -0.10, -0.05, -0.01], 3) is None
    assert decide_floor([0.10, 0.20, 0.05, 0.01], 3) == COND_FLOOR
    # 只用最近 window 期：更早的正 IC 不应把最近为负的状态翻正
    assert decide_floor([0.30, 0.30, -0.01, -0.02, -0.03], 3) is None
    print("合成数据：条件开关窗口/边界/严格大于0判定，均通过")


def synthetic_slots_test():
    """合成测试“仅保只数不调权重”：入选名单相同但动量权重低于完整修复。"""
    data_map, valuation_map, trade_date = _synthetic_market()
    full = _composite(data_map, valuation_map, trade_date, False, DEFAULT_PCT,
                      DEFAULT_VOL_WINDOW, DEFAULT_MOMENTUM_WINDOW,
                      DEFAULT_MOMENTUM_SKIP, COND_FLOOR)
    slots = _sleeve_weights(full["sleeves"], full["allocations"],
                            full["selected"])
    momentum_set = set(full["sleeves"]["momentum"])
    full_weight = _momentum_weight(full["weights"], momentum_set)
    slots_weight = _momentum_weight(slots, momentum_set)
    assert abs(full_weight - COND_FLOOR) < 1e-9, full_weight
    assert slots_weight < COND_FLOOR - 1e-6, slots_weight
    assert abs(float(slots.sum()) - 1) < 1e-12
    assert set(slots.index) == set(full["weights"].index)
    print("合成数据：仅保只数模式入选名单一致，动量权重 %.2f%% < 完整修复 %.0f%%，通过"
          % (slots_weight * 100, COND_FLOOR * 100))


def verify_real_data(result):
    """真实数据回归：五因子主基线 1e-9、条件化生效、四因子口径复现。"""
    full = result["full"].set_index("portfolio")
    periods = result["periods"]
    segments = result["segments"]

    for key, name in (("five_floor", "five_always"), ("five_old", "five_never"),
                      ("four_old", "four_old"),
                      ("four_floor10", "four_floor10")):
        for metric in ("total_return", "annual_return", "max_drawdown"):
            actual = float(full.loc[name, metric])
            expected = REFERENCE[key][metric]
            assert abs(actual - expected) < 1e-9, (
                "%s.%s 偏差 %.12f" % (name, metric, actual - expected))

    benchmark_dd = result["benchmark_metrics"]["max_drawdown"]
    assert abs(benchmark_dd - REFERENCE["benchmark"]["max_drawdown"]) < 1e-9
    # 四因子两种口径：10.72% 通过、12.45% 破基准
    assert float(full.loc["four_old", "max_drawdown"]) <= benchmark_dd
    assert float(full.loc["four_floor10", "max_drawdown"]) > benchmark_dd

    # 条件化：开启期动量权重不低于下限；关闭期与 five_never 逐期收益完全一致
    for window in result["ic_windows"]:
        name = "five_cond_w%d" % int(window)
        on = periods[name + "_floor_on"]
        assert on.any() and (~on).any(), "条件化未发生切换：%s" % name
        assert (periods.loc[on, name + "_momentum_weight"]
                >= result["meta"]["momentum_floor"] - 1e-9).all()
        closed = periods.loc[~on]
        assert np.allclose(closed[name + "_return"],
                           closed["five_never_return"], atol=1e-12, rtol=0)
        opened = periods.loc[on]
        assert np.allclose(opened[name + "_return"],
                           opened["five_always_return"], atol=1e-12, rtol=0)

    # 分段复利必须与全期净值一致
    for name in result["five_names"] + result["four_names"]:
        compounded = float(np.prod(
            1 + segments.loc[segments["portfolio"] == name,
                             "total_return"]) - 1)
        total = float(full.loc[name, "total_return"])
        assert abs(compounded - total) < 1e-9, name

    assert result["segments"]["segment"].nunique() == 5
    assert len(periods) == 30
    print("真实数据回归：五因子主基线误差<1e-9，条件化切换生效，"
          "四因子 10.72%/12.45% 口径复现，五段复利一致，均通过")


if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()
    synthetic_conditional_test()
    synthetic_slots_test()
    result = run_conditional_analysis(
        start_date="2024-05-31", end_date="2026-11-13",
        ic_windows=IC_WINDOWS, months_per_segment=6)
    print_conditional_report(result)
    print_four_report(result)
    paths = save_conditional_reports(result)
    assert len(pd.read_csv(paths["conditional_summary"])) == 4
    assert len(pd.read_csv(paths["segments"])) == 45
    assert len(pd.read_csv(paths["four_reconciliation"])) == 4
    assert len(pd.read_csv(paths["four_variants"])) == 5
    payload = json.loads(paths["nav"].read_text(encoding="utf-8"))
    assert len(payload["segment_nav"]) == 5
    print("\n报告已保存：")
    for path in paths.values():
        print(path)
    verify_real_data(result)
    print("\n联调清单：1 条件开关 → 2 五段重跑 → 3 四因子口径复现 → "
          "4 四因子约束对比 → 5 落盘与末尾自测，均已完成")
    print("smart_beta_conditional_floor.py 自测通过；%s" % DISCLAIMER)
