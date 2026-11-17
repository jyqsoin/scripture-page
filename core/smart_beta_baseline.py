# -*- coding: utf-8 -*-
"""最终配置固化与基线重跑（Step I2-alt-10）。

固化的最终口径（只在本脚本写死，不改任何核心模块默认值）：
- 五因子：并集分支动量最低权重 10%（交集分支不强制）；
- 四因子：维持旧并集规则，不加动量下限；
- 双因子 / 三因子：保持既有配置，作为回归基准。

固化依据（I2-alt-9 四档敏感性）：
- 10% 档五因子全期回撤 10.76% ≤ 基准 11.95%，余量 1.19pp，S4 亏损收敛；
- 20% 档全期回撤 12.36% 破线，30% 档 S5 年化转负，均不采用；
- 四因子在各档回撤均破线，暂不启用下限，留待后续单独优化。

运行口径与 I2-alt-9 完全一致：月频调仓、调仓日收盘选股、下月调仓日收盘计
收益、不做成本扣减（与既有修复验证报告可比），每 6 个自然月分段独立统计。

防未来函数：每个调仓日只用该日及之前的本地行情 / 估值；分段只影响绩效聚合。
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

FINAL_FLOOR = 0.10
DOMINANT_SHARE = 0.60
# four_factor / five_factor 为固化后的最终配置；five_old 仅作固化前对照
PORTFOLIOS = ("dual", "three_equal", "four_factor", "five_old", "five_factor")
PORTFOLIO_LABELS = {
    "dual": "双因子50:50",
    "three_equal": "三因子等权33/33/34",
    "four_factor": "四因子·旧并集规则（最终）",
    "five_old": "五因子·固化前（旧规则）",
    "five_factor": "五因子·动量≥10%（最终）",
    "benchmark": "池内等权基准",
}

# I2-alt-9 10% 口径报告值，用于回归校验（同数据、同调仓日应为逐值一致）
REFERENCE = {
    "dual": {"total_return": 0.26585878707476795,
             "annual_return": 0.09888970389537644,
             "max_drawdown": 0.11860295941224519},
    "three_equal": {"total_return": 0.29173221942802610,
                    "annual_return": 0.10781947832272220,
                    "max_drawdown": 0.09468470564342113},
    "four_factor": {"total_return": 0.39038722042089624,
                    "annual_return": 0.14091767160060176,
                    "max_drawdown": 0.10723814585322256},
    "five_old": {"total_return": 0.40968750351049990,
                 "annual_return": 0.14722641983967820,
                 "max_drawdown": 0.10241368053467859},
    "five_factor": {"total_return": 0.39115649186590207,
                    "annual_return": 0.14117012781891436,
                    "max_drawdown": 0.10759067992734539},
}


def _composite(data_map, valuation_map, trade_date, include_pb, pct,
               vol_window, momentum_window, momentum_skip, floor):
    """统一构造四/五因子组合；floor=None 时与核心模块默认调用逐值一致。"""
    return build_fundamental_composite(
        data_map, valuation_map, trade_date, True, include_pb, pct, vol_window,
        momentum_window, momentum_skip, momentum_floor=floor)


def _segment_label(index, frame):
    return "S%d %s~%s" % (index + 1, frame.iloc[0]["date"],
                          frame.iloc[-1]["next_date"])


def _score_value(scores, column, code):
    """组合评分列可能缺失维度，统一返回 NaN。"""
    return scores[column].get(code) if column in scores else np.nan


def run_baseline(
        codes=None, start_date="2024-05-31", end_date="2026-11-13",
        pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, fq="qfq",
        months_per_segment=6):
    """固化配置下的全期 + 分段回测，并输出四/五因子持仓明细。"""
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
                for name in ("three_equal", "four_factor", "five_old",
                             "five_factor")}
    previous_dual = {"low_price": [], "low_vol": []}
    seg_nav, seg_turnover = {}, {}
    period_rows, holdings_rows = [], []

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
        # 固化配置：四因子不加下限，五因子写死 10%，另跑五因子旧规则作对照
        four = _composite(data_map, valuation_map, trade_date, False,
                          pct, vol_window, momentum_window, momentum_skip, None)
        five_old = _composite(data_map, valuation_map, trade_date, True,
                              pct, vol_window, momentum_window,
                              momentum_skip, None)
        five_final = _composite(data_map, valuation_map, trade_date, True,
                                pct, vol_window, momentum_window,
                                momentum_skip, FINAL_FLOOR)

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
        for name, exposure in (("three_equal", three), ("four_factor", four),
                               ("five_old", five_old),
                               ("five_factor", five_final)):
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
            turnover_map[name].append(turnovers[name])
        nav_map["benchmark"].append(
            nav_map["benchmark"][-1] * (1 + benchmark_return))
        seg_nav[segment]["benchmark"].append(
            seg_nav[segment]["benchmark"][-1] * (1 + benchmark_return))

        row = {"date": trade_date, "next_date": next_date, "segment": segment,
               "benchmark_return": benchmark_return}
        for name in PORTFOLIOS:
            row[name + "_return"] = portfolio_returns[name]
            row[name + "_turnover"] = turnovers[name]
        for prefix, exposure in (("four_factor", four), ("five_old", five_old),
                                 ("five_factor", five_final)):
            row[prefix + "_rule"] = exposure["selection_rule"]
            row[prefix + "_intersection"] = exposure["counts"]["intersection"]
            row[prefix + "_selected"] = exposure["counts"]["selected"]
            row[prefix + "_momentum_count"] = exposure["momentum_selected"]
            row[prefix + "_momentum_weight"] = exposure["momentum_weight"]
        period_rows.append(row)

        # 持仓明细只落最终配置（四因子 + 五因子），逐期记录权重与因子分位
        for name, exposure in (("four_factor", four), ("five_factor", five_final)):
            scores = exposure["scores"]
            momentum_set = set(exposure["sleeves"].get("momentum", []))
            for code, weight in exposure["weights"].items():
                holdings_rows.append({
                    "date": trade_date, "next_date": next_date,
                    "segment": segment, "portfolio": name,
                    "code": code, "weight": float(weight),
                    "selection_rule": exposure["selection_rule"],
                    "in_momentum": code in momentum_set,
                    "price": _score_value(scores, "price", code),
                    "volatility": _score_value(scores, "volatility", code),
                    "momentum": _score_value(scores, "momentum", code),
                    "pe_ttm": _score_value(scores, "pe_ttm", code),
                    "pb_mrq": _score_value(scores, "pb_mrq", code),
                    "period_return": returns.get(code),
                })

    periods = pd.DataFrame(period_rows)
    labels = {segment: _segment_label(segment, frame)
              for segment, frame in periods.groupby("segment")}
    periods["segment_label"] = periods["segment"].map(labels)

    full_nav = pd.DataFrame(nav_map, index=dates)
    benchmark_metrics = performance_metrics(full_nav["benchmark"])
    full_rows = []
    for name in PORTFOLIOS:
        metrics = performance_metrics(full_nav[name], turnover_map[name])
        beat_cash = metrics["annual_return"] > 0
        controlled = metrics["max_drawdown"] <= benchmark_metrics["max_drawdown"]
        full_rows.append({
            "portfolio": name, "label": PORTFOLIO_LABELS[name],
            "final_config": bool(name in ("four_factor", "five_factor")),
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
    full = pd.DataFrame(full_rows)

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
    compare = compare_table(full, benchmark_metrics)

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": len(data_map), "start": dates[0], "end": dates[-1],
            "pct": pct, "vol_window": vol_window,
            "momentum_window": momentum_window, "momentum_skip": momentum_skip,
            "months_per_segment": months_per_segment, "freq": "M", "fq": fq,
            "final_config": {
                "five_factor": {"momentum_floor": FINAL_FLOOR,
                                "rule": "并集分支强制动量入选 3 只且权重≥10%"},
                "four_factor": {"momentum_floor": None,
                                "rule": "维持旧并集规则，不加下限"},
                "dual": {"momentum_floor": None, "rule": "50:50 双因子"},
                "three_equal": {"momentum_floor": None,
                                "rule": "三因子等权 33/33/34"},
            },
            "config_note": "最终配置只写死在基线脚本中，核心模块默认值仍为旧规则",
            "cost_note": "与 I2-alt-9 口径一致，收益未扣交易成本",
            "verdict_note": "年化收益大于0且最大回撤不超过同期池内等权基准",
            "stability_note": "全部段通过且无单段对数收益占比超过%.0f%%记为稳定"
                              % (DOMINANT_SHARE * 100),
            "disclaimer": DISCLAIMER,
        },
        "full": full, "segments": segments, "stability": stability,
        "compare": compare, "periods": periods,
        "holdings": pd.DataFrame(holdings_rows),
        "full_nav": full_nav, "nav": nav_payload,
        "benchmark_metrics": benchmark_metrics,
    }


def stability_table(segments, portfolios=PORTFOLIOS, labels=None):
    """按组合汇总跨段稳定性，口径与 I2-alt-6/9 一致。"""
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
        if float(np.log1p(np.array(returns)).sum()) <= 0:
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


def _row_of(full, name):
    return full.loc[full["portfolio"] == name].iloc[0]


def compare_table(full, benchmark_metrics):
    """固化前后对比：五因子换配置、四因子不变、双/三因子回归核对。"""
    rows = []
    pairs = (
        ("五因子：旧并集 → 动量≥10%", "five_old", "five_factor",
         "本次重跑旧规则", "本次重跑固化配置，唯一变化是并集分支动量下限"),
        ("四因子：旧规则 → 旧规则", None, "four_factor",
         "I2-alt-9 报告", "最终配置不启用下限，应逐值不变"),
        ("双因子：I2-alt-9 基准 → 本次", None, "dual",
         "I2-alt-9 报告", "固化不改双因子，应逐值一致"),
        ("三因子：I2-alt-9 基准 → 本次", None, "three_equal",
         "I2-alt-9 报告", "固化不改三因子，应逐值一致"),
    )
    for item, before, after, source, note in pairs:
        if before is None:
            before_row = REFERENCE[after]
            before_passed = bool(before_row["annual_return"] > 0
                                 and before_row["max_drawdown"]
                                 <= benchmark_metrics["max_drawdown"])
        else:
            before_row = _row_of(full, before)
            before_passed = bool(before_row["passed"])
        after_row = _row_of(full, after)
        rows.append({
            "item": item,
            "before_source": source,
            "before_total_return": before_row["total_return"],
            "after_total_return": after_row["total_return"],
            "delta_total_return": after_row["total_return"] - before_row["total_return"],
            "before_annual_return": before_row["annual_return"],
            "after_annual_return": after_row["annual_return"],
            "delta_annual_return": (after_row["annual_return"]
                                    - before_row["annual_return"]),
            "before_max_drawdown": before_row["max_drawdown"],
            "after_max_drawdown": after_row["max_drawdown"],
            "delta_max_drawdown": (after_row["max_drawdown"]
                                   - before_row["max_drawdown"]),
            "before_excess_return": before_row["total_return"]
            - benchmark_metrics["total_return"],
            "after_excess_return": after_row["excess_return"],
            "before_passed": before_passed,
            "after_passed": bool(after_row["passed"]),
            "note": note,
        })
    return pd.DataFrame(rows)


def save_baseline_reports(result, directory=None):
    """保存固化基线：全期、逐期、分段、稳定性、固化前后对比、持仓与净值。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    stem = "smart_beta_baseline_%s_%s_%s" % (
        meta["start"].replace("-", ""), meta["end"].replace("-", ""),
        meta["codes"])
    paths = {}
    for key in ("full", "periods", "segments", "stability", "compare"):
        path = directory / ("%s_%s.csv" % (stem, key))
        result[key].to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    holdings_path = directory / (stem + "_holdings.parquet")
    result["holdings"].to_parquet(holdings_path, index=False)
    paths["holdings"] = holdings_path
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
        "compare": result["compare"].to_dict("records"),
        "benchmark_metrics": result["benchmark_metrics"],
        "periods": result["periods"].to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def print_baseline_report(result):
    """打印固化配置的全期、固化前后对比、分段与稳定性。"""
    meta = result["meta"]
    benchmark = result["benchmark_metrics"]
    print("\n=== 最终配置固化与基线重跑 ===")
    print("股票池 %d 只 | %s ~ %s | 各前 %.0f%% | 五因子动量下限 %.0f%% 四因子不下限" % (
        meta["codes"], meta["start"], meta["end"], meta["pct"] * 100,
        FINAL_FLOOR * 100))

    print("\n--- 全期绩效（基准年化 %.2f%%，回撤 %.2f%%）---" % (
        benchmark["annual_return"] * 100, benchmark["max_drawdown"] * 100))
    print("%-30s %9s %9s %8s %9s %9s %8s" % (
        "组合", "总收益", "年化", "夏普", "最大回撤", "超额", "判定"))
    for _, row in result["full"].iterrows():
        print("%-30s %8.2f%% %8.2f%% %8.3f %8.2f%% %8.2f%% %8s" % (
            row["label"], row["total_return"] * 100, row["annual_return"] * 100,
            row["sharpe"], row["max_drawdown"] * 100,
            row["excess_return"] * 100, "通过" if row["passed"] else "未通过"))

    print("\n--- 固化前后对比 ---")
    print("%-30s %10s %10s %9s %10s %8s" % (
        "项目", "固化前总收益", "固化后总收益", "变化pp", "固化后回撤", "判定"))
    for _, row in result["compare"].iterrows():
        print("%-30s %9.2f%% %9.2f%% %8.2f%% %9.2f%% %8s" % (
            row["item"], row["before_total_return"] * 100,
            row["after_total_return"] * 100, row["delta_total_return"] * 100,
            row["after_max_drawdown"] * 100,
            "通过" if row["after_passed"] else "未通过"))

    print("\n--- 分段绩效（最终配置）---")
    for label, frame in result["segments"].groupby("segment_label", sort=True):
        head = frame.iloc[0]
        print("\n%s | 基准总收益 %.2f%% 回撤 %.2f%%" % (
            label, head["benchmark_total_return"] * 100,
            head["benchmark_max_drawdown"] * 100))
        print("%-30s %9s %9s %9s %8s %6s" % (
            "组合", "总收益", "年化", "最大回撤", "超额", "判定"))
        for _, row in frame[frame["portfolio"].isin(
                ("four_factor", "five_factor"))].iterrows():
            print("%-30s %8.2f%% %8.2f%% %8.2f%% %7.2f%% %6s" % (
                row["label"], row["total_return"] * 100,
                row["annual_return"] * 100, row["max_drawdown"] * 100,
                row["excess_return"] * 100, "通过" if row["passed"] else "未通过"))

    print("\n--- 跨段稳定性 ---")
    print("%-30s %4s %5s %9s %9s %9s %8s %6s %s" % (
        "组合", "段数", "通过", "年化最低", "年化最高", "最差回撤", "最大占比", "负段", "结论"))
    for _, row in result["stability"].iterrows():
        share = "—" if not np.isfinite(row["max_segment_log_share"]) else "%.0f%%" % (
            row["max_segment_log_share"] * 100)
        print("%-30s %4d %5d %8.2f%% %8.2f%% %8.2f%% %8s %6d %s" % (
            row["label"], row["segments"], row["passed_segments"],
            row["annual_min"] * 100, row["annual_max"] * 100,
            row["worst_drawdown"] * 100, share, row["negative_segments"],
            row["verdict"]))


def _synthetic_market(codes=60, days=400):
    """构造“动量股高价格/高估值、价值股低估值”的合成市场，复现旧规则挤出。"""
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
        else:               # 价值组：低价格、低波动、低估值，且与价格排序相反
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


def synthetic_final_config_test():
    """合成验证：五因子 10% 生效、四因子旧规则不变、默认调用逐值不变。"""
    data_map, valuation_map, trade_date = _synthetic_market()
    four = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, False)
    five_old = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, True)
    five_final = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, True,
        momentum_floor=FINAL_FLOOR)

    assert four["momentum_floor"] is None
    assert five_old["momentum_floor"] is None
    assert four["selection_rule"] == "union_top", four["selection_rule"]
    assert five_old["momentum_selected"] == 0, "旧规则应把动量股全部挤出"
    assert abs(five_old["momentum_weight"]) < 1e-12
    assert five_final["momentum_slots"] == 3, five_final["momentum_slots"]
    assert five_final["momentum_selected"] >= 3
    assert abs(five_final["momentum_weight"] - FINAL_FLOOR) < 1e-9
    assert abs(float(five_final["weights"].sum()) - 1) < 1e-12
    assert len(five_final["selected"]) == 30
    rest = five_final["weights"][
        ~five_final["weights"].index.isin(five_final["sleeves"]["momentum"])]
    assert abs(float(rest.sum()) - (1 - FINAL_FLOOR)) < 1e-9
    # 四因子固化配置 == 核心模块默认调用
    legacy = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, False)
    assert legacy["weights"].equals(four["weights"])
    print("合成数据：五因子动量 3 只/10%%，四因子动量 0 只/0%%，默认调用不变，均通过")


def _validate_baseline(result):
    """真实数据校验：回归一致、下限只作用于并集期、复利自洽、持仓完整。"""
    periods = result["periods"]
    full = result["full"]
    assert len(periods) == 30, len(periods)
    assert result["segments"]["segment"].nunique() == 5
    assert result["segments"]["portfolio"].nunique() == len(PORTFOLIOS)

    for name, reference in REFERENCE.items():
        row = _row_of(full, name)
        assert abs(float(row["total_return"]) - reference["total_return"]) < 1e-9, name
        assert abs(float(row["annual_return"]) - reference["annual_return"]) < 1e-9, name
        assert abs(float(row["max_drawdown"]) - reference["max_drawdown"]) < 1e-9, name

    # 四因子最终配置仍走旧规则，且动量权重未被抬升
    assert set(periods["four_factor_rule"]) <= {"union_top", "intersection"}
    union_five = periods[periods["five_factor_rule"] == "union_top"]
    assert len(union_five) > 0
    assert (union_five["five_factor_momentum_weight"] >= FINAL_FLOOR - 1e-9).all()
    # 交集期下限不生效，新旧五因子收益必须完全相同
    inter = periods[periods["five_factor_rule"] == "intersection"]
    if len(inter):
        assert np.allclose(inter["five_factor_return"], inter["five_old_return"],
                           rtol=0, atol=1e-12)

    # S4 亏损应相对旧规则收敛
    s4 = result["segments"]
    s4 = s4[s4["segment_label"].str.startswith("S4")]
    old_s4 = float(s4[s4["portfolio"] == "five_old"]["total_return"].iloc[0])
    new_s4 = float(s4[s4["portfolio"] == "five_factor"]["total_return"].iloc[0])
    print("\nS4 五因子总收益：旧规则 %.2f%% → 最终配置 %.2f%%" % (
        old_s4 * 100, new_s4 * 100))
    assert new_s4 > old_s4

    # 分段复利与全期净值一致
    for name in PORTFOLIOS:
        compounded = float(np.prod(
            1 + result["segments"].loc[result["segments"]["portfolio"] == name,
                                       "total_return"]) - 1)
        total = float(_row_of(full, name)["total_return"])
        assert abs(compounded - total) < 1e-9, name

    # 持仓明细：每期权重和为 1，五因子并集期动量权重不低于下限
    holdings = result["holdings"]
    for name in ("four_factor", "five_factor"):
        subset = holdings[holdings["portfolio"] == name]
        sums = subset.groupby("date")["weight"].sum()
        assert len(sums) == len(periods), name
        assert (abs(sums - 1) < 1e-9).all(), name
    five_hold = holdings[holdings["portfolio"] == "five_factor"]
    momentum_weight = five_hold[five_hold["in_momentum"]].groupby("date")["weight"].sum()
    for date in union_five["date"]:
        assert float(momentum_weight.get(date, 0.0)) >= FINAL_FLOOR - 1e-9, date


def _run_baseline_main():
    print(DISCLAIMER)
    synthetic_final_config_test()
    result = run_baseline(months_per_segment=6)
    _validate_baseline(result)
    print_baseline_report(result)
    paths = save_baseline_reports(result)

    assert len(pd.read_csv(paths["full"])) == len(PORTFOLIOS)
    assert len(pd.read_csv(paths["segments"])) == len(PORTFOLIOS) * 5
    assert len(pd.read_csv(paths["compare"])) == 4
    assert len(pd.read_parquet(paths["holdings"])) == len(result["holdings"])
    payload = json.loads(paths["nav"].read_text(encoding="utf-8"))
    assert len(payload["segment_nav"]) == 5
    assert len(json.loads(paths["summary"].read_text(encoding="utf-8"))["full"]) == len(PORTFOLIOS)

    print("\n报告已保存：")
    for path in paths.values():
        print(path)
    print("\n联调清单：1 配置固化 → 2 全期重跑 → 3 五段重跑 → 4 固化前后对比 → 5 落盘，均已完成")
    print("smart_beta_baseline.py 自测通过；%s" % DISCLAIMER)


if __name__ == "__main__":
    db.init_db()
    _run_baseline_main()
