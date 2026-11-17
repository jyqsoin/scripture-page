# -*- coding: utf-8 -*-
"""Smart Beta 样本外分段验证（Step I2-alt-6）。

把 2024-05 ~ 2026-11 的月频调仓期按每 6 个自然月切段，逐段独立统计
双因子、三因子三种资金比例、四因子与五因子的年化、夏普和最大回撤，
并与同段池内等权基准比较，检查收益是否集中在某一段。

防未来函数：
- 每个调仓日仍只用该日及之前的本地行情与估值选股，段划分只影响绩效聚合；
- 段首期允许使用段前历史窗口预热（历史数据，不是未来数据）；
- 每期收益严格按调仓日收盘到下月调仓日收盘计算。

不修改既有回测核心；不接券商、不下单、不上传云端。
仅研究用途，不构成投资建议。
"""

import sys
import json
import math
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
from core.valuation_data import load_valuation

PORTFOLIOS = (
    "dual", "three_equal", "three_conservative", "three_aggressive",
    "four_factor", "five_factor",
)
PORTFOLIO_LABELS = {
    "dual": "双因子50:50",
    "three_equal": "三因子等权33/33/34",
    "three_conservative": "三因子保守40/40/20",
    "three_aggressive": "三因子进取20/20/60",
    "four_factor": "四因子等权25%",
    "five_factor": "五因子等权20%",
    "benchmark": "池内等权基准",
}
THREE_MODES = {
    "three_equal": "equal",
    "three_conservative": "conservative",
    "three_aggressive": "aggressive",
}
# 单段对数收益占全期比例超过该阈值，视为收益过度依赖单一段落
DOMINANT_SHARE = 0.60


def assign_segments(dates, start_date, months_per_segment=6):
    """按起始月起每 N 个自然月一段，返回 {调仓日: 段号}；段号从 0 连续递增。"""
    if months_per_segment < 1:
        raise ValueError("months_per_segment 必须为正整数")
    start = pd.Timestamp(start_date)
    base = start.year * 12 + start.month
    mapping = {}
    for date in dates:
        stamp = pd.Timestamp(date)
        index = (stamp.year * 12 + stamp.month - base) // months_per_segment
        if index < 0:
            raise ValueError("调仓日 %s 早于起始日 %s" % (date, start_date))
        mapping[date] = int(index)
    if mapping and sorted(set(mapping.values())) != list(range(max(mapping.values()) + 1)):
        raise ValueError("段号必须从 0 连续且无空缺")
    return mapping


def log_share(values):
    """按对数收益分解各段占比；全期对数收益非正时返回 NaN，避免无意义比值。"""
    logs = np.array([math.log1p(value) for value in values], dtype=float)
    total = float(logs.sum())
    if total <= 1e-12:
        return np.full(len(logs), np.nan)
    return np.abs(logs / total)


def stability_summary(records):
    """汇总各组合的跨段稳定性。

    records 为逐段绩效字典列表，至少包含 portfolio / segment_label /
    total_return / annual_return / max_drawdown / passed 字段。
    判定口径：全部段通过且没有单段对数收益占比超过 60% 记为「稳定」。
    """
    frame = pd.DataFrame(records)
    summaries = []
    for name in PORTFOLIOS:
        subset = frame[frame["portfolio"] == name]
        if subset.empty:
            continue
        returns = subset["total_return"].tolist()
        compound = float(np.prod([1 + value for value in returns]) - 1)
        shares = log_share(returns)
        max_share = float(np.nanmax(shares)) if np.isfinite(shares).any() else np.nan
        if max_share == max_share:
            dominant = subset.iloc[int(np.nanargmax(shares))]["segment_label"]
        else:
            dominant = ""
        passed_count = int(subset["passed"].sum())
        total_log = float(np.log1p(returns).sum())
        if total_log <= 0:
            verdict = "全期亏损"
        elif passed_count < len(subset):
            verdict = "不稳定"
        elif max_share > DOMINANT_SHARE:
            verdict = "单段主导"
        else:
            verdict = "稳定"
        summaries.append({
            "portfolio": name, "label": PORTFOLIO_LABELS[name],
            "segments": len(subset), "passed_segments": passed_count,
            "compound_return": compound,
            "annual_min": float(subset["annual_return"].min()),
            "annual_max": float(subset["annual_return"].max()),
            "annual_std": float(subset["annual_return"].std(ddof=1))
            if len(subset) > 1 else 0.0,
            "worst_drawdown": float(subset["max_drawdown"].max()),
            "worst_drawdown_segment": subset.loc[
                subset["max_drawdown"].idxmax(), "segment_label"],
            "best_return_segment": subset.loc[
                subset["total_return"].idxmax(), "segment_label"],
            "worst_return_segment": subset.loc[
                subset["total_return"].idxmin(), "segment_label"],
            "max_segment_log_share": max_share,
            "dominant_segment": dominant,
            "negative_segments": int((subset["total_return"] < 0).sum()),
            "verdict": verdict,
        })
    return pd.DataFrame(summaries)


def _segment_label(index, periods):
    """段标签用首期调仓日到末期下期日期，便于核对聚合边界。"""
    return "S%d %s~%s" % (
        index + 1, periods.iloc[0]["date"], periods.iloc[-1]["next_date"])


def run_walkforward(
        codes=None, start_date="2024-05-31", end_date="2026-11-13",
        pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, fq="qfq",
        months_per_segment=6):
    """逐段统计六个组合；每段净值从 1.0 重新开始，同时保留全期净值用于校验。"""
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
    # 只有存在下一期的调仓日才产生收益，末个调仓日仅作为上一期的净值终点
    seg_of = assign_segments(dates[:-1], start_date, months_per_segment)

    full_nav = {name: [1.0] for name in PORTFOLIOS + ("benchmark",)}
    seg_nav = {}
    seg_turnover = {}
    previous = {name: pd.Series(dtype=float) for name in PORTFOLIOS
                if name != "dual"}
    previous_dual = {"low_price": [], "low_vol": []}
    period_rows = []

    for i in range(len(dates) - 1):
        trade_date, next_date = dates[i], dates[i + 1]
        segment = seg_of[trade_date]
        seg_nav.setdefault(segment, {name: [1.0] for name in PORTFOLIOS + ("benchmark",)})
        if segment not in seg_turnover:
            seg_turnover[segment] = {name: [] for name in PORTFOLIOS}

        returns = period_returns(data_map, trade_date, next_date)
        dual = build_exposures(
            data_map, trade_date, pct=pct, vol_window=vol_window)
        three = {name: build_three_factor_exposures(
            data_map, trade_date, pct=pct, vol_window=vol_window,
            momentum_window=momentum_window, momentum_skip=momentum_skip,
            mode=mode) for name, mode in THREE_MODES.items()}
        four = build_fundamental_composite(
            data_map, valuation_map, trade_date, True, False, pct, vol_window,
            momentum_window, momentum_skip)
        five = build_fundamental_composite(
            data_map, valuation_map, trade_date, True, True, pct, vol_window,
            momentum_window, momentum_skip)

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
        for name in THREE_MODES:
            weights = three[name]["weights"]
            portfolio_returns[name] = weighted_return(returns, weights)
            turnovers[name] = weighted_turnover(previous[name], weights)
            previous[name] = weights.copy()
        for name, exposure in (("four_factor", four), ("five_factor", five)):
            weights = exposure["weights"]
            portfolio_returns[name] = weighted_return(returns, weights)
            turnovers[name] = weighted_turnover(previous[name], weights)
            previous[name] = weights.copy()

        benchmark_return = equal_weight_return(returns, list(data_map))
        for name in PORTFOLIOS:
            full_nav[name].append(full_nav[name][-1] * (1 + portfolio_returns[name]))
            seg_nav[segment][name].append(
                seg_nav[segment][name][-1] * (1 + portfolio_returns[name]))
            seg_turnover[segment][name].append(turnovers[name])
        full_nav["benchmark"].append(full_nav["benchmark"][-1] * (1 + benchmark_return))
        seg_nav[segment]["benchmark"].append(
            seg_nav[segment]["benchmark"][-1] * (1 + benchmark_return))

        row = {"date": trade_date, "next_date": next_date,
               "segment": segment, "benchmark_return": benchmark_return}
        for name in PORTFOLIOS:
            row[name + "_return"] = portfolio_returns[name]
        period_rows.append(row)

    periods = pd.DataFrame(period_rows)
    labels = {segment: _segment_label(segment, frame)
              for segment, frame in periods.groupby("segment")}
    periods["segment_label"] = periods["segment"].map(labels)

    segment_rows, nav_payload = [], {}
    full_index = dates
    full_metrics = {}
    for name in PORTFOLIOS:
        full_metrics[name] = performance_metrics(
            pd.Series(full_nav[name], index=full_index),
            [value for segment in sorted(seg_turnover)
             for value in seg_turnover[segment][name]])
    full_metrics["benchmark"] = performance_metrics(
        pd.Series(full_nav["benchmark"], index=full_index))

    for segment in sorted(seg_nav):
        frame = periods[periods["segment"] == segment]
        index = [frame.iloc[0]["date"]] + frame["next_date"].tolist()
        label = labels[segment]
        benchmark_metrics = performance_metrics(
            pd.Series(seg_nav[segment]["benchmark"], index=index))
        nav_payload[label] = {
            "periods": len(frame),
            "nav": {
                name: {date: round(float(value), 8)
                       for date, value in zip(index, seg_nav[segment][name])}
                for name in PORTFOLIOS + ("benchmark",)
            },
        }
        for name in PORTFOLIOS:
            metrics = performance_metrics(
                pd.Series(seg_nav[segment][name], index=index),
                seg_turnover[segment][name])
            beat_cash = metrics["annual_return"] > 0
            controlled = metrics["max_drawdown"] <= benchmark_metrics["max_drawdown"]
            row = {
                "segment": segment, "segment_label": label,
                "portfolio": name, "label": PORTFOLIO_LABELS[name],
                "periods": metrics["periods"],
                "total_return": metrics["total_return"],
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "total_turnover": metrics["total_turnover"],
                "benchmark_total_return": benchmark_metrics["total_return"],
                "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
                "excess_return": metrics["total_return"]
                - benchmark_metrics["total_return"],
                "beat_cash": bool(beat_cash),
                "drawdown_not_above_benchmark": bool(controlled),
                "passed": bool(beat_cash and controlled),
            }
            segment_rows.append(row)

    segments = pd.DataFrame(segment_rows)
    stability = stability_summary(segment_rows)
    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": len(data_map), "start": dates[0], "end": dates[-1],
            "pct": pct, "vol_window": vol_window,
            "momentum_window": momentum_window, "momentum_skip": momentum_skip,
            "months_per_segment": months_per_segment,
            "freq": "M", "fq": fq,
            "segment_note": "段划分只影响绩效聚合；每期选股只用调仓日及之前数据",
            "verdict_note": "各段分别判定：年化收益大于0且最大回撤不超过同段池内等权基准",
            "stability_note": "全部段通过且无单段对数收益占比超过%.0f%%记为稳定"
                              % (DOMINANT_SHARE * 100),
            "disclaimer": DISCLAIMER,
        },
        "segments": segments, "stability": stability,
        "periods": periods, "nav": nav_payload,
        "full_metrics": full_metrics,
        "full_nav": pd.DataFrame(full_nav, index=full_index),
    }


def save_walkforward_reports(result, directory=None):
    """保存逐段绩效 CSV、稳定性 CSV、分段净值 JSON 与参数摘要 JSON。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    stem = "smart_beta_walkforward_%s_%s_%s" % (
        meta["start"].replace("-", ""), meta["end"].replace("-", ""), meta["codes"])
    segments_path = directory / (stem + "_segments.csv")
    result["segments"].to_csv(segments_path, index=False, encoding="utf-8-sig")
    stability_path = directory / (stem + "_stability.csv")
    result["stability"].to_csv(stability_path, index=False, encoding="utf-8-sig")
    nav_path = directory / (stem + "_nav.json")
    nav_path.write_text(json.dumps(
        {"meta": meta, "nav": result["nav"]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    summary_path = directory / (stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "meta": meta,
        "stability": result["stability"].to_dict("records"),
        "full_metrics": result["full_metrics"],
        "periods": result["periods"].to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"segments": segments_path, "stability": stability_path,
            "nav": nav_path, "summary": summary_path}


def _verdict_text(row):
    """不用笼统的“未通过”，直接指出是收益还是回撤未达标。"""
    if row["passed"]:
        return "通过"
    if not row["beat_cash"]:
        return "收益为负"
    return "回撤超基准"


def print_walkforward_report(result):
    """打印逐段绩效与稳定性结论。"""
    meta = result["meta"]
    print("\n=== Smart Beta 样本外分段验证 ===")
    print("股票池 %d 只 | %s ~ %s | 各前 %.0f%% | 每 %d 个自然月一段" % (
        meta["codes"], meta["start"], meta["end"], meta["pct"] * 100,
        meta["months_per_segment"]))
    segments = result["segments"]
    for label, frame in segments.groupby("segment_label", sort=True):
        benchmark = frame.iloc[0]
        print("\n--- %s（%d 期）基准总收益 %.2f%%，基准最大回撤 %.2f%% ---" % (
            label, benchmark["periods"], benchmark["benchmark_total_return"] * 100,
            benchmark["benchmark_max_drawdown"] * 100))
        print("%-22s %8s %8s %8s %9s %8s %8s %12s" % (
            "组合", "总收益", "超额", "夏普", "最大回撤", "月胜率", "年化", "判定"))
        for _, row in frame.iterrows():
            print("%-22s %7.2f%% %7.2f%% %8.3f %8.2f%% %7.2f%% %7.2f%% %12s" % (
                row["label"], row["total_return"] * 100,
                row["excess_return"] * 100, row["sharpe"],
                row["max_drawdown"] * 100, row["win_rate"] * 100,
                row["annual_return"] * 100, _verdict_text(row)))
    print("\n--- 跨段稳定性 ---")
    print("%-22s %5s %6s %9s %9s %9s %8s %10s %s" % (
        "组合", "段数", "通过", "年化最低", "年化最高", "最差回撤", "最大占比", "负收益段", "结论"))
    for _, row in result["stability"].iterrows():
        share = "—" if not np.isfinite(row["max_segment_log_share"]) else "%.0f%%" % (
            row["max_segment_log_share"] * 100)
        print("%-22s %5d %6d %8.2f%% %8.2f%% %8.2f%% %8s %10d %s" % (
            row["label"], row["segments"], row["passed_segments"],
            row["annual_min"] * 100, row["annual_max"] * 100,
            row["worst_drawdown"] * 100, share, row["negative_segments"],
            row["verdict"]))


def synthetic_walkforward_test():
    """验证分段连续、对数占比分解与稳定性结论，不依赖本地数据。"""
    dates = ["2024-05-31", "2024-06-28", "2024-07-31", "2024-08-30",
             "2024-09-30", "2024-10-31", "2024-11-29", "2024-12-31",
             "2025-01-27", "2025-02-28", "2025-03-31", "2025-04-30"]
    mapping = assign_segments(dates, "2024-05-31", 6)
    assert [mapping[date] for date in dates] == [0] * 6 + [1] * 6
    assert sorted(set(mapping.values())) == [0, 1]
    try:
        assign_segments(["2024-04-30"], "2024-05-31", 6)
    except ValueError:
        pass
    else:
        raise AssertionError("早于起始日的调仓日应报错")

    shares = log_share([0.10, 0.10, 0.10])
    assert np.allclose(shares, 1.0 / 3), shares
    assert np.isnan(log_share([-0.2, 0.05])).all()

    def records(returns, passed=True):
        return [{"portfolio": "five_factor", "segment_label": "S%d" % (i + 1),
                 "total_return": value, "annual_return": value,
                 "max_drawdown": abs(value) / 2, "passed": passed}
                for i, value in enumerate(returns)]

    stable = stability_summary(records([0.05, 0.04, 0.06])).iloc[0]
    assert stable["verdict"] == "稳定", stable["verdict"]
    assert abs(stable["compound_return"] - (1.05 * 1.04 * 1.06 - 1)) < 1e-12
    assert abs(stable["max_segment_log_share"] - 0.06 / 0.15) < 0.02

    dominant = stability_summary(records([0.30, 0.002, 0.002])).iloc[0]
    assert dominant["verdict"] == "单段主导", dominant["verdict"]
    assert dominant["dominant_segment"] == "S1"

    unstable = stability_summary(
        [dict(row, passed=(row["segment_label"] != "S2"))
         for row in records([0.05, -0.02, 0.03])]).iloc[0]
    assert unstable["verdict"] == "不稳定", unstable["verdict"]

    losing = stability_summary(records([-0.05, -0.02, -0.03])).iloc[0]
    assert losing["verdict"] == "全期亏损", losing["verdict"]
    print("合成数据：分段连续、对数占比分解与四类稳定性结论均通过")


if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()
    synthetic_walkforward_test()
    result = run_walkforward(months_per_segment=6)
    print_walkforward_report(result)

    assert result["meta"]["months_per_segment"] == 6
    assert result["segments"]["segment"].nunique() == 5
    assert set(result["segments"]["portfolio"]) == set(PORTFOLIOS)
    # 分段净值必须从 1.0 开始，且各段复利结果与全期完全一致
    for label, payload in result["nav"].items():
        assert set(payload) == {"periods", "nav"}, label
        for name in PORTFOLIOS + ("benchmark",):
            assert abs(list(payload["nav"][name].values())[0] - 1.0) < 1e-12, (label, name)
    for name in PORTFOLIOS:
        compounded = float(np.prod(
            1 + result["segments"].loc[result["segments"]["portfolio"] == name,
                                       "total_return"]) - 1)
        assert abs(compounded - result["full_metrics"][name]["total_return"]) < 1e-9, name
    paths = save_walkforward_reports(result)
    assert len(pd.read_csv(paths["segments"])) == len(result["segments"])
    assert len(pd.read_csv(paths["stability"])) == len(PORTFOLIOS)
    payload = json.loads(paths["nav"].read_text(encoding="utf-8"))
    assert len(payload["nav"]) == 5
    print("\n报告已保存：")
    for path in paths.values():
        print(path)
    print("\n联调清单：1 分段划分 → 2 六组合逐段回测 → 3 基准与现金判定 → 4 稳定性汇总 → 5 报告落盘，均已完成")
    print("smart_beta_walkforward.py 自测通过；%s" % DISCLAIMER)
