# -*- coding: utf-8 -*-
"""三因子 Smart Beta 月频回测（Step I2-alt-3）。

对比双因子 50:50、三因子等权/保守/进取和池内等权基准。
动量取 252 日累计收益并排除最近 21 日；三个单因子各取前 20%。
三因子交集不足 10 只时，在并集中按三因子综合名次取前 30 只。

只使用本地前复权日线，不接券商、不下单、不上传云端。
仅研究用途，不构成投资建议。
"""

import sys
import copy
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
    momentum_return, build_three_factor_exposures,
)
from core.smart_beta_backtest import (
    SMART_BETA_DIR, period_returns, equal_weight_return,
    performance_metrics, run_smart_beta_backtest,
)

THREE_PORTFOLIOS = {
    "three_equal": "equal",
    "three_conservative": "conservative",
    "three_aggressive": "aggressive",
}
PORTFOLIO_NAMES = ("dual",) + tuple(THREE_PORTFOLIOS)
PORTFOLIO_LABELS = {
    "dual": "双因子50:50",
    "three_equal": "三因子等权33/33/34",
    "three_conservative": "三因子保守40/40/20",
    "three_aggressive": "三因子进取20/20/60",
    "three_custom": "三因子自定义",
    "benchmark": "池内等权基准",
}


def weighted_return(returns, weights):
    """按归一化权重计算持仓收益；缺行情股票在有效权重中重新归一化。"""
    values = returns.reindex(weights.index)
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return 0.0
    valid_weights = weights[valid]
    return float(np.sum(values[valid] * valid_weights) / valid_weights.sum())


def weighted_turnover(previous, current):
    """按权重计算单边换手：0.5 * L1 权重差；首次建仓为 100%。"""
    previous = pd.Series(previous, dtype=float)
    current = pd.Series(current, dtype=float)
    if previous.empty:
        return 1.0 if not current.empty else 0.0
    union = previous.index.union(current.index)
    return float(0.5 * (previous.reindex(union, fill_value=0)
                        - current.reindex(union, fill_value=0)).abs().sum())


def run_three_factor_backtest(
        codes=None, start_date="2024-05-31", end_date="2026-11-13",
        pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, fq="qfq", factor_weights=None):
    """运行三种预设或一组自定义资金比例的月频回测。

    factor_weights 为 None 时保持 I2-alt-3 原行为；提供低价/低波/动量
    自定义比例时仅返回 three_custom 与基准，供 UI 滑块使用。
    """
    portfolio_modes = (THREE_PORTFOLIOS if factor_weights is None
                       else {"three_custom": "equal"})
    if codes is None:
        codes = [row["code"] for row in db.query_all(
            "SELECT DISTINCT code FROM daily_bars WHERE fq=? ORDER BY code", (fq,))]
    data_map = load_all_daily(codes, fq=fq)
    if not data_map:
        raise ValueError("无可用本地日线数据")
    dates = get_rebalance_dates(list(data_map), freq="M", fq=fq)
    dates = [date for date in dates if start_date <= date <= end_date]
    if len(dates) < 2:
        raise ValueError("调仓日不足，至少需要两个自然月末交易日")

    names = tuple(portfolio_modes)
    nav_map = {name: [1.0] for name in names + ("benchmark",)}
    turnover_map = {name: [] for name in names}
    previous = {name: pd.Series(dtype=float) for name in names}
    holding_rows, period_rows = [], []

    for i in range(len(dates) - 1):
        trade_date, next_date = dates[i], dates[i + 1]
        returns = period_returns(data_map, trade_date, next_date)
        exposures = {}
        portfolio_returns = {}
        for name, mode in portfolio_modes.items():
            exposure = build_three_factor_exposures(
                data_map, trade_date, pct=pct, vol_window=vol_window,
                momentum_window=momentum_window, momentum_skip=momentum_skip,
                mode=mode, factor_weights=factor_weights)
            exposures[name] = exposure
            weights = exposure["weights"]
            portfolio_returns[name] = weighted_return(returns, weights)
            turnover_map[name].append(weighted_turnover(previous[name], weights))
            previous[name] = weights.copy()
            for code, weight in weights.items():
                scores = exposure["scores"]
                holding_rows.append({
                    "date": trade_date, "next_date": next_date,
                    "portfolio": name, "sleeve": "three_factor", "code": code,
                    "weight": float(weight),
                    "price": scores["price"].get(code),
                    "volatility": scores["volatility"].get(code),
                    "momentum": scores["momentum"].get(code),
                    "rank_sum": scores["rank_sum"].get(code),
                    "selection_rule": exposure["selection_rule"],
                    "intersection_count": exposure["counts"]["intersection"],
                    "period_return": returns.get(code),
                })

        benchmark_return = equal_weight_return(returns, list(data_map))
        for name in names:
            nav_map[name].append(nav_map[name][-1] * (1 + portfolio_returns[name]))
        nav_map["benchmark"].append(nav_map["benchmark"][-1] * (1 + benchmark_return))
        first = exposures[names[0]]
        period_row = {
            "date": trade_date, "next_date": next_date,
            "benchmark_return": benchmark_return,
            "selection_rule": first["selection_rule"],
            "intersection_count": first["counts"]["intersection"],
            "union_count": first["counts"]["union"],
            "selected_count": first["counts"]["selected"],
        }
        for name in names:
            period_row[name + "_return"] = portfolio_returns[name]
        period_rows.append(period_row)

    nav = pd.DataFrame(nav_map, index=dates)
    nav.index.name = "date"
    metrics = {name: performance_metrics(nav[name], turnover_map[name])
               for name in names}
    metrics["benchmark"] = performance_metrics(nav["benchmark"])
    benchmark_drawdown = metrics["benchmark"]["max_drawdown"]
    verdicts = {}
    for name in names:
        beat_cash = metrics[name]["annual_return"] > 0
        controlled = metrics[name]["max_drawdown"] <= benchmark_drawdown
        verdicts[name] = {
            "beat_cash": bool(beat_cash),
            "drawdown_not_above_benchmark": bool(controlled),
            "passed": bool(beat_cash and controlled),
        }
    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": len(data_map), "start": dates[0], "end": dates[-1],
            "pct": pct, "vol_window": vol_window,
            "momentum_window": momentum_window, "momentum_skip": momentum_skip,
            "custom_factor_weights": (dict(factor_weights) if factor_weights else None),
            "freq": "M", "fq": fq,
            "price_note": "本地仅有前复权行情，低价暂以前复权收盘价排序",
            "selection_note": "三因子交集不足10只时，在并集中按三因子百分位名次之和取前30只",
            "disclaimer": DISCLAIMER,
        },
        "nav": nav, "holdings": pd.DataFrame(holding_rows),
        "periods": pd.DataFrame(period_rows),
        "turnover": pd.DataFrame(turnover_map, index=dates[1:]),
        "metrics": metrics, "verdicts": verdicts,
        "passed": {name: verdicts[name]["passed"] for name in names},
    }


def run_three_factor_comparison(
        codes=None, start_date="2024-05-31", end_date="2026-11-13",
        pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, fq="qfq"):
    """合并既有双因子基线与三种三因子方案，并计算收益/回撤变化。"""
    baseline = run_smart_beta_backtest(
        codes=codes, start_date=start_date, end_date=end_date,
        pct=pct, vol_window=vol_window, fq=fq)
    three = run_three_factor_backtest(
        codes=codes, start_date=start_date, end_date=end_date, pct=pct,
        vol_window=vol_window, momentum_window=momentum_window,
        momentum_skip=momentum_skip, fq=fq)
    if not np.allclose(baseline["nav"]["benchmark"], three["nav"]["benchmark"]):
        raise AssertionError("双因子与三因子回测的池内基准不一致")

    nav = pd.DataFrame({"dual": baseline["nav"]["dual"]})
    for name in THREE_PORTFOLIOS:
        nav[name] = three["nav"][name]
    nav["benchmark"] = three["nav"]["benchmark"]
    metrics = {"dual": baseline["metrics"]["dual"]}
    verdicts = {"dual": baseline["verdicts"]["dual"]}
    for name in THREE_PORTFOLIOS:
        metrics[name] = three["metrics"][name]
        verdicts[name] = three["verdicts"][name]
    metrics["benchmark"] = three["metrics"]["benchmark"]

    comparisons = {}
    for name in THREE_PORTFOLIOS:
        comparisons[name] = {
            "total_return_delta_vs_dual": metrics[name]["total_return"] - metrics["dual"]["total_return"],
            "annual_return_delta_vs_dual": metrics[name]["annual_return"] - metrics["dual"]["annual_return"],
            "max_drawdown_delta_vs_dual": metrics[name]["max_drawdown"] - metrics["dual"]["max_drawdown"],
        }

    # 双因子持仓拆成两个半仓子组合；三因子每行已是归一化组合权重
    dual_holdings = baseline["holdings"].copy()
    dual_holdings["sleeve"] = dual_holdings["portfolio"]
    dual_holdings["portfolio"] = "dual"
    dual_holdings["weight"] *= 0.5
    dual_holdings["momentum"] = np.nan
    dual_holdings["rank_sum"] = np.nan
    dual_holdings["selection_rule"] = "dual_sleeves"
    dual_holdings["intersection_count"] = np.nan
    holdings = pd.concat([dual_holdings, three["holdings"]], ignore_index=True)
    periods = three["periods"].merge(
        baseline["periods"][["date", "next_date", "dual_return"]],
        on=["date", "next_date"], how="left", validate="one_to_one")
    turnover = pd.DataFrame({"dual": baseline["turnover"]["dual"]})
    for name in THREE_PORTFOLIOS:
        turnover[name] = three["turnover"][name]
    meta = dict(three["meta"])
    meta["dual_note"] = "双因子基线直接复用 I2-alt-1 回测结果"
    meta["comparison_note"] = "收益与回撤变化均为三因子指标减双因子指标，回撤差为负表示改善"
    return {
        "meta": meta, "nav": nav, "holdings": holdings, "periods": periods,
        "turnover": turnover, "metrics": metrics, "verdicts": verdicts,
        "comparisons": comparisons,
        "passed": {name: verdicts[name]["passed"] for name in PORTFOLIO_NAMES},
    }


def save_comparison_reports(result, directory=None):
    """保存三预设或自定义组合的绩效、净值、JSON 与持仓；旧结果保持 5 行。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    portfolio_names = tuple(name for name in result["metrics"] if name != "benchmark")
    variant = "custom" if portfolio_names == ("three_custom",) else "3factor"
    stem = "smart_beta_%s_%s_%s_%s" % (
        variant, meta["start"].replace("-", ""), meta["end"].replace("-", ""), meta["codes"])
    comparisons = result.get("comparisons", {})
    rows = []
    for name in portfolio_names + ("benchmark",):
        row = {"portfolio": name, "label": PORTFOLIO_LABELS[name]}
        row.update(result["metrics"][name])
        if name in result["verdicts"]:
            row.update(result["verdicts"][name])
        delta = comparisons.get(name, {})
        row["annual_return_delta_vs_dual"] = delta.get("annual_return_delta_vs_dual", 0.0 if name == "dual" else np.nan)
        row["max_drawdown_delta_vs_dual"] = delta.get("max_drawdown_delta_vs_dual", 0.0 if name == "dual" else np.nan)
        rows.append(row)
    csv_path = directory / (stem + ".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    nav_path = directory / (stem + "_nav.csv")
    result["nav"].to_csv(nav_path, encoding="utf-8-sig")
    holdings_path = directory / (stem + "_holdings.parquet")
    result["holdings"].to_parquet(holdings_path, index=False)
    payload = {
        "meta": meta, "metrics": result["metrics"], "verdicts": result["verdicts"],
        "comparisons": comparisons, "passed": result["passed"],
        "periods": result["periods"].to_dict("records"),
    }
    json_path = directory / (stem + ".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    default=str), encoding="utf-8")
    return {"csv": csv_path, "json": json_path,
            "nav": nav_path, "holdings": holdings_path}


def print_comparison_report(result):
    """打印四方案与基准对比；收益/回撤差均以双因子为参照。"""
    meta = result["meta"]
    print("\n=== 三因子 Smart Beta 月频对比 ===")
    print("股票池 %d 只 | %s ~ %s | 各前 %.0f%% | 动量 %d 日剔除 %d 日" % (
        meta["codes"], meta["start"], meta["end"], meta["pct"] * 100,
        meta["momentum_window"], meta["momentum_skip"]))
    print("%-24s %9s %9s %8s %10s %11s %11s %8s" % (
        "组合", "总收益", "年化", "夏普", "最大回撤", "年化差(pp)", "回撤差(pp)", "判定"))
    for name in PORTFOLIO_NAMES + ("benchmark",):
        metrics = result["metrics"][name]
        delta = result["comparisons"].get(name, {})
        annual_delta = delta.get("annual_return_delta_vs_dual", 0.0 if name == "dual" else np.nan)
        drawdown_delta = delta.get("max_drawdown_delta_vs_dual", 0.0 if name == "dual" else np.nan)
        verdict = ("—" if name == "benchmark" else
                   ("通过" if result["verdicts"][name]["passed"] else "未通过"))
        print("%-24s %8.2f%% %8.2f%% %8.3f %9.2f%% %11.2f %11.2f %8s" % (
            PORTFOLIO_LABELS[name], metrics["total_return"] * 100,
            metrics["annual_return"] * 100, metrics["sharpe"],
            metrics["max_drawdown"] * 100, annual_delta * 100,
            drawdown_delta * 100, verdict))


def synthetic_three_factor_test():
    """300 只合成股票：验证动量剔除近 21 日、交集/并集与三种资金比例。"""
    rng = np.random.default_rng(20261115)
    dates = pd.bdate_range("2025-01-02", periods=280).strftime("%Y-%m-%d").tolist()
    data_map = {}
    for j in range(300):
        code = "%06d" % (j + 1)
        # 低价偏小编号、动量偏大编号，使默认前 20% 子组合交集很小
        returns = 0.0001 + j / 299 * 0.002 + rng.normal(0, 0.001 + (j % 10) * 0.0002, len(dates))
        close = (2 + j * 0.1) * np.cumprod(1 + returns)
        data_map[code] = pd.DataFrame({"date": dates, "close": close})
    trade_date = dates[-2]
    sample = data_map["000001"]
    visible = sample[sample["date"] <= trade_date]
    end_pos = len(visible) - DEFAULT_MOMENTUM_SKIP - 1
    start_pos = end_pos - DEFAULT_MOMENTUM_WINDOW
    expected = float(visible.iloc[end_pos]["close"] / visible.iloc[start_pos]["close"] - 1)
    actual = momentum_return(sample, trade_date)
    assert abs(actual - expected) < 1e-12

    # 最近 21 日及调仓日之后的行情变化均不得影响动量
    changed = copy.deepcopy(sample)
    cutoff = visible.iloc[-DEFAULT_MOMENTUM_SKIP - 1]["date"]
    changed.loc[changed["date"] > cutoff, "close"] *= 0.2
    assert abs(momentum_return(changed, trade_date) - actual) < 1e-12
    changed = copy.deepcopy(sample)
    changed.loc[changed["date"] > trade_date, "close"] *= 100
    assert abs(momentum_return(changed, trade_date) - actual) < 1e-12

    for name, mode in THREE_PORTFOLIOS.items():
        exposure = build_three_factor_exposures(data_map, trade_date, mode=mode)
        assert len(exposure["low_price"]) == 60
        assert len(exposure["low_vol"]) == 60
        assert len(exposure["momentum"]) == 60
        assert exposure["counts"]["intersection"] < 10
        assert exposure["selection_rule"] == "union_top"
        assert len(exposure["selected"]) == 30
        assert set(exposure["selected"]) <= (
            set(exposure["low_price"]) | set(exposure["low_vol"]) | set(exposure["momentum"]))
        assert abs(exposure["weights"].sum() - 1) < 1e-12
        assert (exposure["weights"] > 0).all(), name
    full = build_three_factor_exposures(data_map, trade_date, pct=1.0)
    assert full["selection_rule"] == "intersection" and len(full["selected"]) == 300
    custom = build_three_factor_exposures(
        data_map, trade_date, factor_weights={"low_price": 5, "low_vol": 3, "momentum": 2})
    assert abs(custom["allocations"]["low_price"] - 0.5) < 1e-12
    print("合成数据：300 只动量口径、近 21 日剔除、防未来函数、交集/并集及三种资金比例均通过")


if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()
    synthetic_three_factor_test()
    result = run_three_factor_comparison(
        start_date="2024-05-31", end_date="2026-11-13")
    print_comparison_report(result)
    paths = save_comparison_reports(result)

    assert list(result["nav"].columns) == list(PORTFOLIO_NAMES + ("benchmark",))
    assert len(result["periods"]) == 30 and len(result["nav"]) == 31
    assert (result["nav"] > 0).all().all()
    assert ((result["turnover"] >= 0) & (result["turnover"] <= 1.000001)).all().all()
    for name, frame in result["holdings"].groupby(["date", "portfolio"]):
        assert abs(frame["weight"].sum() - 1) < 1e-9, (name, frame["weight"].sum())
    # 回归：合并结果中的双因子基线必须与 I2-alt-1 完全一致
    assert abs(result["metrics"]["dual"]["annual_return"] - 0.09888970389537644) < 1e-10
    assert round(result["metrics"]["dual"]["max_drawdown"], 4) == 0.1186
    assert len(pd.read_csv(paths["csv"])) == 5

    # I2-alt-5：自定义资金比例只生成 three_custom，且报告导出保持动态组合数。
    custom_result = run_three_factor_backtest(
        start_date="2024-05-31", end_date="2026-11-13",
        factor_weights={"low_price": 0.5, "low_vol": 0.3, "momentum": 0.2})
    assert list(custom_result["nav"].columns) == ["three_custom", "benchmark"]
    assert abs(sum(custom_result["meta"]["custom_factor_weights"].values()) - 1) < 1e-12
    custom_paths = save_comparison_reports(custom_result)
    assert len(pd.read_csv(custom_paths["csv"])) == 2
    assert custom_result["passed"]["three_custom"] == custom_result["verdicts"]["three_custom"]["passed"]
    print("\n三预设报告已保存：")
    for path in paths.values():
        print(path)
    print("自定义比例报告已保存：")
    for path in custom_paths.values():
        print(path)
    print("\n联调清单：1 动量口径 → 2 四方案对比 → 3 双因子回归 → 4 自定义比例及导出，均已完成")
    print("smart_beta_3factor_backtest.py 自测通过；%s" % DISCLAIMER)
