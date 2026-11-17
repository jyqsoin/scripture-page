# -*- coding: utf-8 -*-
"""Smart Beta 交易成本与换手敏感性（Step I2-alt-7）。

在既有月频调仓结果上，按每期实际换手扣减交易成本：
- 单边成本 = 佣金万三 + 滑点 0.1% = 0.13%；
- 期净收益 = (1 + 毛收益) × (1 - 单边换手 × 单边成本) - 1；
- 同时给出成本 0 倍（毛收益）、1 倍、2 倍三档，便于观察拖累幅度；
- 池内等权基准同样按自身再平衡换手扣成本，判定口径保持对齐。

防未来函数：选股与收益计算完全复用既有月频回测口径，本模块只增加成本。
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
COMMISSION = 0.0003      # 佣金：万三，双边各收
SLIPPAGE = 0.001         # 滑点：0.1%，双边各计
COST_MULTIPLIERS = (0, 1, 2)


def one_way_cost(commission=COMMISSION, slippage=SLIPPAGE):
    """单边综合成本率；买卖各按该费率计一次。"""
    if commission < 0 or slippage < 0:
        raise ValueError("佣金与滑点不能为负")
    return float(commission + slippage)


def apply_cost(gross_return, turnover, cost_rate):
    """按单边换手扣成本；成本以期末资产比例近似扣减。"""
    if turnover < 0 or not np.isfinite(turnover):
        raise ValueError("换手必须为非负有限值")
    return float((1 + gross_return) * (1 - turnover * cost_rate) - 1)


def benchmark_turnover(returns):
    """等权基准再平衡换手：0.5 × 漂移权重与目标等权权重之差的绝对值之和。"""
    valid = returns.dropna()
    if valid.empty:
        return 0.0
    gross = 1 + valid
    if gross.sum() <= 0:
        return 0.0
    drift = gross / gross.sum()
    target = 1.0 / len(valid)
    return float(0.5 * (drift - target).abs().sum())


def run_cost_sensitivity(
        codes=None, start_date="2024-05-31", end_date="2026-11-13",
        pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, fq="qfq",
        commission=COMMISSION, slippage=SLIPPAGE,
        multipliers=COST_MULTIPLIERS):
    """逐期计算毛收益与换手，再按不同成本倍数累计净值。"""
    if not multipliers or any(value < 0 for value in multipliers):
        raise ValueError("成本倍数必须为非负序列")
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

    rate = one_way_cost(commission, slippage)
    names = PORTFOLIOS + ("benchmark",)
    nav_map = {multiplier: {name: [1.0] for name in names} for multiplier in multipliers}
    turnover_map = {name: [] for name in names}
    previous = {name: pd.Series(dtype=float) for name in PORTFOLIOS
                if name != "dual"}
    previous_dual = {"low_price": [], "low_vol": []}
    period_rows = []

    for i in range(len(dates) - 1):
        trade_date, next_date = dates[i], dates[i + 1]
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
        gross_returns = {"dual": 0.5 * low_ret + 0.5 * vol_ret}
        turnovers = {"dual": 0.5 * holding_turnover(
            previous_dual["low_price"], dual["low_price"]) + 0.5 * holding_turnover(
            previous_dual["low_vol"], dual["low_vol"])}
        previous_dual = {"low_price": dual["low_price"], "low_vol": dual["low_vol"]}
        for name in THREE_MODES:
            weights = three[name]["weights"]
            gross_returns[name] = weighted_return(returns, weights)
            turnovers[name] = weighted_turnover(previous[name], weights)
            previous[name] = weights.copy()
        for name, exposure in (("four_factor", four), ("five_factor", five)):
            weights = exposure["weights"]
            gross_returns[name] = weighted_return(returns, weights)
            turnovers[name] = weighted_turnover(previous[name], weights)
            previous[name] = weights.copy()

        gross_returns["benchmark"] = equal_weight_return(returns, list(data_map))
        turnovers["benchmark"] = benchmark_turnover(returns)
        for name in names:
            turnover_map[name].append(turnovers[name])
            for multiplier in multipliers:
                net = apply_cost(gross_returns[name], turnovers[name],
                                 rate * multiplier)
                nav_map[multiplier][name].append(
                    nav_map[multiplier][name][-1] * (1 + net))

        row = {"date": trade_date, "next_date": next_date,
               "benchmark_gross_return": gross_returns["benchmark"],
               "benchmark_turnover": turnovers["benchmark"]}
        for name in PORTFOLIOS:
            row[name + "_gross_return"] = gross_returns[name]
            row[name + "_turnover"] = turnovers[name]
        period_rows.append(row)

    nav_frames = {multiplier: pd.DataFrame(nav_map[multiplier], index=dates)
                  for multiplier in multipliers}
    metrics = {}
    for multiplier in multipliers:
        frame = nav_frames[multiplier]
        metrics[multiplier] = {
            name: performance_metrics(frame[name], turnover_map[name])
            for name in PORTFOLIOS
        }
        metrics[multiplier]["benchmark"] = performance_metrics(frame["benchmark"])

    gross_annual = {name: metrics[0][name]["annual_return"] for name in PORTFOLIOS} \
        if 0 in metrics else {name: metrics[min(metrics)][name]["annual_return"]
                              for name in PORTFOLIOS}

    rows = []
    for multiplier in multipliers:
        benchmark = metrics[multiplier]["benchmark"]
        for name in PORTFOLIOS:
            value = metrics[multiplier][name]
            beat_cash = value["annual_return"] > 0
            controlled = value["max_drawdown"] <= benchmark["max_drawdown"]
            rows.append({
                "cost_multiplier": multiplier,
                "one_way_cost": rate * multiplier,
                "portfolio": name, "label": PORTFOLIO_LABELS[name],
                "total_return": value["total_return"],
                "annual_return": value["annual_return"],
                "sharpe": value["sharpe"],
                "max_drawdown": value["max_drawdown"],
                "win_rate": value["win_rate"],
                "total_turnover": value["total_turnover"],
                "annual_cost_drag": gross_annual[name] - value["annual_return"],
                "benchmark_annual_return": benchmark["annual_return"],
                "benchmark_max_drawdown": benchmark["max_drawdown"],
                "beat_cash": bool(beat_cash),
                "drawdown_not_above_benchmark": bool(controlled),
                "passed": bool(beat_cash and controlled),
            })
    summary = pd.DataFrame(rows)
    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": len(data_map), "start": dates[0], "end": dates[-1],
            "pct": pct, "vol_window": vol_window,
            "momentum_window": momentum_window, "momentum_skip": momentum_skip,
            "freq": "M", "fq": fq,
            "commission": commission, "slippage": slippage,
            "one_way_cost": rate, "cost_multipliers": list(multipliers),
            "cost_note": "期净收益=(1+毛收益)×(1-单边换手×单边成本)-1；未计印花税",
            "benchmark_note": "基准按等权再平衡换手扣同口径成本",
            "verdict_note": "同成本档下判定：年化收益大于0且最大回撤不超过基准",
            "disclaimer": DISCLAIMER,
        },
        "summary": summary, "metrics": metrics,
        "periods": pd.DataFrame(period_rows),
        "nav": {multiplier: nav_frames[multiplier] for multiplier in multipliers},
    }


def save_cost_reports(result, directory=None):
    """保存成本敏感度明细 CSV、逐期净值 JSON 与摘要 JSON。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    stem = "smart_beta_cost_%s_%s_%s" % (
        meta["start"].replace("-", ""), meta["end"].replace("-", ""), meta["codes"])
    csv_path = directory / (stem + ".csv")
    result["summary"].to_csv(csv_path, index=False, encoding="utf-8-sig")
    nav_path = directory / (stem + "_nav.json")
    nav_payload = {}
    for multiplier, frame in result["nav"].items():
        nav_payload["cost_x%s" % multiplier] = {
            name: {date: round(float(value), 8) for date, value in frame[name].items()}
            for name in frame.columns
        }
    nav_path.write_text(json.dumps(
        {"meta": meta, "nav": nav_payload}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    summary_path = directory / (stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "meta": meta,
        "rows": result["summary"].to_dict("records"),
        "periods": result["periods"].to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"csv": csv_path, "nav": nav_path, "summary": summary_path}


def print_cost_report(result):
    """按成本档打印净绩效与拖累。"""
    meta = result["meta"]
    print("\n=== Smart Beta 交易成本与换手敏感性 ===")
    print("股票池 %d 只 | %s ~ %s | 单边成本 %.4f%%（佣金 %.2f%% + 滑点 %.2f%%）" % (
        meta["codes"], meta["start"], meta["end"], meta["one_way_cost"] * 100,
        meta["commission"] * 100, meta["slippage"] * 100))
    summary = result["summary"]
    for multiplier in meta["cost_multipliers"]:
        frame = summary[summary["cost_multiplier"] == multiplier]
        benchmark = frame.iloc[0]
        print("\n--- 成本 %d 倍（单边 %.3f%%）基准年化 %.2f%%，基准最大回撤 %.2f%% ---" % (
            multiplier, benchmark["one_way_cost"] * 100,
            benchmark["benchmark_annual_return"] * 100,
            benchmark["benchmark_max_drawdown"] * 100))
        print("%-22s %9s %9s %9s %9s %9s %8s" % (
            "组合", "总收益", "年化", "成本拖累", "夏普", "最大回撤", "累计换手"))
        for _, row in frame.iterrows():
            print("%-22s %8.2f%% %8.2f%% %8.2f%% %9.3f %8.2f%% %8.2f  %s" % (
                row["label"], row["total_return"] * 100,
                row["annual_return"] * 100, row["annual_cost_drag"] * 100,
                row["sharpe"], row["max_drawdown"] * 100, row["total_turnover"],
                "通过" if row["passed"] else "未通过"))


def synthetic_cost_test():
    """验证成本扣减、换手边界与基准换手的计算口径。"""
    assert abs(one_way_cost() - 0.0013) < 1e-12
    assert abs(apply_cost(0.10, 0.0, 0.0013) - 0.10) < 1e-12
    assert abs(apply_cost(0.0, 1.0, 0.0013) + 0.0013) < 1e-12
    assert abs(apply_cost(0.0, 0.5, 0.0013) + 0.00065) < 1e-12
    # 成本翻倍时拖累约翻倍（用零毛收益隔离成本影响）
    single = -apply_cost(0.0, 0.5, 0.0013)
    double = -apply_cost(0.0, 0.5, 0.0026)
    assert abs(double - 2 * single) < 1e-12
    try:
        apply_cost(0.0, -0.1, 0.0013)
    except ValueError:
        pass
    else:
        raise AssertionError("负换手应报错")
    try:
        one_way_cost(-0.0001, 0.001)
    except ValueError:
        pass
    else:
        raise AssertionError("负佣金应报错")

    same = pd.Series({"a": 0.10, "b": 0.10, "c": 0.10})
    assert benchmark_turnover(same) < 1e-12, "同涨幅再平衡不应产生换手"
    spread = pd.Series({"a": 0.30, "b": 0.0, "c": 0.0})
    expected = 0.5 * (abs(1.30 / 3.30 - 1 / 3) + 2 * abs(1.00 / 3.30 - 1 / 3))
    assert abs(benchmark_turnover(spread) - expected) < 1e-12
    assert benchmark_turnover(pd.Series({"a": np.nan})) == 0.0
    print("合成数据：单边成本、零换手、成本翻倍、负值校验与基准换手均通过")


if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()
    synthetic_cost_test()
    result = run_cost_sensitivity(multipliers=COST_MULTIPLIERS)
    print_cost_report(result)
    summary = result["summary"]

    assert set(summary["cost_multiplier"]) == set(COST_MULTIPLIERS)
    assert len(summary) == len(COST_MULTIPLIERS) * len(PORTFOLIOS)
    # 成本越高，同一组合的净年化不得更高；0 倍应等于毛收益
    for name in PORTFOLIOS:
        frame = summary[summary["portfolio"] == name].set_index("cost_multiplier")
        assert abs(frame.loc[0, "annual_cost_drag"]) < 1e-12, name
        assert frame.loc[0, "annual_return"] >= frame.loc[1, "annual_return"] >= \
            frame.loc[2, "annual_return"], name
        assert frame.loc[1, "annual_cost_drag"] > 0, name
    # 全期毛收益应与既有回测一致（双因子 26.59%、五因子 40.97%）
    gross = summary[summary["cost_multiplier"] == 0].set_index("portfolio")
    assert abs(gross.loc["dual", "total_return"] - 0.26586) < 0.001
    assert abs(gross.loc["five_factor", "total_return"] - 0.40969) < 0.001
    for multiplier, frame in result["nav"].items():
        for name in frame.columns:
            assert abs(float(frame[name].iloc[0]) - 1.0) < 1e-12

    paths = save_cost_reports(result)
    assert len(pd.read_csv(paths["csv"])) == len(summary)
    payload = json.loads(paths["nav"].read_text(encoding="utf-8"))
    assert set(payload["nav"]) == {"cost_x%s" % value for value in COST_MULTIPLIERS}
    print("\n报告已保存：")
    for path in paths.values():
        print(path)
    print("\n联调清单：1 实际换手 → 2 单边成本扣减 → 3 基准同口径成本 → 4 三档成本对比 → 5 报告落盘，均已完成")
    print("smart_beta_cost.py 自测通过；%s" % DISCLAIMER)
