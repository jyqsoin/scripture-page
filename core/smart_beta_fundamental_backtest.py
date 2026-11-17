# -*- coding: utf-8 -*-
"""低 PE/PB 与四/五因子 Smart Beta 回测（Step I2-alt-4B）。

低 PE/PB 仅保留正值，各取最低前 20% 等权。
四因子：低价/低波/动量/低 PE 各 25%；五因子再加入低 PB，各 20%。
多因子交集不足 10 只时，在并集中按各因子百分位名次之和取前 30 只。

估值只使用调仓日当日或之前的本地记录，无未来函数；不接券商、不下单。
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
    price_scores, volatility_scores, momentum_scores, select_bottom, select_top,
    build_exposures, build_three_factor_exposures,
)
from core.smart_beta_backtest import (
    SMART_BETA_DIR, period_returns, equal_weight_return, performance_metrics,
    holding_turnover,
)
from core.smart_beta_3factor_backtest import weighted_return, weighted_turnover
from core.valuation_data import load_valuation, latest_valuation

PORTFOLIO_NAMES = (
    "dual", "three_equal", "low_pe", "low_pb", "four_factor", "five_factor",
)
PORTFOLIO_LABELS = {
    "dual": "双因子50:50", "three_equal": "三因子等权",
    "low_pe": "低PE前20%", "low_pb": "低PB前20%",
    "four_factor": "四因子等权25%", "five_factor": "五因子等权20%",
    "benchmark": "池内等权基准",
}


def positive_valuation_scores(valuation_map, trade_date, column):
    """取最近可见 PE/PB，剔除非正值；返回数值越低越好的升序 Series。"""
    if column not in ("pe_ttm", "pb_mrq"):
        raise ValueError("column 只支持 pe_ttm / pb_mrq")
    values = {}
    for code, frame in valuation_map.items():
        latest = latest_valuation(frame, trade_date)
        value = latest.get(column) if latest else np.nan
        if value is not None and pd.notna(value) and float(value) > 0:
            values[code] = float(value)
    return pd.Series(values, dtype=float).sort_values()


def build_value_exposures(valuation_map, trade_date, pct=DEFAULT_PCT):
    """分别构建低 PE、低 PB 单因子前 20% 等权暴露。"""
    pe = positive_valuation_scores(valuation_map, trade_date, "pe_ttm")
    pb = positive_valuation_scores(valuation_map, trade_date, "pb_mrq")
    low_pe, low_pb = select_bottom(pe, pct), select_bottom(pb, pct)
    if not low_pe or not low_pb:
        raise ValueError("正 PE/PB 样本不足，无法构建估值暴露")
    return {
        "date": trade_date, "low_pe": low_pe, "low_pb": low_pb,
        "scores": pd.DataFrame({"pe_ttm": pe, "pb_mrq": pb}),
        "counts": {"low_pe": len(low_pe), "low_pb": len(low_pb)},
    }


def _sleeve_weights(sleeves, allocations, selected):
    """按各单因子子组合等权资金份额累加，再在入选股票内归一化。"""
    raw = pd.Series(0.0, index=selected, dtype=float)
    for factor, members in sleeves.items():
        member_set = set(members)
        share = allocations[factor] / len(members)
        for code in selected:
            if code in member_set:
                raw.loc[code] += share
    if raw.sum() <= 0:
        raise ValueError("组合权重之和为 0")
    return raw / raw.sum()


def build_fundamental_composite(
        data_map, valuation_map, trade_date, include_pe=True, include_pb=False,
        pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, min_intersection=10, union_top_n=30,
        momentum_floor=None):
    """构建四因子或五因子等权组合，选择规则与既有三因子一致。

    momentum_floor 为 I2-alt-9 新增的可选参数，默认 None 保持原有并集行为：
    - 仅在“交集不足、走并集前 union_top_n 只”时生效；
    - 强制动量子组合至少入选 round(union_top_n * momentum_floor) 只；
    - 若动量持仓权重合计低于 momentum_floor，则按比例上调至该下限，
      其余因子按原相对权重等比例下调，总权重仍归一化为 1。
    I2-alt-8 归因显示：S4 期间名次之和会系统性淘汰纯动量股，导致五因子
    实际动量权重长期为 0，故新增该下限以保留动量暴露。
    """
    if not include_pe and not include_pb:
        raise ValueError("四/五因子至少需要一个估值维度")
    scores = {
        "low_price": price_scores(data_map, trade_date),
        "low_vol": volatility_scores(data_map, trade_date, vol_window),
        "momentum": momentum_scores(data_map, trade_date, momentum_window, momentum_skip),
    }
    if include_pe:
        scores["low_pe"] = positive_valuation_scores(valuation_map, trade_date, "pe_ttm")
    if include_pb:
        scores["low_pb"] = positive_valuation_scores(valuation_map, trade_date, "pb_mrq")
    sleeves = {
        "low_price": select_bottom(scores["low_price"], pct),
        "low_vol": select_bottom(scores["low_vol"], pct),
        "momentum": select_top(scores["momentum"], pct),
    }
    if include_pe:
        sleeves["low_pe"] = select_bottom(scores["low_pe"], pct)
    if include_pb:
        sleeves["low_pb"] = select_bottom(scores["low_pb"], pct)
    if any(not members for members in sleeves.values()):
        raise ValueError("有效样本不足，无法构建基本面多因子组合")

    ranks = {}
    for factor, values in scores.items():
        ranks[factor + "_rank"] = values.rank(
            method="average", pct=True, ascending=(factor != "momentum"))
    rank_frame = pd.DataFrame(ranks).fillna(1.0)
    rank_frame["rank_sum"] = rank_frame.sum(axis=1)
    ordered = rank_frame.assign(code=rank_frame.index).sort_values(
        ["rank_sum", "code"], kind="mergesort")
    member_sets = [set(members) for members in sleeves.values()]
    intersection_set = set.intersection(*member_sets)
    union_set = set.union(*member_sets)
    intersection = ordered.loc[ordered.index.isin(intersection_set)].index.tolist()
    union_top = ordered.loc[ordered.index.isin(union_set)].head(union_top_n).index.tolist()
    use_intersection = len(intersection) >= min_intersection
    selected = intersection if use_intersection else union_top

    # 并集规则修复：强制动量因子入选并保底仓位，避免被名次之和挤出
    momentum_set = set(sleeves["momentum"])
    momentum_slots = 0
    if momentum_floor is not None:
        if not 0 < momentum_floor < 1:
            raise ValueError("momentum_floor 必须在 (0, 1) 内")
        if not use_intersection:
            momentum_slots = max(1, int(round(union_top_n * momentum_floor)))
            momentum_slots = min(momentum_slots, len(union_top))
            picked = [code for code in selected if code in momentum_set]
            if len(picked) < momentum_slots:
                candidates = [code for code in ordered.index
                              if code in momentum_set and code not in selected]
                additions = candidates[:momentum_slots - len(picked)]
                droppable = [code for code in reversed(selected)
                             if code not in momentum_set]
                dropped = set(droppable[:len(additions)])
                selected = [code for code in selected if code not in dropped] + additions

    allocations = {factor: 1.0 / len(sleeves) for factor in sleeves}
    weights = _sleeve_weights(sleeves, allocations, selected)
    if momentum_slots:
        in_momentum = weights.index.isin(momentum_set)
        momentum_weight = float(weights[in_momentum].sum())
        if 0 < momentum_weight < momentum_floor and (~in_momentum).any():
            adjusted = weights.copy()
            adjusted[in_momentum] *= momentum_floor / momentum_weight
            adjusted[~in_momentum] *= (
                (1 - momentum_floor) / adjusted[~in_momentum].sum())
            weights = adjusted / adjusted.sum()
    momentum_selected = int(weights.index.isin(momentum_set).sum())
    momentum_weight = float(weights[weights.index.isin(momentum_set)].sum())
    score_frame = pd.DataFrame({
        "price": scores["low_price"], "volatility": scores["low_vol"],
        "momentum": scores["momentum"],
        "pe_ttm": scores.get("low_pe"), "pb_mrq": scores.get("low_pb"),
    }).join(rank_frame, how="outer")
    return {
        "date": trade_date, "sleeves": sleeves, "allocations": allocations,
        "selected": selected, "weights": weights, "scores": score_frame,
        "selection_rule": "intersection" if use_intersection else "union_top",
        "counts": {"intersection": len(intersection), "union": len(union_set),
                   "selected": len(selected)},
        "momentum_floor": momentum_floor, "momentum_slots": momentum_slots,
        "momentum_selected": momentum_selected,
        "momentum_weight": momentum_weight,
    }


def _score_value(scores, column, code):
    """不同组合的评分列不一致，缺失维度统一返回 NaN。"""
    return scores[column].get(code) if column in scores else np.nan


def run_fundamental_comparison(
        codes=None, start_date="2024-05-31", end_date="2026-11-13",
        pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, fq="qfq", momentum_floor=None):
    """同月频日期对比双/三因子、低 PE/PB、四/五因子及池内基准。

    momentum_floor 为 I2-alt-12 新增的可选参数，默认 None 保持原规则：
    仅透传给五因子并集分支，四因子与双/三因子口径完全不变。
    """
    if codes is None:
        codes = [row["code"] for row in db.query_all(
            "SELECT DISTINCT code FROM daily_bars WHERE fq=? ORDER BY code", (fq,))]
    data_map = load_all_daily(codes, fq=fq)
    valuation_map = {code: load_valuation(code) for code in data_map}
    missing = [code for code, frame in valuation_map.items() if frame.empty]
    if missing:
        raise ValueError("以下股票缺少本地估值数据：%s" % ",".join(missing[:10]))
    dates = get_rebalance_dates(list(data_map), freq="M", fq=fq)
    dates = [date for date in dates if start_date <= date <= end_date]
    if len(dates) < 2:
        raise ValueError("调仓日不足")

    nav_map = {name: [1.0] for name in PORTFOLIO_NAMES + ("benchmark",)}
    turnover_map = {name: [] for name in PORTFOLIO_NAMES}
    previous = {name: pd.Series(dtype=float) for name in PORTFOLIO_NAMES}
    previous_dual = {"low_price": [], "low_vol": []}
    holdings, periods = [], []

    for i in range(len(dates) - 1):
        trade_date, next_date = dates[i], dates[i + 1]
        returns = period_returns(data_map, trade_date, next_date)
        dual = build_exposures(data_map, trade_date, pct, vol_window)
        three = build_three_factor_exposures(
            data_map, trade_date, pct=pct, vol_window=vol_window,
            momentum_window=momentum_window, momentum_skip=momentum_skip, mode="equal")
        value = build_value_exposures(valuation_map, trade_date, pct)
        four = build_fundamental_composite(
            data_map, valuation_map, trade_date, True, False, pct, vol_window,
            momentum_window, momentum_skip)
        five = build_fundamental_composite(
            data_map, valuation_map, trade_date, True, True, pct, vol_window,
            momentum_window, momentum_skip, momentum_floor=momentum_floor)
        dual_weights = _sleeve_weights(
            {"low_price": dual["low_price"], "low_vol": dual["low_vol"]},
            {"low_price": 0.5, "low_vol": 0.5},
            list(dict.fromkeys(dual["low_price"] + dual["low_vol"])))
        weights_map = {
            "dual": dual_weights, "three_equal": three["weights"],
            "low_pe": pd.Series(1.0 / len(value["low_pe"]), index=value["low_pe"]),
            "low_pb": pd.Series(1.0 / len(value["low_pb"]), index=value["low_pb"]),
            "four_factor": four["weights"], "five_factor": five["weights"],
        }
        scores_map = {
            "dual": dual["scores"], "three_equal": three["scores"],
            "low_pe": value["scores"], "low_pb": value["scores"],
            "four_factor": four["scores"], "five_factor": five["scores"],
        }
        rules = {
            "dual": "dual_sleeves", "three_equal": three["selection_rule"],
            "low_pe": "positive_pe_top_pct", "low_pb": "positive_pb_top_pct",
            "four_factor": four["selection_rule"], "five_factor": five["selection_rule"],
        }

        period_row = {
            "date": trade_date, "next_date": next_date,
            "three_selection_rule": three["selection_rule"],
            "four_selection_rule": four["selection_rule"],
            "five_selection_rule": five["selection_rule"],
            "four_intersection_count": four["counts"]["intersection"],
            "five_intersection_count": five["counts"]["intersection"],
            "four_selected_count": four["counts"]["selected"],
            "five_selected_count": five["counts"]["selected"],
            "five_momentum_weight": five["momentum_weight"],
            "five_momentum_selected": five["momentum_selected"],
        }
        for name in PORTFOLIO_NAMES:
            weights = weights_map[name]
            if name == "dual":
                # 与 I2-alt-1 完全一致：两个半仓分别计算收益和换手
                low_ret = weighted_return(returns, pd.Series(
                    1.0 / len(dual["low_price"]), index=dual["low_price"]))
                vol_ret = weighted_return(returns, pd.Series(
                    1.0 / len(dual["low_vol"]), index=dual["low_vol"]))
                portfolio_return = 0.5 * low_ret + 0.5 * vol_ret
                turnover = 0.5 * holding_turnover(previous_dual["low_price"], dual["low_price"])
                turnover += 0.5 * holding_turnover(previous_dual["low_vol"], dual["low_vol"])
                previous_dual = {"low_price": dual["low_price"], "low_vol": dual["low_vol"]}
            else:
                portfolio_return = weighted_return(returns, weights)
                turnover = weighted_turnover(previous[name], weights)
                previous[name] = weights.copy()
            nav_map[name].append(nav_map[name][-1] * (1 + portfolio_return))
            turnover_map[name].append(turnover)
            period_row[name + "_return"] = portfolio_return
            scores = scores_map[name]
            for code, weight in weights.items():
                holdings.append({
                    "date": trade_date, "next_date": next_date, "portfolio": name,
                    "code": code, "weight": float(weight), "selection_rule": rules[name],
                    "price": _score_value(scores, "price", code),
                    "volatility": _score_value(scores, "volatility", code),
                    "momentum": _score_value(scores, "momentum", code),
                    "pe_ttm": _score_value(scores, "pe_ttm", code),
                    "pb_mrq": _score_value(scores, "pb_mrq", code),
                    "period_return": returns.get(code),
                })
        benchmark_return = equal_weight_return(returns, list(data_map))
        nav_map["benchmark"].append(nav_map["benchmark"][-1] * (1 + benchmark_return))
        period_row["benchmark_return"] = benchmark_return
        periods.append(period_row)

    nav = pd.DataFrame(nav_map, index=dates)
    metrics = {name: performance_metrics(nav[name], turnover_map[name])
               for name in PORTFOLIO_NAMES}
    metrics["benchmark"] = performance_metrics(nav["benchmark"])
    benchmark_drawdown = metrics["benchmark"]["max_drawdown"]
    verdicts = {}
    for name in PORTFOLIO_NAMES:
        beat_cash = metrics[name]["annual_return"] > 0
        controlled = metrics[name]["max_drawdown"] <= benchmark_drawdown
        verdicts[name] = {"beat_cash": bool(beat_cash),
                          "drawdown_not_above_benchmark": bool(controlled),
                          "passed": bool(beat_cash and controlled)}
    comparisons = {}
    for name in PORTFOLIO_NAMES:
        if name == "three_equal":
            continue
        comparisons[name] = {
            "annual_return_delta_vs_three": metrics[name]["annual_return"] - metrics["three_equal"]["annual_return"],
            "max_drawdown_delta_vs_three": metrics[name]["max_drawdown"] - metrics["three_equal"]["max_drawdown"],
        }
    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": len(data_map), "start": dates[0], "end": dates[-1],
            "pct": pct, "vol_window": vol_window,
            "momentum_window": momentum_window, "momentum_skip": momentum_skip,
            "momentum_floor": momentum_floor,
            "freq": "M", "fq": fq, "valuation_source": "local_baostock",
            "valuation_rule": "仅使用最近可见正 PE/PB，负值和零不参与低估值排序",
            "selection_note": "四/五因子交集不足10只时，在并集中按百分位名次之和取前30只"
                              + ("；五因子并集分支动量最低权重 %g%%" % (momentum_floor * 100)
                                 if momentum_floor else "；五因子未启用动量下限"),
            "disclaimer": DISCLAIMER,
        },
        "nav": nav, "holdings": pd.DataFrame(holdings), "periods": pd.DataFrame(periods),
        "turnover": pd.DataFrame(turnover_map, index=dates[1:]),
        "metrics": metrics, "verdicts": verdicts, "comparisons": comparisons,
        "passed": {name: verdicts[name]["passed"] for name in PORTFOLIO_NAMES},
    }


def save_fundamental_reports(result, directory=None):
    """保存 7 行绩效 CSV、净值 CSV、完整 JSON 与持仓 parquet。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    stem = "smart_beta_fundamental_%s_%s_%s" % (
        meta["start"].replace("-", ""), meta["end"].replace("-", ""), meta["codes"])
    rows = []
    floor = meta.get("momentum_floor")
    for name in PORTFOLIO_NAMES + ("benchmark",):
        row = {"portfolio": name, "label": PORTFOLIO_LABELS[name]}
        if name == "five_factor" and floor:
            row["label"] = "%s（动量≥%g%%）" % (PORTFOLIO_LABELS[name], floor * 100)
        row["momentum_floor"] = floor if name == "five_factor" else np.nan
        row.update(result["metrics"][name])
        if name in result["verdicts"]:
            row.update(result["verdicts"][name])
        delta = result["comparisons"].get(name, {})
        row["annual_return_delta_vs_three"] = delta.get(
            "annual_return_delta_vs_three", 0.0 if name == "three_equal" else np.nan)
        row["max_drawdown_delta_vs_three"] = delta.get(
            "max_drawdown_delta_vs_three", 0.0 if name == "three_equal" else np.nan)
        rows.append(row)
    csv_path = directory / (stem + ".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    nav_path = directory / (stem + "_nav.csv")
    result["nav"].to_csv(nav_path, encoding="utf-8-sig")
    holdings_path = directory / (stem + "_holdings.parquet")
    result["holdings"].to_parquet(holdings_path, index=False)
    payload = {key: result[key] for key in ("meta", "metrics", "verdicts", "comparisons", "passed")}
    payload["periods"] = result["periods"].to_dict("records")
    json_path = directory / (stem + ".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    default=str), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "nav": nav_path, "holdings": holdings_path}


def print_fundamental_report(result):
    print("\n=== 低 PE/PB 与四/五因子 Smart Beta 对比 ===")
    print("%-20s %9s %9s %8s %10s %12s %12s %8s" % (
        "组合", "总收益", "年化", "夏普", "最大回撤", "年化差vs三(pp)", "回撤差vs三(pp)", "判定"))
    for name in PORTFOLIO_NAMES + ("benchmark",):
        metrics = result["metrics"][name]
        delta = result["comparisons"].get(name, {})
        annual_delta = delta.get("annual_return_delta_vs_three", 0.0 if name == "three_equal" else np.nan)
        drawdown_delta = delta.get("max_drawdown_delta_vs_three", 0.0 if name == "three_equal" else np.nan)
        verdict = "—" if name == "benchmark" else ("通过" if result["verdicts"][name]["passed"] else "未通过")
        label = PORTFOLIO_LABELS[name]
        floor = result["meta"].get("momentum_floor")
        if name == "five_factor" and floor:
            label += "（动量≥%g%%）" % (floor * 100)
        print("%-20s %8.2f%% %8.2f%% %8.3f %9.2f%% %12.2f %12.2f %8s" % (
            label, metrics["total_return"] * 100,
            metrics["annual_return"] * 100, metrics["sharpe"],
            metrics["max_drawdown"] * 100, annual_delta * 100,
            drawdown_delta * 100, verdict))


def synthetic_fundamental_test():
    """验证正 PE/PB 过滤、估值可见日期及四/五因子权重。"""
    dates = pd.bdate_range("2025-01-02", periods=280).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(404)
    data_map, valuation_map = {}, {}
    for j in range(100):
        code = "%06d" % (j + 1)
        returns = 0.0001 + j / 99 * 0.002 + rng.normal(0, 0.001 + (j % 10) * 0.0002, len(dates))
        data_map[code] = pd.DataFrame({
            "date": dates, "close": (2 + j * 0.1) * np.cumprod(1 + returns)})
        valuation_map[code] = pd.DataFrame({
            "date": [dates[-30], dates[-1]], "code": [code, code],
            "pe_ttm": [5 + j, 1000 + j], "pb_mrq": [0.5 + j / 100, 2 + j / 100]})
    trade_date = dates[-2]
    pe = positive_valuation_scores(valuation_map, trade_date, "pe_ttm")
    assert pe["000001"] == 5  # 不使用调仓日之后被改成 1000 的 PE
    valuation_map["000001"].loc[:, "pe_ttm"] = [-5, -1000]
    assert "000001" not in positive_valuation_scores(valuation_map, trade_date, "pe_ttm")
    valuation_map["000001"].loc[:, "pe_ttm"] = [5, 1000]
    value = build_value_exposures(valuation_map, trade_date)
    assert len(value["low_pe"]) == 20 and len(value["low_pb"]) == 20
    four = build_fundamental_composite(data_map, valuation_map, trade_date, True, False)
    five = build_fundamental_composite(data_map, valuation_map, trade_date, True, True)
    lowest_price = price_scores(data_map, trade_date).index[0]
    strongest = momentum_scores(data_map, trade_date).index[0]
    lowest_pe = positive_valuation_scores(valuation_map, trade_date, "pe_ttm").index[0]
    assert four["scores"].loc[lowest_price, "low_price_rank"] == four["scores"]["low_price_rank"].min()
    assert four["scores"].loc[strongest, "momentum_rank"] == four["scores"]["momentum_rank"].min()
    assert four["scores"].loc[lowest_pe, "low_pe_rank"] == four["scores"]["low_pe_rank"].min()
    assert len(four["selected"]) == 30 and len(five["selected"]) == 30
    assert abs(four["weights"].sum() - 1) < 1e-12 and abs(five["weights"].sum() - 1) < 1e-12
    assert all(abs(value - 0.25) < 1e-12 for value in four["allocations"].values())
    assert all(abs(value - 0.20) < 1e-12 for value in five["allocations"].values())
    print("合成数据：正 PE/PB 过滤、防未来函数、四/五因子选股及资金比例均通过")


if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()
    synthetic_fundamental_test()
    result = run_fundamental_comparison(
        start_date="2024-05-31", end_date="2026-11-13")
    print_fundamental_report(result)
    paths = save_fundamental_reports(result)
    assert len(result["periods"]) == 30 and len(result["nav"]) == 31
    assert ((result["turnover"] >= 0) & (result["turnover"] <= 1.000001)).all().all()
    for _, frame in result["holdings"].groupby(["date", "portfolio"]):
        assert abs(frame["weight"].sum() - 1) < 1e-9
    assert round(result["metrics"]["dual"]["annual_return"], 4) == 0.0989
    assert round(result["metrics"]["three_equal"]["annual_return"], 4) == 0.1078
    assert len(pd.read_csv(paths["csv"])) == 7
    print("\n报告已保存：")
    for path in paths.values():
        print(path)
    print("smart_beta_fundamental_backtest.py 自测与真实数据验证通过；%s" % DISCLAIMER)
