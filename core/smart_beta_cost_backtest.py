# -*- coding: utf-8 -*-
"""基于 I2-alt-10 固化基线的交易成本与换手敏感性（Step I2-alt-7）。

口径来源：
- 毛收益、每期换手完全来自 smart_beta_baseline.run_baseline 的固化配置：
  五因子动量下限 10%、四因子旧并集规则、三因子等权 33/33/34；
- 单边成本 = 佣金万三 + 滑点 0.1% = 0.13%，买卖各按该费率计一次；
- 期净收益 = (1 + 毛收益) × (1 - 单边换手 × 单边成本) - 1；
- 成本档位：0 倍（毛收益）、1 倍（万三 + 0.1%）、2 倍（万六 + 0.2%）；
- 判定沿用“年化收益 > 0 且最大回撤不超过同期池内等权基准（毛）11.95%”。

注意：smart_beta_cost.py 是 I2-alt-10 之前的版本（五因子未加动量下限），
本模块以固化基线为准，旧模块保留不动，二者数值不可混用。

防未来函数：选股、调仓日与收益区间全部复用固化基线，本模块只增加成本扣减。
不修改核心模块；不接券商、不下单、不上传云端。
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
from core.smart_beta import DISCLAIMER
from core.smart_beta_backtest import SMART_BETA_DIR, performance_metrics
from core.smart_beta_3factor_backtest import weighted_turnover
from core.smart_beta_cost import apply_cost, one_way_cost, COMMISSION, SLIPPAGE
from core.smart_beta_baseline import run_baseline, REFERENCE

PORTFOLIOS = ("three_equal", "four_factor", "five_factor")
PORTFOLIO_LABELS = {
    "three_equal": "三因子等权33/33/34",
    "four_factor": "四因子·旧并集规则（最终）",
    "five_factor": "五因子·动量≥10%（最终）",
}
COST_MULTIPLIERS = (0, 1, 2)
SCENARIO_LABELS = {0: "毛收益（0x）", 1: "基准成本（1x）", 2: "成本翻倍（2x）"}


def scenario_rates(commission=COMMISSION, slippage=SLIPPAGE,
                   multipliers=COST_MULTIPLIERS):
    """把成本倍数换算成各档单边成本率，0 倍表示不扣费。"""
    if not multipliers or any(value < 0 for value in multipliers):
        raise ValueError("成本倍数必须为非负序列")
    rate = one_way_cost(commission, slippage)
    return {multiplier: rate * multiplier for multiplier in multipliers}


def run_cost_backtest(
        codes=None, start_date="2024-05-31", end_date="2026-11-13",
        commission=COMMISSION, slippage=SLIPPAGE,
        multipliers=COST_MULTIPLIERS, months_per_segment=6):
    """复用固化基线逐期毛收益与换手，按多档成本累计净净值。"""
    rates = scenario_rates(commission, slippage, multipliers)
    base = run_baseline(codes=codes, start_date=start_date, end_date=end_date,
                        months_per_segment=months_per_segment)
    periods = base["periods"]
    dates = base["full_nav"].index
    benchmark = base["benchmark_metrics"]

    nav_map = {multiplier: {name: [1.0] for name in PORTFOLIOS}
               for multiplier in multipliers}
    turnover_map = {name: periods[name + "_turnover"].tolist()
                    for name in PORTFOLIOS}
    for _, row in periods.iterrows():
        for name in PORTFOLIOS:
            gross = float(row[name + "_return"])
            turnover = float(row[name + "_turnover"])
            for multiplier in multipliers:
                net = apply_cost(gross, turnover, rates[multiplier])
                nav_map[multiplier][name].append(
                    nav_map[multiplier][name][-1] * (1 + net))

    nav_frames = {multiplier: pd.DataFrame(nav_map[multiplier], index=dates)
                  for multiplier in multipliers}
    metrics = {
        multiplier: {name: performance_metrics(nav_frames[multiplier][name],
                                               turnover_map[name])
                     for name in PORTFOLIOS}
        for multiplier in multipliers
    }

    rows = []
    for multiplier in multipliers:
        for name in PORTFOLIOS:
            value = metrics[multiplier][name]
            gross = metrics[0][name] if 0 in metrics else metrics[min(metrics)][name]
            beat_cash = value["annual_return"] > 0
            controlled = value["max_drawdown"] <= benchmark["max_drawdown"]
            rows.append({
                "cost_multiplier": multiplier,
                "scenario": SCENARIO_LABELS.get(multiplier, "%dx" % multiplier),
                "one_way_cost": rates[multiplier],
                "portfolio": name, "label": PORTFOLIO_LABELS[name],
                "total_return": value["total_return"],
                "annual_return": value["annual_return"],
                "sharpe": value["sharpe"],
                "max_drawdown": value["max_drawdown"],
                "win_rate": value["win_rate"],
                "avg_turnover": value["avg_turnover"],
                "total_turnover": value["total_turnover"],
                "annual_cost_drag": gross["annual_return"] - value["annual_return"],
                "benchmark_annual_return": benchmark["annual_return"],
                "benchmark_max_drawdown": benchmark["max_drawdown"],
                "excess_return": value["total_return"] - benchmark["total_return"],
                "beat_cash": bool(beat_cash),
                "drawdown_not_above_benchmark": bool(controlled),
                "passed": bool(beat_cash and controlled),
            })
    full = pd.DataFrame(rows)
    sensitivity = sensitivity_table(full)

    period_rows = []
    for _, row in periods.iterrows():
        record = {"date": row["date"], "next_date": row["next_date"],
                  "benchmark_gross_return": row["benchmark_return"]}
        for name in PORTFOLIOS:
            gross = float(row[name + "_return"])
            turnover = float(row[name + "_turnover"])
            record[name + "_gross_return"] = gross
            record[name + "_turnover"] = turnover
            for multiplier in multipliers:
                if multiplier == 0:
                    continue
                record["%s_net_%dx" % (name, multiplier)] = apply_cost(
                    gross, turnover, rates[multiplier])
        period_rows.append(record)

    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": base["meta"]["codes"],
            "start": base["meta"]["start"], "end": base["meta"]["end"],
            "pct": base["meta"]["pct"],
            "vol_window": base["meta"]["vol_window"],
            "momentum_window": base["meta"]["momentum_window"],
            "momentum_skip": base["meta"]["momentum_skip"],
            "freq": "M", "fq": base["meta"]["fq"],
            "final_config": base["meta"]["final_config"],
            "commission": commission, "slippage": slippage,
            "one_way_cost": one_way_cost(commission, slippage),
            "scenario_rates": {str(key): value for key, value in rates.items()},
            "cost_multipliers": list(multipliers),
            "cost_note": "期净收益=(1+毛收益)×(1-单边换手×单边成本)-1；"
                         "单边换手=0.5×权重L1差，首次建仓记100%；未计印花税",
            "benchmark_note": "判定基准取固化基线的池内等权毛收益（夏普/回撤同口径）",
            "verdict_note": "年化收益大于0且最大回撤不超过基准（毛）最大回撤",
            "disclaimer": DISCLAIMER,
        },
        "full": full, "sensitivity": sensitivity,
        "periods": pd.DataFrame(period_rows),
        "nav": nav_frames, "benchmark_metrics": benchmark,
    }


def sensitivity_table(full):
    """每个组合一行：毛收益 / 1 倍 / 2 倍成本下的年化、回撤、拖累与判定。"""
    rows = []
    for name in PORTFOLIOS:
        frame = full[full["portfolio"] == name].set_index("cost_multiplier")
        if 0 not in frame.index:
            continue
        gross = frame.loc[0]
        one = frame.loc[1] if 1 in frame.index else gross
        double = frame.loc[2] if 2 in frame.index else one
        rows.append({
            "portfolio": name, "label": PORTFOLIO_LABELS[name],
            "gross_total_return": gross["total_return"],
            "gross_annual_return": gross["annual_return"],
            "gross_max_drawdown": gross["max_drawdown"],
            "avg_turnover": gross["avg_turnover"],
            "total_turnover": gross["total_turnover"],
            "annual_return_1x": one["annual_return"],
            "max_drawdown_1x": one["max_drawdown"],
            "annual_cost_drag_1x": one["annual_cost_drag"],
            "annual_return_2x": double["annual_return"],
            "max_drawdown_2x": double["max_drawdown"],
            "annual_cost_drag_2x": double["annual_cost_drag"],
            "passed_1x": bool(one["passed"]),
            "passed_2x": bool(double["passed"]),
            "still_passed_at_2x": bool(double["passed"]),
        })
    return pd.DataFrame(rows)


def save_cost_backtest_reports(result, directory=None):
    """保存扣费前后绩效、敏感度表、逐期明细、净值与摘要。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    stem = "smart_beta_cost_backtest_%s_%s_%s" % (
        meta["start"].replace("-", ""), meta["end"].replace("-", ""),
        meta["codes"])
    paths = {}
    for key in ("full", "sensitivity", "periods"):
        path = directory / ("%s_%s.csv" % (stem, key))
        result[key].to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    nav_path = directory / (stem + "_nav.json")
    nav_payload = {}
    for multiplier, frame in result["nav"].items():
        nav_payload["cost_x%s" % multiplier] = {
            name: {str(date): round(float(value), 8)
                   for date, value in frame[name].items()}
            for name in frame.columns
        }
    nav_path.write_text(json.dumps(
        {"meta": meta, "nav": nav_payload}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    paths["nav"] = nav_path
    summary_path = directory / (stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "meta": meta,
        "rows": result["full"].to_dict("records"),
        "sensitivity": result["sensitivity"].to_dict("records"),
        "benchmark_metrics": result["benchmark_metrics"],
        "periods": result["periods"].to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def print_cost_report(result):
    """按成本档打印净绩效与拖累，再打印横向敏感度表。"""
    meta = result["meta"]
    benchmark = result["benchmark_metrics"]
    print("\n=== 交易成本与换手敏感性（基于 I2-alt-10 固化基线）===")
    print("股票池 %d 只 | %s ~ %s | 单边成本 %.3f%%（佣金 %.2f%% + 滑点 %.2f%%）" % (
        meta["codes"], meta["start"], meta["end"],
        meta["one_way_cost"] * 100, meta["commission"] * 100,
        meta["slippage"] * 100))
    print("判定基准（毛）：年化 %.2f%% | 最大回撤 %.2f%%" % (
        benchmark["annual_return"] * 100, benchmark["max_drawdown"] * 100))

    for multiplier in meta["cost_multipliers"]:
        frame = result["full"][result["full"]["cost_multiplier"] == multiplier]
        print("\n--- %s（单边 %.3f%%）---" % (
            SCENARIO_LABELS.get(multiplier, "%dx" % multiplier),
            meta["scenario_rates"][str(multiplier)] * 100))
        print("%-30s %9s %9s %9s %8s %9s %8s" % (
            "组合", "总收益", "年化", "成本拖累", "夏普", "最大回撤", "判定"))
        for _, row in frame.iterrows():
            print("%-30s %8.2f%% %8.2f%% %8.2f%% %8.3f %8.2f%% %8s" % (
                row["label"], row["total_return"] * 100,
                row["annual_return"] * 100, row["annual_cost_drag"] * 100,
                row["sharpe"], row["max_drawdown"] * 100,
                "通过" if row["passed"] else "未通过"))

    print("\n--- 成本敏感度汇总（年化 / 最大回撤）---")
    print("%-30s %8s %8s %8s %8s %8s %8s %6s" % (
        "组合", "毛年化", "1x年化", "1x拖累", "2x年化", "2x拖累", "2x回撤", "2x判定"))
    for _, row in result["sensitivity"].iterrows():
        print("%-30s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %6s" % (
            row["label"], row["gross_annual_return"] * 100,
            row["annual_return_1x"] * 100, row["annual_cost_drag_1x"] * 100,
            row["annual_return_2x"] * 100, row["annual_cost_drag_2x"] * 100,
            row["max_drawdown_2x"] * 100,
            "通过" if row["passed_2x"] else "未通过"))
    for _, row in result["sensitivity"].iterrows():
        print("%-30s 平均单边换手 %.2f%% | 累计换手 %.2f | 1x 回撤 %.2f%%" % (
            row["label"], row["avg_turnover"] * 100, row["total_turnover"],
            row["max_drawdown_1x"] * 100))


def synthetic_cost_test():
    """验证单边成本、扣费公式、换手口径与成本翻倍的线性关系。"""
    assert abs(one_way_cost() - 0.0013) < 1e-12
    assert abs(one_way_cost(0.0006, 0.002) - 0.0026) < 1e-12
    # 零换手不扣费；成本为 0 时净收益等于毛收益
    assert abs(apply_cost(0.10, 0.0, 0.0013) - 0.10) < 1e-12
    assert abs(apply_cost(0.10, 0.5, 0.0) - 0.10) < 1e-12
    # 单位换手时，成本直接侵蚀期初资产
    assert abs(apply_cost(0.0, 1.0, 0.0013) + 0.0013) < 1e-12
    single = -apply_cost(0.0, 0.4, 0.0013)
    double = -apply_cost(0.0, 0.4, 0.0026)
    assert abs(double - 2 * single) < 1e-12
    # 换手口径：首次建仓 100%，同权重 0，完全换仓 100%
    empty = pd.Series(dtype=float)
    assert abs(weighted_turnover(empty, pd.Series({"a": 0.5, "b": 0.5})) - 1.0) < 1e-12
    same = pd.Series({"a": 0.5, "b": 0.5})
    assert weighted_turnover(same, same) < 1e-12
    assert abs(weighted_turnover(pd.Series({"a": 1.0}),
                                 pd.Series({"b": 1.0})) - 1.0) < 1e-12
    rates = scenario_rates()
    assert rates[0] == 0.0 and abs(rates[1] - 0.0013) < 1e-12
    assert abs(rates[2] - 0.0026) < 1e-12
    try:
        scenario_rates(multipliers=(-1,))
    except ValueError:
        pass
    else:
        raise AssertionError("负成本倍数应报错")
    print("合成数据：单边成本、零换手/满换手、成本翻倍线性与非法倍数校验均通过")


def _validate_result(result):
    """校验毛收益与固化基线一致、扣费单调、净值起点与判定口径正确。"""
    full = result["full"]
    benchmark = result["benchmark_metrics"]
    # 0 倍成本必须与 I2-alt-10 基线逐值一致
    gross = full[full["cost_multiplier"] == 0].set_index("portfolio")
    for name in PORTFOLIOS:
        reference = REFERENCE[name]
        assert abs(gross.loc[name, "total_return"] - reference["total_return"]) < 1e-9, name
        assert abs(gross.loc[name, "annual_return"] - reference["annual_return"]) < 1e-9, name
        assert abs(gross.loc[name, "max_drawdown"] - reference["max_drawdown"]) < 1e-9, name
        assert abs(gross.loc[name, "annual_cost_drag"]) < 1e-12, name

    # 成本越高净收益越低、拖累越大；2 倍拖累应大于 1 倍
    for name in PORTFOLIOS:
        frame = full[full["portfolio"] == name].set_index("cost_multiplier")
        assert frame.loc[0, "total_return"] > frame.loc[1, "total_return"], name
        assert frame.loc[1, "total_return"] > frame.loc[2, "total_return"], name
        assert frame.loc[1, "annual_cost_drag"] > 0, name
        assert frame.loc[2, "annual_cost_drag"] > frame.loc[1, "annual_cost_drag"], name
        for multiplier in (0, 1, 2):
            row = frame.loc[multiplier]
            expected = bool(row["annual_return"] > 0
                            and row["max_drawdown"] <= benchmark["max_drawdown"])
            assert bool(row["passed"]) == expected, (name, multiplier)

    # 换手合法区间：单边换手应在 [0, 1]，首次建仓为 100%
    for name in PORTFOLIOS:
        turnover = result["periods"][name + "_turnover"]
        assert ((turnover >= 0) & (turnover <= 1 + 1e-12)).all(), name
        assert abs(float(turnover.iloc[0]) - 1.0) < 1e-12, name

    # 每档净值都从 1.0 起算
    for multiplier, frame in result["nav"].items():
        for name in frame.columns:
            assert abs(float(frame[name].iloc[0]) - 1.0) < 1e-12, (multiplier, name)

    # 逐期净收益复利应与净值一致
    for multiplier in (1, 2):
        frame = result["nav"][multiplier]
        for name in PORTFOLIOS:
            compounded = float(np.prod(1 + result["periods"][
                "%s_net_%dx" % (name, multiplier)]) - 1)
            total = float(frame[name].iloc[-1] - 1)
            assert abs(compounded - total) < 1e-9, (multiplier, name)


def _run_cost_backtest_main():
    print(DISCLAIMER)
    synthetic_cost_test()
    result = run_cost_backtest(multipliers=COST_MULTIPLIERS)
    _validate_result(result)
    print_cost_report(result)
    paths = save_cost_backtest_reports(result)

    assert len(pd.read_csv(paths["full"])) == len(PORTFOLIOS) * len(COST_MULTIPLIERS)
    assert len(pd.read_csv(paths["sensitivity"])) == len(PORTFOLIOS)
    assert len(pd.read_csv(paths["periods"])) == 30
    payload = json.loads(paths["nav"].read_text(encoding="utf-8"))
    assert set(payload["nav"]) == {"cost_x0", "cost_x1", "cost_x2"}
    assert len(json.loads(paths["summary"].read_text(encoding="utf-8"))["rows"]) == \
        len(PORTFOLIOS) * len(COST_MULTIPLIERS)

    print("\n报告已保存：")
    for path in paths.values():
        print(path)
    print("\n联调清单：1 复用基线换手 → 2 单边成本扣减 → 3 成本翻倍 → 4 敏感度表 → 5 落盘，均已完成")
    print("smart_beta_cost_backtest.py 自测通过；%s" % DISCLAIMER)


if __name__ == "__main__":
    db.init_db()
    _run_cost_backtest_main()
