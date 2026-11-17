# -*- coding: utf-8 -*-
"""低价 + 低波 Smart Beta 月频回测（Step I2-alt-1）。

组合定义：
- low_price：低价评分前 20% 股票等权；
- low_vol：过去 60 日低波评分前 20% 股票等权；
- dual：低价、低波两个子组合各占 50% 资金；
- benchmark：本地 300 只股票池等权，仅用于比较回撤，不要求策略跑赢它。

目标与判定：
- 目标是跑赢现金（年化收益 > 0），并把最大回撤控制在不高于池内等权基准；
- dual 同时满足两项记为「通过」，否则记为「未通过」；
- 不计交易成本、不接券商、不下单，结果仅用于本地研究。

防前视约束：
- 每月最后一个交易日收盘后，用当日及之前数据确定下一期持仓；
- 持仓收益严格按本月调仓日收盘到下月调仓日收盘计算；
- 全部数据来自本地 SQLite，报告保存到本地 CSV / JSON / parquet。

仅研究用途，不构成投资建议。
"""

import sys
import json
import math
import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.factor_analysis import load_all_daily, get_rebalance_dates
from core.smart_beta import (
    DISCLAIMER, DEFAULT_PCT, DEFAULT_VOL_WINDOW, build_exposures,
)

SMART_BETA_DIR = db.DATA_DIR / "smart_beta"
PORTFOLIO_NAMES = ("low_price", "low_vol", "dual")
PORTFOLIO_LABELS = {
    "low_price": "低价前20%",
    "low_vol": "低波前20%",
    "dual": "低价50%+低波50%",
    "benchmark": "池内等权基准",
}


def period_returns(data_map, start_date, end_date):
    """计算各股票从 start_date 收盘到 end_date 收盘的收益，缺行情记 NaN。"""
    result = {}
    for code, df in data_map.items():
        dates = df["date"].values
        if start_date not in dates or end_date not in dates:
            result[code] = np.nan
            continue
        start_idx = int(np.where(dates == start_date)[0][0])
        end_idx = int(np.where(dates == end_date)[0][0])
        start_close = float(df.iloc[start_idx]["close"])
        end_close = float(df.iloc[end_idx]["close"])
        result[code] = (end_close / start_close - 1
                        if start_close > 0 else np.nan)
    return pd.Series(result, dtype=float)


def equal_weight_return(returns, codes):
    """指定持仓的等权收益；全部缺行情时按 0 处理并保留缺失记录。"""
    if not codes:
        return 0.0
    values = returns.reindex(codes).dropna()
    return float(values.mean()) if len(values) else 0.0


def holding_turnover(previous, current):
    """单边换手：持仓对称差 / 两期持仓数之和；首次建仓记 100%。"""
    previous, current = set(previous), set(current)
    if not previous:
        return 1.0 if current else 0.0
    denominator = len(previous) + len(current)
    return (len(previous.symmetric_difference(current)) / denominator
            if denominator else 0.0)


def performance_metrics(nav, turnovers=None):
    """按月频净值计算收益、夏普、回撤与换手；夏普使用 sqrt(12) 年化。"""
    nav = pd.Series(nav, dtype=float)
    returns = nav.pct_change().dropna()
    periods = len(returns)
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1)
    annual_return = ((1 + total_return) ** (12.0 / periods) - 1
                     if periods > 0 and 1 + total_return > 0 else 0.0)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (float(returns.mean() / std * math.sqrt(12))
              if std > 0 else 0.0)
    max_drawdown = float(abs((nav / nav.cummax() - 1).min()))
    return {
        "total_return": total_return,
        "annual_return": float(annual_return),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "total_turnover": float(np.sum(turnovers)) if turnovers else 0.0,
        "periods": int(periods),
    }


def run_smart_beta_backtest(codes=None, start_date="2024-05-01",
                            end_date="2026-11-13", pct=DEFAULT_PCT,
                            vol_window=DEFAULT_VOL_WINDOW, fq="qfq"):
    """执行月频固定暴露回测，返回净值、持仓、换手与绩效。"""
    if codes is None:
        codes = [r["code"] for r in db.query_all(
            "SELECT DISTINCT code FROM daily_bars WHERE fq=? ORDER BY code", (fq,))]
    data_map = load_all_daily(codes, fq=fq)
    if not data_map:
        raise ValueError("无可用本地日线数据")

    dates = get_rebalance_dates(list(data_map), freq="M", fq=fq)
    dates = [d for d in dates if start_date <= d <= end_date]
    if len(dates) < 2:
        raise ValueError("调仓日不足，至少需要两个自然月末交易日")

    nav_map = {name: [1.0] for name in PORTFOLIO_NAMES + ("benchmark",)}
    turnover_map = {name: [] for name in PORTFOLIO_NAMES}
    previous = {name: set() for name in PORTFOLIO_NAMES[:2]}
    holding_rows = []
    period_rows = []

    for i in range(len(dates) - 1):
        trade_date, next_date = dates[i], dates[i + 1]
        exposure = build_exposures(data_map, trade_date, pct=pct,
                                   vol_window=vol_window)
        returns = period_returns(data_map, trade_date, next_date)
        portfolio_returns = {
            name: equal_weight_return(returns, exposure[name])
            for name in PORTFOLIO_NAMES[:2]
        }
        # 双因子组合不做交集选股，直接按两个子组合各半仓计算
        portfolio_returns["dual"] = (0.5 * portfolio_returns["low_price"]
                                     + 0.5 * portfolio_returns["low_vol"])
        benchmark_return = equal_weight_return(returns, list(data_map))

        current_turnovers = {}
        for name in PORTFOLIO_NAMES[:2]:
            current = exposure[name]
            current_turnovers[name] = holding_turnover(previous[name], current)
            turnover_map[name].append(current_turnovers[name])
            previous[name] = set(current)
            weight = 1.0 / len(current) if current else 0.0
            for code in current:
                holding_rows.append({
                    "date": trade_date, "next_date": next_date,
                    "portfolio": name, "code": code, "weight": weight,
                    "price": exposure["scores"].get("price", pd.Series()).get(code),
                    "volatility": exposure["scores"].get("volatility", pd.Series()).get(code),
                    "period_return": returns.get(code),
                })
        # 双因子组合的换手等于两个半仓子组合换手的资金加权平均
        turnover_map["dual"].append(0.5 * (current_turnovers["low_price"]
                                           + current_turnovers["low_vol"]))

        for name in PORTFOLIO_NAMES:
            nav_map[name].append(nav_map[name][-1]
                                 * (1 + portfolio_returns[name]))
        nav_map["benchmark"].append(nav_map["benchmark"][-1]
                                    * (1 + benchmark_return))
        period_rows.append({
            "date": trade_date, "next_date": next_date,
            "low_price_return": portfolio_returns["low_price"],
            "low_vol_return": portfolio_returns["low_vol"],
            "dual_return": portfolio_returns["dual"],
            "benchmark_return": benchmark_return,
            "low_price_count": len(exposure["low_price"]),
            "low_vol_count": len(exposure["low_vol"]),
        })

    nav_frame = pd.DataFrame(nav_map, index=dates)
    nav_frame.index.name = "date"
    metrics = {name: performance_metrics(nav_frame[name], turnover_map[name])
               for name in PORTFOLIO_NAMES}
    metrics["benchmark"] = performance_metrics(nav_frame["benchmark"])
    benchmark_drawdown = metrics["benchmark"]["max_drawdown"]
    verdicts = {}
    for name in PORTFOLIO_NAMES:
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
            "pct": pct, "vol_window": vol_window, "freq": "M", "fq": fq,
            "price_note": "本地仅有前复权行情，低价暂以前复权收盘价排序",
            "dual_note": "双因子为低价、低波子组合各50%资金",
            "disclaimer": DISCLAIMER,
        },
        "nav": nav_frame,
        "holdings": pd.DataFrame(holding_rows),
        "periods": pd.DataFrame(period_rows),
        "turnover": pd.DataFrame(turnover_map, index=dates[1:]),
        "metrics": metrics,
        "verdicts": verdicts,
        "passed": verdicts["dual"]["passed"],
    }


def save_reports(result, directory=None):
    """保存汇总 CSV/JSON、逐月净值 CSV、持仓明细 parquet。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    stem = "smart_beta_%s_%s_%s" % (meta["start"].replace("-", ""),
                                    meta["end"].replace("-", ""),
                                    meta["codes"])
    rows = []
    for name in PORTFOLIO_NAMES + ("benchmark",):
        row = {"portfolio": name, "label": PORTFOLIO_LABELS[name]}
        row.update(result["metrics"][name])
        if name in result["verdicts"]:
            row.update(result["verdicts"][name])
        rows.append(row)
    csv_path = directory / (stem + ".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    nav_path = directory / (stem + "_nav.csv")
    result["nav"].to_csv(nav_path, encoding="utf-8-sig")
    holdings_path = directory / (stem + "_holdings.parquet")
    result["holdings"].to_parquet(holdings_path, index=False)
    payload = {
        "meta": meta, "metrics": result["metrics"],
        "verdicts": result["verdicts"], "passed": result["passed"],
        "periods": result["periods"].to_dict("records"),
    }
    json_path = directory / (stem + ".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    default=str), encoding="utf-8")
    return {"csv": csv_path, "json": json_path,
            "nav": nav_path, "holdings": holdings_path}


def print_report(result):
    """打印 Smart Beta 回测摘要与双因子判定。"""
    meta = result["meta"]
    print("\n=== 低价 + 低波 Smart Beta（月频）===")
    print("股票池 %d 只 | %s ~ %s | 各前 %.0f%% | 低波 %d 日" % (
        meta["codes"], meta["start"], meta["end"], meta["pct"] * 100,
        meta["vol_window"]))
    print("-" * 92)
    print("%-20s %10s %10s %8s %10s %9s %10s" % (
        "组合", "总收益", "年化", "夏普", "最大回撤", "胜率", "累计换手"))
    print("-" * 92)
    for name in PORTFOLIO_NAMES + ("benchmark",):
        m = result["metrics"][name]
        print("%-20s %9.2f%% %9.2f%% %8.3f %9.2f%% %8.2f%% %9.2f" % (
            PORTFOLIO_LABELS[name], m["total_return"] * 100,
            m["annual_return"] * 100, m["sharpe"], m["max_drawdown"] * 100,
            m["win_rate"] * 100, m["total_turnover"]))
    print("-" * 92)
    for name in PORTFOLIO_NAMES:
        v = result["verdicts"][name]
        print("%s：跑赢现金=%s，回撤不高于基准=%s，判定=%s" % (
            PORTFOLIO_LABELS[name], v["beat_cash"],
            v["drawdown_not_above_benchmark"], "通过" if v["passed"] else "未通过"))


def synthetic_exposure_test():
    """合成数据验证低价/低波排序，并确认修改未来行情不影响当期暴露。"""
    dates = pd.bdate_range("2023-01-02", periods=90).strftime("%Y-%m-%d").tolist()
    data_map = {}
    for j, code in enumerate(("000001", "000002", "000003", "000004", "000005")):
        if code == "000001":
            # 最低价但波动较大，应进入低价组合而非低波组合
            close = 5 * np.cumprod(np.tile([1.02, 0.98], 45))
        elif code == "000002":
            # 价格稍高但几乎不波动，应进入低波组合
            close = 10 * np.cumprod(np.full(90, 1.0001))
        else:
            # 其余股票加入明显但不完全相同的价格波动，避免低波排序退化为浮点误差
            wave = 1 + 0.02 * np.sin(np.arange(90) * 0.7 + j)
            close = (20 + j * 5) * wave * (1 + 0.001 * np.arange(90))
        data_map[code] = pd.DataFrame({"date": dates, "close": close})
    trade_date = dates[70]
    exposure = build_exposures(data_map, trade_date, pct=0.2, vol_window=60)
    assert exposure["low_price"] == ["000001"], exposure["low_price"]
    assert exposure["low_vol"] == ["000002"], exposure["low_vol"]

    changed = copy.deepcopy(data_map)
    for df in changed.values():
        df.loc[df["date"] > trade_date, "close"] *= 100
    after = build_exposures(changed, trade_date, pct=0.2, vol_window=60)
    assert after["low_price"] == exposure["low_price"], "低价暴露使用了未来数据"
    assert after["low_vol"] == exposure["low_vol"], "低波暴露使用了未来数据"
    print("合成数据：低价/低波排序与防未来函数校验通过")


if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()
    synthetic_exposure_test()

    # 真实数据验证：沿用 I1 的 300 只本地股票池与样本外区间
    result = run_smart_beta_backtest(start_date="2024-05-01",
                                     end_date="2026-11-13")
    print_report(result)
    paths = save_reports(result)

    assert result["nav"].index.is_monotonic_increasing
    assert (result["nav"] > 0).all().all(), "净值必须为正"
    assert set(result["metrics"]) == set(PORTFOLIO_NAMES + ("benchmark",))
    assert len(result["periods"]) == len(result["nav"]) - 1
    for name in PORTFOLIO_NAMES:
        turnover = result["turnover"][name]
        assert turnover.between(0, 1).all(), "%s 换手超出 [0,1]" % name
    print("\n报告已保存：")
    for path in paths.values():
        print(path)
    print("\nsmart_beta_backtest.py 自测通过；%s" % DISCLAIMER)
