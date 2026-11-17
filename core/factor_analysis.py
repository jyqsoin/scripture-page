# -*- coding: utf-8 -*-
"""
因子有效性检验（V1.3 Step A2）。

基于 core.factor_engine 的因子框架，提供：
- IC/IR 时序：每个调仓日的因子值与未来 N 日收益的截面相关系数，IR = mean(IC) / std(IC)
- 分层回测：每个调仓日按因子值分 N 层，统计各层未来收益均值，检验单调性
- 因子相关性矩阵：多因子在截面上的相关系数均值

防前视约束：
- 因子值使用调仓日 t 及之前行情计算；
- 未来收益使用 t 日收盘到 t+N 日收盘的收益，严格滞后；
- 调仓日默认取每周最后一个交易日，避免日频噪声。

仅研究用途，不构成投资建议。
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.factor_engine import (
    FACTOR_REGISTRY,
    compute_cross_section,
    list_factors,
)

DISCLAIMER = "仅研究用途，不构成投资建议"


# ---------------------------------------------------------------------------
# 批量数据加载
# ---------------------------------------------------------------------------

def load_all_daily(codes, fq="qfq"):
    """
    批量加载多只股票日线到内存，返回 dict[code] -> DataFrame（按日期升序）。
    用于因子检验时反复按日期切片，避免逐只查询 SQLite。
    """
    # 一次性从 SQLite 读出全部所需数据
    placeholders = ",".join("?" * len(codes))
    rows = db.query_all(
        "SELECT code, date, open, high, low, close, volume, amount "
        "FROM daily_bars WHERE fq = ? AND code IN (%s) ORDER BY code, date" % placeholders,
        (fq, *codes),
    )
    if not rows:
        return {}
    df_all = pd.DataFrame(rows)
    result = {}
    for code, group in df_all.groupby("code"):
        result[code] = group.sort_values("date").reset_index(drop=True)
    return result


def get_rebalance_dates(codes, freq="W", fq="qfq", limit=None):
    """
    生成调仓日序列。

    freq: "W" 周频（每周最后一个交易日），"M" 月频（每月最后一个交易日）
    limit: 最多返回多少个调仓日（从最早开始），None 表示全部
    返回调仓日字符串列表（升序）。
    """
    # 取所有股票的交易日并集，按频率取最后一个
    placeholders = ",".join("?" * len(codes))
    rows = db.query_all(
        "SELECT DISTINCT date FROM daily_bars WHERE fq = ? AND code IN (%s) "
        "ORDER BY date" % placeholders,
        (fq, *codes),
    )
    dates = pd.to_datetime([r["date"] for r in rows])
    if len(dates) == 0:
        return []
    s = pd.Series(dates, index=dates)
    if freq == "W":
        # 每周最后一个交易日
        rebal = s.resample("W").last().dropna()
    elif freq == "M":
        rebal = s.resample("M").last().dropna()
    else:
        raise ValueError("freq 只能是 'W' 或 'M'")
    result = [d.strftime("%Y-%m-%d") for d in rebal]
    if limit and len(result) > limit:
        result = result[-limit:]
    return result


# ---------------------------------------------------------------------------
# 未来收益计算
# ---------------------------------------------------------------------------

def compute_forward_returns(data_map, trade_date, n=5):
    """
    计算每只股票在 trade_date 收盘后的 N 日收益。

    收益 = close[t+n] / close[t] - 1
    若数据不足 n 日则返回 NaN。
    返回 Series，index=code。
    """
    result = {}
    for code, df in data_map.items():
        dates = df["date"].values
        if trade_date not in dates:
            result[code] = np.nan
            continue
        idx = int(np.where(dates == trade_date)[0][0])
        if idx + n >= len(df):
            result[code] = np.nan
            continue
        close_t = float(df.iloc[idx]["close"])
        close_fwd = float(df.iloc[idx + n]["close"])
        if close_t == 0:
            result[code] = np.nan
        else:
            result[code] = close_fwd / close_t - 1
    return pd.Series(result)


# ---------------------------------------------------------------------------
# IC / IR 时序
# ---------------------------------------------------------------------------

def compute_ic_series(data_map, rebalance_dates, factor_name, forward_n=5,
                      method="spearman", fq="qfq"):
    """
    计算单个因子的 IC 时序。

    每个调仓日：
        factor_t = compute_cross_section 在 t 日的因子值截面
        return_t = t 日收盘到 t+forward_n 日收盘的收益
        IC_t = 截面相关系数(factor_t, return_t)

    method: "spearman"（秩相关，默认）或 "pearson"（线性相关）
    返回 DataFrame: index=调仓日, columns=[ic]
    """
    if factor_name not in FACTOR_REGISTRY:
        raise ValueError("未注册的因子：%s" % factor_name)
    ic_list = []
    for trade_date in rebalance_dates:
        # 因子值：从内存数据计算单股因子
        factor_values = {}
        for code, df in data_map.items():
            df_t = df[df["date"] <= trade_date]
            if df_t.empty:
                factor_values[code] = np.nan
            else:
                from core.factor_engine import compute_factor
                factor_values[code] = compute_factor(df_t, factor_name)
        factor_series = pd.Series(factor_values)
        # 未来收益
        return_series = compute_forward_returns(data_map, trade_date, n=forward_n)
        # 对齐并去掉 NaN
        aligned = pd.concat([factor_series, return_series], axis=1).dropna()
        aligned.columns = ["factor", "return"]
        if len(aligned) < 10:
            ic_list.append(np.nan)
            continue
        if method == "spearman":
            ic = aligned["factor"].corr(aligned["return"], method="spearman")
        else:
            ic = aligned["factor"].corr(aligned["return"], method="pearson")
        ic_list.append(ic)
    return pd.DataFrame({"ic": ic_list}, index=rebalance_dates)


def summarize_ic(ic_frame):
    """
    汇总 IC 时序统计。

    返回 dict:
        ic_mean: IC 均值
        ic_std: IC 标准差
        ir: 信息比率 = ic_mean / ic_std
        ic_positive_rate: IC 为正的比例
        t_stat: IC 均值的 t 统计量 = ic_mean / (ic_std / sqrt(n))
        count: 有效样本数
    """
    ic = ic_frame["ic"].dropna()
    if ic.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ir": np.nan,
                "ic_positive_rate": np.nan, "t_stat": np.nan, "count": 0}
    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=1))
    ir = ic_mean / ic_std if ic_std > 0 else np.nan
    positive_rate = float((ic > 0).mean())
    t_stat = ic_mean / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else np.nan
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ir,
        "ic_positive_rate": positive_rate,
        "t_stat": t_stat,
        "count": int(len(ic)),
    }


# ---------------------------------------------------------------------------
# 分层回测
# ---------------------------------------------------------------------------

def compute_layer_returns(data_map, rebalance_dates, factor_name, n_layers=5,
                          forward_n=5, fq="qfq"):
    """
    分层回测：每个调仓日按因子值将股票分为 n_layers 层，
    统计各层未来收益的均值，检验因子单调性。

    第 1 层为因子值最小的组，第 n_layers 层为最大的组。
    返回 DataFrame: index=调仓日, columns=层号(1..n_layers)，值为该层平均未来收益
    """
    if factor_name not in FACTOR_REGISTRY:
        raise ValueError("未注册的因子：%s" % factor_name)
    records = []
    for trade_date in rebalance_dates:
        factor_values = {}
        for code, df in data_map.items():
            df_t = df[df["date"] <= trade_date]
            if df_t.empty:
                factor_values[code] = np.nan
            else:
                from core.factor_engine import compute_factor
                factor_values[code] = compute_factor(df_t, factor_name)
        factor_series = pd.Series(factor_values).dropna()
        return_series = compute_forward_returns(data_map, trade_date, n=forward_n)
        aligned = pd.concat([factor_series, return_series], axis=1).dropna()
        aligned.columns = ["factor", "return"]
        if len(aligned) < n_layers * 2:
            continue
        # 按因子值排序分层
        aligned = aligned.sort_values("factor")
        aligned["layer"] = pd.qcut(aligned["factor"], q=n_layers, labels=False, duplicates="drop") + 1
        layer_mean = aligned.groupby("layer")["return"].mean()
        record = {int(layer): float(layer_mean.get(layer, np.nan)) for layer in range(1, n_layers + 1)}
        record["date"] = trade_date
        records.append(record)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("date")
    return df.reindex(columns=list(range(1, n_layers + 1)))


def summarize_layers(layer_df):
    """
    汇总分层收益：返回各层平均收益 + 多空收益（最高层 - 最低层）。
    """
    if layer_df.empty:
        return {}
    layer_means = layer_df.mean()
    top_layer = int(layer_df.columns[-1])
    bottom_layer = int(layer_df.columns[0])
    long_short = float(layer_means[top_layer] - layer_means[bottom_layer])
    result = {"layer_means": {int(k): float(v) for k, v in layer_means.items()},
              "long_short": long_short,
              "top": float(layer_means[top_layer]),
              "bottom": float(layer_means[bottom_layer])}
    # 单调性检验：层收益的 Spearman 相关系数（层号 vs 层均收益）
    layer_indices = list(layer_means.index)
    layer_vals = list(layer_means.values)
    if len(layer_indices) >= 2:
        result["monotonicity"] = float(pd.Series(layer_indices).corr(pd.Series(layer_vals), method="spearman"))
    else:
        result["monotonicity"] = np.nan
    return result


# ---------------------------------------------------------------------------
# 因子相关性矩阵
# ---------------------------------------------------------------------------

def compute_factor_correlation(data_map, rebalance_dates, factor_names, fq="qfq"):
    """
    计算多因子在截面上的相关系数均值。

    对每个调仓日计算因子值矩阵的相关系数，最后取所有调仓日的均值。
    返回 DataFrame: index=因子名, columns=因子名
    """
    if not factor_names:
        return pd.DataFrame()
    corr_sum = None
    count = 0
    for trade_date in rebalance_dates:
        factor_values = {}
        for name in factor_names:
            values = {}
            for code, df in data_map.items():
                df_t = df[df["date"] <= trade_date]
                if df_t.empty:
                    values[code] = np.nan
                else:
                    from core.factor_engine import compute_factor
                    values[code] = compute_factor(df_t, name)
            factor_values[name] = pd.Series(values)
        frame = pd.DataFrame(factor_values).dropna()
        if len(frame) < 10:
            continue
        corr = frame.corr(method="spearman")
        if corr_sum is None:
            corr_sum = corr
        else:
            corr_sum = corr_sum.add(corr, fill_value=0)
        count += 1
    if corr_sum is None or count == 0:
        return pd.DataFrame(index=factor_names, columns=factor_names)
    return corr_sum / count


# ---------------------------------------------------------------------------
# 综合因子检验入口
# ---------------------------------------------------------------------------

def run_factor_check(factor_names, codes=None, freq="W", forward_n=5, n_layers=5,
                     limit=100, fq="qfq"):
    """
    运行完整的因子有效性检验。

    参数：
        factor_names: 要检验的因子名列表
        codes: 股票池，默认取本地有 qfq 数据的全部股票
        freq: 调仓频率 "W"/"M"
        forward_n: 未来收益天数
        n_layers: 分层数
        limit: 最多使用多少个调仓日（从最近往前取）
        fq: 复权口径

    返回 dict:
        ic_summary: {factor_name: summarize_ic 结果}
        ic_series: {factor_name: IC 时序 DataFrame}
        layer_summary: {factor_name: summarize_layers 结果}
        layer_returns: {factor_name: 分层收益 DataFrame}
        factor_correlation: 因子相关性矩阵 DataFrame
        rebalance_dates: 实际使用的调仓日列表
    """
    if codes is None:
        rows = db.query_all("SELECT DISTINCT code FROM daily_bars WHERE fq=? ORDER BY code", (fq,))
        codes = [r["code"] for r in rows]
    print("加载 %d 只股票日线..." % len(codes), flush=True)
    data_map = load_all_daily(codes, fq=fq)
    codes = list(data_map.keys())
    print("实际加载 %d 只" % len(codes), flush=True)

    rebalance_dates = get_rebalance_dates(codes, freq=freq, fq=fq, limit=limit)
    print("调仓日数: %d（%s ~ %s）" % (len(rebalance_dates), rebalance_dates[0], rebalance_dates[-1]), flush=True)

    ic_summary = {}
    ic_series = {}
    layer_summary = {}
    layer_returns = {}

    for i, name in enumerate(factor_names, 1):
        print("  检验因子 [%d/%d]: %s" % (i, len(factor_names), name), flush=True)
        ic_df = compute_ic_series(data_map, rebalance_dates, name, forward_n=forward_n, fq=fq)
        ic_series[name] = ic_df
        ic_summary[name] = summarize_ic(ic_df)
        layer_df = compute_layer_returns(data_map, rebalance_dates, name, n_layers=n_layers,
                                          forward_n=forward_n, fq=fq)
        layer_returns[name] = layer_df
        layer_summary[name] = summarize_layers(layer_df)

    print("计算因子相关性矩阵...", flush=True)
    corr = compute_factor_correlation(data_map, rebalance_dates, factor_names, fq=fq)

    return {
        "ic_summary": ic_summary,
        "ic_series": ic_series,
        "layer_summary": layer_summary,
        "layer_returns": layer_returns,
        "factor_correlation": corr,
        "rebalance_dates": rebalance_dates,
    }


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()

    # 取本地有数据的前 50 只股票做测试
    rows = db.query_all("SELECT DISTINCT code FROM daily_bars WHERE fq='qfq' ORDER BY code LIMIT 50")
    codes = [r["code"] for r in rows]
    print("测试股票数:", len(codes))

    factors_to_check = ["return_5", "volatility_20", "rsi14", "volume_ratio_5"]
    result = run_factor_check(
        factor_names=factors_to_check, codes=codes, freq="W", forward_n=5,
        n_layers=5, limit=52, fq="qfq",
    )

    print("\n=== IC / IR 汇总 ===")
    print("%-16s %8s %8s %8s %8s %8s" % ("因子", "IC均值", "IC标准差", "IR", "IC胜率", "t值"))
    for name in factors_to_check:
        s = result["ic_summary"][name]
        print("%-16s %8.4f %8.4f %8.4f %8.2f %8.2f" % (
            name, s["ic_mean"], s["ic_std"], s["ir"], s["ic_positive_rate"], s["t_stat"],
        ))

    print("\n=== 分层收益汇总（5层，未来5日）===")
    print("%-16s %8s %8s %8s %8s %8s %8s %10s" % (
        "因子", "L1(低)", "L2", "L3", "L4", "L5(高)", "多空", "单调性"))
    for name in factors_to_check:
        s = result["layer_summary"][name]
        lm = s.get("layer_means", {})
        print("%-16s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %10.4f" % (
            name,
            lm.get(1, 0), lm.get(2, 0), lm.get(3, 0), lm.get(4, 0), lm.get(5, 0),
            s.get("long_short", 0), s.get("monotonicity", 0),
        ))

    print("\n=== 因子相关性矩阵（Spearman）===")
    corr = result["factor_correlation"]
    print(corr.round(3).to_string())

    # 基本断言
    for name in factors_to_check:
        assert result["ic_summary"][name]["count"] > 0, "%s 无有效 IC 样本" % name
    assert not result["factor_correlation"].empty
    print("\nfactor_analysis.py 自测通过；%s" % DISCLAIMER)
