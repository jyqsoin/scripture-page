# -*- coding: utf-8 -*-
"""
因子计算框架（V1.3 Step A1）。

把 core.factors 中的技术指标统一注册为「因子」，支持：
- 单只股票因子计算（取指定交易日及之前的行情，不使用未来数据）；
- 横截面批量计算（多只股票同一日期的因子矩阵）；
- 横截面标准化：MAD 去极值 + Z-Score，可选行业中性；
- 截面原始值 parquet 缓存，避免重复计算。

约束：
- 只使用本地日线行情，不联网；
- 因子值只依赖 trade_date 当日及历史数据，无未来函数；
- 行业中性仅在 stock_info.industry 有值时生效，否则跳过。

仅研究用途，不构成投资建议。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.data_loader import load_local_daily
from core import factors as F

DISCLAIMER = "仅研究用途，不构成投资建议"

# 因子缓存目录：存截面原始因子值（未标准化），按 trade_date + fq 分文件。
FACTOR_DIR = db.DATA_DIR / "factors"


# ---------------------------------------------------------------------------
# 因子注册表
# ---------------------------------------------------------------------------

def _series_ma(df, n):
    """N 日均价时序。"""
    return F.ma(df["close"], n)


def _series_return(df, n):
    """N 日收益率时序。"""
    return df["close"].pct_change(n)


def _series_volatility(df, n):
    """N 日收益率标准差时序。"""
    return df["close"].pct_change().rolling(n, min_periods=n).std()


def _series_macd_dif(df):
    """MACD DIF 时序。"""
    return F.macd(df["close"])["dif"]


def _series_macd_hist(df):
    """MACD 柱（两倍）时序。"""
    return F.macd(df["close"])["macd"]


def _series_atr(df, n=14):
    """ATR 时序。"""
    return F.atr(df, n)


def _series_atr_pct(df, n=14):
    """ATR 占价格比例时序，消除量纲。"""
    return F.atr(df, n) / df["close"].replace(0, np.nan)


def _series_turnover(df, n=5):
    """N 日均量相对 20 日均量的比例时序（换手率近似）。
    真实换手率需要流通股本，本地档案无该字段，用成交量的相对位置代替。"""
    vol = df["volume"]
    base = vol.rolling(20, min_periods=20).mean()
    return vol.rolling(n, min_periods=n).mean() / base.replace(0, np.nan)


def _series_bias(df, n):
    """乖离率时序：close / MA(n) - 1，消除绝对价格量纲。"""
    return df["close"] / F.ma(df["close"], n).replace(0, np.nan) - 1


def _series_ret_risk(df, n=20):
    """风险调整动量时序：N 日收益 / N 日收益波动。"""
    return df["close"].pct_change(n) / _series_volatility(df, n).replace(0, np.nan)


def _series_price_position(df, n):
    """价格在 N 日高低区间中的相对位置时序：0 为区间最低，1 为区间最高。"""
    low = F.llv(df["low"], n)
    high = F.hhv(df["high"], n)
    return (df["close"] - low) / (high - low).replace(0, np.nan)


def _series_volume_trend(df, fast=5, slow=60):
    """量能趋势时序：短均量 / 长均量 - 1。"""
    vol = df["volume"]
    return F.ma(vol, fast) / F.ma(vol, slow).replace(0, np.nan) - 1


def _series_volatility_ratio(df, fast=5, slow=20):
    """波动变化时序：短周期波动 / 长周期波动 - 1。"""
    ret = df["close"].pct_change()
    short = ret.rolling(fast, min_periods=fast).std()
    long = ret.rolling(slow, min_periods=slow).std()
    return short / long.replace(0, np.nan) - 1


def _series_close_to_high(df, n=60):
    """距离 N 日最高收盘价的相对距离时序（<=0）。"""
    return df["close"] / F.hhv(df["close"], n).replace(0, np.nan) - 1


# 注册表：因子名 -> {desc, group, series, needs}
# series 接收按日期升序的日线 DataFrame，返回与 df 同索引的因子值时序；
# 因子值只依赖当日及历史数据，无未来函数。
FACTOR_REGISTRY = {
    # 均线水平类（绝对价格量纲，V1.4 筛查中用于检验是否仅为价格水平暴露）
    "ma5": {"desc": "5日均价", "group": "均线水平", "series": lambda df: _series_ma(df, 5), "needs": ["close"]},
    "ma20": {"desc": "20日均价", "group": "均线水平", "series": lambda df: _series_ma(df, 20), "needs": ["close"]},
    "ma60": {"desc": "60日均价", "group": "均线水平", "series": lambda df: _series_ma(df, 60), "needs": ["close"]},
    # 动量类
    "return_5": {"desc": "5日收益率", "group": "动量", "series": lambda df: _series_return(df, 5), "needs": ["close"]},
    "return_10": {"desc": "10日收益率", "group": "动量", "series": lambda df: _series_return(df, 10), "needs": ["close"]},
    "return_20": {"desc": "20日收益率", "group": "动量", "series": lambda df: _series_return(df, 20), "needs": ["close"]},
    # 波动类
    "volatility_20": {"desc": "20日收益率标准差", "group": "波动", "series": lambda df: _series_volatility(df, 20), "needs": ["close"]},
    # 量能类
    "volume_ratio_5": {"desc": "5日量比", "group": "量能", "series": lambda df: F.volume_ratio(df["volume"], 5), "needs": ["volume"]},
    "turnover_5": {"desc": "5日相对量能", "group": "量能", "series": lambda df: _series_turnover(df, 5), "needs": ["volume"]},
    # 超买超卖
    "rsi14": {"desc": "RSI(14)", "group": "超买超卖", "series": lambda df: F.rsi(df["close"], 14), "needs": ["close"]},
    # 趋势类
    "macd_dif": {"desc": "MACD DIF", "group": "趋势", "series": _series_macd_dif, "needs": ["close"]},
    "macd_hist": {"desc": "MACD柱", "group": "趋势", "series": _series_macd_hist, "needs": ["close"]},
    # 振幅类
    "atr14": {"desc": "ATR(14)", "group": "振幅", "series": lambda df: _series_atr(df, 14), "needs": ["high", "low", "close"]},
    "atr_pct": {"desc": "ATR占比(14)", "group": "振幅", "series": lambda df: _series_atr_pct(df, 14), "needs": ["high", "low", "close"]},
    # ---- V1.4 扩充：无量纲 / 风险调整类 ----
    "bias_5": {"desc": "5日乖离率", "group": "乖离", "series": lambda df: _series_bias(df, 5), "needs": ["close"]},
    "bias_20": {"desc": "20日乖离率", "group": "乖离", "series": lambda df: _series_bias(df, 20), "needs": ["close"]},
    "bias_60": {"desc": "60日乖离率", "group": "乖离", "series": lambda df: _series_bias(df, 60), "needs": ["close"]},
    "ret_risk_20": {"desc": "20日风险调整动量", "group": "风险调整", "series": lambda df: _series_ret_risk(df, 20), "needs": ["close"]},
    "price_position_20": {"desc": "20日价格位置", "group": "位置", "series": lambda df: _series_price_position(df, 20), "needs": ["high", "low", "close"]},
    "price_position_60": {"desc": "60日价格位置", "group": "位置", "series": lambda df: _series_price_position(df, 60), "needs": ["high", "low", "close"]},
    "close_to_high_60": {"desc": "距60日最高收盘价", "group": "位置", "series": lambda df: _series_close_to_high(df, 60), "needs": ["close"]},
    # ---- V1.4 扩充：量能 / 波动结构 / 短反转 ----
    "volume_trend_20": {"desc": "量能趋势(5/60)", "group": "量能", "series": _series_volume_trend, "needs": ["volume"]},
    "volatility_ratio_5_20": {"desc": "波动变化(5/20)", "group": "波动", "series": _series_volatility_ratio, "needs": ["close"]},
    "ret_1": {"desc": "1日收益率", "group": "反转", "series": lambda df: _series_return(df, 1), "needs": ["close"]},
}


def list_factors():
    """返回已注册因子列表：[{name, desc, group}]。"""
    return [{"name": k, "desc": v["desc"], "group": v.get("group", "")}
            for k, v in FACTOR_REGISTRY.items()]


def compute_factor_series(df, factor_name):
    """
    计算单只股票的因子值时序（与 df 同索引）。

    df 必须按日期升序；返回的序列只依赖当日及历史数据，无未来函数。
    """
    spec = FACTOR_REGISTRY.get(factor_name)
    if spec is None:
        raise ValueError("未注册的因子：%s" % factor_name)
    series = spec["series"](df)
    return pd.to_numeric(pd.Series(series, index=df.index), errors="coerce")


# ---------------------------------------------------------------------------
# 单只股票因子计算
# ---------------------------------------------------------------------------

def compute_factor(df, factor_name):
    """
    计算单只股票在 df 最后一日的因子值。

    df 必须按日期升序；只使用历史数据，无未来函数。
    返回 float，无法计算时返回 NaN。
    """
    spec = FACTOR_REGISTRY.get(factor_name)
    if spec is None:
        raise ValueError("未注册的因子：%s" % factor_name)
    try:
        series = compute_factor_series(df, factor_name)
        if series is None or series.empty:
            return np.nan
        value = series.iloc[-1]
    except Exception:
        return np.nan
    if value is None or pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def compute_factors_for_code(code, trade_date, factor_names=None, fq="qfq"):
    """
    加载 code 的本地日线（截取到 trade_date），计算多个因子的最新值。

    返回 dict: {factor_name: value}。
    """
    factor_names = factor_names or list(FACTOR_REGISTRY.keys())
    df = load_local_daily(code, fq=fq)
    if df is None or df.empty:
        return {name: np.nan for name in factor_names}
    # 只保留 trade_date 及之前的数据，防止未来函数
    df = df[df["date"] <= trade_date].copy()
    if df.empty:
        return {name: np.nan for name in factor_names}
    result = {}
    for name in factor_names:
        result[name] = compute_factor(df, name)
    return result


# ---------------------------------------------------------------------------
# 横截面标准化
# ---------------------------------------------------------------------------

def winsorize_mad(series, n=3.0):
    """
    MAD 去极值：用中位数 ± n * 1.4826 * MAD 截断。
    1.4826 是正态分布下 MAD 与标准差的换算系数。
    """
    series = pd.to_numeric(series, errors="coerce")
    median = series.median()
    mad = (series - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return series
    lower = median - n * 1.4826 * mad
    upper = median + n * 1.4826 * mad
    return series.clip(lower=lower, upper=upper)


def zscore(series):
    """Z-Score 标准化：(x - mean) / std。"""
    series = pd.to_numeric(series, errors="coerce")
    mean = series.mean()
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - mean) / std


def neutralize_industry(factor_series, industries):
    """
    行业中性：对每个因子值减去其所属行业的均值。
    industries 是与 factor_series 等长、同索引的 Series。
    行业为空字符串或 NaN 的股票，用全市场均值中性化。
    """
    factor_series = pd.to_numeric(factor_series, errors="coerce")
    industries = pd.Series(industries, index=factor_series.index).fillna("")
    industry_mean = factor_series.groupby(industries).transform("mean")
    # 行业内只有一个值或为空时，transform 会返回 NaN，用全市场均值兜底
    fallback = factor_series.mean()
    industry_mean = industry_mean.fillna(fallback)
    return factor_series - industry_mean


def standardize_cross_section(frame, winsorize=True, z_score=True, industry_neutral=False,
                               industries=None):
    """
    对横截面因子矩阵逐列做标准化。

    frame: DataFrame，index=code, columns=factor_name
    industries: 可选，index=code 的行业 Series
    返回标准化后的 DataFrame（不修改原 frame）。
    """
    result = frame.copy()
    for col in result.columns:
        series = result[col]
        if industry_neutral and industries is not None:
            series = neutralize_industry(series, industries)
        if winsorize:
            series = winsorize_mad(series)
        if z_score:
            series = zscore(series)
        result[col] = series
    return result


# ---------------------------------------------------------------------------
# 横截面批量计算 + 缓存
# ---------------------------------------------------------------------------

def _cache_path(trade_date, fq):
    FACTOR_DIR.mkdir(parents=True, exist_ok=True)
    return FACTOR_DIR / ("cross_%s_%s.parquet" % (trade_date, fq))


def compute_cross_section(codes, trade_date, factor_names=None, fq="qfq",
                          standardize=True, winsorize=True, industry_neutral=False,
                          use_cache=True):
    """
    计算多只股票在 trade_date 的横截面因子矩阵。

    参数：
        codes: 股票代码列表
        trade_date: 交易日（YYYY-MM-DD），只使用该日及之前行情
        factor_names: 因子名列表，默认全部已注册因子
        fq: 复权口径
        standardize: 是否做横截面标准化（MAD去极值 + Z-Score）
        winsorize: 标准化时是否去极值
        industry_neutral: 是否做行业中性（需要 stock_info.industry）
        use_cache: 是否使用 parquet 缓存原始截面值

    返回：
        DataFrame，index=code, columns=factor_names
        若 standardize=True，返回标准化后的值；否则返回原始因子值。
    """
    factor_names = factor_names or list(FACTOR_REGISTRY.keys())
    cache_path = _cache_path(trade_date, fq)

    # 尝试读取缓存（缓存存的是全部因子的原始值）
    raw = None
    if use_cache and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            # 缓存包含所需全部列才使用
            if set(factor_names).issubset(set(cached.columns)):
                raw = cached[factor_names].copy()
                # 只保留请求的 codes
                raw = raw.reindex([c for c in codes if c in raw.index])
        except Exception:
            raw = None

    if raw is None:
        # 逐只计算
        rows = {}
        for code in codes:
            rows[code] = compute_factors_for_code(code, trade_date, factor_names, fq=fq)
        raw = pd.DataFrame.from_dict(rows, orient="index")
        raw.index.name = "code"
        # 写入缓存（存所有因子原始值，便于下次复用）
        if use_cache:
            try:
                # 若已有缓存文件，合并更新
                if cache_path.exists():
                    cached = pd.read_parquet(cache_path)
                    merged = cached.combine_first(raw)
                    merged.to_parquet(cache_path)
                else:
                    raw.to_parquet(cache_path)
            except Exception:
                pass  # 缓存失败不影响计算

    if not standardize:
        return raw

    # 行业中性需要行业数据
    industries = None
    if industry_neutral:
        try:
            stock_rows = db.query_all(
                "SELECT code, industry FROM stock_info WHERE code IN (%s)"
                % ",".join("?" * len(raw.index)),
                tuple(raw.index),
            )
            industries = pd.Series(
                {r["code"]: (r["industry"] or "") for r in stock_rows},
                index=raw.index,
            )
        except Exception:
            industries = None

    return standardize_cross_section(
        raw, winsorize=winsorize, z_score=True,
        industry_neutral=industry_neutral, industries=industries,
    )


def clear_cache(trade_date=None, fq=None):
    """清理因子缓存。指定 trade_date/fq 则只删对应文件，否则清空目录。"""
    if not FACTOR_DIR.exists():
        return
    if trade_date and fq:
        path = _cache_path(trade_date, fq)
        if path.exists():
            path.unlink()
        return
    for path in FACTOR_DIR.glob("*.parquet"):
        path.unlink()


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()

    # 取本地有数据的股票做截面测试（取前 20 只，避免过慢）
    rows = db.query_all(
        "SELECT DISTINCT code FROM daily_bars WHERE fq='qfq' ORDER BY code LIMIT 20"
    )
    codes = [r["code"] for r in rows]
    print("测试股票数:", len(codes))

    # 取最近交易日
    from core.pool import latest_local_trade_date
    trade_date = latest_local_trade_date()
    print("测试交易日:", trade_date)

    # 1. 单只股票因子计算
    factors_0 = compute_factors_for_code(codes[0], trade_date, ["ma5", "return_5", "rsi14"])
    print("\n单股 %s 因子:" % codes[0])
    for k, v in factors_0.items():
        print("  %s = %s" % (k, "%.6f" % v if pd.notna(v) else "NaN"))
    assert len(factors_0) == 3

    # 2. 横截面计算（原始值，不标准化）
    raw = compute_cross_section(codes, trade_date, standardize=False, use_cache=False)
    print("\n截面原始值 shape:", raw.shape)
    print(raw[["ma5", "return_5", "rsi14"]].head())
    assert raw.shape[0] <= len(codes)
    assert "ma5" in raw.columns

    # 3. 横截面标准化
    std = compute_cross_section(codes, trade_date, standardize=True, use_cache=False)
    print("\n标准化后 shape:", std.shape)
    print(std[["ma5", "return_5", "rsi14"]].head())
    # 标准化后每列均值应接近 0，标准差接近 1
    for col in ["return_5", "rsi14"]:
        col_mean = std[col].mean()
        col_std = std[col].std(ddof=0)
        print("  %s: mean=%.4f std=%.4f" % (col, col_mean, col_std))
        # 允许少量 NaN，非 NaN 部分应标准化
        valid = std[col].dropna()
        if len(valid) > 2:
            assert abs(valid.mean()) < 0.1, "标准化后均值偏离 0：%s" % col
            assert abs(valid.std(ddof=0) - 1.0) < 0.1, "标准化后标准差偏离 1：%s" % col

    # 4. 缓存验证
    cached = compute_cross_section(codes, trade_date, standardize=False, use_cache=True)
    print("\n缓存读取 shape:", cached.shape)
    pd.testing.assert_frame_equal(cached.sort_index(), raw.sort_index(), check_dtype=False)

    # 5. 无未来函数验证：用更早的交易日，因子值应该不同
    from datetime import datetime, timedelta
    # 取 30 天前的交易日近似
    dt = datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=30)
    earlier_date = dt.strftime("%Y-%m-%d")
    raw_early = compute_cross_section(codes, earlier_date, factor_names=["ma20"],
                                       standardize=False, use_cache=False)
    # 两个日期的 ma20 不应该完全相同（价格变化）
    overlap = raw.index.intersection(raw_early.index)
    if len(overlap) > 0:
        diff = (raw.loc[overlap, "ma20"] - raw_early.loc[overlap, "ma20"]).abs().sum()
        print("\n早 30 天 ma20 与当日差异总和:", diff)
        assert diff > 0, "不同交易日的因子值不应完全相同（无未来函数检查失败）"

    clear_cache(trade_date, "qfq")
    clear_cache(earlier_date, "qfq")
    print("\nfactor_engine.py 自测通过；%s" % DISCLAIMER)
