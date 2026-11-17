# -*- coding: utf-8 -*-
"""
本地指标与时间序列函数。

所有计算只依赖当前及历史数据，不使用任何未来行情。
主要口径尽量贴近通达信：
- RSI/KDJ 使用 SMA(X, N, 1)，即 Wilder 平滑；
- MACD 柱为 2 * (DIF - DEA)；
- 量比（日线离线简化）= 当日成交量 / 前 N 日平均成交量，分母不含当日。

仅研究用途，不构成投资建议。
"""

import numpy as np
import pandas as pd


def _as_series(value):
    """统一转成 float Series，便于指标计算。"""
    if isinstance(value, pd.Series):
        return value.astype(float)
    return pd.Series(value, dtype=float)


def _pair(a, b):
    """让两个操作数都成为 Series；标量按另一个 Series 的索引广播。"""
    if isinstance(a, pd.Series):
        a = a.astype(float)
        if not isinstance(b, pd.Series):
            b = pd.Series(float(b), index=a.index)
        return a, b.astype(float)
    if isinstance(b, pd.Series):
        return pd.Series(float(a), index=b.index), b.astype(float)
    raise TypeError("CROSS/MAX/MIN 至少需要一个参数为 Series")


def ma(values, n):
    """N 日简单移动平均；不足 N 日时返回 NaN。"""
    n = int(n)
    return _as_series(values).rolling(n, min_periods=n).mean()


def ema(values, n):
    """指数移动平均，adjust=False 对应常见交易软件递推口径。"""
    return _as_series(values).ewm(span=int(n), adjust=False).mean()


def sma_cn(values, n, m=1):
    """通达信 SMA(X,N,M)：Y=(X*M+Y'*(N-M))/N。"""
    return _as_series(values).ewm(alpha=float(m) / int(n), adjust=False).mean()


def ref(values, n=1):
    """引用 N 周期前的数据；N 必须不小于 0，禁止负周期引用未来。"""
    n = int(n)
    if n < 0:
        raise ValueError("REF 周期不能为负数，负周期会引用未来数据")
    return _as_series(values).shift(n)


def hhv(values, n):
    """N 周期最高值；N=0 表示从首个数据开始。"""
    series = _as_series(values)
    n = int(n)
    return series.expanding().max() if n == 0 else series.rolling(n, min_periods=n).max()


def llv(values, n):
    """N 周期最低值；N=0 表示从首个数据开始。"""
    series = _as_series(values)
    n = int(n)
    return series.expanding().min() if n == 0 else series.rolling(n, min_periods=n).min()


def rolling_sum(values, n):
    """N 周期求和；N=0 表示从首个数据开始累计。"""
    series = _as_series(values)
    n = int(n)
    return series.expanding().sum() if n == 0 else series.rolling(n, min_periods=n).sum()


def count(condition, n):
    """统计 N 周期内条件成立次数；N=0 表示历史累计。"""
    return rolling_sum(_as_series(condition).fillna(0), n)


def cross(a, b):
    """上穿：当前 A>B，且前一周期 A<=B。"""
    a, b = _pair(a, b)
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False)


def barslast(condition):
    """
    上一次条件成立到当前的周期数；当前成立为 0。
    历史从未成立时返回 NaN。
    """
    series = _as_series(condition).fillna(0).to_numpy()
    result = np.full(len(series), np.nan)
    last = -1
    for i, value in enumerate(series):
        if bool(value):
            last = i
        if last >= 0:
            result[i] = i - last
    return pd.Series(result, index=getattr(condition, "index", None))


def rsi(close, n=14):
    """相对强弱指标：SMA(MAX(C-LC,0),N,1)/SMA(ABS(C-LC),N,1)*100。"""
    close = _as_series(close)
    delta = close.diff()
    up = delta.clip(lower=0)
    absolute = delta.abs()
    denominator = sma_cn(absolute, n, 1)
    result = sma_cn(up, n, 1) / denominator.replace(0, np.nan) * 100
    return result.rename("rsi")


def macd(close, fast=12, slow=26, signal=9):
    """返回 DIF、DEA、MACD 三列；MACD 为通达信常见的两倍柱体。"""
    close = _as_series(close)
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = 2 * (dif - dea)
    return pd.DataFrame({"dif": dif, "dea": dea, "macd": hist})


def kdj(df, n=9, m1=3, m2=3):
    """KDJ：输入需包含 high/low/close，返回 k/d/j 三列。"""
    low_n = llv(df["low"], n)
    high_n = hhv(df["high"], n)
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = sma_cn(rsv, m1, 1)
    d = sma_cn(k, m2, 1)
    j = 3 * k - 2 * d
    return pd.DataFrame({"k": k, "d": d, "j": j})


def boll(close, n=20, k=2):
    """布林带：中轨 MA20，上下轨为中轨加减 K 倍总体标准差。"""
    close = _as_series(close)
    mid = ma(close, n)
    std = close.rolling(int(n), min_periods=int(n)).std(ddof=0)
    return pd.DataFrame({
        "mid": mid,
        "upper": mid + float(k) * std,
        "lower": mid - float(k) * std,
    })


def atr(df, n=14):
    """真实波幅均值，使用 Wilder 平滑；输入需包含 high/low/close。"""
    previous_close = ref(df["close"], 1)
    tr = pd.concat([
        _as_series(df["high"]) - _as_series(df["low"]),
        (_as_series(df["high"]) - previous_close).abs(),
        (_as_series(df["low"]) - previous_close).abs(),
    ], axis=1).max(axis=1)
    return sma_cn(tr, n, 1).rename("atr")


def cci(df, n=14):
    """顺势指标 CCI：(TP-MA(TP,N))/(0.015*MD)，MD 为 TP 与均线绝对偏差的 N 日均值。"""
    tp = (_as_series(df["high"]) + _as_series(df["low"]) + _as_series(df["close"])) / 3
    ma_tp = ma(tp, n)
    deviation = (tp - ma_tp).abs()
    md = ma(deviation, n)
    return ((tp - ma_tp) / (0.015 * md.replace(0, np.nan))).rename("cci")


def wr(df, n=14):
    """威廉指标 WR：(HHV(HIGH,N)-CLOSE)/(HHV(HIGH,N)-LLV(LOW,N))*100。"""
    high_n = hhv(df["high"], n)
    low_n = llv(df["low"], n)
    close = _as_series(df["close"])
    return ((high_n - close) / (high_n - low_n).replace(0, np.nan) * 100).rename("wr")


def dmi(df, n=14, m=6):
    """趋向指标 DMI：返回 pdi/mdi/adx 三列；使用 Wilder 平滑(SMA)。"""
    high = _as_series(df["high"])
    low = _as_series(df["low"])
    close = _as_series(df["close"])
    previous_close = ref(close, 1)
    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)
    hd = high - ref(high, 1)
    ld = ref(low, 1) - low
    dmp = pd.Series(np.where((hd > 0) & (hd > ld), hd, 0.0), index=high.index)
    dmm = pd.Series(np.where((ld > 0) & (ld > hd), ld, 0.0), index=high.index)
    tr_smooth = sma_cn(tr, n, 1)
    pdi = sma_cn(dmp, n, 1) / tr_smooth.replace(0, np.nan) * 100
    mdi = sma_cn(dmm, n, 1) / tr_smooth.replace(0, np.nan) * 100
    dx = (mdi - pdi).abs() / (mdi + pdi).replace(0, np.nan) * 100
    adx = sma_cn(dx, m, 1)
    return pd.DataFrame({"pdi": pdi, "mdi": mdi, "adx": adx})


def obv(df):
    """能量潮 OBV：累计量，上涨加成交量、下跌减成交量、平盘不变。"""
    close = _as_series(df["close"])
    volume = _as_series(df["volume"])
    direction = pd.Series(0.0, index=close.index)
    direction[close > close.shift(1)] = 1.0
    direction[close < close.shift(1)] = -1.0
    signed_volume = direction * volume
    return signed_volume.cumsum().rename("obv")


def roc(close, n=12):
    """变动率 ROC：(CLOSE-REF(CLOSE,N))/REF(CLOSE,N)*100。"""
    close = _as_series(close)
    previous = ref(close, n)
    return ((close - previous) / previous.replace(0, np.nan) * 100).rename("roc")


def mtm(close, n=12):
    """动量 MTM：CLOSE - REF(CLOSE, N)。"""
    close = _as_series(close)
    return (close - ref(close, n)).rename("mtm")


def vr(df, n=26):
    """成交量比率 VR：N 日上涨量之和 / 下跌量之和 * 100。"""
    close = _as_series(df["close"])
    volume = _as_series(df["volume"])
    up_vol = pd.Series(np.where(close > close.shift(1), volume, 0.0), index=close.index)
    down_vol = pd.Series(np.where(close < close.shift(1), volume, 0.0), index=close.index)
    av = rolling_sum(up_vol, n)
    bv = rolling_sum(down_vol, n)
    return (av / bv.replace(0, np.nan) * 100).rename("vr")


def volume_ratio(volume, n=5):
    """日线简化量比：当日成交量 / 前 N 日平均成交量（不含当日）。"""
    volume = _as_series(volume)
    previous_average = ma(volume, n).shift(1)
    return (volume / previous_average.replace(0, np.nan)).rename("volume_ratio")


def add_basic_factors(df):
    """
    为日线 DataFrame 添加 MA5、MA20、RSI14、量比四列。

    输入至少需要 date/close/volume；函数会按日期升序排列并返回副本。
    """
    required = {"date", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("缺少行情列：%s" % sorted(missing))
    result = df.copy().sort_values("date").reset_index(drop=True)
    result["ma5"] = ma(result["close"], 5)
    result["ma20"] = ma(result["close"], 20)
    result["rsi14"] = rsi(result["close"], 14)
    result["volume_ratio"] = volume_ratio(result["volume"], 5)
    return result


if __name__ == "__main__":
    # 使用合成行情自测，不联网，也不依赖 SQLite。
    print("仅研究用途，不构成投资建议")
    dates = pd.bdate_range("2024-01-01", periods=120)
    close = pd.Series(np.linspace(10, 18, 120) + np.sin(np.arange(120) / 5))
    demo = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.linspace(100000, 300000, 120),
    })
    enriched = add_basic_factors(demo)
    assert abs(enriched.loc[4, "ma5"] - close.iloc[:5].mean()) < 1e-9
    assert len(rsi(close, 14)) == len(close)
    assert list(macd(close).columns) == ["dif", "dea", "macd"]
    assert list(kdj(demo).columns) == ["k", "d", "j"]
    assert list(boll(close).columns) == ["mid", "upper", "lower"]
    assert len(atr(demo)) == len(demo)
    # ---- 新增指标自测 ----
    assert len(cci(demo, 14)) == len(demo)
    assert len(wr(demo, 14)) == len(demo)
    assert list(dmi(demo).columns) == ["pdi", "mdi", "adx"]
    assert len(obv(demo)) == len(demo)
    assert len(roc(close, 12)) == len(demo)
    assert len(mtm(close, 12)) == len(demo)
    assert len(vr(demo, 26)) == len(demo)
    # CCI/WR/ROC/MTM 在有足够数据后应为有限值
    assert np.isfinite(cci(demo, 14).iloc[-1])
    assert np.isfinite(wr(demo, 14).iloc[-1])
    assert np.isfinite(roc(close, 12).iloc[-1])
    assert np.isfinite(mtm(close, 12).iloc[-1])
    print(enriched[["date", "close", "ma5", "ma20", "rsi14", "volume_ratio"]].tail())
    print("factors.py 自测通过（含 CCI/WR/DMI/OBV/ROC/MTM/VR）")
