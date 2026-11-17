# -*- coding: utf-8 -*-
"""多股并行数据加载、可选 20/50 因子和严格历史滚动标准化。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 242

BASE_FEATURE_COLUMNS = [
    "momentum_1", "momentum_5", "momentum_10", "momentum_20", "momentum_60",
    "volume_ratio_5", "volume_ratio_20", "volume_ratio_60",
    "atr_14", "volatility_20", "amplitude", "volume_std_20",
    "ma_diff_5_20", "ma_diff_20_60", "price_position_52w", "bollinger_position",
    "obv", "volume_corr_20", "vwap_deviation", "dist_high_60",
]
EXTRA_FEATURE_COLUMNS = [
    "momentum_3", "momentum_reversal_12_1", "rs_20",
    "volatility_5", "volatility_10", "volatility_60", "downside_volatility_20",
    "garman_klass_20", "turnover_proxy_20", "ma10_price", "ma60_price",
    "ma_bull_alignment", "ma_bear_alignment", "ma_entanglement",
    "atr_channel_position", "big_order_proxy", "limit_up_count_20", "limit_down_count_20",
    "market_advance_ratio", "market_limit_up_count", "market_limit_down_count",
    "market_cap_quantile_proxy", "value_score_proxy", "growth_score_proxy",
    "dist_low_60", "gap", "intraday_return", "rsi_14", "macd_hist", "return_skew_20",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS
assert len(BASE_FEATURE_COLUMNS) == 20 and len(FEATURE_COLUMNS) == 50


def get_feature_columns(feature_count):
    if int(feature_count) == 20:
        return BASE_FEATURE_COLUMNS
    if int(feature_count) == 50:
        return FEATURE_COLUMNS
    raise ValueError("feature_count 仅支持 20 或 50")


@dataclass
class MarketView:
    code: str
    dates: np.ndarray
    features: np.ndarray
    close: np.ndarray
    ma20: np.ndarray
    momentum20: np.ndarray
    volatility60: np.ndarray
    volatility_median60: np.ndarray

    def __len__(self):
        return len(self.close)


def _market(code):
    if code.startswith("6"):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    raise ValueError("暂不支持该 A 股代码：%s" % code)


def _clean(raw, code):
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["code"] = code
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "open", "high", "low", "close"]).drop_duplicates(
        "date", keep="last"
    ).sort_values("date").reset_index(drop=True)


def _fetch_tencent(code, start_date, end_date, adjust):
    import akshare as ak
    raw = ak.stock_zh_a_hist_tx(
        symbol=_market(code) + code, start_date=start_date, end_date=end_date, adjust=adjust
    )
    return _clean(raw, code)


def _fetch_eastmoney(code, start_date, end_date, adjust):
    import akshare as ak
    func = getattr(ak, "stock_zh_a_hist_em", None) or getattr(ak, "stock_zh_a_hist", None)
    raw = func(
        symbol=code, period="daily", start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""), adjust="" if adjust == "none" else adjust,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = raw.rename(columns={
        "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount",
    })
    frame = _clean(raw, code)
    if not frame.empty:
        frame["volume"] *= 100
    return frame


FETCHERS = {"tencent": _fetch_tencent, "eastmoney": _fetch_eastmoney}


def fetch_daily(code, config):
    errors = []
    for provider in config.get("providers", ["tencent", "eastmoney"]):
        if provider not in FETCHERS:
            continue
        try:
            frame = FETCHERS[provider](
                code, config.get("start_date", "20150101"),
                config.get("end_date", datetime.now().strftime("%Y%m%d")),
                config.get("adjust", "hfq"),
            )
            if not frame.empty:
                return frame, "akshare/" + provider
        except Exception as exc:
            errors.append("%s: %s" % (provider, exc))
    raise RuntimeError("%s 数据获取失败：%s" % (code, "；".join(errors) or "无可用数据源"))


def _base_factors(frame, code):
    d = frame.copy()
    close, open_, high, low = d["close"], d["open"], d["high"], d["low"]
    volume = d["volume"].replace(0, np.nan)
    ret = close.pct_change().fillna(0.0)
    prev_close = close.shift(1)
    d["_ret1"] = ret

    for period in [1, 3, 5, 10, 20, 60]:
        d["momentum_%d" % period] = close.pct_change(period)
        d["volatility_%d" % period] = ret.rolling(period).std() * np.sqrt(TRADING_DAYS)
    d["momentum_reversal_12_1"] = close.shift(20) / close.shift(250) - 1
    downside = ret.clip(upper=0)
    d["downside_volatility_20"] = np.sqrt((downside ** 2).rolling(20).mean() * TRADING_DAYS)

    true_range = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    d["atr_14"] = atr / close
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    d["garman_klass_20"] = np.sqrt(gk.clip(lower=0).rolling(20).mean() * TRADING_DAYS)

    for period in [5, 20, 60]:
        d["volume_ratio_%d" % period] = volume / volume.rolling(period).mean()
    d["turnover_proxy_20"] = volume / volume.rolling(20).sum()
    signed_volume = np.sign(ret) * volume.fillna(0)
    d["obv"] = signed_volume.cumsum() / volume.rolling(20).mean()
    rolling_vwap = d["amount"] / volume
    typical = (high + low + close) / 3
    rolling_vwap = rolling_vwap.where(np.isfinite(rolling_vwap), typical).rolling(20).mean()
    d["vwap_deviation"] = close / rolling_vwap - 1
    d["volume_corr_20"] = ret.rolling(20).corr(volume.pct_change())

    moving_averages = {period: close.rolling(period).mean() for period in [5, 10, 20, 60]}
    d["ma_diff_5_20"] = moving_averages[5] / moving_averages[20] - 1
    d["ma_diff_20_60"] = moving_averages[20] / moving_averages[60] - 1
    for period, ma in moving_averages.items():
        d["ma%d_price" % period] = close / ma - 1
    d["ma_bull_alignment"] = (
        (moving_averages[5] > moving_averages[10])
        & (moving_averages[10] > moving_averages[20])
        & (moving_averages[20] > moving_averages[60])
    ).astype(float)
    d["ma_bear_alignment"] = (
        (moving_averages[5] < moving_averages[10])
        & (moving_averages[10] < moving_averages[20])
        & (moving_averages[20] < moving_averages[60])
    ).astype(float)
    ma_frame = pd.DataFrame(moving_averages)
    ma_spread = ma_frame.max(axis=1) / ma_frame.min(axis=1) - 1
    d["ma_entanglement"] = 1 / (1 + ma_spread * 100)

    high_252, low_252 = high.rolling(252).max(), low.rolling(252).min()
    d["price_position_52w"] = (close - low_252) / (high_252 - low_252)
    middle = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    d["bollinger_position"] = (close - (middle - 2 * std20)) / (4 * std20)
    d["atr_channel_position"] = (close - (moving_averages[20] - 2 * atr)) / (4 * atr)

    close_location = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    d["big_order_proxy"] = (close_location * d["volume_ratio_20"]).rolling(5).mean()
    is_growth_reform = code.startswith("30") & (d["date"] >= pd.Timestamp("2020-08-24"))
    limit_threshold = np.where(is_growth_reform, 0.195, 0.095)
    d["_limit_up"] = (ret >= limit_threshold).astype(float)
    d["_limit_down"] = (ret <= -limit_threshold).astype(float)
    d["limit_up_count_20"] = d["_limit_up"].rolling(20).sum()
    d["limit_down_count_20"] = d["_limit_down"].rolling(20).sum()

    d["_size_proxy"] = np.log((close * volume.rolling(60).median()).clip(lower=1))
    d["value_score_proxy"] = close.rolling(252).median() / close - 1
    d["growth_score_proxy"] = d["momentum_60"] + d["momentum_reversal_12_1"]
    d["amplitude"] = (high - low) / prev_close
    d["volume_std_20"] = volume.pct_change().rolling(20).std()
    d["dist_high_60"] = close / close.rolling(60).max() - 1
    d["dist_low_60"] = close / close.rolling(60).min() - 1
    d["gap"] = open_ / prev_close - 1
    d["intraday_return"] = close / open_ - 1
    gains = ret.clip(lower=0).rolling(14).mean()
    losses = (-ret.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gains / losses.replace(0, np.nan))
    rsi = rsi.mask((losses == 0) & (gains > 0), 100).mask((losses == 0) & (gains == 0), 50)
    d["rsi_14"] = rsi
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    d["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    d["return_skew_20"] = ret.rolling(20).skew()
    d["ma20"] = moving_averages[20]
    return d


def _add_cross_sectional_factors(frames):
    panel = pd.concat(frames, ignore_index=True)
    grouped = panel.groupby("date", sort=False)
    panel["rs_20"] = panel["momentum_20"] - grouped["momentum_20"].transform("mean")
    panel["market_advance_ratio"] = grouped["_ret1"].transform(lambda values: float((values > 0).mean()))
    panel["market_limit_up_count"] = grouped["_limit_up"].transform("sum")
    panel["market_limit_down_count"] = grouped["_limit_down"].transform("sum")
    panel["market_cap_quantile_proxy"] = grouped["_size_proxy"].rank(pct=True)
    panel["volatility_median60"] = grouped["volatility_60"].transform("median")
    return panel


def _rolling_normalize(panel, window, min_periods, feature_count):
    columns = get_feature_columns(feature_count)
    results = []
    keep = [
        "date", "code", "open", "high", "low", "close", "volume", "amount", "ma20",
        "volatility_median60",
    ]
    for _, group in panel.groupby("code", sort=False):
        group = group.sort_values("date").copy()
        raw = group[columns].astype(float)
        mean = raw.rolling(window, min_periods=min_periods).mean()
        std = raw.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
        normalized = ((raw - mean) / std.fillna(1.0)).replace([np.inf, -np.inf], np.nan).clip(-8, 8)
        result = group[keep].copy()
        result[columns] = normalized
        result["trend_momentum20"] = group["momentum_20"]
        result["raw_volatility60"] = group["volatility_60"]
        result = result.dropna(subset=columns + [
            "ma20", "trend_momentum20", "raw_volatility60", "volatility_median60",
        ])
        results.append(result.reset_index(drop=True))
    return results


class StockData:
    def __init__(self, frame, config):
        self.code = str(frame["code"].iloc[0])
        self.feature_columns = get_feature_columns(config.get("feature_count", 20))
        self.frame = frame.reset_index(drop=True)
        self.split = int(len(frame) * (1 - float(config.get("test_ratio", 0.2))))
        self.validation_split = int(self.split * (1 - float(config.get("validation_ratio", 0.2))))
        if min(self.validation_split, self.split - self.validation_split, len(frame) - self.split) < 100:
            raise RuntimeError("%s 因子预热后数据不足，训练/验证/测试均需至少 100 条" % self.code)

    def view(self, dataset, window=None):
        if dataset == "train":
            indexes = slice(0, self.validation_split)
        elif dataset == "validation":
            indexes = slice(self.validation_split, self.split)
        elif dataset == "test":
            indexes = slice(self.split, len(self.frame))
        else:
            raise ValueError("未知数据集：%s" % dataset)
        part = self.frame.iloc[indexes]
        if window is not None:
            part = part.iloc[-int(window):]
        return MarketView(
            self.code,
            part["date"].to_numpy(),
            np.ascontiguousarray(part[self.feature_columns].to_numpy(dtype=np.float32)),
            part["close"].to_numpy(dtype=np.float64),
            part["ma20"].to_numpy(dtype=np.float64),
            part["trend_momentum20"].to_numpy(dtype=np.float64),
            part["raw_volatility60"].to_numpy(dtype=np.float64),
            part["volatility_median60"].to_numpy(dtype=np.float64),
        )


def load_stock_data(config):
    codes = [str(code).zfill(6) for code in config["codes"]]
    cache_dir = Path(config.get("cache_dir", "cache_v2"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_text = ",".join(sorted(codes)) + str(config.get("start_date")) + str(config.get("end_date"))
    key_text += str(config.get("adjust")) + str(config.get("normalization_window", 120))
    key_text += "-factor-schema-v3-count-%s" % int(config.get("feature_count", 20))
    cache_key = hashlib.md5(key_text.encode("utf-8")).hexdigest()[:10]
    window = int(config.get("normalization_window", 120))
    min_periods = int(config.get("normalization_min_periods", 60))
    cache_paths = {code: cache_dir / ("%s_%s.pkl" % (code, cache_key)) for code in codes}
    use_cache = all(path.exists() for path in cache_paths.values()) and not bool(
        config.get("refresh", False)
    )

    def load_one(code):
        cache_path = cache_paths[code]
        if use_cache:
            return pd.read_pickle(cache_path), "本地缓存"
        raw, source = fetch_daily(code, config)
        return raw, source

    raw_frames = []
    sources = {}
    workers = max(1, int(config.get("parallel_workers", 3)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_one, code): code for code in codes}
        for future in as_completed(futures):
            code = futures[future]
            frame, source = future.result()
            sources[code] = source
            if source == "本地缓存":
                raw_frames.append(frame)
            else:
                raw_frames.append(_base_factors(frame, code))
            print("  %s 使用%s，%d 根日线" % (code, source, len(frame)), flush=True)

    if all(sources[code] == "本地缓存" for code in codes):
        # as_completed 会改变顺序，按代码重新映射。
        by_code = {str(frame["code"].iloc[0]): frame for frame in raw_frames}
        return {code: StockData(by_code[code], config) for code in codes}

    if any(sources[code] == "本地缓存" for code in codes):
        raise RuntimeError("请勿混用已处理缓存和原始数据；请清理本次缓存键对应文件后重试")
    panel = _add_cross_sectional_factors(raw_frames)
    processed = _rolling_normalize(
        panel, window, min_periods, int(config.get("feature_count", 20))
    )
    result = {}
    for frame in processed:
        code = str(frame["code"].iloc[0])
        frame.to_pickle(cache_dir / ("%s_%s.pkl" % (code, cache_key)))
        result[code] = StockData(frame, config)
    return {code: result[code] for code in codes}
