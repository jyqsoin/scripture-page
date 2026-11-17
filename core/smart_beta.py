# -*- coding: utf-8 -*-
"""Smart Beta 固定暴露（Step I2-alt-1）。

本步只实现两类固定暴露，不做 IC 筛选、不预测收益：
- 低价暴露：按调仓日可见的最新收盘价升序，取前 20% 等权；
- 低波暴露：按过去 60 个交易日日收益率标准差升序，取前 20% 等权。

口径与限制：
- 本地当前只有前复权日线，低价暂以前复权收盘价为代理；后续接入不复权价或
  PE 后可替换评分函数，不影响回测流程；
- 所有评分只使用调仓日及之前的数据，停牌股票取当时最近一个可见交易日；
- 暴露选取不代表收益预测，本策略定位为组合层面的风控工具。

仅研究用途，不构成投资建议。
"""

import math

import numpy as np
import pandas as pd

DISCLAIMER = "仅研究用途，不构成投资建议"

# 默认参数：每种暴露选取股票池前 20%，低波观察 60 个交易日
DEFAULT_PCT = 0.20
DEFAULT_VOL_WINDOW = 60

# 动量采用 252 日形成期，并剔除最近 21 个交易日，避免短期反转干扰
DEFAULT_MOMENTUM_WINDOW = 252
DEFAULT_MOMENTUM_SKIP = 21

# 三因子预设资金比例；自定义比例接口为后续 UI 滑块预留
THREE_FACTOR_WEIGHT_PRESETS = {
    "equal": {"low_price": 0.33, "low_vol": 0.33, "momentum": 0.34},
    "conservative": {"low_price": 0.40, "low_vol": 0.40, "momentum": 0.20},
    "aggressive": {"low_price": 0.20, "low_vol": 0.20, "momentum": 0.60},
}


def latest_close(df, trade_date):
    """返回 trade_date 当日或之前最近一个交易日的收盘价；无数据返回 NaN。"""
    visible = df[df["date"] <= trade_date]
    if visible.empty:
        return np.nan
    close = float(visible.iloc[-1]["close"])
    return close if np.isfinite(close) and close > 0 else np.nan


def realized_volatility(df, trade_date, window=DEFAULT_VOL_WINDOW):
    """计算截至 trade_date 的 window 日已实现波动率（日收益率标准差）。"""
    visible = df[df["date"] <= trade_date]
    # window 个收益率需要 window + 1 个收盘价，不足则不参与排序
    if len(visible) < window + 1:
        return np.nan
    close = visible["close"].tail(window + 1).astype(float)
    returns = close.pct_change().dropna()
    if len(returns) != window or (close <= 0).any():
        return np.nan
    value = float(returns.std(ddof=1))
    return value if np.isfinite(value) else np.nan


def price_scores(data_map, trade_date):
    """低价评分：值越低排名越靠前。返回 index=code 的收盘价 Series。"""
    values = {code: latest_close(df, trade_date)
              for code, df in data_map.items()}
    return pd.Series(values, dtype=float).dropna().sort_values()


def volatility_scores(data_map, trade_date, window=DEFAULT_VOL_WINDOW):
    """低波评分：值越低排名越靠前。返回 index=code 的波动率 Series。"""
    values = {code: realized_volatility(df, trade_date, window)
              for code, df in data_map.items()}
    return pd.Series(values, dtype=float).dropna().sort_values()


def select_bottom(scores, pct=DEFAULT_PCT):
    """从已升序排列的评分中选取前 pct，返回代码列表；至少选 1 只。"""
    if not 0 < pct <= 1:
        raise ValueError("pct 必须在 (0, 1] 内")
    count = max(1, int(math.floor(len(scores) * pct)))
    return scores.index[:count].tolist()


def build_exposures(data_map, trade_date, pct=DEFAULT_PCT,
                    vol_window=DEFAULT_VOL_WINDOW):
    """一次性生成低价与低波持仓，并保留评分供持仓明细追溯。"""
    prices = price_scores(data_map, trade_date)
    volatilities = volatility_scores(data_map, trade_date, vol_window)
    low_price = select_bottom(prices, pct)
    low_vol = select_bottom(volatilities, pct)
    scores = pd.DataFrame({
        "price": prices,
        "volatility": volatilities,
    })
    return {
        "date": trade_date,
        "low_price": low_price,
        "low_vol": low_vol,
        "scores": scores,
        "counts": {"low_price": len(low_price), "low_vol": len(low_vol)},
    }


def momentum_return(df, trade_date, window=DEFAULT_MOMENTUM_WINDOW,
                    skip=DEFAULT_MOMENTUM_SKIP):
    """过去 window 日累计收益，形成期终点前再排除最近 skip 个交易日。"""
    if window <= 0 or skip < 0:
        raise ValueError("window 必须为正数，skip 不能为负数")
    visible = df[df["date"] <= trade_date]
    # window 个收益需要 window + 1 个收盘价，skip 日完全不参与形成期
    if len(visible) < window + skip + 1:
        return np.nan
    formation = visible.iloc[:len(visible) - skip] if skip else visible
    close = formation["close"].tail(window + 1).astype(float)
    if len(close) != window + 1 or (close <= 0).any():
        return np.nan
    value = float(close.iloc[-1] / close.iloc[0] - 1)
    return value if np.isfinite(value) else np.nan


def momentum_scores(data_map, trade_date, window=DEFAULT_MOMENTUM_WINDOW,
                    skip=DEFAULT_MOMENTUM_SKIP):
    """动量评分：值越高排名越靠前，返回按动量降序排列的 Series。"""
    values = {code: momentum_return(df, trade_date, window, skip)
              for code, df in data_map.items()}
    return pd.Series(values, dtype=float).dropna().sort_values(ascending=False)


def select_top(scores, pct=DEFAULT_PCT):
    """从已降序排列的评分中选取前 pct，返回代码列表；至少选 1 只。"""
    if not 0 < pct <= 1:
        raise ValueError("pct 必须在 (0, 1] 内")
    count = max(1, int(math.floor(len(scores) * pct)))
    return scores.index[:count].tolist()


def _resolve_three_factor_weights(mode, factor_weights):
    """解析预设或自定义资金比例；自定义比例会归一化，供后续 UI 滑块调用。"""
    if factor_weights is None:
        if mode not in THREE_FACTOR_WEIGHT_PRESETS:
            raise ValueError("未知三因子模式：%s" % mode)
        return dict(THREE_FACTOR_WEIGHT_PRESETS[mode])
    required = {"low_price", "low_vol", "momentum"}
    if set(factor_weights) != required:
        raise ValueError("自定义权重必须包含 low_price/low_vol/momentum")
    weights = {key: float(factor_weights[key]) for key in required}
    if any(value < 0 for value in weights.values()):
        raise ValueError("因子资金比例不能为负数")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("因子资金比例之和必须大于 0")
    return {key: value / total for key, value in weights.items()}


def build_three_factor_exposures(
        data_map, trade_date, pct=DEFAULT_PCT, vol_window=DEFAULT_VOL_WINDOW,
        momentum_window=DEFAULT_MOMENTUM_WINDOW,
        momentum_skip=DEFAULT_MOMENTUM_SKIP, mode="equal", factor_weights=None,
        min_intersection=10, union_top_n=30):
    """构建低价、低波、动量三因子暴露及归一化持仓权重。

    三个单因子子组合各取前 pct；交集达到 min_intersection 只时使用交集，
    否则在并集中按三个因子百分位名次之和取前 union_top_n 只。
    入选股票按所属单因子子组合的等权资金份额累加，再归一化为组合权重。
    """
    if min_intersection < 1 or union_top_n < 1:
        raise ValueError("min_intersection 与 union_top_n 必须为正整数")
    allocations = _resolve_three_factor_weights(mode, factor_weights)
    prices = price_scores(data_map, trade_date)
    volatilities = volatility_scores(data_map, trade_date, vol_window)
    momentums = momentum_scores(data_map, trade_date,
                                momentum_window, momentum_skip)
    sleeves = {
        "low_price": select_bottom(prices, pct),
        "low_vol": select_bottom(volatilities, pct),
        "momentum": select_top(momentums, pct),
    }
    if any(not members for members in sleeves.values()):
        raise ValueError("有效行情不足，无法生成完整三因子暴露")

    # 三个因子统一转换为“越小越好”的百分位名次，缺失因子按最差名次处理
    rank_frame = pd.DataFrame({
        "price_rank": prices.rank(method="average", pct=True),
        "volatility_rank": volatilities.rank(method="average", pct=True),
        "momentum_rank": momentums.rank(method="average", pct=True,
                                        ascending=False),
    }).fillna(1.0)
    rank_frame["rank_sum"] = rank_frame.sum(axis=1)
    ordered = rank_frame.assign(code=rank_frame.index).sort_values(
        ["rank_sum", "code"], kind="mergesort")
    intersection_set = set(sleeves["low_price"]) & set(sleeves["low_vol"]) & set(sleeves["momentum"])
    union_set = set().union(*(set(members) for members in sleeves.values()))
    intersection = ordered.loc[ordered.index.isin(intersection_set)].index.tolist()
    union_top = ordered.loc[ordered.index.isin(union_set)].head(union_top_n).index.tolist()
    use_intersection = len(intersection) >= min_intersection
    selected = intersection if use_intersection else union_top
    if not selected:
        raise ValueError("三因子交集与并集均无可用股票")

    raw_weights = pd.Series(0.0, index=selected, dtype=float)
    for factor, members in sleeves.items():
        member_set = set(members)
        sleeve_weight = allocations[factor] / len(members)
        for code in selected:
            if code in member_set:
                raw_weights.loc[code] += sleeve_weight
    if raw_weights.sum() <= 0:
        raise ValueError("三因子组合权重之和为 0")
    weights = raw_weights / raw_weights.sum()
    scores = pd.DataFrame({
        "price": prices, "volatility": volatilities, "momentum": momentums,
    }).join(rank_frame, how="outer")
    return {
        "date": trade_date, "mode": mode, "allocations": allocations,
        "low_price": sleeves["low_price"], "low_vol": sleeves["low_vol"],
        "momentum": sleeves["momentum"], "intersection": intersection,
        "union_top": union_top, "selected": selected, "weights": weights,
        "selection_rule": "intersection" if use_intersection else "union_top",
        "scores": scores,
        "counts": {
            "low_price": len(sleeves["low_price"]),
            "low_vol": len(sleeves["low_vol"]),
            "momentum": len(sleeves["momentum"]),
            "intersection": len(intersection), "union": len(union_set),
            "selected": len(selected),
        },
    }
