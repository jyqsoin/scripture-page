# -*- coding: utf-8 -*-
"""部署侧状态构造参考实现（与训练口径一致）。

用途：为 IR 模型准备 (1, 1208) 的 state。
注意：
1. 必须传入**自上市以来的完整日线**，因为 OBV 使用 cumsum，截断历史会改变因子值；
2. 归一化是逐日 120 日滚动 z-score（min_periods=60、std ddof=1、clip ±8），
   不是固定均值/方差，因此不能把它固化成常量嵌进 IR；
3. 输出顺序：最近 60 天 × 20 因子（按日展开）→ 1200 维，再拼 8 维账户状态。
"""

import numpy as np
import pandas as pd

from data_loader_v2 import _base_factors, get_feature_columns

TRADING_DAYS = 242


def build_normalized_frame(raw_frame, code, feature_count=20, window=120, min_periods=60):
    """raw_frame 需包含 date/open/high/low/close/volume/amount 的完整历史。"""
    columns = get_feature_columns(feature_count)
    enriched = _base_factors(raw_frame.copy(), str(code).zfill(6))
    raw = enriched[columns].astype(float)
    mean = raw.rolling(window, min_periods=min_periods).mean()
    std = raw.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    normalized = ((raw - mean) / std.fillna(1.0)).replace([np.inf, -np.inf], np.nan).clip(-8, 8)
    result = enriched[["date", "code", "open", "high", "low", "close", "volume", "amount"]].copy()
    result[columns] = normalized
    result["ma20"] = enriched["ma20"]
    result["trend_momentum20"] = enriched["momentum_20"]
    result["raw_volatility60"] = enriched["volatility_60"]
    result = result.dropna(subset=columns + ["ma20", "trend_momentum20", "raw_volatility60"])
    return result.reset_index(drop=True)


def build_state(normalized_frame, index, account, lookback=60, feature_count=20):
    """取第 index 行（含）往前 lookback 天，拼账户状态，返回 (1208,)。"""
    columns = get_feature_columns(feature_count)
    if index < lookback - 1:
        raise ValueError("index 不足以构造 60 天状态")
    window = normalized_frame.iloc[index - lookback + 1:index + 1][columns].to_numpy(np.float32)
    account = np.asarray(account, dtype=np.float32).reshape(-1)
    if account.size != 8:
        raise ValueError("account 需为 8 维")
    return np.concatenate([window.reshape(-1), account])


def account_state(cash, shares, price, initial_cash, nav, drawdown, last_action,
                  hold_steps, unrealized_pnl, ret5):
    """与 trading_env_v2._state 中的 8 维账户特征一致。"""
    portfolio = cash + shares * price
    cash_ratio = cash / max(portfolio, 1e-8)
    position_fraction = shares * price / max(portfolio, 1e-8)
    return np.array([
        cash_ratio,
        position_fraction,
        nav,
        drawdown,
        last_action,
        min(hold_steps / 60.0, 3.0),
        float(np.clip(unrealized_pnl, -1, 1)),
        float(np.clip(ret5, -0.2, 0.2)) * 5,
    ], dtype=np.float32)
