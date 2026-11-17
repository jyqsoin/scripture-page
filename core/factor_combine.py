# -*- coding: utf-8 -*-
"""
多因子合成（V1.3 Step A3）。

把多个标准化后的横截面因子合成为一个综合得分，支持三种方法：
- 等权：各因子等权平均
- IC 加权：权重为历史 IC 均值（带符号，自动统一因子方向）
- 回归加权：用历史截面回归的系数作为权重

防前视约束：
- IC 加权只使用 t 日之前的 IC 均值，不使用当期或未来 IC；
- 回归加权只使用 t 日之前的因子-收益截面做回归，不使用当期收益；
- 因子方向通过 IC 均值符号自动统一，合成后综合得分越大越看多。

仅研究用途，不构成投资建议。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISCLAIMER = "仅研究用途，不构成投资建议"


# ---------------------------------------------------------------------------
# 因子方向统一
# ---------------------------------------------------------------------------

def infer_directions(ic_summary):
    """
    从 IC 汇总推断因子方向。

    IC 均值 > 0 → 方向为 1（因子越大，未来收益越高）；
    IC 均值 < 0 → 方向为 -1（因子越小，未来收益越高）；
    IC 均值 = 0 或 NaN → 方向为 1（默认正向，由调用方决定是否覆盖）。

    返回 dict: {factor_name: 1 或 -1}
    """
    directions = {}
    for name, summary in ic_summary.items():
        ic_mean = summary.get("ic_mean")
        if ic_mean is None or pd.isna(ic_mean) or ic_mean == 0:
            directions[name] = 1
        else:
            directions[name] = 1 if ic_mean > 0 else -1
    return directions


def align_direction(std_frame, directions):
    """
    按方向统一因子符号：方向为 -1 的因子取反。
    返回对齐后的 DataFrame（不修改原 frame）。
    """
    result = std_frame.copy()
    for col in result.columns:
        direction = directions.get(col, 1)
        if direction == -1:
            result[col] = -result[col]
    return result


# ---------------------------------------------------------------------------
# 三种合成方法
# ---------------------------------------------------------------------------

def combine_equal_weight(std_frame, directions=None):
    """
    等权合成：各因子标准化值等权平均。

    std_frame: 标准化后的截面因子矩阵（index=code, columns=factor_name）
    directions: 可选，因子方向 dict；None 时不翻转方向
    返回 Series：综合得分，index=code
    """
    if directions is not None:
        std_frame = align_direction(std_frame, directions)
    # 每行等权平均，跳过 NaN
    return std_frame.mean(axis=1, skipna=True)


def combine_ic_weighted(std_frame, weights, directions=None):
    """
    IC 加权合成：权重为各因子的历史 IC 均值（带符号）。

    weights: dict {factor_name: weight}，权重已含方向符号
    directions: 可选，若提供则先按方向翻转；但 IC 权重本身已含方向，
                一般 directions=None，让权重自带方向。
    返回 Series：综合得分，index=code
    """
    if directions is not None:
        std_frame = align_direction(std_frame, directions)
    # 只保留有权重的因子列
    factor_names = [c for c in std_frame.columns if c in weights]
    if not factor_names:
        return pd.Series(np.nan, index=std_frame.index)
    weight_series = pd.Series({n: float(weights[n]) for n in factor_names})
    # 权重归一化（绝对值之和为 1），避免量纲影响
    weight_sum = weight_series.abs().sum()
    if weight_sum == 0:
        return pd.Series(np.nan, index=std_frame.index)
    weight_series = weight_series / weight_sum
    sub_frame = std_frame[factor_names]
    # 加权平均
    return sub_frame.mul(weight_series, axis=1).sum(axis=1, skipna=True)


def combine_regression(std_frame, return_series):
    """
    回归加权合成：用当期截面因子值对未来收益做线性回归，
    回归系数作为权重，再用这些权重对因子加权合成。

    注意：此函数使用当期 return_series 做回归，存在前视风险。
    实际使用中应传入「历史」收益序列，或用 rolling_regression_weights
    计算 t 日之前的回归系数。

    返回 (score_series, coef_series)
    """
    aligned = pd.concat([std_frame, return_series.rename("__ret__")], axis=1).dropna()
    if len(aligned) < std_frame.shape[1] + 5:
        return pd.Series(np.nan, index=std_frame.index), pd.Series(dtype=float)
    X = aligned[std_frame.columns].values
    y = aligned["__ret__"].values
    # 加截距项
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    try:
        coef, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=std_frame.index), pd.Series(dtype=float)
    # coef[0] 是截距，后面的是因子系数
    factor_coef = pd.Series(coef[1:], index=std_frame.columns)
    score = std_frame.mul(factor_coef, axis=1).sum(axis=1, skipna=True)
    return score, factor_coef


# ---------------------------------------------------------------------------
# IC 权重计算（防前视：只用历史 IC）
# ---------------------------------------------------------------------------

def compute_ic_weights(ic_series_dict, as_of_date=None, window=None):
    """
    从 IC 时序计算各因子的权重。

    ic_series_dict: {factor_name: DataFrame(index=调仓日, columns=['ic'])}
    as_of_date: 截止日期（YYYY-MM-DD），只使用该日期及之前的 IC；None 表示全部
    window: 滚动窗口大小（最近 N 个调仓日），None 表示使用全部历史

    返回 dict: {factor_name: weight}，权重 = 历史 IC 均值（带符号）
    """
    weights = {}
    for name, ic_df in ic_series_dict.items():
        ic = ic_df["ic"].copy()
        if as_of_date is not None:
            ic = ic[ic.index <= as_of_date]
        if window is not None and len(ic) > window:
            ic = ic.iloc[-window:]
        ic_mean = ic.mean()
        weights[name] = float(ic_mean) if pd.notna(ic_mean) else 0.0
    return weights


# ---------------------------------------------------------------------------
# 滚动回归权重（防前视）
# ---------------------------------------------------------------------------

def compute_rolling_regression_weights(factor_history, return_history, as_of_date):
    """
    用历史截面数据做回归，得到 t 日的因子权重。

    factor_history: dict {date: DataFrame(index=code, columns=factor_name)}
    return_history: dict {date: Series(index=code)}
    as_of_date: 只使用严格早于该日期的截面做回归

    返回 Series: {factor_name: weight}
    """
    frames = []
    for date in sorted(factor_history.keys()):
        if date >= as_of_date:
            break
        if date not in return_history:
            continue
        f = factor_history[date].copy()
        f["__ret__"] = return_history[date]
        frames.append(f.dropna())
    if not frames:
        return pd.Series(dtype=float)
    pool = pd.concat(frames)
    if len(pool) < pool.shape[1] + 5:
        return pd.Series(dtype=float)
    factor_cols = [c for c in pool.columns if c != "__ret__"]
    X = pool[factor_cols].values
    y = pool["__ret__"].values
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    try:
        coef, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(dtype=float)
    return pd.Series(coef[1:], index=factor_cols)


# ---------------------------------------------------------------------------
# 统一合成入口
# ---------------------------------------------------------------------------

def combine_factors(std_frame, method="equal", directions=None, weights=None,
                    return_series=None):
    """
    多因子合成统一入口。

    method:
        "equal"     — 等权平均
        "ic"        — IC 加权（需提供 weights）
        "regression"— 回归加权（需提供 return_series，注意前视风险）

    std_frame: 标准化后的截面因子矩阵
    directions: 因子方向 dict（equal/ic 模式可选）
    weights: IC 权重 dict（ic 模式必填）
    return_series: 收益序列（regression 模式必填）

    返回 Series：综合得分，index=code
    """
    if method == "equal":
        return combine_equal_weight(std_frame, directions=directions)
    if method == "ic":
        if weights is None:
            raise ValueError("IC 加权需要提供 weights 参数")
        return combine_ic_weighted(std_frame, weights, directions=directions)
    if method == "regression":
        if return_series is None:
            raise ValueError("回归加权需要提供 return_series 参数")
        score, _ = combine_regression(std_frame, return_series)
        return score
    raise ValueError("未知合成方法：%s" % method)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(DISCLAIMER)

    # 构造 100 只股票、5 个因子的标准化截面矩阵
    np.random.seed(42)
    n_codes = 100
    codes = ["%06d" % i for i in range(n_codes)]
    factors = ["mom", "value", "quality", "lowvol", "liquidity"]
    # 随机标准化因子值
    raw = np.random.randn(n_codes, len(factors))
    std_frame = pd.DataFrame(raw, index=codes, columns=factors)
    # 构造未来收益：与 mom 正相关、与 lowvol 负相关
    future_returns = (
        std_frame["mom"] * 0.03
        - std_frame["lowvol"] * 0.02
        + np.random.randn(n_codes) * 0.01
    )

    # 1. 等权合成
    score_eq = combine_factors(std_frame, method="equal")
    print("等权合成：")
    print("  得分均值: %.4f 标准差: %.4f" % (score_eq.mean(), score_eq.std()))
    print("  与未来收益相关: %.4f" % score_eq.corr(future_returns))

    # 2. IC 加权（模拟历史 IC：mom 正向、lowvol 负向）
    ic_weights = {"mom": 0.05, "value": 0.02, "quality": 0.03,
                  "lowvol": -0.04, "liquidity": 0.01}
    score_ic = combine_factors(std_frame, method="ic", weights=ic_weights)
    print("\nIC 加权合成：")
    print("  权重:", {k: "%.4f" % v for k, v in ic_weights.items()})
    print("  与未来收益相关: %.4f" % score_ic.corr(future_returns))
    # IC 加权应该比等权更能捕捉 mom 正向、lowvol 负向
    assert abs(score_ic.corr(future_returns)) >= abs(score_eq.corr(future_returns)) - 0.05

    # 3. 方向推断
    ic_summary = {
        "mom": {"ic_mean": 0.05},
        "value": {"ic_mean": 0.02},
        "lowvol": {"ic_mean": -0.04},
    }
    directions = infer_directions(ic_summary)
    print("\n方向推断:", directions)
    assert directions["mom"] == 1
    assert directions["lowvol"] == -1

    # 4. 回归加权
    score_reg, coef = combine_regression(std_frame, future_returns)
    print("\n回归加权合成：")
    print("  系数:", {k: "%.4f" % v for k, v in coef.items()})
    print("  与未来收益相关: %.4f" % score_reg.corr(future_returns))
    # 回归加权与收益相关性应最高
    assert score_reg.corr(future_returns) > score_ic.corr(future_returns)

    # 5. 方向对齐验证：lowvol 方向为 -1，对齐后应该取反
    aligned = align_direction(std_frame[["lowvol"]], {"lowvol": -1})
    assert (aligned["lowvol"] == -std_frame["lowvol"]).all()

    print("\nfactor_combine.py 自测通过；%s" % DISCLAIMER)
