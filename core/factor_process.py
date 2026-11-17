# -*- coding: utf-8 -*-
"""
因子处理（V1.4 Step B2，轻量版）。

提供两项与截面无关未来信息的处理能力：
- 去冗余：按显著性排序贪心挑选代表因子，|相关性| 超阈值的同组因子剔除；
- 残差正交化：按给定顺序做截面 OLS 残差化，消除因子间共线。

防前视约束：
- 所有运算只使用当期截面数据，不涉及未来收益；
- 去冗余使用的相关性与显著性来自因子筛查（B1）的训练段统计。

仅研究用途，不构成投资建议。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.factor_engine import zscore

DISCLAIMER = "仅研究用途，不构成投资建议"


def deduplicate(corr_mean, scores, directions=None, threshold=0.7):
    """
    因子去冗余。

    参数：
        corr_mean: 因子相关性矩阵（DataFrame，index/columns 为因子名）
        scores: dict {factor_name: 显著性得分}，通常用训练段 |t 值| 或 |IC|
        directions: dict {factor_name: 1/-1}，用于方向对齐后再比较相关性
        threshold: |相关性| 达到该值即视为冗余

    返回 (selected, dropped)：
        selected: 保留的因子名列表（按显著性降序）
        dropped: dict {被剔除因子: (保留因子, 对齐后相关系数)}
    """
    directions = directions or {}
    if corr_mean is None or corr_mean.empty:
        return list(scores.keys()), {}

    def _score(name):
        value = scores.get(name, 0)
        try:
            value = abs(float(value))
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if not np.isfinite(value) else value

    order = sorted(list(corr_mean.index), key=_score, reverse=True)
    selected = []
    dropped = {}
    for name in order:
        duplicate = None
        for kept in selected:
            try:
                value = float(corr_mean.loc[name, kept])
            except (KeyError, TypeError, ValueError):
                continue
            value *= directions.get(name, 1) * directions.get(kept, 1)
            if pd.notna(value) and abs(value) >= threshold:
                duplicate = (kept, value)
                break
        if duplicate:
            dropped[name] = duplicate
        else:
            selected.append(name)
    return selected, dropped


def orthogonalize(frame, order=None, min_count=20):
    """
    截面残差正交化。

    按 order 顺序逐个处理：第 i 个因子对前面所有因子（含截距）做 OLS 回归，
    用残差替代原值，使因子之间近似正交。最后逐列 Z-Score 标准化。

    参数：
        frame: 因子矩阵（index=股票代码, columns=因子名），值应为原始或已标准化
        order: 处理顺序（通常是显著性降序）；None 表示按列顺序
        min_count: 单次回归所需最少样本数，不足时保留原值

    返回新的 DataFrame（不修改原 frame）。
    """
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()

    columns = [c for c in (order or list(frame.columns)) if c in frame.columns]
    if not columns:
        return frame.copy()

    result = pd.DataFrame(index=frame.index)
    result[columns[0]] = frame[columns[0]]
    for i, name in enumerate(columns[1:], 1):
        previous = columns[:i]
        data = pd.concat([frame[name].rename("__y__"), result[previous]], axis=1).dropna()
        if len(data) < max(min_count, i + 2):
            result[name] = frame[name]
            continue
        design = np.column_stack([np.ones(len(data)), data[previous].values])
        response = data["__y__"].values
        try:
            coef, *_ = np.linalg.lstsq(design, response, rcond=None)
        except np.linalg.LinAlgError:
            result[name] = frame[name]
            continue
        residual = pd.Series(np.nan, index=frame.index)
        residual.loc[data.index] = response - design @ coef
        result[name] = residual

    for name in result.columns:
        result[name] = zscore(result[name])
    return result


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(DISCLAIMER)
    np.random.seed(7)
    n = 200
    base = np.random.randn(n)
    frame = pd.DataFrame({
        "a": base + np.random.randn(n) * 0.1,
        "b": base * 0.9 + np.random.randn(n) * 0.1,   # 与 a 高度共线
        "c": np.random.randn(n),                      # 与 a 独立
    }, index=["%06d" % i for i in range(n)])

    # 1. 去冗余：b 与 a 相关 0.99，应被剔除
    corr = frame.corr(method="spearman")
    selected, dropped = deduplicate(corr, {"a": 3.2, "b": 1.9, "c": 2.5}, threshold=0.7)
    print("去冗余后保留:", selected)
    print("被剔除:", {k: (v[0], round(v[1], 3)) for k, v in dropped.items()})
    assert "a" in selected and "c" in selected and "b" in dropped
    assert dropped["b"][0] == "a"

    # 2. 正交化：残差与前一因子的相关性应接近 0
    orth = orthogonalize(frame, order=["a", "b", "c"])
    r_ab = orth["a"].corr(orth["b"])
    print("正交化后 a~b 相关:", round(r_ab, 6))
    assert abs(r_ab) < 1e-6
    # 各列标准化后均值 0、标准差 1
    for col in orth.columns:
        valid = orth[col].dropna()
        assert abs(valid.mean()) < 1e-6 and abs(valid.std(ddof=0) - 1) < 1e-6

    # 3. 顺序不同结果不同，但都应正交
    orth2 = orthogonalize(frame, order=["b", "a", "c"])
    assert abs(orth2["b"].corr(orth2["a"])) < 1e-6
    print("不同顺序正交化均通过")

    print("\nfactor_process.py 自测通过；%s" % DISCLAIMER)
