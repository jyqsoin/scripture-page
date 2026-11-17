# -*- coding: utf-8 -*-
"""
独立 LightGBM 横截面排序模块（研究用途）。

- 仅使用本地行情和 Step 3 的 MA5、MA20、RSI14、量比因子；
- 标签为每只股票未来 5 根有效交易日 K 线的收益，并转换为当日横截面五级相关度；
- 按日期依次划分训练、验证、测试集，边界执行 5 日 purge，避免标签区间跨集合重叠；
- 输出未来收益排序分数、NDCG、Rank IC 等研究指标，不输出涨跌概率或确定收益；
- 模块不接入 UI、回测下单或券商接口，不联网，也不阻塞桌面主流程。

仅研究用途，不构成投资建议。历史排序表现不代表未来收益。
"""

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.data_loader import load_local_daily
from core.factors import add_basic_factors

DISCLAIMER = "仅研究用途，不构成投资建议"
SCORE_MEANING = "分数仅用于同一交易日横截面相对排序，不是涨跌概率或收益承诺"
FEATURE_COLUMNS = ["ma5", "ma20", "rsi14", "volume_ratio"]


@dataclass
class RankingConfig:
    horizon: int = 5
    min_group_size: int = 5
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    purge_days: int = 5
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    colsample_bytree: float = 0.90
    reg_lambda: float = 0.0
    early_stopping_rounds: int = 30
    random_state: int = 42
    n_jobs: int = 1


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _factor_frame(df):
    """清洗有效交易日并计算历史因子；标签位移在该函数外按股票独立完成。"""
    required = {"date", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("排序行情缺少列：%s" % sorted(missing))
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[(frame["close"] > 0) & (frame["volume"].fillna(0) > 0)]
    frame = frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if frame.empty:
        return frame
    return add_basic_factors(frame)


def build_samples(data_map, horizon=5, min_group_size=5):
    """
    构造训练样本。每只股票单独计算未来收益，杜绝跨股票 shift。

    每日样本不足 min_group_size 时整日剔除；label 为 0–4 级横截面相关度，
    future_return 仅用于评估，不作为模型输入。
    """
    horizon = int(horizon)
    min_group_size = int(min_group_size)
    if horizon <= 0:
        raise ValueError("horizon 必须大于 0")
    if min_group_size < 2:
        raise ValueError("min_group_size 至少为 2，才能进行横截面排序")

    samples = []
    for code, raw in data_map.items():
        if raw is None or raw.empty:
            continue
        frame = _factor_frame(raw)
        if frame.empty:
            continue
        frame["future_return"] = frame["close"].shift(-horizon) / frame["close"] - 1
        frame["label_end_date"] = frame["date"].shift(-horizon)
        columns = ["date"] + FEATURE_COLUMNS + ["future_return", "label_end_date"]
        item = frame[columns].copy()
        item["code"] = str(code)
        item = item.replace([np.inf, -np.inf], np.nan)
        item = item.dropna(subset=FEATURE_COLUMNS + ["future_return", "label_end_date"])
        samples.append(item)

    output_columns = ["date", "code"] + FEATURE_COLUMNS + ["future_return", "label_end_date", "label"]
    if not samples:
        return pd.DataFrame(columns=output_columns)
    result = pd.concat(samples, ignore_index=True)
    group_size = result.groupby("date")["code"].transform("size")
    result = result[group_size >= min_group_size].copy()
    if result.empty:
        return pd.DataFrame(columns=output_columns)

    group_size = result.groupby("date")["code"].transform("size")
    ranks = result.groupby("date")["future_return"].rank(method="average", ascending=True)
    relevance = np.floor((ranks - 1) / group_size * 5).clip(0, 4)
    result["label"] = relevance.astype(int)
    return result[output_columns].sort_values(["date", "code"]).reset_index(drop=True)


def time_split(samples, train_ratio=0.70, valid_ratio=0.15, purge_days=5):
    """按日期顺序切分；训练/验证、验证/测试之间各保留 purge_days 个空档。"""
    if samples.empty or "label_end_date" not in samples.columns:
        raise ValueError("time_split 需要 build_samples 生成的非空样本")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 0 和 1 之间")
    if not 0 <= valid_ratio < 1 or train_ratio + valid_ratio >= 1:
        raise ValueError("valid_ratio 必须不小于 0，且 train_ratio + valid_ratio 必须小于 1")
    purge_days = int(purge_days)
    if purge_days < 0:
        raise ValueError("purge_days 不能为负数")

    dates = np.array(sorted(samples["date"].unique()))
    train_boundary = int(len(dates) * train_ratio)
    test_boundary = int(len(dates) * (train_ratio + valid_ratio))
    train_end = train_boundary - purge_days
    valid_end = test_boundary - purge_days if valid_ratio > 0 else test_boundary

    train_dates = dates[:train_end]
    valid_dates = dates[train_boundary:valid_end] if valid_ratio > 0 else dates[:0]
    test_dates = dates[test_boundary:]
    # 除固定空档外，再检查真实标签结束日，覆盖停牌导致的五根 K 线跨期情况。
    train = samples[
        samples["date"].isin(train_dates)
        & (samples["label_end_date"] < dates[train_boundary])
    ].copy()
    valid = samples[
        samples["date"].isin(valid_dates)
        & (samples["label_end_date"] < dates[test_boundary])
    ].copy()
    test = samples[samples["date"].isin(test_dates)].copy()
    if train.empty or test.empty:
        raise ValueError("日期切分后训练集或测试集为空，请增加历史行情或调整切分比例")
    if valid_ratio > 0 and valid.empty:
        raise ValueError("日期切分后验证集为空，请增加历史行情、减少 purge_days 或调整比例")
    return train, valid, test


def _groups(frame):
    """LightGBM group 必须与已按 date/code 排序的行一一对应。"""
    return frame.groupby("date", sort=False).size().to_numpy(dtype=int)


def _predict_scores(model, frame):
    kwargs = {}
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration:
        kwargs["num_iteration"] = int(best_iteration)
    scores = model.predict(frame[FEATURE_COLUMNS], **kwargs)
    return np.asarray(scores, dtype=float)


def evaluate_ranking(frame, scores, k=5):
    """评估每日排序；top/bottom 收益只是样本诊断，不是可交易策略收益。"""
    k = int(k)
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if frame.empty:
        return {"periods": 0}
    work = frame[["date", "code", "future_return", "label"]].copy()
    work["score"] = np.asarray(scores, dtype=float)
    ndcgs, ics, top_returns, bottom_returns = [], [], [], []

    for _, group in work.groupby("date", sort=False):
        group = group.dropna(subset=["future_return", "label", "score"])
        if group.empty:
            continue
        ranked = group.sort_values("score", ascending=False)
        count = min(k, len(group))
        relevance = group["label"].to_numpy(dtype=float)
        ideal_relevance = np.sort(relevance)[::-1]
        gains = np.power(2.0, ranked["label"].to_numpy(dtype=float)[:count]) - 1
        ideal_gains = np.power(2.0, ideal_relevance[:count]) - 1
        discounts = np.log2(np.arange(2, count + 2))
        ideal_dcg = float(np.sum(ideal_gains / discounts))
        ndcgs.append(float(np.sum(gains / discounts) / ideal_dcg) if ideal_dcg else 1.0)

        if group["future_return"].nunique() > 1 and group["score"].nunique() > 1:
            ic = group["future_return"].rank(method="average").corr(
                group["score"].rank(method="average"), method="pearson"
            )
            if pd.notna(ic):
                ics.append(float(ic))
        top_returns.append(float(ranked.head(count)["future_return"].mean()))
        bottom_returns.append(float(ranked.tail(count)["future_return"].mean()))

    top_mean = _finite(np.mean(top_returns)) if top_returns else None
    bottom_mean = _finite(np.mean(bottom_returns)) if bottom_returns else None
    return {
        "periods": len(ndcgs),
        "ndcg_at_%d" % k: _finite(np.mean(ndcgs)) if ndcgs else None,
        "mean_spearman_ic": _finite(np.mean(ics)) if ics else None,
        "top_%d_mean_return" % k: top_mean,
        "bottom_%d_mean_return" % k: bottom_mean,
        "top_bottom_spread": _finite(top_mean - bottom_mean) if top_mean is not None and bottom_mean is not None else None,
        "disclaimer": DISCLAIMER,
    }


def _period_info(frame):
    if frame.empty:
        return {"samples": 0, "dates": 0, "start": "", "end": ""}
    return {
        "samples": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
    }


def train_ranker(data_map, config=None):
    """训练 LightGBM Ranker；返回模型、测试指标、切分信息和因子重要度。"""
    config = config or RankingConfig()
    if config.horizon <= 0 or config.purge_days < config.horizon:
        raise ValueError("horizon 必须大于 0，且 purge_days 不能小于 horizon")
    if config.n_estimators <= 0:
        raise ValueError("n_estimators 必须大于 0")
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("缺少 lightgbm，请先执行：python -m pip install lightgbm") from exc

    samples = build_samples(data_map, config.horizon, config.min_group_size)
    if samples.empty:
        raise ValueError("没有有效训练样本：需足够的本地历史行情，且每个交易日有足够股票")
    train, valid, test = time_split(
        samples, config.train_ratio, config.valid_ratio, config.purge_days
    )
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(config.n_estimators),
        learning_rate=float(config.learning_rate),
        num_leaves=int(config.num_leaves),
        min_child_samples=int(config.min_child_samples),
        colsample_bytree=float(config.colsample_bytree),
        reg_lambda=float(config.reg_lambda),
        random_state=int(config.random_state),
        n_jobs=int(config.n_jobs),
        device="cpu",
        verbosity=-1,
    )
    fit_kwargs = {"group": _groups(train)}
    if not valid.empty:
        fit_kwargs.update({
            "eval_set": [(valid[FEATURE_COLUMNS], valid["label"])],
            "eval_group": [_groups(valid)],
            "eval_names": ["validation"],
            "eval_at": (5,),
        })
        if config.early_stopping_rounds > 0:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(config.early_stopping_rounds, verbose=False)
            ]
    model.fit(train[FEATURE_COLUMNS], train["label"], **fit_kwargs)

    validation_metrics = (
        evaluate_ranking(valid, _predict_scores(model, valid)) if not valid.empty else {"periods": 0}
    )
    test_metrics = evaluate_ranking(test, _predict_scores(model, test))
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    return {
        "model": model,
        "config": asdict(config),
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "split": {
            "train": _period_info(train),
            "validation": _period_info(valid),
            "test": _period_info(test),
            "purge_days": int(config.purge_days),
        },
        "feature_importance": importance,
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "score_meaning": SCORE_MEANING,
        "disclaimer": DISCLAIMER,
    }


def build_feature_snapshot(data_map, date=None):
    """构造同一交易日快照；默认使用全部输入股票中的最新日期，不混用各股旧日期。"""
    frames = {}
    for code, raw in data_map.items():
        if raw is None or raw.empty:
            continue
        try:
            frame = _factor_frame(raw)
            if not frame.empty:
                frames[str(code)] = frame
        except Exception:
            continue
    if not frames:
        raise ValueError("没有可用于排序的本地因子数据")
    target = pd.Timestamp(date) if date else max(frame["date"].max() for frame in frames.values())
    rows = []
    for code, frame in frames.items():
        current = frame[frame["date"] == target].tail(1)
        if current.empty:
            continue
        row = current.iloc[0]
        values = {key: row[key] for key in FEATURE_COLUMNS}
        if any(pd.isna(value) or not np.isfinite(float(value)) for value in values.values()):
            continue
        rows.append({"date": target, "code": code, **values})
    result = pd.DataFrame(rows, columns=["date", "code"] + FEATURE_COLUMNS)
    if len(result) < 2:
        raise ValueError("同一交易日有效股票不足 2 只，无法进行横截面排序")
    return result.sort_values("code").reset_index(drop=True)


def rank_latest(data_map, model, date=None):
    """输出最新或指定日期的相对排名，不预测单只股票必然上涨或下跌。"""
    snapshot = build_feature_snapshot(data_map, date)
    snapshot["score"] = _predict_scores(model, snapshot)
    snapshot = snapshot.sort_values(["score", "code"], ascending=[False, True]).reset_index(drop=True)
    snapshot.insert(0, "rank", np.arange(1, len(snapshot) + 1))
    snapshot["score_meaning"] = SCORE_MEANING
    snapshot["disclaimer"] = DISCLAIMER
    return snapshot


def train_from_local(codes, fq="qfq", config=None, save=False, model_path=None):
    """从本地 parquet/SQLite 加载股票并训练；缺失行情单独记录，不访问网络。"""
    if isinstance(codes, str):
        codes = [codes]
    data_map, errors = {}, {}
    for code in codes:
        code = str(code).strip().zfill(6)
        try:
            frame = load_local_daily(code, fq)
            if frame.empty:
                raise ValueError("本地无%s行情" % fq)
            data_map[code] = frame
        except Exception as exc:
            errors[code] = str(exc)
    if not data_map:
        raise ValueError("没有可用本地行情：%s" % errors)
    result = train_ranker(data_map, config)
    result["fq"] = fq
    result["load_errors"] = errors
    if save:
        result["model_path"], result["metadata_path"] = save_model(result, model_path)
    return result


def save_model(result, path=None):
    """模型保存为 LightGBM 文本，配置和评估信息保存为同名 JSON。"""
    if path is None:
        model_dir = db.DATA_DIR / "models"
        path = model_dir / ("lgbm_ranker_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_iteration = int(result.get("best_iteration", 0) or 0)
    result["model"].booster_.save_model(
        str(path), num_iteration=best_iteration if best_iteration > 0 else -1
    )
    metadata = {
        "created_at": db.now_str(),
        "config": result["config"],
        "fq": result.get("fq", ""),
        "load_errors": result.get("load_errors", {}),
        "feature_columns": FEATURE_COLUMNS,
        "metrics": result["metrics"],
        "split": result["split"],
        "best_iteration": best_iteration,
        "score_meaning": SCORE_MEANING,
        "disclaimer": DISCLAIMER,
    }
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return str(path), str(metadata_path)


def load_model(path):
    """读取本地 LightGBM 文本模型和同名 JSON 元数据。"""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("缺少 lightgbm，请先执行：python -m pip install lightgbm") from exc
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError("模型或同名 JSON 元数据不存在：%s" % path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("模型因子口径与当前模块不一致")
    model = lgb.Booster(model_file=str(path))
    return {"model": model, "metadata": metadata}


if __name__ == "__main__":
    # 合成横截面自测：不联网、不读正式数据库、不保存模型。
    print(DISCLAIMER)
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=280)
    demo_map = {}
    for offset in range(8):
        returns = rng.normal(0, 0.012, len(dates)) + np.sin(np.arange(len(dates)) / 17 + offset) * 0.001
        close = pd.Series((10 + offset) * np.exp(np.cumsum(returns)))
        open_price = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, len(dates)))
        demo_map["SYNTH%02d" % offset] = pd.DataFrame({
            "date": dates,
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.01,
            "low": np.minimum(open_price, close) * 0.99,
            "close": close,
            "volume": rng.integers(100000, 1000000, len(dates)),
        })

    demo_config = RankingConfig(
        n_estimators=20, num_leaves=7, min_child_samples=10, early_stopping_rounds=5
    )
    demo_result = train_ranker(demo_map, demo_config)
    latest = rank_latest(demo_map, demo_result["model"])
    print("测试集指标：", demo_result["metrics"]["test"])
    print("最新横截面排名：\n", latest[["rank", "date", "code", "score"]].head())
    assert not latest.empty
    assert np.isfinite(latest["score"]).all()
    assert demo_result["metrics"]["test"]["periods"] > 0
    print("ml_model.py 自测通过；排序结果仅研究用途")
