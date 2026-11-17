# -*- coding: utf-8 -*-
"""
因子批量筛查（V1.4 Step B1）。

对因子库做一次性批量有效性筛查，产出「保留 / 反向 / 剔除」清单：
- IC/IR：调仓日因子值与未来 N 日收益的截面秩相关；
- 稳健性：样本按时间前 60% 训练、后 40% 验证，两段方向必须一致；
- IC 衰减：未来 1/5/10/20 日 IC，判断因子衰减速度；
- 分层：5 层未来收益、多空差、单调性；
- 冗余：因子间截面相关系数均值，|r| 超阈值的组内只保留 |t| 最大者。

防前视约束：
- 因子值只使用调仓日 t 及之前行情（compute_factor_series 为时序递推）；
- 未来收益使用 t 日收盘到 t+N 日收盘，严格滞后；
- 训练段用于筛选，验证段只用于复核，不参与因子取舍。

仅研究用途，不构成投资建议。
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.factor_engine import FACTOR_REGISTRY, compute_factor_series
from core.factor_analysis import load_all_daily, get_rebalance_dates
from core.factor_process import deduplicate

DISCLAIMER = "仅研究用途，不构成投资建议"

# 筛查结果落盘目录
SCREEN_DIR = db.DATA_DIR / "factor_screen"

# IC 衰减考察的未来收益期
FORWARD_DAYS = (1, 5, 10, 20)


# ---------------------------------------------------------------------------
# 面板与截面构建
# ---------------------------------------------------------------------------

def build_panel(df, factor_names, forward_days=FORWARD_DAYS):
    """
    对单只股票一次性计算全部因子时序与未来收益。

    df: 按日期升序的日线（需含 date/high/low/close/volume）
    返回 DataFrame：index=date，columns=因子名 + fwdN（未来 N 日收益）
    """
    x = df.set_index("date")
    panel = pd.DataFrame(index=x.index)
    for name in factor_names:
        panel[name] = compute_factor_series(x, name)
    close = x["close"]
    for n in forward_days:
        panel["fwd%d" % n] = close.shift(-n) / close.replace(0, np.nan) - 1
    return panel


def prepare_panels(codes, factor_names, fq="qfq", min_rows=120):
    """批量加载日线并构建因子面板，返回 dict[code] -> DataFrame。"""
    data_map = load_all_daily(codes, fq=fq)
    panels = {}
    for code, df in data_map.items():
        if df is None or len(df) < min_rows:
            continue
        panels[code] = build_panel(df, factor_names)
    return panels, data_map


def build_cross_sections(panels, dates):
    """按调仓日取出横截面，返回 DataFrame 列表（index=code）。"""
    frames = []
    for date in dates:
        rec = {}
        for code, panel in panels.items():
            if date in panel.index:
                rec[code] = panel.loc[date]
        if rec:
            frames.append(pd.DataFrame(rec).T)
    return frames


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------

def _ic_series(frames, factor_name, ret_col="fwd5", min_count=30):
    """计算单个因子的 IC 时序（Spearman）。"""
    ics = []
    for frame in frames:
        sub = frame[[factor_name, ret_col]].dropna()
        if len(sub) < min_count:
            continue
        value = sub[factor_name].corr(sub[ret_col], method="spearman")
        if pd.notna(value):
            ics.append(float(value))
    return pd.Series(ics, dtype=float)


def _summarize_ic(ics):
    """汇总 IC 时序：均值、标准差、IR、t 值、胜率、样本数。"""
    if ics is None or ics.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ir": np.nan,
                "t_stat": np.nan, "positive_rate": np.nan, "count": 0}
    mean = float(ics.mean())
    std = float(ics.std(ddof=1)) if len(ics) > 1 else np.nan
    ir = mean / std if std and std > 0 else np.nan
    t_stat = mean / (std / np.sqrt(len(ics))) if std and std > 0 else np.nan
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ir": ir,
        "t_stat": t_stat,
        "positive_rate": float((ics > 0).mean()),
        "count": int(len(ics)),
    }


def _layer_stats(frames, factor_name, n_layers=5, ret_col="fwd5", min_count=None):
    """分层统计：各层平均未来收益、多空差、单调性。"""
    min_count = min_count or max(30, n_layers * 6)
    buckets = {i: [] for i in range(1, n_layers + 1)}
    for frame in frames:
        sub = frame[[factor_name, ret_col]].dropna()
        if len(sub) < min_count:
            continue
        sub = sub.sort_values(factor_name)
        try:
            labels = pd.qcut(sub[factor_name], n_layers, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        for i in range(1, n_layers + 1):
            values = sub[ret_col][labels == i]
            if len(values):
                buckets[i].append(float(values.mean()))
    means = {i: (float(np.mean(v)) if v else np.nan) for i, v in buckets.items()}
    long_short = means[n_layers] - means[1]
    idx = pd.Series(list(means.keys()), dtype=float)
    val = pd.Series(list(means.values()), dtype=float)
    monotonicity = float(idx.corr(val, method="spearman")) if val.notna().sum() >= 2 else np.nan
    return {"layer_means": means, "long_short": long_short, "monotonicity": monotonicity}


def mean_correlation(frames, factor_names, min_count=30):
    """因子间截面相关系数均值矩阵。"""
    total = None
    count = 0
    for frame in frames:
        sub = frame[factor_names].dropna()
        if len(sub) < min_count:
            continue
        corr = sub.corr(method="spearman")
        total = corr if total is None else total.add(corr, fill_value=0)
        count += 1
    if total is None or count == 0:
        return pd.DataFrame(index=factor_names, columns=factor_names, dtype=float)
    return total / count


# ---------------------------------------------------------------------------
# 统计明细与判定（供筛查与分段回测复用）
# ---------------------------------------------------------------------------

def compute_details(frames, factor_names, train_frames=None, valid_frames=None,
                    forward_n=5, n_layers=5, include_decay=True, include_layers=True):
    """
    计算因子的完整统计明细。

    frames: 全样本截面列表
    train_frames / valid_frames: 训练段与验证段截面列表（None 表示与 frames 相同 / 空）
    forward_n: 主检验未来收益期
    n_layers: 分层数
    include_decay / include_layers: 是否计算 IC 衰减与分层（分段走查可关闭以提速）

    返回 dict {name: {desc, group, full, train, valid, decay, layer_means, long_short, monotonicity}}
    """
    train_frames = frames if train_frames is None else train_frames
    valid_frames = [] if valid_frames is None else valid_frames
    details = {}
    for name in factor_names:
        decay = {}
        if include_decay:
            for n in FORWARD_DAYS:
                decay[n] = _summarize_ic(_ic_series(frames, name, "fwd%d" % n))["ic_mean"]
        item = {
            "desc": FACTOR_REGISTRY[name]["desc"],
            "group": FACTOR_REGISTRY[name].get("group", ""),
            "full": _summarize_ic(_ic_series(frames, name, "fwd%d" % forward_n)),
            "train": _summarize_ic(_ic_series(train_frames, name, "fwd%d" % forward_n)),
            "valid": _summarize_ic(_ic_series(valid_frames, name, "fwd%d" % forward_n)),
            "decay": decay,
        }
        if include_layers:
            item.update(_layer_stats(frames, name, n_layers=n_layers,
                                     ret_col="fwd%d" % forward_n))
        details[name] = item
    return details


def compute_direction_stability(frames, factor_names, n_windows=4, forward_n=5,
                                min_count=10):
    """
    因子方向稳定性。

    把训练窗口按时间等分为 n_windows 个子窗口，逐个子窗口独立计算 IC，
    统计 IC 方向的同号比例（一致率）。一致率越高，说明因子方向越不依赖单一时段。

    frames: 训练段截面列表（按时间升序）
    返回 dict {name: {ics, signs, consistency, direction, valid_windows, windows}}
    """
    frames = list(frames)
    if not frames:
        return {}
    n_windows = max(2, min(int(n_windows), len(frames)))
    size = len(frames) / float(n_windows)
    blocks = [frames[int(round(i * size)):int(round((i + 1) * size))]
              for i in range(n_windows)]
    blocks = [b for b in blocks if b]
    result = {}
    for name in factor_names:
        ics, signs = [], []
        for block in blocks:
            ic_mean = _summarize_ic(
                _ic_series(block, name, "fwd%d" % forward_n, min_count=min_count))["ic_mean"]
            if pd.isna(ic_mean):
                ics.append(None)
                signs.append(0)
                continue
            ics.append(float(ic_mean))
            signs.append(1 if ic_mean > 0 else (-1 if ic_mean < 0 else 0))
        valid = [s for s in signs if s != 0]
        if valid:
            positive = sum(1 for s in valid if s > 0)
            negative = len(valid) - positive
            direction = 1 if positive >= negative else -1
            consistency = max(positive, negative) / float(len(valid))
        else:
            direction, consistency = 1, 0.0
        result[name] = {
            "ics": ics,
            "signs": signs,
            "consistency": float(consistency),
            "direction": int(direction),
            "valid_windows": len(valid),
            "windows": len(blocks),
        }
    return result


def judge_factors(details, min_abs_ic=0.02, min_abs_t=2.0, use_validation=True,
                  stability=None, min_consistency=0.7):
    """
    判定每个因子的状态与方向。

    use_validation=False 时只依据训练段（分段走查回测不可使用未来数据）。
    stability: compute_direction_stability 的结果；提供时「方向稳定性」作为前置门槛，
               一致率低于 min_consistency 直接剔除，方向取多数子窗口的符号。
    返回 dict {name: {status: keep/reverse/drop, direction: 1/-1, reason}}
    """
    decisions = {}
    for name, item in details.items():
        train = item["train"]
        valid = item["valid"]
        ic_mean = train["ic_mean"]
        t_stat = train["t_stat"]
        stable_info = (stability or {}).get(name)
        need_windows = 0
        if stable_info:
            need_windows = max(2, int(round(stable_info.get("windows", 0) * 0.7)))
        direction = 1
        if train["count"] < 20:
            status, reason = "drop", "训练段有效样本不足（%d）" % train["count"]
        elif pd.isna(ic_mean) or abs(ic_mean) < min_abs_ic:
            status, reason = "drop", "IC 强度不足（%.4f）" % (ic_mean if pd.notna(ic_mean) else np.nan)
        elif pd.isna(t_stat) or abs(t_stat) < min_abs_t:
            status, reason = "drop", "t 值不显著（%.2f）" % (t_stat if pd.notna(t_stat) else np.nan)
        elif stable_info and (stable_info["consistency"] < min_consistency
                              or stable_info["valid_windows"] < need_windows):
            status, reason = "drop", "方向不稳定（一致率 %.0f%%，有效子窗口 %d/%d）" % (
                stable_info["consistency"] * 100, stable_info["valid_windows"],
                stable_info["windows"])
        elif use_validation and pd.notna(valid["ic_mean"]) and np.sign(valid["ic_mean"]) != np.sign(ic_mean):
            status, reason = "drop", "前后段方向不一致（训练 %.4f / 验证 %.4f）" % (
                ic_mean, valid["ic_mean"])
        elif use_validation and pd.notna(valid["ic_mean"]) and abs(valid["ic_mean"]) < min_abs_ic * 0.5:
            status, reason = "drop", "验证段衰减过大（%.4f）" % valid["ic_mean"]
        else:
            base_sign = 1 if ic_mean > 0 else -1
            direction = stable_info["direction"] if stable_info else base_sign
            status = "keep" if direction > 0 else "reverse"
            parts = ["训练段 %.4f" % ic_mean]
            if stable_info:
                parts.append("方向一致率 %.0f%%" % (stable_info["consistency"] * 100))
            if use_validation:
                parts.append("验证段 %.4f" % valid["ic_mean"])
            reason = " / ".join(parts)
        decisions[name] = {"status": status, "direction": direction, "reason": reason}
    return decisions


# ---------------------------------------------------------------------------
# 筛查主流程
# ---------------------------------------------------------------------------

def screen_factors(factor_names=None, codes=None, start_date=None, end_date=None,
                   freq="W", forward_n=5, n_layers=5, train_ratio=0.6,
                   min_abs_ic=0.02, min_abs_t=2.0, corr_threshold=0.7,
                   n_stability_windows=4, min_consistency=0.7, check_stability=True,
                   fq="qfq", progress_cb=None):
    """
    批量筛查因子。

    参数：
        factor_names: 待筛因子，默认全部已注册因子
        codes: 股票池，默认本地全部有 qfq 数据的股票
        start_date / end_date: 检验区间
        freq: 调仓频率 "W" / "2W" / "M"
        forward_n: 主检验使用的未来收益天数
        n_layers: 分层数
        train_ratio: 训练段占比（按调仓日数量切分）
        min_abs_ic: 训练段 |IC| 下限
        min_abs_t: 训练段 |t 值| 下限
        corr_threshold: 冗余判定阈值（|r| >= 该值视为重复）
        n_stability_windows: 方向稳定性检验的子窗口数（训练段内等分）
        min_consistency: 方向一致率下限（默认 0.7）
        check_stability: 是否启用方向稳定性前置门槛
        progress_cb: 进度回调 callback(done, total, text)

    返回 dict：meta / factors / correlation / decisions / selected / stability
    """
    factor_names = factor_names or list(FACTOR_REGISTRY.keys())
    for name in factor_names:
        if name not in FACTOR_REGISTRY:
            raise ValueError("未注册的因子：%s" % name)

    if codes is None:
        codes = [r["code"] for r in db.query_all(
            "SELECT DISTINCT code FROM daily_bars WHERE fq=? ORDER BY code", (fq,))]

    panels, data_map = prepare_panels(codes, factor_names, fq=fq)
    usable = list(panels.keys())
    if not usable:
        raise ValueError("无可用日线数据")

    dates = get_rebalance_dates(usable, freq=freq, fq=fq)
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    if len(dates) < 20:
        raise ValueError("调仓日不足，无法筛查")

    if progress_cb:
        progress_cb(0, 4, "构建截面")
    frames = build_cross_sections(panels, dates)

    split = max(1, int(len(frames) * train_ratio))
    train_frames = frames[:split]
    valid_frames = frames[split:]
    if not valid_frames:
        raise ValueError("验证段为空，请延长检验区间或降低 train_ratio")

    if progress_cb:
        progress_cb(1, 4, "计算 IC / 分层 / 衰减")

    details = compute_details(frames, factor_names, train_frames=train_frames,
                              valid_frames=valid_frames, forward_n=forward_n,
                              n_layers=n_layers)

    if progress_cb:
        progress_cb(2, 4, "计算方向稳定性")
    stability = compute_direction_stability(
        train_frames, factor_names, n_windows=n_stability_windows,
        forward_n=forward_n) if check_stability else {}

    if progress_cb:
        progress_cb(3, 4, "计算相关性并判定")
    corr_mean = mean_correlation(frames, factor_names)

    # 1) 单因子判定：强度 + 显著性 + 方向稳定性（前置门槛）+ 前后段一致性
    decisions = judge_factors(details, min_abs_ic=min_abs_ic, min_abs_t=min_abs_t,
                              stability=stability if check_stability else None,
                              min_consistency=min_consistency)

    # 2) 冗余剔除：组内保留 |t| 最大者（方向对齐后比较相关性）
    candidates = [n for n in factor_names
                  if decisions[n]["status"] in ("keep", "reverse")]
    scores = {n: details[n]["train"]["t_stat"] for n in candidates}
    directions = {n: decisions[n]["direction"] for n in candidates}
    sub_corr = corr_mean.loc[candidates, candidates] if candidates else corr_mean
    selected, dropped = deduplicate(sub_corr, scores, directions=directions,
                                    threshold=corr_threshold)
    for name, (kept, value) in dropped.items():
        decisions[name]["status"] = "drop"
        decisions[name]["reason"] = "与 %s 冗余（r=%.3f）" % (kept, value)

    result = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "codes": len(usable),
            "freq": freq,
            "forward_n": forward_n,
            "start": dates[0],
            "end": dates[-1],
            "periods": len(frames),
            "train_periods": len(train_frames),
            "valid_periods": len(valid_frames),
            "train_ratio": train_ratio,
            "min_abs_ic": min_abs_ic,
            "min_abs_t": min_abs_t,
            "corr_threshold": corr_threshold,
            "n_stability_windows": n_stability_windows,
            "min_consistency": min_consistency,
            "check_stability": bool(check_stability),
            "disclaimer": DISCLAIMER,
        },
        "factors": details,
        "correlation": _corr_to_dict(corr_mean),
        "decisions": decisions,
        "selected": selected,
        "stability": stability,
    }
    if progress_cb:
        progress_cb(4, 4, "筛查完成")
    return result


def _corr_to_dict(corr_mean):
    """把相关性矩阵转成嵌套 dict，便于 JSON 落盘。"""
    if corr_mean is None or corr_mean.empty:
        return {}
    data = {}
    for row in corr_mean.index:
        data[row] = {col: _safe_float(corr_mean.loc[row, col]) for col in corr_mean.columns}
    return data


def _safe_float(value):
    """把值转为 JSON 可序列化的 float 或 None。"""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


def save_screen_result(result, path=None):
    """把筛查结果落盘为 JSON，返回文件路径。"""
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        meta = result["meta"]
        path = SCREEN_DIR / ("screen_%s_%s_%s.json" % (
            meta["start"].replace("-", ""), meta["end"].replace("-", ""), meta["codes"]))
    path = Path(path)
    payload = json.loads(json.dumps(result, ensure_ascii=False, default=_default_json))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _default_json(obj):
    """json.dumps 的兜底转换：numpy / pandas 标量。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def save_stability_report(result, path=None):
    """
    把方向一致性报告落盘为 CSV + JSON，返回 (csv_path, json_path)。

    字段：因子名、各子窗口 IC、方向符号、一致率、是否方向稳定、是否入选、判定原因。
    """
    stability = result.get("stability") or {}
    if not stability:
        raise ValueError("结果中没有方向稳定性数据，请启用 check_stability=True")
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)

    n_windows = max(len(v.get("ics", [])) for v in stability.values()) if stability else 0
    rows = []
    for name, info in stability.items():
        decision = (result.get("decisions") or {}).get(name, {})
        row = {
            "factor": name,
            "group": (result.get("factors", {}).get(name, {}) or {}).get("group", ""),
        }
        for i in range(n_windows):
            ics = info.get("ics", [])
            row["window%d_ic" % (i + 1)] = ics[i] if i < len(ics) else None
            row["window%d_sign" % (i + 1)] = info.get("signs", [0] * n_windows)[i]
        row.update({
            "consistency": info.get("consistency"),
            "direction": info.get("direction"),
            "valid_windows": info.get("valid_windows"),
            "stable": bool(info.get("consistency", 0) >= result["meta"]["min_consistency"]),
            "selected": name in (result.get("selected") or []),
            "status": decision.get("status", ""),
            "reason": decision.get("reason", ""),
        })
        rows.append(row)
    frame = pd.DataFrame(rows)

    if path is None:
        meta = result["meta"]
        path = SCREEN_DIR / ("dir_stability_%s_%s_%s.csv" % (
            meta["start"].replace("-", ""), meta["end"].replace("-", ""), meta["codes"]))
    csv_path = Path(path)
    if csv_path.suffix.lower() != ".csv":
        csv_path = csv_path.with_suffix(".csv")
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = csv_path.with_suffix(".json")
    payload = {
        "meta": result["meta"],
        "windows": n_windows,
        "items": rows,
    }
    payload = json.loads(json.dumps(payload, ensure_ascii=False, default=_default_json))
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def print_report(result):
    """打印筛查报告。"""
    meta = result["meta"]
    print("\n=== 因子筛查报告 ===")
    print("股票池 %d 只 | %s 频 | 未来 %d 日 | %s ~ %s | %d 期（训练 %d / 验证 %d）" % (
        meta["codes"], meta["freq"], meta["forward_n"], meta["start"], meta["end"],
        meta["periods"], meta["train_periods"], meta["valid_periods"]))
    print("-" * 126)
    print("%-20s %-8s %8s %7s %8s %8s %8s %8s %8s %8s" % (
        "因子", "分组", "IC训练", "IR", "t值", "IC验证", "一致率", "稳定", "IC_5日", "多空%"))
    print("-" * 126)
    stability = result.get("stability") or {}
    for name, item in result["factors"].items():
        train = item["train"]
        valid = item["valid"]
        info = stability.get(name) or {}
        consistency = info.get("consistency")
        stable = "-" if not info else ("是" if consistency >= meta["min_consistency"] else "否")
        print("%-20s %-8s %8.4f %7.3f %8.2f %8.4f %8.2f %8s %9.4f %8.3f" % (
            name, item.get("group", ""), train["ic_mean"], train["ir"], train["t_stat"],
            valid["ic_mean"],
            consistency if consistency is not None else np.nan,
            stable,
            (item.get("decay") or {}).get(5, np.nan),
            (item.get("long_short") or np.nan) * 100))
    print("-" * 126)
    groups = {"keep": [], "reverse": [], "drop": []}
    for name, dec in result["decisions"].items():
        groups[dec["status"]].append(name)
    print("保留（正向）：%s" % (", ".join(groups["keep"]) or "无"))
    print("反向（取负）：%s" % (", ".join(groups["reverse"]) or "无"))
    print("剔除：%s" % (", ".join(groups["drop"]) or "无"))
    print("最终入选：%s（%d 个）" % (", ".join(result["selected"]) or "无", len(result["selected"])))


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(DISCLAIMER)

    # 0. 合成数据自测：子窗口方向翻转场景
    np.random.seed(11)

    def _make_block(sign, n_dates=6, n_stocks=40):
        """构造一批截面：因子与未来收益按 sign 同向/反向。"""
        block = []
        for _ in range(n_dates):
            factor = np.random.randn(n_stocks)
            ret = sign * 0.05 * factor + np.random.randn(n_stocks) * 0.004
            block.append(pd.DataFrame({"x": factor, "fwd5": ret},
                                      index=["%03d" % i for i in range(n_stocks)]))
        return block

    # x：4 个子窗口中 3 个正向、1 个反向 → 一致率 0.75，应判定为稳定
    frames_syn = _make_block(1) + _make_block(1) + _make_block(1) + _make_block(-1)
    stability = compute_direction_stability(frames_syn, ["x"], n_windows=4)
    print("\n合成数据方向稳定性:", {k: {"consistency": round(v["consistency"], 2),
                                    "direction": v["direction"]} for k, v in stability.items()})
    assert stability["x"]["consistency"] == 0.75, "一致率应为 0.75"
    assert stability["x"]["direction"] == 1

    # 方向交替的因子：一致率 0.5，应被稳定性门槛剔除
    frames_flip = _make_block(1) + _make_block(-1) + _make_block(1) + _make_block(-1)
    stability_flip = compute_direction_stability(frames_flip, ["x"], n_windows=4)
    fake_details = {"x": {"train": {"ic_mean": 0.05, "t_stat": 3.0, "count": 100},
                          "valid": {"ic_mean": np.nan}}}
    dec = judge_factors(fake_details, stability=stability_flip, min_consistency=0.7)
    print("方向交替因子的判定:", dec["x"])
    assert dec["x"]["status"] == "drop", "方向不稳定应被剔除"
    dec2 = judge_factors(fake_details, stability=stability, min_consistency=0.7)
    assert dec2["x"]["status"] == "keep" and dec2["x"]["direction"] == 1
    print("方向稳定性门槛逻辑校验通过")

    db.init_db()

    rows = db.query_all("SELECT DISTINCT code FROM daily_bars WHERE fq='qfq' ORDER BY code LIMIT 50")
    codes = [r["code"] for r in rows]
    print("测试股票数:", len(codes))

    names = ["ma5", "bias_20", "ret_risk_20", "price_position_60", "atr_pct", "ret_1"]

    # 1. 面板与单点计算一致性（验证无未来函数、口径一致）
    panels, data_map = prepare_panels(codes, names)
    code = list(panels.keys())[0]
    date = panels[code].index[-40]
    df_hist = data_map[code]
    df_hist = df_hist[df_hist["date"] <= date].reset_index(drop=True)
    from core.factor_engine import compute_factor
    for name in names:
        expected = compute_factor(df_hist, name)
        actual = float(panels[code].loc[date, name])
        if pd.notna(expected):
            assert abs(expected - actual) < 1e-9, "%s 面板值与截面值不一致：%s vs %s" % (
                name, expected, actual)
    print("面板与单点计算一致性校验通过")

    # 2. 筛查流程（含方向稳定性前置门槛）
    result = screen_factors(factor_names=names, codes=codes, start_date="2023-01-01",
                            freq="W", forward_n=5, train_ratio=0.6,
                            n_stability_windows=4, min_consistency=0.7)
    print_report(result)

    # 3. 落盘（筛查结果 + 方向一致性报告 CSV/JSON）
    path = save_screen_result(result)
    csv_path, json_path = save_stability_report(result)
    print("\n结果已保存:", path)
    print("方向一致性报告:", csv_path)
    print("方向一致性报告:", json_path)

    assert result["meta"]["periods"] >= 20
    assert set(result["decisions"].keys()) == set(names)
    assert set(result["stability"].keys()) == set(names)
    for name, item in result["factors"].items():
        assert item["train"]["count"] > 0, "%s 训练段无有效 IC" % name
    assert csv_path.exists() and json_path.exists()
    print("\nfactor_screen.py 自测通过；%s" % DISCLAIMER)
