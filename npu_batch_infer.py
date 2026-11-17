# -*- coding: utf-8 -*-
"""OpenVINO NPU 独立批量推理：FP16 IR 逐日 replay，逐票输出仓位与 NAV。

- 输入：1208 维标准化状态（60×20 因子 + 8 维账户），由 load_stock_data 按
  120 日滚动 z-score 计算，依赖完整历史（obv 为 cumsum，不可截断）；
- 推理：model.xml/model.bin（FP16）编译到 NPU，同一步把多只股票拼成动态 batch；
- 参考：同一 FP16 IR 在 CPU 上并行 replay，逐日核对动作/仓位/NAV；
- 判定：仓位与 NAV 相对误差 < 1e-3 视为一致（--threshold 可改）。
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import openvino as ov
import pandas as pd
import yaml

from data_loader_v2 import MarketView, get_feature_columns, load_stock_data
from trading_env_v2 import TradingEnvV2

ROOT = Path(__file__).resolve().parent


def build_view(stock, feature_columns, rows):
    frame = stock.frame.tail(rows).reset_index(drop=True)
    return MarketView(
        stock.code,
        frame["date"].to_numpy(),
        np.ascontiguousarray(frame[feature_columns].to_numpy(dtype=np.float32)),
        frame["close"].to_numpy(dtype=np.float64),
        frame["ma20"].to_numpy(dtype=np.float64),
        frame["trend_momentum20"].to_numpy(dtype=np.float64),
        frame["raw_volatility60"].to_numpy(dtype=np.float64),
        frame["volatility_median60"].to_numpy(dtype=np.float64),
    )


def resolve_device(core, requested, require_npu):
    available = list(core.available_devices)
    if requested in available:
        return requested
    if requested == "NPU" and require_npu:
        raise SystemExit("未检测到 NPU，可用设备：%s" % ",".join(available))
    fallback = "CPU" if "CPU" in available else available[0]
    print("警告：未检测到 %s，回退到 %s（可用：%s）" % (
        requested, fallback, ",".join(available)))
    return fallback


def compile_ir(core, path, device, batch):
    """优先动态 batch；设备不支持时改成静态 batch 再编译。"""
    model = core.read_model(str(path))
    try:
        return core.compile_model(model, device), "dynamic"
    except Exception as exc:
        print("动态 batch 编译失败（%s），改用静态 batch=%d" % (exc, batch))
        model.reshape({model.input(0): ov.PartialShape([batch, 1208])})
        return core.compile_model(model, device), "static"


def max_position_for(code, env_config):
    override = env_config.get("risk_overrides", {}).get(str(code), {})
    return float(override.get("max_position", env_config.get("max_position", 1.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=63)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--model", default="ir_model/model.xml")
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--require-npu", action="store_true",
                        help="未检测到 NPU 时直接退出，不回退 CPU")
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--output", default="npu_output")
    args = parser.parse_args()
    as_of = args.as_of or datetime.now().strftime("%Y%m%d")

    with open(ROOT / "config_v2.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["data"]["cache_dir"] = "cache_shadow"
    config["data"]["end_date"] = as_of
    config["data"]["refresh"] = bool(args.refresh)
    feature_count = int(config["network"].get("feature_count", 20))
    lookback = int(config["network"].get("lookback", 60))
    feature_columns = get_feature_columns(feature_count)

    stocks = load_stock_data(config["data"])
    views = {}
    for code, stock in stocks.items():
        if len(stock.frame) < lookback + args.days:
            print("%s 历史不足 %d 天，跳过" % (code, lookback + args.days))
            continue
        views[code] = build_view(stock, feature_columns, lookback + args.days)
    if not views:
        raise SystemExit("没有可推理的股票")

    codes = list(views)
    batch = len(codes)
    core = ov.Core()
    device = resolve_device(core, args.device, args.require_npu)
    model_path = ROOT / args.model

    npu_compiled, batch_mode = compile_ir(core, model_path, device, batch)
    ref_compiled, _ = compile_ir(core, model_path, "CPU", batch)
    print("IR=%s  device=%s  batch=%d(%s)  codes=%s" % (
        model_path.name, device, batch, batch_mode, ",".join(codes)))

    envs_npu = {code: TradingEnvV2(view, config["env"]) for code, view in views.items()}
    envs_ref = {code: TradingEnvV2(view, config["env"]) for code, view in views.items()}
    states_npu = {code: np.asarray(env.reset(), dtype=np.float32)
                  for code, env in envs_npu.items()}
    states_ref = {code: np.asarray(env.reset(), dtype=np.float32)
                  for code, env in envs_ref.items()}
    actions_npu = {code: [] for code in codes}
    actions_ref = {code: [] for code in codes}

    latencies = []
    calls = 0
    while any(not env.done for env in envs_npu.values()):
        active = [code for code in codes if not envs_npu[code].done]
        batch_npu = np.stack([states_npu[code] for code in active])
        batch_ref = np.stack([states_ref[code] for code in active])
        start = time.perf_counter()
        out_npu = np.asarray(npu_compiled([batch_npu])[0]).reshape(-1)
        latencies.append((time.perf_counter() - start) * 1000)
        out_ref = np.asarray(ref_compiled([batch_ref])[0]).reshape(-1)
        calls += 1
        for index, code in enumerate(active):
            action = float(out_npu[index])
            action_ref = float(out_ref[index])
            actions_npu[code].append(action)
            actions_ref[code].append(action_ref)
            states_npu[code], _, _, _ = envs_npu[code].step(action)
            states_ref[code], _, _, _ = envs_ref[code].step(action_ref)

    run_dir = ROOT / args.output / ("run_" + datetime.now().strftime("%Y%m%d"))
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries, daily_frames = [], []
    for code in codes:
        env_npu, env_ref = envs_npu[code], envs_ref[code]
        nav_npu = np.asarray(env_npu.nav_history, dtype=float)
        pos_npu = np.asarray(env_npu.position_history, dtype=float)
        nav_ref = np.asarray(env_ref.nav_history, dtype=float)
        pos_ref = np.asarray(env_ref.position_history, dtype=float)

        # 以决策步数对齐各曲线（不同环境的历史数组可能差一位）
        steps = min(
            len(actions_npu[code]), len(actions_ref[code]),
            len(nav_npu) - 1, len(pos_npu) - 1,
            len(nav_ref) - 1, len(pos_ref) - 1,
        )
        nav_npu, nav_ref = nav_npu[:steps + 1], nav_ref[:steps + 1]
        pos_npu, pos_ref = pos_npu[:steps + 1], pos_ref[:steps + 1]

        prices = np.asarray(env_npu.price_history, dtype=float)[:steps + 1]
        dates = [str(value)[:10] for value in env_npu.date_history][:steps + 1]
        bh = prices / prices[0]
        strategy_dd = float(np.max(1 - nav_npu / np.maximum.accumulate(nav_npu)))

        action_npu = np.asarray(actions_npu[code], dtype=float)[:steps]
        action_ref = np.asarray(actions_ref[code], dtype=float)[:steps]
        action_diff = np.abs(action_npu - action_ref)
        pos_diff = np.abs(pos_npu - pos_ref)
        nav_diff = np.abs(nav_npu - nav_ref)
        nav_rel = nav_diff / np.maximum(np.abs(nav_npu), 1e-12)

        latest = float(np.clip(
            np.tanh(action_npu[-1]), 0.0, max_position_for(code, config["env"])
        ))
        summary = {
            "code": code,
            "days": int(len(nav_npu) - 1),
            "start": dates[1],
            "end": dates[-1],
            "latest_target": latest,
            "avg_position": float(pos_npu[1:].mean()),
            "nav": float(nav_npu[-1]),
            "bh_nav": float(bh[-1]),
            "excess": float(nav_npu[-1] - bh[-1]),
            "max_drawdown": strategy_dd,
            "trades": int(env_npu.trade_count),
            "closed_trades": int(len(env_npu.trade_returns)),
            "max_action_diff": float(action_diff.max()) if len(action_diff) else 0.0,
            "max_position_diff": float(pos_diff.max()) if len(pos_diff) else 0.0,
            "max_nav_diff": float(nav_diff.max()) if len(nav_diff) else 0.0,
            "max_nav_rel_diff": float(nav_rel.max()) if len(nav_rel) else 0.0,
        }
        summaries.append(summary)
        columns = {
            "code": code,
            "date": dates[1:],
            "action_npu": action_npu,
            "action_ref": action_ref,
            "position_npu": pos_npu[1:],
            "position_ref": pos_ref[1:],
            "nav_npu": nav_npu[1:],
            "nav_ref": nav_ref[1:],
            "bh_nav": bh[1:],
            "action_diff": action_diff,
            "position_diff": pos_diff[1:],
            "nav_rel_diff": nav_rel[1:],
        }
        width = min(
            len(value) for key, value in columns.items() if key != "code"
        )
        columns = {
            key: (value if key == "code" else value[:width])
            for key, value in columns.items()
        }
        daily_frames.append(pd.DataFrame(columns))

    summary_frame = pd.DataFrame(summaries)
    daily_frame = pd.concat(daily_frames, ignore_index=True)
    summary_frame.to_csv(run_dir / "npu_summary.csv", index=False, encoding="utf-8-sig")
    daily_frame.to_csv(run_dir / "npu_daily_all.csv", index=False, encoding="utf-8-sig")

    max_pos = float(summary_frame["max_position_diff"].max())
    max_nav_rel = float(summary_frame["max_nav_rel_diff"].max())
    passed = max_pos < args.threshold and max_nav_rel < args.threshold
    stable_latencies = latencies[3:] if len(latencies) > 3 else latencies
    report = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": as_of,
        "model": str(model_path.relative_to(ROOT)),
        "device": device,
        "batch_mode": batch_mode,
        "batch_size": batch,
        "calls": calls,
        "latency_ms": {
            "mean": float(np.mean(stable_latencies)),
            "p95": float(np.quantile(stable_latencies, 0.95)),
            "max": float(np.max(stable_latencies)),
        },
        "threshold": args.threshold,
        "consistency": {
            "max_position_diff": max_pos,
            "max_nav_rel_diff": max_nav_rel,
            "passed": passed,
        },
        "stocks": summaries,
    }
    with open(run_dir / "npu_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    view_frame = summary_frame[[
        "code", "latest_target", "avg_position", "nav", "bh_nav", "excess",
        "trades", "closed_trades", "max_position_diff", "max_nav_rel_diff",
    ]].copy()
    for column in ["latest_target", "avg_position", "excess"]:
        view_frame[column] = view_frame[column].map(lambda v: "%.2f%%" % (100 * v))
    for column in ["max_position_diff", "max_nav_rel_diff"]:
        view_frame[column] = view_frame[column].map(lambda v: "%.3e" % v)
    view_frame = view_frame.rename(columns={
        "latest_target": "最新目标仓位", "avg_position": "平均仓位",
        "excess": "超额", "max_position_diff": "仓位误差",
        "max_nav_rel_diff": "NAV相对误差",
    })
    print(view_frame.to_string(index=False))
    print("一致性：max_position_diff=%.3e max_nav_rel_diff=%.3e -> %s（阈值 %.0e）" % (
        max_pos, max_nav_rel, "PASS" if passed else "FAIL", args.threshold))
    print("推理：%d 次 batch=%d，平均 %.3f ms/次，p95 %.3f ms" % (
        calls, batch, report["latency_ms"]["mean"], report["latency_ms"]["p95"]))
    print("输出：%s" % run_dir.resolve())


if __name__ == "__main__":
    main()
