# -*- coding: utf-8 -*-
"""盘后影子推理：用 OpenVINO IR 逐日决策，并与 PyTorch 参考对齐。

- 窗口：最近 N 个交易日（默认 63，约 3 个月）+ 前置 60 天 lookback；
- 决策：IR 输出 action_mean，经与训练一致的环境执行（目标仓位、风控、成本）；
- 对齐：同一天并行跑 PyTorch 确定性策略，记录动作/仓位/净值偏差；
- 监控：600519 上行捕获率、000001 趋势参与度、300750 超额收益。
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import openvino as ov
import pandas as pd
import torch
import yaml

from data_loader_v2 import MarketView, get_feature_columns, load_stock_data
from live.event_risk_filter import EventRiskFilter, BLACKLIST, WATCH, NORMAL
from networks_v2 import PPOAgent
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


def run_stock(stock, agent, compiled, config, feature_columns, lookback, days):
    view = build_view(stock, feature_columns, lookback + days)
    env = TradingEnvV2(view, config["env"])
    state = env.reset()

    ir_actions, pt_actions = [], []
    while not env.done:
        ir_action = float(np.asarray(compiled([state.astype(np.float32)[None]])[0])[0])
        pt_action = float(agent.select_action(state, deterministic=True)[0])
        ir_actions.append(ir_action)
        pt_actions.append(pt_action)
        state, _, _, _ = env.step(ir_action)

    nav = np.asarray(env.nav_history, dtype=float)
    prices = np.asarray(env.price_history, dtype=float)
    positions = np.asarray(env.position_history, dtype=float)
    dates = [str(value)[:10] for value in env.date_history]

    bh = prices / prices[0]
    bh_peak = np.maximum.accumulate(bh)
    bh_dd = float(np.max(1 - bh / bh_peak))
    nav_peak = np.maximum.accumulate(nav)
    strategy_dd = float(np.max(1 - nav / nav_peak))

    strategy_returns = np.diff(nav) / nav[:-1]
    bh_returns = np.diff(prices) / prices[:-1]
    trend = np.array([
        view.close[index] > view.ma20[index] and view.momentum20[index] > 0
        for index in range(lookback - 1, len(view))
    ])
    trend_strategy = float(np.prod(1 + strategy_returns[trend]) - 1) if trend.any() else 0.0
    trend_bh = float(np.prod(1 + bh_returns[trend]) - 1) if trend.any() else 0.0
    capture = trend_strategy / trend_bh if trend_bh > 0 else None

    daily = pd.DataFrame({
        "date": dates[1:],
        "action_ir": ir_actions,
        "action_pt": pt_actions,
        "position": positions[1:],
        "nav": nav[1:],
        "bh_nav": bh[1:],
        "trend": trend.astype(int),
    })
    daily["action_diff"] = (daily["action_ir"] - daily["action_pt"]).abs()

    pt_nav, pt_positions = replay_pt(compiled, view, config, feature_columns, agent)
    pt_nav = pt_nav[:len(nav)]
    pt_positions = pt_positions[:len(positions)]
    daily["position_pt"] = pt_positions[1:len(positions)]
    daily["nav_pt"] = pt_nav[1:len(nav)]
    daily["position_err"] = (daily["position"] - daily["position_pt"]).abs()
    daily["nav_err"] = (daily["nav"] - daily["nav_pt"]).abs()
    daily["nav_rel_err"] = daily["nav_err"] / daily["nav"].abs()

    max_nav_diff = float(daily["nav_err"].max())
    max_position_diff = float(daily["position_err"].max())
    max_nav_rel_diff = float(daily["nav_rel_err"].max())
    block = max(1, min(10, len(daily) // 5))

    def stats(series):
        return {
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p95": float(series.quantile(0.95)),
            "head_mean": float(series.head(block).mean()),
            "tail_mean": float(series.tail(block).mean()),
        }

    detail = {
        "code": stock.code,
        "days": int(len(daily)),
        "max_position_diff": max_position_diff,
        "max_nav_diff": max_nav_diff,
        "max_nav_rel_diff": max_nav_rel_diff,
        "worst_position_date": str(daily.loc[daily["position_err"].idxmax(), "date"]),
        "worst_nav_date": str(daily.loc[daily["nav_err"].idxmax(), "date"]),
        "position_error": stats(daily["position_err"]),
        "nav_rel_error": stats(daily["nav_rel_err"]),
    }

    summary = {
        "code": stock.code,
        "days": int(len(daily)),
        "start": daily["date"].iloc[0],
        "end": daily["date"].iloc[-1],
        "nav": float(nav[-1]),
        "bh_nav": float(bh[-1]),
        "excess": float(nav[-1] - bh[-1]),
        "max_drawdown": strategy_dd,
        "bh_max_drawdown": bh_dd,
        "avg_position": float(positions[1:].mean()),
        "uptrend_share": float(trend.mean()),
        "avg_position_uptrend": float(positions[1:][trend].mean()) if trend.any() else 0.0,
        "uptrend_capture": capture,
        "trades": int(env.trade_count),
        "closed_trades": int(len(env.trade_returns)),
        "max_action_diff": float(daily["action_diff"].max()),
        "max_position_diff": max_position_diff,
        "max_nav_diff": max_nav_diff,
        "max_nav_rel_diff": max_nav_rel_diff,
    }
    return summary, daily, detail


def replay_pt(compiled, view, config, feature_columns, agent):
    """用 PyTorch 确定性动作完整重放，返回净值与仓位曲线。"""
    env = TradingEnvV2(view, config["env"])
    state = env.reset()
    while not env.done:
        action = float(agent.select_action(state, deterministic=True)[0])
        state, _, _, _ = env.step(action)
    return (
        np.asarray(env.nav_history, dtype=float),
        np.asarray(env.position_history, dtype=float),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=63)
    parser.add_argument("--output", default="shadow_output")
    parser.add_argument("--model", default="ir_model/model.xml",
                        help="IR xml 路径；一致性核对用 model_fp32.xml，部署用 model.xml")
    parser.add_argument("--as-of", default=None,
                        help="增量数据截止日 YYYYMMDD，默认今天；每日盘后运行时自动前移")
    parser.add_argument("--refresh", action="store_true",
                        help="忽略当日缓存，强制重新拉取原始行情")
    args = parser.parse_args()
    as_of = args.as_of or datetime.now().strftime("%Y%m%d")

    with open("config_v2.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # 影子推理使用独立缓存目录，避免污染训练/回测的冻结缓存；
    # 截止日按自然日推进，缓存键随之变化，保证因子按最新完整历史重算（OBV 依赖 cumsum）。
    config["data"]["cache_dir"] = "cache_shadow"
    config["data"]["end_date"] = as_of
    config["data"]["refresh"] = bool(args.refresh)
    feature_count = int(config["network"].get("feature_count", 20))
    lookback = int(config["network"].get("lookback", 60))
    feature_columns = get_feature_columns(feature_count)

    stocks = load_stock_data(config["data"])
    agent = PPOAgent(config["network"], config["ppo"], "cpu")
    checkpoint = torch.load(
        ROOT / "rl_output_final_baseline" / "model_best.pth",
        map_location="cpu", weights_only=False,
    )
    agent.policy.load_state_dict(checkpoint["model_state"])
    agent.policy.eval()

    core = ov.Core()
    model_path = ROOT / args.model
    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, "CPU")

    run_dir = ROOT / args.output / ("run_" + datetime.now().strftime("%Y%m%d"))
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    details = []
    all_daily = []
    for code, stock in stocks.items():
        if len(stock.frame) < lookback + args.days:
            print("%s 历史不足以覆盖 %d 天，跳过" % (code, args.days))
            continue
        summary, daily, detail = run_stock(
            stock, agent, compiled, config, feature_columns, lookback, args.days
        )
        summaries.append(summary)
        details.append(detail)
        daily.insert(0, "code", code)
        all_daily.append(daily)
        daily.to_csv(run_dir / ("daily_%s.csv" % code), index=False, encoding="utf-8-sig")
        print(json.dumps(summary, ensure_ascii=False))

    # ── 事件风控层：插在 NPU 输出之后、写总 CSV 之前 ────────────────
    # 只对最新交易日（每个 code 一行）应用风控，历史行保持原样。
    full_daily = pd.concat(all_daily, ignore_index=True)
    # 关键：按 (date, code) 去重取最后一行，防止同一标的同日重复
    full_daily = full_daily.drop_duplicates(subset=["date", "code"], keep="last").copy()
    trade_date = str(full_daily["date"].max())
    risk_filter = EventRiskFilter(project_root=ROOT, use_network=True)

    # 构造 picks（每个 code 一行，最新交易日的真实仓位）
    picks_raw = []
    for code, _ in stocks.items():
        sub = full_daily[(full_daily["code"] == code)
                         & (full_daily["date"].astype(str) == trade_date)]
        if sub.empty:
            continue
        picks_raw.append({
            "code": code,
            "position": float(sub["position"].iloc[0]),
            "action": float(sub["action_ir"].iloc[0]) if "action_ir" in sub.columns else 0.0,
            "position_raw": float(sub["position"].iloc[0]),
        })

    filtered_picks = risk_filter.apply(picks_raw, trade_date=trade_date)
    risk_filter.save_outputs(filtered_picks, run_dir, trade_date)

    # 把风控结果回写到 full_daily 最新交易日对应的所有行（去重后每个 code 只有一行）
    risk_map = {fp["code"]: fp for fp in filtered_picks}
    full_daily["risk"] = ""
    full_daily["risk_reason"] = ""
    full_daily["filtered_position"] = full_daily["position"]
    for i in full_daily.index[full_daily["date"].astype(str) == trade_date]:
        code = full_daily.at[i, "code"]
        if code in risk_map:
            fp = risk_map[code]
            full_daily.at[i, "risk"] = fp["risk"]
            full_daily.at[i, "risk_reason"] = fp["risk_reason"]
            full_daily.at[i, "filtered_position"] = fp["filtered_position"]

    # 风控层日志汇总（只影响最新交易日，历史不回填）
    blocks = sum(1 for fp in filtered_picks if fp["risk"] == BLACKLIST)
    watches = sum(1 for fp in filtered_picks if fp["risk"] == WATCH)
    if blocks or watches:
        print("\n[EventRiskFilter] trade_date=%s：BLACKLIST=%d, WATCH=%d" % (trade_date, blocks, watches))
        for fp in filtered_picks:
            if fp["risk"] != NORMAL:
                print("  → %s %s: %s (原=%.4f → 风控后=%.4f)" % (
                    fp["code"], fp["risk"], fp["risk_reason"][:50],
                    fp["position_raw"], fp["filtered_position"]))

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(run_dir / "shadow_summary.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_json(run_dir / "shadow_summary.json", orient="records",
                          force_ascii=False, indent=2)
    full_daily.to_csv(
        run_dir / "shadow_daily_all.csv", index=False, encoding="utf-8-sig"
    )

    # ── 同步到 npu_output/run_latest：供「影子推理选股」Tab 按约定路径读取 ──
    # 只写入选股需要的产物，不覆盖 npu_batch_infer 的日期目录。
    latest_dir = ROOT / "npu_output" / "run_latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    full_daily.to_csv(latest_dir / "npu_daily_all.csv",
                      index=False, encoding="utf-8-sig")
    summary_frame.to_csv(latest_dir / "npu_summary.csv",
                         index=False, encoding="utf-8-sig")
    expires_at = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    pd.DataFrame([{
        "trade_date": trade_date,
        "code": fp["code"],
        "risk_level": fp["risk"],
        "reason": fp.get("risk_reason", ""),
        "expires_at": expires_at,
    } for fp in filtered_picks]).to_csv(
        latest_dir / "risk_blacklist.csv", index=False, encoding="utf-8-sig")
    print("[EventRiskFilter] 已同步 npu_output/run_latest（%d 只）" % len(filtered_picks))

    watch = []
    by_code = {item["code"]: item for item in summaries}
    if "600519" in by_code and (by_code["600519"]["uptrend_capture"] or 0) < 0.40:
        watch.append("600519 上行捕获率 %.2f%% < 40%%" % (
            100 * (by_code["600519"]["uptrend_capture"] or 0)))
    if "000001" in by_code:
        item = by_code["000001"]
        if item["avg_position_uptrend"] < 0.15 or (item["uptrend_capture"] or 0) < 0.40:
            watch.append("000001 趋势日仓位 %.2f%% / 捕获率 %.2f%% 偏低" % (
                100 * item["avg_position_uptrend"],
                100 * (item["uptrend_capture"] or 0)))
    if "300750" in by_code and by_code["300750"]["excess"] <= 0:
        watch.append("300750 未跑赢基准，超额 %.2f%%" % (100 * by_code["300750"]["excess"]))

    precision = "FP32" if "fp32" in model_path.name else "FP16"
    history_path = ROOT / args.output / "monitor_history.csv"

    # 把风控结果合并到 monitor_history rows
    risk_by_code = {fp["code"]: fp for fp in filtered_picks}
    rows = []
    for item in summaries:
        code = item["code"]
        risk_info = risk_by_code.get(code, {"risk": NORMAL, "filtered_position": item["avg_position"]})
        rows.append({
            "as_of": as_of,
            "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "window_days": args.days,
            "code": code,
            "start": item["start"],
            "end": item["end"],
            "nav": item["nav"],
            "bh_nav": item["bh_nav"],
            "excess": item["excess"],
            "max_drawdown": item["max_drawdown"],
            "avg_position": item["avg_position"],
            "avg_position_uptrend": item["avg_position_uptrend"],
            "uptrend_capture": item["uptrend_capture"],
            "trades": item["trades"],
            "closed_trades": item["closed_trades"],
            "max_position_diff": item["max_position_diff"],
            "max_nav_diff": item["max_nav_diff"],
            "max_nav_rel_diff": item["max_nav_rel_diff"],
            "risk_level": risk_info["risk"],
            "risk_reason": risk_info.get("risk_reason", ""),
            "filtered_position_latest": risk_info.get("filtered_position", item["avg_position"]),
        })
    history = pd.DataFrame(rows)
    if history_path.exists():
        history = pd.concat([
            pd.read_csv(history_path, dtype={"as_of": str, "code": str}), history
        ], ignore_index=True)
        history = history.drop_duplicates(
            subset=["as_of", "precision", "code"], keep="last"
        )
    if "capture_streak_600519" not in history.columns:
        history["capture_streak_600519"] = pd.NA

    # 600519 上行捕获率连续低于 40% 的次数：按最新行情日去重，周末/节假日重复运行不重复计数
    subset = history[(history["code"] == "600519")
                     & (history["precision"] == precision)]
    subset = subset.drop_duplicates(subset=["end"], keep="last").sort_values("end")
    streak = 0
    for value in reversed(subset["uptrend_capture"].tolist()):
        if pd.notna(value) and float(value) < 0.40:
            streak += 1
        else:
            break
    history.loc[
        (history["code"] == "600519") & (history["as_of"] == as_of)
        & (history["precision"] == precision), "capture_streak_600519"
    ] = streak
    history.sort_values(["as_of", "precision", "code"]).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )

    # 异常告警：仅记录，不自动调参
    alerts = []
    max_pos = max(item["max_position_diff"] for item in summaries)
    max_nav_rel = max(item["max_nav_rel_diff"] for item in summaries)
    if max_pos >= 1e-3 or max_nav_rel >= 1e-3:
        alerts.append(
            "IR 一致性超差：max_position_diff=%.3e，max_nav_rel_diff=%.3e" % (
                max_pos, max_nav_rel)
        )
    capture_600519 = next(
        (item["uptrend_capture"] for item in summaries if item["code"] == "600519"), None
    )
    if streak >= 5 and (capture_600519 or 0) < 0.40:
        alerts.append(
            "600519 上行捕获率连续 %d 次 <40%%（当前 %.2f%%）" % (
                streak, 100 * (capture_600519 or 0))
        )

    # 风控层告警（同 alerts 逻辑：仅记录，不自动调参）
    if blocks > 0:
        blocked_codes = [fp["code"] for fp in filtered_picks if fp["risk"] == BLACKLIST]
        alerts.append("事件风控 BLACKLIST 拦截 %d 只：%s" % (blocks, ", ".join(blocked_codes)))
    if watches > 0:
        watch_codes = [fp["code"] for fp in filtered_picks if fp["risk"] == WATCH]
        alerts.append("事件风控 WATCH 预警 %d 只：%s" % (watches, ", ".join(watch_codes)))

    report = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": as_of,
        "window_days": args.days,
        "lookback": lookback,
        "ir": str(model_path.relative_to(ROOT)),
        "precision": precision,
        "device": "CPU (NPU 部署前的等价前向)",
        "consistency": {
            "threshold": 1e-3,
            "nav_rel_threshold": 1e-3,
            "max_position_diff": max(item["max_position_diff"] for item in summaries),
            "max_nav_diff": max(item["max_nav_diff"] for item in summaries),
            "max_nav_rel_diff": max(item["max_nav_rel_diff"] for item in summaries),
            "passed": all(
                item["max_position_diff"] < 1e-3 or item["max_nav_rel_diff"] < 1e-3
                for item in summaries
            ),
            "detail": details,
        },
        "summary": summaries,
        "monitor": {
            "code": "600519",
            "capture_threshold": 0.40,
            "capture_streak_below_threshold": streak,
            "floor_adjustment": "not_triggered",
        },
        "watch": watch,
        "alerts": alerts,
        "verdict": "stable" if not watch else "watch",
    }
    with open(run_dir / "shadow_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    if alerts:
        alert_path = ROOT / args.output / "alerts.log"
        with open(alert_path, "a", encoding="utf-8") as handle:
            handle.write("[%s] as_of=%s precision=%s\n" % (
                report["run_at"], as_of, precision))
            for item in alerts:
                handle.write("  - %s\n" % item)

    print("\n一致性核对（门槛 1e-3 / NAV 相对 0.1%）：")
    for item in details:
        print("  %s 仓位 max=%.3e mean=%.3e head10=%.3e tail10=%.3e | "
              "NAV相对 max=%.3e head10=%.3e tail10=%.3e" % (
                  item["code"], item["position_error"]["max"],
                  item["position_error"]["mean"],
                  item["position_error"]["head_mean"],
                  item["position_error"]["tail_mean"],
                  item["nav_rel_error"]["max"],
                  item["nav_rel_error"]["head_mean"],
                  item["nav_rel_error"]["tail_mean"],
              ))
    print("  汇总 max_position_diff=%.3e max_nav_rel_diff=%.3e -> %s" % (
        report["consistency"]["max_position_diff"],
        report["consistency"]["max_nav_rel_diff"],
        "PASS" if report["consistency"]["passed"] else "FAIL",
    ))
    print("监控告警：%s" % ("无" if not watch else "；".join(watch)))
    print("异常告警：%s" % ("无" if not alerts else "；".join(alerts)))
    print("600519 捕获率连续<40%% 次数：%d" % streak)
    print("监控历史：%s（%d 行）" % (history_path.resolve(), len(history)))
    print("影子推理完成：%s" % run_dir.resolve())


if __name__ == "__main__":
    main()
