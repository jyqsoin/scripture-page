# -*- coding: utf-8 -*-
"""影子推理监控看板：只读 monitor_history.csv，展示逐日指标并给出告警提示。

- 不修改任何模型/风控参数，不触发调参；
- 告警：600519 捕获率 <40% 且 streak ≥5 → 人工 floor 讨论；
        FP16 仓位/NAV 相对误差 ≥1e-3 → 一致性超差；
        日志中出现 FAILED exit= → 脚本失败记录。
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / "shadow_output" / "monitor_history.csv"
LOG_DIR = ROOT / "logs"
NUMERIC = [
    "nav", "bh_nav", "excess", "avg_position", "avg_position_uptrend",
    "uptrend_capture", "capture_streak_600519", "max_position_diff",
    "max_nav_diff", "max_nav_rel_diff",
]


def load_history(path):
    if not path.exists():
        raise SystemExit("未找到 %s，请先运行 shadow_infer.py" % path)
    history = pd.read_csv(path, dtype={"as_of": str, "code": str})
    for column in NUMERIC:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    if "capture_streak_600519" not in history.columns:
        history["capture_streak_600519"] = pd.NA
    return history


def build_table(history, precision, limit):
    subset = history if precision == "all" else history[history["precision"] == precision]
    if subset.empty:
        raise SystemExit("没有 precision=%s 的记录" % precision)
    rows = []
    for (as_of, end), group in subset.groupby(["as_of", "end"], sort=True):
        by_code = {str(item["code"]): item for _, item in group.iterrows()}
        c600 = by_code.get("600519", {})
        c000 = by_code.get("000001", {})
        c300 = by_code.get("300750", {})
        rows.append({
            "as_of": as_of,
            "end": end,
            "600519_capture": c600.get("uptrend_capture", float("nan")),
            "600519_streak": c600.get("capture_streak_600519", float("nan")),
            "000001_pos": c000.get("avg_position_uptrend", float("nan")),
            "300750_excess": c300.get("excess", float("nan")),
            "pos_err": group["max_position_diff"].max(),
            "nav_rel_err": group["max_nav_rel_diff"].max(),
        })
    return pd.DataFrame(rows).sort_values(["as_of", "end"]).tail(limit).reset_index(drop=True)


def evaluate_alerts(frame):
    alerts = []
    latest = frame.iloc[-1]
    capture = latest["600519_capture"]
    streak = latest["600519_streak"]
    if pd.notna(capture) and pd.notna(streak) and capture < 0.40 and streak >= 5:
        alerts.append(
            "600519 捕获率 %.2f%% 且 streak=%d（≥5）→ 进入人工 floor 讨论（不自动调参）" % (
                100 * capture, int(streak))
        )
    if pd.notna(latest["pos_err"]) and latest["pos_err"] >= 1e-3:
        alerts.append("一致性超差：FP16 仓位误差 %.3e ≥ 1e-3" % latest["pos_err"])
    if pd.notna(latest["nav_rel_err"]) and latest["nav_rel_err"] >= 1e-3:
        alerts.append("一致性超差：FP16 NAV 相对误差 %.3e ≥ 1e-3" % latest["nav_rel_err"])
    return alerts


def scan_failed_logs(days):
    if not LOG_DIR.exists():
        return []
    hits = []
    for path in sorted(LOG_DIR.glob("shadow_*.log"))[-days:]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if "FAILED exit=" in line:
                hits.append("%s: %s" % (path.name, line.strip()))
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--precision", default="FP16", choices=["FP16", "FP32", "all"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--log-days", type=int, default=7)
    parser.add_argument("--output", default=str(ROOT / "shadow_output" / "dashboard.txt"))
    args = parser.parse_args()

    history = load_history(Path(args.history))
    frame = build_table(history, args.precision, args.limit)
    alerts = evaluate_alerts(frame)
    failed = scan_failed_logs(args.log_days)

    view = frame.copy()
    view["600519_capture"] = view["600519_capture"].map(
        lambda v: "-" if pd.isna(v) else "%.2f%%" % (100 * v))
    view["600519_streak"] = view["600519_streak"].map(
        lambda v: "-" if pd.isna(v) else "%d" % int(v))
    view["000001_pos"] = view["000001_pos"].map(
        lambda v: "-" if pd.isna(v) else "%.2f%%" % (100 * v))
    view["300750_excess"] = view["300750_excess"].map(
        lambda v: "-" if pd.isna(v) else "%+.2f%%" % (100 * v))
    view["pos_err"] = view["pos_err"].map(
        lambda v: "-" if pd.isna(v) else "%.3e" % v)
    view["nav_rel_err"] = view["nav_rel_err"].map(
        lambda v: "-" if pd.isna(v) else "%.3e" % v)
    view = view.rename(columns={
        "as_of": "运行日", "end": "行情截止", "600519_capture": "600519捕获",
        "600519_streak": "streak", "000001_pos": "000001趋势仓位",
        "300750_excess": "300750超额", "pos_err": "FP16仓位误差",
        "nav_rel_err": "FP16_NAV相对误差",
    })

    lines = [
        "影子推理监控看板  %s（precision=%s，最近 %d 个记录）" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.precision, len(view)),
        "",
        view.to_string(index=False),
        "",
        "锁版参数：floor 600519/000001=0.20，300750=0.25；只记录不调参",
    ]
    if alerts:
        lines.append("告警：")
        lines.extend("  - %s" % item for item in alerts)
    else:
        lines.append("告警：无")
    if failed:
        lines.append("失败记录（近 %d 份日志）：" % args.log_days)
        lines.extend("  - %s" % item for item in failed)
    else:
        lines.append("失败记录：无")

    text = "\n".join(lines)
    print(text)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print("\n看板快照：%s" % output.resolve())


if __name__ == "__main__":
    main()
