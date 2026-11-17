# -*- coding: utf-8 -*-
"""NPU 推理监控看板：对比 npu_output 与 CPU 影子（shadow_output）结果。

- 读取最近一次 NPU run 的 npu_report.json / npu_summary.csv / npu_daily_all.csv；
- 读取同日期 CPU 影子 run 的 shadow_summary.csv / daily_*.csv；
- 展示逐票 NAV、仓位、交易闭环差异，以及逐日 NAV 最大偏离；
- 仅展示与告警，不修改任何参数。
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def latest_run(root):
    runs = sorted(root.glob("run_*"))
    return runs[-1] if runs else None


def read_shadow_daily(run_dir):
    frames = []
    for path in sorted(run_dir.glob("daily_*.csv")):
        frame = pd.read_csv(path, dtype={"code": str})
        frames.append(frame[["code", "date", "nav", "position"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compare_daily(npu_run, shadow_run):
    npu_path = npu_run / "npu_daily_all.csv"
    if not npu_path.exists():
        return pd.DataFrame()
    npu_daily = pd.read_csv(npu_path, dtype={"code": str})
    shadow_daily = read_shadow_daily(shadow_run)
    if shadow_daily.empty:
        return pd.DataFrame()
    merged = npu_daily.merge(
        shadow_daily, on=["code", "date"], suffixes=("_npu", "_cpu")
    )
    merged["nav_diff"] = (merged["nav_npu"] - merged["nav"]).abs()
    merged["nav_rel"] = merged["nav_diff"] / np.maximum(merged["nav"].abs(), 1e-12)
    rows = []
    for code, group in merged.groupby("code"):
        rows.append({
            "code": code,
            "max_nav_diff": float(group["nav_diff"].max()),
            "max_nav_rel": float(group["nav_rel"].max()),
            "worst_date": str(group.loc[group["nav_rel"].idxmax(), "date"]),
            "last_nav_npu": float(group["nav_npu"].iloc[-1]),
            "last_nav_cpu": float(group["nav"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu-output", default=str(ROOT / "npu_output"))
    parser.add_argument("--shadow-output", default=str(ROOT / "shadow_output"))
    parser.add_argument("--run", default=None, help="指定 run_YYYYMMDD，默认取最新")
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--output", default=str(ROOT / "npu_output" / "npu_dashboard.txt"))
    args = parser.parse_args()

    npu_root = Path(args.npu_output)
    shadow_root = Path(args.shadow_output)
    npu_run = npu_root / args.run if args.run else latest_run(npu_root)
    shadow_run = shadow_root / args.run if args.run else latest_run(shadow_root)
    if npu_run is None or not npu_run.exists():
        raise SystemExit("未找到 npu_output/run_*，请先运行 npu_batch_infer.py")
    if shadow_run is None or not shadow_run.exists():
        raise SystemExit("未找到 shadow_output/run_*，请先运行 shadow_infer.py")

    npu_summary = pd.read_csv(npu_run / "npu_summary.csv", dtype={"code": str})
    shadow_summary = pd.read_csv(shadow_run / "shadow_summary.csv", dtype={"code": str})
    report_path = npu_run / "npu_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    merged = npu_summary.merge(
        shadow_summary, on="code", suffixes=("_npu", "_cpu")
    )
    table = pd.DataFrame({
        "代码": merged["code"],
        "NPU_NAV": merged["nav_npu"].map(lambda v: "%.4f" % v),
        "CPU影子_NAV": merged["nav_cpu"].map(lambda v: "%.4f" % v),
        "NAV差": merged.apply(lambda row: "%.2e" % abs(row["nav_npu"] - row["nav_cpu"]), axis=1),
        "NPU仓位": merged["avg_position_npu"].map(lambda v: "%.2f%%" % (100 * v)),
        "CPU仓位": merged["avg_position_cpu"].map(lambda v: "%.2f%%" % (100 * v)),
        "NPU交易/平仓": merged["trades_npu"].astype(int).astype(str) + " / "
                        + merged["closed_trades_npu"].astype(int).astype(str),
        "CPU交易/平仓": merged["trades_cpu"].astype(int).astype(str) + " / "
                        + merged["closed_trades_cpu"].astype(int).astype(str),
        "FP16仓位误差": merged["max_position_diff_npu"].map(lambda v: "%.3e" % v),
        "FP16_NAV相对误差": merged["max_nav_rel_diff_npu"].map(lambda v: "%.3e" % v),
        "NPU延迟ms": "%.3f" % report.get("latency_ms", {}).get("mean", float("nan")),
    })

    daily_cmp = compare_daily(npu_run, shadow_run)
    alerts = []
    consistency = report.get("consistency", {})
    max_pos = consistency.get("max_position_diff", float("nan"))
    max_nav_rel = consistency.get("max_nav_rel_diff", float("nan"))
    if consistency and (max_pos >= args.threshold or max_nav_rel >= args.threshold):
        alerts.append("NPU 一致性超差：position=%.3e / NAV相对=%.3e ≥ %.0e" % (
            max_pos, max_nav_rel, args.threshold))
    if not consistency.get("passed", True):
        alerts.append("NPU 报告标记一致性未通过")
    if not daily_cmp.empty:
        worst = daily_cmp.loc[daily_cmp["max_nav_rel"].idxmax()]
        if worst["max_nav_rel"] >= args.threshold:
            alerts.append("与 CPU 影子逐日 NAV 偏离超阈值：%s %.3e（%s）" % (
                worst["code"], worst["max_nav_rel"], worst["worst_date"]))

    lines = [
        "NPU 推理监控看板  %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "NPU run: %s  |  CPU 影子 run: %s" % (npu_run.name, shadow_run.name),
        "device=%s  batch=%s(%s)  calls=%s  延迟 mean=%.3fms p95=%.3fms" % (
            report.get("device", "-"),
            report.get("batch_size", "-"),
            report.get("batch_mode", "-"),
            report.get("calls", "-"),
            report.get("latency_ms", {}).get("mean", float("nan")),
            report.get("latency_ms", {}).get("p95", float("nan")),
        ),
        "一致性：position=%.3e  NAV相对=%.3e  -> %s（阈值 %.0e）" % (
            max_pos, max_nav_rel,
            "PASS" if consistency.get("passed") else "FAIL", args.threshold),
        "",
        table.to_string(index=False),
    ]
    if not daily_cmp.empty:
        daily_view = pd.DataFrame({
            "代码": daily_cmp["code"],
            "逐日NAV最大差": daily_cmp["max_nav_diff"].map(lambda v: "%.2e" % v),
            "逐日NAV最大相对差": daily_cmp["max_nav_rel"].map(lambda v: "%.2e" % v),
            "最差日期": daily_cmp["worst_date"],
            "末值_NPU": daily_cmp["last_nav_npu"].map(lambda v: "%.4f" % v),
            "末值_CPU": daily_cmp["last_nav_cpu"].map(lambda v: "%.4f" % v),
        })
        lines.extend(["", "与 CPU 影子逐日 NAV 对比：", daily_view.to_string(index=False)])

    lines.append("")
    if alerts:
        lines.append("告警：")
        lines.extend("  - %s" % item for item in alerts)
    else:
        lines.append("告警：无")
    lines.append("说明：NPU 为部署推理，CPU 影子为同权重 FP16 IR 参考；只读、不调参")

    text = "\n".join(lines)
    print(text)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print("\n看板快照：%s" % output.resolve())


if __name__ == "__main__":
    main()
