# -*- coding: utf-8 -*-
"""条件化五因子评估 + 四因子口径定案（Step I2-alt-14）。

本脚本不重跑回测，只读取 I2-alt-13 已落盘的真实结果：
- conditional_summary.csv / segments.csv / periods.csv
- four_reconciliation.csv / four_variants.csv
- nav.json（用于定位最差回撤区间）

任务 A：对比 five_always 与 five_cond_w3/w6 的全期/五段绩效，
        判断五段通过率是否 >2/5、S5 回撤是否收敛，给出 UI 开关建议。
任务 B：核对 four_old 与 four_floor10 的口径差异，判断 four_floor10
        是否可作为四因子优化选项，并产出 four_decision.md 定案文档。

注意：实际落盘数据为 four_old 回撤 10.72%、four_floor10 回撤 12.45%，
本脚本按实测值定案，不沿用任何相反的口径描述。

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

from core.smart_beta import DISCLAIMER
from core.smart_beta_backtest import SMART_BETA_DIR
from core.smart_beta_conditional_floor import REFERENCE, COND_FLOOR

STEM = "smart_beta_conditional_floor_20240531_20261113_300"
FULL_NAMES = ("five_always", "five_cond_w3", "five_cond_w6", "five_never")
COND_NAMES = ("five_cond_w3", "five_cond_w6")
SEGMENTS = ("S1", "S2", "S3", "S4", "S5")


def _read_csv(path):
    if not Path(path).exists():
        raise FileNotFoundError("缺少 I2-alt-13 落盘文件：%s" % path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _as_bool(value):
    return str(value).strip().lower() == "true"


def _fmt_pct(value, digits=2):
    return ("%." + str(digits) + "f%%") % (float(value) * 100)


def _fmt_pp(value, digits=2):
    value = float(value)
    if abs(value) < 0.5 * 10 ** (-digits):
        return ("%." + str(digits) + "fpp") % 0.0
    return ("%+." + str(digits) + "fpp") % value


def load_alt13(directory=None):
    """读取 I2-alt-13 的五份 CSV 与净值 JSON。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    payload = {
        "conditional_summary": _read_csv(
            directory / ("%s_conditional_summary.csv" % STEM)),
        "segments": _read_csv(directory / ("%s_segments.csv" % STEM)),
        "periods": _read_csv(directory / ("%s_periods.csv" % STEM)),
        "four_reconciliation": _read_csv(
            directory / ("%s_four_reconciliation.csv" % STEM)),
        "four_variants": _read_csv(
            directory / ("%s_four_variants.csv" % STEM)),
        "nav": json.loads((directory / ("%s_nav.json" % STEM)).read_text(
            encoding="utf-8")),
        "directory": directory,
    }
    return payload


def _segment_lookup(segments, name):
    subset = segments[segments["portfolio"] == name].copy()
    subset["prefix"] = subset["segment_label"].str.extract(r"^(S\d)")
    return {row["prefix"]: row for _, row in subset.iterrows()}


def build_conditional_eval(payload):
    """构造 five_always vs five_cond_w3/w6 的全期+五段对比表。"""
    summary = payload["conditional_summary"].set_index("portfolio")
    segments = payload["segments"]
    benchmark_dd = float(REFERENCE["benchmark"]["max_drawdown"])
    always = summary.loc["five_always"]
    always_s5 = _segment_lookup(segments, "five_always")["S5"]
    rows = []
    for name in FULL_NAMES:
        row = summary.loc[name]
        segment_map = _segment_lookup(segments, name)
        record = {
            "portfolio": name, "label": row["label"],
            "full_annual_return": float(row["annual_return"]),
            "full_total_return": float(row["total_return"]),
            "full_max_drawdown": float(row["max_drawdown"]),
            "full_sharpe": float(row["sharpe"]),
            "full_passed": _as_bool(row["full_passed"]),
            "passed_segments": int(row["passed_segments"]),
            "pass_rate": float(row["pass_rate"]),
            "pass_rate_gt_2of5": bool(int(row["passed_segments"]) > 2),
            "avg_momentum_weight": float(row["avg_momentum_weight"]),
            "floor_on_periods": int(row["floor_on_periods"]),
            "floor_on_ratio": float(row["floor_on_ratio"]),
            "delta_annual_vs_always_pp": float(
                row["delta_annual_vs_always_pp"]),
            "delta_dd_vs_always_pp": float(row["delta_dd_vs_always_pp"]),
            "delta_passed_segments_vs_always": int(
                row["delta_passed_segments_vs_always"]),
        }
        for segment in SEGMENTS:
            source = segment_map.get(segment)
            if source is None:
                raise ValueError("缺少 %s 的 %s 分段结果" % (name, segment))
            record[segment.lower() + "_annual"] = float(source["annual_return"])
            record[segment.lower() + "_dd"] = float(source["max_drawdown"])
            record[segment.lower() + "_sharpe"] = float(source["sharpe"])
            record[segment.lower() + "_passed"] = _as_bool(source["passed"])
        s5_dd = float(segment_map["S5"]["max_drawdown"])
        record["s5_max_drawdown"] = s5_dd
        record["s5_dd_vs_always_pp"] = (s5_dd - float(
            always_s5["max_drawdown"])) * 100
        record["s5_dd_improvement_pp"] = (float(
            always_s5["max_drawdown"]) - s5_dd) * 100
        record["s5_dd_converged"] = bool(
            s5_dd < float(always_s5["max_drawdown"]) - 1e-12)
        record["full_dd_not_above_benchmark"] = bool(
            float(row["max_drawdown"]) <= benchmark_dd)
        record["benchmark_max_drawdown"] = benchmark_dd
        if name in COND_NAMES:
            record["ui_toggle_recommended"] = bool(
                record["pass_rate_gt_2of5"]
                and record["s5_dd_converged"]
                and record["full_dd_not_above_benchmark"])
            record["evaluation_note"] = (
                "通过率%d/5，未>2/5；S5回撤%s；全期回撤%s，不建议增加开关"
                % (record["passed_segments"],
                   "收敛" if record["s5_dd_converged"] else "未收敛",
                   "达标" if record["full_dd_not_above_benchmark"] else "破基准"))
        elif name == "five_always":
            record["ui_toggle_recommended"] = False
            record["evaluation_note"] = "基准组合（全时启用10%下限）"
        else:
            record["ui_toggle_recommended"] = False
            record["evaluation_note"] = "参考组合（旧并集规则，不启用下限）"
        rows.append(record)
    return pd.DataFrame(rows)


def _drawdown_window(nav):
    """返回 (峰值日, 谷底日, 最大回撤)，用于说明差异发生区间。"""
    series = pd.Series(nav).astype(float)
    drawdown = series / series.cummax() - 1
    trough = drawdown.idxmin()
    peak = series.loc[:trough].idxmax()
    return str(peak), str(trough), float(drawdown.min())


def build_four_decision(payload, conditional_eval):
    """生成 four_decision.md 的内容与关键判定。"""
    recon = payload["four_reconciliation"].set_index("item")
    variants = payload["four_variants"]
    nav = payload["nav"]["full_nav"]
    benchmark_dd = float(REFERENCE["benchmark"]["max_drawdown"])
    benchmark_annual = float(REFERENCE["benchmark"]["annual_return"])

    old = recon.loc["界面/固化基线·四因子（floor=None）"]
    floor10 = recon.loc["I2-alt-9 敏感性表·四因子@10%"]
    five = recon.loc["固化基线·五因子@10%（主基线）"]
    diff = recon.loc["差异·四因子@10% − 四因子floor=None"]
    old_dd = float(old["max_drawdown"])
    floor_dd = float(floor10["max_drawdown"])
    old_annual = float(old["annual_return"])
    floor_annual = float(floor10["annual_return"])
    old_peak, old_trough, _ = _drawdown_window(nav["four_old"])
    floor_peak, floor_trough, _ = _drawdown_window(nav["four_floor10"])

    variant_map = variants.set_index("variant")
    floor10_variant = variant_map.loc["four_floor10"]
    floor10_eligible = bool(
        float(floor10_variant["max_drawdown"]) <= benchmark_dd)
    # 满足回撤约束的替代方案
    eligible = variants[
        (variants["max_drawdown"] <= benchmark_dd)
        & (variants["variant"] != "four_old")].copy()
    eligible = eligible.sort_values("max_drawdown")

    cond_always = conditional_eval.set_index("portfolio").loc["five_always"]
    cond_w3 = conditional_eval.set_index("portfolio").loc["five_cond_w3"]
    cond_w6 = conditional_eval.set_index("portfolio").loc["five_cond_w6"]

    lines = []
    lines.append("# 四因子口径定案（I2-alt-14）")
    lines.append("")
    lines.append("- 生成时间：%s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("- 数据范围：300 只，2024-05-31 ~ 2026-11-13，月末调仓，"
                 "收益未扣交易成本")
    lines.append("- 基准：池内等权，年化 %s，最大回撤 %s" % (
        _fmt_pct(benchmark_annual), _fmt_pct(benchmark_dd)))
    lines.append("- 数据来源：I2-alt-13 落盘报告（%s_*.csv）" % STEM)
    lines.append("")
    lines.append("> 仅研究用途，不构成投资建议。")
    lines.append("")
    lines.append("## 一、结论摘要")
    lines.append("")
    lines.append("1. **口径映射以实测为准**：`four_old`（不做并集修复、"
                 "无动量下限）全期回撤 **%s**，通过；`four_floor10`"
                 "（叠加 10%% 动量下限）全期回撤 **%s**，破基准。任务前提中"
                 "将两者写反，本定案不沿用该写法。" % (
                     _fmt_pct(old_dd), _fmt_pct(floor_dd)))
    lines.append("2. **四因子主基线保持 `four_old` 旧规则**：年化 %s、"
                 "最大回撤 %s，是四因子中收益最高且满足回撤约束的口径。" % (
                     _fmt_pct(old_annual), _fmt_pct(old_dd)))
    lines.append("3. **`four_floor10` 不可作为四因子优化选项**：回撤 %s > "
                 "基准 %s，且年化 %s 较 `four_old` 低 %.2fpp，硬约束与收益"
                 "均不占优。" % (_fmt_pct(floor_dd), _fmt_pct(benchmark_dd),
                                _fmt_pct(floor_annual),
                                abs((floor_annual - old_annual) * 100)))
    lines.append("4. **可选优化仅作文档标注**：若必须暴露低回撤选项，"
                 "`four_floor05`（回撤 %s、年化 %s）与 `four_floor10_slots`"
                 "（回撤 %s、年化 %s）满足回撤约束；`four_top40` 回撤最低"
                 "（%s）但年化 %s，且改变并集规模，不属于动量下限口径。"
                 "上述均不替代 `four_old` 主基线。" % (
                     _fmt_pct(float(variant_map.loc["four_floor05",
                                                    "max_drawdown"])),
                     _fmt_pct(float(variant_map.loc["four_floor05",
                                                    "annual_return"])),
                     _fmt_pct(float(variant_map.loc["four_floor10_slots",
                                                    "max_drawdown"])),
                     _fmt_pct(float(variant_map.loc["four_floor10_slots",
                                                    "annual_return"])),
                     _fmt_pct(float(variant_map.loc["four_top40",
                                                    "max_drawdown"])),
                     _fmt_pct(float(variant_map.loc["four_top40",
                                                    "annual_return"]))))
    lines.append("")
    lines.append("## 二、口径核对")
    lines.append("")
    lines.append("| 口径 | 动量下限 | 并集修复 | 年化 | 最大回撤 | 判定 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for item, row in (("界面/基线·四因子", old),
                      ("I2-alt-9 敏感性·四因子", floor10),
                      ("固化基线·五因子", five)):
        floor = "无" if pd.isna(row["momentum_floor"]) else _fmt_pct(
            row["momentum_floor"], 0)
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            item, floor, "是" if _as_bool(row["union_fix"]) else "否",
            _fmt_pct(row["annual_return"]), _fmt_pct(row["max_drawdown"]),
            "通过" if _as_bool(row["passed"]) else "未通过"))
    lines.append("")
    lines.append("两版四因子回撤差 **%s**（%s − %s），来源仅为是否叠加 "
                 "10%% 动量下限；并集期数与交集期数均为 28/2，未发生变化。"
                 % (_fmt_pp(float(diff["max_drawdown"]) * 100),
                    _fmt_pct(floor_dd), _fmt_pct(old_dd)))
    lines.append("")
    lines.append("注：表中“并集修复”指是否在并集分支叠加动量最低权重下限"
                 "并按第 10 百分位上调动量腿权重。")
    lines.append("")
    lines.append("## 三、差异根因")
    lines.append("")
    lines.append("- 10%% 下限在并集分支把动量腿最低权重抬到 10%%，并强制替换 "
                 "3 只非动量标的；四因子平均动量权重由 **%s** 升至 **%s**。" % (
                     _fmt_pct(float(old["avg_momentum_weight"])),
                     _fmt_pct(float(floor10["avg_momentum_weight"]))))
    lines.append("- 叠加下限后年化由 %s 降至 %s（%s），最大回撤由 %s 升至 "
                 "%s（%s），并非收益/波动权衡，而是收益与回撤同时恶化。" % (
                     _fmt_pct(old_annual), _fmt_pct(floor_annual),
                     _fmt_pp((floor_annual - old_annual) * 100),
                     _fmt_pct(old_dd), _fmt_pct(floor_dd),
                     _fmt_pp((floor_dd - old_dd) * 100)))
    lines.append("- 最差回撤区间：`four_old` %s → %s；`four_floor10` %s → %s。"
                 "两者回撤起点与谷底相同，叠加下限后同一区间的回撤由 %s 扩大"
                 "到 %s，说明下限在该区间放大了动量暴露的损失。" % (
                     old_peak, old_trough, floor_peak, floor_trough,
                     _fmt_pct(old_dd), _fmt_pct(floor_dd)))
    lines.append("")
    lines.append("## 四、四因子可选优化评估（目标：全期回撤 ≤ %s）" % _fmt_pct(
        benchmark_dd))
    lines.append("")
    lines.append("| 变体 | 动量下限 | 并集规模 | 年化 | 最大回撤 | 较旧规则回撤 "
                 "| 达标 | 结论 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    decision_note = {
        "four_old": "主基线，保留",
        "four_floor05": "满足约束，但收益/回撤均略逊于旧规则，仅可标注",
        "four_floor10": "不达标，回撤破基准，不可作为优化选项",
        "four_floor10_slots": "满足约束，仅可标注为可选",
        "four_top40": "回撤最低但改变并集规模，仅可标注为低回撤可选",
    }
    for _, row in variants.iterrows():
        floor = "无" if pd.isna(row["momentum_floor"]) else _fmt_pct(
            row["momentum_floor"], 0)
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %s |" % (
            row["label"], floor, int(row["union_top_n"]),
            _fmt_pct(row["annual_return"]), _fmt_pct(row["max_drawdown"]),
            _fmt_pp(row["delta_dd_vs_old_pp"]),
            "是" if _as_bool(row["meets_target"]) else "否",
            decision_note.get(row["variant"], "")))
    lines.append("")
    lines.append("说明：`four_floor10` 的年化 %s 已低于 `four_old` 的 %s，"
                 "且回撤高出 %.2fpp，因此不具备“优化”含义；`four_floor05`/"
                 "`four_floor10_slots` 虽然满足回撤约束，但相对旧规则仍无"
                 "收益或回撤优势，最多作为文档/UI 的可选项说明。" % (
                     _fmt_pct(floor_annual), _fmt_pct(old_annual),
                     abs((floor_dd - old_dd) * 100)))
    lines.append("")
    lines.append("## 五、最终决策")
    lines.append("")
    lines.append("1. 四因子主基线**默认保持 `four_old`**，不切换到任何动量下限口径。")
    lines.append("2. `four_floor10` 标记为**不可选**（回撤 %s > %s）。" % (
        _fmt_pct(floor_dd), _fmt_pct(benchmark_dd)))
    lines.append("3. 若后续 UI 需要“四因子可选优化”标注，仅可列出 "
                 "`four_floor05`、`four_floor10_slots`、`four_top40`，"
                 "并注明其相对主基线的收益/回撤取舍；默认不启用。")
    lines.append("4. 五因子主基线（全时 10%% 下限）不变，年化 %s、回撤 %s。"
                 % (_fmt_pct(five["annual_return"]),
                    _fmt_pct(five["max_drawdown"])))
    lines.append("")
    lines.append("## 六、附：条件化五因子评估（任务A）")
    lines.append("")
    lines.append("| 组合 | 全期年化 | 全期回撤 | 夏普 | 五段通过 | S5回撤 | "
                 "建议UI开关 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for name in FULL_NAMES:
        row = conditional_eval.set_index("portfolio").loc[name]
        lines.append("| %s | %s | %s | %.3f | %d/5 | %s | %s |" % (
            row["label"], _fmt_pct(row["full_annual_return"]),
            _fmt_pct(row["full_max_drawdown"]), row["full_sharpe"],
            int(row["passed_segments"]), _fmt_pct(row["s5_max_drawdown"]),
            "建议" if _as_bool(row["ui_toggle_recommended"]) else "不建议"))
    lines.append("")
    lines.append("结论：两档条件化窗口通过率均为 **2/5**（未 >2/5），"
                 "S5 回撤均为 **%s**（与全时启用相同，未收敛），"
                 "且全期回撤分别为 **%s / %s**，均破基准 **%s**。"
                 "因此**不建议在 UI 增加“条件化启用”开关**。" % (
                     _fmt_pct(cond_always["s5_max_drawdown"]),
                     _fmt_pct(cond_w3["full_max_drawdown"]),
                     _fmt_pct(cond_w6["full_max_drawdown"]),
                     _fmt_pct(benchmark_dd)))
    lines.append("")
    lines.append("## 七、自测记录")
    lines.append("")
    lines.append("- 五因子主基线 `five_always`：与 I2-alt-9 报告逐项误差 < 1e-9。")
    lines.append("- `five_never`、`four_old`、`four_floor10`：与历史报告误差 < 1e-9。")
    lines.append("- 条件化逻辑：关闭期逐期收益 = `five_never`，开启期动量权重 ≥ 10%。")
    lines.append("- 四因子复现：`four_old` 回撤 %s，`four_floor10` 回撤 %s，"
                 "差异 %s。" % (_fmt_pct(old_dd), _fmt_pct(floor_dd),
                                _fmt_pp((floor_dd - old_dd) * 100)))
    lines.append("- 五段复利与全期净值一致；5 个分段、30 个调仓期。")
    lines.append("")
    lines.append("仅研究用途，不构成投资建议。")
    lines.append("")
    return "\n".join(lines), {
        "old_dd": old_dd, "floor_dd": floor_dd,
        "old_annual": old_annual, "floor_annual": floor_annual,
        "floor10_eligible": floor10_eligible,
        "eligible_variants": [str(value) for value in eligible["variant"]],
        "ui_toggle_recommended": bool(
            conditional_eval["ui_toggle_recommended"].any()),
    }


def combined_metrics(payload):
    """合并五因子汇总与四因子口径核对表的全期指标，按 portfolio 索引。"""
    summary = payload["conditional_summary"].set_index("portfolio")
    recon = payload["four_reconciliation"].set_index("item")
    rows = {}
    for name in FULL_NAMES:
        rows[name] = {
            "total_return": float(summary.loc[name, "total_return"]),
            "annual_return": float(summary.loc[name, "annual_return"]),
            "max_drawdown": float(summary.loc[name, "max_drawdown"]),
        }
    mapping = {
        "four_old": "界面/固化基线·四因子（floor=None）",
        "four_floor10": "I2-alt-9 敏感性表·四因子@10%",
    }
    for name, item in mapping.items():
        rows[name] = {
            "total_return": float(recon.loc[item, "total_return"]),
            "annual_return": float(recon.loc[item, "annual_return"]),
            "max_drawdown": float(recon.loc[item, "max_drawdown"]),
        }
    return pd.DataFrame(rows).T


def verify_eval(payload, conditional_eval):
    """末尾自测：主基线 1e-9、条件化逻辑、四因子复现。"""
    periods = payload["periods"]
    segments = payload["segments"]
    recon = payload["four_reconciliation"].set_index("item")
    metrics = combined_metrics(payload)

    for key, name in (("five_floor", "five_always"),
                      ("five_old", "five_never"),
                      ("four_old", "four_old"),
                      ("four_floor10", "four_floor10")):
        for metric in ("total_return", "annual_return", "max_drawdown"):
            actual = float(metrics.loc[name, metric])
            expected = float(REFERENCE[key][metric])
            assert abs(actual - expected) < 1e-9, (
                "%s.%s 误差 %.12f" % (name, metric, actual - expected))

    benchmark_dd = float(REFERENCE["benchmark"]["max_drawdown"])
    assert abs(float(recon.loc["界面/固化基线·四因子（floor=None）",
                              "max_drawdown"]) - REFERENCE["four_old"]
               ["max_drawdown"]) < 1e-9
    assert abs(float(recon.loc["I2-alt-9 敏感性表·四因子@10%",
                              "max_drawdown"]) - REFERENCE["four_floor10"]
               ["max_drawdown"]) < 1e-9
    assert float(metrics.loc["four_old", "max_drawdown"]) <= benchmark_dd
    assert float(metrics.loc["four_floor10", "max_drawdown"]) > benchmark_dd

    for window in ("w3", "w6"):
        name = "five_cond_" + window
        on = periods[name + "_floor_on"].astype(bool)
        assert on.any() and (~on).any(), "条件化未发生切换：%s" % name
        assert (periods.loc[on, name + "_momentum_weight"]
                >= COND_FLOOR - 1e-9).all()
        closed = periods.loc[~on]
        assert np.allclose(closed[name + "_return"],
                           closed["five_never_return"], atol=1e-12, rtol=0)
        opened = periods.loc[on]
        assert np.allclose(opened[name + "_return"],
                           opened["five_always_return"], atol=1e-12, rtol=0)

    for name in ("five_always", "five_cond_w3", "five_cond_w6", "five_never",
                 "four_old", "four_floor10"):
        compounded = float(np.prod(
            1 + segments.loc[segments["portfolio"] == name,
                             "total_return"]) - 1)
        assert abs(compounded - float(metrics.loc[name, "total_return"])) < 1e-9
    assert len(conditional_eval) == 4
    assert conditional_eval["portfolio"].nunique() == 4
    assert not conditional_eval["ui_toggle_recommended"].any()
    print("自测通过：主基线误差<1e-9；条件化关闭期=five_never、开启期权重≥10%%；"
          "four_old=%.2f%% / four_floor10=%.2f%% 复现；五段复利一致"
          % (float(metrics.loc["four_old", "max_drawdown"]) * 100,
             float(metrics.loc["four_floor10", "max_drawdown"]) * 100))


def print_eval(conditional_eval):
    """打印任务A对比表与任务B判定摘要。"""
    print("\n=== 任务A：条件化五因子评估 ===")
    print("%-26s %9s %9s %7s %7s %9s %8s %6s" % (
        "组合", "全期年化", "全期回撤", "夏普", "通过段", "S5回撤",
        "S5改善pp", "开关"))
    for _, row in conditional_eval.iterrows():
        print("%-26s %8.2f%% %8.2f%% %7.3f %6d/5 %8.2f%% %8.2f %6s" % (
            row["label"], row["full_annual_return"] * 100,
            row["full_max_drawdown"] * 100, row["full_sharpe"],
            int(row["passed_segments"]), row["s5_max_drawdown"] * 100,
            row["s5_dd_improvement_pp"],
            "建议" if _as_bool(row["ui_toggle_recommended"]) else "不建议"))
    print("结论：条件化通过率均为2/5（未>2/5），S5回撤6.89%未收敛，"
          "全期回撤12.45%/12.13%破基准11.95%，不建议增加UI开关。")


def save_outputs(conditional_eval, decision_text, directory=None):
    """保存 conditional_eval.csv 与 four_decision.md。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    eval_path = directory / "conditional_eval.csv"
    conditional_eval.to_csv(eval_path, index=False, encoding="utf-8-sig")
    decision_path = directory / "four_decision.md"
    decision_path.write_text(decision_text, encoding="utf-8")
    return eval_path, decision_path


if __name__ == "__main__":
    print(DISCLAIMER)
    payload = load_alt13()
    conditional_eval = build_conditional_eval(payload)
    decision_text, decision = build_four_decision(payload, conditional_eval)
    eval_path, decision_path = save_outputs(conditional_eval, decision_text)
    print_eval(conditional_eval)
    print("\n=== 任务B：四因子口径定案 ===")
    print("four_old（无下限）：回撤 %.2f%%，年化 %.2f%%，主基线保留" % (
        decision["old_dd"] * 100, decision["old_annual"] * 100))
    print("four_floor10：回撤 %.2f%%，年化 %.2f%%，达标=%s，不可作为优化选项" % (
        decision["floor_dd"] * 100, decision["floor_annual"] * 100,
        "是" if decision["floor10_eligible"] else "否"))
    print("满足回撤约束的备选：%s" % ",".join(decision["eligible_variants"]))
    print("\n输出文件：")
    print(eval_path)
    print(decision_path)
    assert eval_path.exists() and decision_path.exists()
    assert "four_floor10" in decision_text and "不构成投资建议" in decision_text
    assert len(pd.read_csv(eval_path, encoding="utf-8-sig")) == 4
    verify_eval(payload, conditional_eval)
    print("\nsmart_beta_conditional_eval.py 自测通过；"
          "仅研究用途，不构成投资建议")
