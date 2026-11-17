# -*- coding: utf-8 -*-
"""全路线终版汇总 HTML + 最新一期盯盘清单导出（Step I2-alt-19）。

任务 A：生成离线 summary_report.html，整合 I2-alt-1~18 的全部路线：
- 双因子 / 三因子基线；
- 低 PE / 低 PB 单因子；
- 四因子（旧规则 vs 动量下限变体）；
- 五因子（全时下限 vs 条件化 vs 不启用）；
- 五段稳定性、成本 1x/2x 敏感性、并集修复回顾、动量下限敏感性；
- 五因子 vs 四因子当期对比（与持仓盯盘页一致）；
- 每个方案标注判定、年化、最大回撤、夏普与局限。

任务 B：把最新一期调仓（2026-10-30 → 2026-11-13）的入选股导出为
只读盯盘清单（CSV + HTML 表格），字段含代码、名称、组合、因子归属、
综合名次和、动量、波动、PE/PB、权重、下期收益。

口径说明：
- 所有绩效直接读取既有的落盘报告 CSV/JSON，不在本脚本重算历史；
- 仅最新一期持仓会用核心选股函数复算一次，用于补全因子归属与综合名次，
  并与落盘持仓权重做 1e-9 一致性断言；
- HTML 复用 core.export_html 的墨水屏样式与文档骨架，纯静态、无 JavaScript、
  无外链、无图片、不联网；
- 不接券商、不下单、不上传云端；仅研究用途，不构成投资建议。
- 不改动任何既有核心模块，只新增本脚本与输出文件。
"""

import sys
import json
import html
import hashlib
from pathlib import Path
from datetime import datetime
from html.parser import HTMLParser

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.export_html import (
    DISCLAIMER, _document, _escape, _format, _table_html, _write_html,
)
from core.factor_analysis import load_all_daily
from core.smart_beta import (
    DEFAULT_PCT, DEFAULT_VOL_WINDOW,
    DEFAULT_MOMENTUM_WINDOW, DEFAULT_MOMENTUM_SKIP,
)
from core.smart_beta_backtest import SMART_BETA_DIR
from core.smart_beta_fundamental_backtest import build_fundamental_composite
from core.valuation_data import load_valuation

REPORT_NAME = "summary_report.html"
WATCHLIST_PREFIX = "watchlist"
FINAL_FLOOR = 0.10

# 只读保护清单：本脚本只允许读取这些核心模块，任何写入都会触发自测失败
GUARDED_MODULES = (
    "db.py", "factors.py", "strategy.py", "backtest.py", "ml_model.py",
    "export_html.py", "backup.py", "logger.py", "smart_beta.py",
    "smart_beta_backtest.py",
)

SLEEVE_LABELS = {
    "low_price": "低价", "low_vol": "低波", "momentum": "动量",
    "low_pe": "低PE", "low_pb": "低PB",
}
PORTFOLIO_LABELS = {
    "five_factor": "五因子主基线（动量≥10%）",
    "four_factor": "四因子主基线（旧并集规则）",
}
PORTFOLIO_SHORT = {
    "five_factor": "五因子主基线",
    "four_factor": "四因子主基线",
}
WATCHLIST_CSV_COLUMNS = [
    ("code", "代码"), ("name", "名称"), ("portfolio_label", "组合"),
    ("factor_tags", "因子归属"), ("rank_sum", "综合百分位和"),
    ("momentum", "动量"), ("volatility", "年化波动"),
    ("pe_ttm", "PE(TTM)"), ("pb_mrq", "PB(MRQ)"),
    ("weight", "权重"), ("period_return", "下期收益"),
    ("date", "调仓日"), ("next_date", "收益截止日"),
]
WATCHLIST_HTML_COLUMNS = [
    ("code", "代码"), ("name", "名称"), ("portfolio_short", "组合"),
    ("factor_tags", "因子归属"), ("rank_sum", "综合百分位和"),
    ("momentum", "动量"), ("volatility", "年化波动"),
    ("pe_ttm", "PE(TTM)"), ("pb_mrq", "PB(MRQ)"),
    ("weight", "权重"), ("period_return", "下期收益"),
]

# 补充样式：全路线总表与盯盘表采用固定列宽，允许长文本换行，避免横向截断
EXTRA_CSS = """
<style>
.route-table table { table-layout: fixed; }
.route-table td, .route-table th { white-space: normal; }
.route-table td:first-child, .route-table th:first-child { width: 17%; text-align: left; }
.route-table td:last-child, .route-table th:last-child {
  width: 30%; text-align: left;
}
.watch-table table { table-layout: fixed; font-size: 12px; }
.watch-table td, .watch-table th { white-space: normal; padding: 4px 3px; }
.watch-table th:nth-child(1), .watch-table td:nth-child(1) { width: 7%; }
.watch-table th:nth-child(2), .watch-table td:nth-child(2) { width: 8%; text-align: left; }
.watch-table th:nth-child(3), .watch-table td:nth-child(3) { width: 13%; text-align: left; }
.watch-table th:nth-child(4), .watch-table td:nth-child(4) { width: 18%; text-align: left; }
.watch-table th:nth-child(5), .watch-table td:nth-child(5) { width: 8%; }
.watch-table th:nth-child(6), .watch-table td:nth-child(6) { width: 8%; }
.watch-table th:nth-child(7), .watch-table td:nth-child(7) { width: 8%; }
.watch-table th:nth-child(8), .watch-table td:nth-child(8) { width: 7%; }
.watch-table th:nth-child(9), .watch-table td:nth-child(9) { width: 7%; }
.watch-table th:nth-child(10), .watch-table td:nth-child(10) { width: 8%; }
.watch-table th:nth-child(11), .watch-table td:nth-child(11) { width: 8%; }
.compare-table table { table-layout: fixed; }
.compare-table td, .compare-table th { white-space: normal; text-align: left; }
.compare-table th:nth-child(1), .compare-table td:nth-child(1) { width: 24%; }
.compare-table th:nth-child(2), .compare-table td:nth-child(2) { width: 38%; }
.compare-table th:nth-child(3), .compare-table td:nth-child(3) { width: 38%; }
</style>
"""


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def _latest_file(directory, pattern):
    """按修改时间取最新匹配文件，避免硬编码日期。"""
    files = sorted(Path(directory).glob(pattern), key=lambda item: item.stat().st_mtime)
    if not files:
        raise FileNotFoundError("未找到匹配 %s 的报告文件" % pattern)
    return files[-1]


def _latest_csv(directory, pattern):
    return pd.read_csv(_latest_file(directory, pattern), encoding="utf-8-sig")


def load_reports(directory=None):
    """读取全路线落盘报告，返回统一字典。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    nav_payload = json.loads(
        _latest_file(directory, "smart_beta_baseline_*_nav.json").read_text(
            encoding="utf-8"))
    nav = pd.DataFrame(nav_payload["full_nav"])
    nav.index = pd.to_datetime(nav.index)
    return {
        "directory": directory,
        "baseline_full": _latest_csv(directory, "smart_beta_baseline_*_full.csv"),
        "baseline_segments": _latest_csv(directory, "smart_beta_baseline_*_segments.csv"),
        "baseline_stability": _latest_csv(directory, "smart_beta_baseline_*_stability.csv"),
        "fundamental": _latest_csv(directory, "smart_beta_fundamental_*_300.csv"),
        "cost": _latest_csv(directory, "smart_beta_cost_backtest_*_sensitivity.csv"),
        "floor": _latest_csv(directory, "smart_beta_floor_sensitivity_*_sensitivity.csv"),
        "conditional_summary": _latest_csv(
            directory, "smart_beta_conditional_floor_*_conditional_summary.csv"),
        "conditional_segments": _latest_csv(
            directory, "smart_beta_conditional_floor_*_segments.csv"),
        "four_reconciliation": _latest_csv(
            directory, "smart_beta_conditional_floor_*_four_reconciliation.csv"),
        "four_variants": _latest_csv(
            directory, "smart_beta_conditional_floor_*_four_variants.csv"),
        "union_full": _latest_csv(directory, "smart_beta_union_fix_*_full.csv"),
        "union_s4": _latest_csv(directory, "smart_beta_union_fix_*_s4.csv"),
        "walkforward_stability": _latest_csv(
            directory, "smart_beta_walkforward_*_stability.csv"),
        "nav": nav,
    }


# ---------------------------------------------------------------------------
# 最新一期盯盘清单
# ---------------------------------------------------------------------------
def _latest_holdings(directory, portfolio):
    """读取固化基线持仓 parquet 中指定组合的最新一期。"""
    path = _latest_file(directory, "smart_beta_baseline_*_holdings.parquet")
    frame = pd.read_parquet(path)
    latest = frame["date"].max()
    subset = frame[(frame["date"] == latest) & (frame["portfolio"] == portfolio)]
    if subset.empty:
        raise ValueError("最新一期缺少 %s 持仓" % portfolio)
    return subset.reset_index(drop=True), latest


def _stock_names():
    return {row["code"]: row["name"] for row in db.query_all(
        "SELECT code, name FROM stock_info")}


def _score_value(scores, column, code):
    """组合评分列可能整体缺失（如四因子无低 PB），统一返回 NaN。"""
    if column not in scores or scores[column] is None:
        return np.nan
    value = scores[column].get(code, np.nan)
    return np.nan if value is None else float(value)


def _factor_tags(code, sleeves):
    """返回该股命中的单因子子组合名称，至少一个。"""
    tags = [SLEEVE_LABELS[factor] for factor, members in sleeves.items()
            if code in set(members)]
    return "、".join(tags)


def build_watchlist(directory=None, verify=True):
    """复算最新一期四/五因子组合，生成只读盯盘清单。

    复算只用于补全因子归属与综合名次，并与落盘权重逐值核对；
    不重新计算历史收益，不改变任何基线结果。
    """
    directory = Path(directory) if directory else SMART_BETA_DIR
    five_holdings, trade_date = _latest_holdings(directory, "five_factor")
    four_holdings, four_date = _latest_holdings(directory, "four_factor")
    if four_date != trade_date:
        raise ValueError("四/五因子最新调仓日不一致：%s vs %s" % (four_date, trade_date))
    next_date = str(five_holdings["next_date"].iloc[0])

    codes = [row["code"] for row in db.query_all(
        "SELECT DISTINCT code FROM daily_bars WHERE fq=? ORDER BY code", ("qfq",))]
    data_map = load_all_daily(codes, fq="qfq")
    if not data_map:
        raise ValueError("无可用本地日线数据")
    valuation_map = {code: load_valuation(code) for code in data_map}
    missing = [code for code, frame in valuation_map.items() if frame.empty]
    if missing:
        raise ValueError("以下股票缺少本地估值数据：%s" % ",".join(missing[:10]))

    five = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, True, DEFAULT_PCT,
        DEFAULT_VOL_WINDOW, DEFAULT_MOMENTUM_WINDOW, DEFAULT_MOMENTUM_SKIP,
        momentum_floor=FINAL_FLOOR)
    four = build_fundamental_composite(
        data_map, valuation_map, trade_date, True, False, DEFAULT_PCT,
        DEFAULT_VOL_WINDOW, DEFAULT_MOMENTUM_WINDOW, DEFAULT_MOMENTUM_SKIP,
        momentum_floor=None)

    names = _stock_names()
    rows = []
    for portfolio, exposure, holdings in (
            ("five_factor", five, five_holdings), ("four_factor", four, four_holdings)):
        returns = holdings.set_index("code")["period_return"]
        scores = exposure["scores"]
        for code, weight in exposure["weights"].sort_values(
                ascending=False).items():
            rows.append({
                "code": code,
                "name": names.get(code, ""),
                "portfolio": portfolio,
                "portfolio_label": PORTFOLIO_LABELS[portfolio],
                "portfolio_short": PORTFOLIO_SHORT[portfolio],
                "factor_tags": _factor_tags(code, exposure["sleeves"]),
                "rank_sum": _score_value(scores, "rank_sum", code),
                "momentum": _score_value(scores, "momentum", code),
                "volatility": _score_value(scores, "volatility", code),
                "pe_ttm": _score_value(scores, "pe_ttm", code),
                "pb_mrq": _score_value(scores, "pb_mrq", code),
                "weight": float(weight),
                "period_return": float(returns.get(code, np.nan)),
                "date": str(trade_date),
                "next_date": str(next_date),
            })
    frame = pd.DataFrame(rows)
    if verify:
        for exposure, holdings in ((five, five_holdings), (four, four_holdings)):
            stored = holdings.set_index("code")["weight"]
            joined = pd.concat(
                [exposure["weights"].rename("calc"), stored.rename("stored")],
                axis=1, join="outer")
            assert joined.notna().all().all(), "复算权重与落盘持仓存在缺失"
            assert float((joined["calc"] - joined["stored"]).abs().max()) < 1e-9, (
                "复算权重与落盘持仓不一致")
    return frame, str(trade_date), next_date


def save_watchlist_csv(frame, path):
    """导出看盘友好的 CSV：百分比字段换算为百分数，表头中文。"""
    csv_frame = frame[[key for key, _label in WATCHLIST_CSV_COLUMNS]].copy()
    csv_frame.columns = [label for _key, label in WATCHLIST_CSV_COLUMNS]
    percent_labels = ("动量", "年化波动", "权重", "下期收益")
    for column in percent_labels:
        csv_frame[column] = (csv_frame[column].astype(float) * 100).round(4)
    csv_frame["综合百分位和"] = csv_frame["综合百分位和"].astype(float).round(6)
    csv_frame.columns = [
        "%s(%%)" % label if label in percent_labels else label
        for label in csv_frame.columns]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


# ---------------------------------------------------------------------------
# 静态 SVG：多方案净值对比（无脚本、无动画、无外链）
# ---------------------------------------------------------------------------
def _nav_svg(nav, series, labels, title, width=960, height=300):
    frame = nav[list(series)].dropna(how="all")
    if frame.empty:
        return "<p>暂无净值数据。</p>"
    left, right, top, bottom = 70, 200, 38, 42
    plot_width, plot_height = width - left - right, height - top - bottom
    values = frame.to_numpy(dtype=float)
    minimum, maximum = float(np.nanmin(values)), float(np.nanmax(values))
    padding = (maximum - minimum) * 0.08 or max(abs(maximum) * 0.01, 0.01)
    minimum, maximum = minimum - padding, maximum + padding
    value_range = maximum - minimum
    count = len(frame)

    def x_at(index):
        return left + index / max(count - 1, 1) * plot_width

    def y_at(value):
        return top + (maximum - value) / value_range * plot_height

    parts = [
        '<svg class="chart" viewBox="0 0 %d %d" role="img" aria-label="%s">'
        % (width, height, _escape(title)),
        "<title>%s</title>" % _escape(title),
        '<text x="%d" y="21" font-size="13" font-weight="bold">%s</text>'
        % (left, _escape(title)),
    ]
    for index in range(5):
        value = minimum + value_range * index / 4
        y = y_at(value)
        parts.append(
            '<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#888" '
            'stroke-width="1" />' % (left, y, width - right, y))
        parts.append(
            '<text x="%d" y="%.2f" text-anchor="end" font-size="11">%.3f</text>'
            % (left - 8, y + 4, value))
    dash_styles = ("", "6 4", "2 3", "10 4 2 4")
    for order, name in enumerate(series):
        dash = dash_styles[order % len(dash_styles)]
        dash_attr = ' stroke-dasharray="%s"' % dash if dash else ""
        points = []
        for index, value in enumerate(frame[name].to_numpy(dtype=float)):
            if np.isfinite(value):
                points.append("%.2f,%.2f" % (x_at(index), y_at(value)))
        parts.append(
            '<polyline fill="none" stroke="#000" stroke-width="1.8"%s '
            'points="%s" />' % (dash_attr, " ".join(points)))
        legend_y = top + 14 + order * 20
        parts.append(
            '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#000" '
            'stroke-width="1.8"%s />'
            % (width - right + 14, legend_y, width - right + 42, legend_y, dash_attr))
        parts.append(
            '<text x="%d" y="%d" font-size="12">%s</text>'
            % (width - right + 48, legend_y + 4, _escape(labels[name])))
    parts.append(
        '<text x="%d" y="%d" font-size="11">%s</text>'
        % (left, height - 10, _escape(str(frame.index[0].date()))))
    parts.append(
        '<text x="%d" y="%d" text-anchor="end" font-size="11">%s</text>'
        % (width - right, height - 10, _escape(str(frame.index[-1].date()))))
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 呈现辅助
# ---------------------------------------------------------------------------
def _pass_text(value):
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "—"
    return "通过" if bool(value) else "未通过"


def _yes_no(value):
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "—"
    return "是" if bool(value) else "否"


def _pp(value):
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "—" if pd.isna(number) else "%+.2fpp" % (float(number) * 100)


def _floor_text(value):
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "无" if pd.isna(number) else "%.0f%%" % (float(number) * 100)


def _cards(items):
    cards = "".join(
        '<div class="metric"><div class="metric-label">%s</div>'
        '<div class="metric-value">%s</div></div>'
        % (_escape(label), _escape(value)) for label, value in items)
    return '<div class="metrics">%s</div>' % cards


def _section(title, content, note=None):
    block = "<section><h2>%s</h2>" % _escape(title)
    if note:
        block += '<p class="note">%s</p>' % _escape(note)
    return block + content + "</section>"


def _route_frame(data):
    """汇总全部路线为一张表；判定统一为“通过/未通过”。"""
    base = data["baseline_full"].set_index("portfolio")
    fund = data["fundamental"].set_index("portfolio")
    variants = data["four_variants"].set_index("variant")
    cond = data["conditional_summary"].set_index("portfolio")
    benchmark_row = fund.loc["benchmark"]
    benchmark_total = float(benchmark_row["total_return"])
    benchmark_dd = float(benchmark_row["max_drawdown"])

    def add(rows, label, category, total, annual, drawdown, sharpe, passed, note,
            judged=True):
        rows.append({
            "label": label, "category": category,
            "total_return": total, "annual_return": annual,
            "max_drawdown": drawdown, "sharpe": sharpe,
            "excess": total - benchmark_total,
            "dd_vs_benchmark": drawdown - benchmark_dd,
            "verdict": (_pass_text(passed) if judged else "基准，不作判定"),
            "note": note,
        })

    rows = []
    add(rows, "双因子 50:50", "基础基线", base.loc["dual", "total_return"],
        base.loc["dual", "annual_return"], base.loc["dual", "max_drawdown"],
        base.loc["dual", "sharpe"], base.loc["dual", "passed"],
        "保守基线，回撤 11.86% 接近基准")
    add(rows, "三因子等权 33/33/34", "基础基线",
        base.loc["three_equal", "total_return"],
        base.loc["three_equal", "annual_return"],
        base.loc["three_equal", "max_drawdown"],
        base.loc["three_equal", "sharpe"], base.loc["three_equal", "passed"],
        "回撤 9.47%，为四/五因子的对照基准")
    add(rows, "低 PE 前 20%", "单因子", fund.loc["low_pe", "total_return"],
        fund.loc["low_pe", "annual_return"], fund.loc["low_pe", "max_drawdown"],
        fund.loc["low_pe", "sharpe"], fund.loc["low_pe", "passed"],
        "单因子中年化最高，但无动量暴露")
    add(rows, "低 PB 前 20%", "单因子", fund.loc["low_pb", "total_return"],
        fund.loc["low_pb", "annual_return"], fund.loc["low_pb", "max_drawdown"],
        fund.loc["low_pb", "sharpe"], fund.loc["low_pb", "passed"],
        "回撤 12.60% 破基准，不采用")
    add(rows, "四因子·旧并集规则", "四因子主基线",
        variants.loc["four_old", "total_return"],
        variants.loc["four_old", "annual_return"],
        variants.loc["four_old", "max_drawdown"],
        variants.loc["four_old", "sharpe"], variants.loc["four_old", "passed"],
        "四因子默认口径，年化 14.09%")
    add(rows, "四因子·动量≥5%", "四因子可选",
        variants.loc["four_floor05", "total_return"],
        variants.loc["four_floor05", "annual_return"],
        variants.loc["four_floor05", "max_drawdown"],
        variants.loc["four_floor05", "sharpe"],
        variants.loc["four_floor05", "passed"],
        "满足回撤约束，仅作文档标注")
    add(rows, "四因子·动量≥10%", "四因子可选",
        variants.loc["four_floor10", "total_return"],
        variants.loc["four_floor10", "annual_return"],
        variants.loc["four_floor10", "max_drawdown"],
        variants.loc["four_floor10", "sharpe"],
        variants.loc["four_floor10", "passed"],
        "回撤 12.45% 破基准，不可选")
    add(rows, "四因子·动量≥10%·仅保只数", "四因子可选",
        variants.loc["four_floor10_slots", "total_return"],
        variants.loc["four_floor10_slots", "annual_return"],
        variants.loc["four_floor10_slots", "max_drawdown"],
        variants.loc["four_floor10_slots", "sharpe"],
        variants.loc["four_floor10_slots", "passed"],
        "满足约束，仅作文档标注")
    add(rows, "四因子·并集扩至 40 只", "四因子可选",
        variants.loc["four_top40", "total_return"],
        variants.loc["four_top40", "annual_return"],
        variants.loc["four_top40", "max_drawdown"],
        variants.loc["four_top40", "sharpe"],
        variants.loc["four_top40", "passed"],
        "回撤最低 8.55%，但改变并集规模")
    add(rows, "五因子·全时 10% 下限", "五因子主基线",
        cond.loc["five_always", "total_return"],
        cond.loc["five_always", "annual_return"],
        cond.loc["five_always", "max_drawdown"],
        cond.loc["five_always", "sharpe"], cond.loc["five_always", "full_passed"],
        "五因子默认口径，年化 14.12%")
    add(rows, "五因子·不启用下限", "五因子参考",
        cond.loc["five_never", "total_return"],
        cond.loc["five_never", "annual_return"],
        cond.loc["five_never", "max_drawdown"],
        cond.loc["five_never", "sharpe"], cond.loc["five_never", "full_passed"],
        "参考口径，年化 14.72%")
    add(rows, "五因子·近 3 期 IC>0 条件化", "五因子条件化",
        cond.loc["five_cond_w3", "total_return"],
        cond.loc["five_cond_w3", "annual_return"],
        cond.loc["five_cond_w3", "max_drawdown"],
        cond.loc["five_cond_w3", "sharpe"],
        cond.loc["five_cond_w3", "full_passed"],
        "回撤 12.45% 破基准，不建议 UI 开关")
    add(rows, "五因子·近 6 期 IC>0 条件化", "五因子条件化",
        cond.loc["five_cond_w6", "total_return"],
        cond.loc["five_cond_w6", "annual_return"],
        cond.loc["five_cond_w6", "max_drawdown"],
        cond.loc["five_cond_w6", "sharpe"],
        cond.loc["five_cond_w6", "full_passed"],
        "回撤 12.13% 破基准，不建议 UI 开关")
    add(rows, "池内等权基准", "基准", benchmark_row["total_return"],
        benchmark_row["annual_return"], benchmark_row["max_drawdown"],
        benchmark_row["sharpe"], False, "300 只股票池等权，回撤 11.95%",
        judged=False)
    return pd.DataFrame(rows)


def build_watch_compare(watchlist, five_stats, four_stats, benchmark_drawdown):
    """当期清单对比：五因子 vs 四因子（指标、重叠、独有个股）。

    watchlist 为 build_watchlist 的输出；five_stats/four_stats 至少包含
    annual / drawdown / sharpe / passed 四项，均为小数或布尔值。
    """
    five = watchlist[watchlist["portfolio"] == "five_factor"]
    four = watchlist[watchlist["portfolio"] == "four_factor"]
    if five.empty or four.empty:
        raise ValueError("盯盘清单缺少五因子或四因子组合")

    five_codes = set(five["code"])
    four_codes = set(four["code"])
    overlap = five_codes & four_codes

    def weighted_return(frame):
        return float((frame["weight"] * frame["period_return"]).sum())

    def updown(frame):
        return "%d / %d" % (
            int((frame["period_return"] > 0).sum()),
            int((frame["period_return"] < 0).sum()))

    def unique_names(frame, exclude):
        subset = frame[~frame["code"].isin(exclude)].sort_values(
            "weight", ascending=False)
        return "、".join(
            "%s %.2f%%" % (row["name"], float(row["weight"]) * 100)
            for _, row in subset.iterrows()) or "—"

    def ratio(value):
        return "%.2f%%" % (float(value) * 100)

    def verdict(stats):
        if not stats["passed"]:
            return "未通过"
        return "通过" if float(stats["annual"]) > 0 else "未通过"

    rows = [
        {"item": "组合规则", "five": "并集修复 + 动量最低 10%",
         "four": "旧并集规则（不设动量下限）"},
        {"item": "当期持仓数", "five": "%d 只" % len(five),
         "four": "%d 只" % len(four)},
        {"item": "当期权重合计", "five": ratio(five["weight"].sum()),
         "four": ratio(four["weight"].sum())},
        {"item": "当期加权下期收益",
         "five": "%+.2f%%" % (weighted_return(five) * 100),
         "four": "%+.2f%%" % (weighted_return(four) * 100)},
        {"item": "当期上涨 / 下跌", "five": updown(five), "four": updown(four)},
        {"item": "与对方重叠", "five": "%d 只" % len(overlap),
         "four": "%d 只" % len(overlap)},
        {"item": "组合独有", "five": unique_names(five, four_codes),
         "four": unique_names(four, five_codes)},
        {"item": "全期年化收益", "five": ratio(five_stats["annual"]),
         "four": ratio(four_stats["annual"])},
        {"item": "全期最大回撤", "five": ratio(five_stats["drawdown"]),
         "four": ratio(four_stats["drawdown"])},
        {"item": "基准最大回撤", "five": ratio(benchmark_drawdown),
         "four": ratio(benchmark_drawdown)},
        {"item": "全期夏普", "five": "%.3f" % float(five_stats["sharpe"]),
         "four": "%.3f" % float(four_stats["sharpe"])},
        {"item": "判定（年化>0 且回撤≤11.95%）", "five": verdict(five_stats),
         "four": verdict(four_stats)},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 报告主体
# ---------------------------------------------------------------------------
def _render(data, watchlist, trade_date, next_date):
    base = data["baseline_full"].set_index("portfolio")
    benchmark = data["fundamental"].set_index("portfolio").loc["benchmark"]
    cond = data["conditional_summary"].set_index("portfolio")

    sections = []

    # 1. 结论速览
    overview = _cards([
        ("池内等权基准年化", _format(benchmark["annual_return"], "percent")),
        ("基准最大回撤", _format(benchmark["max_drawdown"], "percent")),
        ("五因子主基线年化", _format(cond.loc["five_always", "annual_return"], "percent")),
        ("五因子主基线回撤", _format(cond.loc["five_always", "max_drawdown"], "percent")),
        ("四因子主基线年化", _format(base.loc["four_factor", "annual_return"], "percent")),
        ("四因子主基线回撤", _format(base.loc["four_factor", "max_drawdown"], "percent")),
        ("五因子五段通过", "2/5"),
        ("最新调仓日", trade_date),
    ])
    sections.append(_section(
        "结论速览", overview,
        "判定口径：年化收益 > 0 且最大回撤不超过同期池内等权基准（11.95%）。"
        "全部收益为毛收益，成本影响见“成本敏感性”一节。"))

    # 1.1 版本与最终口径（收口说明）
    decisions = (
        "<ul>"
        "<li><b>主基线</b>：五因子 + 并集修复 + 动量最低 10% 权重；"
        "全期年化 14.12%、最大回撤 10.76%，1x/2x 成本下均通过回撤约束。</li>"
        "<li><b>四因子</b>：定案为旧并集规则（年化 14.09%、回撤 10.72%）；"
        "动量下限变体仅作可选对照，不替代主基线。</li>"
        "<li><b>条件化下限</b>：近 3/6 期 IC&gt;0 两档全期回撤均破基准，"
        "不增设 UI 开关。</li>"
        "<li><b>稳定性</b>：五段通过 2/5，S4 为共同亏损段，全期年化不可外推。</li>"
        "<li><b>盯盘</b>：持仓盯盘页只读接入最新清单与五/四因子对比表，"
        "不接券商、不下单、不上传云端。</li>"
        "<li><b>报告版本</b>：I2-alt-19 终版，整合 I2-alt-1~18 全部路线与里程碑。</li>"
        "</ul>")
    sections.append(_section(
        "版本与最终口径（I2-alt-19）", decisions,
        "本报告整合既有落盘结果，不重跑回测；全部数字以 data/smart_beta 下 CSV/JSON 为准。"))

    # 2. 净值对比
    nav_series = ("five_factor", "four_factor", "dual", "benchmark")
    nav_labels = {
        "five_factor": "五因子主基线", "four_factor": "四因子主基线",
        "dual": "双因子 50:50", "benchmark": "池内等权基准",
    }
    sections.append(_section(
        "全期净值对比",
        _nav_svg(data["nav"], nav_series, nav_labels, "全期净值（起点=1.000）"),
        "月末调仓、毛收益口径；净值仅作路线间相对比较，不代表可交易净值。"))

    # 3. 全路线总表
    route = _route_frame(data)
    sections.append(_section(
        "全路线总表",
        '<div class="route-table">%s</div>' % _table_html(route, [
            ("label", "方案"), ("category", "类别"),
            ("annual_return", "年化"), ("max_drawdown", "最大回撤"),
            ("sharpe", "夏普"), ("excess", "超额收益"),
            ("dd_vs_benchmark", "回撤差 vs 基准"),
            ("verdict", "判定"), ("note", "说明 / 局限"),
        ], {
            "annual_return": lambda value: _format(value, "percent"),
            "max_drawdown": lambda value: _format(value, "percent"),
            "sharpe": lambda value: _format(value, "number"),
            "excess": lambda value: _format(value, "percent"),
            "dd_vs_benchmark": _pp,
        }),
        "超额收益 = 累计收益 − 基准累计收益；回撤差为正表示回撤高于基准。"))

    # 4. 双/三因子基线
    baseline_rows = data["baseline_full"][
        data["baseline_full"]["portfolio"].isin(("dual", "three_equal"))]
    sections.append(_section(
        "双因子 / 三因子基线",
        _table_html(baseline_rows, [
            ("label", "方案"), ("total_return", "总收益"),
            ("annual_return", "年化"), ("sharpe", "夏普"),
            ("max_drawdown", "最大回撤"), ("win_rate", "胜率"),
            ("total_turnover", "总换手"), ("passed", "判定"),
        ], {
            "total_return": lambda value: _format(value, "percent"),
            "annual_return": lambda value: _format(value, "percent"),
            "sharpe": lambda value: _format(value, "number"),
            "max_drawdown": lambda value: _format(value, "percent"),
            "win_rate": lambda value: _format(value, "percent"),
            "total_turnover": lambda value: _format(value, "number"),
            "passed": _pass_text,
        }),
        "双因子为低价/低波各半仓，三因子为低价/低波/动量等权；"
        "两者均通过回撤约束，但年化低于四/五因子。"))

    # 5. 低 PE / 低 PB 单因子
    value_rows = data["fundamental"][
        data["fundamental"]["portfolio"].isin(("three_equal", "low_pe", "low_pb"))]
    sections.append(_section(
        "低 PE / 低 PB 单因子",
        _table_html(value_rows, [
            ("label", "方案"), ("total_return", "总收益"),
            ("annual_return", "年化"), ("sharpe", "夏普"),
            ("max_drawdown", "最大回撤"),
            ("annual_return_delta_vs_three", "年化差 vs 三因子"),
            ("max_drawdown_delta_vs_three", "回撤差 vs 三因子"),
            ("passed", "判定"),
        ], {
            "total_return": lambda value: _format(value, "percent"),
            "annual_return": lambda value: _format(value, "percent"),
            "sharpe": lambda value: _format(value, "number"),
            "max_drawdown": lambda value: _format(value, "percent"),
            "annual_return_delta_vs_three": _pp,
            "max_drawdown_delta_vs_three": _pp,
            "passed": _pass_text,
        }),
        "低 PE 前 20% 年化 12.81%、回撤 8.71%；低 PB 前 20% 回撤 12.60% 破基准，"
        "单因子均缺少动量暴露。"))

    # 6. 四因子定案
    four_recon = data["four_reconciliation"].copy()
    four_variants = data["four_variants"].copy()
    four_html = _table_html(four_recon, [
        ("item", "口径"), ("momentum_floor", "动量下限"),
        ("union_fix", "并集修复"), ("annual_return", "年化"),
        ("max_drawdown", "最大回撤"),
        ("dd_vs_benchmark_pp", "回撤 vs 基准(pp)"),
        ("passed", "判定"),
    ], {
        "momentum_floor": _floor_text,
        "union_fix": _yes_no,
        "annual_return": lambda value: _format(value, "percent"),
        "max_drawdown": lambda value: _format(value, "percent"),
        "dd_vs_benchmark_pp": lambda value: _format(value, "number"),
        "passed": _pass_text,
    })
    four_html += "<h3>四因子约束变体</h3>" + _table_html(four_variants, [
        ("label", "变体"), ("momentum_floor", "动量下限"),
        ("union_top_n", "并集规模"), ("annual_return", "年化"),
        ("max_drawdown", "最大回撤"),
        ("delta_dd_vs_old_pp", "回撤差 vs 旧规则(pp)"),
        ("meets_target", "回撤达标"), ("avg_momentum_weight", "平均动量权重"),
    ], {
        "momentum_floor": _floor_text,
        "annual_return": lambda value: _format(value, "percent"),
        "max_drawdown": lambda value: _format(value, "percent"),
        "delta_dd_vs_old_pp": lambda value: _format(value, "number"),
        "meets_target": _yes_no,
        "avg_momentum_weight": lambda value: _format(value, "percent"),
    })
    four_html += (
        '<p class="note">定案：四因子主基线保持 <b>旧并集规则</b>（年化 14.09%、'
        '回撤 10.72%）。叠加 10% 动量下限后年化降至 12.70%、回撤升至 12.45%，'
        '收益与回撤同时恶化，<b>不可作为优化选项</b>；5% 下限、仅保只数、'
        '并集扩至 40 只仅可在文档/UI 标注为可选，均不替代主基线。</p>')
    sections.append(_section("四因子口径定案", four_html))

    # 7. 五因子条件化
    five_names = ("five_always", "five_never", "five_cond_w3", "five_cond_w6")
    five_rows = data["conditional_summary"][
        data["conditional_summary"]["portfolio"].isin(five_names)].copy()
    five_rows["verdict"] = five_rows["full_passed"].map(_pass_text)
    five_rows["passed_segments_text"] = five_rows["passed_segments"].astype(str) + "/5"
    five_html = _table_html(five_rows, [
        ("label", "组合"), ("annual_return", "全期年化"),
        ("max_drawdown", "全期回撤"), ("sharpe", "夏普"),
        ("passed_segments_text", "五段通过"), ("floor_on_periods", "开启下限期数"),
        ("s5_annual_return", "S5 年化"), ("s5_max_drawdown", "S5 回撤"),
        ("verdict", "判定"),
    ], {
        "annual_return": lambda value: _format(value, "percent"),
        "max_drawdown": lambda value: _format(value, "percent"),
        "sharpe": lambda value: _format(value, "number"),
        "s5_annual_return": lambda value: _format(value, "percent"),
        "s5_max_drawdown": lambda value: _format(value, "percent"),
    })

    mark_rows = []
    for name in five_names:
        subset = data["conditional_segments"][
            data["conditional_segments"]["portfolio"] == name].sort_values("segment")
        row = {"label": cond.loc[name, "label"]}
        for _, record in subset.iterrows():
            row["S%d" % (int(record["segment"]) + 1)] = "%s %.2f%%" % (
                "✓" if bool(record["passed"]) else "✗",
                float(record["max_drawdown"]) * 100)
        row["passed_segments_text"] = "%d/5" % int(subset["passed"].sum())
        mark_rows.append(row)
    five_html += "<h3>五段判定明细</h3>" + _table_html(
        pd.DataFrame(mark_rows), [
            ("label", "组合"), ("S1", "S1"), ("S2", "S2"), ("S3", "S3"),
            ("S4", "S4"), ("S5", "S5"), ("passed_segments_text", "通过段"),
        ])
    five_html += (
        '<p class="note">条件化两档均只通过 2/5 段（未超过 2/5），S5 回撤 6.89% '
        '与全时启用相同（未收敛），全期回撤 12.45%/12.13% 破基准，'
        '因此不建议在 UI 增加“条件化启用”开关。✓/✗ 为分段判定，数值为段内最大回撤。</p>')
    sections.append(_section("五因子：全时 vs 条件化 vs 不启用", five_html))

    # 8. 五段稳定性
    stability = data["baseline_stability"]
    sections.append(_section(
        "五段稳定性（时间分段）",
        _table_html(stability, [
            ("label", "组合"), ("passed_segments", "通过段数"),
            ("compound_return", "复利收益"), ("annual_min", "最差段年化"),
            ("annual_max", "最好段年化"), ("worst_drawdown", "最差段回撤"),
            ("max_segment_log_share", "最大单段占比"), ("verdict", "结论"),
        ], {
            "compound_return": lambda value: _format(value, "percent"),
            "annual_min": lambda value: _format(value, "percent"),
            "annual_max": lambda value: _format(value, "percent"),
            "worst_drawdown": lambda value: _format(value, "percent"),
            "max_segment_log_share": lambda value: _format(value, "percent"),
        }),
        "“五段”为 2024-05-31 起的五个连续时间分段，并非严格样本外检验；"
        "双因子通过 4/5 段，三/四/五因子均只通过 2/5 段，S4（2025-11-28~2026-05-29）"
        "为共同亏损段，稳定性结论为“不稳定”，不应把全期年化外推为未来收益。"))

    # 9. 成本敏感性
    cost = data["cost"].copy()
    sections.append(_section(
        "成本敏感性（1x / 2x）",
        _table_html(cost, [
            ("label", "组合"), ("gross_annual_return", "毛收益年化"),
            ("annual_return_1x", "1x 年化"), ("max_drawdown_1x", "1x 回撤"),
            ("passed_1x", "1x 判定"), ("annual_return_2x", "2x 年化"),
            ("max_drawdown_2x", "2x 回撤"), ("passed_2x", "2x 判定"),
            ("annual_cost_drag_2x", "2x 成本拖累"),
        ], {
            "gross_annual_return": lambda value: _format(value, "percent"),
            "annual_return_1x": lambda value: _format(value, "percent"),
            "max_drawdown_1x": lambda value: _format(value, "percent"),
            "passed_1x": _pass_text,
            "annual_return_2x": lambda value: _format(value, "percent"),
            "max_drawdown_2x": lambda value: _format(value, "percent"),
            "passed_2x": _pass_text,
            "annual_cost_drag_2x": lambda value: _format(value, "percent"),
        }),
        "1x = 佣金万三 + 滑点 0.1%（单边 0.13%），2x = 万六 + 0.2%；"
        "三/四/五因子在 2x 成本下仍通过回撤约束，但年化拖累约 0.8~1.0pp。"))

    # 10. 并集修复回顾
    union = data["union_full"].copy()
    union_html = _table_html(union, [
        ("label", "组合"), ("total_return", "总收益"),
        ("annual_return", "年化"), ("sharpe", "夏普"),
        ("max_drawdown", "最大回撤"), ("win_rate", "胜率"),
        ("passed", "判定"),
    ], {
        "total_return": lambda value: _format(value, "percent"),
        "annual_return": lambda value: _format(value, "percent"),
        "sharpe": lambda value: _format(value, "number"),
        "max_drawdown": lambda value: _format(value, "percent"),
        "win_rate": lambda value: _format(value, "percent"),
        "passed": _pass_text,
    })
    union_html += "<h3>S4 亏损段修复效果</h3>" + _table_html(
        data["union_s4"], [
            ("pair", "对比"), ("old_total_return", "修复前 S4 收益"),
            ("new_total_return", "修复后 S4 收益"),
            ("delta_max_drawdown", "S4 回撤变化"),
            ("old_passed", "修复前判定"), ("new_passed", "修复后判定"),
        ], {
            "old_total_return": lambda value: _format(value, "percent"),
            "new_total_return": lambda value: _format(value, "percent"),
            "delta_max_drawdown": _pp,
            "old_passed": _pass_text,
            "new_passed": _pass_text,
        })
    union_html += (
        '<p class="note">并集修复指在并集分支强制动量股入选并保底权重，'
        '用于解决 S4 期间动量暴露被名次之和挤出为 0 的问题。修复改善了 S4 回撤，'
        '但也提高了全期动量暴露：四因子叠加 10% 下限后全期回撤反而破基准，'
        '故四因子维持旧规则，五因子采用 10% 下限。</p>')
    sections.append(_section("并集修复回顾", union_html))

    # 11. 动量下限敏感性
    floor = data["floor"].copy()
    floor["factor_label"] = floor["factor"].map({"four": "四因子", "five": "五因子"})
    floor["full_text"] = floor["full_passed"].map(_pass_text)
    floor["dd_text"] = floor["drawdown_ok"].map(_yes_no)
    sections.append(_section(
        "动量下限敏感性",
        _table_html(floor, [
            ("factor_label", "因子"), ("variant", "下限档"),
            ("floor", "下限参数"), ("annual_return", "年化"),
            ("max_drawdown", "最大回撤"), ("dd_text", "回撤达标"),
            ("full_text", "全期判定"), ("avg_momentum_weight", "平均动量权重"),
            ("s4_max_drawdown", "S4 回撤"),
        ], {
            "floor": lambda value: _format(value, "percent"),
            "annual_return": lambda value: _format(value, "percent"),
            "max_drawdown": lambda value: _format(value, "percent"),
            "avg_momentum_weight": lambda value: _format(value, "percent"),
            "s4_max_drawdown": lambda value: _format(value, "percent"),
        }),
        "四因子在各下限档回撤普遍高于基准，仅旧规则（不启用下限）兼顾收益与回撤；"
        "五因子 10% 档回撤 10.76% 达标，15% 档回撤 11.81% 仍达标但年化下降，"
        "20% 档破基准，30% 档 S5 年化转负。"))

    # 12. 最新一期盯盘清单
    watch_html = (
        '<p class="note">最新一期：调仓日 %s，收益统计截至 %s。'
        '本清单只读跟踪，不接券商、不下单、不上传云端。</p>'
        % (_escape(trade_date), _escape(next_date)))
    for portfolio in ("five_factor", "four_factor"):
        subset = watchlist[watchlist["portfolio"] == portfolio]
        watch_html += "<h3>%s（%d 只）</h3>" % (
            _escape(PORTFOLIO_LABELS[portfolio]), len(subset))
        watch_html += '<div class="watch-table">%s</div>' % _table_html(
            subset, WATCHLIST_HTML_COLUMNS, {
                "rank_sum": lambda value: _format(value, "number"),
                "momentum": lambda value: _format(value, "percent"),
                "volatility": lambda value: _format(value, "percent"),
                "pe_ttm": lambda value: _format(value, "number"),
                "pb_mrq": lambda value: _format(value, "number"),
                "weight": lambda value: _format(value, "percent"),
                "period_return": lambda value: _format(value, "percent"),
            })
    watch_html += (
        '<p class="note">综合百分位和为各单因子百分位名次之和，数值越低越靠前；'
        '因子归属表示该股命中的单因子子组合，可能同时命中多个。'
        '下期收益为 %s → %s 的区间收益，仅用于事后核对，不构成未来收益预测。</p>'
        % (_escape(trade_date), _escape(next_date)))
    sections.append(_section("最新一期盯盘清单", watch_html))

    # 13. 五因子 vs 四因子对比（与持仓盯盘页同口径）
    five_stats = {
        "annual": float(cond.loc["five_always", "annual_return"]),
        "drawdown": float(cond.loc["five_always", "max_drawdown"]),
        "sharpe": float(cond.loc["five_always", "sharpe"]),
        "passed": bool(cond.loc["five_always", "full_passed"]),
    }
    four_stats = {
        "annual": float(base.loc["four_factor", "annual_return"]),
        "drawdown": float(base.loc["four_factor", "max_drawdown"]),
        "sharpe": float(base.loc["four_factor", "sharpe"]),
        "passed": bool(base.loc["four_factor", "passed"]),
    }
    compare = build_watch_compare(
        watchlist, five_stats, four_stats, float(benchmark["max_drawdown"]))
    sections.append(_section(
        "五因子 vs 四因子对比（当期）",
        '<div class="compare-table">%s</div>' % _table_html(compare, [
            ("item", "指标"), ("five", "五因子主基线（动量≥10%）"),
            ("four", "四因子主基线（旧并集规则）"),
        ]),
        "当期：%s → %s；全期：2024-05-31 ~ 2026-11-13，300 只池，等权基准，毛收益。"
        "判定口径：年化 &gt; 0 且最大回撤不超过 11.95%%。"
        % (_escape(trade_date), _escape(next_date))))

    # 13. 局限
    limits = (
        "<ul>"
        "<li>区间为 2024-05-31 ~ 2026-11-13 的单一历史区间，300 只本地股票池，"
        "不能代表全市场或更长周期。</li>"
        "<li>“五段稳定性”是连续时间分段，不是严格样本外；各方案均只通过 2/5 段，"
        "S4 为共同亏损段。</li>"
        "<li>除成本敏感性一节外，其余绩效均为毛收益，未扣佣金、滑点、冲击成本与税费。</li>"
        "<li>月频调仓、月末收盘选股、下月月末收盘计收益；不处理停牌、涨跌停与成交约束。</li>"
        "<li>四因子叠加动量下限后收益与回撤同时恶化，说明简单提高动量暴露并非稳健改进；"
        "参数敏感性结论仅基于当前样本。</li>"
        "<li>盯盘清单为研究记录，不含交易指令；任何实盘决策需自行复核数据与风险。</li>"
        "</ul>")
    sections.append(_section("局限与使用说明", limits))

    title = "Smart Beta 全路线研究汇总（I2-alt-19 终版）"
    subtitle = (
        "数据范围：300 只本地股票，2024-05-31 ~ 2026-11-13，月末调仓｜"
        "生成时间：%s｜共 13 个方案路线｜仅研究用途，不构成投资建议"
        % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return _document(title, subtitle, EXTRA_CSS + "".join(sections))


# ---------------------------------------------------------------------------
# 导出入口
# ---------------------------------------------------------------------------
def build_summary_report(directory=None, output=None, watchlist_output=None):
    """生成汇总 HTML 与盯盘 CSV，返回输出路径与清单数据。"""
    directory = Path(directory) if directory else SMART_BETA_DIR
    data = load_reports(directory)
    watchlist, trade_date, next_date = build_watchlist(directory)
    content = _render(data, watchlist, trade_date, next_date)
    html_path = _write_html(content, output or (directory / REPORT_NAME))
    csv_name = "%s_%s_%s.csv" % (
        WATCHLIST_PREFIX, trade_date.replace("-", ""), next_date.replace("-", ""))
    csv_path = save_watchlist_csv(
        watchlist, watchlist_output or (directory / csv_name))
    return {
        "html": html_path,
        "watchlist_csv": csv_path,
        "watchlist": watchlist,
        "trade_date": trade_date,
        "next_date": next_date,
    }


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
class _StructureParser(HTMLParser):
    """统计标签并确保 HTML 可解析；不执行任何脚本。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = {}

    def handle_starttag(self, tag, attrs):
        self.tags[tag] = self.tags.get(tag, 0) + 1


def assert_offline_html(text):
    """HTML 渲染自测：无脚本、无外链、无图片、无动画。"""
    lowered = text.lower()
    forbidden = ("<script", "<link", "@import", "url(", "javascript:",
                 "http://", "https://", "src=", "<iframe", "@font-face",
                 "animation:", "@keyframes")
    for token in forbidden:
        assert token not in lowered, "HTML 含禁用内容：%s" % token
    parser = _StructureParser()
    parser.feed(text)
    assert parser.tags.get("table", 0) >= 8, "表格数量不足，报告结构异常"
    assert parser.tags.get("svg", 0) >= 1, "缺少静态 SVG 图表"
    assert DISCLAIMER in text
    return parser.tags


def assert_watchlist_fields(frame, trade_date, next_date):
    """盯盘清单字段完整性、权重归一与期间一致性断言。"""
    keys = {key for key, _label in WATCHLIST_CSV_COLUMNS}
    html_keys = {key for key, _label in WATCHLIST_HTML_COLUMNS}
    assert keys.issubset(set(frame.columns)), "盯盘清单缺少字段"
    assert html_keys.issubset(set(frame.columns)), "盯盘 HTML 缺少字段"
    assert len(frame) >= 60, "最新一期入选股数量异常"
    assert frame["code"].astype(str).str.fullmatch(r"\d{6}").all()
    assert frame["name"].astype(str).str.strip().ne("").all()
    assert frame["factor_tags"].astype(str).str.strip().ne("").all()
    for column in ("weight", "period_return", "rank_sum", "momentum", "volatility"):
        assert frame[column].notna().all(), "字段存在空值：%s" % column
    sums = frame.groupby("portfolio_label")["weight"].sum()
    assert ((sums - 1.0).abs() < 1e-9).all(), "组合权重未归一：%s" % sums.to_dict()
    assert (frame["date"] == trade_date).all()
    assert (frame["next_date"] == next_date).all()
    assert set(frame["portfolio"]) == {"five_factor", "four_factor"}


def _guard_snapshot():
    root = Path(__file__).resolve().parent
    snapshot = {}
    for name in GUARDED_MODULES:
        path = root / name
        snapshot[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def synthetic_nav_svg_test():
    """合成净值：确认 SVG 为纯静态、含图例且不含外部资源。"""
    index = pd.bdate_range("2026-01-01", periods=8)
    nav = pd.DataFrame({
        "a": np.linspace(1.0, 1.1, len(index)),
        "b": np.linspace(1.0, 1.05, len(index)),
    }, index=index)
    svg = _nav_svg(nav, ("a", "b"), {"a": "A", "b": "B"}, "合成<净值>")
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "polyline" in svg and "合成&lt;净值&gt;" in svg
    for token in ("http://", "https://", "<script", "animation"):
        assert token not in svg
    print("合成数据：静态 SVG 多方案净值图，转义与离线检查通过")


if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()
    synthetic_nav_svg_test()
    before = _guard_snapshot()

    result = build_summary_report()
    html_text = Path(result["html"]).read_text(encoding="utf-8")
    tags = assert_offline_html(html_text)
    assert_watchlist_fields(
        result["watchlist"], result["trade_date"], result["next_date"])

    csv_frame = pd.read_csv(result["watchlist_csv"], encoding="utf-8-sig")
    assert len(csv_frame) == len(result["watchlist"])
    for column in ("代码", "名称", "组合", "因子归属", "权重(%)", "下期收益(%)"):
        assert column in csv_frame.columns, "CSV 缺少列：%s" % column
    weight_sums = csv_frame.groupby("组合")["权重(%)"].sum()
    # 30 只股票各自四舍五入到 0.0001%，累计误差上限约 0.0015 个百分点
    assert ((weight_sums - 100.0).abs() < 1e-2).all(), weight_sums.to_dict()

    # 盯盘清单加权收益必须与固化基线最新一期组合收益一致
    periods = _latest_csv(
        Path(result["html"]).parent, "smart_beta_baseline_*_periods.csv")
    latest = periods[periods["date"] == result["trade_date"]].iloc[0]
    weighted_returns = {}
    for portfolio in ("five_factor", "four_factor"):
        subset = result["watchlist"][result["watchlist"]["portfolio"] == portfolio]
        weighted = float((subset["weight"] * subset["period_return"]).sum())
        expected = float(latest[portfolio + "_return"])
        assert abs(weighted - expected) < 1e-6, (portfolio, weighted, expected)
        weighted_returns[portfolio] = weighted

    after = _guard_snapshot()
    assert before == after, "核心模块在本次运行中被修改"

    # 五/四因子对比表必须与落盘清单和全期核对指标一致
    data = load_reports(Path(result["html"]).parent)
    cond = data["conditional_summary"].set_index("portfolio")
    base = data["baseline_full"].set_index("portfolio")
    bench_dd = float(
        data["fundamental"].set_index("portfolio").loc["benchmark", "max_drawdown"])
    compare = build_watch_compare(
        result["watchlist"],
        {
            "annual": cond.loc["five_always", "annual_return"],
            "drawdown": cond.loc["five_always", "max_drawdown"],
            "sharpe": cond.loc["five_always", "sharpe"],
            "passed": bool(cond.loc["five_always", "full_passed"]),
        },
        {
            "annual": base.loc["four_factor", "annual_return"],
            "drawdown": base.loc["four_factor", "max_drawdown"],
            "sharpe": base.loc["four_factor", "sharpe"],
            "passed": bool(base.loc["four_factor", "passed"]),
        },
        bench_dd,
    ).set_index("item")
    assert compare.loc["当期持仓数", "five"] == "30 只"
    assert compare.loc["与对方重叠", "five"] == "27 只"
    assert "南山铝业" in compare.loc["组合独有", "five"]
    assert compare.loc["当期加权下期收益", "five"] == "%+.2f%%" % (
        weighted_returns["five_factor"] * 100)
    assert compare.loc["当期加权下期收益", "four"] == "%+.2f%%" % (
        weighted_returns["four_factor"] * 100)
    assert compare.loc["全期年化收益", "five"] == "14.12%"
    assert compare.loc["全期年化收益", "four"] == "14.09%"
    assert compare.loc["全期最大回撤", "four"] == "10.72%"
    assert compare.loc["全期夏普", "five"] == "0.940"
    assert "五因子 vs 四因子对比（当期）" in html_text
    assert "27 只" in html_text

    print("\n最新一期盯盘清单：调仓日 %s，收益截至 %s，共 %d 行" % (
        result["trade_date"], result["next_date"], len(result["watchlist"])))
    for portfolio, subset in result["watchlist"].groupby("portfolio"):
        print("  %-14s %d 只｜权重合计 %.6f｜加权下期收益 %.4f%%" % (
            portfolio, len(subset), subset["weight"].sum(),
            weighted_returns[portfolio] * 100))
    print("HTML 结构：%d 张表、%d 个 SVG" % (
        tags.get("table", 0), tags.get("svg", 0)))
    print("五/四因子对比：全期 %s vs %s｜当期 %s vs %s｜重叠 %s" % (
        compare.loc["全期年化收益", "five"], compare.loc["全期年化收益", "four"],
        compare.loc["当期加权下期收益", "five"],
        compare.loc["当期加权下期收益", "four"],
        compare.loc["与对方重叠", "five"]))
    print("\n输出文件：")
    print(result["html"])
    print(result["watchlist_csv"])
    print("\n自测通过：HTML 无脚本/外链/图片，盯盘清单字段完整且权重归一，"
          "五/四因子对比表与落盘数据一致，核心模块哈希未变化；%s" % DISCLAIMER)
