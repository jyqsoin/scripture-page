# -*- coding: utf-8 -*-
"""
极简静态 HTML 导出，适配墨水屏和离线浏览。

- 仅 HTML + 内联 CSS + 静态 SVG；无 JavaScript、动画、外部字体、图片或网络资源；
- 不启动 HTTP 服务，不依赖 matplotlib，可直接用浏览器或墨水屏阅读器打开；
- 支持回测结果、已有回测导出目录和选股结果；
- 所有外部文本均进行 HTML 转义，所有报告标注“仅研究用途，不构成投资建议”。
"""

import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db

DISCLAIMER = "仅研究用途，不构成投资建议"

CSS = """
:root {
  --paper: #ffffff;        /* 背景：纯白，保证墨水屏最高对比度 */
  --paper-deep: #d4d4d4;   /* 表头底色：中灰，低灰阶屏上仍可与白底区分 */
  --stripe: #f0f0f0;       /* 斑马纹：浅灰，与表头拉开层次 */
  --ink: #000000;          /* 主文字/边框：纯黑 */
  --ink-soft: #333333;     /* 次要文字 */
  --rule: #000000;         /* 分隔线 */
}
* { box-sizing: border-box; }
html { background: var(--paper); color: var(--ink); }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Microsoft YaHei", "SimSun", sans-serif;
  line-height: 1.6;
  /* 极淡点状纸纹：点距 6px、透明度 0.03，普通屏有纸感，墨水屏几乎不可见不干扰阅读 */
  background-image: radial-gradient(rgba(0,0,0,0.03) 1px, transparent 1px);
  background-size: 6px 6px;
}
main { max-width: 1080px; margin: 0 auto; padding: 20px; background: var(--paper); }
h1 { margin: 0 0 8px; font-size: 25px; border-bottom: 3px solid var(--rule); padding-bottom: 8px; }
h2 { margin: 26px 0 8px; font-size: 19px; border-bottom: 2px solid var(--rule); padding-bottom: 5px; }
p { margin: 8px 0; }
.meta, .note { font-size: 13px; color: var(--ink-soft); }
.disclaimer { margin: 14px 0; padding: 10px; border: 2px solid var(--ink); font-weight: 700; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); border-top: 1px solid var(--rule); border-left: 1px solid var(--rule); }
.metric { min-height: 70px; padding: 8px; border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }
.metric-label { font-size: 13px; color: var(--ink-soft); }
.metric-value { margin-top: 5px; font-size: 20px; font-weight: 700; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 7px; border: 1px solid var(--rule); text-align: right; white-space: nowrap; }
th { background: var(--paper-deep); font-weight: 700; }
th:first-child, td:first-child { text-align: left; }
tbody tr:nth-child(even) { background: var(--stripe); }
.chart { display: block; width: 100%; height: auto; margin: 10px 0; border: 1px solid var(--rule); background: var(--paper); }
.code { font-family: Consolas, monospace; }
footer { margin-top: 28px; padding-top: 10px; border-top: 2px solid var(--rule); font-size: 13px; color: var(--ink-soft); }
@media (max-width: 600px) {
  main { padding: 12px; }
  h1 { font-size: 21px; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .metric-value { font-size: 17px; }
}
@media print {
  main { max-width: none; padding: 0; }
  body { background-image: none; }
  .metrics, section { break-inside: avoid; }
  tbody tr:nth-child(even) { background: #fff; }
  h2 { break-after: avoid; }
}
"""

METRICS = [
    ("初始资金", "initial_cash", "money"),
    ("期末权益", "final_value", "money"),
    ("总收益率", "total_return", "percent"),
    ("年化收益率", "annual_return", "percent"),
    ("夏普比率", "sharpe", "number"),
    ("最大回撤", "max_drawdown", "percent"),
    ("胜率", "win_rate", "percent"),
    ("平均盈亏比", "pl_ratio", "number"),
    ("盈利因子", "profit_factor", "number"),
    ("已平仓净盈亏", "net_pnl", "money"),
    ("已平仓交易", "trades_closed", "integer"),
    ("未平仓交易", "trades_open", "integer"),
]

TRADE_COLUMNS = [
    ("code", "代码"), ("buy_date", "买入日期"), ("buy_price", "买入价"),
    ("size", "数量"), ("sell_date", "卖出日期"), ("sell_price", "卖出价"),
    ("commission", "双边佣金"), ("pnl", "净盈亏"), ("return_pct", "收益率"),
    ("status", "状态"),
]


def _escape(value):
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _format(value, kind="text"):
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "—"
    number = _finite(value)
    if kind == "money":
        return "—" if number is None else "%.2f" % number
    if kind == "percent":
        return "—" if number is None else "%.2f%%" % (number * 100)
    if kind == "number":
        return "—" if number is None else "%.4f" % number
    if kind == "integer":
        return "—" if number is None else str(int(number))
    if kind == "price":
        return "—" if number is None else "%.3f" % number
    if kind == "percent_value":
        return "—" if number is None else "%.2f%%" % number
    return str(value)


def _document(title, subtitle, body):
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'none'; connect-src 'none'; img-src 'none'; font-src 'none'; form-action 'none'; base-uri 'none'">
<title>""" + _escape(title) + """</title>
<style>""" + CSS + """</style>
</head>
<body><main>
<h1>""" + _escape(title) + """</h1>
<p class="meta">""" + _escape(subtitle) + """</p>
<div class="disclaimer">""" + DISCLAIMER + "。历史结果不代表未来表现。" + """</div>
""" + body + """
<footer>""" + DISCLAIMER + """。本文件为完全静态页面，不联网、不执行脚本、不提供交易功能。</footer>
</main></body></html>
"""


def _write_html(content, path):
    path = Path(path)
    if path.suffix.lower() not in (".html", ".htm"):
        path = path.with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def _default_path(prefix):
    return db.EXPORT_DIR / (prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".html")


def _metrics_html(summary):
    cards = []
    for label, key, kind in METRICS:
        cards.append(
            '<div class="metric"><div class="metric-label">%s</div>'
            '<div class="metric-value">%s</div></div>'
            % (_escape(label), _escape(_format(summary.get(key), kind)))
        )
    extra = "交易日 %s｜盈利 %s 笔｜亏损 %s 笔" % (
        _escape(summary.get("periods", 0)), _escape(summary.get("wins", 0)), _escape(summary.get("losses", 0))
    )
    return '<section><h2>核心指标</h2><div class="metrics">%s</div><p class="note">%s</p></section>' % (
        "".join(cards), extra
    )


def _table_html(frame, columns, formatters=None):
    """columns 为 [(字段, 中文标题)]；缺失字段补空列，所有单元格默认转义。"""
    formatters = formatters or {}
    if frame is None:
        frame = pd.DataFrame()
    head = "".join("<th>%s</th>" % _escape(label) for _, label in columns)
    rows = []
    for _, record in frame.iterrows():
        cells = []
        for key, _label in columns:
            value = record.get(key) if key in frame.columns else None
            if key in formatters:
                value = formatters[key](value)
            elif value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
                value = ""
            css_class = ' class="code"' if key == "code" else ""
            cells.append("<td%s>%s</td>" % (css_class, _escape(value)))
        rows.append("<tr>%s</tr>" % "".join(cells))
    if not rows:
        rows.append('<tr><td colspan="%d">暂无记录</td></tr>' % len(columns))
    return '<div class="table-wrap"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (
        head, "".join(rows)
    )


def _curve_frame(curves, column):
    if curves is None or column not in curves.columns:
        return pd.DataFrame(columns=["date", "value"])
    dates = curves["date"] if "date" in curves.columns else curves.index.to_series()
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates, errors="coerce").to_numpy(),
        "value": pd.to_numeric(curves[column], errors="coerce").to_numpy(),
    })
    return frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _chart_svg(curves, column, title, percent=False, max_points=240):
    """直接生成静态 SVG；降采样仅影响展示，不改变已计算的绩效指标。"""
    max_points = int(max_points)
    if max_points < 2:
        raise ValueError("max_points 至少为 2")
    frame = _curve_frame(curves, column)
    if frame.empty:
        return "<p>暂无%s数据。</p>" % _escape(title)
    if len(frame) > max_points:
        positions = np.unique(np.linspace(0, len(frame) - 1, max_points, dtype=int))
        frame = frame.iloc[positions].reset_index(drop=True)

    width, height = 960, 250
    left, right, top, bottom = 68, 18, 32, 44
    plot_width, plot_height = width - left - right, height - top - bottom
    values = frame["value"].to_numpy(dtype=float)
    minimum, maximum = float(values.min()), float(values.max())
    padding = (maximum - minimum) * 0.08 or max(abs(maximum) * 0.01, 0.01)
    minimum, maximum = minimum - padding, maximum + padding
    value_range = maximum - minimum
    count = len(frame)

    def x_at(index):
        return left + (index / max(count - 1, 1)) * plot_width

    def y_at(value):
        return top + (maximum - value) / value_range * plot_height

    parts = [
        '<svg class="chart" viewBox="0 0 %d %d" role="img" aria-label="%s">' % (width, height, _escape(title)),
        "<title>%s</title>" % _escape(title),
        '<text x="%d" y="19" font-size="13" font-weight="bold">%s</text>' % (left, _escape(title)),
    ]
    for index in range(5):
        value = minimum + value_range * index / 4
        y = y_at(value)
        label = "%.2f%%" % (value * 100) if percent else "%.3f" % value
        parts.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#888" stroke-width="1" />' % (left, y, width - right, y))
        parts.append('<text x="%d" y="%.2f" text-anchor="end" font-size="11">%s</text>' % (left - 6, y + 4, label))
    if percent and minimum <= 0 <= maximum:
        parts.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#000" stroke-dasharray="5 4" />' % (left, y_at(0), width - right, y_at(0)))

    points = " ".join("%.2f,%.2f" % (x_at(index), y_at(value)) for index, value in enumerate(values))
    parts.append('<polyline points="%s" fill="none" stroke="#000" stroke-width="1.8" />' % points)
    if count == 1:
        parts.append('<circle cx="%.2f" cy="%.2f" r="3" fill="#000" />' % (x_at(0), y_at(values[0])))

    tick_count = min(6, count)
    tick_positions = np.unique(np.linspace(0, count - 1, tick_count, dtype=int))
    for index in tick_positions:
        label = frame.iloc[int(index)]["date"].strftime("%Y-%m-%d")
        parts.append('<text x="%.2f" y="%d" text-anchor="middle" font-size="11">%s</text>' % (x_at(int(index)), height - 14, label))
    parts.append("</svg>")
    return "".join(parts)


def _to_dataframe(value):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value:
        return pd.DataFrame(value)
    return pd.DataFrame()


def _read_csv(path, **kwargs):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_backtest_folder(output_dir):
    """从 Step 6 已导出的 JSON/CSV 重新组织报告数据，不读取网络或正式数据库。"""
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("回测汇总不存在：%s" % summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trades = _read_csv(output_dir / "trades.csv", dtype={"code": str})
    curves = _read_csv(output_dir / "equity_curve.csv")
    if "date" in curves.columns:
        curves["date"] = pd.to_datetime(curves["date"], errors="coerce")
        curves = curves.set_index("date")
    warnings_frame = _read_csv(output_dir / "warnings.csv")
    return {
        "summary": summary,
        "trades": trades,
        "curves": curves,
        "warnings": warnings_frame.to_dict("records"),
        "output_dir": str(output_dir),
    }


def export_backtest_html(result, path=None, max_trades=0):
    """
    导出回测静态报告。result 可为 run_backtest 返回值或已有回测导出目录。
    max_trades=0 表示全部交易；正数时仅显示前 N 笔，完整明细仍以 CSV 为准。
    """
    if isinstance(result, (str, Path)):
        result = load_backtest_folder(result)
    summary = result.get("summary") or {}
    trades = _to_dataframe(result.get("trades"))
    curves = result.get("curves")
    warnings = result.get("warnings") or []
    if path is None:
        path = Path(result["output_dir"]) / "report.html" if result.get("output_dir") else _default_path("backtest_report")

    loaded = summary.get("codes_loaded") or []
    skipped = summary.get("codes_skipped") or []
    overview = (
        "<section><h2>范围</h2><p>已加载：%s</p><p>已跳过：%s</p>"
        "<p>V1.0 默认规则：手续费万三、滑点 0.1%%、单票≤10%%、总仓≤80%%、5%% 止损、A 股 T+1；"
        "T 日收盘判定，T+1 日开盘成交。</p></section>"
        % (_escape(", ".join(map(str, loaded)) or "—"), _escape(", ".join(map(str, skipped)) or "—"))
    )
    body = [
        _metrics_html(summary), overview,
        "<section><h2>净值曲线</h2>" + _chart_svg(curves, "net_value", "组合净值") + "</section>",
        "<section><h2>回撤曲线</h2>" + _chart_svg(curves, "drawdown", "组合回撤", percent=True) + "</section>",
    ]
    shown_trades = trades
    if max_trades and max_trades > 0:
        shown_trades = trades.head(int(max_trades))
    trade_note = ""
    if len(shown_trades) < len(trades):
        trade_note = '<p class="note">仅显示前 %d 笔，共 %d 笔；完整明细请查看 trades.csv。</p>' % (len(shown_trades), len(trades))
    body.append(
        "<section><h2>交易明细</h2><p class=\"note\">open 表示期末仍持仓；收益率已扣除配对买卖双边佣金。</p>"
        + trade_note
        + _table_html(
            shown_trades, TRADE_COLUMNS,
            {
                "buy_price": lambda value: _format(value, "price"),
                "sell_price": lambda value: _format(value, "price"),
                "commission": lambda value: _format(value, "money"),
                "pnl": lambda value: _format(value, "money"),
                "return_pct": lambda value: _format(value, "percent"),
            },
        ) + "</section>"
    )
    if warnings:
        body.append(
            "<section><h2>规则拦截与警告</h2>"
            + _table_html(pd.DataFrame(warnings), [("date", "日期"), ("code", "代码"), ("action", "事件"), ("reason", "原因")])
            + "</section>"
        )
    title = str(summary.get("name") or "A 股本地回测报告")
    subtitle = "生成时间：%s｜交易日数：%s｜交易记录：%d 笔" % (
        db.now_str(), summary.get("periods", 0), len(trades)
    )
    return _write_html(_document(title, subtitle, "".join(body)), path)


def export_selection_html(rows, path=None, title="公式选股结果"):
    """将 Step 7 选股结果导出为墨水屏友好的静态表格。"""
    frame = _to_dataframe(rows)
    if path is None:
        path = _default_path("selection_report")
    columns = [("code", "代码"), ("name", "名称"), ("date", "日期"), ("close", "收盘价"), ("change_pct", "当日涨跌幅")]
    body = (
        "<section><h2>命中股票</h2><p>共 %d 只。结果仅为本地公式计算输出，不代表买卖建议。</p>%s</section>"
        % (
            len(frame),
            _table_html(
                frame, columns,
                {
                    "close": lambda value: _format(value, "price"),
                    "change_pct": lambda value: _format(value, "percent_value"),
                },
            ),
        )
    )
    dates = frame["date"].dropna().astype(str) if "date" in frame.columns and not frame.empty else pd.Series(dtype=str)
    subtitle = "生成时间：%s｜结果日期：%s" % (db.now_str(), dates.max() if not dates.empty else "—")
    return _write_html(_document(title, subtitle, body), path)


if __name__ == "__main__":
    # 临时目录合成自测：不联网、不写正式数据库、不在项目中遗留 HTML。
    import tempfile

    demo_dates = pd.bdate_range("2024-01-02", periods=40)
    demo_curves = pd.DataFrame({
        "net_value": np.linspace(1.0, 1.08, len(demo_dates)),
        "drawdown": np.minimum(0, np.linspace(0, 0.04, len(demo_dates)) - 0.02),
    }, index=demo_dates)
    demo_summary = {
        "name": "合成回测<静态测试>", "codes_loaded": ["000001", "600519"], "codes_skipped": [],
        "initial_cash": 1000000, "final_value": 1080000, "total_return": 0.08,
        "annual_return": 0.12, "sharpe": 1.2, "max_drawdown": 0.02, "periods": 40,
        "trades_closed": 1, "trades_open": 1, "wins": 1, "losses": 0,
        "win_rate": 1.0, "pl_ratio": None, "profit_factor": None, "net_pnl": 2000,
    }
    demo_trades = [
        {"code": "000001", "buy_date": "2024-01-03", "buy_price": 10, "size": 100,
         "sell_date": "2024-01-05", "sell_price": 11, "commission": 6.3,
         "pnl": 93.7, "return_pct": 0.0093, "status": "closed"},
        {"code": "600519", "buy_date": "2024-01-04", "buy_price": 1600, "size": 100,
         "sell_date": "", "sell_price": None, "commission": None,
         "pnl": None, "return_pct": None, "status": "open"},
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        backtest_path = export_backtest_html(
            {"summary": demo_summary, "curves": demo_curves, "trades": demo_trades, "warnings": []},
            Path(temp_dir) / "backtest.html",
        )
        selection_path = export_selection_html(
            [{"code": "000001", "name": "测试<股票>", "date": "2024-02-01", "close": 10.5, "change_pct": 1.2}],
            Path(temp_dir) / "selection.html",
        )
        backtest_text = Path(backtest_path).read_text(encoding="utf-8")
        selection_text = Path(selection_path).read_text(encoding="utf-8")
        assert "<svg" in backtest_text and DISCLAIMER in backtest_text
        assert "合成回测&lt;静态测试&gt;" in backtest_text
        assert "测试&lt;股票&gt;" in selection_text
        for content in (backtest_text, selection_text):
            lowered = content.lower()
            assert "<script" not in lowered
            assert "@keyframes" not in lowered and "animation:" not in lowered
            assert "http://" not in lowered and "https://" not in lowered
        print("export_html.py 自测通过：静态 HTML、SVG、转义与离线资源检查通过")
        print(DISCLAIMER)
