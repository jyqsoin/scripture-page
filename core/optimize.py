# -*- coding: utf-8 -*-
"""
参数寻优结果深化 + 多策略对比。

- 样本外验证：按日期把回测区间切成样本内（寻优）和样本外（验证）两段，
  样本外数据在寻优时完全不可见，用于检验过拟合；
- 热力图：把两维参数组合的某绩效指标渲染为静态 SVG 灰阶热力图，适配墨水屏；
- 多策略对比：对多个策略配置分别回测，生成绩效汇总表与净值曲线叠加图；
- 仅使用本地行情，不联网、不下单、不接券商。

仅研究用途，不构成投资建议。历史回测结果不代表未来收益。
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest import run_backtest, run_parameter_grid

DISCLAIMER = "仅研究用途，不构成投资建议"

# 越小越优的指标（回撤类），用于最优参数选取与热力图灰度反转。
LOWER_IS_BETTER = {"max_drawdown"}


def _resolve_end_date(base_config, data_map):
    """确定回测结束日：优先用 config.end_date，否则取 data_map 中最晚日期。"""
    if base_config.end_date:
        return pd.Timestamp(base_config.end_date)
    if data_map:
        latest = None
        for frame in data_map.values():
            if frame is None or frame.empty or "date" not in frame.columns:
                continue
            current = pd.to_datetime(frame["date"], errors="coerce").max()
            if pd.notna(current) and (latest is None or current > latest):
                latest = current
        if latest is not None:
            return latest
    return pd.Timestamp.today().normalize()


def _split_date(start, end, in_sample_ratio):
    """按自然日比例计算切分日；不保证是交易日，由回测引擎自行过滤。"""
    if not 0 < in_sample_ratio < 1:
        raise ValueError("in_sample_ratio 必须在 0 和 1 之间")
    total_days = (end - start).days
    if total_days <= 0:
        raise ValueError("回测起止日期无效，无法切分样本内/外")
    split = start + pd.Timedelta(days=int(total_days * in_sample_ratio))
    return split


def run_out_of_sample_validation(
    codes, base_config, param_grid, data_map=None,
    in_sample_ratio=0.7, metric="sharpe", max_runs=None, progress_cb=None,
):
    """
    样本外验证：在样本内区间寻优，用最优参数在样本外区间回测。

    参数
    ----
    codes : list[str]
        股票代码列表。
    base_config : BacktestConfig
        基础回测配置；start_date/end_date 用于确定整体区间。
    param_grid : dict
        参数网格，同 run_parameter_grid。
    data_map : dict[str, DataFrame] | None
        本地行情映射；None 时由回测引擎自行加载。
    in_sample_ratio : float
        样本内区间占比，默认 0.7。
    metric : str
        选优指标列名，如 sharpe / annual_return / max_drawdown / total_return。
    max_runs : int | None
        样本内寻优的最大组合数，同 run_parameter_grid。
    progress_cb : callable | None
        进度回调 current/total/overrides，仅样本内寻优阶段触发。

    返回
    ----
    dict
        in_sample_frame: 样本内所有组合的绩效 DataFrame
        best_params: 最优参数字典
        in_sample_best: 最优组合在样本内的绩效
        out_of_sample: 最优参数在样本外的绩效汇总
        split_date: 切分日期字符串
        in_sample_period / out_of_sample_period: 区间起止
        metric: 选优指标
    """
    start = pd.Timestamp(base_config.start_date)
    end = _resolve_end_date(base_config, data_map)
    split = _split_date(start, end, in_sample_ratio)
    split_str = split.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    # 样本内：寻优区间 [start, split]
    in_config = replace(base_config, start_date=base_config.start_date, end_date=split_str)
    in_sample_frame = run_parameter_grid(
        codes, in_config, param_grid, data_map=data_map,
        max_runs=max_runs, progress_cb=progress_cb,
    )
    if in_sample_frame.empty:
        raise ValueError("样本内寻优无有效结果，请扩大参数范围或检查行情")
    if metric not in in_sample_frame.columns:
        raise ValueError("选优指标 %s 不在寻优结果列中：%s" % (metric, list(in_sample_frame.columns)))

    # 选取最优参数：越小越优的指标升序，其余降序。
    ascending = metric in LOWER_IS_BETTER
    sorted_frame = in_sample_frame.sort_values(metric, ascending=ascending, na_position="last")
    best_row = sorted_frame.iloc[0]
    param_keys = list(param_grid.keys())
    best_params = {key: best_row[key] for key in param_keys}
    # numpy 标量转原生类型，便于序列化。
    best_params = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in best_params.items()
    }

    # 样本外：用最优参数在 [split, end] 回测，寻优阶段完全未见过该区间。
    out_config = replace(base_config, start_date=split_str, end_date=end_str, **best_params)
    out_result = run_backtest(
        codes, out_config, data_map=data_map, export=False, save_record=False,
    )

    return {
        "in_sample_frame": in_sample_frame,
        "best_params": best_params,
        "in_sample_best": best_row.to_dict(),
        "out_of_sample": out_result["summary"],
        "split_date": split_str,
        "in_sample_period": (base_config.start_date, split_str),
        "out_of_sample_period": (split_str, end_str),
        "metric": metric,
        "disclaimer": DISCLAIMER,
    }


def heatmap_svg(frame, x_param, y_param, value_metric, title="参数热力图"):
    """
    把参数寻优结果渲染为静态 SVG 灰阶热力图（适配墨水屏，无外部资源）。

    颜色语义：颜色越深表示该指标越优。对于 max_drawdown 等越小越优的指标自动反转。
    缺失组合显示为空白格。
    """
    if frame is None or frame.empty:
        return "<p>暂无热力图数据。</p>"
    for col in (x_param, y_param, value_metric):
        if col not in frame.columns:
            raise ValueError("列 %s 不在寻优结果中：%s" % (col, list(frame.columns)))

    x_values = sorted(frame[x_param].dropna().unique())
    y_values = sorted(frame[y_param].dropna().unique(), reverse=True)  # y 轴大值在上
    if not x_values or not y_values:
        return "<p>参数 %s/%s 取值为空，无法绘制热力图。</p>" % (x_param, y_param)

    value_series = pd.to_numeric(frame[value_metric], errors="coerce")
    valid = value_series.dropna()
    if valid.empty:
        return "<p>指标 %s 无有效值，无法绘制热力图。</p>" % value_metric
    vmin, vmax = float(valid.min()), float(valid.max())
    value_range = vmax - vmin or 1.0
    lower_is_better = value_metric in LOWER_IS_BETTER

    cell_w, cell_h = 64, 44
    margin_left, margin_right = 72, 18
    margin_top, margin_bottom = 36, 56
    width = margin_left + cell_w * len(x_values) + margin_right
    height = margin_top + cell_h * len(y_values) + margin_bottom

    def x_at(col):
        return margin_left + col * cell_w

    def y_at(row):
        return margin_top + row * cell_h

    def gray_level(value):
        """返回 0-1 的灰度强度，越大越深（越优）。"""
        normalized = (float(value) - vmin) / value_range
        return 1.0 - normalized if lower_is_better else normalized

    parts = [
        '<svg class="chart" viewBox="0 0 %d %d" role="img" aria-label="%s">' % (width, height, _esc(title)),
        "<title>%s</title>" % _esc(title),
        '<text x="%d" y="22" font-size="13" font-weight="bold">%s</text>' % (margin_left, _esc(title)),
    ]

    # 绘制单元格
    for row, y_val in enumerate(y_values):
        for col, x_val in enumerate(x_values):
            match = frame[(frame[x_param] == x_val) & (frame[y_param] == y_val)]
            if match.empty or pd.isna(match.iloc[0][value_metric]):
                # 缺失组合：浅灰底
                fill = "#f0f0f0"
                text = ""
            else:
                value = float(match.iloc[0][value_metric])
                level = gray_level(value)
                # 灰阶：0=白，1=黑
                gray = int(round((1.0 - level) * 255))
                fill = "#%02x%02x%02x" % (gray, gray, gray)
                # 数值文字：深色格用白字，浅色格用黑字
                text_color = "#fff" if level > 0.5 else "#000"
                if value_metric in ("sharpe",):
                    text = "%.2f" % value
                elif value_metric in ("annual_return", "total_return", "max_drawdown"):
                    text = "%.1f%%" % (value * 100)
                else:
                    text = "%.3f" % value
                parts.append(
                    '<rect x="%d" y="%d" width="%d" height="%d" fill="%s" stroke="#000" stroke-width="1" />'
                    % (x_at(col), y_at(row), cell_w, cell_h, fill)
                )
                parts.append(
                    '<text x="%d" y="%d" text-anchor="middle" font-size="11" fill="%s">%s</text>'
                    % (x_at(col) + cell_w // 2, y_at(row) + cell_h // 2 + 4, text_color, _esc(text))
                )
                continue
            parts.append(
                '<rect x="%d" y="%d" width="%d" height="%d" fill="%s" stroke="#000" stroke-width="1" />'
                % (x_at(col), y_at(row), cell_w, cell_h, fill)
            )

    # x 轴标签
    for col, x_val in enumerate(x_values):
        parts.append(
            '<text x="%d" y="%d" text-anchor="middle" font-size="11">%s</text>'
            % (x_at(col) + cell_w // 2, height - margin_bottom + 18, _esc(str(x_val)))
        )
    parts.append(
        '<text x="%d" y="%d" text-anchor="middle" font-size="12" font-weight="bold">%s</text>'
        % (margin_left + cell_w * len(x_values) // 2, height - 10, _esc(x_param))
    )

    # y 轴标签
    for row, y_val in enumerate(y_values):
        parts.append(
            '<text x="%d" y="%d" text-anchor="end" font-size="11">%s</text>'
            % (margin_left - 6, y_at(row) + cell_h // 2 + 4, _esc(str(y_val)))
        )
    parts.append(
        '<text x="%d" y="%d" text-anchor="middle" font-size="12" font-weight="bold" transform="rotate(-90 %d %d)">%s</text>'
        % (16, margin_top + cell_h * len(y_values) // 2, 16, margin_top + cell_h * len(y_values) // 2, _esc(y_param))
    )

    parts.append("</svg>")
    return "".join(parts)


# 墨水屏灰阶色板：从黑到浅灰，用于区分多条曲线
_GRAY_PALETTE = ["#000000", "#333333", "#666666", "#999999", "#555555", "#777777", "#aaaaaa", "#444444"]

# 对比汇总表展示的绩效指标列
COMPARISON_METRICS = [
    ("总收益率", "total_return", "percent"),
    ("年化收益率", "annual_return", "percent"),
    ("夏普比率", "sharpe", "number"),
    ("最大回撤", "max_drawdown", "percent"),
    ("已平仓交易", "trades_closed", "integer"),
    ("胜率", "win_rate", "percent"),
    ("盈利因子", "profit_factor", "number"),
]


def run_strategy_comparison(codes, named_configs, data_map=None, progress_cb=None):
    """
    多策略对比：对多个命名策略配置分别回测，返回绩效汇总与净值曲线。

    参数
    ----
    codes : list[str]
        股票代码列表。
    named_configs : list[tuple[str, BacktestConfig]]
        策略名称与配置的列表。
    data_map : dict | None
        本地行情映射。
    progress_cb : callable | None
        进度回调 current/total/name。

    返回
    ----
    dict
        comparisons: [{"name", "config", "summary", "curves"}, ...]
        summary_table: DataFrame（行为策略，列为绩效指标）
        equity_overlay_svg: 多策略净值叠加 SVG
    """
    if not named_configs:
        raise ValueError("策略配置列表为空")
    total = len(named_configs)
    comparisons = []
    for index, (name, config) in enumerate(named_configs):
        if progress_cb:
            progress_cb(index + 1, total, name)
        result = run_backtest(
            codes, config, data_map=data_map, export=False, save_record=False,
        )
        comparisons.append({
            "name": name,
            "config": config,
            "summary": result["summary"],
            "curves": result["curves"],
        })

    summary_table = build_comparison_table(comparisons)
    overlay_svg = equity_overlay_svg(comparisons)
    return {
        "comparisons": comparisons,
        "summary_table": summary_table,
        "equity_overlay_svg": overlay_svg,
        "disclaimer": DISCLAIMER,
    }


def build_comparison_table(comparisons):
    """把多策略绩效汇总为 DataFrame，行为策略名，列为指标。"""
    rows = []
    for item in comparisons:
        summary = item["summary"]
        row = {"策略": item["name"]}
        for label, key, _fmt in COMPARISON_METRICS:
            row[label] = summary.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def equity_overlay_svg(comparisons, title="策略净值对比"):
    """多策略净值曲线叠加 SVG（墨水屏灰阶）。缺失日期不插值，按各自交易日绘制。"""
    if not comparisons:
        return "<p>暂无策略对比数据。</p>"
    width, height = 960, 300
    left, right, top, bottom = 68, 18, 32, 60
    plot_w, plot_h = width - left - right, height - top - bottom

    # 合并所有策略的净值序列，确定值域
    all_series = []
    for item in comparisons:
        curves = item["curves"]
        if curves is None or curves.empty or "net_value" not in curves.columns:
            continue
        series = curves["net_value"].dropna()
        if not series.empty:
            all_series.append((item["name"], series))
    if not all_series:
        return "<p>暂无净值曲线数据。</p>"

    all_values = pd.concat([s for _, s in all_series])
    vmin, vmax = float(all_values.min()), float(all_values.max())
    padding = (vmax - vmin) * 0.08 or 0.01
    vmin, vmax = vmin - padding, vmax + padding
    vrange = vmax - vmin

    # 统一时间轴：取所有策略日期的并集
    all_dates = pd.DatetimeIndex([])
    for _, series in all_series:
        all_dates = all_dates.union(series.index)
    all_dates = all_dates.sort_values()
    n = len(all_dates)

    def x_at(date_ts):
        pos = all_dates.get_loc(date_ts) if isinstance(all_dates, pd.DatetimeIndex) else list(all_dates).index(date_ts)
        return left + (pos / max(n - 1, 1)) * plot_w

    def y_at(value):
        return top + (vmax - value) / vrange * plot_h

    parts = [
        '<svg class="chart" viewBox="0 0 %d %d" role="img" aria-label="%s">' % (width, height, _esc(title)),
        "<title>%s</title>" % _esc(title),
        '<text x="%d" y="19" font-size="13" font-weight="bold">%s</text>' % (left, _esc(title)),
    ]
    # Y 轴网格与刻度
    for i in range(5):
        val = vmin + vrange * i / 4
        y = y_at(val)
        parts.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#888" stroke-width="1" />' % (left, y, width - right, y))
        parts.append('<text x="%d" y="%.2f" text-anchor="end" font-size="11">%.3f</text>' % (left - 6, y + 4, val))
    # 零线（净值=1）
    if vmin <= 1 <= vmax:
        y0 = y_at(1)
        parts.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#000" stroke-dasharray="5 4" />' % (left, y0, width - right, y0))

    # 绘制每条净值曲线
    for idx, (name, series) in enumerate(all_series):
        color = _GRAY_PALETTE[idx % len(_GRAY_PALETTE)]
        points = " ".join("%.2f,%.2f" % (x_at(ts), y_at(v)) for ts, v in series.items())
        parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.8" />' % (points, color))

    # X 轴日期刻度
    tick_count = min(6, n)
    tick_positions = np.unique(np.linspace(0, n - 1, tick_count, dtype=int))
    for pos in tick_positions:
        label = all_dates[int(pos)].strftime("%Y-%m-%d")
        x = left + (int(pos) / max(n - 1, 1)) * plot_w
        parts.append('<text x="%.2f" y="%d" text-anchor="middle" font-size="11">%s</text>' % (x, height - 40, label))

    # 图例
    legend_y = height - 20
    legend_x = left
    for idx, (name, _) in enumerate(all_series):
        color = _GRAY_PALETTE[idx % len(_GRAY_PALETTE)]
        parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="2" />' % (legend_x, legend_y, legend_x + 20, legend_y, color))
        parts.append('<text x="%d" y="%d" font-size="11">%s</text>' % (legend_x + 24, legend_y + 4, _esc(name)))
        legend_x += 110

    parts.append("</svg>")
    return "".join(parts)


def _esc(value):
    import html
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    # 合成数据自测：不联网、不写正式数据库、不导出文件。
    print(DISCLAIMER)
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=240)
    close = pd.Series(10 + np.sin(np.arange(len(dates)) / 12) * 2 + np.arange(len(dates)) * 0.01)
    data_map = {"000001": pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 0.2, "low": close - 0.2, "close": close,
        "volume": rng.integers(100000, 1000000, len(dates)),
    })}

    from core.backtest import BacktestConfig
    base = BacktestConfig(start_date="2023-01-02", end_date="2023-12-15", save_plot=False)
    grid = {"fast_period": [3, 5], "slow_period": [10, 20], "stop_loss": [0.05]}

    # 样本外验证
    result = run_out_of_sample_validation(
        ["000001"], base, grid, data_map=data_map, in_sample_ratio=0.7, metric="sharpe", max_runs=10,
    )
    print("切分日:", result["split_date"])
    print("样本内区间:", result["in_sample_period"])
    print("样本外区间:", result["out_of_sample_period"])
    print("最优参数:", result["best_params"])
    print("样本内最优 sharpe:", result["in_sample_best"].get("sharpe"))
    print("样本外 sharpe:", result["out_of_sample"].get("sharpe"))
    assert result["in_sample_frame"].shape[0] <= 10
    assert result["best_params"]
    assert "sharpe" in result["out_of_sample"]
    assert result["in_sample_period"][1] == result["out_of_sample_period"][0]

    # 热力图
    svg = heatmap_svg(result["in_sample_frame"], "fast_period", "slow_period", "sharpe")
    assert "<svg" in svg and "stroke=\"#000\"" in svg
    assert "http://" not in svg and "https://" not in svg
    print("热力图 SVG 长度:", len(svg))

    # ---- 多策略对比自测 ----
    cfg1 = replace(base, name="MA5/20", fast_period=5, slow_period=20, stop_loss=0.05)
    cfg2 = replace(base, name="MA3/10", fast_period=3, slow_period=10, stop_loss=0.03)
    cfg3 = replace(base, name="MA8/30+ATR", fast_period=8, slow_period=30, stop_loss=0.05,
                   enable_atr=True, atr_stop_multiplier=2.0, atr_trailing_multiplier=3.0)
    named = [("MA5/20", cfg1), ("MA3/10", cfg2), ("MA8/30+ATR", cfg3)]
    cmp_cb = [0]
    def cmp_progress(cur, total, name):
        cmp_cb[0] += 1
    cmp_result = run_strategy_comparison(["000001"], named, data_map=data_map, progress_cb=cmp_progress)
    assert len(cmp_result["comparisons"]) == 3
    assert cmp_cb[0] == 3
    table = cmp_result["summary_table"]
    print("对比表列:", list(table.columns))
    print(table.to_string(index=False))
    assert list(table["策略"]) == ["MA5/20", "MA3/10", "MA8/30+ATR"]
    overlay = cmp_result["equity_overlay_svg"]
    assert "<svg" in overlay and "polyline" in overlay
    assert "http://" not in overlay and "https://" not in overlay
    print("多策略对比 SVG 长度:", len(overlay))
    print("optimize.py 自测通过（含多策略对比）；结果仅研究用途")
