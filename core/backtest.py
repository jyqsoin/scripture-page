# -*- coding: utf-8 -*-
"""
A 股日线回测执行、绩效报表与导出。

- 数据完全来自本地 parquet/SQLite，不联网；
- 默认手续费万三、滑点 0.1%，启用 cheat-on-open 配合策略的收盘确认模式；
- 输出总收益、年化、夏普、最大回撤、胜率、平均盈亏比、盈利因子；
- 导出净值/回撤 CSV、交易明细 CSV、事件 CSV、汇总 JSON，可选 PNG 曲线；
- 期末不强制平仓，未平仓交易在明细中标记 open；
- 提供串行参数遍历接口，V1.0 不做寻优 UI。

仅研究用途，不构成投资建议，不承诺任何收益。
"""

import itertools
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.data_loader import load_local_daily, load_local_minute
from core.strategy import MACrossStrategy

DISCLAIMER = "仅研究用途，不构成投资建议"


@dataclass
class BacktestConfig:
    """回测参数；日期包含首尾交易日。"""
    start_date: str = "2020-01-01"
    end_date: str = ""
    initial_cash: float = 1000000.0
    fq: str = "qfq"
    # ---- V1.2 Step G3：分钟回测（默认 daily，保持 V1.0/V1.1 行为） ----
    period: str = "daily"  # "daily" 或 "1min"/"5min"/"15min"/"30min"/"60min"
    # ------------------------------------------------------------------
    fast_period: int = 5
    slow_period: int = 20
    stop_loss: float = 0.05
    # ---- V1.1 Step F2：ATR 追踪止盈止损（默认禁用，保持 V1.0 行为） ----
    enable_atr: bool = False
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    atr_trailing_multiplier: float = 3.0
    # ------------------------------------------------------------------
    commission: float = 0.0003
    slippage: float = 0.001
    max_position_per_stock: float = 0.10
    max_total_position: float = 0.80
    fixed_amount: float = None       # None 表示按组合百分比等权买入
    name: str = "MA金叉止损回测"
    save_plot: bool = True


def _finite(value):
    """把 numpy/NaN/inf 统一转成可写入 JSON 的有限 float 或 None。"""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _prepare_feed(df, start_date, end_date, period="daily"):
    """
    清洗行情并转为 backtrader PandasData。

    - daily：用 date 列，零成交量行视为不可交易日并移除；
    - 分钟：用 datetime 列，不过滤零成交量（集合竞价/午休可能为 0），
      避免丢失有效分钟 K 线；时间索引兼容 backtrader。
    """
    if period == "daily":
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError("回测行情缺少列：%s" % sorted(missing))
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["date", "open", "high", "low", "close"])
        if start_date:
            data = data[data["date"] >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data["date"] <= pd.Timestamp(end_date)]
        data = data[data["volume"].fillna(0) > 0]
        data = data.drop_duplicates("date", keep="last").sort_values("date").set_index("date")
    else:
        required = {"datetime", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError("分钟回测行情缺少列：%s" % sorted(missing))
        data = df.copy()
        data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["datetime", "open", "high", "low", "close"])
        if start_date:
            data = data[data["datetime"] >= pd.Timestamp(start_date)]
        if end_date:
            # 结束日包含全天分钟数据
            end_ts = pd.Timestamp(end_date)
            if end_ts.hour == 0 and end_ts.minute == 0:
                end_ts = end_ts.replace(hour=23, minute=59, second=59)
            data = data[data["datetime"] <= end_ts]
        # 分钟数据不过滤零成交量，只按时间去重排序
        data = data.drop_duplicates("datetime", keep="last").sort_values("datetime").set_index("datetime")
    data["openinterest"] = 0.0
    return data[["open", "high", "low", "close", "volume", "openinterest"]]


def _build_trade_records(events, commission):
    """
    把策略实际成交事件配对为开平仓明细。

    当前策略不加仓、卖出时全平，因此每只股票同时只保留一笔开仓。
    成交价已包含滑点，这里再扣除买卖双边百分比佣金。
    """
    records = []
    open_positions = {}
    for event in events:
        action = event.get("action")
        code = event.get("code")
        if action == "BUY":
            open_positions[code] = {
                "code": code,
                "buy_date": event["date"],
                "buy_price": float(event["price"]),
                "size": abs(int(event["size"])),
            }
        elif action == "SELL" and code in open_positions:
            buy = open_positions.pop(code)
            sell_size = abs(int(event["size"]))
            buy_value = buy["buy_price"] * buy["size"]
            sell_value = float(event["price"]) * sell_size
            fees = (buy_value + sell_value) * commission
            pnl = sell_value - buy_value - fees
            records.append({
                **buy,
                "sell_date": event["date"],
                "sell_price": float(event["price"]),
                "buy_value": buy_value,
                "sell_value": sell_value,
                "commission": fees,
                "pnl": pnl,
                "return_pct": pnl / (buy_value + buy_value * commission) if buy_value else None,
                "status": "closed",
            })
    for buy in open_positions.values():
        records.append({
            **buy, "sell_date": "", "sell_price": None,
            "buy_value": buy["buy_price"] * buy["size"], "sell_value": None,
            "commission": None, "pnl": None, "return_pct": None, "status": "open",
        })
    return records


def _trade_metrics(records):
    """胜率、平均盈亏比和盈利因子均只统计已平仓交易。"""
    closed = [item for item in records if item["status"] == "closed"]
    wins = [item for item in closed if item["pnl"] > 0]
    losses = [item for item in closed if item["pnl"] < 0]
    gross_profit = sum(item["pnl"] for item in wins)
    gross_loss = abs(sum(item["pnl"] for item in losses))
    average_win = gross_profit / len(wins) if wins else None
    average_loss = gross_loss / len(losses) if losses else None
    return {
        "trades_closed": len(closed),
        "trades_open": len(records) - len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(closed) if closed else None,
        "pl_ratio": average_win / average_loss if average_win and average_loss else None,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "net_pnl": sum(item["pnl"] for item in closed),
    }


def _curve_metrics(time_returns, initial_cash, final_value):
    """由每日权益收益率生成净值和回撤曲线，并计算年化、夏普、最大回撤。"""
    columns = ["daily_return", "equity", "net_value", "drawdown"]
    if not time_returns:
        curves = pd.DataFrame(columns=columns)
        curves.index = pd.DatetimeIndex([], name="date")
    else:
        curves = pd.DataFrame.from_dict(time_returns, orient="index", columns=["daily_return"])
        curves.index = pd.to_datetime(curves.index)
        curves = curves.sort_index()
        curves["equity"] = initial_cash * (1 + curves["daily_return"].fillna(0)).cumprod()
        curves["net_value"] = curves["equity"] / initial_cash
        running_max = curves["equity"].cummax().clip(lower=initial_cash)
        curves["drawdown"] = curves["equity"] / running_max - 1

    total_return = final_value / initial_cash - 1
    periods = len(curves)
    annual_return = None
    if periods > 0 and final_value > 0:
        annual_return = (final_value / initial_cash) ** (252.0 / periods) - 1
    sharpe = None
    if periods > 1:
        std = curves["daily_return"].std(ddof=1)
        if std and np.isfinite(std):
            sharpe = curves["daily_return"].mean() / std * math.sqrt(252)
    max_drawdown = abs(curves["drawdown"].min()) if periods else 0.0
    return curves, {
        "total_return": total_return,
        "annual_return": _finite(annual_return),
        "sharpe": _finite(sharpe),
        "max_drawdown": _finite(max_drawdown),
        "periods": periods,
    }


def save_curve_plot(curves, path):
    """使用 Agg 画布保存净值/回撤 PNG，不启动窗口或本地服务。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 6), dpi=120)
    FigureCanvasAgg(figure)
    ax1 = figure.add_subplot(211)
    ax2 = figure.add_subplot(212)
    if not curves.empty:
        ax1.plot(curves.index, curves["net_value"], color="black", linewidth=1)
        ax2.fill_between(curves.index, curves["drawdown"] * 100, 0, color="gray", alpha=0.6)
    ax1.set_title("Net Value - Research Only")
    ax2.set_title("Drawdown (%)")
    ax1.grid(alpha=0.2)
    ax2.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(str(path))


def _save_record(config, summary, summary_path, trades_path):
    """回测汇总写入 SQLite；大文件只保存路径。"""
    db.init_db()
    with db.get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO backtest_record(
                name, strategy, params, start_date, end_date, initial_cash, final_value,
                annual_return, sharpe, max_drawdown, win_rate, pl_ratio,
                trades_count, report_path, trades_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config.name, "MACrossStrategy", json.dumps(asdict(config), ensure_ascii=False),
                config.start_date, config.end_date, config.initial_cash, summary["final_value"],
                summary["annual_return"], summary["sharpe"], summary["max_drawdown"],
                summary["win_rate"], summary["pl_ratio"], summary["trades_closed"],
                str(summary_path or ""), str(trades_path or ""), db.now_str(),
            ),
        )
        return cursor.lastrowid


def run_backtest(codes=None, config=None, data_map=None, export=True, save_record=True):
    """
    运行回测。

    codes：股票代码列表；data_map 可传入 {code: DataFrame} 用于合成数据测试，
           未提供的代码仍从本地 load_local_daily 读取。
    export=False 时跳过文件导出；save_record=False 时跳过 SQLite 记录，供参数遍历使用。
    """
    config = config or BacktestConfig()
    if config.initial_cash <= 0:
        raise ValueError("初始资金必须大于 0")
    if config.fast_period >= config.slow_period:
        raise ValueError("fast_period 必须小于 slow_period")
    codes = list(codes) if codes is not None else list((data_map or {}).keys())
    codes = [str(code).strip().zfill(6) for code in codes]
    if not codes:
        raise ValueError("回测股票列表为空")

    cerebro = bt.Cerebro(cheat_on_open=True)  # 启用 next_open，配合收盘信号次日开盘成交
    cerebro.broker.set_coo(True)  # 与 next_open 配合：收盘信号次日开盘成交
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(
        commission=config.commission,
        commtype=bt.CommInfoBase.COMM_PERC,
        stocklike=True,
    )
    cerebro.broker.set_slippage_perc(
        perc=config.slippage, slip_open=True, slip_match=True
    )

    loaded = []
    skipped = {}
    for code in codes:
        try:
            if data_map and code in data_map:
                raw = data_map[code]
            elif config.period == "daily":
                raw = load_local_daily(code, config.fq)
            else:
                raw = load_local_minute(code, config.period, config.fq)
            frame = _prepare_feed(raw, config.start_date, config.end_date, config.period)
            if len(frame) < config.slow_period + 2:
                skipped[code] = "有效行情不足 %d 根" % (config.slow_period + 2)
                continue
            cerebro.adddata(bt.feeds.PandasData(dataname=frame), name=code)
            loaded.append(code)
        except Exception as exc:
            skipped[code] = str(exc)
    if not loaded:
        raise ValueError("没有可回测行情：%s" % skipped)

    cerebro.addstrategy(
        MACrossStrategy,
        fast_period=config.fast_period,
        slow_period=config.slow_period,
        stop_loss=config.stop_loss,
        enable_atr=config.enable_atr,
        atr_period=config.atr_period,
        atr_stop_multiplier=config.atr_stop_multiplier,
        atr_trailing_multiplier=config.atr_trailing_multiplier,
        max_position_per_stock=config.max_position_per_stock,
        max_total_position=config.max_total_position,
        commission=config.commission,
        slippage=config.slippage,
        fixed_amount=config.fixed_amount,
    )
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
    strategy = cerebro.run()[0]
    final_value = float(cerebro.broker.getvalue())

    time_returns = strategy.analyzers.time_return.get_analysis()
    curves, curve_summary = _curve_metrics(time_returns, config.initial_cash, final_value)
    trade_records = _build_trade_records(strategy.events, config.commission)
    trades = pd.DataFrame(trade_records)
    events = pd.DataFrame(strategy.events)
    warnings = list(strategy.warnings)

    summary = {
        "name": config.name,
        "codes_loaded": loaded,
        "codes_skipped": skipped,
        "initial_cash": config.initial_cash,
        "final_value": final_value,
        **curve_summary,
        **_trade_metrics(trade_records),
        "disclaimer": DISCLAIMER,
    }

    output_dir = ""
    summary_path = trades_path = ""
    if export:
        output = db.EXPORT_DIR / ("backtest_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        output.mkdir(parents=True, exist_ok=True)
        output_dir = str(output)
        summary_path = output / "summary.json"
        trades_path = output / "trades.csv"
        curves.to_csv(output / "equity_curve.csv", encoding="utf-8-sig", index_label="date")
        trades.to_csv(trades_path, encoding="utf-8-sig", index=False)
        events.to_csv(output / "events.csv", encoding="utf-8-sig", index=False)
        if warnings:
            pd.DataFrame(warnings).to_csv(output / "warnings.csv", encoding="utf-8-sig", index=False)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        if config.save_plot:
            try:
                save_curve_plot(curves, output / "curves.png")
            except ImportError as exc:
                warnings.append({"action": "PLOT", "reason": "matplotlib 不可用：%s" % exc})

    record_id = _save_record(config, summary, summary_path, trades_path) if save_record else None
    return {
        "summary": summary,
        "curves": curves,
        "trades": trades,
        "events": events,
        "warnings": warnings,
        "output_dir": output_dir,
        "record_id": record_id,
    }


def iter_param_grid(param_grid):
    """参数遍历生成器：{"fast_period":[3,5],"slow_period":[10,20]}。"""
    keys = list(param_grid)
    values = [value if isinstance(value, (list, tuple)) else [value] for value in param_grid.values()]
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def run_parameter_grid(codes, base_config, param_grid, data_map=None, max_runs=None, progress_cb=None):
    """
    V1.0 串行参数遍历接口；每次不导出文件、不写回测记录，返回汇总 DataFrame。

    progress_cb(current, total, overrides) 可选；用于 UI 进度展示。
    寻优结果只表示历史样本表现，不代表未来收益。
    """
    # 先计算总组合数，便于进度展示。
    total = 1
    for value in param_grid.values():
        total *= len(value) if isinstance(value, (list, tuple)) else 1
    if max_runs is not None:
        total = min(total, max_runs)

    rows = []
    for index, overrides in enumerate(iter_param_grid(param_grid)):
        if max_runs is not None and index >= max_runs:
            break
        config = replace(base_config, **overrides)
        result = run_backtest(
            codes, config, data_map=data_map, export=False, save_record=False
        )
        rows.append({**overrides, **result["summary"]})
        if progress_cb:
            progress_cb(index + 1, total, overrides)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 两只合成股票自测：不联网、不写正式数据库、不导出文件。
    print(DISCLAIMER)
    size = 120
    index = pd.bdate_range("2024-01-01", periods=size)
    data_map = {}
    for offset, code in enumerate(["600000", "000001"]):
        close = pd.Series(
            10 + offset + np.sin(np.arange(size) / 4) * 2 + np.arange(size) * 0.02
        )
        data_map[code] = pd.DataFrame({
            "date": index.strftime("%Y-%m-%d"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(size, 1000000.0),
        })

    config = BacktestConfig(
        start_date="2024-01-01", fast_period=3, slow_period=8, save_plot=False
    )
    result = run_backtest(
        config=config, data_map=data_map, export=False, save_record=False
    )
    print("汇总：", result["summary"])
    assert not result["curves"].empty
    assert result["summary"]["final_value"] > 0
    assert "max_drawdown" in result["summary"]
    print("backtest.py 自测通过；结果仅研究用途")
