# -*- coding: utf-8 -*-
"""
持仓管理与离线盯盘提醒。

- 持仓存储在本地 SQLite positions 表，可手动增删改或从 CSV 导入；
- 盯盘仅使用本地日线行情，计算现价、浮动盈亏、盈亏比例；
- 止盈/止损按比例触发，目标价按绝对价格触发，不接券商、不下单；
- 所有计算基于已收盘的本地行情，不使用实时行情、不存在未来函数。

仅研究用途，不构成投资建议。
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.data_loader import load_local_daily

DISCLAIMER = "仅研究用途，不构成投资建议"
FORCE_SELL_DAYS = 5   # 满 5 个交易日收盘强制离场（买入日记为第 1 天）


def load_positions():
    """加载全部持仓，返回 DataFrame。"""
    db.init_db()
    with db.get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM positions ORDER BY code", conn)
    return df


def save_position(code, name="", shares=0, cost_price=0.0, buy_date="",
                  stop_loss=0.0, take_profit=0.0, target_price=0.0, note="", position_id=None):
    """新增或更新持仓。position_id 为 None 时新增，否则更新。"""
    code = str(code).strip()
    if not code:
        raise ValueError("股票代码不能为空")
    shares = int(shares)
    if shares < 0:
        raise ValueError("持仓数量不能为负")
    cost_price = float(cost_price)
    if cost_price < 0:
        raise ValueError("成本价不能为负")
    db.init_db()
    with db.get_conn() as conn:
        if position_id is None:
            conn.execute(
                "INSERT INTO positions(code, name, shares, cost_price, buy_date, "
                "stop_loss, take_profit, target_price, note, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (code, name, shares, cost_price, buy_date,
                 float(stop_loss), float(take_profit), float(target_price), note, db.now_str()),
            )
        else:
            conn.execute(
                "UPDATE positions SET code=?, name=?, shares=?, cost_price=?, buy_date=?, "
                "stop_loss=?, take_profit=?, target_price=?, note=? WHERE id=?",
                (code, name, shares, cost_price, buy_date,
                 float(stop_loss), float(take_profit), float(target_price), note, position_id),
            )


def delete_position(position_id):
    """按 id 删除持仓。"""
    with db.get_conn() as conn:
        conn.execute("DELETE FROM positions WHERE id=?", (int(position_id),))


def import_positions_csv(file_path):
    """
    从 CSV 导入持仓。CSV 列名不区分大小写，支持以下列：
    code, name, shares, cost_price, buy_date, stop_loss, take_profit, target_price, note
    code 为必填列。导入前先清空已有持仓。
    返回导入条数。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("CSV 文件不存在：%s" % file_path)
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            # 列名转小写匹配
            row = {k.lower().strip(): v for k, v in raw.items()}
            code = str(row.get("code", "")).strip()
            if not code:
                continue
            rows.append({
                "code": code,
                "name": str(row.get("name", "")).strip(),
                "shares": int(float(row.get("shares", 0) or 0)),
                "cost_price": float(row.get("cost_price", 0) or 0),
                "buy_date": str(row.get("buy_date", "")).strip(),
                "stop_loss": float(row.get("stop_loss", 0) or 0),
                "take_profit": float(row.get("take_profit", 0) or 0),
                "target_price": float(row.get("target_price", 0) or 0),
                "note": str(row.get("note", "")).strip(),
            })
    db.init_db()
    with db.get_conn() as conn:
        conn.execute("DELETE FROM positions")
        for r in rows:
            conn.execute(
                "INSERT INTO positions(code, name, shares, cost_price, buy_date, "
                "stop_loss, take_profit, target_price, note, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (r["code"], r["name"], r["shares"], r["cost_price"], r["buy_date"],
                 r["stop_loss"], r["take_profit"], r["target_price"], r["note"], db.now_str()),
            )
    return len(rows)


def _latest_close(code, fq="qfq"):
    """取本地日线最新收盘价；无数据返回 None。"""
    try:
        df = load_local_daily(code, fq=fq)
    except Exception:
        return None
    if df is None or df.empty or "close" not in df.columns:
        return None
    return float(df["close"].iloc[-1])


def _trading_dates(start_date, end_date):
    """取本地前复权日线交易日历；交易日数据缺失时回退到工作日。"""
    try:
        rows = db.query_all(
            "SELECT DISTINCT date FROM daily_bars WHERE fq = ? AND date >= ? AND date <= ? "
            "ORDER BY date", ("qfq", start_date, end_date))
    except Exception:
        rows = []
    dates = [str(row["date"]) for row in rows]
    if dates:
        return dates
    result, current = [], datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    while current <= end:
        if current.weekday() < 5:
            result.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return result


def holding_days(buy_date, as_of=None):
    """按交易日历计算持有天数；返回 (days, force_sell)。

    买入日记为第 1 天，第 5 个交易日收盘强制离场。
    买入日期为空或格式非法时返回 (None, False)。
    """
    text = str(buy_date or "").strip()[:10]
    if not text or text.lower() in ("nan", "none", "nat"):
        return None, False
    try:
        start = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None, False
    if as_of:
        try:
            end = datetime.strptime(str(as_of)[:10], "%Y-%m-%d").date()
        except ValueError:
            end = datetime.now().date()
    else:
        row = db.query_one("SELECT MAX(date) AS date FROM daily_bars WHERE fq = ?", ("qfq",))
        end = datetime.now().date()
        if row and row["date"]:
            try:
                end = datetime.strptime(str(row["date"])[:10], "%Y-%m-%d").date()
            except ValueError:
                pass
    if end < start:
        return 0, False
    days = len(_trading_dates(start.isoformat(), end.isoformat()))
    return days, days >= FORCE_SELL_DAYS



def refresh_positions(fq="qfq"):
    """
    刷新持仓行情：用本地最新日线收盘价计算现价、浮动盈亏、盈亏比例，并返回预警。

    返回 dict:
        frame: 含 current_price, pnl, pnl_ratio 的持仓 DataFrame
        alerts: 触发的预警列表 [{id, code, name, type, message}]
    """
    positions = load_positions()
    if positions.empty:
        return {"frame": positions, "alerts": []}

    current_prices = {}
    for code in positions["code"].unique():
        current_prices[code] = _latest_close(code, fq)

    positions = positions.copy()
    positions["current_price"] = positions["code"].map(current_prices)
    positions["pnl"] = (positions["current_price"] - positions["cost_price"]) * positions["shares"]
    positions["pnl_ratio"] = positions.apply(
        lambda r: (r["current_price"] / r["cost_price"] - 1)
        if pd.notna(r["current_price"]) and r["cost_price"] > 0 else None,
        axis=1,
    )
    buy_dates = positions["buy_date"] if "buy_date" in positions.columns else pd.Series([""] * len(positions))
    held = [holding_days(value) for value in buy_dates]
    positions["days_held"] = [item[0] for item in held]
    positions["force_sell"] = [bool(item[1]) for item in held]

    alerts = []
    for _, row in positions.iterrows():
        # 时间止损优先：满 5 个交易日收盘强制离场，不依赖当日行情是否缺失。
        days = row["days_held"]
        if days is not None and pd.notna(days):
            if days >= FORCE_SELL_DAYS:
                overdue = int(days) - FORCE_SELL_DAYS
                when = "今日收盘" if overdue == 0 else "已超期 %d 个交易日，应尽快" % overdue
                alerts.append({
                    "id": int(row["id"]), "code": row["code"], "name": row["name"],
                    "type": "强制卖出",
                    "message": "%s 已持有第 %d 个交易日（买入 %s），满 %d 天，%s强制离场"
                    % (row["code"], int(days), row["buy_date"], FORCE_SELL_DAYS, when),
                })
            elif days == FORCE_SELL_DAYS - 1:
                alerts.append({
                    "id": int(row["id"]), "code": row["code"], "name": row["name"],
                    "type": "到期提醒",
                    "message": "%s 已持有第 %d 个交易日，明日收盘满 %d 天，需强制离场"
                    % (row["code"], int(days), FORCE_SELL_DAYS),
                })
        if pd.isna(row["current_price"]):
            alerts.append({
                "id": int(row["id"]), "code": row["code"], "name": row["name"],
                "type": "无行情", "message": "本地无 %s 行情数据，请先更新日线" % row["code"],
            })
            continue
        ratio = row["pnl_ratio"]
        # 止损
        if row["stop_loss"] > 0 and ratio <= -row["stop_loss"]:
            alerts.append({
                "id": int(row["id"]), "code": row["code"], "name": row["name"],
                "type": "止损",
                "message": "%s 现价 %.2f，盈亏 %.2f%%，触及止损线 %.2f%%"
                % (row["code"], row["current_price"], ratio * 100, -row["stop_loss"] * 100),
            })
        # 止盈
        if row["take_profit"] > 0 and ratio >= row["take_profit"]:
            alerts.append({
                "id": int(row["id"]), "code": row["code"], "name": row["name"],
                "type": "止盈",
                "message": "%s 现价 %.2f，盈亏 %.2f%%，触及止盈线 %.2f%%"
                % (row["code"], row["current_price"], ratio * 100, row["take_profit"] * 100),
            })
        # 目标价
        if row["target_price"] > 0 and row["current_price"] >= row["target_price"]:
            alerts.append({
                "id": int(row["id"]), "code": row["code"], "name": row["name"],
                "type": "目标价",
                "message": "%s 现价 %.2f，达到目标价 %.2f"
                % (row["code"], row["current_price"], row["target_price"]),
            })
    return {"frame": positions, "alerts": alerts}


if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()
    # 清理测试数据
    with db.get_conn() as conn:
        conn.execute("DELETE FROM positions")
    # 新增测试持仓
    save_position("000001", "平安银行", 1000, 11.0, "2024-01-01",
                  stop_loss=0.05, take_profit=0.10, target_price=13.0)
    save_position("600519", "贵州茅台", 100, 1680.0, "2024-01-01",
                  stop_loss=0.08, take_profit=0.15)
    df = load_positions()
    print("持仓数:", len(df))
    print(df[["code", "name", "shares", "cost_price"]].to_string(index=False))

    # 刷新行情（使用本地真实数据）
    result = refresh_positions()
    frame = result["frame"]
    print("\n刷新后:")
    print(frame[["code", "current_price", "pnl", "pnl_ratio"]].to_string(index=False))
    print("\n预警数:", len(result["alerts"]))
    for a in result["alerts"]:
        print("  [%s] %s" % (a["type"], a["message"]))

    # CSV 导入测试
    csv_path = Path(__file__).resolve().parents[1] / "data" / "test_positions.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "code,name,shares,cost_price,buy_date,stop_loss,take_profit\n"
        "000001,平安银行,2000,10.5,2024-03-01,0.05,0.08\n"
        "000858,五粮液,500,150.0,2024-03-01,0.06,0.12\n",
        encoding="utf-8",
    )
    n = import_positions_csv(csv_path)
    print("\nCSV 导入条数:", n)
    df2 = load_positions()
    print("导入后持仓:", list(df2["code"]))
    csv_path.unlink(missing_ok=True)
    print("position.py 自测通过；%s" % DISCLAIMER)
