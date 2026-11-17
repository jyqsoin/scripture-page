# -*- coding: utf-8 -*-
"""
HAMMER_BOLL 买入/卖出公式与卖点核对（纯本地，仅研究用途）。

落点配置：
- 买入公式 HAMMER_BOLL_BUY：写入本地公式库，在“公式选股”页执行并导出候选池 CSV；
- 卖出公式 HAMMER_BOLL_SELL：导出为文本，由用户在通达信 Ctrl+F 新建条件选股公式后粘贴，
  本模块不操作通达信、不联网、不下单；
- 卖点核对：读取通达信 SELL 扫描结果或本模块回测输出的 hammer_trades.csv，
  与持仓表逐条比对，只做预警提示，是否清仓由人工确认。

策略口径（与五因子/回测逻辑无关）：
- 买入：阳线锤子线（下影≥实体2倍、下影≥振幅40%）+ 最低价触及 20 日均线 + MA20 或 MA60 向上；
- 卖出：跌破布林中轨、冲高上轨后回落、跌破下轨、5 日累计跌幅达到 8%；
- 时间止损：持仓表按“买入日期”计第 5 个交易日收盘强制离场（由 core.position 计算）。

仅研究用途，不构成投资建议。
"""

import csv
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db, formula_engine
from core.data_loader import load_local_daily
from core.formula_engine import FormulaEngine

DISCLAIMER = "仅研究用途，不构成投资建议"
BUY_NAME = "HAMMER_BOLL_BUY"
SELL_NAME = "HAMMER_BOLL_SELL"
DEFAULT_N = 20
FORCE_SELL_DAYS = 5

DATA_DIR = db.DATA_DIR
FORMULA_DIR = DATA_DIR / "formula"
HAMMER_TRADES_CSV = DATA_DIR / "hammer_trades.csv"
HAMMER_POOL_CSV = DATA_DIR / "hammer_pool.csv"
BUY_TXT = FORMULA_DIR / (BUY_NAME + ".txt")
SELL_TXT = FORMULA_DIR / (SELL_NAME + ".txt")

TDX_STEPS = (
    "通达信步骤：Ctrl+F → 条件选股公式 → 其它类型 → 新建 → "
    "公式名称填 %s → 粘贴公式文本 → 确定。" % SELL_NAME
)

# 买入公式：锤子线 + 最低价触及 20 日均线 + 均线趋势向上，兼容通达信语法。
BUY_FORMULA = """N1:=5; N2:=20; N3:=60;
MA5:=MA(C,N1);
MA20:=MA(C,N2);
MA60:=MA(C,N3);

MID:=MA(C,N2);

Body:=ABS(C-O);
Range:=H-L;
OpenClose:=IF(O<C,O,C);
LowShadow:=OpenClose-L;

Hammer:=LowShadow>=Body*2 AND C>O AND LowShadow>=Range*0.4;

BreakMID:=L<=MID;

TrendOK:=(MA20>REF(MA20,1)) OR (MA60>REF(MA60,1));
NoDoubleDead:=(MA20>REF(MA20,1) OR MA60>REF(MA60,1));

XG:Hammer AND BreakMID AND TrendOK AND NoDoubleDead;"""

# 卖出公式：内置 N=20，只使用 MA/STD/REF 等通达信原生函数，可直接粘贴；
# 满 5 个交易日的时间止损由本地持仓表按买入日期计算。
SELL_FORMULA = """N:=20;
MID:=MA(C,N);
SD:=STD(C,N);
UPPER:=MID+2*SD;
LOWER:=MID-2*SD;
BREAK_MID:=C<MID AND REF(C,1)>=REF(MID,1);
TOUCH_UPPER:=H>=UPPER;
FALL_BACK:=C<UPPER AND C<O;
BREAK_LOWER:=C<LOWER;
DROP_5D:=C<=REF(C,5)*0.92;
XG:BREAK_MID OR (TOUCH_UPPER AND FALL_BACK) OR BREAK_LOWER OR DROP_5D;"""


def formula_bundle():
    """返回 (BUY 文本, SELL 文本)，供界面展示或导出。"""
    return BUY_FORMULA, SELL_FORMULA


def ensure_buy_formula():
    """把 BUY 公式写入本地公式库（select 类型），重复调用为幂等覆盖。"""
    return formula_engine.save_formula(
        BUY_NAME, "select", BUY_FORMULA, params={},
    )


def save_formula_files():
    """导出 BUY/SELL 文本文件；SELL 文本用于通达信手动新建公式。"""
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    buy_text = (
        "{HAMMER_BOLL_BUY —— 本工具“公式选股”页可直接载入；%s}\n"
        "{参数：N1=5、N2=20、N3=60；不含未来函数}\n%s\n"
        % (DISCLAIMER, BUY_FORMULA)
    )
    sell_text = (
        "{HAMMER_BOLL_SELL —— 通达信条件选股公式，用于持仓股每日扫描}\n"
        "{Ctrl+F → 条件选股公式 → 其它类型 → 新建 → 名称 %s → 粘贴本文件内容 → 确定}\n"
        "{说明：满 %d 个交易日的时间止损由本工具持仓表按买入日期计算，不在本公式内判断}\n"
        "{%s}\n%s\n"
        % (SELL_NAME, FORCE_SELL_DAYS, DISCLAIMER, SELL_FORMULA)
    )
    BUY_TXT.write_text(buy_text, encoding="utf-8")
    SELL_TXT.write_text(sell_text, encoding="utf-8")
    return BUY_TXT, SELL_TXT


def _signals(frame):
    """计算单只股票的 BUY/SELL 布尔序列。"""
    buy = FormulaEngine(frame, {"N": DEFAULT_N}).run(BUY_FORMULA, "select").value
    sell = FormulaEngine(frame, {"N": DEFAULT_N}).run(SELL_FORMULA, "select").value
    return buy.fillna(False).astype(bool), sell.fillna(False).astype(bool)


def candidate_rows(rows, fq="qfq"):
    """把选股结果补充为候选池字段：代码/名称/选股日/收盘价/MA5/布林下轨。"""
    output = []
    for item in rows:
        code = str(item.get("code", "")).zfill(6)
        date = str(item.get("date", ""))[:10]
        close = item.get("close")
        ma5 = lower = None
        try:
            frame = load_local_daily(code, fq)
            frame = frame[frame["date"].astype(str).str[:10] <= date]
            closes = frame["close"].astype(float)
            if len(closes) >= 5:
                ma5 = float(closes.tail(5).mean())
            if len(closes) >= DEFAULT_N:
                tail = closes.tail(DEFAULT_N)
                lower = float(tail.mean()) - 2 * float(tail.std(ddof=0))
                if close is None:
                    close = float(closes.iloc[-1])
        except Exception:
            pass
        output.append({
            "代码": code,
            "名称": item.get("name", ""),
            "选股日": date,
            "收盘价": round(float(close), 3) if close is not None else "",
            "MA5": round(ma5, 3) if ma5 is not None else "",
            "布林下轨": round(lower, 3) if lower is not None else "",
        })
    return output


def _simulate_trades(code, frame, buy, sell):
    """单只股票逐笔模拟：信号次日开盘买入，SELL 或满 5 个交易日收盘卖出。

    买入日记为第 1 天，第 5 个交易日收盘强制离场；不计算费用，仅用于卖点核对。
    """
    rows = []
    dates = frame["date"].astype(str).tolist()
    opens = frame["open"].astype(float).tolist()
    closes = frame["close"].astype(float).tolist()
    entry = None
    for index in range(1, len(frame)):
        if entry is None:
            if bool(buy.iloc[index]) and index + 1 < len(frame):
                entry = {
                    "entry": index + 1,
                    "buy_date": dates[index + 1],
                    "buy_price": opens[index + 1],
                }
            continue
        if index <= entry["entry"]:
            continue  # T+1：买入当日不卖出
        held_days = index - entry["entry"] + 1
        reason = None
        if bool(sell.iloc[index]):
            reason = "SELL公式"
        elif held_days >= FORCE_SELL_DAYS:
            reason = "满%d天" % FORCE_SELL_DAYS
        if reason is None:
            continue
        sell_price = closes[index]
        cost = entry["buy_price"]
        rows.append({
            "code": code,
            "name": "",
            "buy_date": entry["buy_date"],
            "buy_price": round(cost, 3),
            "sell_date": dates[index],
            "sell_price": round(sell_price, 3),
            "holding_days": held_days,
            "return_pct": round((sell_price / cost - 1) * 100, 2) if cost else 0.0,
            "reason": reason,
        })
        entry = None
    still_open = 1 if entry else 0
    return rows, still_open


def write_trades_csv(rows, path=None):
    """写 hammer_trades.csv（utf-8-sig），返回路径。"""
    path = Path(path) if path else HAMMER_TRADES_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "code", "name", "buy_date", "buy_price", "sell_date", "sell_price",
        "holding_days", "return_pct", "reason",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return path


def run_sell_backtest(codes, start=None, end=None, progress=None):
    """对给定股票回放 BUY/SELL 公式，输出 hammer_trades.csv。

    返回 dict：rows（成交列表）、path（CSV 路径）、open_count（回测期末仍持仓数）。
    """
    codes = [str(code).strip().zfill(6) for code in codes if str(code).strip()]
    codes = sorted(set(codes))
    names = {str(row["code"]).zfill(6): row["name"]
             for row in db.query_all("SELECT code, name FROM stock_info")}
    rows, open_count, skipped = [], 0, []
    for index, code in enumerate(codes, start=1):
        if progress:
            progress.emit({"current": index - 1, "total": len(codes), "text": "回放卖点 " + code})
        try:
            frame = load_local_daily(code, "qfq")
            if frame.empty or len(frame) < DEFAULT_N + 10:
                skipped.append(code)
                continue
            if start:
                frame = frame[frame["date"] >= str(start)]
            if end:
                frame = frame[frame["date"] <= str(end)]
            frame = frame.reset_index(drop=True)
            if len(frame) < DEFAULT_N + 5:
                skipped.append(code)
                continue
            buy, sell = _signals(frame)
            trades, still_open = _simulate_trades(code, frame, buy, sell)
            for row in trades:
                row["name"] = names.get(code, "")
            rows.extend(trades)
            open_count += still_open
        except Exception:
            skipped.append(code)
        if progress:
            progress.emit({"current": index, "total": len(codes), "text": "已处理 " + code})
    rows.sort(key=lambda item: (item["buy_date"], item["code"]))
    path = write_trades_csv(rows)
    return {
        "rows": rows,
        "path": path,
        "open_count": open_count,
        "skipped": skipped,
        "codes": len(codes),
    }


def load_sell_signals(path=None):
    """读取 SELL 信号 CSV，兼容通达信导出列名与 hammer_trades.csv。

    返回 DataFrame：code / name / sell_date / reason，按日期升序。
    文件不存在时返回空表（不抛错），便于界面直接调用。
    """
    path = Path(path) if path else HAMMER_TRADES_CSV
    if not path.exists():
        return pd.DataFrame(columns=["code", "name", "sell_date", "reason"])
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    frame.columns = [str(col).strip() for col in frame.columns]
    lowered = {col.lower(): col for col in frame.columns}

    def _pick(*names):
        for name in names:
            if name in frame.columns:
                return name
            if name.lower() in lowered:
                return lowered[name.lower()]
        return None

    code_col = _pick("code", "代码", "股票代码", "证券代码")
    date_col = _pick("sell_date", "卖出日期", "date", "日期", "信号日期")
    name_col = _pick("name", "名称", "股票名称")
    reason_col = _pick("reason", "卖出原因", "原因", "信号")
    if code_col is None or date_col is None:
        raise ValueError("SELL 结果缺少代码或日期列：需要 code/代码 与 sell_date/日期")
    result = pd.DataFrame({
        "code": frame[code_col].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6),
        "name": frame[name_col].astype(str) if name_col else "",
        "sell_date": frame[date_col].astype(str).str[:10],
        "reason": frame[reason_col].astype(str) if reason_col else "SELL公式",
    })
    result = result.dropna(subset=["code"])
    result = result[result["sell_date"].str.len() == 10]
    return result.sort_values(["code", "sell_date"]).reset_index(drop=True)


def sell_alerts(positions, signals):
    """把 SELL 信号与持仓逐条比对，返回预警列表（只提醒，不自动清仓）。"""
    if positions is None or getattr(positions, "empty", True):
        return []
    if signals is None or getattr(signals, "empty", True):
        return []
    alerts = []
    for _, row in positions.iterrows():
        code = str(row.get("code", "")).zfill(6)
        subset = signals[signals["code"] == code]
        if subset.empty:
            continue
        buy_date = str(row.get("buy_date") or "")[:10]
        if buy_date:
            subset = subset[subset["sell_date"] >= buy_date]
        if subset.empty:
            continue
        latest = subset.iloc[-1]
        alerts.append({
            "id": int(row["id"]) if pd.notna(row.get("id")) else None,
            "code": code,
            "name": row.get("name", ""),
            "type": "SELL信号",
            "message": "%s 于 %s 触发卖出信号（%s），请人工核对是否清仓"
                       % (code, latest["sell_date"], latest["reason"]),
        })
    return alerts


if __name__ == "__main__":
    print(DISCLAIMER)
    db.init_db()
    buy_path = ensure_buy_formula()
    print("BUY 公式已写入本地公式库：", buy_path)
    buy_txt, sell_txt = save_formula_files()
    print("BUY 文本：", buy_txt)
    print("SELL 文本（通达信手动粘贴）：", sell_txt)
    print(TDX_STEPS)

    from core.position import load_positions

    positions = load_positions()
    codes = positions["code"].astype(str).tolist() if not positions.empty else []
    if not codes:
        print("持仓为空，跳过卖点回测；可先在“持仓盯盘”页导入持仓 CSV。")
    else:
        result = run_sell_backtest(codes, start="2024-01-01")
        print("卖点回测完成：%d 只股票，%d 笔成交，期末仍持仓 %d 笔，跳过 %d 只"
              % (result["codes"], len(result["rows"]), result["open_count"], len(result["skipped"])))
        print("成交明细：", result["path"])
        signals = load_sell_signals(result["path"])
        alerts = sell_alerts(positions, signals)
        print("持仓命中 SELL 信号：%d 条" % len(alerts))
        for alert in alerts:
            print("  [%s] %s" % (alert["type"], alert["message"]))
