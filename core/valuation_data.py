# -*- coding: utf-8 -*-
"""历史估值数据接入（Step I2-alt-4A）。

使用 Baostock 历史日频 peTTM / pbMRQ，保存到本地 SQLite 与 parquet。
估值日期是数据可见日期，后续回测只能取 trade_date 当日或之前的记录；
PE/PB 不属于复权价格，因此本表不使用 fq 字段。

仅研究用途，不构成投资建议。
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core.data_loader import _market_of

DISCLAIMER = "仅研究用途，不构成投资建议"
VALUATION_DIR = db.DATA_DIR / "valuation"
VALUATION_COLUMNS = ["date", "code", "pe_ttm", "pb_mrq", "tradestatus", "is_st"]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS valuation_daily (
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,
    pe_ttm      REAL,
    pb_mrq      REAL,
    tradestatus INTEGER,
    is_st       INTEGER,
    source      TEXT NOT NULL DEFAULT 'baostock',
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_valuation_code_date
    ON valuation_daily(code, date);
"""


def ensure_valuation_schema():
    """独立创建估值表，不改动 V1.0-V1.4 的数据库 schema 常量。"""
    db.init_db()
    with db.get_conn() as conn:
        conn.executescript(SCHEMA_SQL)


def valuation_parquet_path(code):
    """每只股票一个估值 parquet，目录与日线数据分离。"""
    return db.PARQUET_DIR / "valuation" / (code + ".parquet")


def empty_valuation():
    return pd.DataFrame(columns=VALUATION_COLUMNS)


def clean_valuation(raw, code):
    """统一日期、6 位代码和数值字段；保留 PE/PB 缺失及非正值供覆盖率统计。"""
    if raw is None or raw.empty:
        return empty_valuation()
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["code"] = str(code).zfill(6)
    for column in ("pe_ttm", "pb_mrq", "tradestatus", "is_st"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return frame[VALUATION_COLUMNS].reset_index(drop=True)


def _query_baostock(code, start_date, end_date):
    """在已登录状态下查询单只股票；adjustflag=3 表示不复权。"""
    import baostock as bs
    market = _market_of(code)
    if not market:
        raise ValueError("无法识别股票交易所：%s" % code)
    result = bs.query_history_k_data_plus(
        market + "." + code,
        "date,code,peTTM,pbMRQ,tradestatus,isST",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3",
    )
    if result.error_code != "0":
        raise RuntimeError("Baostock 查询失败：%s" % result.error_msg)
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    raw = pd.DataFrame(rows, columns=result.fields)
    return clean_valuation(raw.rename(columns={"peTTM": "pe_ttm", "pbMRQ": "pb_mrq"}), code)


def fetch_valuation_baostock(code, start_date, end_date):
    """单只股票下载接口，独立登录；批量任务应使用 update_valuations。"""
    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError("Baostock 登录失败：%s" % login.error_msg)
    try:
        return _query_baostock(code, start_date, end_date)
    finally:
        bs.logout()


def load_valuation(code, start_date=None, end_date=None):
    """离线读取估值历史，优先 parquet，缺失时回退 SQLite。"""
    code = str(code).zfill(6)
    path = valuation_parquet_path(code)
    if path.exists():
        frame = clean_valuation(pd.read_parquet(path), code)
    else:
        rows = db.query_all(
            "SELECT date, code, pe_ttm, pb_mrq, tradestatus, is_st "
            "FROM valuation_daily WHERE code=? ORDER BY date", (code,))
        frame = clean_valuation(pd.DataFrame(rows), code)
    if start_date:
        frame = frame[frame["date"] >= start_date]
    if end_date:
        frame = frame[frame["date"] <= end_date]
    return frame.reset_index(drop=True)


def latest_valuation(frame, trade_date):
    """从单只股票估值表中取 trade_date 可见的最新一条，无记录返回 None。"""
    if frame is None or frame.empty:
        return None
    visible = frame[frame["date"] <= trade_date]
    if visible.empty:
        return None
    return visible.iloc[-1].to_dict()


def save_valuation(code, new_frame, source="baostock", full_replace=False):
    """同步写 parquet 与 SQLite；默认合并去重，非空数据才允许全量替换。"""
    ensure_valuation_schema()
    code = str(code).zfill(6)
    new_frame = clean_valuation(new_frame, code)
    if new_frame.empty:
        return empty_valuation()
    path = valuation_parquet_path(code)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = clean_valuation(pd.read_parquet(path), code) if path.exists() else empty_valuation()
    if full_replace or old.empty:
        final = new_frame.copy()
    else:
        final = pd.concat([old, new_frame], ignore_index=True)
    final = clean_valuation(final, code)
    final["source"] = source
    final.to_parquet(path, index=False)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # parquet 成功后才更新 SQLite；增量模式只写本次新数据，避免重复批量写旧历史
    to_write = final if full_replace else new_frame
    with db.get_conn() as conn:
        if full_replace:
            conn.execute("DELETE FROM valuation_daily WHERE code=?", (code,))
        rows = [(
            code, row["date"], row["pe_ttm"], row["pb_mrq"],
            row["tradestatus"], row["is_st"], source, now,
        ) for row in to_write.to_dict("records")]
        conn.executemany(
            """
            INSERT INTO valuation_daily(
                code, date, pe_ttm, pb_mrq, tradestatus, is_st, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code, date) DO UPDATE SET
                pe_ttm=excluded.pe_ttm, pb_mrq=excluded.pb_mrq,
                tradestatus=excluded.tradestatus, is_st=excluded.is_st,
                source=excluded.source, updated_at=excluded.updated_at
            """, rows)
    return final


def _summary_row(code, frame, status="ok", message=""):
    """生成单只股票覆盖率统计；非正值占比以非空估值记录为分母。"""
    rows = len(frame)
    pe = frame["pe_ttm"] if rows else pd.Series(dtype=float)
    pb = frame["pb_mrq"] if rows else pd.Series(dtype=float)
    return {
        "code": code, "status": status, "message": message, "rows": rows,
        "first_date": frame["date"].min() if rows else "",
        "last_date": frame["date"].max() if rows else "",
        "pe_valid_rows": int(pe.notna().sum()), "pb_valid_rows": int(pb.notna().sum()),
        "pe_coverage": float(pe.notna().mean()) if rows else 0.0,
        "pb_coverage": float(pb.notna().mean()) if rows else 0.0,
        "pe_nonpositive_rows": int((pe.dropna() <= 0).sum()),
        "pb_nonpositive_rows": int((pb.dropna() <= 0).sum()),
        "pe_nonpositive_ratio": float((pe.dropna() <= 0).mean()) if pe.notna().any() else 0.0,
        "pb_nonpositive_ratio": float((pb.dropna() <= 0).mean()) if pb.notna().any() else 0.0,
    }


def update_valuations(codes, start_date="2024-05-01", end_date="2026-11-13",
                      request_interval=0.2, progress_cb=None):
    """共用一次登录批量更新，逐股容错并返回每只股票覆盖率。"""
    import baostock as bs
    codes = sorted({str(code).zfill(6) for code in codes})
    if not codes:
        raise ValueError("股票代码列表为空")
    ensure_valuation_schema()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError("Baostock 登录失败：%s" % login.error_msg)
    summaries = []
    try:
        for index, code in enumerate(codes, 1):
            try:
                downloaded = _query_baostock(code, start_date, end_date)
                if downloaded.empty:
                    summaries.append(_summary_row(code, empty_valuation(), "no_data", "接口无数据"))
                else:
                    final = save_valuation(code, downloaded)
                    summaries.append(_summary_row(code, final))
            except Exception as exc:
                summaries.append(_summary_row(code, empty_valuation(), "error", str(exc)))
            if progress_cb:
                progress_cb(index, len(codes), "估值 %s %s" % (code, summaries[-1]["status"]))
            if request_interval > 0 and index < len(codes):
                time.sleep(request_interval)
    finally:
        bs.logout()
    return pd.DataFrame(summaries)


def save_coverage_report(summary, start_date, end_date, directory=None):
    """保存逐股 CSV 与总体 JSON 覆盖率报告。"""
    directory = Path(directory) if directory else VALUATION_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stem = "valuation_coverage_%s_%s_%d" % (
        start_date.replace("-", ""), end_date.replace("-", ""), len(summary))
    csv_path = directory / (stem + ".csv")
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    ok = summary[summary["status"] == "ok"]
    total_rows = int(ok["rows"].sum())
    pe_valid = int(ok["pe_valid_rows"].sum())
    pb_valid = int(ok["pb_valid_rows"].sum())
    payload = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "baostock", "start": start_date, "end": end_date,
            "codes": int(len(summary)), "success_codes": int(len(ok)),
            "error_codes": summary.loc[summary["status"] == "error", "code"].tolist(),
            "no_data_codes": summary.loc[summary["status"] == "no_data", "code"].tolist(),
            "disclaimer": DISCLAIMER,
        },
        "overall": {
            "rows": total_rows,
            "pe_coverage": pe_valid / total_rows if total_rows else 0.0,
            "pb_coverage": pb_valid / total_rows if total_rows else 0.0,
            "pe_nonpositive_ratio": float(ok["pe_nonpositive_rows"].sum() / pe_valid) if pe_valid else 0.0,
            "pb_nonpositive_ratio": float(ok["pb_nonpositive_rows"].sum() / pb_valid) if pb_valid else 0.0,
        },
        "by_code": summary.to_dict("records"),
    }
    json_path = directory / (stem + ".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    default=str), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "overall": payload["overall"]}


def synthetic_valuation_test():
    """合成数据验证清洗、去重、最近可见日期与防未来函数。"""
    raw = pd.DataFrame({
        "date": ["2024-05-10", "2024-05-06", "2024-05-08", "2024-05-08"],
        "peTTM": [30, 20, 24, 25], "pbMRQ": [3, 2, 2.4, 2.5],
        "tradestatus": [1] * 4, "isST": [0] * 4,
    }).rename(columns={"peTTM": "pe_ttm", "pbMRQ": "pb_mrq"})
    frame = clean_valuation(raw, "000001")
    assert len(frame) == 3 and frame["date"].is_monotonic_increasing
    assert frame.loc[frame["date"] == "2024-05-08", "pe_ttm"].iloc[0] == 25
    latest = latest_valuation(frame, "2024-05-09")
    assert latest["date"] == "2024-05-08" and latest["pe_ttm"] == 25
    changed = frame.copy()
    changed.loc[changed["date"] > "2024-05-09", "pe_ttm"] = 999
    assert latest_valuation(changed, "2024-05-09")["pe_ttm"] == 25
    assert latest_valuation(frame, "2024-05-01") is None
    print("合成数据：估值清洗、去重、最近可见日期与防未来函数校验通过")


if __name__ == "__main__":
    print(DISCLAIMER)
    synthetic_valuation_test()
    if "--self-test" in sys.argv:
        print("valuation_data.py 仅合成自测完成；%s" % DISCLAIMER)
        sys.exit(0)

    codes = [row["code"] for row in db.query_all(
        "SELECT DISTINCT code FROM daily_bars WHERE fq='qfq' ORDER BY code")]
    summary = update_valuations(codes, "2024-05-01", "2026-11-13",
                                request_interval=float(db.get_config("request_interval", 0.2)),
                                progress_cb=lambda done, total, text: print("[%d/%d] %s" % (done, total, text)))
    paths = save_coverage_report(summary, "2024-05-01", "2026-11-13")
    assert len(summary) == len(codes) == 300
    print("\n=== 历史估值覆盖率 ===")
    print(json.dumps(paths["overall"], ensure_ascii=False, indent=2))
    print("成功 %d / %d 只" % ((summary["status"] == "ok").sum(), len(summary)))
    print("CSV:", paths["csv"])
    print("JSON:", paths["json"])
    print("valuation_data.py 自测与真实数据接入完成；%s" % DISCLAIMER)
