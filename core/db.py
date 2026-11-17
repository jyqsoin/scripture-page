# -*- coding: utf-8 -*-
"""
SQLite 本地存储层。

职责：
1. 统一管理本机数据目录与 SQLite 单文件连接；
2. 创建行情、股票池、公式、回测记录、配置等数据表；
3. 提供少量通用读写函数，供数据更新、公式、回测和 UI 层复用。

说明：
- 行情明细同时保存到 parquet 与 SQLite；parquet 适合高速读取，SQLite 便于统一查询和备份。
- 本模块不联网，也不依赖第三方库。
- 仅研究用途，不构成投资建议。
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

# aquant 项目根目录：本文件位于 aquant/core/db.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "aquant.db"
PARQUET_DIR = DATA_DIR / "parquet"
BACKUP_DIR = DATA_DIR / "backup"
EXPORT_DIR = DATA_DIR / "export"
LOG_DIR = PROJECT_ROOT / "logs"

DISCLAIMER = "仅研究用途，不构成投资建议"

# 回测红线对应的默认配置。后续 UI 可修改，但默认值严格按需求设置。
DEFAULT_CONFIG = {
    "commission": 0.0003,          # 手续费：万三
    "slippage": 0.001,             # 滑点：0.1%
    "max_position_per_stock": 0.10,  # 单票仓位不超过 10%
    "max_total_position": 0.80,    # 总仓位不超过 80%
    "stop_loss": 0.05,             # 跌破买入价 5% 强制出场
    "t_plus_one": True,            # A 股 T+1
    "close_confirm": True,         # 收盘判定，下一交易日开盘成交
    "data_source_order": ["ths", "akshare", "sina", "baostock", "tushare"],
    "tushare_token": "",           # 可选；为空时自动跳过 tushare
    "request_interval": 0.20,      # 每次联网请求后的间隔，避免请求过快
    "disclaimer": DISCLAIMER,
}

SCHEMA_SQL = """
-- 股票基础信息。code 统一为 6 位数字字符串。
CREATE TABLE IF NOT EXISTS stock_info (
    code        TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    market      TEXT NOT NULL DEFAULT '',       -- sh / sz / bj
    industry    TEXT NOT NULL DEFAULT '',
    list_date   TEXT NOT NULL DEFAULT '',       -- YYYY-MM-DD，未知时为空
    is_hs300    INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL
);

-- 日线行情。fq：none=不复权，qfq=前复权，hfq=后复权。
-- volume 统一为“股”，amount 统一为“元”。
CREATE TABLE IF NOT EXISTS daily_bars (
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,                  -- YYYY-MM-DD
    fq          TEXT NOT NULL DEFAULT 'none',
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    amount      REAL,
    PRIMARY KEY (code, date, fq)
);
CREATE INDEX IF NOT EXISTS idx_daily_bars_code_fq_date
    ON daily_bars(code, fq, date);

-- 每只股票、每种复权口径的更新状态，用于增量更新与错误排查。
CREATE TABLE IF NOT EXISTS daily_meta (
    code        TEXT NOT NULL,
    fq          TEXT NOT NULL,
    last_date   TEXT NOT NULL DEFAULT '',
    rows        INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'ok',     -- ok / error
    source      TEXT NOT NULL DEFAULT '',
    message     TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (code, fq)
);

-- 股票池。type：all / custom / industry / concept / blacklist。
CREATE TABLE IF NOT EXISTS pool (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'custom',
    remark      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    UNIQUE(name, type)
);

-- 股票池成员。黑名单也作为一种 pool 保存，避免维护多套结构。
CREATE TABLE IF NOT EXISTS pool_item (
    pool_id     INTEGER NOT NULL,
    code        TEXT NOT NULL,
    added_at    TEXT NOT NULL,
    PRIMARY KEY (pool_id, code),
    FOREIGN KEY (pool_id) REFERENCES pool(id) ON DELETE CASCADE
);

-- 本地公式库。kind：select=选股公式（布尔），indicator=指标公式（数值）。
CREATE TABLE IF NOT EXISTS formula (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL,
    content     TEXT NOT NULL,
    params      TEXT NOT NULL DEFAULT '{}',     -- JSON 参数定义
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- 回测记录。大段净值与交易明细保存为 CSV，这里只存路径和汇总指标。
CREATE TABLE IF NOT EXISTS backtest_record (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    params          TEXT NOT NULL DEFAULT '{}', -- JSON 回测参数
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    initial_cash    REAL,
    final_value     REAL,
    annual_return   REAL,
    sharpe          REAL,
    max_drawdown    REAL,
    win_rate        REAL,
    pl_ratio        REAL,
    trades_count    INTEGER NOT NULL DEFAULT 0,
    report_path     TEXT NOT NULL DEFAULT '',
    trades_path     TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

-- 简单键值配置。value 统一存 JSON 文本，读取时自动解析。
CREATE TABLE IF NOT EXISTS config (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- V1.2 Step G5：持仓管理（离线，不接券商、不下单）
CREATE TABLE IF NOT EXISTS positions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    code         TEXT NOT NULL,
    name         TEXT NOT NULL DEFAULT '',
    shares       INTEGER NOT NULL DEFAULT 0,
    cost_price   REAL NOT NULL DEFAULT 0,
    buy_date     TEXT NOT NULL DEFAULT '',
    stop_loss    REAL NOT NULL DEFAULT 0,   -- 止损比例（如 0.05 = -5%），0 表示不设
    take_profit  REAL NOT NULL DEFAULT 0,   -- 止盈比例（如 0.10 = +10%），0 表示不设
    target_price REAL NOT NULL DEFAULT 0,   -- 目标价预警，0 表示不设
    note         TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);
"""


def ensure_dirs():
    """创建运行所需目录；重复执行不会清空已有数据。"""
    for path in (DATA_DIR, PARQUET_DIR, BACKUP_DIR, EXPORT_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def now_str():
    """统一的时间字符串格式。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class LocalConnection(sqlite3.Connection):
    """with 语句结束时先提交/回滚，再关闭连接，避免残留文件句柄。"""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def get_conn(db_path=None):
    """
    获取 SQLite 连接。

    每次调用返回独立连接，避免 UI 线程与后台更新线程共享连接。
    可用 with 自动提交并关闭；手动使用时应调用 conn.close()。
    """
    ensure_dirs()
    conn = sqlite3.connect(
        str(db_path or DB_PATH), timeout=30, factory=LocalConnection
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL 允许读取与写入并行；备份时后续会同时处理 wal/shm 文件。
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db():
    """建表并写入默认配置；可重复执行，不会删除已有数据。"""
    ensure_dirs()
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)
        for key, value in DEFAULT_CONFIG.items():
            conn.execute(
                "INSERT OR IGNORE INTO config(key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), now_str()),
            )
    return DB_PATH


def execute(sql, params=()):
    """执行单条写 SQL，返回受影响行数。"""
    with get_conn() as conn:
        cursor = conn.execute(sql, params)
        return cursor.rowcount


def executemany(sql, rows):
    """批量执行写 SQL；空列表直接返回。"""
    rows = list(rows)
    if not rows:
        return 0
    with get_conn() as conn:
        cursor = conn.executemany(sql, rows)
        return cursor.rowcount


def query_all(sql, params=()):
    """查询多行，统一返回 list[dict]，方便转成 DataFrame。"""
    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def query_one(sql, params=()):
    """查询单行；没有结果时返回 None。"""
    conn = get_conn()
    try:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def get_config(key, default=None):
    """读取配置并解析 JSON；配置不存在或 JSON 非法时返回默认值。"""
    row = query_one("SELECT value FROM config WHERE key = ?", (key,))
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (TypeError, ValueError):
        return row["value"]


def set_config(key, value):
    """写入配置；列表、字典、数字均以 JSON 保存。"""
    execute(
        """
        INSERT INTO config(key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, json.dumps(value, ensure_ascii=False), now_str()),
    )


def upsert_stock_info(stocks, hs300=False):
    """
    批量保存股票信息。

    stocks 中每项至少包含 code，可包含 name/market/industry/list_date。
    hs300=True 时先把旧成分标记清零，再将本次名单标为沪深300成分。
    """
    stocks = list(stocks)
    if not stocks:
        return 0
    now = now_str()
    with get_conn() as conn:
        if hs300:
            conn.execute("UPDATE stock_info SET is_hs300 = 0")
        rows = []
        for item in stocks:
            code = str(item.get("code", "")).strip().zfill(6)
            if not code.isdigit():
                continue
            rows.append((
                code,
                str(item.get("name", "") or ""),
                str(item.get("market", "") or ""),
                str(item.get("industry", "") or ""),
                str(item.get("list_date", "") or ""),
                1 if hs300 else int(item.get("is_hs300", 0) or 0),
                now,
            ))
        conn.executemany(
            """
            INSERT INTO stock_info(code, name, market, industry, list_date, is_hs300, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                name = excluded.name,
                market = excluded.market,
                industry = CASE WHEN excluded.industry <> '' THEN excluded.industry
                                ELSE stock_info.industry END,
                list_date = CASE WHEN excluded.list_date <> '' THEN excluded.list_date
                                 ELSE stock_info.list_date END,
                is_hs300 = excluded.is_hs300,
                updated_at = excluded.updated_at
            """,
            rows,
        )
    return len(rows)


def insert_daily_bars(code, fq, bars):
    """
    批量写入日线；主键冲突时替换，因此同一股票同一天可安全重复更新。

    bars 为 dict 列表，字段至少包括 date/open/high/low/close/volume/amount。
    """
    rows = []
    for bar in bars:
        rows.append((
            code,
            str(bar["date"]),
            fq,
            bar.get("open"),
            bar.get("high"),
            bar.get("low"),
            bar.get("close"),
            bar.get("volume"),
            bar.get("amount"),
        ))
    return executemany(
        """
        INSERT INTO daily_bars(code, date, fq, open, high, low, close, volume, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(code, date, fq) DO UPDATE SET
            open = excluded.open, high = excluded.high, low = excluded.low,
            close = excluded.close, volume = excluded.volume, amount = excluded.amount
        """,
        rows,
    )


def delete_daily_bars(code, fq):
    """全量重拉成功后才调用，删除该股票该口径的旧 SQLite 行情。"""
    return execute("DELETE FROM daily_bars WHERE code = ? AND fq = ?", (code, fq))


def get_daily_meta(code, fq):
    """读取某只股票、某种复权口径的更新元数据。"""
    return query_one("SELECT * FROM daily_meta WHERE code = ? AND fq = ?", (code, fq))


def upsert_daily_meta(code, fq, last_date="", rows=0, status="ok", source="", message=""):
    """保存行情更新状态。message 用于记录失败原因，方便用户贴报错。"""
    execute(
        """
        INSERT INTO daily_meta(code, fq, last_date, rows, status, source, message, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(code, fq) DO UPDATE SET
            last_date = excluded.last_date,
            rows = excluded.rows,
            status = excluded.status,
            source = excluded.source,
            message = excluded.message,
            updated_at = excluded.updated_at
        """,
        (code, fq, last_date, int(rows), status, source, message, now_str()),
    )


if __name__ == "__main__":
    # 最小自测：只操作本地 SQLite，不联网。
    print(DISCLAIMER)
    path = init_db()
    print("数据库已初始化：", path)
    tables = query_all("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    print("数据表：", [item["name"] for item in tables])
    print("默认手续费：", get_config("commission"))
    print("默认回测模式：收盘确认 =", get_config("close_confirm"))
    assert get_config("max_total_position") == 0.80
    assert get_config("stop_loss") == 0.05
    print("db.py 自测通过")
