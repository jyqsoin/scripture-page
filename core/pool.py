# -*- coding: utf-8 -*-
"""
股票池管理与离线过滤。

池类型：
- all：本地 stock_info 中全部沪深 A 股，不保存成员，不联网补全；
- custom：自选池；
- industry / concept：行业池、概念池，可手工维护或通过 JSON 导入；
- blacklist：黑名单，可建多个，过滤时取所有黑名单的并集。

过滤规则：ST、停牌/当日无行情、上市天数、最近交易日成交额、黑名单。
停牌与成交额使用不复权行情；若没有不复权数据，则回退前复权行情中的量额字段。

仅研究用途，不构成投资建议。
"""

import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db

DISCLAIMER = "仅研究用途，不构成投资建议"
POOL_TYPES = ("custom", "industry", "concept", "blacklist")


@dataclass
class FilterConfig:
    """股票池过滤参数。min_amount 单位为元，0 表示不限制。"""
    exclude_st: bool = True
    exclude_suspended: bool = True
    min_list_days: int = 0
    min_amount: float = 0.0
    exclude_blacklist: bool = True


def _market_of(code):
    if code.startswith("6"):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    if code.startswith(("4", "8")):
        return "bj"
    return ""


def _normalize_code(value):
    """统一为 6 位数字代码；非法代码返回 None。"""
    code = re.sub(r"\D", "", str(value or ""))
    if len(code) > 6:
        code = code[-6:]
    code = code.zfill(6)
    return code if len(code) == 6 and code.isdigit() else None


def _as_date_text(value):
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    raise ValueError("日期格式应为 YYYY-MM-DD 或 YYYYMMDD：%s" % value)


def _chunks(items, size=500):
    """SQLite 变量数量有限，批量查询分块执行。"""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _to_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def create_pool(name, pool_type="custom", remark=""):
    """创建股票池；同名同类型已存在时直接返回其 id。"""
    name = str(name).strip()
    if not name:
        raise ValueError("股票池名称不能为空")
    if pool_type not in POOL_TYPES:
        raise ValueError("pool_type 必须是 %s" % (POOL_TYPES,))
    db.init_db()
    existing = db.query_one(
        "SELECT id FROM pool WHERE name = ? AND type = ?", (name, pool_type)
    )
    if existing:
        return existing["id"]
    db.execute(
        "INSERT INTO pool(name, type, remark, created_at) VALUES (?, ?, ?, ?)",
        (name, pool_type, remark, db.now_str()),
    )
    return db.query_one(
        "SELECT id FROM pool WHERE name = ? AND type = ?", (name, pool_type)
    )["id"]


def delete_pool(pool_id):
    """删除股票池；pool_item 通过外键级联删除。"""
    return db.execute("DELETE FROM pool WHERE id = ?", (int(pool_id),))


def list_pools(pool_type=None):
    """列出股票池及成员数量。"""
    where = "WHERE p.type = ?" if pool_type else ""
    params = (pool_type,) if pool_type else ()
    return db.query_all(
        """
        SELECT p.id, p.name, p.type, p.remark, p.created_at,
               COUNT(i.code) AS member_count
        FROM pool p LEFT JOIN pool_item i ON p.id = i.pool_id
        %s GROUP BY p.id ORDER BY p.type, p.name
        """ % where,
        params,
    )


def get_pool(pool_id):
    return db.query_one("SELECT * FROM pool WHERE id = ?", (int(pool_id),))


def get_pool_codes(pool_id):
    """按代码排序返回池成员。"""
    rows = db.query_all(
        "SELECT code FROM pool_item WHERE pool_id = ? ORDER BY code", (int(pool_id),)
    )
    return [row["code"] for row in rows]


def add_codes(pool_id, codes):
    """
    批量加入股票池，自动去重。

    对本地尚无档案的代码只补 code/market，不用空名称覆盖已有股票名称。
    """
    pool_id = int(pool_id)
    if not get_pool(pool_id):
        raise ValueError("股票池不存在：%s" % pool_id)
    normalized = sorted({_normalize_code(code) for code in codes} - {None})
    if not normalized:
        return 0
    existing = _stock_map(normalized)
    new_stocks = [
        {"code": code, "market": _market_of(code)}
        for code in normalized if code not in existing
    ]
    db.upsert_stock_info(new_stocks)
    return db.executemany(
        "INSERT OR IGNORE INTO pool_item(pool_id, code, added_at) VALUES (?, ?, ?)",
        [(pool_id, code, db.now_str()) for code in normalized],
    )


def remove_codes(pool_id, codes):
    normalized = sorted({_normalize_code(code) for code in codes} - {None})
    return db.executemany(
        "DELETE FROM pool_item WHERE pool_id = ? AND code = ?",
        [(int(pool_id), code) for code in normalized],
    )


def replace_codes(pool_id, codes):
    """用给定代码整体替换池成员。"""
    db.execute("DELETE FROM pool_item WHERE pool_id = ?", (int(pool_id),))
    return add_codes(pool_id, codes)


def get_all_a_codes():
    """全 A 定义为本地股票档案中的沪深 A 股；是否完整取决于此前数据导入范围。"""
    rows = db.query_all("SELECT code FROM stock_info ORDER BY code")
    return [row["code"] for row in rows if _market_of(row["code"]) in ("sh", "sz")]


def add_blacklist(codes, name="默认黑名单"):
    """便捷加入黑名单；可传单个代码或代码列表。"""
    if isinstance(codes, str):
        codes = [codes]
    pool_id = create_pool(name, "blacklist", "过滤时自动排除")
    return add_codes(pool_id, codes)


def get_blacklist_codes():
    """所有黑名单池的并集。"""
    rows = db.query_all(
        """
        SELECT DISTINCT i.code FROM pool_item i
        JOIN pool p ON p.id = i.pool_id WHERE p.type = 'blacklist'
        """
    )
    return {row["code"] for row in rows}


def rebuild_industry_pools():
    """按 stock_info.industry 重建同名行业池；概念池不受影响。"""
    rows = db.query_all(
        "SELECT code, industry FROM stock_info WHERE industry <> '' ORDER BY code"
    )
    groups = {}
    for row in rows:
        groups.setdefault(row["industry"], []).append(row["code"])
    result = {}
    for industry, codes in groups.items():
        pool_id = create_pool(industry, "industry", "按本地股票档案自动分组")
        replace_codes(pool_id, codes)
        result[industry] = len(codes)
    return result


def export_pool(pool_id, path=None):
    """导出单池 JSON；代码附带名称，便于人工检查和再次导入。"""
    pool = get_pool(pool_id)
    if not pool:
        raise ValueError("股票池不存在：%s" % pool_id)
    codes = get_pool_codes(pool_id)
    stocks = _stock_map(codes)
    payload = {
        "name": pool["name"],
        "type": pool["type"],
        "remark": pool["remark"],
        "codes": [
            {"code": code, "name": stocks.get(code, {}).get("name", "")}
            for code in codes
        ],
        "exported_at": db.now_str(),
        "disclaimer": DISCLAIMER,
    }
    if path is None:
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", pool["name"])
        path = db.EXPORT_DIR / ("pool_%s_%s.json" % (pool["type"], safe_name))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def import_pool(path, overwrite=False):
    """
    从 JSON 导入股票池。

    codes 支持 ["600000"] 或 [{"code":"600000","name":"浦发银行"}] 两种元素。
    overwrite=True 时先清空同名同类型池的原有成员。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    name = str(payload.get("name", "")).strip()
    pool_type = payload.get("type", "custom")
    if not name or pool_type not in POOL_TYPES:
        raise ValueError("JSON 必须包含有效 name 和 type（custom/industry/concept/blacklist）")

    entries = []
    for item in payload.get("codes", []):
        entry = {"code": item} if isinstance(item, str) else dict(item)
        code = _normalize_code(entry.get("code"))
        if code:
            entry["code"] = code
            entry.setdefault("market", _market_of(code))
            entries.append(entry)
    # 只有携带额外档案信息时才更新 stock_info，避免空名称覆盖已有档案。
    informative = [
        entry for entry in entries
        if entry.get("name") or entry.get("industry") or entry.get("list_date")
    ]
    db.upsert_stock_info(informative)

    pool_id = create_pool(name, pool_type, payload.get("remark", ""))
    if overwrite:
        replace_codes(pool_id, [entry["code"] for entry in entries])
    else:
        add_codes(pool_id, [entry["code"] for entry in entries])
    return {"pool_id": pool_id, "imported": len(entries), "members": len(get_pool_codes(pool_id))}


def _stock_map(codes):
    """批量读取股票档案。"""
    result = {}
    for chunk in _chunks(list(codes)):
        if not chunk:
            continue
        placeholders = ",".join("?" * len(chunk))
        rows = db.query_all(
            "SELECT * FROM stock_info WHERE code IN (%s)" % placeholders, chunk
        )
        result.update({row["code"]: row for row in rows})
    return result


def _first_bar_map(codes):
    """批量读取本地最早行情日，供缺失上市日期的股票保守推断上市天数。"""
    result = {}
    for chunk in _chunks(list(codes)):
        if not chunk:
            continue
        placeholders = ",".join("?" * len(chunk))
        rows = db.query_all(
            """
            SELECT code, MIN(date) AS first_date FROM daily_bars
            WHERE code IN (%s) GROUP BY code
            """ % placeholders,
            chunk,
        )
        result.update({row["code"]: row["first_date"] for row in rows})
    return result


def _bar_map(codes, trade_date):
    """读取同一交易日的量额；每只股票优先 none，其次 qfq。"""
    result = {}
    for chunk in _chunks(list(codes)):
        if not chunk:
            continue
        placeholders = ",".join("?" * len(chunk))
        rows = db.query_all(
            """
            SELECT code, date, fq, volume, amount FROM daily_bars
            WHERE date = ? AND fq IN ('none', 'qfq') AND code IN (%s)
            """ % placeholders,
            [trade_date] + chunk,
        )
        for row in rows:
            old = result.get(row["code"])
            if old is None or row["fq"] == "none":
                result[row["code"]] = row
    return result


def latest_local_trade_date():
    """获取全库行情最近日期，避免混合复权口径时取到旧日期。"""
    row = db.query_one("SELECT MAX(date) AS date FROM daily_bars")
    return row["date"] if row else None


def latest_trade_date_on_or_before(date_text):
    """返回不晚于指定日期的最近行情日；非交易日自动前移。"""
    text = _as_date_text(date_text)
    row = db.query_one(
        "SELECT MAX(date) AS date FROM daily_bars WHERE date <= ?", (text,)
    )
    return row["date"] if row else None


def filter_stocks(codes, config=None, trade_date=None):
    """
    按统一交易日过滤股票，返回保留代码和每只剔除股票的具体原因。

    注意：trade_date 缺省为本地最近行情日。若数据没有更新到该日，股票也会因
    suspended_or_no_data 被剔除，因此正式选股前应先手动更新数据。
    上市天数按自然日计算；list_date 缺失时回退到本地最早行情日，
    仍无任何本地行情且设置上市天数限制时安全剔除。
    """
    db.init_db()
    config = config or FilterConfig()
    normalized = sorted({_normalize_code(code) for code in codes} - {None})
    trade_date = _as_date_text(trade_date) if trade_date else latest_local_trade_date()
    if not trade_date:
        raise ValueError("本地没有日线行情，无法执行停牌/成交额过滤")

    stocks = _stock_map(normalized)
    first_bars = _first_bar_map(normalized) if config.min_list_days > 0 else {}
    bars = _bar_map(normalized, trade_date)
    blacklist = get_blacklist_codes() if config.exclude_blacklist else set()
    kept, rejected = [], []

    def reject(code, reason, detail):
        rejected.append({
            "code": code, "name": stocks.get(code, {}).get("name", ""),
            "reason": reason, "detail": detail,
        })

    for code in normalized:
        stock = stocks.get(code, {})
        name = stock.get("name", "")
        if code in blacklist:
            reject(code, "blacklist", "属于黑名单")
            continue
        if config.exclude_st and "ST" in name.upper():
            reject(code, "st", "ST/*ST 股票")
            continue
        if config.min_list_days > 0:
            list_date = stock.get("list_date", "") or first_bars.get(code, "")
            if not list_date:
                reject(code, "missing_list_date", "缺少上市日期且本地无行情，无法验证上市天数")
                continue
            try:
                list_date = _as_date_text(list_date)
            except ValueError:
                reject(code, "missing_list_date", "上市日期格式非法：%s" % list_date)
                continue
            days = (datetime.strptime(trade_date, "%Y-%m-%d") -
                    datetime.strptime(list_date, "%Y-%m-%d")).days
            if days < config.min_list_days:
                reject(code, "list_days", "上市仅 %d 天，小于 %d 天" % (days, config.min_list_days))
                continue

        bar = bars.get(code)
        volume = _to_float(bar.get("volume")) if bar else None
        if config.exclude_suspended and (bar is None or volume is None or volume <= 0):
            reject(code, "suspended_or_no_data", "当日无行情或成交量为 0，可能停牌或数据未更新")
            continue
        if config.min_amount > 0:
            amount = _to_float(bar.get("amount")) if bar else None
            if amount is None or amount < config.min_amount:
                reject(code, "amount", "当日成交额低于 %.0f 元" % config.min_amount)
                continue
        kept.append(code)

    return {
        "trade_date": trade_date,
        "total": len(normalized),
        "codes": kept,
        "rejected": rejected,
        "disclaimer": DISCLAIMER,
    }


def get_universe(pool_id=None, config=None, trade_date=None):
    """pool_id=None 时使用本地全 A，否则使用指定池，然后应用过滤规则。"""
    codes = get_all_a_codes() if pool_id is None else get_pool_codes(pool_id)
    return filter_stocks(codes, config=config, trade_date=trade_date)


if __name__ == "__main__":
    # 使用临时 SQLite 自测，不联网，也不污染正式 data/aquant.db。
    import tempfile

    print(DISCLAIMER)
    with tempfile.TemporaryDirectory() as temp_dir:
        db.DB_PATH = Path(temp_dir) / "pool_test.db"
        db.init_db()
        trade_date = "2024-01-05"
        samples = [
            {"code": "600000", "name": "浦发银行", "market": "sh", "list_date": "1999-11-10"},
            {"code": "300001", "name": "特锐德", "market": "sz", "list_date": "2009-10-30"},
            {"code": "600001", "name": "*ST测试", "market": "sh", "list_date": "2000-01-01"},
            {"code": "600002", "name": "新股测试", "market": "sh", "list_date": "2023-12-25"},
            {"code": "600003", "name": "停牌测试", "market": "sh", "list_date": "2000-01-01"},
            {"code": "600004", "name": "黑名单测试", "market": "sh", "list_date": "2000-01-01"},
            {"code": "600005", "name": "低额测试", "market": "sh", "list_date": "2000-01-01"},
        ]
        db.upsert_stock_info(samples)
        for code, amount in [
            ("600000", 100000000), ("300001", 80000000), ("600001", 100000000),
            ("600002", 100000000), ("600004", 100000000), ("600005", 1000000),
        ]:
            db.insert_daily_bars(code, "none", [{
                "date": trade_date, "open": 10, "high": 11, "low": 9, "close": 10.5,
                "volume": 1000000, "amount": amount,
            }])

        pool_id = create_pool("自测自选", "custom")
        add_codes(pool_id, [item["code"] for item in samples])
        add_blacklist("600004")
        result = get_universe(
            pool_id,
            FilterConfig(min_list_days=60, min_amount=50000000),
            trade_date=trade_date,
        )
        print("保留：", result["codes"])
        print("剔除：", [(item["code"], item["reason"]) for item in result["rejected"]])
        assert result["codes"] == ["300001", "600000"]
        assert len(result["rejected"]) == 5

        json_path = Path(temp_dir) / "pool.json"
        export_pool(pool_id, json_path)
        imported = import_pool(json_path, overwrite=True)
        assert imported["members"] == len(samples)
        print("pool.py 自测通过")
