# -*- coding: utf-8 -*-
"""
日线与分钟 K 线下载、增量更新和本地持久化。

日线数据源优先级：akshare -> baostock -> tushare（需先在 config 中配置 token）。
分钟当前使用 akshare 东财接口，失败只记录错误并保留旧数据。
仅在用户主动调用更新函数时联网；读取本地行情完全离线。

日线持久化规则：
- 每只股票、每种复权口径保存一个 parquet 文件，同时写入 SQLite daily_bars。
- 不复权数据按最后交易日增量追加。
- 前/后复权数据会因新的除权除息而改变历史价格，因此每次全量重拉并覆盖；
  只有接口成功返回非空数据后才替换旧数据，避免失败时清空历史。
- 分钟原始数据独立保存到 data/minute，SQLite minute_meta 只存索引；
  前复权在读取时复用本地日线因子，不写日线表。
- volume 统一为“股”，amount 统一为“元”。

仅研究用途，不构成投资建议。
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 允许两种运行方式：
# 1. 在项目根目录运行 python -m core.data_loader；
# 2. 直接运行 python core/data_loader.py。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db, logger

FQ_TYPES = ("none", "qfq", "hfq")
AK_ADJUST = {"none": "", "qfq": "qfq", "hfq": "hfq"}
SINA_ADJUST = {"none": "", "qfq": "qfq", "hfq": "hfq"}
# baostock：1=后复权，2=前复权，3=不复权
BS_ADJUST = {"none": "3", "qfq": "2", "hfq": "1"}

DAILY_COLUMNS = ["date", "code", "open", "high", "low", "close", "volume", "amount"]

# 分钟原始数据独立存放，不写 daily_bars，也不覆盖任何日线 parquet。
MINUTE_PERIODS = ("1min", "5min", "15min", "30min", "60min")
MINUTE_API_PERIOD = {"1min": "1", "5min": "5", "15min": "15", "30min": "30", "60min": "60"}
MINUTE_COLUMNS = ["datetime", "code", "open", "high", "low", "close", "volume", "amount"]
MINUTE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS minute_meta (
    code          TEXT NOT NULL,
    period        TEXT NOT NULL,
    last_datetime TEXT NOT NULL DEFAULT '',
    rows          INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'ok',
    source        TEXT NOT NULL DEFAULT '',
    message       TEXT NOT NULL DEFAULT '',
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (code, period)
);
CREATE INDEX IF NOT EXISTS idx_minute_meta_updated_at
    ON minute_meta(updated_at);
"""


def empty_daily():
    """返回结构统一的空行情表。"""
    return pd.DataFrame(columns=DAILY_COLUMNS)


def _as_date(value):
    """把 YYYYMMDD、YYYY-MM-DD、datetime 统一转为 date。"""
    if value is None:
        return datetime.now().date()
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError("无法识别日期：%s，请使用 YYYY-MM-DD 或 YYYYMMDD" % value)


def _market_of(code):
    """按 A 股代码前缀判断交易所。"""
    if code.startswith("6"):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    if code.startswith(("4", "8")):
        return "bj"
    return ""


def _normalize_constituents(raw):
    """兼容不同 akshare 成分股接口的列名，输出 code/name/market。"""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["code", "name", "market"])
    code_candidates = ["成分券代码", "品种代码", "股票代码", "证券代码", "A股代码", "代码", "code", "con_code"]
    name_candidates = ["成分券名称", "品种名称", "股票名称", "证券简称", "A股简称", "名称", "name", "con_name"]
    code_col = next((c for c in code_candidates if c in raw.columns), None)
    name_col = next((c for c in name_candidates if c in raw.columns), None)
    if code_col is None:
        raise RuntimeError("成分股接口返回列发生变化，实际列为：%s" % list(raw.columns))

    result = pd.DataFrame()
    result["code"] = (raw[code_col].astype(str).str.strip()
                      .str.replace(r"\.0$", "", regex=True).str.zfill(6))
    result["name"] = raw[name_col].astype(str).str.strip() if name_col else ""
    result = result[result["code"].str.fullmatch(r"\d{6}")].copy()
    result["market"] = result["code"].map(_market_of)
    return result.drop_duplicates("code").reset_index(drop=True)


def get_all_ashare_constituents():
    """
    获取上证、深证、创业板全部 A 股（去重），替代原沪深300成分列表。

    优先沪 A / 深 A / 创业板快照；东方财富限流时改用沪深交易所官网名单；
    最后才尝试全 A 快照。只保留 60/00/30 开头代码，排除科创板、北交所与 B 股。
    """
    import akshare as ak

    errors = []
    frames = []
    for name in ("stock_sh_a_spot_em", "stock_sz_a_spot_em", "stock_cy_a_spot_em"):
        func = getattr(ak, name, None)
        if func is None:
            errors.append("%s：当前 akshare 版本无此接口" % name)
            continue
        try:
            part = _normalize_constituents(func())
            if part.empty:
                errors.append("%s：返回空数据" % name)
            else:
                frames.append(part)
        except Exception as exc:  # 单接口失败不阻塞其他市场
            errors.append("%s：%s" % (name, exc))

    if frames:
        result = pd.concat(frames, ignore_index=True)
        source = "沪A/深A/创业板快照"
    else:
        # 备用：交易所官网名单，不依赖东方财富快照
        exchange_sources = (
            ("stock_info_sh_name_code", {"symbol": "主板A股"}),
            ("stock_info_sz_name_code", {"symbol": "A股列表"}),
        )
        for name, kwargs in exchange_sources:
            func = getattr(ak, name, None)
            if func is None:
                errors.append("%s：当前 akshare 版本无此接口" % name)
                continue
            try:
                part = _normalize_constituents(func(**kwargs))
                if part.empty:
                    errors.append("%s(%s)：返回空数据" % (name, kwargs.get("symbol")))
                else:
                    frames.append(part)
            except Exception as exc:
                errors.append("%s(%s)：%s" % (name, kwargs.get("symbol"), exc))
        if frames:
            result = pd.concat(frames, ignore_index=True)
            source = "沪深交易所官网名单"
        else:
            func = getattr(ak, "stock_zh_a_spot_em", None)
            if func is None:
                raise RuntimeError("全 A 股票列表获取失败：\n" + "\n".join(errors))
            try:
                result = _normalize_constituents(func())
                source = "全 A 快照"
            except Exception as exc:
                errors.append("stock_zh_a_spot_em：%s" % exc)
                raise RuntimeError("全 A 股票列表获取失败：\n" + "\n".join(errors))

    result = result[result["code"].str.startswith(("60", "00", "30"))]
    result = result.drop_duplicates("code").reset_index(drop=True)
    print("全 A 股票来源：%s，共 %d 只" % (source, len(result)))
    return result


def _clean_daily(raw, code):
    """统一行情字段类型、日期格式、排序与去重。"""
    if raw is None or raw.empty:
        return empty_daily()
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["code"] = code
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return df[DAILY_COLUMNS].reset_index(drop=True)


def fetch_daily_akshare(code, start_date, end_date, fq):
    """东方财富 A 股日线接口；start/end 为 date 对象。"""
    import akshare as ak

    func = getattr(ak, "stock_zh_a_hist_em", None) or getattr(ak, "stock_zh_a_hist", None)
    if func is None:
        raise RuntimeError("当前 akshare 版本无 stock_zh_a_hist_em 接口，请升级 akshare")
    raw = func(
        symbol=code,
        period="daily",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust=AK_ADJUST[fq],
    )
    if raw is None or raw.empty:
        return empty_daily()
    rename = {
        "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount",
    }
    missing = [col for col in rename if col not in raw.columns]
    if missing:
        raise RuntimeError("stock_zh_a_hist 返回列变化，缺少 %s；实际为 %s" % (missing, list(raw.columns)))
    df = raw.rename(columns=rename)
    # 东财成交量单位为“手”，统一换算成“股”；成交额本身为“元”。
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100
    return _clean_daily(df, code)


def fetch_daily_baostock(code, start_date, end_date, fq):
    """baostock 备用日线接口；volume 原本就是股，amount 为元。"""
    import baostock as bs

    bs_code = _market_of(code) + "." + code
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError("baostock 登录失败：%s" % login.error_msg)
    try:
        result = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag=BS_ADJUST[fq],
        )
        if result.error_code != "0":
            raise RuntimeError("baostock 查询失败：%s" % result.error_msg)
        rows = []
        while result.next():
            rows.append(result.get_row_data())
        raw = pd.DataFrame(rows, columns=result.fields)
    finally:
        bs.logout()
    return _clean_daily(raw, code)


def fetch_daily_tushare(code, start_date, end_date, fq):
    """
    tushare 可选数据源。

    未配置 token 时直接抛出明确异常，调度层会跳过它。
    复权使用 pro_bar；不复权使用 pro.daily。
    """
    token = db.get_config("tushare_token", "")
    if not token:
        raise RuntimeError("未配置 tushare_token")
    import tushare as ts

    ts.set_token(token)
    ts_code = code + "." + _market_of(code).upper()
    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")
    if fq == "none":
        raw = ts.pro_api().daily(ts_code=ts_code, start_date=start, end_date=end)
    else:
        func = getattr(ts, "pro_bar", None)
        if func is None:
            raise RuntimeError("当前 tushare 版本无 pro_bar 接口")
        raw = func(ts_code=ts_code, adj=fq, start_date=start, end_date=end)
    if raw is None or raw.empty:
        return empty_daily()

    df = raw.rename(columns={"trade_date": "date", "vol": "volume"})
    # tushare vol 单位为手、amount 单位为千元，分别换算为股和元。
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000
    return _clean_daily(df, code)


def fetch_daily_sina(code, start_date, end_date, fq):
    """新浪 A 股日线接口；volume 为股，amount 为元，作为东财限流时的备用源。"""
    import akshare as ak

    func = getattr(ak, "stock_zh_a_daily_sina", None)
    if func is None:
        raise RuntimeError("当前 akshare 版本无 stock_zh_a_daily_sina 接口，请升级 akshare")
    raw = func(
        symbol=_market_of(code) + code,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        adjust=SINA_ADJUST[fq],
    )
    if raw is None or raw.empty:
        return empty_daily()
    return _clean_daily(raw, code)


# 同花顺 v4 年度日线：00=不复权、01=前复权、02=后复权；
# 返回纯文本 CSV，每根 11 列：日期,开,高,低,收,成交量(股),成交额(元),换手率,...
THS_KIND = {"none": "00", "qfq": "01", "hfq": "02"}
THS_URL = "https://d.10jqka.com.cn/v4/line/hs_%s/%s/%d.js"
THS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "http://q.10jqka.com.cn",
    "Host": "d.10jqka.com.cn",
}


def fetch_daily_ths(code, start_date, end_date, fq):
    """
    同花顺 A 股日线接口，按年份取数后合并；volume 为股，amount 为元。

    同花顺一次只返回一个自然年，跨年区间会拆成多次请求；
    最新一根的复权价与东财/新浪一致，因此增量追加不会出现拼接跳空。
    """
    import json

    import requests

    kind = THS_KIND[fq]
    frames = []
    for year in range(start_date.year, end_date.year + 1):
        resp = requests.get(
            THS_URL % (code, kind, year), headers=THS_HEADERS, timeout=15)
        if resp.status_code != 200:
            continue
        text = resp.text
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left:
            continue
        try:
            payload = json.loads(text[left:right + 1])
        except ValueError:
            continue
        rows = []
        for bar in str(payload.get("data") or "").split(";"):
            fields = bar.split(",")
            if len(fields) < 7:
                continue
            rows.append({
                "date": fields[0],
                "open": fields[1],
                "high": fields[2],
                "low": fields[3],
                "close": fields[4],
                "volume": fields[5],
                "amount": fields[6],
            })
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return empty_daily()
    df = _clean_daily(pd.concat(frames, ignore_index=True), code)
    start_text = start_date.strftime("%Y-%m-%d")
    end_text = end_date.strftime("%Y-%m-%d")
    df = df[(df["date"] >= start_text) & (df["date"] <= end_text)]
    return df.reset_index(drop=True)


FETCHERS = {
    "ths": fetch_daily_ths,
    "akshare": fetch_daily_akshare,
    "sina": fetch_daily_sina,
    "baostock": fetch_daily_baostock,
    "tushare": fetch_daily_tushare,
}


def parquet_path(code, fq):
    """本地 parquet 路径：data/parquet/daily/{fq}/{code}.parquet。"""
    return db.PARQUET_DIR / "daily" / fq / (code + ".parquet")


def load_local_daily(code, fq="qfq"):
    """
    离线读取本地日线。优先 parquet；文件不存在时回退 SQLite。

    返回按日期升序的 DataFrame，不访问网络。
    """
    path = parquet_path(code, fq)
    if path.exists():
        return _clean_daily(pd.read_parquet(path), code)
    rows = db.query_all(
        """
        SELECT date, code, open, high, low, close, volume, amount
        FROM daily_bars WHERE code = ? AND fq = ? ORDER BY date
        """,
        (code, fq),
    )
    return _clean_daily(pd.DataFrame(rows), code)


def save_daily(code, fq, new_df, full_replace, source):
    """
    保存一次成功下载的数据，并同步 parquet/SQLite/更新元数据。

    full_replace=True：用新数据整体替换该口径旧数据；
    full_replace=False：旧数据与新数据按日期合并去重。
    """
    new_df = _clean_daily(new_df, code)
    if new_df.empty:
        return 0
    path = parquet_path(code, fq)
    path.parent.mkdir(parents=True, exist_ok=True)

    if full_replace:
        final = new_df
    else:
        old = _clean_daily(pd.read_parquet(path), code) if path.exists() else empty_daily()
        # 旧数据为空时直接使用新数据，避免 concat 触发 FutureWarning 刷屏。
        final = new_df if old.empty else pd.concat([old, new_df], ignore_index=True)
        final = _clean_daily(final, code)

    final.to_parquet(path, index=False)

    # 先成功写 parquet，再同步 SQLite；全量替换只在有非空新数据时执行。
    if full_replace:
        db.delete_daily_bars(code, fq)
    db.insert_daily_bars(code, fq, final.to_dict("records") if full_replace else new_df.to_dict("records"))
    db.upsert_daily_meta(
        code, fq,
        last_date=str(final["date"].max()),
        rows=len(final),
        status="ok",
        source=source,
        message="",
    )
    return len(final)


def update_one(
    code,
    fq="qfq",
    start_date="20050101",
    end_date=None,
    sources=None,
    retry=2,
    request_interval=None,
):
    """
    更新单只股票、单种复权口径。

    每个数据源失败后立即尝试下一个；一轮全部失败后等待再重试。
    返回统计字典，不把单票异常抛给全量任务。
    """
    code = str(code).strip().zfill(6)
    if fq not in FQ_TYPES:
        raise ValueError("fq 必须是 none/qfq/hfq，收到：%s" % fq)
    db.init_db()
    start = _as_date(start_date)
    end = _as_date(end_date)
    meta = db.get_daily_meta(code, fq)

    # 复权历史会变化，所以 qfq/hfq 全量覆盖；none 只做真正增量。
    full_replace = fq in ("qfq", "hfq") or not meta or not meta.get("last_date")
    fetch_start = start
    if not full_replace:
        fetch_start = max(start, _as_date(meta["last_date"]) + timedelta(days=1))
    if fetch_start > end:
        return {"code": code, "fq": fq, "status": "up_to_date", "rows": 0}

    sources = sources or db.get_config(
        "data_source_order", ["ths", "akshare", "sina", "baostock", "tushare"])
    interval = db.get_config("request_interval", 0.2) if request_interval is None else request_interval
    errors = []

    for attempt in range(retry + 1):
        for source in sources:
            fetcher = FETCHERS.get(source)
            if fetcher is None:
                errors.append("%s：未知数据源" % source)
                continue
            if source == "tushare" and not db.get_config("tushare_token", ""):
                continue  # 未配置 token 属于正常跳过，不算失败
            try:
                df = fetcher(code, fetch_start, end, fq)
                if df.empty:
                    local = load_local_daily(code, fq)
                    last_date = str(local["date"].max()) if not local.empty else (meta or {}).get("last_date", "")
                    db.upsert_daily_meta(
                        code, fq, last_date=last_date, rows=len(local), status="ok",
                        source=source, message="接口返回空数据，未覆盖旧数据",
                    )
                    return {"code": code, "fq": fq, "status": "no_new_data", "rows": 0}
                total = save_daily(code, fq, df, full_replace=full_replace, source=source)
                return {
                    "code": code, "fq": fq, "status": "ok", "source": source,
                    "rows": len(df), "total_rows": total,
                }
            except Exception as exc:
                errors.append("第%d轮 %s：%s" % (attempt + 1, source, exc))
        if attempt < retry:
            time.sleep(max(interval, 0.2))

    message = " | ".join(errors)
    old_rows = db.query_one(
        "SELECT COUNT(*) AS n FROM daily_bars WHERE code = ? AND fq = ?", (code, fq)
    )["n"]
    db.upsert_daily_meta(
        code, fq,
        last_date=(meta or {}).get("last_date", ""),
        rows=old_rows,
        status="error",
        source="",
        message=message,
    )
    return {"code": code, "fq": fq, "status": "error", "rows": 0, "error": message}


def update_all_ashare_daily(
    fq_list=("qfq",),
    start_date="20050101",
    end_date=None,
    codes=None,
    limit=None,
    retry=2,
    progress_cb=None,
    workers=6,
    incremental=True,
):
    """
    更新上证、深证、创业板全部 A 股日线（并发下载、单线程写库）。

    参数：
    - fq_list：可传 ('none', 'qfq', 'hfq')；默认只更新研究最常用的前复权，避免首次三倍请求。
    - codes：可指定子集；不传则联网获取全部 A 股列表并写入 stock_info。
    - limit：仅取前 N 只，便于本地小范围测试。
    - progress_cb(current, total, code, fq, result)：供后续 UI 显示进度。
    - workers：联网并发线程数；写库始终在主线程串行，避免 SQLite 写冲突。
    - incremental：本地已有数据时只补最后交易日之后的缺口，不再整段重拉复权历史；
      仅新股或本地无数据时才从 start_date 全量下载。日常全量更新由此从约 90 分钟
      降到几分钟。

    单票失败不会中断整体任务，最终返回汇总并保存到 config。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    db.init_db()
    if codes is None:
        constituents = get_all_ashare_constituents()
        with db.get_conn() as conn:
            conn.execute("UPDATE stock_info SET is_hs300 = 0")
        db.upsert_stock_info(constituents.to_dict("records"), hs300=False)
        target_codes = constituents["code"].tolist()
    else:
        target_codes = list(codes)
    if limit is not None:
        target_codes = target_codes[:int(limit)]
    summary = {"total": 0, "ok": 0, "no_change": 0, "error": 0, "errors": []}

    start = _as_date(start_date)
    end = _as_date(end_date)
    sources = db.get_config(
        "data_source_order", ["ths", "akshare", "sina", "baostock", "tushare"])
    sources = list(sources)
    if "ths" not in sources:  # 兼容旧配置：同花顺未列入时放到最前
        sources.insert(0, "ths")

    def fetch_only(code, fq, fetch_start, has_local):
        """线程内只联网取数，不碰 SQLite；返回 (df, source, errors)。"""
        errors = []
        for attempt in range(retry + 1):
            had_error = False
            for source in sources:
                if source == "sina":
                    # akshare 新浪接口要初始化 py_mini_racer/V8，子线程中会直接
                    # 触发 FATAL 使进程退出；批量并发只走同花顺/东财/baostock。
                    continue
                fetcher = FETCHERS.get(source)
                if fetcher is None:
                    continue
                if source == "tushare" and not db.get_config("tushare_token", ""):
                    continue
                try:
                    df = fetcher(code, fetch_start, end, fq)
                    if df is not None and not df.empty:
                        return df, source, errors
                    errors.append("第%d轮 %s：返回空数据" % (attempt + 1, source))
                    if source == "ths" and has_local:
                        # 增量区间同花顺为空即该区间无行情（周末/停牌），
                        # 不再逐个源空跑，也不再重试。
                        return None, "ths", errors
                except Exception as exc:
                    had_error = True
                    errors.append("第%d轮 %s：%s" % (attempt + 1, source, exc))
            if not had_error:
                break  # 全部为空而非异常，重试没有意义
        return None, "", errors

    def record(index, total, fq, result):
        summary["total"] += 1
        if result["status"] == "ok":
            summary["ok"] += 1
        elif result["status"] in ("up_to_date", "no_new_data"):
            summary["no_change"] += 1
        else:
            summary["error"] += 1
            summary["errors"].append(result)
        if index % 100 == 0 or index == total or result["status"] == "error":
            print("[%s %d/%d] %s" % (fq, index, total, result), flush=True)
        if progress_cb:
            progress_cb(index, total, result["code"], fq, result)

    for fq in fq_list:
        if fq not in FQ_TYPES:
            raise ValueError("未知复权类型：%s" % fq)
        total = len(target_codes)
        tasks = {}
        finished = 0

        # 主线程先读 meta 决定每只是增量还是全量，避免线程内访问数据库。
        for code in target_codes:
            code = str(code).strip().zfill(6)
            meta = db.get_daily_meta(code, fq)
            has_local = bool(meta and meta.get("last_date"))
            full_replace = fq in ("qfq", "hfq") or not has_local
            if incremental and has_local:
                full_replace = False
            fetch_start = start
            if not full_replace:
                fetch_start = max(start, _as_date(meta["last_date"]) + timedelta(days=1))
            if fetch_start > end:
                finished += 1
                record(finished, total, fq,
                       {"code": code, "fq": fq, "status": "up_to_date", "rows": 0})
                continue
            tasks[code] = (meta, full_replace, fetch_start)

        if not tasks:
            continue
        print("[%s] 需联网 %d / %d 只，并发 %d" % (fq, len(tasks), total, workers), flush=True)

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {
                pool.submit(fetch_only, code, fq, task[2],
                            bool(task[0] and task[0].get("last_date"))): code
                for code, task in tasks.items()
            }
            for future in as_completed(futures):
                code = futures[future]
                meta, full_replace, fetch_start = tasks[code]
                finished += 1
                try:
                    df, source, errors = future.result()
                except Exception as exc:  # 线程自身异常也记为单票失败
                    df, source, errors = None, "", [str(exc)]
                if df is None or df.empty:
                    # 已有本地数据且接口为空，通常表示停牌/无新行情，不算失败。
                    if meta and meta.get("last_date"):
                        result = {"code": code, "fq": fq, "status": "no_new_data", "rows": 0}
                    else:
                        message = " | ".join(errors) or "接口返回空数据"
                        db.upsert_daily_meta(
                            code, fq, last_date="", rows=0, status="error",
                            source="", message=message)
                        result = {"code": code, "fq": fq, "status": "error",
                                  "rows": 0, "error": message}
                    record(finished, total, fq, result)
                    continue
                try:
                    rows_total = save_daily(code, fq, df, full_replace=full_replace,
                                            source=source)
                    result = {"code": code, "fq": fq, "status": "ok", "source": source,
                              "rows": len(df), "total_rows": rows_total}
                except Exception as exc:
                    result = {"code": code, "fq": fq, "status": "error",
                              "rows": 0, "error": str(exc)}
                record(finished, total, fq, result)

    db.set_config("last_all_ashare_update", {"time": db.now_str(), "summary": summary})
    print("更新完成：", summary)
    print(db.DISCLAIMER)
    return summary


# ---------------------------------------------------------------------------
# V1.1 Step F1：分钟 K 线。仅原始分钟落盘，前复权在读取时由本地日线因子计算。
# ---------------------------------------------------------------------------


def normalize_minute_period(period):
    """统一 5/5min 两种写法，并拒绝未声明的周期。"""
    text = str(period).strip().lower()
    if text.isdigit():
        text += "min"
    if text not in MINUTE_PERIODS:
        raise ValueError("分钟周期必须是 1/5/15/30/60min，收到：%s" % period)
    return text


def empty_minute():
    """返回结构统一的空分钟行情表。"""
    return pd.DataFrame(columns=MINUTE_COLUMNS)


def ensure_minute_db():
    """在不修改 V1.0 db.py 的前提下扩展分钟索引表与保留期配置。"""
    db.init_db()
    with db.get_conn() as conn:
        conn.executescript(MINUTE_SCHEMA_SQL)
    if db.get_config("minute_retention_days") is None:
        db.set_config("minute_retention_days", 30)


def minute_path(code, period):
    """分钟 parquet 精确路径：data/minute/{code}_{period}.parquet。"""
    return db.DATA_DIR / "minute" / ("%s_%s.parquet" % (code, normalize_minute_period(period)))


def get_minute_meta(code, period):
    ensure_minute_db()
    return db.query_one(
        "SELECT * FROM minute_meta WHERE code = ? AND period = ?",
        (code, normalize_minute_period(period)),
    )


def upsert_minute_meta(
    code, period, last_datetime="", rows=0, status="ok", source="", message=""
):
    """记录分钟更新状态；失败时由调用方保留原时间戳和行数。"""
    ensure_minute_db()
    db.execute(
        """
        INSERT INTO minute_meta(
            code, period, last_datetime, rows, status, source, message, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(code, period) DO UPDATE SET
            last_datetime = excluded.last_datetime,
            rows = excluded.rows,
            status = excluded.status,
            source = excluded.source,
            message = excluded.message,
            updated_at = excluded.updated_at
        """,
        (
            code, normalize_minute_period(period), last_datetime, int(rows),
            status, source, message, db.now_str(),
        ),
    )


def _as_minute_datetime(value, end_of_day=False):
    """日期/时间统一为 naive datetime；日期形式的结束时间取当日 23:59:59。"""
    if value is None:
        return datetime.now()
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value).strip()
    result = pd.to_datetime(text, errors="raise").to_pydatetime().replace(tzinfo=None)
    if end_of_day and " " not in text and "T" not in text:
        result = result.replace(hour=23, minute=59, second=59)
    return result


def _clean_minute(raw, code, period):
    """统一分钟字段、时间格式、排序与去重；period 仅用于校验。"""
    normalize_minute_period(period)
    if raw is None or raw.empty:
        return empty_minute()
    if "datetime" not in raw.columns:
        raise RuntimeError("分钟数据缺少 datetime 列，实际列为：%s" % list(raw.columns))
    df = raw.copy()
    parsed = pd.to_datetime(df["datetime"], errors="coerce")
    # 数据源若带时区，先去掉时区再按本地交易所时间保存。
    try:
        parsed = parsed.dt.tz_localize(None)
    except TypeError:
        pass
    df["datetime"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["code"] = code
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime")
    return df[MINUTE_COLUMNS].reset_index(drop=True)


def fetch_minute_akshare(code, period, start_datetime, end_datetime):
    """东方财富分钟接口；当前 akshare 使用 stock_zh_a_hist_min_em。"""
    import akshare as ak

    period = normalize_minute_period(period)
    func = getattr(ak, "stock_zh_a_hist_min_em", None)
    if func is None:
        raise RuntimeError("当前 akshare 版本无 stock_zh_a_hist_min_em 分钟接口")
    raw = func(
        symbol=code,
        start_date=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        period=MINUTE_API_PERIOD[period],
        adjust="",  # 只保存不复权原始分钟，复权由本地日线因子计算。
    )
    if raw is None or raw.empty:
        return empty_minute()
    rename = {
        "时间": "datetime", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount",
    }
    missing = [col for col in rename if col not in raw.columns]
    if missing:
        raise RuntimeError(
            "stock_zh_a_hist_min_em 返回列变化，缺少 %s；实际为 %s"
            % (missing, list(raw.columns))
        )
    df = raw.rename(columns=rename)
    # 东财成交量单位为手，与日线一致统一换算为股；成交额单位为元。
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100
    return _clean_minute(df, code, period)


# 分钟暂只有 akshare 适配器；保留字典入口，后续可独立扩展其他数据源。
MINUTE_FETCHERS = {"akshare": fetch_minute_akshare}


def load_local_minute(code, period, fq="none"):
    """
    离线读取分钟数据。none 直接读 parquet；qfq 复用本地日线因子重算 OHLC。

    任一交易日缺少 none/qfq 日线因子时明确报错，不静默混用不复权价格。
    """
    code = str(code).strip().zfill(6)
    period = normalize_minute_period(period)
    if fq not in ("none", "qfq"):
        raise ValueError("分钟 K 线当前仅支持 none/qfq，不支持：%s" % fq)
    path = minute_path(code, period)
    frame = _clean_minute(pd.read_parquet(path), code, period) if path.exists() else empty_minute()
    if frame.empty or fq == "none":
        return frame

    daily_none = load_local_daily(code, "none")
    daily_qfq = load_local_daily(code, "qfq")
    if daily_none.empty or daily_qfq.empty:
        raise RuntimeError("分钟前复权需要本地不复权和前复权日线，请先在数据管理更新两种日线")
    none_close = daily_none[["date", "close"]].rename(columns={"close": "none_close"})
    qfq_close = daily_qfq[["date", "close"]].rename(columns={"close": "qfq_close"})
    factors = none_close.merge(qfq_close, on="date", how="inner")
    factors = factors[pd.to_numeric(factors["none_close"], errors="coerce") != 0].copy()
    factors["factor"] = factors["qfq_close"] / factors["none_close"]
    factor_map = factors.set_index("date")["factor"]

    result = frame.copy()
    result["_date"] = result["datetime"].str.slice(0, 10)
    result["_factor"] = result["_date"].map(factor_map)
    if result["_factor"].isna().any():
        missing = sorted(result.loc[result["_factor"].isna(), "_date"].unique())
        raise RuntimeError(
            "分钟前复权缺少以下交易日的日线因子：%s；请更新该股票的不复权和前复权日线"
            % ", ".join(missing[:10])
        )
    for column in ["open", "high", "low", "close"]:
        result[column] = result[column] * result["_factor"]
    # 成交量与成交额不随价格复权，时间戳与原始行顺序也保持不变。
    return result[MINUTE_COLUMNS].reset_index(drop=True)


def _write_minute_parquet(path, frame):
    """先写临时文件再替换，避免写盘中断破坏已有分钟文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_minute(code, period, new_df, source="synthetic", retention_days=None):
    """合并增量分钟、按保留期清理并更新索引；空输入不会覆盖旧文件。"""
    code = str(code).strip().zfill(6)
    period = normalize_minute_period(period)
    ensure_minute_db()
    new_df = _clean_minute(new_df, code, period)
    if new_df.empty:
        return 0
    if retention_days is None:
        retention_days = int(db.get_config("minute_retention_days", 30))
    retention_days = int(retention_days)
    if retention_days <= 0:
        raise ValueError("分钟保留天数必须大于 0")

    path = minute_path(code, period)
    old = _clean_minute(pd.read_parquet(path), code, period) if path.exists() else empty_minute()
    combined = new_df if old.empty else pd.concat([old, new_df], ignore_index=True)
    final = _clean_minute(combined, code, period)
    cutoff = datetime.now() - timedelta(days=retention_days)
    final = final[pd.to_datetime(final["datetime"]) >= cutoff].reset_index(drop=True)
    if final.empty:
        return 0
    _write_minute_parquet(path, final)
    upsert_minute_meta(
        code, period, last_datetime=str(final["datetime"].max()), rows=len(final),
        status="ok", source=source, message="",
    )
    return len(final)


def cleanup_minute(code, period, retention_days=None, reference_datetime=None):
    """按保留天数清理单只分钟数据；reference_datetime 供离线自测指定基准。"""
    code = str(code).strip().zfill(6)
    period = normalize_minute_period(period)
    ensure_minute_db()
    if retention_days is None:
        retention_days = int(db.get_config("minute_retention_days", 30))
    retention_days = int(retention_days)
    if retention_days <= 0:
        raise ValueError("分钟保留天数必须大于 0")
    path = minute_path(code, period)
    if not path.exists():
        return 0
    frame = load_local_minute(code, period, "none")
    reference = _as_minute_datetime(reference_datetime) if reference_datetime else datetime.now()
    cutoff = reference - timedelta(days=retention_days)
    final = frame[pd.to_datetime(frame["datetime"]) >= cutoff].reset_index(drop=True)
    _write_minute_parquet(path, final)
    upsert_minute_meta(
        code, period,
        last_datetime=str(final["datetime"].max()) if not final.empty else "",
        rows=len(final), status="ok", source="local_cleanup", message="",
    )
    return len(final)


def update_minute(
    code, period="5min", start_date=None, end_date=None, retention_days=None, sources=None
):
    """按最后时间戳增量更新分钟；接口失败保留旧文件并写 minute_meta 与日志。"""
    code = str(code).strip().zfill(6)
    period = normalize_minute_period(period)
    ensure_minute_db()
    if retention_days is None:
        retention_days = int(db.get_config("minute_retention_days", 30))
    retention_days = int(retention_days)
    if retention_days <= 0:
        raise ValueError("分钟保留天数必须大于 0")
    end = _as_minute_datetime(end_date, end_of_day=True)
    # 首次也只请求保留期内的数据，避免 UI 沿用日线较早开始日期导致过量请求。
    default_start = end - timedelta(days=retention_days)
    start = _as_minute_datetime(start_date) if start_date is not None else default_start
    start = max(start, default_start)
    if start > end:
        raise ValueError("分钟开始时间不能晚于结束时间")

    meta = get_minute_meta(code, period) or {}
    local = load_local_minute(code, period, "none")
    local_last = str(local["datetime"].max()) if not local.empty else ""
    old_last = local_last or meta.get("last_datetime", "")
    old_rows = len(local)
    fetch_start = start
    if old_last:
        fetch_start = max(start, _as_minute_datetime(old_last) + timedelta(minutes=1))
    if fetch_start > end:
        return {
            "code": code, "period": period, "status": "up_to_date",
            "rows": 0, "total_rows": old_rows,
        }

    sources = sources or db.get_config("minute_data_source_order", ["akshare"])
    errors = []
    for source in sources:
        fetcher = MINUTE_FETCHERS.get(source)
        if fetcher is None:
            errors.append("%s：分钟数据源尚未接入" % source)
            continue
        try:
            frame = fetcher(code, period, fetch_start, end)
            if frame.empty:
                message = "接口返回空数据，未覆盖旧数据"
                upsert_minute_meta(
                    code, period, last_datetime=old_last, rows=old_rows,
                    status="ok", source=source, message=message,
                )
                return {"code": code, "period": period, "status": "no_new_data", "rows": 0}
            total = save_minute(
                code, period, frame, source=source, retention_days=retention_days
            )
            if total == 0:
                message = "返回数据均超出保留期，未覆盖旧数据"
                upsert_minute_meta(
                    code, period, last_datetime=old_last, rows=old_rows,
                    status="ok", source=source, message=message,
                )
                return {"code": code, "period": period, "status": "no_new_data", "rows": 0}
            result = {
                "code": code, "period": period, "status": "ok", "source": source,
                "rows": len(frame), "total_rows": total,
            }
            logger.log_data_update(code, period, "ok", message="分钟行情更新成功")
            return result
        except Exception as exc:
            errors.append("%s：%s" % (source, exc))

    # 当前无第二个分钟数据源时，降级结果是保留本地旧数据并明确展示错误。
    message = " | ".join(errors) or "分钟数据源不可用"
    upsert_minute_meta(
        code, period, last_datetime=old_last, rows=old_rows,
        status="error", source="", message=message,
    )
    logger.log_data_update(code, period, "error", message="分钟行情更新失败", error=message)
    return {"code": code, "period": period, "status": "error", "rows": 0, "error": message}


def update_all_ashare_minute(
    period="5min", start_date=None, end_date=None, codes=None, limit=None,
    retention_days=None, progress_cb=None,
):
    """上证、深证、创业板全部 A 股分钟批量更新；沿用日线汇总与单票失败不中断规则。"""
    ensure_minute_db()
    period = normalize_minute_period(period)
    if codes is None:
        constituents = get_all_ashare_constituents()
        with db.get_conn() as conn:
            conn.execute("UPDATE stock_info SET is_hs300 = 0")
        db.upsert_stock_info(constituents.to_dict("records"), hs300=False)
        target_codes = constituents["code"].tolist()
    else:
        target_codes = list(codes)
    if limit is not None:
        target_codes = target_codes[:int(limit)]
    summary = {"total": 0, "ok": 0, "no_change": 0, "error": 0, "errors": []}
    total = len(target_codes)
    for index, code in enumerate(target_codes, start=1):
        result = update_minute(
            code, period=period, start_date=start_date, end_date=end_date,
            retention_days=retention_days,
        )
        summary["total"] += 1
        if result["status"] == "ok":
            summary["ok"] += 1
        elif result["status"] in ("up_to_date", "no_new_data"):
            summary["no_change"] += 1
        else:
            summary["error"] += 1
            summary["errors"].append(result)
        print("[%s %d/%d] %s" % (period, index, total, result))
        if progress_cb:
            progress_cb(index, total, code, period, result)
        if index < total:
            time.sleep(float(db.get_config("request_interval", 0.2)))
    db.set_config(
        "last_all_ashare_minute_update",
        {"time": db.now_str(), "period": period, "summary": summary},
    )
    return summary


if __name__ == "__main__":
    # 快速自测只下载一只股票，避免首次运行就对全 A 发起数千次请求。
    # 需要 akshare、pandas、pyarrow：pip install akshare pandas pyarrow
    print(db.DISCLAIMER)
    db.init_db()
    test_result = update_one("600519", fq="qfq", start_date="20240101")
    print("单票更新结果：", test_result)
    local = load_local_daily("600519", "qfq")
    print("本地共 %d 行，最后 5 行：" % len(local))
    print(local.tail())

    # 全量更新示例（确认单票接口正常后再取消注释）：
    # update_all_ashare_daily(fq_list=("none", "qfq", "hfq"))
    # 只测试前 3 只股票：
    # update_all_ashare_daily(fq_list=("qfq",), limit=3)
