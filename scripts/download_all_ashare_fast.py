# -*- coding: utf-8 -*-
"""
全 A 股日线快速补全（同花顺并发下载，断点续传）。

- 股票范围：stock_info 中 60/00/30 开头代码，与“更新全部A股”一致；
- 已存在数据的股票只补最新缺口，缺失股票默认从 2024-01-01 起的按年接口下载；
- 下载在线程池并发，写库在单线程串行，避免 SQLite 写冲突；
- 失败股票自动用 baostock 兜底重试一次。

仅研究用途，不构成投资建议。
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db
from core import data_loader as dl

# akshare 的新浪请求默认不带超时，个别代码会永久挂起；统一补上超时。
_original_get = requests.get


def _get_with_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", 15)
    return _original_get(*args, **kwargs)


requests.get = _get_with_timeout

FQ = "qfq"
PREFIXES = ("60", "00", "30")


def _codes():
    rows = db.query_all("SELECT code FROM stock_info ORDER BY code")
    codes = [str(row["code"]).zfill(6) for row in rows]
    return [code for code in codes if code.startswith(PREFIXES)]


def _last_date(code):
    row = db.query_one(
        "SELECT MAX(date) AS d FROM daily_bars WHERE code = ? AND fq = ?", (code, FQ))
    return row["d"] if row and row["d"] else None


def _fetch(code, start, end, source):
    """高并发下接口可能返回空表，空结果或异常自动重试。"""
    for attempt in range(3):
        try:
            if source == "ths":
                frame = dl.fetch_daily_ths(code, start, end, FQ)
            elif source == "sina":
                frame = dl.fetch_daily_sina(code, start, end, FQ)
            else:
                frame = dl.fetch_daily_akshare(code, start, end, FQ)
            if frame is not None and not frame.empty:
                return frame
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
    return None


def main():
    parser = argparse.ArgumentParser(description="全 A 股日线快速补全")
    parser.add_argument("--workers", type=int, default=6, help="并发线程数")
    parser.add_argument("--source", default="ths",
                        choices=["ths", "sina", "akshare"], help="首选数据源")
    parser.add_argument("--start", default="", help="缺失股票起始日期")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 只，调试用")
    args = parser.parse_args()

    # 同花顺按年请求，起始越早越慢；2024 年起已覆盖 MA60 与回测所需区间。
    default_start = "2024-01-01" if args.source == "ths" else "2005-01-01"
    start_all = datetime.strptime(args.start or default_start, "%Y-%m-%d").date()
    if args.source == "sina" and args.workers > 1:
        # akshare 新浪接口需要初始化 V8，多线程并发可能直接让进程退出。
        print("新浪源不支持并发，已改为单线程", flush=True)
        args.workers = 1
    end = date.today()
    codes = _codes()
    global_last = db.query_one(
        "SELECT MAX(date) AS d FROM daily_bars WHERE fq = ?", (FQ,))["d"]
    latest_local = str(global_last)[:10] if global_last else ""
    tasks = []
    for code in codes:
        last = _last_date(code)
        if last and latest_local and str(last)[:10] >= latest_local:
            continue
        start = start_all
        if last:
            start = datetime.strptime(str(last)[:10], "%Y-%m-%d").date() + timedelta(days=1)
        if start <= end:
            tasks.append((code, start))
    if args.limit:
        tasks = tasks[:args.limit]
    print("待补全 %d / %d 只，数据源 %s，并发 %d，区间至 %s" % (
        len(tasks), len(codes), args.source, args.workers, end), flush=True)

    done = empty = failed = skipped = 0
    errors = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch, code, start, end, args.source): code
                   for code, start in tasks}
        for future in as_completed(futures):
            code = futures[future]
            try:
                frame = future.result()
                if frame is None or frame.empty:
                    empty += 1
                else:
                    db.init_db()
                    dl.save_daily(code, FQ, frame, False, args.source)
                    done += 1
            except Exception:
                try:
                    frame = dl.fetch_daily_baostock(code, start_all, end, FQ)
                    if frame is not None and not frame.empty:
                        dl.save_daily(code, FQ, frame, False, "baostock")
                        done += 1
                        skipped += 1
                    else:
                        empty += 1
                except Exception as exc:
                    failed += 1
                    errors.append((code, str(exc)[:120]))
            total = done + empty + failed
            if total % 100 == 0:
                elapsed = max(time.time() - started, 0.1)
                print("进度 %d/%d，成功 %d，空数据 %d，失败 %d，%.1f 只/分钟" % (
                    total, len(tasks), done, empty, failed, total / elapsed * 60),
                    flush=True)
    elapsed = max(time.time() - started, 0.1)
    print("完成：成功 %d，空数据 %d，失败 %d，baostock 兜底 %d，耗时 %.1f 分钟" % (
        done, empty, failed, skipped, elapsed / 60), flush=True)
    if errors:
        path = db.DATA_DIR / "logs" / "download_all_ashare_errors.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        pd.DataFrame(errors, columns=["code", "error"]).to_csv(
            path, index=False, encoding="utf-8-sig")
        print("失败明细：%s" % path, flush=True)


if __name__ == "__main__":
    main()
