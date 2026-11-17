# -*- coding: utf-8 -*-
"""事件风控拦截层 —— NPU/CPU 推理输出之后、写入 CSV/看板之前执行。

架构位置：
    NPU 推理 → 原始仓位(position) → 【EventRiskFilter.apply()】
                                         │
                                         ├── 命中 BLACKLIST → 仓位强制置 0
                                         ├── 命中 WATCH     → 仓位 × 0.5
                                         └── 无命中         → 原仓位透传
                                         │
                                         ▼
                                    过滤后仓位 → 看板/CSV/日志

设计原则：
    - 只读不写模型：不修改 floor、不修改网络、不修改 1208 维因子。
    - 离线优先：先查本地 SQLite（stock_info 表 name 字段识别 ST）+
      本地公告缓存 CSV，网络拉取 akshare 作为可选增强。
    - 公告 T+1 生效：公告日 < 信号日 才匹配，盘后出的公告次日信号才生效，
      禁止当日回填（防未来函数）。
    - 不把公告文本塞进 PPO 输入，不影响训练。
"""

import csv
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# 常量
# ────────────────────────────────────────────────────────────────────

# 风险分级
BLACKLIST = "BLACKLIST"   # 硬拦截：仓位强制置 0，界面置灰
WATCH = "WATCH"           # 软预警：仓位 × 0.5，界面标黄
NORMAL = "NORMAL"         # 无命中：原仓位透传

# 本地公告缓存路径（相对项目根）
LOCAL_NOTICE_CSV = "data/parquet/announcements.csv"  # 可选，不存在就跳过
LOCAL_ST_CSV = "data/cache/st_list.csv"              # 可选

# 生效延迟：公告必须在信号日之前发布（T+1 规则）
NOTICE_EFFECTIVE_LAG_DAYS = 0  # 公告日 < 信号日 即生效；设为 1 则严格 T+1


class EventRiskFilter:
    """事件风控拦截器。

    用法：
        filter = EventRiskFilter(project_root=Path("."))
        risk_map = filter.load_daily_risk(trade_date="2026-11-13")
        filtered_picks = filter.apply(npu_raw_output, trade_date="2026-11-13")
        filter.save_outputs(filtered_picks, run_dir)
    """

    def __init__(self, project_root: Optional[Path] = None,
                 use_network: bool = False, keywords_path: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        :param project_root: 项目根目录，用于定位本地缓存 CSV、db 和关键词文件。
        :param use_network: 是否启用 akshare 网络拉取。默认 False（离线优先）。
        :param keywords_path: MVP 兼容参数，以其所在目录定位两个关键词文件。
        :param output_dir: risk_blacklist.csv 输出目录，默认 live/。
        """
        if project_root is not None:
            self.root = Path(project_root)
        elif output_dir is not None:
            # 独立测试仅传 output_dir 时，数据缓存也限定在临时目录
            self.root = Path(output_dir)
        else:
            self.root = Path(__file__).resolve().parent.parent
        self.use_network = use_network
        self.output_dir = Path(output_dir) if output_dir else self.root / "live"
        keyword_dir = Path(keywords_path).parent if keywords_path else self.root / "live"
        self.keywords_path = keywords_path or str(keyword_dir / "blacklist_hard.txt")
        self.hard_path = keyword_dir / "blacklist_hard.txt"
        self.soft_path = keyword_dir / "blacklist_soft.txt"

        # 分别加载硬拦截、软预警关键词；保留两组属性名以兼容压缩包接口
        self.hard_kw, self.soft_kw = self._load_keywords_split()
        self.blacklist_kw = self.hard_kw
        self.watch_kw = self.soft_kw

        # 缓存：避免同一 trade_date 重复拉取
        self._cache_date: Optional[str] = None
        self._cache_map: Dict[str, Tuple[str, str]] = {}

    # ────────────────────────────────────────────────────────────────
    # 关键词加载
    # ────────────────────────────────────────────────────────────────

    def _load_keywords_split(self) -> Tuple[List[str], List[str]]:
        """分别加载 blacklist_hard.txt 和 blacklist_soft.txt。

        空行与 # 开头的注释行跳过；不再使用 WATCH: 前缀区分等级。
        """
        def read_keywords(path: Path) -> List[str]:
            keywords = []
            if not path.exists():
                logger.warning("关键词文件不存在：%s", path)
                return keywords
            with open(path, "r", encoding="utf-8-sig") as handle:
                for raw in handle:
                    line = raw.strip()
                    if line and not line.startswith("#"):
                        keywords.append(line)
            return keywords

        hard_kw = read_keywords(self.hard_path)
        soft_kw = read_keywords(self.soft_path)
        logger.info("关键词加载：BLACKLIST=%d 个，WATCH=%d 个",
                    len(hard_kw), len(soft_kw))
        return hard_kw, soft_kw

    # ────────────────────────────────────────────────────────────────
    # 每日风险数据加载
    # ────────────────────────────────────────────────────────────────

    def load_daily_risk(self, trade_date: str) -> Dict[str, Tuple[str, str]]:
        """加载指定交易日的风险数据，返回 {code: (level, reason)}。

        数据源优先级：
            1. 本地 SQLite（stock_info 表 name 字段识别 ST）— 最快最稳；
            2. 本地公告缓存 CSV（data/parquet/announcements.csv）— 硬/软关键词分级扫描；
            3. 网络 akshare（stock_zh_a_st_em, stock_em_tfp）— 可选增强。

        结果按风险等级覆盖：BLACKLIST > WATCH > NORMAL。
        """
        # 命中缓存就直接返回
        if self._cache_date == trade_date and self._cache_map:
            return self._cache_map

        risk_map: Dict[str, Tuple[str, str]] = {}

        # ── 1. 本地 DB：ST/*ST 标记 ──────────────────────────────────
        self._load_st_from_db(risk_map)

        # ── 2. 本地 ST 缓存 CSV ──────────────────────────────────────
        self._load_st_from_csv(risk_map)

        # ── 3. 网络 ST / 停牌（可选，优先级高于公告） ─────────────────
        if self.use_network:
            self._load_st_from_akshare(risk_map)
            self._load_halt_from_akshare(risk_map, trade_date)

        # ── 4. 本地公告缓存（已 ST/停牌的标的不会被公告原因覆盖） ──────
        self._load_from_notice_cache(risk_map, trade_date)

        self._cache_date = trade_date
        self._cache_map = risk_map
        logger.info("[EventRiskFilter] trade_date=%s 风险标记：%d 只 BLACKLIST, %d 只 WATCH",
                     trade_date,
                     sum(1 for v in risk_map.values() if v[0] == BLACKLIST),
                     sum(1 for v in risk_map.values() if v[0] == WATCH))
        return risk_map

    # ── 各数据源实现 ────────────────────────────────────────────────

    def _load_st_from_db(self, risk_map: Dict[str, Tuple[str, str]]):
        """从本地 stock_info 识别 ST/*ST，列名保持项目现有 schema。

        stock_info 没有统一的 risk_warning 字段，因此直接读取证券名称；
        名称中包含 ST 即视为风险警示，避免离线运行时 ST 硬拦截失效。
        """
        try:
            import sys
            if str(self.root) not in sys.path:
                sys.path.insert(0, str(self.root))
            from core import db as local_db

            rows = local_db.query_all("SELECT code, name FROM stock_info")
            for row in rows:
                code = str(row.get("code", "")).strip().zfill(6)
                name = str(row.get("name", "")).strip()
                if code and "ST" in name.upper():
                    self._set_if_higher(risk_map, code, BLACKLIST,
                                        f"风险警示: {name}")
        except Exception as e:
            logger.debug("_load_st_from_db 跳过：%s", e)

    def _load_from_notice_cache(self, risk_map: Dict[str, Tuple[str, str]],
                                trade_date: str):
        """扫描本地公告缓存 CSV（如有），按关键词匹配，T+1 生效。

        CSV 期望列：
            code, notice_date, title, source
        """
        csv_path = self.root / LOCAL_NOTICE_CSV
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path, dtype={"code": str})
            # 公告必须在信号日之前发布（T+1 规则）
            df["notice_date"] = pd.to_datetime(df["notice_date"], errors="coerce")
            signal_dt = pd.to_datetime(trade_date)
            lag = pd.Timedelta(days=NOTICE_EFFECTIVE_LAG_DAYS)
            eligible = df[df["notice_date"] < signal_dt - lag].copy()

            for _, row in eligible.iterrows():
                code = str(row.get("code", "")).strip()
                title = str(row.get("title", ""))
                if not code or not title:
                    continue
                # 先扫 BLACKLIST（硬拦截优先级高）
                for kw in self.blacklist_kw:
                    if kw in title:
                        self._set_if_higher(
                            risk_map, code, BLACKLIST,
                            f"公告硬预警:{kw}（{title[:40]}）",
                        )
                        break
                else:
                    # BLACKLIST 没命中才扫 WATCH
                    for kw in self.watch_kw:
                        if kw in title:
                            self._set_if_higher(
                                risk_map, code, WATCH,
                                f"公告预警:{kw}（{title[:40]}）",
                            )
                            break
        except Exception as e:
            logger.warning("_load_from_notice_cache 异常：%s", e)

    def scan_notice_keywords(self, trade_date=None, target_codes=None):
        """网络公告分级扫描，返回 {code: {'hard': [...], 'soft': [...]}}。

        单票接口失败只记录日志，不影响其他标的；历史信号严格剔除信号日
        及之后发布的公告，避免把未来公告回填到历史仓位。
        """
        result = {}
        if not self.use_network or not target_codes:
            return result
        try:
            import akshare as ak
        except Exception as exc:
            logger.warning("akshare 导入失败，跳过网络公告扫描：%s", exc)
            return result

        signal_dt = pd.to_datetime(trade_date) if trade_date else None
        for raw_code in target_codes:
            code = str(raw_code).strip().zfill(6)
            try:
                if hasattr(ak, "stock_individual_notice_report"):
                    notice_df = ak.stock_individual_notice_report(symbol=code)
                else:
                    # 兼容压缩包所用的旧版 akshare 接口
                    notice_df = ak.stock_notice_report(symbol=code)
                if notice_df is None or len(notice_df) == 0:
                    continue

                # 先按公告日期过滤，再取最近 10 条，保证历史信号无未来函数
                eligible = notice_df
                if signal_dt is not None:
                    date_col = next((col for col in (
                        "公告日期", "日期", "公告时间", "发布时间"
                    ) if col in notice_df.columns), None)
                    if date_col:
                        notice_dates = pd.to_datetime(notice_df[date_col], errors="coerce")
                        eligible = notice_df[notice_dates < signal_dt]
                    elif signal_dt.date() < datetime.now().date():
                        logger.debug("%s 公告缺少日期列，历史信号跳过网络扫描", code)
                        continue

                hard_hits, soft_hits = [], []
                for _, notice_row in eligible.head(10).iterrows():
                    text = " ".join(str(v) for v in notice_row.values if pd.notna(v))
                    for kw in self.hard_kw:
                        if kw in text and kw not in hard_hits:
                            hard_hits.append(kw)
                    for kw in self.soft_kw:
                        if kw in text and kw not in soft_hits:
                            soft_hits.append(kw)
                if hard_hits or soft_hits:
                    result[code] = {"hard": hard_hits, "soft": soft_hits}
            except Exception as exc:
                logger.debug("%s 网络公告扫描失败：%s", code, exc)
        return result

    def _load_st_from_csv(self, risk_map: Dict[str, Tuple[str, str]]):
        """读本地 ST 缓存 CSV（如存在）。"""
        csv_path = self.root / LOCAL_ST_CSV
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path, dtype={"code": str})
            for _, row in df.iterrows():
                code = str(row.get("code", "")).strip()
                reason = row.get("reason", "风险警示")
                if code:
                    self._set_if_higher(risk_map, code, BLACKLIST,
                                        f"ST 缓存: {reason}")
        except Exception as e:
            logger.debug("_load_st_from_csv 跳过：%s", e)

    def _load_st_from_akshare(self, risk_map: Dict[str, Tuple[str, str]]):
        """网络拉取 akshare ST/*ST 列表。"""
        try:
            import akshare as ak
            st_df = ak.stock_zh_a_st_em()
            # akshare 返回列名：代码 / 名称
            code_col = "代码" if "代码" in st_df.columns else "证券代码"
            name_col = "名称" if "名称" in st_df.columns else "证券简称"
            for _, row in st_df.iterrows():
                code = str(row.get(code_col, "")).strip()
                name = str(row.get(name_col, "")).strip()
                if code:
                    self._set_if_higher(risk_map, code, BLACKLIST,
                                        f"ST 板: {name}")
        except Exception as e:
            logger.warning("akshare ST 拉取失败：%s", e)

    def _load_halt_from_akshare(self, risk_map: Dict[str, Tuple[str, str]],
                                trade_date: str):
        """网络拉取 akshare 停牌清单。"""
        try:
            import akshare as ak
            td = trade_date.replace("-", "")
            if hasattr(ak, "stock_tfp_em"):
                tfp_df = ak.stock_tfp_em(date=td)
            else:
                # 兼容旧版 akshare
                tfp_df = ak.stock_em_tfp(trade_date=td)
            if tfp_df is None or len(tfp_df) == 0:
                return
            if "停牌状态" in tfp_df.columns:
                halted = tfp_df[tfp_df["停牌状态"].astype(str).str.contains("停牌", na=False)]
            else:
                halted = tfp_df
            code_col = next((col for col in ("代码", "证券代码") if col in halted.columns), None)
            if not code_col:
                return
            for _, row in halted.iterrows():
                code = str(row.get(code_col, "")).strip().zfill(6)
                if code:
                    self._set_if_higher(risk_map, code, BLACKLIST, "停牌中")
        except Exception as e:
            logger.warning("akshare 停牌拉取失败：%s", e)

    @staticmethod
    def _set_if_higher(risk_map: Dict[str, Tuple[str, str]],
                       code: str, level: str, reason: str):
        """设置风险标记，但等级只升不降：BLACKLIST > WATCH > NORMAL。"""
        prev = risk_map.get(code)
        rank = {BLACKLIST: 3, WATCH: 2, NORMAL: 1}
        if prev is None or rank.get(level, 0) > rank.get(prev[0], 0):
            risk_map[code] = (level, reason)

    # ────────────────────────────────────────────────────────────────
    # 风险汇总：ST/停牌 > 公告硬关键词 > 公告软关键词 > 正常
    # ────────────────────────────────────────────────────────────────

    def load_st_blacklist(self):
        """加载 ST/*ST 列表，返回 {code: reason}，便于独立测试替换数据源。"""
        risk_map = {}
        self._load_st_from_db(risk_map)
        self._load_st_from_csv(risk_map)
        if self.use_network:
            self._load_st_from_akshare(risk_map)
        return {code: reason for code, (_, reason) in risk_map.items()}

    def load_halt_list(self, trade_date):
        """加载停牌列表，返回 {code: reason}。"""
        risk_map = {}
        if self.use_network:
            self._load_halt_from_akshare(risk_map, trade_date)
        return {code: reason for code, (_, reason) in risk_map.items()}

    def build_risk_map(self, trade_date: str, target_codes=None):
        """汇总风险源：ST/停牌 > 公告硬关键词 > 公告软关键词。"""
        risk_map = {}
        for raw_code, reason in self.load_st_blacklist().items():
            code = str(raw_code).strip().zfill(6)
            self._set_if_higher(risk_map, code, BLACKLIST, reason)
        for raw_code, reason in self.load_halt_list(trade_date).items():
            code = str(raw_code).strip().zfill(6)
            self._set_if_higher(risk_map, code, BLACKLIST, reason)

        # 本地公告缓存始终可用；网络公告由 scan_notice_keywords 内部开关控制
        self._load_from_notice_cache(risk_map, trade_date)
        if target_codes:
            notice_result = self.scan_notice_keywords(trade_date, target_codes)
            for raw_code, hits in notice_result.items():
                code = str(raw_code).strip().zfill(6)
                if risk_map.get(code, (None, ""))[0] == BLACKLIST:
                    continue
                if hits.get("hard"):
                    reasons = "; ".join(hits["hard"])
                    self._set_if_higher(risk_map, code, BLACKLIST,
                                        f"公告硬预警:{reasons}")
                elif hits.get("soft"):
                    reasons = "; ".join(hits["soft"])
                    self._set_if_higher(risk_map, code, WATCH,
                                        f"公告预警:{reasons}")
        self._cache_map = risk_map
        return risk_map

    # ────────────────────────────────────────────────────────────────
    # 主入口：apply
    # ────────────────────────────────────────────────────────────────

    def apply(self, raw_picks: List[dict], trade_date: str,
              target_codes: Optional[List[str]] = None) -> List[dict]:
        """对 NPU 推理原始输出应用事件风控。

        :param raw_picks: list[dict]，每个 dict 至少含 code, position, name(可选)。
        :param trade_date: 信号日期，格式 YYYY-MM-DD 或 YYYYMMDD。
        :param target_codes: 兼容 MVP 调用签名；本地缓存模式无需额外传参。
        :return: list[dict]，在原有字段基础上新增：
                 risk / risk_reason / filtered_position
                 并覆盖 position 为 filtered_position（供下游看板直接使用）。
        """
        if target_codes is None:
            target_codes = [row.get("code", "") for row in raw_picks]
        risk_map = self.build_risk_map(trade_date, target_codes)

        out: List[dict] = []
        for row in raw_picks:
            result = dict(row)
            code = str(row.get("code", "")).strip().zfill(6)
            raw_pos = float(row.get("position", 0.0))
            # 先保留原始仓位，再用过滤仓位覆盖 position，避免下游丢失原值
            result["position_raw"] = float(row.get("position_raw", raw_pos))

            if code in risk_map:
                level, reason = risk_map[code]
                result["risk"] = level
                result["risk_reason"] = reason
                if level == BLACKLIST:
                    result["filtered_position"] = 0.0
                elif level == WATCH:
                    result["filtered_position"] = round(raw_pos * 0.5, 6)
                else:
                    result["filtered_position"] = raw_pos
            else:
                result["risk"] = NORMAL
                result["risk_reason"] = ""
                result["filtered_position"] = raw_pos

            # 同时覆盖 position 字段，方便下游看板直接消费
            result["position"] = result["filtered_position"]
            out.append(result)

        # 统计日志
        blocks = sum(1 for r in out if r["risk"] == BLACKLIST)
        watches = sum(1 for r in out if r["risk"] == WATCH)
        if blocks or watches:
            logger.info("[EventRiskFilter] trade_date=%s: BLACKLIST=%d, WATCH=%d, 合计 %d 只受影响",
                         trade_date, blocks, watches, blocks + watches)
            for r in out:
                if r["risk"] != NORMAL:
                    logger.info("  → %s %s: %s (原仓位=%.4f → 风控后=%.4f)",
                                r.get("code"), r.get("name", ""), r["risk_reason"],
                                r.get("position_raw", 0.0),
                                r.get("filtered_position", 0.0))
        # MVP 约定：每次 apply 自动追加当日风险快照
        self._save_blacklist_csv(trade_date, risk_map)
        return out

    # ────────────────────────────────────────────────────────────────
    # 输出文件
    # ────────────────────────────────────────────────────────────────

    def _save_blacklist_csv(self, trade_date, risk_map):
        """向 output_dir/risk_blacklist.csv 追加完整风险映射。"""
        live_path = self.output_dir / "risk_blacklist.csv"
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_exists = live_path.exists()
        expires = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        with open(live_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if not live_exists:
                writer.writerow(["trade_date", "code", "risk_level", "reason", "expires_at"])
            for code, (level, reason) in risk_map.items():
                writer.writerow([trade_date, code, level, reason, expires])

    def save_outputs(self, filtered_picks: List[dict], run_dir: Path,
                     trade_date: str):
        """生成两个输出文件：

        - risk_blacklist.csv     当日风控结果（含所有标的，即使 NORMAL）
        - event_log.csv          全量追加（保留历史，方便回测审计）
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── risk_blacklist.csv ──────────────────────────────────────
        fields = ["trade_date", "code", "name", "risk", "risk_reason",
                  "source", "position_raw", "position_filtered"]
        with open(run_dir / "risk_blacklist.csv", "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in filtered_picks:
                writer.writerow({
                    "trade_date": trade_date,
                    "code": row.get("code", ""),
                    "name": row.get("name", ""),
                    "risk": row.get("risk", NORMAL),
                    "risk_reason": row.get("risk_reason", ""),
                    "source": "event_risk_filter_v1",
                    "position_raw": row.get("position_raw", row.get("position", "")),
                    "position_filtered": row.get("filtered_position", ""),
                })

        # live/risk_blacklist.csv 已由 apply() 自动追加，这里不重复写入

        # ── event_log.csv（追加，全历史） ────────────────────────────
        log_path = run_dir.parent / "event_log.csv"
        file_exists = log_path.exists()
        with open(log_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            for row in filtered_picks:
                if row.get("risk") != NORMAL:
                    writer.writerow({
                        "trade_date": trade_date,
                        "code": row.get("code", ""),
                        "name": row.get("name", ""),
                        "risk": row.get("risk", NORMAL),
                        "risk_reason": row.get("risk_reason", ""),
                        "source": "event_risk_filter_v1",
                        "position_raw": row.get("position_raw", row.get("position", "")),
                        "position_filtered": row.get("filtered_position", ""),
                    })

        logger.info("[EventRiskFilter] 输出：%s（%d 行）, %s（追加模式）",
                     run_dir / "risk_blacklist.csv", len(filtered_picks),
                     log_path)


# ────────────────────────────────────────────────────────────────────
# 快捷入口：命令行独立跑一次风控验证（不依赖 shadow_infer.py）
# ────────────────────────────────────────────────────────────────────

def main():
    """独立运行：读 run_latest/npu_daily_all.csv → 应用风控 → 写回。

    这是一个验证工具，方便在 shadow_infer.py 之外单独调试风控层。
    正式调度入口仍然是 shadow_infer.py --risk 参数。
    """
    import argparse
    parser = argparse.ArgumentParser(description="EventRiskFilter 独立运行")
    parser.add_argument("--run-dir", required=True, help="run_YYYYMMDD 目录")
    parser.add_argument("--csv", default=None, help="覆盖输入 CSV 路径（默认 run-dir/npu_daily_all.csv）")
    parser.add_argument("--trade-date", default=None, help="信号日期，默认 CSV 最大 date")
    parser.add_argument("--use-network", action="store_true", help="启用 akshare 网络拉取")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    input_csv = Path(args.csv) if args.csv else run_dir / "npu_daily_all.csv"

    if not input_csv.exists():
        SystemExit(f"输入 CSV 不存在：{input_csv}")

    df = pd.read_csv(input_csv, dtype={"code": str})
    trade_date = args.trade_date or str(df["date"].max())
    logger.info("加载：%s，trade_date=%s，%d 行", input_csv, trade_date, len(df))

    # 只取 trade_date 当日
    today = df[df["date"].astype(str) == str(trade_date)].copy()
    picks = []
    for _, row in today.iterrows():
        picks.append({
            "code": str(row["code"]),
            "name": row.get("name", ""),
            "position": float(row.get("position_npu", row.get("position", 0))),
            "action": row.get("action_npu", row.get("action", "")),
            "position_raw": float(row.get("position_npu", row.get("position", 0))),
        })

    rf = EventRiskFilter(use_network=args.use_network)
    filtered = rf.apply(picks, trade_date=trade_date)
    rf.save_outputs(filtered, run_dir, trade_date)

    print("\n=== 风控结果 ===")
    for r in filtered:
        flag = {"BLACKLIST": "🚫 BLOCK", "WATCH": "⚠️  WATCH", "NORMAL": "✅ NORMAL"}[r["risk"]]
        print(f"  {flag} {r['code']} {r['name']}: "
              f"原={r['position_raw']:.4f} → 风控后={r['filtered_position']:.4f}"
              f"  {r['risk_reason'][:50]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
