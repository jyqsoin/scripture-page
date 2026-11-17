# -*- coding: utf-8 -*-
"""
A 股本地量化研究工具 V1.0 - PyQt5 桌面入口。

页面：数据管理、公式选股、K 线看图、回测。
只有数据管理页的两个“更新”按钮会联网，其余功能仅使用本地数据。
不启动本地服务，不连接券商，不下单。

仅研究用途，不构成投资建议。
"""

import base64
import io
import json
from html import escape
import math
import re
import subprocess
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import QDate, QProcess, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QDesktopServices, QIcon
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import db, formula_engine, hammer_boll, ma_break, pool, ydzq_watchlist
from core.backtest import BacktestConfig, run_backtest, run_parameter_grid
from core.factor_analysis import load_all_daily
from core.smart_beta import build_exposures, build_three_factor_exposures
from core.smart_beta_backtest import (
    SMART_BETA_DIR, run_smart_beta_backtest, save_reports,
)
from core.smart_beta_3factor_backtest import (
    run_three_factor_backtest, run_three_factor_comparison,
    save_comparison_reports,
)
from core.smart_beta_fundamental_backtest import (
    build_fundamental_composite, run_fundamental_comparison,
    save_fundamental_reports,
)
from core.smart_beta_summary_report import build_summary_report
from sim import (
    NAV_CSV, REPORT_HTML, TARGETS_CSV, TRADES_CSV,
    latest_trade_date, load_price_map,
)
from sim.account import holdings_frame, init_account, load_account
from sim.nav import benchmark_curve, run as run_sim_nav
from sim.paper_executor import run as run_sim_execution
from sim.report import build_report as build_sim_report
from sim.target_position import run as run_sim_targets
from sim.threshold_engine import (
    DRIFT_CSV, EMERGENCY_CSV, next_check_time, run as run_threshold_scan,
)
from sim.rolling import (
    ROLLING_NAV_CSV, ROLLING_PERIODS_CSV, ROLLING_THRESHOLDS_CSV,
    ROLLING_TRADES_CSV, build_report as build_rolling_report,
    run as run_rolling_sim,
)
from core.valuation_data import load_valuation
from core.optimize import run_out_of_sample_validation, heatmap_svg
from core.data_loader import (
    ensure_minute_db,
    load_local_daily,
    load_local_minute,
    update_all_ashare_daily,
    update_all_ashare_minute,
    update_minute,
    update_one,
)

DISCLAIMER = "仅研究用途，不构成投资建议"
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
FQ_OPTIONS = [("前复权", "qfq"), ("不复权", "none"), ("后复权", "hfq")]
MINUTE_FQ_OPTIONS = [("不复权", "none"), ("前复权", "qfq")]
PERIOD_OPTIONS = [
    ("日线", "daily"), ("1 分钟", "1min"), ("5 分钟", "5min"),
    ("15 分钟", "15min"), ("30 分钟", "30min"), ("60 分钟", "60min"),
]


def add_fq_options(combo):
    for label, value in FQ_OPTIONS:
        combo.addItem(label, value)


def add_period_options(combo):
    for label, value in PERIOD_OPTIONS:
        combo.addItem(label, value)


def make_date_edit(value):
    editor = QDateEdit(value)
    editor.setCalendarPopup(True)
    editor.setDisplayFormat("yyyy-MM-dd")
    return editor


def parse_codes(text):
    """支持逗号、分号、空白分隔，以及 sh600519/600519.SH 等写法。"""
    codes = []
    for token in re.split(r"[\s,，;；]+", str(text).strip()):
        if not token:
            continue
        digits = re.sub(r"\D", "", token)
        if len(digits) > 6:
            digits = digits[-6:]
        if len(digits) != 6:
            raise ValueError("股票代码格式不正确：%s" % token)
        codes.append(digits.zfill(6))
    return list(dict.fromkeys(codes))


def prepare_table(table, headers):
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


def set_cell(table, row, column, value, user_data=None, numeric=False):
    if value is None:
        text = ""
    elif isinstance(value, float) and math.isnan(value):
        text = ""
    else:
        text = str(value)
    item = QTableWidgetItem(text)
    if user_data is not None:
        item.setData(Qt.UserRole, user_data)
    if numeric:
        item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
    table.setItem(row, column, item)


def validate_date_range(start, end):
    if start > end:
        raise ValueError("开始日期不能晚于结束日期")


# 本期调仓清单（Smart Beta 导出 CSV）的固定列，顺序与导出文件一致。
WATCHLIST_COLUMNS = [
    "代码", "名称", "组合", "因子归属", "综合百分位和", "动量(%)", "年化波动(%)",
    "PE(TTM)", "PB(MRQ)", "权重(%)", "下期收益(%)", "调仓日", "收益截止日",
]


def _load_latest_watchlist(directory=None):
    """读取最新一份 watchlist_*.csv（按修改时间取最新），只读不写。

    返回 (路径, DataFrame)；目录下没有清单时返回 (None, None)。
    缺列时抛出 ValueError，供界面提示。
    """
    base = Path(directory) if directory else SMART_BETA_DIR
    paths = sorted(base.glob("watchlist_*.csv"), key=lambda p: p.stat().st_mtime)
    if not paths:
        return None, None
    path = paths[-1]
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"代码": str})
    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in WATCHLIST_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("清单缺少列：%s" % "、".join(missing))
    df = df[WATCHLIST_COLUMNS].copy()
    df["代码"] = df["代码"].astype(str).str.strip().str.zfill(6)
    return path, df


def _watchlist_summary(df):
    """汇总清单：行数、调仓日、收益截止日、权重合计与加权下期收益。"""
    info = {
        "rows": int(len(df)),
        "rebalance_date": "",
        "return_end": "",
        "weight_sum": 0.0,
        "weighted_return": None,
    }
    if df.empty:
        return info
    info["rebalance_date"] = str(df["调仓日"].iloc[0])
    info["return_end"] = str(df["收益截止日"].iloc[0])
    weight = pd.to_numeric(df["权重(%)"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(df["下期收益(%)"], errors="coerce").fillna(0.0)
    info["weight_sum"] = float(weight.sum())
    if float(weight.sum()) > 0:
        info["weighted_return"] = float((weight * ret).sum() / weight.sum())
    return info


def _load_strategy_metrics(directory=None):
    """读取最新一份 *_four_reconciliation.csv，取五因子/四因子主基线全期指标。

    返回 {"five_always": {...}, "four_old": {...}}；文件缺失或列不全时返回 {}。
    只读取已有核对结果，不重跑回测。
    """
    base = Path(directory) if directory else SMART_BETA_DIR
    paths = sorted(base.glob("*_four_reconciliation.csv"), key=lambda p: p.stat().st_mtime)
    if not paths:
        return {}
    df = pd.read_csv(paths[-1], encoding="utf-8-sig")
    if "portfolio" not in df.columns:
        return {}
    metrics = {}
    for key in ("five_always", "four_old"):
        rows = df[df["portfolio"] == key]
        if not rows.empty:
            metrics[key] = rows.iloc[0].to_dict()
    return metrics


class WorkerThread(QThread):
    """单一后台任务；UI 控件值在启动前捕获，线程内不访问控件。"""

    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(object)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            result = self.function(self.progress)
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.result_ready.emit(result)


class DataPage(QWidget):
    @staticmethod
    def default_end_date():
        """不联网估算最近交易日：盘中取上一工作日，收盘后取当日。"""
        now = datetime.now()
        candidate = now.date()
        if now.weekday() >= 5 or now.hour < 15:
            candidate -= timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
        # 法定节假日不依赖网络日历，用户仍可在日期框中手动调整。
        return QDate(candidate.year, candidate.month, candidate.day)

    def __init__(self, window):
        super().__init__()
        self.window = window
        layout = QVBoxLayout(self)

        update_group = QGroupBox("手动更新行情（只有以下两个更新按钮会联网）")
        grid = QGridLayout(update_group)
        self.codes_edit = QPlainTextEdit()
        self.codes_edit.setPlaceholderText("输入股票代码，例如：600519, 000001；支持空格或换行分隔")
        self.codes_edit.setFixedHeight(64)
        self.period_combo = QComboBox()
        add_period_options(self.period_combo)
        self.period_combo.currentIndexChanged.connect(self.period_changed)
        self.fq_combo = QComboBox()
        add_fq_options(self.fq_combo)
        self.fq_combo.setToolTip("日线前/后复权全量覆盖；分钟只拉取原始数据，在看图页选择复权")
        self.retention_spin = QSpinBox()
        self.retention_spin.setRange(1, 365)
        self.retention_spin.setValue(int(db.get_config("minute_retention_days", 30)))
        self.retention_spin.setSuffix(" 天")
        self.retention_spin.setEnabled(False)
        self.retention_spin.setToolTip("分钟数据本地保留时长，更新时自动清理更早数据")
        self.start_edit = make_date_edit(QDate(2020, 1, 1))
        self.end_edit = make_date_edit(self.default_end_date())
        self.end_edit.setToolTip("默认取最近交易日；法定节假日可手动调整，不会默认到未来日期")
        self.update_codes_button = QPushButton("更新输入股票")
        self.update_all_button = QPushButton("更新全部A股")
        self.update_all_button.setToolTip("上证、深证、创业板全部股票（不含科创板），同花顺并发+增量更新，已有数据只补缺口")
        self.update_codes_button.clicked.connect(self.update_codes)
        self.update_all_button.clicked.connect(self.update_all_ashare)
        grid.addWidget(QLabel("股票代码"), 0, 0)
        grid.addWidget(self.codes_edit, 0, 1, 1, 5)
        grid.addWidget(QLabel("周期"), 1, 0)
        grid.addWidget(self.period_combo, 1, 1)
        grid.addWidget(QLabel("复权口径"), 1, 2)
        grid.addWidget(self.fq_combo, 1, 3)
        grid.addWidget(QLabel("分钟保留"), 1, 4)
        grid.addWidget(self.retention_spin, 1, 5)
        grid.addWidget(QLabel("开始日期"), 2, 0)
        grid.addWidget(self.start_edit, 2, 1)
        grid.addWidget(QLabel("结束日期"), 2, 2)
        grid.addWidget(self.end_edit, 2, 3)
        grid.addWidget(self.update_codes_button, 3, 1)
        grid.addWidget(self.update_all_button, 3, 2)
        note = QLabel(
            "日线规则保持不变；分钟按最后时间戳增量，首次仅拉保留期内数据，仅用于 K 线看图。"
            "接口失败会记录原因且不会清空旧数据。"
        )
        note.setWordWrap(True)
        grid.addWidget(note, 3, 3, 1, 3)
        layout.addWidget(update_group)

        status_header = QHBoxLayout()
        status_header.addWidget(QLabel("本地行情状态（双击记录可查看 K 线）"))
        status_header.addStretch()
        self.refresh_button = QPushButton("刷新本地状态")
        self.refresh_button.clicked.connect(self.refresh_status)
        status_header.addWidget(self.refresh_button)
        layout.addLayout(status_header)

        self.status_table = QTableWidget()
        self.status_headers = [
            "code", "period", "fq", "last_date", "rows", "status", "source", "updated_at", "message"
        ]
        prepare_table(self.status_table, self.status_headers)
        # 所有列均可拖拽调宽；错误信息换行，并在列宽变化后重新计算行高。
        header = self.status_table.horizontalHeader()
        column_widths = {
            "code": 90, "period": 70, "fq": 65, "last_date": 145, "rows": 70,
            "status": 75, "source": 100, "updated_at": 150, "message": 360,
        }
        for column, name in enumerate(self.status_headers):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            self.status_table.setColumnWidth(column, column_widths[name])
        header.setStretchLastSection(True)
        header.sectionResized.connect(lambda *_: self.status_table.resizeRowsToContents())
        self.status_table.setWordWrap(True)
        self.status_table.setTextElideMode(Qt.ElideNone)
        self.status_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.status_table.cellDoubleClicked.connect(self.open_kline_from_status)

        self.status_empty_label = QLabel("暂无本地数据，请先点击【更新全部A股】或输入代码更新")
        self.status_empty_label.setAlignment(Qt.AlignCenter)
        self.status_empty_label.setStyleSheet("color: #777; padding: 24px;")
        self.status_stack = QStackedWidget()
        self.status_stack.addWidget(self.status_table)
        self.status_stack.addWidget(self.status_empty_label)
        self.status_stack.setCurrentWidget(self.status_empty_label)
        layout.addWidget(self.status_stack)

    def period_changed(self, _index=None):
        """分钟更新只拉原始数据，复权选择移到 K 线看图页。"""
        is_daily = self.period_combo.currentData() == "daily"
        self.fq_combo.setEnabled(is_daily)
        self.retention_spin.setEnabled(not is_daily)

    def set_update_pending(self, pending):
        """点击更新后立即给出忙碌状态；任务结束仍由主窗口统一恢复。"""
        worker_running = self.window.worker is not None and self.window.worker.isRunning()
        self.update_codes_button.setEnabled(not pending and not worker_running)
        self.update_all_button.setEnabled(not pending and not worker_running)
        if pending:
            self.window.progress_bar.setRange(0, 0)
            self.window.progress_bar.show()
        elif not worker_running:
            self.window.progress_bar.hide()

    def update_codes(self):
        self.set_update_pending(True)
        try:
            codes = parse_codes(self.codes_edit.toPlainText())
            if not codes:
                raise ValueError("请先输入至少一个股票代码")
            start = self.start_edit.date().toString("yyyy-MM-dd")
            end = self.end_edit.date().toString("yyyy-MM-dd")
            validate_date_range(start, end)
        except ValueError as exc:
            self.set_update_pending(False)
            QMessageBox.warning(self, "无法更新", str(exc))
            return
        fq = self.fq_combo.currentData()
        period = self.period_combo.currentData()
        retention_days = self.retention_spin.value()
        if period != "daily":
            db.set_config("minute_retention_days", retention_days)

        def task(progress):
            summary = {"total": 0, "ok": 0, "no_change": 0, "error": 0, "errors": []}
            for index, code in enumerate(codes, start=1):
                progress.emit({"current": index - 1, "total": len(codes), "text": "正在更新 " + code})
                if period == "daily":
                    result = update_one(code, fq=fq, start_date=start, end_date=end)
                else:
                    result = update_minute(
                        code, period=period, start_date=start, end_date=end,
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
                progress.emit({"current": index, "total": len(codes), "text": code + "：" + result["status"]})
                if index < len(codes):
                    time.sleep(float(db.get_config("request_interval", 0.2)))
            return summary

        self.window.start_task(task, self.update_finished, "正在手动更新输入股票")

    def update_all_ashare(self):
        self.set_update_pending(True)
        start = self.start_edit.date().toString("yyyy-MM-dd")
        end = self.end_edit.date().toString("yyyy-MM-dd")
        try:
            validate_date_range(start, end)
        except ValueError as exc:
            self.set_update_pending(False)
            QMessageBox.warning(self, "无法更新", str(exc))
            return
        fq = self.fq_combo.currentData()
        period = self.period_combo.currentData()
        period_name = self.period_combo.currentText()
        retention_days = self.retention_spin.value()
        answer = QMessageBox.question(
            self,
            "确认联网更新",
            "将联网获取上证、深证、创业板全部股票（不含科创板，约 4600 只），并更新其%s。\n"
            "开始日期：%s\n结束日期：%s\n\n同花顺并发下载，已有数据只补缺口，"
            "单票失败会继续，是否开始？"
            % (period_name, start, end),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.set_update_pending(False)
            return

        def task(progress):
            def callback(current, total, code, current_period, result):
                progress.emit({
                    "current": current,
                    "total": total,
                    "text": "%s %s：%s" % (current_period, code, result["status"]),
                })

            if period == "daily":
                return update_all_ashare_daily(
                    fq_list=(fq,), start_date=start, end_date=end, progress_cb=callback
                )
            db.set_config("minute_retention_days", retention_days)
            return update_all_ashare_minute(
                period=period, start_date=start, end_date=end,
                retention_days=retention_days, progress_cb=callback,
            )

        self.window.start_task(task, self.update_finished, "正在更新全部A股行情")

    def update_finished(self, summary):
        self.refresh_status()
        message = "更新结束：成功 %d，无需更新 %d，失败 %d。" % (
            summary["ok"], summary["no_change"], summary["error"]
        )
        self.window.statusBar().showMessage(message, 10000)
        if summary["errors"]:
            box = QMessageBox(QMessageBox.Warning, "部分行情更新失败", message, parent=self)
            box.setDetailedText(json.dumps(summary["errors"], ensure_ascii=False, indent=2))
            box.exec_()
        else:
            QMessageBox.information(self, "更新完成", message)

    def refresh_status(self):
        try:
            ensure_minute_db()
            rows = db.query_all(
                """
                SELECT code, 'daily' AS period, fq, last_date, rows, status, source, updated_at, message
                FROM daily_meta
                UNION ALL
                SELECT code, period, 'none' AS fq, last_datetime AS last_date,
                       rows, status, source, updated_at, message
                FROM minute_meta
                ORDER BY updated_at DESC, code LIMIT 1000
                """
            )
            self.status_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                for column, key in enumerate(self.status_headers):
                    set_cell(
                        self.status_table, row_index, column, row.get(key),
                        user_data=row.get("code") if column == 0 else None,
                        numeric=key == "rows",
                    )
                    if key == "message":
                        # 单元格完整换行显示，同时保留完整悬浮提示便于复制核对。
                        item = self.status_table.item(row_index, column)
                        item.setToolTip(item.text())
            self.status_stack.setCurrentWidget(self.status_table if rows else self.status_empty_label)
            if rows:
                self.status_table.resizeRowsToContents()
        except Exception as exc:
            self.window.show_error("读取本地行情状态失败", traceback.format_exc(), str(exc))

    def open_kline_from_status(self, row, _column):
        code_item = self.status_table.item(row, self.status_headers.index("code"))
        period_item = self.status_table.item(row, self.status_headers.index("period"))
        fq_item = self.status_table.item(row, self.status_headers.index("fq"))
        if code_item and period_item and fq_item:
            code = code_item.data(Qt.UserRole) or code_item.text()
            self.window.show_kline(code, fq_item.text(), period_item.text())


class ShadowSelectPage(QWidget):
    """影子推理选股：读取 NPU 批量推理产物，展示最新交易日 NPU 给出目标仓位的标的。

    锁版冻结状态保持不变：不重训、不调参、不改 floor、不下单、不接券商。
    """

    def __init__(self, window):
        super().__init__()
        self.window = window
        self._latest_rows = []  # 最新交易日 position >= 0.01 的标的
        self._latest_date = ""
        layout = QVBoxLayout(self)

        # 顶部信息栏
        top_bar = QHBoxLayout()
        top_bar.setSpacing(16)
        self.date_label = QLabel("信号日期：—")
        self.date_label.setStyleSheet("font-weight: bold; color:#1f3a5f;")
        self.precision_label = QLabel("精度：FP16")
        self.precision_label.setStyleSheet("color:#1f7a3a;")
        self.device_label = QLabel("推理：NPU 盘后批量 ✓")
        self.device_label.setStyleSheet("color:#1f7a3a;")
        self.floor_label = QLabel("锁版 floor：0.20 / 0.20 / 0.25")
        self.floor_label.setStyleSheet("color:#555;")
        self.schedule_label = QLabel("自动刷新：每日 16:30")
        self.schedule_label.setStyleSheet("color:#555;")
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.setToolTip("重新读取 NPU 影子推理产物")
        self.refresh_button.clicked.connect(self.refresh)
        top_bar.addWidget(self.date_label)
        top_bar.addWidget(self.precision_label)
        top_bar.addWidget(self.device_label)
        top_bar.addWidget(self.floor_label)
        top_bar.addWidget(self.schedule_label)
        top_bar.addStretch(1)
        top_bar.addWidget(self.refresh_button)
        layout.addLayout(top_bar)

        # 表格
        self.table = QTableWidget()
        # 9 列：选股信息 + 均线破位标记 + 一致性 + 风控状态/原因
        self.headers = ["代码", "名称", "仓位", "信号", "均线破位",
                        "捕获率", "FP16误差", "风控状态", "风控原因"]
        self.header_tooltips = [
            "股票代码（6 位数字，本地 stock_info 表补全名称）",
            "股票名称（本地缓存，不发起网络请求）",
            "风控后的有效目标仓位（BLACKLIST 拦截置 0，WATCH 预警减半）",
            "由最新日仓位与前一日仓位差分推导：新开仓 / 加仓 / 减仓 / 清仓 / 持有",
            "MA5/10/20/60 中最低一根即 MIN_MA；开盘价与收盘价同时低于 MIN_MA 判为破位↓",
            "回测窗口内上涨日的收益捕获率（取自 shadow_summary.csv）",
            "FP16 与参考实现的仓位误差（阈值 1e-3），悬停可看 NAV 相对误差",
            "事件风控层分级：正常 / 预警(WATCH, 仓位×0.5) / 拦截(BLACKLIST, 仓位=0)",
            "风控命中的具体原因（鼠标悬停查看明细）",
        ]
        prepare_table(self.table, self.headers)
        self._setup_header_tooltips()
        self._setup_column_resize_modes()
        self.table.cellDoubleClicked.connect(self._on_double_clicked)
        layout.addWidget(self.table, 1)

        # 操作按钮栏
        action_bar = QHBoxLayout()
        self.kline_button = QPushButton("查看选中 K 线")
        self.kline_button.setEnabled(False)
        self.kline_button.clicked.connect(self._open_selected_kline)
        self.export_button = QPushButton("导出当前列表 CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        action_bar.addWidget(self.kline_button)
        action_bar.addWidget(self.export_button)
        action_bar.addStretch(1)
        layout.addLayout(action_bar)

        # 底部告警区（复用影子推理页逻辑）
        alert_group = QGroupBox("影子推理监控告警")
        alert_layout = QVBoxLayout(alert_group)
        self.alert_label = QLabel("加载中")
        self.alert_label.setWordWrap(True)
        alert_layout.addWidget(self.alert_label)
        layout.addWidget(alert_group)

        # 启动时读一次
        QTimer.singleShot(200, self.refresh)

    # -------- 表格初始化辅助 --------

    def _setup_header_tooltips(self):
        """为每列表头设置鼠标悬停提示，解释字段含义和数据来源。"""
        for col, tip in enumerate(self.header_tooltips):
            item = self.table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)

    def _setup_column_resize_modes(self):
        """列宽策略：前 7 列按内容自适应，风控状态/原因两列 Stretch。"""
        header = self.table.horizontalHeader()
        # 前 7 列：代码、名称、仓位、信号、均线破位、捕获率、FP16误差 — 紧凑
        for col in range(7):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        # 风控状态（col 7）+ 风控原因（col 8）：Stretch 分摊剩余空间
        for col in range(7, len(self.headers)):
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        header.setStretchLastSection(True)

    # -------- 数据加载 --------

    @staticmethod
    def _find_latest_run():
        """查找最新影子推理 CSV，依次尝试：

        1. npu_output/run_latest/npu_daily_all.csv   — shadow_infer 同步的风控标注产物
        2. shadow_output/run_latest/shadow_daily_all.csv
        3. 两个目录下日期最大的 run_YYYYMMDD
        """
        root = Path(__file__).resolve().parent
        candidates = [
            (root / "npu_output", "npu_daily_all.csv"),
            (root / "shadow_output", "shadow_daily_all.csv"),
        ]
        for base, fname in candidates:
            if not base.exists():
                continue
            # run_latest 软链接优先
            latest = base / "run_latest"
            if latest.exists():
                csv_path = latest / fname
                if csv_path.exists():
                    return csv_path
            # 否则日期最大的 run_*
            runs = sorted([p for p in base.iterdir()
                           if p.is_dir() and p.name.startswith("run_") and p.name != "run_latest"],
                          key=lambda p: p.name, reverse=True)
            for run in runs:
                csv_path = run / fname
                if csv_path.exists():
                    return csv_path
        return None

    @staticmethod
    def _load_name_map():
        try:
            from . import db as _db
            return {row["code"]: row["name"] for row in
                    _db.query_all("SELECT code, name FROM stock_info")}
        except Exception:
            return {}

    @staticmethod
    def _ma_break_info(code, as_of):
        """计算单只标的在 as_of 日的均线破位情况。

        口径与「均线破位选股」公式一致：MA5/10/20/60 中数值最小的即 MIN_MA，
        当日开盘价与收盘价同时低于 MIN_MA 视为破位。数据不足时返回 None。
        """
        try:
            from core.data_loader import load_local_daily

            frame = load_local_daily(code)
            if frame is None or frame.empty:
                return None
            frame = frame[frame["date"] <= str(as_of)]
            periods = sorted(ma_break.DEFAULT_PERIODS.values())
            if len(frame) < max(periods):
                return None
            closes = frame["close"].astype(float)
            means = {period: float(closes.tail(period).mean()) for period in periods}
            min_period = min(means, key=means.get)
            min_ma = means[min_period]
            last = frame.iloc[-1]
            close = float(last["close"])
            open_price = float(last["open"])
            broken = open_price < min_ma and close < min_ma
            deviation = (close / min_ma - 1) * 100 if min_ma else 0.0
            if broken:
                tip = ("跌破%d日均线 %.2f（收盘 %.2f / 开盘 %.2f，偏离 %.2f%%）"
                       % (min_period, min_ma, close, open_price, deviation))
            else:
                tip = ("未破位：最低为 MA%d=%.2f（收盘 %.2f / 开盘 %.2f，偏离 %+.2f%%）"
                       % (min_period, min_ma, close, open_price, deviation))
            return {"broken": broken, "period": min_period, "min_ma": min_ma,
                    "close": close, "open": open_price,
                    "deviation": deviation, "tip": tip}
        except Exception:
            return None

    def _ma_break_map(self, codes, as_of):
        """批量计算均线破位标记，同一代码只读一次本地日线。"""
        result = {}
        for code in set(codes):
            result[code] = self._ma_break_info(code, as_of)
        return result

    def _build_funnel_text(self, rows):
        """空态漏斗：初选 → 事件风控剔除 → 均线破位 → 最终信号。"""
        first = [r for r in rows if r.get("raw_position", 0.0) >= 0.01]
        blocked = [r for r in first if r.get("risk") == "BLACKLIST"]
        survived = [r for r in first if r.get("risk") != "BLACKLIST"]
        broken = [r for r in survived if (r.get("ma_break") or {}).get("broken")]
        return (
            "今日 NPU 未选出标的（所有标的目标仓位 < 1%%）\n"
            "漏斗：初选池 %d 只 → 事件风控剔除 %d 只 → 剩余 %d 只（其中均线破位 %d 只）"
            " → 最终信号 0 只"
            % (len(first), len(blocked), len(survived), len(broken))
        )

    def refresh(self):
        try:
            csv_path = self._find_latest_run()
            if csv_path is None:
                self._show_empty("影子数据尚未生成，请先运行 run_shadow_daily.bat 或任务计划")
                return
            try:
                frame = pd.read_csv(csv_path, dtype={"code": str})
            except Exception as exc:
                self._show_empty("影子数据异常（CSV 损坏）：%s" % exc)
                return

            # 宽松字段校验：必须有 code + date，至少有一个 position 列
            cols = set(frame.columns)
            if "code" not in cols or "date" not in cols:
                self._show_empty("影子数据异常（缺少 code/date 列）")
                return

            # 两套 schema 自动适配：
            #   shadow_output: position / position_err / nav_rel_err
            #   npu_output:    position_npu / position_diff / nav_rel_diff
            pos_col = "position_npu" if "position_npu" in cols else "position"
            if pos_col not in cols:
                self._show_empty("影子数据异常（缺少 position 列）")
                return
            pos_err_col = "position_diff" if "position_diff" in cols \
                else ("position_err" if "position_err" in cols else None)
            nav_err_col = "nav_rel_diff" if "nav_rel_diff" in cols \
                else ("nav_rel_err" if "nav_rel_err" in cols
                      else ("nav_err" if "nav_err" in cols else None))

            # 去重：按 (date, code) 保留最后一行
            frame = frame.drop_duplicates(subset=["date", "code"], keep="last").copy()
            latest_date = str(frame["date"].max())
            latest = frame[frame["date"] == latest_date].copy()
            latest = latest.sort_values(pos_col, ascending=False)

            self._latest_date = latest_date
            self.date_label.setText("信号日期：%s（来自 %s）" % (latest_date, csv_path.parent.name))

            name_map = self._load_name_map()
            threshold_pos = 1e-3
            threshold_nav = 1e-5

            # 捕获率：优先同目录 shadow_summary.csv，兜底 npu_summary.csv
            capture_map = {}
            summary_path = csv_path.parent / "shadow_summary.csv"
            if not summary_path.exists():
                summary_path = csv_path.parent / "npu_summary.csv"
            if summary_path.exists():
                try:
                    summary_df = pd.read_csv(summary_path, dtype={"code": str})
                    if "code" in summary_df.columns and "uptrend_capture" in summary_df.columns:
                        for _, srow in summary_df.iterrows():
                            try:
                                capture_map[str(srow["code"])] = float(srow["uptrend_capture"])
                            except Exception:
                                continue
                except Exception:
                    capture_map = {}

            # 风控字段（shadow_infer.py 的 EventRiskFilter 产出）；不存在则 fallback NORMAL
            has_risk = "risk" in latest.columns
            has_filtered = "filtered_position" in latest.columns

            # 构建每只股票前一个交易日的仓位（用于推导信号方向）
            pos_col = "position_npu" if "position_npu" in frame.columns else "position"
            prev_pos_map = {}
            codes_all = latest["code"].astype(str).unique()
            for code in codes_all:
                code_hist = frame[frame["code"].astype(str) == code].sort_values("date")
                earlier = code_hist[code_hist["date"] < latest_date]
                if not earlier.empty:
                    prev_pos_map[code] = float(earlier.iloc[-1][pos_col])
                else:
                    prev_pos_map[code] = 0.0

            rows = []
            for _, row in latest.iterrows():
                code = str(row["code"])
                # 原始 NPU 仓位（风控前）——用于信号推导
                raw_pos = float(row[pos_col])
                # 风控后有效仓位——用于展示和筛选；若无风控列则 fallback 到 raw_pos
                filtered_pos = (float(row["filtered_position"]) if has_filtered else raw_pos)
                # 风控分级
                risk_level = str(row.get("risk", "NORMAL") or "NORMAL").strip() if has_risk else "NORMAL"
                risk_reason = str(row.get("risk_reason", "") or "").strip() if has_risk else ""

                # FP16 一致性字段（全量标的都算，用于告警）—— 用提前算出的列名
                pos_err = float(row[pos_err_col]) if pos_err_col else 0.0
                nav_err = float(row[nav_err_col]) if nav_err_col else 0.0
                pos_exceed = pos_err >= threshold_pos
                nav_exceed = nav_err >= threshold_nav

                # 推导信号：对比前一日原始仓位变化（信号是策略动作，不受风控后仓位影响）
                prev_pos = prev_pos_map.get(code, 0.0)
                if prev_pos == 0.0 and raw_pos > 0:
                    signal = "新开仓"
                elif prev_pos > 0 and raw_pos > prev_pos + 0.005:
                    signal = "加仓"
                elif prev_pos > raw_pos + 0.005 and raw_pos > 0:
                    signal = "减仓"
                elif prev_pos > 0 and raw_pos == 0.0:
                    signal = "清仓"
                else:
                    signal = "持有"

                # 被硬拦截但原始仓位达到展示阈值的标的仍保留，确保能看见“拦截”状态
                hidden = filtered_pos < 0.01 and not (
                    risk_level == "BLACKLIST" and raw_pos >= 0.01
                )
                name = name_map.get(code, "")

                rows.append({
                    "code": code,
                    "name": name,
                    "date": latest_date,
                    "position": filtered_pos,    # 风控后仓位（展示用）
                    "raw_position": raw_pos,     # 原始仓位（参考）
                    "signal": signal,
                    "capture": capture_map.get(code),  # 上涨日捕获率
                    "risk": risk_level,
                    "risk_reason": risk_reason,
                    "pos_err": pos_err,
                    "nav_err": nav_err,
                    "pos_exceed": pos_exceed,
                    "nav_exceed": nav_exceed,
                    "_hidden": hidden,
                })

            # 均线破位标记：仅对候选标的读取本地日线计算，缺数据时置 None 不报错
            ma_map = self._ma_break_map([r["code"] for r in rows], latest_date)
            for r in rows:
                r["ma_break"] = ma_map.get(r["code"])

            # 只展示 position >= 0.01 的标的；但 rows 保留全量用于告警
            display_rows = [r for r in rows if not r.get("_hidden", False)]
            self._latest_rows = display_rows
            if not display_rows:
                # 空态不再只给一句话，改为横向漏斗，让空跑也有诊断价值
                self._show_empty(self._build_funnel_text(rows))
            else:
                self._fill_table(display_rows)
                self.table.show()
                self.kline_button.setEnabled(True)
                self.export_button.setEnabled(True)

            self._refresh_alerts(rows)
        except Exception as exc:
            self._show_empty("刷新失败：%s" % exc)
            self.alert_label.setText("刷新异常：%s" % exc)
            self.alert_label.setStyleSheet("color:#b00020;")

    def _show_empty(self, text):
        # 保留完整表头，避免后续有数据时列错位
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.setRowCount(1)
        self.table.clearContents()
        # 空态可能携带漏斗诊断文本，启用换行避免被截断
        self.table.setWordWrap(True)
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(Qt.NoItemFlags)
        self.table.setItem(0, 0, item)
        # 合并所有单元格让文案居中横跨整表
        self.table.setSpan(0, 0, 1, len(self.headers))
        # 恢复表头 tooltip（setHorizontalHeaderLabels 会重置表头 item）
        self._setup_header_tooltips()
        # 内容自适应 + 精细 Stretch 策略
        self.table.resizeColumnsToContents()
        self._setup_column_resize_modes()
        self.kline_button.setEnabled(False)
        self.export_button.setEnabled(False)

    def _fill_table(self, rows):
        self.table.clearSpans()
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            risk = row.get("risk", "NORMAL")
            reason = row.get("risk_reason", "")

            # 代码（风控状态作为 tooltip 附加）
            code_item = QTableWidgetItem(row["code"])
            code_item.setData(Qt.UserRole, row["code"])
            code_item.setData(Qt.UserRole + 1, row.get("date", ""))
            if risk == "BLACKLIST":
                code_item.setForeground(QBrush(QColor("#888")))
            if reason:
                code_item.setToolTip("风控原因：" + reason)
            self.table.setItem(r, 0, code_item)

            # 名称
            name_item = QTableWidgetItem(row.get("name") or "—")
            if risk == "BLACKLIST":
                name_item.setForeground(QBrush(QColor("#888")))
            self.table.setItem(r, 1, name_item)

            # 仓位（风控后）—— BLACKLIST 置灰显示 0
            pos_item = QTableWidgetItem("%.2f%%" % (row["position"] * 100))
            if risk == "BLACKLIST":
                pos_item.setForeground(QBrush(QColor("#888")))
                if row.get("raw_position", 0) > 0:
                    pos_item.setToolTip("原始 NPU 仓位 %.2f%% → 被 BLACKLIST 拦截置 0"
                                        % (row["raw_position"] * 100))
            self.table.setItem(r, 2, pos_item)

            # 信号（带颜色）
            sig = row.get("signal", "")
            sig_item = QTableWidgetItem(sig or "—")
            if sig == "新开仓" or sig == "加仓":
                sig_item.setForeground(QBrush(QColor("#b00020")))
            elif sig == "减仓":
                sig_item.setForeground(QBrush(QColor("#b8860b")))
            elif sig == "清仓":
                sig_item.setForeground(QBrush(QColor("#555")))
            if risk == "BLACKLIST":
                sig_item.setForeground(QBrush(QColor("#888")))
            self.table.setItem(r, 3, sig_item)

            # 均线破位标记（第 4 列）：破位橙底白字，正常灰字
            ma = row.get("ma_break") or {}
            ma_item = QTableWidgetItem()
            if ma.get("broken"):
                ma_item.setText("破位↓")
                ma_item.setBackground(QBrush(QColor("#e0781f")))
                ma_item.setForeground(QBrush(QColor("#ffffff")))
                ma_item.setToolTip(ma.get("tip", ""))
            elif ma:
                ma_item.setText("正常")
                ma_item.setForeground(QBrush(QColor("#888")))
                ma_item.setToolTip(ma.get("tip", ""))
            else:
                ma_item.setText("—")
                ma_item.setForeground(QBrush(QColor("#999")))
                ma_item.setToolTip("本地日线不足 %d 根，无法计算均线"
                                   % max(ma_break.DEFAULT_PERIODS.values()))
            self.table.setItem(r, 4, ma_item)

            # 捕获率（上涨日收益捕获率）
            capture = row.get("capture")
            cap_item = QTableWidgetItem("—" if capture is None else "%.2f%%" % (capture * 100))
            cap_item.setToolTip("回测窗口内上涨日策略收益 / 基准收益")
            if capture is not None and capture < 0.40:
                cap_item.setForeground(QBrush(QColor("#b8860b")))
            if risk == "BLACKLIST":
                cap_item.setForeground(QBrush(QColor("#888")))
            self.table.setItem(r, 5, cap_item)

            # FP16 一致性误差（仓位误差，tooltip 附 NAV 相对误差）
            err_item = QTableWidgetItem("%.2e" % row.get("pos_err", 0.0))
            err_item.setToolTip("仓位误差阈值 1e-3；NAV 相对误差 %.2e（阈值 1e-5）"
                                % row.get("nav_err", 0.0))
            if row.get("pos_exceed") or row.get("nav_exceed"):
                err_item.setForeground(QBrush(QColor("#b00020")))
            if risk == "BLACKLIST":
                err_item.setForeground(QBrush(QColor("#888")))
            self.table.setItem(r, 6, err_item)

            # 风控状态（第 7 列）
            risk_item = QTableWidgetItem()
            if risk == "BLACKLIST":
                risk_item.setText("已拦截")
                risk_item.setForeground(QBrush(QColor("#b00020")))
                risk_item.setToolTip(reason or "BLACKLIST：硬拦截，仓位强制置 0")
            elif risk == "WATCH":
                risk_item.setText("⚠️ 预警")
                risk_item.setForeground(QBrush(QColor("#b8860b")))
                risk_item.setToolTip(reason or "WATCH：软预警，仓位减半")
            else:
                risk_item.setText("✓ 正常")
                risk_item.setForeground(QBrush(QColor("#0a6b3d")))
            self.table.setItem(r, 7, risk_item)

            # 风控原因（第 8 列）—— 短显示，完整看 tooltip
            reason_item = QTableWidgetItem()
            if reason:
                display = reason[:30] + ("…" if len(reason) > 30 else "")
                reason_item.setText(display)
                reason_item.setToolTip(reason)
                if risk == "BLACKLIST":
                    reason_item.setForeground(QBrush(QColor("#b00020")))
                elif risk == "WATCH":
                    reason_item.setForeground(QBrush(QColor("#b8860b")))
            else:
                reason_item.setText("—")
                reason_item.setForeground(QBrush(QColor("#999")))
            self.table.setItem(r, 8, reason_item)

        # 恢复表头 tooltip
        self._setup_header_tooltips()
        # 先让 Qt 根据内容算一遍理想宽度
        self.table.resizeColumnsToContents()
        # 再应用精细 Stretch / ResizeToContents 混合策略
        self._setup_column_resize_modes()

    def _refresh_alerts(self, rows):
        """与影子推理页告警规则保持一致。"""
        messages = []
        try:
            # 读取 monitor_history.csv 拿 600519 捕获率与 streak
            mon_path = Path(__file__).resolve().parent / "shadow_output" / "monitor_history.csv"
            streak = 0
            capture = None
            if mon_path.exists():
                df = pd.read_csv(mon_path)
                sub = df[df["code"].astype(str) == "600519"]
                if not sub.empty:
                    latest = sub.iloc[-1]
                    try:
                        streak = int(float(latest.get("capture_streak_600519", 0) or 0))
                    except Exception:
                        streak = 0
                    cap_val = latest.get("uptrend_capture")
                    try:
                        capture = float(cap_val) if cap_val is not None and not pd.isna(cap_val) else None
                    except Exception:
                        capture = None
            if capture is not None and capture < 0.40 and streak >= 5:
                messages.append(
                    "<span style='color:#b00020;font-weight:bold;'>600519 捕获率 %.2f%% 且 streak=%d，"
                    "进入人工 floor 讨论（不自动调参）</span>"
                    % (capture * 100, streak))
            elif capture is not None and capture < 0.40:
                messages.append(
                    "<span style='color:#7a5b00;'>600519 捕获率 %.2f%%（streak=%d，观察中）</span>"
                    % (capture * 100, streak))
        except Exception:
            pass

        # FP16 一致性
        for row in rows:
            if row["pos_exceed"] or row["nav_exceed"]:
                messages.append(
                    "<span style='color:#b00020;'>%s 一致性超差：pos_err=%.3e nav_err=%.3e</span>"
                    % (row["code"], row["pos_err"], row["nav_err"]))

        # 事件风控告警（扩展）
        blocked = [r for r in rows if r.get("risk") == "BLACKLIST"]
        watched = [r for r in rows if r.get("risk") == "WATCH"]
        if blocked:
            codes = ", ".join("%s(%s)" % (r["code"], r.get("name", ""))
                              for r in blocked if not r.get("_hidden"))
            messages.append(
                "<span style='color:#b00020;font-weight:bold;'>"
                "🚫 事件风控 BLACKLIST 拦截 %d 只：%s</span>"
                % (len(blocked), codes or "(全被置 0，无进表格)"))
        if watched:
            codes = ", ".join("%s(%s)" % (r["code"], r.get("name", ""))
                              for r in watched if not r.get("_hidden"))
            messages.append(
                "<span style='color:#b8860b;'>⚠️ 事件风控 WATCH 预警 %d 只：%s</span>"
                % (len(watched), codes or "(全被减半，无进表格)"))

        # 失败日志
        try:
            log_dir = Path(__file__).resolve().parent / "logs"
            if log_dir.exists():
                failed = []
                for p in sorted(log_dir.glob("shadow_*.log"))[-10:]:
                    try:
                        content = p.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
                    for line in content.splitlines():
                        if "FAILED exit=" in line:
                            failed.append("%s: %s" % (p.name, line.strip()))
                            break
                if failed:
                    messages.append(
                        "<span style='color:#b00020;'>后台影子推理失败：</span>"
                        + "<br>".join(failed))
        except Exception:
            pass

        if not messages:
            self.alert_label.setText(
                "<span style='color:#0a6b3d;'>当前无告警（streak=%d）</span>" % streak)
        else:
            self.alert_label.setText("<br>".join(messages))

        # 同步主窗口底部徽标：BLACKLIST 红色计数，只有 WATCH 时黄色计数
        self.window.set_alert_badge(
            any("#b00020" in item and "事件风控 BLACKLIST" not in item
                for item in messages),
            streak, loaded=True,
            risk_blacklist=len(blocked), risk_watch=len(watched),
        )

    # -------- 交互 --------

    def _on_double_clicked(self, row, _col):
        self._open_kline_for_row(row)

    def _open_selected_kline(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "未选择股票", "请先选择一行影子推理选股结果")
            return
        self._open_kline_for_row(row)

    def _open_kline_for_row(self, row):
        code_item = self.table.item(row, 0)
        if code_item is None:
            return
        code = code_item.data(Qt.UserRole) or code_item.text()
        date = code_item.data(Qt.UserRole + 1) or None
        self.window.show_kline(code, anchor_date=date)

    def _export(self):
        if not self._latest_rows:
            QMessageBox.information(self, "无数据", "当前没有可导出的影子推理选股结果")
            return
        default_name = "shadow_select_%s.csv" % (self._latest_date.replace("-", ""))
        path, _ = QFileDialog.getSaveFileName(
            self, "导出影子推理选股结果", default_name, "CSV (*.csv)")
        if not path:
            return
        frame = pd.DataFrame([{
            "代码": r["code"],
            "名称": r.get("name", ""),
            "仓位": r["position"],
            "原始仓位": r.get("raw_position", r["position"]),
            "信号": r.get("signal", ""),
            "均线破位": ("破位" if (r.get("ma_break") or {}).get("broken")
                        else ("正常" if r.get("ma_break") else "")),
            "MIN_MA": (r.get("ma_break") or {}).get("min_ma", ""),
            "捕获率": ("" if r.get("capture") is None
                       else round(r["capture"], 6)),
            "FP16仓位误差": r.get("pos_err", 0.0),
            "FP16_NAV相对误差": r.get("nav_err", 0.0),
            "风控状态": r.get("risk", "NORMAL"),
            "风控原因": r.get("risk_reason", ""),
        } for r in self._latest_rows])
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        QMessageBox.information(self, "导出成功", "已导出到\n%s" % path)


class FormulaPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.selection_rows = []
        layout = QVBoxLayout(self)

        # 两个并列入口：规则选股（原有）与影子推理选股（新增）
        self.tabs = QTabWidget()
        self.rule_tab = QWidget()
        rule_layout = QVBoxLayout(self.rule_tab)
        self.tabs.addTab(self.rule_tab, "规则选股")
        self.shadow_select_page = ShadowSelectPage(window)
        self.tabs.addTab(self.shadow_select_page, "影子推理选股")
        layout.addWidget(self.tabs)

        top_layout = QHBoxLayout()
        rule_layout.addLayout(top_layout, 0)

        universe_group = QGroupBox("选股范围与过滤")
        universe_layout = QGridLayout(universe_group)
        self.pool_combo = QComboBox()
        self.refresh_pools_button = QPushButton("刷新股票池")
        self.refresh_pools_button.clicked.connect(self.refresh_pools)
        self.fq_combo = QComboBox()
        add_fq_options(self.fq_combo)
        self.manual_codes = QPlainTextEdit()
        self.manual_codes.setPlaceholderText("手动输入优先于股票池（空格/换行分隔）")
        self.manual_codes.setToolTip("填写后优先于股票池，例如：600519 000001，或使用逗号、换行分隔")
        self.manual_codes.setFixedHeight(46)
        self.exclude_st = QCheckBox("排除 ST/*ST")
        self.exclude_st.setChecked(True)
        self.exclude_suspended = QCheckBox("排除停牌/当日无行情")
        self.exclude_suspended.setChecked(True)
        self.exclude_blacklist = QCheckBox("排除黑名单")
        self.exclude_blacklist.setChecked(True)
        self.min_list_days = QSpinBox()
        self.min_list_days.setRange(0, 100000)
        self.min_list_days.setSuffix(" 天")
        self.min_amount = QDoubleSpinBox()
        self.min_amount.setRange(0, 100000000)
        self.min_amount.setDecimals(0)
        self.min_amount.setSuffix(" 万元")
        self.list_days_hint = QLabel("0 表示不限制")
        self.amount_hint = QLabel("0 表示不限制")
        for hint in (self.list_days_hint, self.amount_hint):
            hint.setStyleSheet("color: #777; font-size: 11px;")
        universe_layout.addWidget(QLabel("股票池"), 0, 0)
        universe_layout.addWidget(self.pool_combo, 0, 1)
        universe_layout.addWidget(self.refresh_pools_button, 0, 2)
        universe_layout.addWidget(QLabel("行情口径"), 1, 0)
        universe_layout.addWidget(self.fq_combo, 1, 1)
        universe_layout.addWidget(self.manual_codes, 2, 0, 1, 3)
        universe_layout.addWidget(self.exclude_st, 3, 0, 1, 2)
        universe_layout.addWidget(self.exclude_suspended, 4, 0, 1, 2)
        universe_layout.addWidget(self.exclude_blacklist, 5, 0, 1, 2)
        universe_layout.addWidget(QLabel("最少上市天数"), 6, 0)
        universe_layout.addWidget(self.min_list_days, 6, 1)
        universe_layout.addWidget(self.list_days_hint, 6, 2)
        universe_layout.addWidget(QLabel("最低成交额"), 7, 0)
        universe_layout.addWidget(self.min_amount, 7, 1)
        universe_layout.addWidget(self.amount_hint, 7, 2)
        top_layout.addWidget(universe_group, 1)

        formula_group = QGroupBox("本地选股公式")
        formula_layout = QGridLayout(formula_group)
        self.formula_edit = QPlainTextEdit(hammer_boll.BUY_FORMULA)
        self.formula_edit.setPlaceholderText(
            "在此输入选股公式，或点击下方按钮载入内置公式")
        self.param_n = QSpinBox()
        self.param_n.setRange(1, 250)
        self.param_n.setValue(20)
        self.formula_name = QLineEdit()
        self.formula_name.setPlaceholderText("保存时填写公式名称")
        self.saved_formulas = QComboBox()
        self.save_formula_button = QPushButton("保存公式")
        self.load_formula_button = QPushButton("加载公式")
        self.run_button = QPushButton("开始选股")
        self.save_formula_button.clicked.connect(self.save_formula)
        self.load_formula_button.clicked.connect(self.load_formula)
        self.run_button.clicked.connect(self.run_selection)
        self.formula_hint = QLabel(
            "已内置未来函数（ZIG/BACKSET）检测，回测需注意防未来\n"
            "一致性提示：参数 N 仅传递给公式中的 N（如 MA(C,N)）；"
            "MA(C,5)、MA(C,20) 等固定周期不会随 N 自动修改。"
        )
        self.formula_hint.setWordWrap(True)
        self.formula_hint.setStyleSheet("color: #777; font-size: 11px;")
        formula_layout.addWidget(self.formula_edit, 0, 0, 1, 4)
        formula_layout.addWidget(self.formula_hint, 1, 0, 1, 4)
        formula_layout.addWidget(QLabel("参数 N"), 2, 0)
        formula_layout.addWidget(self.param_n, 2, 1)
        formula_layout.addWidget(self.run_button, 2, 3)
        formula_layout.addWidget(self.formula_name, 3, 0)
        formula_layout.addWidget(self.save_formula_button, 3, 1)
        formula_layout.addWidget(self.saved_formulas, 3, 2)
        formula_layout.addWidget(self.load_formula_button, 3, 3)
        self.hammer_load_button = QPushButton("载入 HAMMER_BOLL_BUY")
        self.hammer_install_button = QPushButton("写入公式库并导出 SELL 公式")
        self.hammer_load_button.clicked.connect(self.load_hammer_preset)
        self.hammer_install_button.clicked.connect(self.install_hammer_formula)
        formula_layout.addWidget(self.hammer_load_button, 4, 0, 1, 2)
        formula_layout.addWidget(self.hammer_install_button, 4, 2, 1, 2)
        hammer_hint = QLabel(
            "卖出落点：通达信 Ctrl+F → 条件选股公式 → 新建 HAMMER_BOLL_SELL → 粘贴导出的 SELL 文本；"
            "满 5 个交易日强制离场由“持仓盯盘”页按买入日期计算。本工具不自动操作通达信、不下单。"
        )
        hammer_hint.setWordWrap(True)
        hammer_hint.setStyleSheet("color: #777; font-size: 11px;")
        formula_layout.addWidget(hammer_hint, 5, 0, 1, 4)

        # 均线破位选股参数：N1~N4 对应 MA5/MA10/MA20/MA60，可在此按需修改。
        self.ma_period_spins = {}
        ma_controls = QWidget()
        ma_controls_layout = QHBoxLayout(ma_controls)
        ma_controls_layout.setContentsMargins(0, 0, 0, 0)
        for key, value in ma_break.DEFAULT_PERIODS.items():
            spin = QSpinBox()
            spin.setRange(1, 250)
            spin.setValue(value)
            spin.setPrefix(key + " ")
            spin.setToolTip("均线周期参数 %s，默认 %d 日" % (key, value))
            self.ma_period_spins[key] = spin
            ma_controls_layout.addWidget(spin)
        self.ma_yin_check = QCheckBox("阴线确认（收盘<开盘，可选）")
        self.ma_yin_check.setChecked(False)
        ma_controls_layout.addWidget(self.ma_yin_check)
        self.ma_break_load_button = QPushButton("载入均线破位选股")
        self.ma_break_load_button.clicked.connect(self.load_ma_break_preset)
        ma_controls_layout.addWidget(self.ma_break_load_button)
        ma_controls_layout.addStretch()
        formula_layout.addWidget(QLabel("均线破位参数"), 6, 0)
        formula_layout.addWidget(ma_controls, 6, 1, 1, 3)

        self.base_date = make_date_edit(QDate.currentDate())
        self.base_date.setToolTip("按该日期及以前的本地行情选股；非交易日自动取最近交易日")
        base_hint = QLabel("按基准日收盘计算信号，行情只使用该日及以前数据")
        base_hint.setStyleSheet("color: #777; font-size: 11px;")
        formula_layout.addWidget(QLabel("选股基准日"), 7, 0)
        formula_layout.addWidget(self.base_date, 7, 1)
        formula_layout.addWidget(base_hint, 7, 2, 1, 2)
        top_layout.addWidget(formula_group, 2)
        rule_layout.addLayout(top_layout, 0)

        result_header = QHBoxLayout()
        self.result_label = QLabel("尚未执行选股；双击结果可查看 K 线")
        result_header.addWidget(self.result_label)
        result_header.addStretch()
        self.kline_button = QPushButton("查看选中 K 线")
        self.export_button = QPushButton("导出结果 CSV")
        self.ydzq_button = QPushButton("写入英大证券")
        self.ydzq_button.setToolTip(
            "一键追加到英大证券自选股 zxg.blk（自动定位安装目录）；保留原有自选股，"
            "写入前自动备份。请先完全退出英大证券客户端。")
        self.export_button.setEnabled(False)
        self.ydzq_button.setEnabled(False)
        self.kline_button.setEnabled(False)
        self.kline_button.clicked.connect(self.open_selected_kline)
        self.export_button.clicked.connect(self.export_results)
        self.ydzq_button.clicked.connect(self.write_ydzq_watchlist)
        result_header.addWidget(self.kline_button)
        result_header.addWidget(self.export_button)
        result_header.addWidget(self.ydzq_button)
        rule_layout.addLayout(result_header)

        self.result_table = QTableWidget()
        self.default_result_headers = ["code", "name", "date", "close", "change_pct"]
        self.result_headers = self.default_result_headers
        prepare_table(self.result_table, self.result_headers)
        self.result_table.cellDoubleClicked.connect(self.open_kline_from_row)
        self.result_empty_label = QLabel("尚未执行选股或无符合标的")
        self.result_empty_label.setAlignment(Qt.AlignCenter)
        self.result_empty_label.setStyleSheet("color: #777; padding: 24px;")
        self.result_stack = QStackedWidget()
        self.result_stack.addWidget(self.result_table)
        self.result_stack.addWidget(self.result_empty_label)
        rule_layout.addWidget(self.result_stack, 1)
        self.update_result_state()
        try:
            ma_break.ensure_formula()  # 确保“均线破位选股”常驻本地公式库
        except Exception as exc:
            self.window.statusBar().showMessage("均线破位公式写入失败：%s" % exc, 8000)
        self.refresh_pools()
        self.refresh_formula_list()

    def refresh_pools(self):
        try:
            current = self.pool_combo.currentData()
            items = pool.list_pools()
            self.pool_combo.clear()
            self.pool_combo.addItem("本地全 A（以本地股票档案为准）", None)
            type_names = {"custom": "自选", "industry": "行业", "concept": "概念"}
            blacklist_count = 0
            for item in items:
                if item["type"] == "blacklist":
                    # 黑名单由过滤配置统一取并集，不能作为正向选股范围。
                    blacklist_count += 1
                    continue
                label = "%s（%s，%d只）" % (
                    item["name"], type_names.get(item["type"], item["type"]), item["member_count"]
                )
                self.pool_combo.addItem(label, item["id"])
            index = self.pool_combo.findData(current)
            self.pool_combo.setCurrentIndex(max(0, index))
            self.window.statusBar().showMessage(
                "股票池已刷新：全A/自选/行业/概念共 %d 项；%d 个黑名单池由过滤项处理"
                % (self.pool_combo.count(), blacklist_count), 5000
            )
        except Exception as exc:
            self.window.show_error("读取股票池失败", traceback.format_exc(), str(exc))

    def refresh_formula_list(self):
        self.saved_formulas.clear()
        for spec in formula_engine.list_formulas("select"):
            self.saved_formulas.addItem(spec["name"])

    def _formula_params(self):
        """汇总界面参数：N 为通用周期，N1~N4/YIN 供均线破位公式使用。"""
        params = {"N": self.param_n.value()}
        for key, spin in self.ma_period_spins.items():
            params[key] = spin.value()
        params["YIN"] = 1 if self.ma_yin_check.isChecked() else 0
        return params

    def save_formula(self):
        name = self.formula_name.text().strip()
        if not name:
            QMessageBox.warning(self, "无法保存", "请先填写公式名称")
            return
        try:
            params = {key: {"default": value} for key, value in self._formula_params().items()}
            path = formula_engine.save_formula(
                name,
                "select",
                self.formula_edit.toPlainText(),
                params=params,
            )
            self.refresh_formula_list()
            self.saved_formulas.setCurrentText(name)
            QMessageBox.information(self, "保存成功", "公式已保存到：\n%s" % path)
        except Exception as exc:
            self.window.show_error("公式保存失败", traceback.format_exc(), str(exc))

    def load_formula(self):
        name = self.saved_formulas.currentText()
        if not name:
            QMessageBox.warning(self, "无法加载", "本地公式库为空")
            return
        try:
            spec = formula_engine.load_formula(name, "select")
            self.formula_edit.setPlainText(spec["content"])
            self.formula_name.setText(name)
            saved_params = spec.get("params") or {}

            def _default(key):
                definition = saved_params.get(key)
                if isinstance(definition, dict):
                    return definition.get("default")
                return definition

            definition = _default("N")
            if definition is not None:
                self.param_n.setValue(int(definition))
            for key, spin in self.ma_period_spins.items():
                value = _default(key)
                if value is not None:
                    spin.setValue(int(value))
            yin = _default("YIN")
            if yin is not None:
                self.ma_yin_check.setChecked(bool(int(yin)))
        except Exception as exc:
            self.window.show_error("公式加载失败", traceback.format_exc(), str(exc))

    def load_hammer_preset(self):
        """把 HAMMER_BOLL_BUY 买入公式载入编辑器（不自动运行选股）。"""
        self.formula_edit.setPlainText(hammer_boll.BUY_FORMULA)
        self.formula_name.setText(hammer_boll.BUY_NAME)
        self.param_n.setValue(hammer_boll.DEFAULT_N)
        self.window.statusBar().showMessage(
            "已载入 %s（N=%d）；点击“开始选股”后可导出候选池 CSV"
            % (hammer_boll.BUY_NAME, hammer_boll.DEFAULT_N), 8000
        )

    def load_ma_break_preset(self):
        """载入“均线破位选股”公式，并同步界面参数（不自动运行选股）。"""
        self.formula_edit.setPlainText(ma_break.FORMULA)
        self.formula_name.setText(ma_break.NAME)
        self.ma_yin_check.setChecked(False)
        periods = "/".join(str(spin.value()) for spin in self.ma_period_spins.values())
        self.window.statusBar().showMessage(
            "已载入 %s（N1~N4=%s）；点击“开始选股”查看 MIN_MA 与偏离幅度"
            % (ma_break.NAME, periods), 8000
        )

    def install_hammer_formula(self):
        """BUY 写入本地公式库，并导出 BUY/SELL 文本供通达信手动新建公式。"""
        try:
            buy_path = hammer_boll.ensure_buy_formula()
            buy_txt, sell_txt = hammer_boll.save_formula_files()
        except Exception as exc:
            self.window.show_error("HAMMER 公式写入失败", traceback.format_exc(), str(exc))
            return
        self.refresh_formula_list()
        self.saved_formulas.setCurrentText(hammer_boll.BUY_NAME)
        self.load_hammer_preset()
        QMessageBox.information(
            self, "HAMMER 公式已就绪",
            "买入公式已写入本地公式库：\n%s\n\n"
            "SELL 公式（通达信手动创建）：\n%s\n\n%s\n\n"
            "卖出结果 CSV 可在“持仓盯盘”页导入核对；仅研究用途，不构成投资建议。"
            % (buy_path, sell_txt, hammer_boll.TDX_STEPS)
        )

    def run_selection(self):
        formula = self.formula_edit.toPlainText().strip()
        if not formula:
            QMessageBox.warning(self, "无法选股", "公式不能为空")
            return
        try:
            manual_codes = parse_codes(self.manual_codes.toPlainText())
        except ValueError as exc:
            QMessageBox.warning(self, "无法选股", str(exc))
            return
        pool_id = self.pool_combo.currentData()
        fq = self.fq_combo.currentData()
        params = self._formula_params()
        # 公式含 MINMA 时按均线破位口径展示专属结果列（MIN_MA/开盘价/偏离幅度）。
        ma_break_mode = "MINMA" in formula.upper()
        requested_date = self.base_date.date().toString("yyyy-MM-dd")
        filter_config = pool.FilterConfig(
            exclude_st=self.exclude_st.isChecked(),
            exclude_suspended=self.exclude_suspended.isChecked(),
            min_list_days=self.min_list_days.value(),
            min_amount=self.min_amount.value() * 10000,
            exclude_blacklist=self.exclude_blacklist.isChecked(),
        )

        def task(progress):
            effective = pool.latest_trade_date_on_or_before(requested_date)
            if not effective:
                raise ValueError("本地没有 %s 及以前的日线行情" % requested_date)
            if manual_codes:
                universe = pool.filter_stocks(
                    manual_codes, config=filter_config, trade_date=effective)
            else:
                universe = pool.get_universe(
                    pool_id=pool_id, config=filter_config, trade_date=effective)
            codes = universe["codes"]
            data_map = {}
            load_errors = {}
            for index, code in enumerate(codes, start=1):
                progress.emit({"current": index - 1, "total": len(codes), "text": "加载本地行情 " + code})
                try:
                    frame = load_local_daily(code, fq)
                    if frame.empty:
                        raise ValueError("本地无%s行情" % fq)
                    frame = frame[frame["date"].astype(str).str[:10] <= effective]
                    frame = frame.reset_index(drop=True)
                    if frame.empty or str(frame.iloc[-1]["date"])[:10] != effective:
                        raise ValueError("基准日 %s 无行情（可能停牌）" % effective)
                    data_map[code] = frame
                except Exception as exc:
                    load_errors[code] = str(exc)
                progress.emit({"current": index, "total": len(codes), "text": "已处理 " + code})
            selected, formula_errors = formula_engine.select_latest(data_map, formula, params)
            enriched = []
            for item in selected:
                code = item["code"]
                frame = data_map[code]
                close = float(frame.iloc[-1]["close"])
                change_pct = None
                if len(frame) > 1 and float(frame.iloc[-2]["close"]):
                    change_pct = (close / float(frame.iloc[-2]["close"]) - 1) * 100
                stock = db.query_one("SELECT name FROM stock_info WHERE code = ?", (code,)) or {}
                extra = {}
                if ma_break_mode:
                    extra = ma_break.extra_fields(
                        frame, params["N1"], params["N2"], params["N3"], params["N4"])
                enriched.append({
                    **item,
                    "name": stock.get("name", ""),
                    "close": close,
                    "change_pct": change_pct,
                    **extra,
                })
            return {
                "selected": enriched,
                "load_errors": load_errors,
                "formula_errors": formula_errors,
                "rejected": universe["rejected"],
                "trade_date": universe["trade_date"],
                "requested_date": requested_date,
                "ma_break_mode": ma_break_mode,
            }

        self.window.start_task(task, self.selection_finished, "正在执行本地公式选股")

    def update_result_state(self):
        """结果为空时展示引导，并同步禁用导出和 K 线按钮。"""
        has_results = bool(self.selection_rows)
        self.result_stack.setCurrentWidget(self.result_table if has_results else self.result_empty_label)
        self.export_button.setEnabled(has_results)
        self.kline_button.setEnabled(has_results)
        self.ydzq_button.setEnabled(has_results)

    def apply_result_mode(self, ma_break_mode):
        """按公式类型切换结果列：均线破位展示 MIN_MA/开盘价/偏离幅度。"""
        if ma_break_mode:
            self.result_headers = [
                "code", "name", "date", "min_ma", "close", "open", "deviation_pct"]
            labels = ["代码", "名称", "选股日", "MIN_MA", "收盘价", "开盘价", "偏离幅度%"]
        else:
            self.result_headers = self.default_result_headers
            labels = list(self.result_headers)
        prepare_table(self.result_table, labels)

    def selection_finished(self, result):
        self.selection_rows = result["selected"]
        self.apply_result_mode(result.get("ma_break_mode", False))
        self.result_table.setRowCount(len(self.selection_rows))
        for row_index, row in enumerate(self.selection_rows):
            for column, key in enumerate(self.result_headers):
                value = row.get(key)
                if isinstance(value, (int, float)):
                    if key in ("close", "min_ma", "open"):
                        value = "%.3f" % value
                    elif key in ("change_pct", "deviation_pct"):
                        value = "%.2f%%" % value
                set_cell(
                    self.result_table, row_index, column, value,
                    user_data=row["code"] if column == 0 else None,
                    numeric=key in ("close", "min_ma", "open", "change_pct", "deviation_pct"),
                )
        error_count = len(result["load_errors"]) + len(result["formula_errors"])
        base_text = result["trade_date"]
        if result.get("requested_date") and result["requested_date"] != base_text:
            base_text = "%s（请求 %s，非交易日取前一交易日）" % (
                base_text, result["requested_date"])
        self.result_label.setText(
            "基准日 %s：命中 %d 只；过滤剔除 %d 只；加载/公式错误 %d 只"
            % (base_text, len(self.selection_rows), len(result["rejected"]), error_count)
        )
        self.update_result_state()
        if error_count:
            details = {
                "load_errors": result["load_errors"],
                "formula_errors": result["formula_errors"],
            }
            box = QMessageBox(
                QMessageBox.Warning, "部分股票未参与选股", self.result_label.text(), parent=self
            )
            box.setDetailedText(json.dumps(details, ensure_ascii=False, indent=2))
            box.exec_()

    def export_results(self):
        if not self.selection_rows:
            QMessageBox.warning(self, "无法导出", "当前没有选股结果")
            return
        db.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        is_hammer = self.formula_name.text().strip().upper().startswith("HAMMER")
        if is_hammer:
            as_of = str(self.selection_rows[0].get("date", ""))[:10]
            as_of = as_of or datetime.now().strftime("%Y-%m-%d")
            default_path = db.DATA_DIR / "sim" / (
                "candidates_%s.csv" % as_of.replace("-", ""))
            default_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            default_path = db.EXPORT_DIR / (
                "selection_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
        path, _ = QFileDialog.getSaveFileName(self, "导出选股结果", str(default_path), "CSV 文件 (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            if is_hammer:
                rows = hammer_boll.candidate_rows(self.selection_rows, self.fq_combo.currentData())
                rows = [{**row, "disclaimer": DISCLAIMER} for row in rows]
            else:
                rows = [{**row, "disclaimer": DISCLAIMER} for row in self.selection_rows]
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            self.window.statusBar().showMessage("选股结果已导出：" + path, 10000)
            QMessageBox.information(self, "导出完成", path)
        except Exception as exc:
            self.window.show_error("导出失败", traceback.format_exc(), str(exc))

    def write_ydzq_watchlist(self):
        """把当前选股结果追加到英大证券自选股（保留原有自选股，不删除）。"""
        codes = [str(row.get("code", "")).zfill(6) for row in self.selection_rows]
        codes = [code for code in codes if re.fullmatch(r"\d{6}", code)]
        if not codes:
            QMessageBox.warning(self, "无法写入", "当前没有可写入的选股结果")
            return
        unique_count = len(set(codes))
        path = ydzq_watchlist.find_default_path()
        if path is None:
            # 自动找不到时才回退到手动选择，选中一次后会被记住
            QMessageBox.information(
                self, "未自动找到 zxg.blk",
                "未在注册表或常见安装目录找到英大证券自选股文件，请手动选择一次。\n"
                "文件通常位于客户端安装目录 T0002\\blocknew\\zxg.blk。")
            path, _ = QFileDialog.getOpenFileName(
                self, "选择英大证券自选股文件 zxg.blk", str(Path.home()),
                "自选股文件 (zxg.blk);;所有文件 (*.*)")
            if not path:
                return
        answer = QMessageBox.question(
            self, "写入前确认",
            "已定位自选股文件：\n%s\n\n"
            "请先完全退出英大证券客户端，否则客户端退出时可能覆盖本次写入结果。\n\n"
            "将把 %d 只股票追加到自选股（保留原有自选股）。是否继续？"
            % (path, unique_count),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            result = ydzq_watchlist.append_codes(path, codes)
            ydzq_watchlist.save_last_path(path)
        except Exception as exc:
            self.window.show_error("写入英大证券失败", traceback.format_exc(), str(exc))
            return
        backup = str(result["backup"]) if result["backup"] else "（新建文件，无备份）"
        self.window.statusBar().showMessage(
            "英大证券自选股已追加 %d 只" % result["added"], 10000)
        QMessageBox.information(
            self, "写入完成",
            "自选股文件：%s\n本次新增 %d 只，已存在跳过 %d 只\n原文件备份：%s\n\n"
            "请勿在英大证券客户端运行时重复写入。%s。"
            % (result["path"], result["added"], result["skipped"], backup, DISCLAIMER)
        )

    def open_selected_kline(self):
        row = self.result_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "未选择股票", "请先选择一行选股结果")
            return
        self.open_kline_from_row(row, 0)

    def open_kline_from_row(self, row, _column):
        item = self.result_table.item(row, 0)
        if item:
            # 双击结果时把 K 线定位到信号日，避免用最新行情误判历史选股结果
            anchor = None
            if 0 <= row < len(self.selection_rows):
                anchor = str(self.selection_rows[row].get("date", ""))[:10] or None
            self.window.show_kline(item.text(), self.fq_combo.currentData(), anchor_date=anchor)


class KlinePage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.code_edit = QLineEdit("600519")
        self.code_edit.setMaximumWidth(120)
        self.period_combo = QComboBox()
        add_period_options(self.period_combo)
        self.fq_combo = QComboBox()
        add_fq_options(self.fq_combo)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(30, 500)
        self.count_spin.setValue(120)
        self.count_spin.setSuffix(" 根")
        self.load_button = QPushButton("读取本地 K 线")
        self.load_button.clicked.connect(self.load_stock)
        self.code_edit.returnPressed.connect(self.load_stock)
        # 周期、复权口径或显示根数变化后只重读本地行情，不触发网络请求。
        self.period_combo.currentIndexChanged.connect(self.period_changed)
        self.fq_combo.currentIndexChanged.connect(self.load_stock)
        self.count_spin.valueChanged.connect(self.load_stock)
        self.indicator_button = QPushButton("指标叠加（V1.1 扩展）")
        self.indicator_button.setEnabled(False)
        self.indicator_button.setToolTip("预留入口，V1.0 暂不实现指标叠加")
        controls.addWidget(QLabel("股票代码"))
        controls.addWidget(self.code_edit)
        controls.addWidget(QLabel("周期"))
        controls.addWidget(self.period_combo)
        controls.addWidget(QLabel("复权"))
        controls.addWidget(self.fq_combo)
        controls.addWidget(QLabel("显示"))
        controls.addWidget(self.count_spin)
        controls.addWidget(self.load_button)
        controls.addWidget(self.indicator_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.price_axis = self.figure.add_subplot(211)
        self.volume_axis = self.figure.add_subplot(212, sharex=self.price_axis)
        self.figure.text(0.99, 0.01, DISCLAIMER, ha="right", fontsize=8, color="gray")
        self.chart_empty_label = QLabel("请输入股票代码并点击【读取本地 K 线】")
        self.chart_empty_label.setAlignment(Qt.AlignCenter)
        self.chart_empty_label.setWordWrap(True)
        self.chart_empty_label.setStyleSheet("color: #777; padding: 24px;")
        self.chart_stack = QStackedWidget()
        self.chart_stack.addWidget(self.canvas)
        self.chart_stack.addWidget(self.chart_empty_label)
        self.chart_stack.setCurrentWidget(self.chart_empty_label)
        layout.addWidget(self.chart_stack)

    def show_chart_message(self, message):
        """无数据或读取失败时显示文字，不留下空白坐标轴或上一只股票旧图。"""
        self.price_axis.clear()
        self.volume_axis.clear()
        self.chart_empty_label.setText(message)
        self.chart_stack.setCurrentWidget(self.chart_empty_label)
        self.canvas.draw_idle()

    def _reset_fq_options(self, preferred=None):
        """日线保留三种口径；分钟仅提供不复权和本地重算前复权。"""
        is_daily = self.period_combo.currentData() == "daily"
        options = FQ_OPTIONS if is_daily else MINUTE_FQ_OPTIONS
        desired = preferred or self.fq_combo.currentData() or ("qfq" if is_daily else "none")
        blocked = self.fq_combo.blockSignals(True)
        self.fq_combo.clear()
        for label, value in options:
            self.fq_combo.addItem(label, value)
        index = self.fq_combo.findData(desired)
        if index < 0:
            index = self.fq_combo.findData("qfq" if is_daily else "none")
        self.fq_combo.setCurrentIndex(index)
        self.fq_combo.blockSignals(blocked)

    def period_changed(self, _index=None):
        # 每次切入分钟默认不复权，避免未准备日线因子时直接报错。
        self._reset_fq_options(preferred="qfq" if self.period_combo.currentData() == "daily" else "none")
        self.load_stock()

    def open_stock(self, code, fq="qfq", period="daily", anchor_date=None):
        self.code_edit.setText(code)
        period_index = self.period_combo.findData(period)
        if period_index >= 0:
            blocked = self.period_combo.blockSignals(True)
            self.period_combo.setCurrentIndex(period_index)
            self.period_combo.blockSignals(blocked)
        # 跳转只显式读取一次，避免周期/复权信号和 open_stock 重复加载。
        self._reset_fq_options(preferred=fq)
        self.load_stock(anchor_date=anchor_date)

    def load_stock(self, anchor_date=None):
        try:
            codes = parse_codes(self.code_edit.text())
            if len(codes) != 1:
                raise ValueError("K 线图一次只显示一只股票")
            code = codes[0]
            fq = self.fq_combo.currentData()
            period = self.period_combo.currentData()
            if period == "daily":
                frame = load_local_daily(code, fq)
                time_column = "date"
            else:
                frame = load_local_minute(code, period, fq)
                time_column = "datetime"
            if frame.empty:
                message = "本地无该代码%s数据，请先在【数据管理】更新" % self.period_combo.currentText()
                self.show_chart_message(message)
                self.window.statusBar().showMessage("%s：%s" % (code, message), 10000)
                return
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame[time_column], errors="coerce")
            for column in ["open", "high", "low", "close", "volume"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
            frame = frame.sort_values("date")
            frame["ma5"] = frame["close"].rolling(5).mean()
            frame["ma20"] = frame["close"].rolling(20).mean()

            # 如果给了 anchor_date，优先以该日期为中心取 count_spin 根
            if anchor_date is not None and period == "daily":
                try:
                    anchor_ts = pd.to_datetime(str(anchor_date), errors="coerce")
                    if not pd.isna(anchor_ts):
                        mask = frame["date"] <= anchor_ts
                        if mask.any():
                            idx = int(mask.sum()) - 1
                            start = max(0, idx - self.count_spin.value() // 2)
                            end = min(len(frame), start + self.count_spin.value())
                            frame = frame.iloc[start:end].reset_index(drop=True)
                except Exception:
                    frame = frame.tail(self.count_spin.value()).reset_index(drop=True)
            else:
                frame = frame.tail(self.count_spin.value()).reset_index(drop=True)
            if frame.empty:
                self.show_chart_message("本地行情没有可绘制的有效 OHLC 数据")
                return
            self.draw_frame(code, fq, frame, period)
        except Exception as exc:
            self.show_chart_message("K 线读取失败，请检查代码、复权因子或本地数据")
            self.window.show_error("K 线读取失败", traceback.format_exc(), str(exc))

    def draw_frame(self, code, fq, frame, period="daily"):
        self.price_axis.clear()
        self.volume_axis.clear()
        colors = []
        for index, row in frame.iterrows():
            open_price = float(row["open"])
            close = float(row["close"])
            color = "#d62728" if close >= open_price else "#2ca02c"
            colors.append(color)
            self.price_axis.vlines(index, float(row["low"]), float(row["high"]), color=color, linewidth=0.8)
            bottom = min(open_price, close)
            height = abs(close - open_price)
            if height < 1e-10:
                self.price_axis.hlines(close, index - 0.32, index + 0.32, color=color, linewidth=1)
            else:
                self.price_axis.add_patch(
                    Rectangle((index - 0.32, bottom), 0.64, height, facecolor=color, edgecolor=color)
                )
        self.price_axis.plot(frame.index, frame["ma5"], color="#1f77b4", linewidth=1, label="MA5")
        self.price_axis.plot(frame.index, frame["ma20"], color="#ff7f0e", linewidth=1, label="MA20")
        self.volume_axis.bar(frame.index, frame["volume"].fillna(0) / 10000, color=colors, width=0.64)
        stock = db.query_one("SELECT name FROM stock_info WHERE code = ?", (code,)) or {}
        fq_name = dict((value, label) for label, value in FQ_OPTIONS).get(fq, fq)
        period_name = dict((value, label) for label, value in PERIOD_OPTIONS).get(period, period)
        time_format = "%Y-%m-%d" if period == "daily" else "%Y-%m-%d %H:%M"
        self.price_axis.set_title(
            "%s %s - %s %s（截至 %s）" % (
                code, stock.get("name", ""), period_name, fq_name,
                frame.iloc[-1]["date"].strftime(time_format),
            )
        )
        self.price_axis.set_ylabel("价格")
        self.volume_axis.set_ylabel("成交量（万股）")
        self.price_axis.legend(loc="upper left")
        self.price_axis.grid(alpha=0.2)
        self.volume_axis.grid(alpha=0.2)
        self.price_axis.tick_params(labelbottom=False)
        tick_count = min(10, len(frame))
        if tick_count:
            positions = sorted({
                int(i * (len(frame) - 1) / max(tick_count - 1, 1))
                for i in range(tick_count)
            })
            self.volume_axis.set_xticks(positions)
            self.volume_axis.set_xticklabels(
                [frame.iloc[position]["date"].strftime(time_format) for position in positions],
                rotation=30,
                ha="right",
            )
        self.price_axis.set_xlim(-1, len(frame))
        self.figure.tight_layout(rect=(0, 0.025, 1, 1))
        self.canvas.draw_idle()
        self.chart_stack.setCurrentWidget(self.canvas)
        self.window.statusBar().showMessage("已显示 %s %s本地 K 线" % (code, period_name), 5000)


class BacktestPage(QWidget):
    SUMMARY_FIELDS = [
        ("期末权益", "final_value", "money"),
        ("总收益率", "total_return", "percent"),
        ("年化收益率", "annual_return", "percent"),
        ("夏普比率", "sharpe", "number"),
        ("最大回撤", "max_drawdown", "percent"),
        ("已平仓交易", "trades_closed", "integer"),
        ("未平仓交易", "trades_open", "integer"),
        ("胜率", "win_rate", "percent"),
        ("平均盈亏比", "pl_ratio", "number"),
        ("盈利因子", "profit_factor", "number"),
        ("已平仓净盈亏", "net_pnl", "money"),
    ]

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.last_result = None
        layout = QVBoxLayout(self)

        codes_group = QGroupBox("回测股票")
        codes_layout = QHBoxLayout(codes_group)
        self.codes_edit = QPlainTextEdit()
        self.codes_edit.setPlaceholderText("输入代码，例如：600519, 000001；回测只读取本地行情")
        self.codes_edit.setFixedHeight(58)
        self.use_selection_button = QPushButton("填入选股结果")
        self.use_selection_button.clicked.connect(self.use_selection_codes)
        codes_layout.addWidget(self.codes_edit)
        codes_layout.addWidget(self.use_selection_button)
        layout.addWidget(codes_group)

        settings_group = QGroupBox("回测设置")
        settings = QGridLayout(settings_group)
        self.fq_combo = QComboBox()
        add_fq_options(self.fq_combo)
        # ---- V1.2 Step G3：分钟回测周期选择 ----
        self.period_combo = QComboBox()
        self.period_combo.addItem("日线", "daily")
        self.period_combo.addItem("1 分钟", "1min")
        self.period_combo.addItem("5 分钟", "5min")
        self.period_combo.addItem("15 分钟", "15min")
        self.period_combo.addItem("30 分钟", "30min")
        self.period_combo.addItem("60 分钟", "60min")
        self.period_combo.setToolTip("分钟回测需先在数据管理更新对应分钟周期数据")
        # ----------------------------------------
        self.start_edit = make_date_edit(QDate(2020, 1, 1))
        self.end_edit = make_date_edit(QDate.currentDate())
        self.cash_edit = QDoubleSpinBox()
        self.cash_edit.setRange(10000, 1000000000000)
        self.cash_edit.setDecimals(2)
        self.cash_edit.setValue(1000000)
        self.cash_edit.setSuffix(" 元")
        self.fast_period = QSpinBox()
        self.fast_period.setRange(2, 250)
        self.fast_period.setValue(5)
        self.slow_period = QSpinBox()
        self.slow_period.setRange(3, 500)
        self.slow_period.setValue(20)
        # ---- V1.1 Step F2：ATR 追踪止盈止损参数 ----
        self.enable_atr = QCheckBox("启用 ATR 追踪止盈止损")
        self.enable_atr.setToolTip("启用后叠加 ATR 初始止损和 ATR 追踪止盈；关闭则只用固定止损")
        self.atr_period = QSpinBox()
        self.atr_period.setRange(2, 120)
        self.atr_period.setValue(14)
        self.atr_period.setToolTip("ATR 计算周期")
        self.atr_stop_mult = QDoubleSpinBox()
        self.atr_stop_mult.setRange(0.5, 10.0)
        self.atr_stop_mult.setSingleStep(0.5)
        self.atr_stop_mult.setValue(2.0)
        self.atr_stop_mult.setToolTip("初始 ATR 止损 = 买入价 - N × ATR")
        self.atr_trail_mult = QDoubleSpinBox()
        self.atr_trail_mult.setRange(0.5, 20.0)
        self.atr_trail_mult.setSingleStep(0.5)
        self.atr_trail_mult.setValue(3.0)
        self.atr_trail_mult.setToolTip("追踪止盈 = 持仓最高价 - N × ATR(最新)")
        # --------------------------------------------
        self.save_plot = QCheckBox("导出净值/回撤 PNG")
        self.save_plot.setChecked(True)
        self.run_button = QPushButton("开始本地回测")
        self.run_button.clicked.connect(self.run_backtest)
        settings.addWidget(QLabel("行情口径"), 0, 0)
        settings.addWidget(self.fq_combo, 0, 1)
        settings.addWidget(QLabel("周期"), 0, 2)
        settings.addWidget(self.period_combo, 0, 3)
        settings.addWidget(QLabel("开始日期"), 0, 4)
        settings.addWidget(self.start_edit, 0, 5)
        settings.addWidget(QLabel("结束日期"), 0, 6)
        settings.addWidget(self.end_edit, 0, 7)
        settings.addWidget(QLabel("初始资金"), 1, 0)
        settings.addWidget(self.cash_edit, 1, 1)
        settings.addWidget(QLabel("MA 快线"), 1, 2)
        settings.addWidget(self.fast_period, 1, 3)
        settings.addWidget(QLabel("MA 慢线"), 1, 4)
        settings.addWidget(self.slow_period, 1, 5)
        settings.addWidget(self.enable_atr, 2, 0)
        settings.addWidget(QLabel("ATR 周期"), 2, 1)
        settings.addWidget(self.atr_period, 2, 2)
        settings.addWidget(QLabel("ATR 止损倍数"), 2, 3)
        settings.addWidget(self.atr_stop_mult, 2, 4)
        settings.addWidget(QLabel("ATR 追踪倍数"), 2, 5)
        settings.addWidget(self.atr_trail_mult, 2, 6)
        settings.addWidget(self.save_plot, 3, 0, 1, 2)
        settings.addWidget(self.run_button, 3, 2)
        self.rules_label = QLabel(
            "固定规则：手续费万三｜滑点 0.1%｜单票≤10%｜总仓≤80%｜买入价下方 5% 止损（底线）｜"
            "ATR 启用后叠加初始止损与追踪止盈｜"
            "A 股 T+1｜涨停不买、跌停不卖｜T 日收盘判定、T+1 开盘成交｜期末不强制平仓｜"
            "分钟回测：信号在分钟 K 线收盘生成、下一分钟开盘成交，T+1 同样生效"
        )
        self.rules_label.setWordWrap(True)
        self.rules_label.setToolTip(
            "规则说明：手续费按成交额万三收取，滑点为 0.1%；单票仓位不超过 10%，总仓位不超过 80%；\n"
            "固定止损为跌破实际买入价 5%；启用 ATR 后增加：初始 ATR 止损（买入价-N×ATR）和"
            "ATR 追踪止盈（持仓最高价-N×ATR，ATR 每日重算）；\n"
            "遵守 A 股 T+1，涨停不买、跌停不卖，停牌禁开仓；T 日收盘确认信号，下一交易日开盘成交；"
            "回测结束保留未平仓记录，不强制平仓。"
        )
        settings.addWidget(self.rules_label, 4, 0, 1, 7)
        layout.addWidget(settings_group)

        # ---- V1.1 Step F3：参数寻优（批量遍历 + 汇总表） ----
        opt_group = QGroupBox("参数寻优（批量遍历，仅使用本地行情，不联网）")
        opt_grid = QGridLayout(opt_group)
        opt_grid.addWidget(QLabel("参数"), 0, 0)
        opt_grid.addWidget(QLabel("最小值"), 0, 1)
        opt_grid.addWidget(QLabel("最大值"), 0, 2)
        opt_grid.addWidget(QLabel("步长"), 0, 3)
        opt_grid.addWidget(QLabel("说明"), 0, 4, 1, 3)

        self.opt_fast_min = QSpinBox(); self.opt_fast_min.setRange(2, 60); self.opt_fast_min.setValue(3)
        self.opt_fast_max = QSpinBox(); self.opt_fast_max.setRange(2, 60); self.opt_fast_max.setValue(7)
        self.opt_fast_step = QSpinBox(); self.opt_fast_step.setRange(1, 20); self.opt_fast_step.setValue(2)
        opt_grid.addWidget(QLabel("MA 快线"), 1, 0)
        opt_grid.addWidget(self.opt_fast_min, 1, 1)
        opt_grid.addWidget(self.opt_fast_max, 1, 2)
        opt_grid.addWidget(self.opt_fast_step, 1, 3)
        opt_grid.addWidget(QLabel("fast_period"), 1, 4, 1, 3)

        self.opt_slow_min = QSpinBox(); self.opt_slow_min.setRange(3, 120); self.opt_slow_min.setValue(10)
        self.opt_slow_max = QSpinBox(); self.opt_slow_max.setRange(3, 120); self.opt_slow_max.setValue(30)
        self.opt_slow_step = QSpinBox(); self.opt_slow_step.setRange(1, 30); self.opt_slow_step.setValue(5)
        opt_grid.addWidget(QLabel("MA 慢线"), 2, 0)
        opt_grid.addWidget(self.opt_slow_min, 2, 1)
        opt_grid.addWidget(self.opt_slow_max, 2, 2)
        opt_grid.addWidget(self.opt_slow_step, 2, 3)
        opt_grid.addWidget(QLabel("slow_period，需大于快线"), 2, 4, 1, 3)

        self.opt_stop_min = QDoubleSpinBox(); self.opt_stop_min.setRange(0.01, 0.30); self.opt_stop_min.setSingleStep(0.01); self.opt_stop_min.setValue(0.03)
        self.opt_stop_max = QDoubleSpinBox(); self.opt_stop_max.setRange(0.01, 0.30); self.opt_stop_max.setSingleStep(0.01); self.opt_stop_max.setValue(0.08)
        self.opt_stop_step = QDoubleSpinBox(); self.opt_stop_step.setRange(0.005, 0.10); self.opt_stop_step.setSingleStep(0.005); self.opt_stop_step.setValue(0.01)
        opt_grid.addWidget(QLabel("止损比例"), 3, 0)
        opt_grid.addWidget(self.opt_stop_min, 3, 1)
        opt_grid.addWidget(self.opt_stop_max, 3, 2)
        opt_grid.addWidget(self.opt_stop_step, 3, 3)
        opt_grid.addWidget(QLabel("stop_loss，如 0.05 = 5%"), 3, 4, 1, 3)

        self.opt_atr = QCheckBox("同时遍历 ATR 倍数")
        self.opt_atr.setToolTip("勾选后会额外遍历 ATR 止损/追踪倍数，组合数会倍增")
        self.opt_atr_stop_min = QDoubleSpinBox(); self.opt_atr_stop_min.setRange(1.0, 6.0); self.opt_atr_stop_min.setSingleStep(0.5); self.opt_atr_stop_min.setValue(1.5)
        self.opt_atr_stop_max = QDoubleSpinBox(); self.opt_atr_stop_max.setRange(1.0, 6.0); self.opt_atr_stop_max.setSingleStep(0.5); self.opt_atr_stop_max.setValue(3.0)
        self.opt_atr_stop_step = QDoubleSpinBox(); self.opt_atr_stop_step.setRange(0.5, 2.0); self.opt_atr_stop_step.setSingleStep(0.5); self.opt_atr_stop_step.setValue(0.5)
        self.opt_atr_trail_min = QDoubleSpinBox(); self.opt_atr_trail_min.setRange(1.0, 10.0); self.opt_atr_trail_min.setSingleStep(0.5); self.opt_atr_trail_min.setValue(2.0)
        self.opt_atr_trail_max = QDoubleSpinBox(); self.opt_atr_trail_max.setRange(1.0, 10.0); self.opt_atr_trail_max.setSingleStep(0.5); self.opt_atr_trail_max.setValue(4.0)
        self.opt_atr_trail_step = QDoubleSpinBox(); self.opt_atr_trail_step.setRange(0.5, 2.0); self.opt_atr_trail_step.setSingleStep(0.5); self.opt_atr_trail_step.setValue(1.0)
        opt_grid.addWidget(self.opt_atr, 4, 0)
        opt_grid.addWidget(QLabel("ATR止损倍数"), 4, 1)
        opt_grid.addWidget(self.opt_atr_stop_min, 4, 2)
        opt_grid.addWidget(self.opt_atr_stop_max, 4, 3)
        opt_grid.addWidget(self.opt_atr_stop_step, 4, 4)
        opt_grid.addWidget(QLabel("ATR追踪倍数"), 5, 1)
        opt_grid.addWidget(self.opt_atr_trail_min, 5, 2)
        opt_grid.addWidget(self.opt_atr_trail_max, 5, 3)
        opt_grid.addWidget(self.opt_atr_trail_step, 5, 4)

        self.opt_sort = QComboBox()
        self.opt_sort.addItem("夏普比率 降序", ("sharpe", False))
        self.opt_sort.addItem("最大回撤 升序", ("max_drawdown", True))
        self.opt_sort.addItem("年化收益 降序", ("annual_return", False))
        self.opt_sort.addItem("总收益 降序", ("total_return", False))
        self.opt_max_runs = QSpinBox()
        self.opt_max_runs.setRange(1, 5000)
        self.opt_max_runs.setValue(60)
        self.opt_max_runs.setToolTip("安全上限：超过此数量的组合会被截断，避免回测过久")
        # ---- V1.2 Step G1：样本外验证 ----
        self.opt_oos = QCheckBox("样本外验证")
        self.opt_oos.setToolTip("勾选后按日期切分：前半段寻优，后半段用最优参数回测，检验过拟合")
        self.opt_oos_ratio = QDoubleSpinBox()
        self.opt_oos_ratio.setRange(0.5, 0.9)
        self.opt_oos_ratio.setSingleStep(0.05)
        self.opt_oos_ratio.setValue(0.7)
        self.opt_oos_ratio.setToolTip("样本内区间占比，剩余部分为样本外验证区间")
        self.opt_heat_x = QComboBox()
        self.opt_heat_x.addItem("快线", "fast_period")
        self.opt_heat_x.addItem("慢线", "slow_period")
        self.opt_heat_x.addItem("止损", "stop_loss")
        self.opt_heat_y = QComboBox()
        self.opt_heat_y.addItem("慢线", "slow_period")
        self.opt_heat_y.addItem("快线", "fast_period")
        self.opt_heat_y.addItem("止损", "stop_loss")
        # -----------------------------------
        self.opt_run_button = QPushButton("开始参数寻优")
        self.opt_run_button.clicked.connect(self.run_optimization)
        opt_grid.addWidget(QLabel("排序指标"), 6, 0)
        opt_grid.addWidget(self.opt_sort, 6, 1)
        opt_grid.addWidget(QLabel("最大组合数"), 6, 2)
        opt_grid.addWidget(self.opt_max_runs, 6, 3)
        opt_grid.addWidget(self.opt_run_button, 6, 4)
        opt_grid.addWidget(self.opt_oos, 7, 0)
        opt_grid.addWidget(QLabel("样本内比例"), 7, 1)
        opt_grid.addWidget(self.opt_oos_ratio, 7, 2)
        opt_grid.addWidget(QLabel("热力图X"), 7, 3)
        opt_grid.addWidget(self.opt_heat_x, 7, 4)
        opt_grid.addWidget(QLabel("热力图Y"), 7, 5)
        opt_grid.addWidget(self.opt_heat_y, 7, 6)
        layout.addWidget(opt_group)

        self.opt_result_label = QLabel("尚未运行参数寻优")
        layout.addWidget(self.opt_result_label)
        self.opt_table = QTableWidget()
        # 表头显示名（短）
        self.opt_headers = [
            "快线", "慢线", "止损", "ATR",
            "ATR止损倍数", "ATR追踪倍数", "年化收益", "夏普",
            "最大回撤", "总收益", "交易次数",
        ]
        # 对应的 DataFrame 列名（与 param_grid / run_parameter_grid 输出一致）
        self.opt_columns = [
            "fast_period", "slow_period", "stop_loss", "enable_atr",
            "atr_stop_multiplier", "atr_trailing_multiplier", "annual_return", "sharpe",
            "max_drawdown", "total_return", "trades_closed",
        ]
        prepare_table(self.opt_table, self.opt_headers)
        self.opt_table.setFixedHeight(200)
        self.opt_empty_label = QLabel("尚未运行参数寻优")
        self.opt_empty_label.setAlignment(Qt.AlignCenter)
        self.opt_empty_label.setStyleSheet("color: #777;")
        self.opt_stack = QStackedWidget()
        self.opt_stack.addWidget(self.opt_table)
        self.opt_stack.addWidget(self.opt_empty_label)
        self.opt_stack.setCurrentWidget(self.opt_empty_label)
        self.opt_stack.setFixedHeight(220)
        layout.addWidget(self.opt_stack)
        # ---- V1.2 Step G1：样本外验证对比 + 热力图 ----
        self.oos_result_label = QLabel("")
        self.oos_result_label.setWordWrap(True)
        self.oos_result_label.hide()
        layout.addWidget(self.oos_result_label)
        self.opt_heatmap_widget = QSvgWidget()
        self.opt_heatmap_widget.setFixedHeight(320)
        self.opt_heatmap_widget.hide()
        layout.addWidget(self.opt_heatmap_widget)
        # ----------------------------------------------------------

        # ---- V1.2 Step G4：多策略对比 ----
        cmp_group = QGroupBox("多策略对比")
        cmp_layout = QVBoxLayout(cmp_group)
        cmp_tip = QLabel("每行一个策略：名称,快线,慢线,止损,ATR开关(0/1)，ATR 开关省略或为 0 表示不启用")
        cmp_tip.setStyleSheet("color: #555;")
        cmp_layout.addWidget(cmp_tip)
        self.cmp_edit = QPlainTextEdit()
        self.cmp_edit.setPlaceholderText(
            "MA5/20,5,20,0.05,0\nMA3/10,3,10,0.03,0\nMA8/30+ATR,8,30,0.05,1"
        )
        self.cmp_edit.setFixedHeight(70)
        cmp_layout.addWidget(self.cmp_edit)
        self.cmp_run_button = QPushButton("开始策略对比")
        self.cmp_run_button.clicked.connect(self.run_comparison)
        cmp_layout.addWidget(self.cmp_run_button)
        layout.addWidget(cmp_group)

        self.cmp_result_label = QLabel("尚未运行策略对比")
        layout.addWidget(self.cmp_result_label)
        self.cmp_table = QTableWidget()
        self.cmp_empty_label = QLabel("尚未运行策略对比")
        self.cmp_stack = QStackedWidget()
        self.cmp_stack.addWidget(self.cmp_empty_label)
        self.cmp_stack.addWidget(self.cmp_table)
        self.cmp_stack.setCurrentWidget(self.cmp_empty_label)
        self.cmp_stack.setFixedHeight(180)
        layout.addWidget(self.cmp_stack)
        self.cmp_chart_widget = QSvgWidget()
        self.cmp_chart_widget.setFixedHeight(340)
        self.cmp_chart_widget.hide()
        layout.addWidget(self.cmp_chart_widget)
        # -----------------------------------

        self.result_label = QLabel("尚未运行回测")
        layout.addWidget(self.result_label)
        self.summary_table = QTableWidget()
        prepare_table(self.summary_table, ["指标", "数值"])
        self.summary_table.setFixedHeight(210)
        self.summary_empty_label = QLabel("尚未运行回测")
        self.summary_empty_label.setAlignment(Qt.AlignCenter)
        self.summary_empty_label.setStyleSheet("color: #777;")
        self.summary_stack = QStackedWidget()
        self.summary_stack.addWidget(self.summary_table)
        self.summary_stack.addWidget(self.summary_empty_label)
        self.summary_stack.setCurrentWidget(self.summary_empty_label)
        self.summary_stack.setFixedHeight(230)

        self.figure = Figure(figsize=(9, 3), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.equity_axis = self.figure.add_subplot(211)
        self.drawdown_axis = self.figure.add_subplot(212, sharex=self.equity_axis)
        self.figure.text(0.99, 0.01, DISCLAIMER, ha="right", fontsize=8, color="gray")
        self.chart_empty_label = QLabel("尚未运行回测")
        self.chart_empty_label.setAlignment(Qt.AlignCenter)
        self.chart_empty_label.setStyleSheet("color: #777;")
        self.chart_stack = QStackedWidget()
        self.chart_stack.addWidget(self.canvas)
        self.chart_stack.addWidget(self.chart_empty_label)
        self.chart_stack.setCurrentWidget(self.chart_empty_label)
        self.chart_stack.setFixedHeight(230)
        result_visual_layout = QHBoxLayout()
        result_visual_layout.addWidget(self.summary_stack, 1)
        result_visual_layout.addWidget(self.chart_stack, 2)
        layout.addLayout(result_visual_layout)

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("导出目录"))
        self.output_path = QLineEdit()
        self.output_path.setReadOnly(True)
        self.export_button = QPushButton("导出 PNG/CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_files)
        self.open_output_button = QPushButton("打开目录")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self.open_output_directory)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.export_button)
        output_layout.addWidget(self.open_output_button)
        layout.addLayout(output_layout)

        layout.addWidget(QLabel("交易明细（open 表示期末仍持仓）"))
        self.trade_table = QTableWidget()
        self.trade_headers = [
            "code", "buy_date", "buy_price", "size", "sell_date", "sell_price",
            "commission", "pnl", "return_pct", "status",
        ]
        prepare_table(self.trade_table, self.trade_headers)
        self.trade_empty_label = QLabel("暂无交易记录")
        self.trade_empty_label.setAlignment(Qt.AlignCenter)
        self.trade_empty_label.setStyleSheet("color: #777; padding: 24px;")
        self.trade_stack = QStackedWidget()
        self.trade_stack.addWidget(self.trade_table)
        self.trade_stack.addWidget(self.trade_empty_label)
        self.trade_stack.setCurrentWidget(self.trade_empty_label)
        layout.addWidget(self.trade_stack)

    def use_selection_codes(self):
        raw_codes = "\n".join(
            str(row.get("code") or "") for row in self.window.formula_page.selection_rows
        )
        # 与回测入口共用代码规范化和去重规则，保持原有选股顺序。
        codes = parse_codes(raw_codes)
        if not codes:
            QMessageBox.warning(self, "没有选股结果", "请先在公式选股页执行选股")
            return
        self.codes_edit.setPlainText("\n".join(codes))
        self.window.tabs.setCurrentWidget(self)
        self.window.statusBar().showMessage("已填入 %d 只选股结果（已去重）" % len(codes), 5000)

    def run_backtest(self):
        try:
            codes = parse_codes(self.codes_edit.toPlainText())
            if not codes:
                raise ValueError("请输入至少一个回测股票代码")
            start = self.start_edit.date().toString("yyyy-MM-dd")
            end = self.end_edit.date().toString("yyyy-MM-dd")
            validate_date_range(start, end)
            if self.fast_period.value() >= self.slow_period.value():
                raise ValueError("MA 快线周期必须小于慢线周期")
        except ValueError as exc:
            QMessageBox.warning(self, "无法回测", str(exc))
            return
        config = BacktestConfig(
            start_date=start,
            end_date=end,
            initial_cash=self.cash_edit.value(),
            fq=self.fq_combo.currentData(),
            period=self.period_combo.currentData(),
            fast_period=self.fast_period.value(),
            slow_period=self.slow_period.value(),
            enable_atr=self.enable_atr.isChecked(),
            atr_period=self.atr_period.value(),
            atr_stop_multiplier=self.atr_stop_mult.value(),
            atr_trailing_multiplier=self.atr_trail_mult.value(),
            save_plot=self.save_plot.isChecked(),
        )

        def task(progress):
            progress.emit({"total": 0, "text": "正在本地回测，不联网"})
            return run_backtest(codes=codes, config=config, export=True, save_record=True)

        self.window.start_task(task, self.backtest_finished, "正在运行本地回测")

    @staticmethod
    def _frange(start, stop, step):
        """生成含首尾的浮点区间，避免浮点累加误差。"""
        values = []
        current = float(start)
        while current <= float(stop) + 1e-9:
            values.append(round(current, 6))
            current += float(step)
        return values

    def run_optimization(self):
        """根据界面范围构建参数网格，后台批量回测并汇总排序。"""
        try:
            codes = parse_codes(self.codes_edit.toPlainText())
            if not codes:
                raise ValueError("请输入至少一个回测股票代码")
            start = self.start_edit.date().toString("yyyy-MM-dd")
            end = self.end_edit.date().toString("yyyy-MM-dd")
            validate_date_range(start, end)
            fast_min, fast_max, fast_step = (
                self.opt_fast_min.value(), self.opt_fast_max.value(), self.opt_fast_step.value()
            )
            slow_min, slow_max, slow_step = (
                self.opt_slow_min.value(), self.opt_slow_max.value(), self.opt_slow_step.value()
            )
            if fast_min > fast_max or slow_min > slow_max:
                raise ValueError("最小值不能大于最大值")
            if fast_max >= slow_min:
                raise ValueError("快线最大值必须小于慢线最小值，否则慢线始终不大于快线")
            stop_min, stop_max, stop_step = (
                self.opt_stop_min.value(), self.opt_stop_max.value(), self.opt_stop_step.value()
            )
            if stop_min > stop_max:
                raise ValueError("止损最小值不能大于最大值")
        except ValueError as exc:
            QMessageBox.warning(self, "无法寻优", str(exc))
            return

        fast_values = list(range(fast_min, fast_max + 1, fast_step))
        slow_values = [s for s in range(slow_min, slow_max + 1, slow_step) if s > fast_min]
        stop_values = self._frange(stop_min, stop_max, stop_step)
        param_grid = {
            "fast_period": fast_values,
            "slow_period": slow_values,
            "stop_loss": stop_values,
            "enable_atr": [False],
        }
        if self.opt_atr.isChecked():
            param_grid["enable_atr"] = [True]
            param_grid["atr_stop_multiplier"] = self._frange(
                self.opt_atr_stop_min.value(), self.opt_atr_stop_max.value(), self.opt_atr_stop_step.value()
            )
            param_grid["atr_trailing_multiplier"] = self._frange(
                self.opt_atr_trail_min.value(), self.opt_atr_trail_max.value(), self.opt_atr_trail_step.value()
            )

        max_runs = self.opt_max_runs.value()
        base_config = BacktestConfig(
            start_date=start, end_date=end, fq=self.fq_combo.currentData(),
            period=self.period_combo.currentData(), save_plot=False,
        )
        sort_data = self.opt_sort.currentData()
        use_oos = self.opt_oos.isChecked()
        oos_ratio = self.opt_oos_ratio.value()
        heat_x = self.opt_heat_x.currentData()
        heat_y = self.opt_heat_y.currentData()
        sort_key, _ascending = sort_data

        def task(progress):
            def progress_cb(current, total, overrides):
                progress.emit({
                    "current": current, "total": total,
                    "text": "寻优 %d/%d" % (current, total),
                })
            if use_oos:
                result = run_out_of_sample_validation(
                    codes, base_config, param_grid,
                    in_sample_ratio=oos_ratio, metric=sort_key,
                    max_runs=max_runs, progress_cb=progress_cb,
                )
                frame = result["in_sample_frame"].sort_values(
                    sort_key, ascending=(sort_key in ("max_drawdown",)), na_position="last"
                )
                result["in_sample_frame"] = frame
                result["heatmap_svg"] = heatmap_svg(frame, heat_x, heat_y, sort_key)
                return result
            frame = run_parameter_grid(
                codes, base_config, param_grid, max_runs=max_runs, progress_cb=progress_cb
            )
            sort_key_local, ascending = sort_data
            if sort_key_local in frame.columns:
                frame = frame.sort_values(sort_key_local, ascending=ascending, na_position="last")
            return frame

        busy_text = "正在样本外验证寻优，不联网" if use_oos else "正在参数寻优，不联网"
        self.window.start_task(task, self.optimization_finished, busy_text)

    def optimization_finished(self, result):
        """展示参数寻优汇总表；支持普通寻优(DataFrame)与样本外验证(dict)两种结果。"""
        # 先隐藏样本外专属区域，普通寻优时不显示
        self.oos_result_label.hide()
        self.opt_heatmap_widget.hide()

        # 样本外验证返回 dict，普通寻优返回 DataFrame
        if isinstance(result, dict):
            frame = result.get("in_sample_frame")
            oos_summary = result.get("out_of_sample")
            best_params = result.get("best_params", {})
            in_best = result.get("in_sample_best", {})
            split_date = result.get("split_date", "")
        else:
            frame = result
            oos_summary = None

        if frame is None or frame.empty:
            self.opt_result_label.setText("参数寻优无有效结果，请扩大参数范围或检查行情")
            self.opt_stack.setCurrentWidget(self.opt_empty_label)
            return
        self.opt_table.setRowCount(len(frame))
        for row_index, (_, row) in enumerate(frame.iterrows()):
            for col_index, key in enumerate(self.opt_columns):
                value = row.get(key) if key in frame.columns else None
                if pd.isna(value):
                    text = ""
                elif key in ("stop_loss", "annual_return", "max_drawdown", "total_return"):
                    text = "%.2f%%" % (float(value) * 100)
                elif key in ("sharpe", "atr_stop_multiplier", "atr_trailing_multiplier"):
                    text = "%.3f" % float(value)
                elif key == "enable_atr":
                    text = "是" if bool(value) else "否"
                else:
                    text = str(int(value))
                set_cell(self.opt_table, row_index, col_index, text, numeric=True)
        self.opt_stack.setCurrentWidget(self.opt_table)
        self.opt_result_label.setText(
            "参数寻优完成：共 %d 组，按 %s 排序。%s"
            % (len(frame), self.opt_sort.currentText(), DISCLAIMER)
        )

        # 样本外验证：展示对比与热力图
        if oos_summary is not None:
            metric = result.get("metric", "sharpe")
            in_val = in_best.get(metric)
            out_val = oos_summary.get(metric)
            in_dd = in_best.get("max_drawdown")
            out_dd = oos_summary.get("max_drawdown")
            params_text = "、".join("%s=%s" % (k, v) for k, v in best_params.items())
            oos_text = (
                "样本外验证：切分日 %s，样本内 %s ~ %s，样本外 %s ~ %s\n"
                "最优参数：%s\n"
                "样本内  %s=%.4f  年化=%.2f%%  最大回撤=%.2f%%\n"
                "样本外  %s=%.4f  年化=%.2f%%  最大回撤=%.2f%%\n"
                "%s"
            ) % (
                split_date,
                result["in_sample_period"][0], result["in_sample_period"][1],
                result["out_of_sample_period"][0], result["out_of_sample_period"][1],
                params_text,
                metric, float(in_val or 0), float(in_best.get("annual_return") or 0) * 100,
                float(in_dd or 0) * 100,
                metric, float(out_val or 0), float(oos_summary.get("annual_return") or 0) * 100,
                float(out_dd or 0) * 100,
                DISCLAIMER,
            )
            self.oos_result_label.setText(oos_text)
            self.oos_result_label.show()
            # 热力图
            svg = result.get("heatmap_svg", "")
            if svg and "<svg" in svg:
                from PyQt5.QtCore import QByteArray
                self.opt_heatmap_widget.load(QByteArray(svg.encode("utf-8")))
                self.opt_heatmap_widget.show()

        self.window.statusBar().showMessage("参数寻优完成，共 %d 组" % len(frame), 10000)

    def run_comparison(self):
        """解析多策略配置文本，后台对比回测。"""
        try:
            codes = parse_codes(self.codes_edit.toPlainText())
            if not codes:
                raise ValueError("请输入至少一个回测股票代码")
            start = self.start_edit.date().toString("yyyy-MM-dd")
            end = self.end_edit.date().toString("yyyy-MM-dd")
            validate_date_range(start, end)
            lines = [ln.strip() for ln in self.cmp_edit.toPlainText().splitlines() if ln.strip()]
            if not lines:
                raise ValueError("请输入至少一个策略配置（每行：名称,快线,慢线,止损[,ATR开关]）")
            named = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    raise ValueError("策略配置格式错误：%s（应为 名称,快线,慢线,止损[,ATR]）" % line)
                name, fast, slow, stop = parts[0], int(parts[1]), int(parts[2]), float(parts[3])
                atr_on = len(parts) > 4 and parts[4] in ("1", "true", "True", "是")
                if fast >= slow:
                    raise ValueError("策略 %s：快线必须小于慢线" % name)
                cfg = BacktestConfig(
                    start_date=start, end_date=end,
                    fq=self.fq_combo.currentData(), period=self.period_combo.currentData(),
                    fast_period=fast, slow_period=slow, stop_loss=stop,
                    enable_atr=atr_on, atr_stop_multiplier=2.0, atr_trailing_multiplier=3.0,
                    save_plot=False, name=name,
                )
                named.append((name, cfg))
        except ValueError as exc:
            QMessageBox.warning(self, "无法对比", str(exc))
            return

        def task(progress):
            # 在 worker 线程内导入，避免闭包变量跨线程访问导致的崩溃
            from core.optimize import run_strategy_comparison
            def progress_cb(cur, total, name):
                progress.emit({"current": cur, "total": total, "text": "对比 %d/%d %s" % (cur, total, name)})
            return run_strategy_comparison(codes, named, progress_cb=progress_cb)

        self.window.start_task(task, self.comparison_finished, "正在多策略对比，不联网")

    def comparison_finished(self, result):
        """展示多策略对比汇总表与净值叠加图。"""
        table = result.get("summary_table")
        if table is None or table.empty:
            self.cmp_result_label.setText("策略对比无有效结果")
            self.cmp_stack.setCurrentWidget(self.cmp_empty_label)
            self.cmp_chart_widget.hide()
            return
        headers = list(table.columns)
        self.cmp_table.setColumnCount(len(headers))
        self.cmp_table.setHorizontalHeaderLabels(headers)
        self.cmp_table.setRowCount(len(table))
        for row_index in range(len(table)):
            for col_index, key in enumerate(headers):
                value = table.iloc[row_index][key]
                if pd.isna(value):
                    text = "—"
                elif key in ("总收益率", "年化收益率", "最大回撤", "胜率"):
                    text = "%.2f%%" % (float(value) * 100)
                elif key in ("夏普比率", "盈利因子"):
                    text = "%.3f" % float(value)
                elif key == "已平仓交易":
                    text = str(int(value))
                else:
                    text = str(value)
                set_cell(self.cmp_table, row_index, col_index, text, numeric=(col_index > 0))
        self.cmp_stack.setCurrentWidget(self.cmp_table)
        self.cmp_result_label.setText("策略对比完成：共 %d 个策略。%s" % (len(table), DISCLAIMER))

        svg = result.get("equity_overlay_svg", "")
        if svg and "<svg" in svg:
            try:
                from PyQt5.QtCore import QByteArray
                self.cmp_chart_widget.load(QByteArray(svg.encode("utf-8")))
                self.cmp_chart_widget.show()
            except Exception:
                self.cmp_chart_widget.hide()
        else:
            self.cmp_chart_widget.hide()
        self.window.statusBar().showMessage("策略对比完成，共 %d 个策略" % len(table), 10000)

    @staticmethod
    def format_metric(value, kind):
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "—"
        if kind == "percent":
            return "%.2f%%" % (value * 100)
        if kind == "money":
            return "%.2f" % value
        if kind == "integer":
            return str(int(value))
        return "%.4f" % value

    def draw_curves(self, curves):
        """仅展示回测结果中的净值和回撤，不重新计算绩效。"""
        self.equity_axis.clear()
        self.drawdown_axis.clear()
        if curves.empty:
            self.chart_empty_label.setText("暂无净值数据")
            self.chart_stack.setCurrentWidget(self.chart_empty_label)
            return
        frame = curves.copy()
        frame.index = pd.to_datetime(frame.index)
        self.equity_axis.plot(
            frame.index, frame["net_value"].astype(float).to_numpy(),
            color="black", linewidth=1, label="净值",
        )
        self.drawdown_axis.fill_between(
            frame.index, frame["drawdown"].astype(float).to_numpy() * 100,
            0, color="gray", alpha=0.6,
        )
        self.equity_axis.set_ylabel("净值")
        self.drawdown_axis.set_ylabel("回撤（%）")
        self.equity_axis.grid(alpha=0.2)
        self.drawdown_axis.grid(alpha=0.2)
        self.equity_axis.tick_params(labelbottom=False)
        self.figure.tight_layout(rect=(0, 0.04, 1, 1))
        self.canvas.draw_idle()
        self.chart_stack.setCurrentWidget(self.canvas)

    def export_files(self):
        """把当前展示结果保存到本次回测目录，覆盖同名展示文件。"""
        if not self.last_result:
            return
        output_dir = Path(self.last_result.get("output_dir") or "")
        if not output_dir.is_dir():
            QMessageBox.warning(self, "导出目录不可用", "本次回测导出目录不存在，请重新运行回测")
            return
        try:
            self.last_result["curves"].to_csv(
                output_dir / "equity_curve.csv", encoding="utf-8-sig", index_label="date"
            )
            self.last_result["trades"].to_csv(
                output_dir / "trades.csv", encoding="utf-8-sig", index=False
            )
            self.figure.savefig(str(output_dir / "curves.png"))
        except Exception as exc:
            self.window.show_error("导出回测文件失败", traceback.format_exc(), str(exc))
            return
        self.window.statusBar().showMessage("PNG/CSV 已保存到 " + str(output_dir), 10000)

    def backtest_finished(self, result):
        self.last_result = result
        summary = result["summary"]
        self.summary_table.setRowCount(len(self.SUMMARY_FIELDS))
        for row_index, (label, key, kind) in enumerate(self.SUMMARY_FIELDS):
            set_cell(self.summary_table, row_index, 0, label)
            set_cell(
                self.summary_table, row_index, 1,
                self.format_metric(summary.get(key), kind), numeric=True,
            )
        self.summary_stack.setCurrentWidget(self.summary_table)
        trades = result["trades"]
        self.trade_table.setRowCount(len(trades))
        for row_index, (_, trade) in enumerate(trades.iterrows()):
            for column, key in enumerate(self.trade_headers):
                value = trade.get(key) if key in trades.columns else None
                if key in ("buy_price", "sell_price") and pd.notna(value):
                    value = "%.3f" % float(value)
                elif key in ("commission", "pnl") and pd.notna(value):
                    value = "%.2f" % float(value)
                elif key == "return_pct" and pd.notna(value):
                    value = "%.2f%%" % (float(value) * 100)
                set_cell(
                    self.trade_table, row_index, column, value,
                    numeric=key in ("buy_price", "size", "sell_price", "commission", "pnl", "return_pct"),
                )
        self.trade_stack.setCurrentWidget(self.trade_table if len(trades) else self.trade_empty_label)
        output_dir = result.get("output_dir") or ""
        self.output_path.setText(output_dir)
        self.export_button.setEnabled(bool(output_dir))
        self.open_output_button.setEnabled(bool(output_dir))
        self.draw_curves(result["curves"])
        self.result_label.setText(
            "回测完成：加载 %d 只，跳过 %d 只，规则拦截/警告 %d 条。%s"
            % (
                len(summary["codes_loaded"]), len(summary["codes_skipped"]),
                len(result["warnings"]), DISCLAIMER,
            )
        )
        message = "回测完成"
        if output_dir:
            message += "，结果已保存到 " + output_dir
        self.window.statusBar().showMessage(message, 10000)

    def open_output_directory(self):
        path = self.output_path.text()
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))


class SmartBetaPage(QWidget):
    """Smart Beta 多策略研究面板（I2-alt-5：滑块、多组合与敏感度）。"""

    PORTFOLIO_LABELS = {
        "low_price": "低价组合", "low_vol": "低波组合",
        "dual": "双因子 50:50", "three_equal": "三因子等权 33/33/34",
        "three_conservative": "三因子保守 40/40/20",
        "three_aggressive": "三因子进取 20/20/60",
        "three_custom": "三因子自定义", "low_pe": "低 PE 组合",
        "low_pb": "低 PB 组合", "four_factor": "四因子等权 25%",
        "five_factor": "五因子等权 20%", "benchmark": "池内等权基准",
    }
    PRIMARY_PORTFOLIO = {
        "dual": "dual", "three": "three_equal",
        "custom": "three_custom", "fundamental": "five_factor",
    }
    SENSITIVITY_PORTFOLIOS = {
        "dual": ("dual",),
        "three": ("three_equal", "three_conservative", "three_aggressive"),
        "custom": ("three_custom",),
        "fundamental": ("low_pe", "low_pb", "four_factor", "five_factor"),
    }

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.last_result = None
        self.last_exposure = None
        self.last_export_paths = None
        self.last_sensitivity = None
        layout = QVBoxLayout(self)

        intro = QLabel(
            "低价、低波、动量与低 PE/PB 固定暴露研究，支持三因子自定义比例、"
            "多组合对比和暴露比例敏感度。只使用本地行情与估值，不接券商、不下单、不上传云端。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        settings_group = QGroupBox("固定暴露参数")
        settings = QGridLayout(settings_group)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("双因子 50:50（原有）", "dual")
        self.strategy_combo.addItem("三因子预设对比", "three")
        self.strategy_combo.addItem("三因子自定义比例", "custom")
        self.strategy_combo.addItem("低 PE/PB 与四/五因子对比", "fundamental")
        self.strategy_combo.currentIndexChanged.connect(self.update_strategy_controls)
        self.start_edit = make_date_edit(QDate(2024, 5, 1))
        self.end_edit = make_date_edit(QDate.currentDate())
        self.pct_edit = QDoubleSpinBox()
        self.pct_edit.setRange(1.0, 100.0)
        self.pct_edit.setDecimals(1)
        self.pct_edit.setSingleStep(1.0)
        self.pct_edit.setValue(20.0)
        self.pct_edit.setSuffix(" %")
        self.pct_edit.setToolTip("低价与低波分别选取评分最低的前 N% 股票，组内等权")
        self.vol_window_edit = QSpinBox()
        self.vol_window_edit.setRange(20, 250)
        self.vol_window_edit.setValue(60)
        self.vol_window_edit.setSuffix(" 个交易日")
        self.vol_window_edit.setToolTip("低波评分使用过去 N 个交易日的日收益率标准差")
        self.momentum_window_edit = QSpinBox()
        self.momentum_window_edit.setRange(60, 500)
        self.momentum_window_edit.setValue(252)
        self.momentum_window_edit.setSuffix(" 日")
        self.momentum_skip_edit = QSpinBox()
        self.momentum_skip_edit.setRange(0, 60)
        self.momentum_skip_edit.setValue(21)
        self.momentum_skip_edit.setSuffix(" 日")
        # I2-alt-12：五因子并集分支的动量最低权重，默认 10%，0 表示关闭
        self.momentum_floor_edit = QDoubleSpinBox()
        self.momentum_floor_edit.setRange(0.0, 50.0)
        self.momentum_floor_edit.setDecimals(1)
        self.momentum_floor_edit.setSingleStep(1.0)
        self.momentum_floor_edit.setValue(10.0)
        self.momentum_floor_edit.setSuffix(" %")
        self.momentum_floor_edit.setToolTip(
            "仅作用于五因子并集分支：强制动量持仓达到该权重下限；0 表示关闭（等同旧并集规则）")
        self.fq_combo = QComboBox()
        self.fq_combo.addItem("前复权（本地现有行情）", "qfq")
        self.fq_combo.setEnabled(False)

        settings.addWidget(QLabel("研究方案"), 0, 0)
        settings.addWidget(self.strategy_combo, 0, 1, 1, 3)
        settings.addWidget(QLabel("开始日期"), 1, 0)
        settings.addWidget(self.start_edit, 1, 1)
        settings.addWidget(QLabel("结束日期"), 1, 2)
        settings.addWidget(self.end_edit, 1, 3)
        settings.addWidget(QLabel("暴露选取比例"), 2, 0)
        settings.addWidget(self.pct_edit, 2, 1)
        settings.addWidget(QLabel("低波观察窗口"), 2, 2)
        settings.addWidget(self.vol_window_edit, 2, 3)
        settings.addWidget(QLabel("动量形成期"), 3, 0)
        settings.addWidget(self.momentum_window_edit, 3, 1)
        settings.addWidget(QLabel("排除最近"), 3, 2)
        settings.addWidget(self.momentum_skip_edit, 3, 3)
        settings.addWidget(QLabel("行情口径"), 4, 0)
        settings.addWidget(self.fq_combo, 4, 1)
        settings.addWidget(QLabel("调仓频率"), 4, 2)
        settings.addWidget(QLabel("月频（每月最后一个交易日）"), 4, 3)
        settings.addWidget(QLabel("动量最低权重（五因子）"), 5, 0)
        settings.addWidget(self.momentum_floor_edit, 5, 1)
        settings.addWidget(QLabel("仅并集分支生效，0 表示关闭"), 5, 2, 1, 2)

        self.rules_label = QLabel(
            "固定规则：低价前 N% 等权 + 低波前 N% 等权；双因子组合为两个子组合各占 50% 资金；"
            "不计交易成本；低价暂以前复权收盘价排序。仅研究用途，不构成投资建议。"
        )
        self.rules_label.setWordWrap(True)
        settings.addWidget(self.rules_label, 6, 0, 1, 4)
        layout.addWidget(settings_group)

        # 三因子自定义资金比例：滑块原始值可为任意非负整数，回测前统一归一化
        self.weight_group = QGroupBox("三因子资金比例（自动归一化）")
        weight_layout = QGridLayout(self.weight_group)
        self.weight_preset = QComboBox()
        self.weight_preset.addItem("等权 33/33/34", (33, 33, 34))
        self.weight_preset.addItem("保守 40/40/20", (40, 40, 20))
        self.weight_preset.addItem("进取 20/20/60", (20, 20, 60))
        self.weight_preset.currentIndexChanged.connect(self.apply_weight_preset)
        weight_layout.addWidget(QLabel("预设"), 0, 0)
        weight_layout.addWidget(self.weight_preset, 0, 1, 1, 3)
        self.weight_sliders = {}
        self.weight_value_labels = {}
        for row, (key, label, value) in enumerate((
                ("low_price", "低价", 33), ("low_vol", "低波", 33),
                ("momentum", "动量", 34)), 1):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(value)
            slider.valueChanged.connect(self.update_weight_labels)
            value_label = QLabel()
            self.weight_sliders[key] = slider
            self.weight_value_labels[key] = value_label
            weight_layout.addWidget(QLabel(label), row, 0)
            weight_layout.addWidget(slider, row, 1)
            weight_layout.addWidget(value_label, row, 2)
        self.weight_normalized_label = QLabel()
        self.weight_normalized_label.setWordWrap(True)
        weight_layout.addWidget(self.weight_normalized_label, 4, 0, 1, 3)
        layout.addWidget(self.weight_group)
        self.update_weight_labels()
        self.update_strategy_controls()

        action_layout = QHBoxLayout()
        self.exposure_button = QPushButton("生成期末暴露")
        self.exposure_button.setToolTip("使用结束日期及之前的本地行情构建低价、低波名单，不执行回测")
        self.exposure_button.clicked.connect(self.generate_exposure)
        self.run_button = QPushButton("开始月频回测")
        self.run_button.setEnabled(True)
        self.run_button.setToolTip("在后台线程调用 smart_beta_backtest.py，只使用本地行情")
        self.run_button.clicked.connect(self.run_smart_beta)
        self.sensitivity_button = QPushButton("比例敏感度 10/20/30%")
        self.sensitivity_button.setToolTip("后台依次运行三组暴露比例，不改变当前主回测结果")
        self.sensitivity_button.clicked.connect(self.run_sensitivity)
        self.export_button = QPushButton("导出 HTML/CSV/parquet")
        self.export_button.setEnabled(False)
        self.export_button.setToolTip("回测完成后启用；报告仅保存到本地 data/smart_beta 目录")
        self.export_button.clicked.connect(self.export_report)
        self.summary_report_button = QPushButton("导出终版 HTML 报告")
        self.summary_report_button.setToolTip(
            "后台生成全路线离线报告 summary_report.html 与最新盯盘 CSV，"
            "只读取本地落盘结果并写入 data/smart_beta，不联网")
        self.summary_report_button.clicked.connect(self.export_summary_report)
        action_layout.addWidget(self.exposure_button)
        action_layout.addWidget(self.run_button)
        action_layout.addWidget(self.sensitivity_button)
        action_layout.addWidget(self.export_button)
        action_layout.addWidget(self.summary_report_button)
        action_layout.addStretch()
        layout.addLayout(action_layout)

        result_group = QGroupBox("Smart Beta 研究面板")
        result_layout = QVBoxLayout(result_group)
        self.result_tabs = QTabWidget()
        result_layout.addWidget(self.result_tabs)

        # 回测绩效与净值：判定着色只依赖核心返回的 verdicts，不在 UI 重复判定
        self.performance_tab = QWidget()
        performance_layout = QVBoxLayout(self.performance_tab)
        self.backtest_status = QLabel("尚未运行月频回测")
        self.backtest_status.setWordWrap(True)
        performance_layout.addWidget(self.backtest_status)
        self.performance_table = QTableWidget()
        prepare_table(self.performance_table, [
            "组合", "总收益", "年化", "夏普", "最大回撤", "月胜率", "累计换手", "判定"])
        self.performance_table.setFixedHeight(230)
        performance_layout.addWidget(self.performance_table)
        self.smart_beta_figure = Figure(figsize=(9, 4), dpi=100)
        self.smart_beta_canvas = FigureCanvasQTAgg(self.smart_beta_figure)
        self.smart_beta_canvas.setMinimumHeight(250)
        self.smart_beta_figure.text(
            0.99, 0.01, DISCLAIMER, ha="right", fontsize=8, color="gray")
        performance_layout.addWidget(self.smart_beta_canvas, 1)
        self.result_tabs.addTab(self.performance_tab, "绩效与净值")

        # 逐期持仓明细：默认只显示最新一期，避免一次渲染数千行
        self.holdings_tab = QWidget()
        holdings_layout = QVBoxLayout(self.holdings_tab)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("调仓日"))
        self.holdings_date = QComboBox()
        self.holdings_date.currentIndexChanged.connect(self.refresh_holdings_table)
        filter_layout.addWidget(self.holdings_date, 1)
        filter_layout.addWidget(QLabel("子组合"))
        self.holdings_portfolio = QComboBox()
        self.holdings_portfolio.addItem("全部子组合", "all")
        self.holdings_portfolio.addItem("低价", "low_price")
        self.holdings_portfolio.addItem("低波", "low_vol")
        self.holdings_portfolio.currentIndexChanged.connect(self.refresh_holdings_table)
        filter_layout.addWidget(self.holdings_portfolio, 1)
        holdings_layout.addLayout(filter_layout)
        self.holdings_summary = QLabel("尚未运行月频回测")
        holdings_layout.addWidget(self.holdings_summary)
        self.holdings_table = QTableWidget()
        prepare_table(self.holdings_table, [
            "调仓日", "组合", "代码", "名称", "权重", "前复权收盘", "日波动率",
            "动量", "PE", "PB", "下期收益"])
        holdings_layout.addWidget(self.holdings_table, 1)
        self.result_tabs.addTab(self.holdings_tab, "持仓明细")

        # Step 2 的期末暴露保留为独立子页
        self.exposure_tab = QWidget()
        exposure_layout = QVBoxLayout(self.exposure_tab)
        self.exposure_summary = QLabel("点击“生成期末暴露”查看低价与低波名单。")
        self.exposure_summary.setWordWrap(True)
        exposure_layout.addWidget(self.exposure_summary)
        tables_layout = QGridLayout()
        self.low_price_title = QLabel("低价暴露")
        self.low_price_table = self._make_exposure_table()
        tables_layout.addWidget(self.low_price_title, 0, 0)
        tables_layout.addWidget(self.low_price_table, 1, 0)
        self.low_vol_title = QLabel("低波暴露")
        self.low_vol_table = self._make_exposure_table()
        tables_layout.addWidget(self.low_vol_title, 0, 1)
        tables_layout.addWidget(self.low_vol_table, 1, 1)
        self.momentum_title = QLabel("动量暴露")
        self.momentum_table = self._make_exposure_table()
        tables_layout.addWidget(self.momentum_title, 0, 2)
        tables_layout.addWidget(self.momentum_table, 1, 2)
        self.low_pe_title = QLabel("低 PE 暴露")
        self.low_pe_table = self._make_exposure_table()
        tables_layout.addWidget(self.low_pe_title, 2, 0)
        tables_layout.addWidget(self.low_pe_table, 3, 0)
        self.low_pb_title = QLabel("低 PB 暴露")
        self.low_pb_table = self._make_exposure_table()
        tables_layout.addWidget(self.low_pb_title, 2, 2)
        tables_layout.addWidget(self.low_pb_table, 3, 2)
        exposure_layout.addLayout(tables_layout, 1)
        self.result_tabs.addTab(self.exposure_tab, "期末暴露")

        # 参数敏感度独立展示，避免覆盖当前主回测与导出结果
        self.sensitivity_tab = QWidget()
        sensitivity_layout = QVBoxLayout(self.sensitivity_tab)
        self.sensitivity_summary = QLabel("点击“比例敏感度 10/20/30%”开始分组回测。")
        self.sensitivity_summary.setWordWrap(True)
        sensitivity_layout.addWidget(self.sensitivity_summary)
        self.sensitivity_table = QTableWidget()
        prepare_table(self.sensitivity_table, [
            "暴露比例", "组合", "总收益", "年化", "夏普", "最大回撤", "判定"])
        sensitivity_layout.addWidget(self.sensitivity_table, 1)
        self.export_sensitivity_button = QPushButton("导出敏感度 CSV")
        self.export_sensitivity_button.setEnabled(False)
        self.export_sensitivity_button.clicked.connect(self.export_sensitivity)
        sensitivity_layout.addWidget(self.export_sensitivity_button)
        self.result_tabs.addTab(self.sensitivity_tab, "参数敏感度")
        self.update_exposure_visibility()
        layout.addWidget(result_group, 1)

    def update_strategy_controls(self, _=None):
        """按研究方案启停参数，并同步期末暴露页的可见维度。"""
        strategy = self.strategy_combo.currentData()
        uses_momentum = strategy != "dual"
        self.weight_group.setVisible(strategy == "custom")
        self.momentum_window_edit.setEnabled(uses_momentum)
        self.momentum_skip_edit.setEnabled(uses_momentum)
        self.momentum_floor_edit.setEnabled(strategy == "fundamental")
        rules = {
            "dual": "低价、低波各取前 N% 等权，两个子组合各占 50% 资金；不计交易成本。",
            "three": "低价、低波、动量各取前 N%；对比等权、保守、进取三种资金比例；交集不足 10 只时按并集综合名次取前 30 只。",
            "custom": "低价、低波、动量各取前 N%，资金采用下方自定义归一化比例；选股规则与三因子预设一致。",
            "fundamental": "低 PE/PB 仅保留正值；对比双/三因子、低 PE/PB、四因子与五因子；估值仅取调仓日及之前最近记录；"
                           "五因子并集分支动量最低权重按上方设置（默认 10%，四因子不启用下限）。",
        }
        self.rules_label.setText(
            rules[strategy] + "低价暂以前复权收盘价排序。仅研究用途，不构成投资建议。")
        # 初次策略初始化早于操作按钮和暴露表格创建。
        if hasattr(self, "low_price_table"):
            self.update_exposure_visibility()

    def apply_weight_preset(self, _=None):
        """套用预设原始比例，统一由标签和回测入口负责归一化。"""
        values = self.weight_preset.currentData()
        if values is None:
            return
        for key, value in zip(("low_price", "low_vol", "momentum"), values):
            slider = self.weight_sliders[key]
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        self.update_weight_labels()

    def update_weight_labels(self, _=None):
        """显示滑块原始值及实际用于回测的归一化资金比例。"""
        raw = {key: slider.value() for key, slider in self.weight_sliders.items()}
        total = sum(raw.values())
        for key, value in raw.items():
            self.weight_value_labels[key].setText("%d" % value)
        if total <= 0:
            self.weight_normalized_label.setText("三项均为 0，无法回测；请至少将一项调整为正数。")
            self.weight_normalized_label.setStyleSheet("color: #b3261e;")
            return
        normalized = {key: value / total * 100 for key, value in raw.items()}
        self.weight_normalized_label.setText(
            "归一化后：低价 %.2f%%　低波 %.2f%%　动量 %.2f%%（原始值合计 %d）" % (
                normalized["low_price"], normalized["low_vol"],
                normalized["momentum"], total))
        self.weight_normalized_label.setStyleSheet("color: #555;")

    def _normalized_factor_weights(self):
        raw = {key: float(slider.value()) for key, slider in self.weight_sliders.items()}
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("三因子资金比例不能全部为 0")
        return {key: value / total for key, value in raw.items()}

    def update_exposure_visibility(self, _=None):
        """双因子仅显示低价/低波，其余方案增加动量及估值维度。"""
        strategy = self.strategy_combo.currentData()
        visible = {
            "low_price": True, "low_vol": True,
            "momentum": strategy in ("three", "custom", "fundamental"),
            "low_pe": strategy == "fundamental", "low_pb": strategy == "fundamental",
        }
        for factor, is_visible in visible.items():
            getattr(self, factor + "_title").setVisible(is_visible)
            getattr(self, factor + "_table").setVisible(is_visible)

    def _capture_parameters(self):
        """在 UI 线程捕获参数，后台线程不再读取任何控件。"""
        start = self.start_edit.date().toString("yyyy-MM-dd")
        end = self.end_edit.date().toString("yyyy-MM-dd")
        validate_date_range(start, end)
        strategy = self.strategy_combo.currentData()
        floor = self.momentum_floor_edit.value() / 100.0
        return {
            "start_date": start, "end_date": end,
            "pct": self.pct_edit.value() / 100.0,
            "vol_window": self.vol_window_edit.value(),
            "momentum_window": self.momentum_window_edit.value(),
            "momentum_skip": self.momentum_skip_edit.value(),
            "fq": self.fq_combo.currentData() or "qfq",
            "momentum_floor": (floor if 0 < floor < 1 else None)
                              if strategy == "fundamental" else None,
            "factor_weights": (self._normalized_factor_weights()
                               if strategy == "custom" else None),
        }

    def _execute_backtest(self, strategy, parameters, pct=None):
        """统一分发四种本地方案；pct 覆盖参数供敏感度任务复用。"""
        kwargs = {key: parameters[key] for key in (
            "start_date", "end_date", "pct", "vol_window", "fq")}
        if pct is not None:
            kwargs["pct"] = pct
        if strategy == "dual":
            result = run_smart_beta_backtest(codes=None, **kwargs)
        else:
            kwargs.update({key: parameters[key] for key in (
                "momentum_window", "momentum_skip")})
            if strategy == "three":
                result = run_three_factor_comparison(codes=None, **kwargs)
            elif strategy == "custom":
                result = run_three_factor_backtest(
                    codes=None, factor_weights=parameters["factor_weights"], **kwargs)
            elif strategy == "fundamental":
                result = run_fundamental_comparison(
                    codes=None, momentum_floor=parameters.get("momentum_floor"),
                    **kwargs)
            else:
                raise ValueError("未知 Smart Beta 研究方案：%s" % strategy)
        # 记录启动回测时的方案，避免之后切换下拉框导致导出分发错误。
        result["meta"]["ui_strategy"] = strategy
        return result

    def _result_names(self, result):
        names = [name for name in result["metrics"] if name != "benchmark"]
        if "benchmark" in result["metrics"]:
            names.append("benchmark")
        return names

    def _primary_portfolio(self, result):
        strategy = result["meta"].get("ui_strategy", "dual")
        return self.PRIMARY_PORTFOLIO.get(strategy, "dual")

    def _passed_map(self, result):
        passed = result.get("passed", {})
        if isinstance(passed, dict):
            return passed
        return {self._primary_portfolio(result): bool(passed)}

    def populate_backtest_results(self, result):
        """按核心返回的组合动态刷新绩效、净值/回撤和持仓筛选器。"""
        pct = result["meta"]["pct"] * 100
        labels = dict(self.PORTFOLIO_LABELS)
        labels["low_price"] = "低价前%g%%" % pct
        labels["low_vol"] = "低波前%g%%" % pct
        labels["low_pe"] = "低 PE 前%g%%" % pct
        labels["low_pb"] = "低 PB 前%g%%" % pct
        floor = result["meta"].get("momentum_floor")
        if floor:
            labels["five_factor"] = "五因子等权 20%%（动量≥%g%%）" % (floor * 100)
        custom = result["meta"].get("custom_factor_weights")
        if custom:
            labels["three_custom"] = "三因子自定义 %.0f/%.0f/%.0f" % (
                custom["low_price"] * 100, custom["low_vol"] * 100,
                custom["momentum"] * 100)
        names = self._result_names(result)
        self.performance_table.setRowCount(len(names))
        for row, name in enumerate(names):
            metrics = result["metrics"][name]
            verdict = result["verdicts"].get(name)
            verdict_text = ("参考" if verdict is None
                            else ("通过" if verdict["passed"] else "未通过"))
            values = [
                labels.get(name, name), "%.2f%%" % (metrics["total_return"] * 100),
                "%.2f%%" % (metrics["annual_return"] * 100),
                "%.3f" % metrics["sharpe"],
                "%.2f%%" % (metrics["max_drawdown"] * 100),
                "%.2f%%" % (metrics["win_rate"] * 100),
                "%.2f" % metrics["total_turnover"], verdict_text,
            ]
            for col, value in enumerate(values):
                set_cell(self.performance_table, row, col, value,
                         numeric=col in (1, 2, 3, 4, 5, 6))
            if verdict is not None:
                color = QColor("#d9ead3") if verdict["passed"] else QColor("#f4cccc")
                self.performance_table.item(row, 7).setBackground(color)

        # 净值展示全部组合；回撤仅展示当前主组合与基准，避免线条过密。
        self.smart_beta_figure.clear()
        self.smart_beta_figure.text(
            0.99, 0.01, DISCLAIMER, ha="right", fontsize=8, color="gray")
        nav_axis = self.smart_beta_figure.add_subplot(211)
        drawdown_axis = self.smart_beta_figure.add_subplot(212, sharex=nav_axis)
        nav = result["nav"]
        dates = pd.to_datetime(nav.index)
        for name in names:
            if name in nav:
                nav_axis.plot(dates, nav[name].values, linewidth=1.2,
                              label=labels.get(name, name))
        for name in (self._primary_portfolio(result), "benchmark"):
            if name not in nav:
                continue
            drawdown = nav[name] / nav[name].cummax() - 1
            drawdown_axis.plot(dates, drawdown.values, linewidth=1.2,
                               label=labels.get(name, name))
        nav_axis.set_ylabel("净值")
        nav_axis.legend(loc="upper left", fontsize=8, ncol=min(4, len(names)))
        nav_axis.grid(alpha=0.2)
        drawdown_axis.set_ylabel("回撤")
        drawdown_axis.grid(alpha=0.2)
        drawdown_axis.legend(loc="lower left", fontsize=8)
        self.smart_beta_figure.autofmt_xdate()
        self.smart_beta_figure.tight_layout(rect=(0, 0.03, 1, 1))
        self.smart_beta_canvas.draw()

        # 调仓日倒序排列，默认显示最新一期；组合选项完全来自本次持仓结果。
        holdings = result["holdings"]
        dates_in_result = sorted(holdings["date"].unique().tolist(), reverse=True)
        self.holdings_date.blockSignals(True)
        self.holdings_date.clear()
        for date in dates_in_result:
            self.holdings_date.addItem(date, date)
        self.holdings_date.addItem("全部调仓日", "all")
        self.holdings_date.setCurrentIndex(0)
        self.holdings_date.blockSignals(False)
        present = set(holdings["portfolio"].unique())
        portfolio_names = [name for name in names if name in present]
        portfolio_names += sorted(present - set(portfolio_names))
        self.holdings_portfolio.blockSignals(True)
        self.holdings_portfolio.clear()
        self.holdings_portfolio.addItem("全部组合", "all")
        for name in portfolio_names:
            self.holdings_portfolio.addItem(labels.get(name, name), name)
        self.holdings_portfolio.setCurrentIndex(0)
        self.holdings_portfolio.blockSignals(False)
        self.stock_names = {row["code"]: row["name"] for row in db.query_all(
            "SELECT code, name FROM stock_info")}
        self.refresh_holdings_table()

    def refresh_holdings_table(self, _=None):
        """按调仓日和组合过滤持仓，统一展示动量与 PE/PB 评分。"""
        if self.last_result is None:
            return
        frame = self.last_result["holdings"].copy()
        date = self.holdings_date.currentData()
        portfolio = self.holdings_portfolio.currentData()
        if date and date != "all":
            frame = frame[frame["date"] == date]
        if portfolio and portfolio != "all":
            frame = frame[frame["portfolio"] == portfolio]
        frame = frame.sort_values(["date", "portfolio", "code"],
                                  ascending=[False, True, True])
        self.holdings_table.setRowCount(len(frame))
        for row, record in enumerate(frame.to_dict("records")):
            def number(key, digits=3, percent=False):
                value = record.get(key)
                if value is None or pd.isna(value):
                    return ""
                return ("%.*f%%" % (digits, value * 100)) if percent else ("%.*f" % (digits, value))

            values = [
                record["date"], self.PORTFOLIO_LABELS.get(record["portfolio"], record["portfolio"]),
                record["code"], self.stock_names.get(record["code"], record["code"]),
                number("weight", 2, True), number("price", 3), number("volatility", 4),
                number("momentum", 2, True), number("pe_ttm", 2), number("pb_mrq", 2),
                number("period_return", 2, True),
            ]
            for col, value in enumerate(values):
                set_cell(self.holdings_table, row, col, value, numeric=col >= 4)
        weight_note = "权重为对应组合的资金权重"
        if self.last_result["meta"].get("ui_strategy") == "dual":
            weight_note = "原双因子表中权重为子组合内部权重，低价/低波各占总资金 50%"
        floor = self.last_result["meta"].get("momentum_floor")
        if floor:
            weight_note += "；五因子并集分支动量最低权重 %.0f%%" % (floor * 100)
        self.holdings_summary.setText(
            "当前筛选共 %d 条；%s；%s" % (len(frame), weight_note, DISCLAIMER))

    def export_report(self):
        """按回测时的方案复用核心落盘逻辑，并生成自包含离线 HTML。"""
        if self.last_result is None:
            QMessageBox.warning(self, "无法导出", "请先运行 Smart Beta 月频回测")
            return
        self.export_button.setEnabled(False)
        try:
            strategy = self.last_result["meta"].get("ui_strategy", "dual")
            saver = {
                "dual": save_reports, "three": save_comparison_reports,
                "custom": save_comparison_reports, "fundamental": save_fundamental_reports,
            }[strategy]
            paths = saver(self.last_result)
            html_path = self._write_html_report(paths)
            self.last_export_paths = dict(paths, html=html_path)
            self.window.statusBar().showMessage(
                "Smart Beta 报告已保存到本地：" + str(html_path.parent), 10000)
            QMessageBox.information(
                self, "导出完成",
                "HTML、CSV、JSON、净值与持仓 parquet 已保存到本地目录：\n%s\n\nHTML：%s" % (
                    html_path.parent, html_path.name))
        except Exception as exc:
            self.window.show_error("Smart Beta 导出失败", traceback.format_exc(), str(exc))
        finally:
            self.export_button.setEnabled(self.last_result is not None)

    def _write_html_report(self, paths):
        """生成多策略通用 HTML；图表内嵌，不引用任何网络资源。"""
        result = self.last_result
        meta = result["meta"]
        image_buffer = io.BytesIO()
        self.smart_beta_figure.savefig(
            image_buffer, format="png", dpi=130, bbox_inches="tight")
        chart_base64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")

        pct = meta["pct"] * 100
        labels = dict(self.PORTFOLIO_LABELS)
        labels["low_price"] = "低价前%g%%" % pct
        labels["low_vol"] = "低波前%g%%" % pct
        labels["low_pe"] = "低 PE 前%g%%" % pct
        labels["low_pb"] = "低 PB 前%g%%" % pct
        floor = meta.get("momentum_floor")
        if floor:
            labels["five_factor"] = "五因子等权 20%%（动量≥%g%%）" % (floor * 100)
        rows = []
        for name in self._result_names(result):
            metrics = result["metrics"][name]
            verdict = result["verdicts"].get(name)
            verdict_text = ("参考" if verdict is None
                            else ("通过" if verdict["passed"] else "未通过"))
            css_class = ("reference" if verdict is None
                         else ("pass" if verdict["passed"] else "fail"))
            rows.append(
                "<tr><td>%s</td><td>%.2f%%</td><td>%.2f%%</td><td>%.3f</td>"
                "<td>%.2f%%</td><td>%.2f%%</td><td>%.2f</td>"
                "<td class='%s'>%s</td></tr>" % (
                    escape(labels.get(name, name)), metrics["total_return"] * 100,
                    metrics["annual_return"] * 100, metrics["sharpe"],
                    metrics["max_drawdown"] * 100, metrics["win_rate"] * 100,
                    metrics["total_turnover"], css_class, verdict_text))
        links = []
        for key, label in (("csv", "绩效 CSV"), ("json", "完整 JSON"),
                           ("nav", "逐月净值 CSV"), ("holdings", "持仓明细 parquet")):
            links.append("<li><a href='%s'>%s</a></li>" % (
                escape(paths[key].name), label))
        primary = self._primary_portfolio(result)
        primary_passed = self._passed_map(result).get(
            primary, result["verdicts"].get(primary, {}).get("passed", False))
        strategy_labels = {
            "dual": "双因子 50:50", "three": "三因子预设对比",
            "custom": "三因子自定义比例", "fundamental": "低 PE/PB 与四/五因子对比",
        }
        strategy = meta.get("ui_strategy", "dual")
        parameters = ["暴露比例：各前 %g%%" % pct, "低波窗口：%d 日" % meta["vol_window"]]
        if "momentum_window" in meta:
            parameters.append("动量：%d 日并排除最近 %d 日" % (
                meta["momentum_window"], meta["momentum_skip"]))
        if meta.get("custom_factor_weights"):
            custom = meta["custom_factor_weights"]
            parameters.append("自定义低价/低波/动量：%.2f%%/%.2f%%/%.2f%%" % (
                custom["low_price"] * 100, custom["low_vol"] * 100, custom["momentum"] * 100))
        if meta.get("valuation_rule"):
            parameters.append(meta["valuation_rule"])
        if meta.get("selection_note"):
            parameters.append(meta["selection_note"])
        if meta.get("momentum_floor"):
            parameters.append("五因子动量最低权重：%.0f%%（仅并集分支生效）" % (
                meta["momentum_floor"] * 100))
        replacements = {
            "__GENERATED__": escape(meta["generated_at"]),
            "__CODES__": str(meta["codes"]), "__START__": escape(meta["start"]),
            "__END__": escape(meta["end"]), "__STRATEGY__": escape(strategy_labels[strategy]),
            "__PARAMETERS__": escape("<br>".join(parameters)),
            "__PRICE_NOTE__": escape(meta.get("price_note", "低价以前复权收盘价排序")),
            "__PRIMARY__": escape(labels.get(primary, primary)),
            "__VERDICT__": "通过" if primary_passed else "未通过",
            "__VERDICT_CLASS__": "pass" if primary_passed else "fail",
            "__ROWS__": "\n".join(rows), "__CHART__": chart_base64,
            "__LINKS__": "\n".join(links), "__DISCLAIMER__": DISCLAIMER,
        }
        # 参数在上方已拼入换行，此处恢复为 HTML 换行，其余内容仍统一转义。
        replacements["__PARAMETERS__"] = replacements["__PARAMETERS__"].replace("&lt;br&gt;", "<br>")
        html_text = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Smart Beta 多策略月频回测</title>
<style>
body{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1100px;margin:28px auto;color:#222;padding:0 18px}
h1{font-size:25px}h2{margin-top:26px;font-size:19px}.meta{background:#f5f6f8;padding:14px;border-radius:6px;line-height:1.8}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ddd;padding:8px;text-align:right}th{background:#eee}td:first-child,th:first-child{text-align:left}
.pass{color:#187a1f;background:#d9ead3;font-weight:bold}.fail{color:#b3261e;background:#f4cccc;font-weight:bold}.reference{color:#666}
img{max-width:100%;border:1px solid #ddd}.note{color:#666;font-size:13px}.disclaimer{color:#a33;font-weight:bold;margin-top:24px}
</style></head><body>
<h1>Smart Beta 多策略月频回测</h1>
<div class="meta">生成时间：__GENERATED__<br>研究方案：__STRATEGY__<br>本地股票池：__CODES__ 只　区间：__START__ ～ __END__<br>
__PARAMETERS__<br>__PRICE_NOTE__</div>
<h2>__PRIMARY__ 判定：<span class="__VERDICT_CLASS__">__VERDICT__</span></h2>
<p>判定口径：年化收益大于 0（跑赢现金），且最大回撤不高于池内等权基准；不要求跑赢基准或沪深300。</p>
<h2>绩效汇总</h2>
<table><thead><tr><th>组合</th><th>总收益</th><th>年化</th><th>夏普</th><th>最大回撤</th><th>月胜率</th><th>累计换手</th><th>判定</th></tr></thead>
<tbody>__ROWS__</tbody></table>
<h2>净值与主组合回撤</h2><img alt="Smart Beta 净值与回撤" src="data:image/png;base64,__CHART__">
<h2>本地附件</h2><ul>__LINKS__</ul>
<p class="note">未计交易成本；持仓明细 parquet 保留每期代码、权重、评分与下期收益。HTML 图片已内嵌，可离线打开。</p>
<p class="disclaimer">__DISCLAIMER__。不接券商、不下单、不上传云端。</p>
</body></html>"""
        for token, value in replacements.items():
            html_text = html_text.replace(token, str(value))
        html_path = paths["csv"].with_name(paths["csv"].stem + "_report.html")
        html_path.write_text(html_text, encoding="utf-8")
        return html_path

    def _make_exposure_table(self):
        """创建只读暴露表格；保留各因子原始评分以核对排序口径。"""
        table = QTableWidget(0, 9)
        table.setHorizontalHeaderLabels([
            "排名", "代码", "名称", "前复权收盘", "日波动率", "动量", "PE", "PB", "子组合权重"])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _fill_exposure_table(self, table, title, factor, codes, scores, names):
        """按核心返回顺序填充，不在 UI 重新排序或重新选股。"""
        factor_labels = {
            "low_price": "低价暴露", "low_vol": "低波暴露", "momentum": "动量暴露",
            "low_pe": "低 PE 暴露", "low_pb": "低 PB 暴露",
        }
        title.setText("%s（%d 只，子组合等权）" % (factor_labels[factor], len(codes)))
        table.setRowCount(len(codes))
        weight = 100.0 / len(codes) if codes else 0.0
        for row, code in enumerate(codes):
            def score(column, digits=3, percent=False):
                value = scores[column].get(code) if column in scores else None
                if value is None or pd.isna(value):
                    return ""
                return ("%.*f%%" % (digits, value * 100)) if percent else ("%.*f" % (digits, value))

            values = [
                str(row + 1), code, names.get(code, code), score("price"),
                score("volatility", 4), score("momentum", 2, True),
                score("pe_ttm", 2), score("pb_mrq", 2), "%.2f%%" % weight,
            ]
            for col, value in enumerate(values):
                set_cell(table, row, col, value, numeric=col == 0 or col >= 3)

    def generate_exposure(self):
        """按当前方案构建结束日暴露；所有评分仅使用该日及之前数据。"""
        try:
            parameters = self._capture_parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        strategy = self.strategy_combo.currentData()
        fq = parameters["fq"]
        trade_date = parameters["end_date"]
        self.exposure_button.setEnabled(False)
        self.exposure_button.setText("正在构建…")
        QApplication.processEvents()
        try:
            codes = [row["code"] for row in db.query_all(
                "SELECT DISTINCT code FROM daily_bars WHERE fq=? ORDER BY code", (fq,))]
            if not codes:
                raise ValueError("本地没有 %s 日线行情，请先在数据管理页更新" % fq)
            data_map = load_all_daily(codes, fq=fq)
            common = {
                "pct": parameters["pct"], "vol_window": parameters["vol_window"],
                "momentum_window": parameters["momentum_window"],
                "momentum_skip": parameters["momentum_skip"],
            }
            if strategy == "dual":
                raw = build_exposures(
                    data_map, trade_date, pct=parameters["pct"],
                    vol_window=parameters["vol_window"])
                exposure = {
                    "date": trade_date, "scores": raw["scores"],
                    "sleeves": {"low_price": raw["low_price"], "low_vol": raw["low_vol"]},
                    "selection_rule": "dual_sleeves", "counts": {},
                }
            elif strategy in ("three", "custom"):
                raw = build_three_factor_exposures(
                    data_map, trade_date, mode="equal",
                    factor_weights=(parameters["factor_weights"]
                                    if strategy == "custom" else None), **common)
                exposure = dict(raw)
                exposure["sleeves"] = {key: raw[key] for key in (
                    "low_price", "low_vol", "momentum")}
            else:
                valuation_map = {code: load_valuation(code) for code in data_map}
                missing = [code for code, frame in valuation_map.items() if frame.empty]
                if missing:
                    raise ValueError("以下股票缺少本地估值数据：%s" % ",".join(missing[:10]))
                exposure = build_fundamental_composite(
                    data_map, valuation_map, trade_date,
                    include_pe=True, include_pb=True, **common,
                    momentum_floor=parameters.get("momentum_floor"))
            sleeves = exposure["sleeves"]
            factors = [key for key in ("low_price", "low_vol", "momentum", "low_pe", "low_pb")
                       if key in sleeves]
            if not factors or any(not sleeves[key] for key in factors):
                raise ValueError("有效样本不足，无法生成当前方案的期末暴露")
            names = {row["code"]: row["name"] for row in db.query_all(
                "SELECT code, name FROM stock_info")}
            self.last_exposure = exposure
            for factor in factors:
                self._fill_exposure_table(
                    getattr(self, factor + "_table"), getattr(self, factor + "_title"),
                    factor, sleeves[factor], exposure["scores"], names)
            factor_labels = {
                "low_price": "低价", "low_vol": "低波", "momentum": "动量",
                "low_pe": "低 PE", "low_pb": "低 PB",
            }
            counts = "、".join("%s %d 只" % (factor_labels[key], len(sleeves[key]))
                               for key in factors)
            intersection = len(set.intersection(*(set(sleeves[key]) for key in factors)))
            self.exposure_summary.setText(
                "截至 %s：%s；共同交集 %d 只；选股规则 %s。仅研究用途，不构成投资建议。" % (
                    trade_date, counts, intersection, exposure.get("selection_rule", "固定暴露")))
            self.result_tabs.setCurrentWidget(self.exposure_tab)
            self.window.statusBar().showMessage("期末暴露已生成（未执行回测）", 5000)
        except Exception as exc:
            QMessageBox.warning(self, "暴露构建失败", str(exc))
        finally:
            self.exposure_button.setEnabled(True)
            self.exposure_button.setText("生成期末暴露")

    def run_smart_beta(self):
        """捕获界面参数后在后台执行月频回测，线程内不访问 UI 控件。"""
        strategy = self.strategy_combo.currentData()
        try:
            parameters = self._capture_parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "无法回测", str(exc))
            return

        def task(progress):
            # 核心回测暂无逐期回调，主任务使用不确定进度条。
            progress.emit({"total": 0, "text": "正在本地运行 Smart Beta 月频回测，不联网"})
            result = self._execute_backtest(strategy, parameters)
            progress.emit({"total": 1, "current": 1, "text": "Smart Beta 回测计算完成"})
            return result

        self.window.start_task(task, self.smart_beta_finished,
                               "正在运行 Smart Beta 月频回测")

    def smart_beta_finished(self, result):
        """保存完整结果并刷新动态绩效、净值、持仓和主组合判定。"""
        self.last_result = result
        primary = self._primary_portfolio(result)
        metrics = result["metrics"][primary]
        passed = self._passed_map(result).get(
            primary, result["verdicts"].get(primary, {}).get("passed", False))
        verdict = "通过" if passed else "未通过"
        primary_label = self.PORTFOLIO_LABELS.get(primary, primary)
        floor = result["meta"].get("momentum_floor")
        if primary == "five_factor" and floor:
            primary_label = "五因子等权 20%%（动量≥%g%%）" % (floor * 100)
        self.backtest_status.setText(
            "回测完成：%d 期，%s年化 %.2f%%、夏普 %.3f、最大回撤 %.2f%%，判定%s；%s" % (
                metrics["periods"], primary_label,
                metrics["annual_return"] * 100, metrics["sharpe"],
                metrics["max_drawdown"] * 100, verdict, DISCLAIMER))
        self.backtest_status.setStyleSheet(
            "color: #187a1f; font-weight: bold;" if passed
            else "color: #b3261e; font-weight: bold;")
        self.populate_backtest_results(result)
        self.result_tabs.setCurrentWidget(self.performance_tab)
        self.export_button.setEnabled(True)
        self.window.statusBar().showMessage(
            "Smart Beta 月频回测完成，绩效、净值与持仓已刷新", 10000)

    def run_sensitivity(self):
        """后台依次运行 10%/20%/30% 暴露比例，显示真实分段进度。"""
        strategy = self.strategy_combo.currentData()
        try:
            parameters = self._capture_parameters()
        except ValueError as exc:
            QMessageBox.warning(self, "无法运行敏感度", str(exc))
            return

        def task(progress):
            rows = []
            ratios = (0.10, 0.20, 0.30)
            for index, pct in enumerate(ratios):
                progress.emit({
                    "total": len(ratios), "current": index,
                    "text": "正在运行暴露比例 %.0f%%（%d/%d），不联网" % (
                        pct * 100, index + 1, len(ratios)),
                })
                result = self._execute_backtest(strategy, parameters, pct=pct)
                for name in self.SENSITIVITY_PORTFOLIOS[strategy]:
                    metrics = result["metrics"][name]
                    verdict = result["verdicts"].get(name)
                    row = {"pct": pct, "portfolio": name}
                    row.update(metrics)
                    row["passed"] = verdict["passed"] if verdict else None
                    rows.append(row)
                progress.emit({"total": len(ratios), "current": index + 1,
                               "text": "暴露比例 %.0f%% 已完成" % (pct * 100)})
            return {
                "strategy": strategy, "rows": rows,
                "start": parameters["start_date"], "end": parameters["end_date"],
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        self.window.start_task(task, self.sensitivity_finished,
                               "正在运行 Smart Beta 暴露比例敏感度")

    def sensitivity_finished(self, result):
        """展示各比例下的收益、风险与核心判定，不覆盖主回测。"""
        self.last_sensitivity = result
        rows = result["rows"]
        self.sensitivity_table.setRowCount(len(rows))
        for row_index, record in enumerate(rows):
            verdict = "参考" if record["passed"] is None else ("通过" if record["passed"] else "未通过")
            values = [
                "%.0f%%" % (record["pct"] * 100),
                self.PORTFOLIO_LABELS.get(record["portfolio"], record["portfolio"]),
                "%.2f%%" % (record["total_return"] * 100),
                "%.2f%%" % (record["annual_return"] * 100),
                "%.3f" % record["sharpe"],
                "%.2f%%" % (record["max_drawdown"] * 100), verdict,
            ]
            for column, value in enumerate(values):
                set_cell(self.sensitivity_table, row_index, column, value,
                         numeric=column in (0, 2, 3, 4, 5))
            if record["passed"] is not None:
                color = QColor("#d9ead3") if record["passed"] else QColor("#f4cccc")
                self.sensitivity_table.item(row_index, 6).setBackground(color)
        self.sensitivity_summary.setText(
            "已完成 10%%/20%%/30%% 三档暴露比例，共 %d 行结果；生成时间 %s；"
            "每组判定均为年化收益大于 0 且最大回撤不超过当次池内等权基准。%s" % (
                len(rows), result["generated_at"], DISCLAIMER))
        self.export_sensitivity_button.setEnabled(True)
        self.result_tabs.setCurrentWidget(self.sensitivity_tab)
        self.window.statusBar().showMessage("Smart Beta 暴露比例敏感度已完成", 10000)

    def export_summary_report(self):
        """后台生成 I2-alt-19 终版离线 HTML 报告与最新盯盘 CSV，不联网。"""
        def task(progress):
            progress.emit({
                "total": 0,
                "text": "正在生成终版 HTML 报告，仅读取本地落盘结果，不联网",
            })
            from core.smart_beta_summary_report import build_summary_report
            result = build_summary_report()
            progress.emit({"total": 1, "current": 1, "text": "终版报告生成完成"})
            return result

        self.window.start_task(task, self.summary_report_finished,
                               "正在生成 Smart Beta 终版 HTML 报告")

    def summary_report_finished(self, result):
        """报告生成后提示路径，可选直接在本机浏览器打开。"""
        html_path = Path(result["html"])
        csv_path = Path(result["watchlist_csv"])
        self.window.statusBar().showMessage(
            "终版报告已生成：%s（只读离线，不下单）" % html_path, 10000)
        answer = QMessageBox.question(
            self, "终版报告已生成",
            "已保存到本地：\n%s\n\n盯盘清单：%s\n\n是否立即用默认浏览器打开？\n%s"
            % (html_path, csv_path.name, DISCLAIMER),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path)))

    def export_sensitivity(self):
        """将当前敏感度明细保存为带免责声明的本地 CSV。"""
        if not self.last_sensitivity:
            QMessageBox.warning(self, "无法导出", "请先运行暴露比例敏感度")
            return
        try:
            result = self.last_sensitivity
            rows = []
            for record in result["rows"]:
                row = dict(record)
                row["label"] = self.PORTFOLIO_LABELS.get(record["portfolio"], record["portfolio"])
                row["generated_at"] = result["generated_at"]
                row["disclaimer"] = DISCLAIMER
                rows.append(row)
            SMART_BETA_DIR.mkdir(parents=True, exist_ok=True)
            path = SMART_BETA_DIR / ("smart_beta_sensitivity_%s_%s_%s.csv" % (
                result["strategy"], result["start"].replace("-", ""),
                result["end"].replace("-", "")))
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            self.window.statusBar().showMessage("敏感度 CSV 已保存：" + str(path), 10000)
            QMessageBox.information(self, "导出完成", "敏感度 CSV 已保存到本地：\n%s" % path)
        except Exception as exc:
            self.window.show_error("敏感度导出失败", traceback.format_exc(), str(exc))


class PositionPage(QWidget):
    """持仓管理与盯盘提醒（离线，仅用本地日线行情，不接券商、不下单）。"""

    # 持仓表格列：(显示名, 数据键, 是否数值)
    POS_COLUMNS = [
        ("代码", "code", False),
        ("名称", "name", False),
        ("持股数", "shares", True),
        ("成本价", "cost_price", True),
        ("买入日期", "buy_date", False),
        ("现价", "current_price", True),
        ("浮动盈亏", "pnl", True),
        ("盈亏比例", "pnl_ratio", True),
        ("持有交易日", "days_held", True),
        ("强制卖出", "force_sell", False),
        ("止损", "stop_loss", True),
        ("止盈", "take_profit", True),
        ("目标价", "target_price", True),
        ("备注", "note", False),
    ]

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.editing_id = None  # 当前编辑的持仓 id，None 表示新增
        layout = QVBoxLayout(self)

        # 新增/编辑表单
        form_group = QGroupBox("新增 / 编辑持仓（双击表格行可载入编辑）")
        form = QGridLayout(form_group)
        self.code_edit = QLineEdit()
        self.code_edit.setPlaceholderText("6位代码，如 600519")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("可选")
        self.shares_spin = QSpinBox()
        self.shares_spin.setRange(0, 100000000)
        self.shares_spin.setSingleStep(100)
        self.shares_spin.setSuffix(" 股")
        self.cost_spin = QDoubleSpinBox()
        self.cost_spin.setRange(0, 1000000)
        self.cost_spin.setDecimals(3)
        self.cost_spin.setSingleStep(0.01)
        self.buy_date_edit = make_date_edit(QDate.currentDate())
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0, 1)
        self.stop_loss_spin.setDecimals(4)
        self.stop_loss_spin.setSingleStep(0.01)
        self.stop_loss_spin.setToolTip("止损比例，如 0.05 = -5%；0 表示不设")
        self.take_profit_spin = QDoubleSpinBox()
        self.take_profit_spin.setRange(0, 5)
        self.take_profit_spin.setDecimals(4)
        self.take_profit_spin.setSingleStep(0.01)
        self.take_profit_spin.setToolTip("止盈比例，如 0.10 = +10%；0 表示不设")
        self.target_price_spin = QDoubleSpinBox()
        self.target_price_spin.setRange(0, 1000000)
        self.target_price_spin.setDecimals(3)
        self.target_price_spin.setSingleStep(0.01)
        self.target_price_spin.setToolTip("目标价，0 表示不设")
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("备注，可选")
        self.save_button = QPushButton("新增持仓")
        self.save_button.clicked.connect(self.save_position)
        self.reset_button = QPushButton("清空表单")
        self.reset_button.clicked.connect(self.reset_form)

        form.addWidget(QLabel("代码*"), 0, 0)
        form.addWidget(self.code_edit, 0, 1)
        form.addWidget(QLabel("名称"), 0, 2)
        form.addWidget(self.name_edit, 0, 3)
        form.addWidget(QLabel("持股数"), 0, 4)
        form.addWidget(self.shares_spin, 0, 5)
        form.addWidget(QLabel("成本价"), 0, 6)
        form.addWidget(self.cost_spin, 0, 7)
        form.addWidget(QLabel("买入日期"), 1, 0)
        form.addWidget(self.buy_date_edit, 1, 1)
        form.addWidget(QLabel("止损比例"), 1, 2)
        form.addWidget(self.stop_loss_spin, 1, 3)
        form.addWidget(QLabel("止盈比例"), 1, 4)
        form.addWidget(self.take_profit_spin, 1, 5)
        form.addWidget(QLabel("目标价"), 1, 6)
        form.addWidget(self.target_price_spin, 1, 7)
        form.addWidget(QLabel("备注"), 2, 0)
        form.addWidget(self.note_edit, 2, 1, 1, 5)
        form.addWidget(self.save_button, 2, 6)
        form.addWidget(self.reset_button, 2, 7)
        layout.addWidget(form_group)

        # 操作按钮行
        action_row = QHBoxLayout()
        self.refresh_button = QPushButton("刷新行情（本地日线，不联网）")
        self.refresh_button.clicked.connect(self.refresh_positions)
        self.import_button = QPushButton("导入 CSV")
        self.import_button.clicked.connect(self.import_csv)
        self.template_button = QPushButton("导出 CSV 模板")
        self.template_button.clicked.connect(self.export_template)
        self.hammer_button = QPushButton("生成 hammer_trades.csv")
        self.hammer_button.setToolTip("用本地日线回放 HAMMER_BOLL_BUY/SELL，输出卖点核对清单；不联网、不下单")
        self.hammer_button.clicked.connect(self.run_hammer_backtest)
        self.sell_import_button = QPushButton("导入 SELL 结果 CSV")
        self.sell_import_button.setToolTip("导入通达信 HAMMER_BOLL_SELL 扫描结果或 hammer_trades.csv，仅做预警核对")
        self.sell_import_button.clicked.connect(self.import_sell_signals)
        self.delete_button = QPushButton("删除选中")
        self.delete_button.clicked.connect(self.delete_selected)
        action_row.addWidget(self.refresh_button)
        action_row.addWidget(self.import_button)
        action_row.addWidget(self.template_button)
        action_row.addWidget(self.hammer_button)
        action_row.addWidget(self.sell_import_button)
        action_row.addWidget(self.delete_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        # 持仓表格
        self.position_table = QTableWidget()
        prepare_table(self.position_table, [c[0] for c in self.POS_COLUMNS])
        header_hints = {
            "买入日期": "用于计算持有交易日；买入日记为第 1 天",
            "持有交易日": "按本地交易日历计数，买入日记为第 1 天",
            "强制卖出": "满 5 个交易日收盘强制离场；此处仅提示，需人工确认",
        }
        for index, (label, _key, _numeric) in enumerate(self.POS_COLUMNS):
            item = self.position_table.horizontalHeaderItem(index)
            if label in header_hints and item is not None:
                item.setToolTip(header_hints[label])
        self.position_table.cellDoubleClicked.connect(self.load_row_to_form)
        self.position_empty_label = QLabel("暂无持仓，可在上方表单新增或导入 CSV")
        self.position_empty_label.setAlignment(Qt.AlignCenter)
        self.position_empty_label.setStyleSheet("color: #777; padding: 24px;")
        self.position_stack = QStackedWidget()
        self.position_stack.addWidget(self.position_table)
        self.position_stack.addWidget(self.position_empty_label)
        self.position_stack.setCurrentWidget(self.position_empty_label)
        layout.addWidget(self.position_stack, 3)

        # 预警列表
        alert_header = QHBoxLayout()
        self.alert_label = QLabel("预警（满5天强制卖出 / SELL信号 / 止损 / 止盈 / 目标价 / 无行情）")
        alert_header.addWidget(self.alert_label)
        alert_header.addStretch()
        layout.addLayout(alert_header)
        self.alert_table = QTableWidget()
        prepare_table(self.alert_table, ["类型", "代码", "名称", "信息"])
        self.alert_table.setFixedHeight(130)
        self.alert_empty_label = QLabel("暂无预警")
        self.alert_empty_label.setAlignment(Qt.AlignCenter)
        self.alert_empty_label.setStyleSheet("color: #777;")
        self.alert_stack = QStackedWidget()
        self.alert_stack.addWidget(self.alert_table)
        self.alert_stack.addWidget(self.alert_empty_label)
        self.alert_stack.setCurrentWidget(self.alert_empty_label)
        self.alert_stack.setFixedHeight(150)
        layout.addWidget(self.alert_stack)

        # 本期调仓清单 + 五因子/四因子对比：只读展示，不接券商、不下单、不上传云端。
        watch_group = QGroupBox("本期调仓清单与方案对比（只读跟踪；不接券商、不下单、不上传云端）")
        watch_layout = QVBoxLayout(watch_group)
        self.watch_tabs = QTabWidget()

        # 子页 1：清单明细
        detail_page = QWidget()
        detail_layout = QVBoxLayout(detail_page)
        watch_row = QHBoxLayout()
        watch_row.addWidget(QLabel("组合筛选"))
        self.watch_filter_combo = QComboBox()
        self.watch_filter_combo.addItem("全部组合", "全部")
        self.watch_filter_combo.addItem("五因子主基线", "五因子")
        self.watch_filter_combo.addItem("四因子主基线", "四因子")
        self.watch_filter_combo.currentIndexChanged.connect(self._fill_watchlist_table)
        watch_row.addWidget(self.watch_filter_combo)
        self.watch_refresh_button = QPushButton("读取最新清单")
        self.watch_refresh_button.clicked.connect(self.load_watchlist)
        watch_row.addWidget(self.watch_refresh_button)
        self.watch_summary_label = QLabel("尚未加载清单")
        self.watch_summary_label.setStyleSheet("color: #555;")
        watch_row.addWidget(self.watch_summary_label, 1)
        detail_layout.addLayout(watch_row)

        self.watch_table = QTableWidget()
        prepare_table(self.watch_table, WATCHLIST_COLUMNS)
        header = self.watch_table.horizontalHeader()
        watch_widths = {
            "代码": 70, "名称": 90, "组合": 200, "因子归属": 190, "综合百分位和": 100,
            "动量(%)": 80, "年化波动(%)": 95, "PE(TTM)": 80, "PB(MRQ)": 80,
            "权重(%)": 80, "下期收益(%)": 95, "调仓日": 100, "收益截止日": 105,
        }
        for column, name in enumerate(WATCHLIST_COLUMNS):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            self.watch_table.setColumnWidth(column, watch_widths[name])
        header.setStretchLastSection(True)
        self.watch_empty_label = QLabel("暂无调仓清单，请先在 Smart Beta 页生成并导出")
        self.watch_empty_label.setAlignment(Qt.AlignCenter)
        self.watch_empty_label.setStyleSheet("color: #777; padding: 16px;")
        self.watch_stack = QStackedWidget()
        self.watch_stack.addWidget(self.watch_table)
        self.watch_stack.addWidget(self.watch_empty_label)
        self.watch_stack.setCurrentWidget(self.watch_empty_label)
        detail_layout.addWidget(self.watch_stack, 1)
        self.watch_tabs.addTab(detail_page, "清单明细")

        # 子页 2：五因子 vs 四因子对比（当期清单 + 全期核对指标）
        compare_page = QWidget()
        compare_layout = QVBoxLayout(compare_page)
        self.compare_table = QTableWidget()
        prepare_table(self.compare_table, ["指标", "五因子主基线（动量≥10%）", "四因子主基线（旧并集规则）"])
        compare_header = self.compare_table.horizontalHeader()
        for column, width in enumerate((230, 300, 300)):
            compare_header.setSectionResizeMode(column, QHeaderView.Interactive)
            self.compare_table.setColumnWidth(column, width)
        compare_header.setStretchLastSection(True)
        compare_layout.addWidget(self.compare_table, 1)
        self.compare_note = QLabel("尚未加载对比数据")
        self.compare_note.setStyleSheet("color: #777;")
        self.compare_note.setWordWrap(True)
        compare_layout.addWidget(self.compare_note)
        self.watch_tabs.addTab(compare_page, "五因子 vs 四因子")

        self.watch_tabs.setFixedHeight(260)
        watch_layout.addWidget(self.watch_tabs)
        layout.addWidget(watch_group)

        self._watchlist = None       # 原始清单 DataFrame（只读缓存）
        self._watchlist_path = None   # 当前清单文件路径
        self._strategy_metrics = {}   # 全期核对指标（来自 four_reconciliation.csv）
        self.sell_signals = None      # 已导入/生成的 SELL 信号（只读缓存）
        self.load_positions()
        self.load_watchlist()

    def load_watchlist(self):
        """读取最新 watchlist_*.csv 并填充只读表格，全程离线、不写文件。"""
        try:
            path, df = _load_latest_watchlist()
        except Exception as exc:
            self._watchlist = None
            self._watchlist_path = None
            self._strategy_metrics = {}
            self.watch_empty_label.setText("清单读取失败：%s" % exc)
            self.watch_summary_label.setText("清单读取失败：%s" % exc)
            self.watch_stack.setCurrentWidget(self.watch_empty_label)
            self._fill_compare_table()
            return
        if path is None:
            self._watchlist = None
            self._watchlist_path = None
            self._strategy_metrics = {}
            message = "未找到 watchlist_*.csv，请先在 Smart Beta 页生成并导出"
            self.watch_empty_label.setText(message)
            self.watch_summary_label.setText(message)
            self.watch_stack.setCurrentWidget(self.watch_empty_label)
            self._fill_compare_table()
            return
        self._watchlist = df
        self._watchlist_path = path
        try:
            self._strategy_metrics = _load_strategy_metrics()
        except Exception:
            self._strategy_metrics = {}
        self.watch_empty_label.setText("暂无调仓清单，请先在 Smart Beta 页生成并导出")
        self._fill_watchlist_table()
        self._fill_compare_table()
        self.window.statusBar().showMessage(
            "已读取调仓清单：%s（只读，不下单）" % path.name, 5000
        )

    def _fill_watchlist_table(self, _index=None):
        """按组合筛选填充表格；表格只读，单元格不提供编辑入口。"""
        df = self._watchlist
        if df is None or df.empty:
            self.watch_stack.setCurrentWidget(self.watch_empty_label)
            return
        key = self.watch_filter_combo.currentData()
        view = df
        if key == "五因子":
            view = df[df["组合"].astype(str).str.startswith("五因子")]
        elif key == "四因子":
            view = df[df["组合"].astype(str).str.startswith("四因子")]
        view = view.reset_index(drop=True)
        if view.empty:
            self.watch_empty_label.setText("当前筛选下没有记录")
            self.watch_stack.setCurrentWidget(self.watch_empty_label)
            self.watch_summary_label.setText("当前筛选下没有记录")
            return

        info = _watchlist_summary(view)
        weighted = "—" if info["weighted_return"] is None else "%+.2f%%" % info["weighted_return"]
        self.watch_summary_label.setText(
            "%s ｜ %d 行 ｜ 调仓日 %s ｜ 收益截止 %s ｜ 权重合计 %.2f%% ｜ 加权下期收益 %s"
            % (
                self._watchlist_path.name, info["rows"], info["rebalance_date"],
                info["return_end"], info["weight_sum"], weighted,
            )
        )

        self.watch_table.setRowCount(len(view))
        for row_index, row in view.iterrows():
            for col_index, column in enumerate(WATCHLIST_COLUMNS):
                value = row[column]
                numeric = column not in ("代码", "名称", "组合", "因子归属", "调仓日", "收益截止日")
                set_cell(
                    self.watch_table, row_index, col_index,
                    self._format_watch_value(column, value), numeric=numeric,
                )
                item = self.watch_table.item(row_index, col_index)
                if item is None:
                    continue
                # 双重保证只读：除 NoEditTriggers 外，再移除单元格可编辑标志。
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if column == "下期收益(%)":
                    try:
                        ret = float(value)
                    except (TypeError, ValueError):
                        ret = 0.0
                    if ret:
                        item.setForeground(Qt.red if ret > 0 else Qt.darkGreen)
        self.watch_stack.setCurrentWidget(self.watch_table)

    def _fill_compare_table(self):
        """用当期清单与全期核对结果填五因子/四因子对比表（只读）。"""
        df = self._watchlist
        if df is None or df.empty:
            self.compare_table.setRowCount(0)
            self.compare_note.setText("暂无对比数据：请先生成调仓清单")
            return
        five = df[df["组合"].astype(str).str.startswith("五因子")]
        four = df[df["组合"].astype(str).str.startswith("四因子")]
        if five.empty or four.empty:
            self.compare_table.setRowCount(0)
            self.compare_note.setText("清单中缺少五因子或四因子组合，无法对比")
            return

        codes_five = set(five["代码"])
        codes_four = set(four["代码"])
        overlap = codes_five & codes_four
        five_only = "、".join(five[~five["代码"].isin(codes_four)]["名称"].astype(str)) or "—"
        four_only = "、".join(four[~four["代码"].isin(codes_five)]["名称"].astype(str)) or "—"

        def _weight_stats(frame):
            weight = pd.to_numeric(frame["权重(%)"], errors="coerce").fillna(0.0)
            ret = pd.to_numeric(frame["下期收益(%)"], errors="coerce").fillna(0.0)
            total = float(weight.sum())
            return total, (float((weight * ret).sum() / total) if total else 0.0)

        five_weight, five_ret = _weight_stats(five)
        four_weight, four_ret = _weight_stats(four)
        five_m = self._strategy_metrics.get("five_always", {})
        four_m = self._strategy_metrics.get("four_old", {})

        def _num(value):
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                return None
            return float(value)

        def _pct(value):
            v = _num(value)
            return "—" if v is None else "%.2f%%" % v

        def _ratio(value):
            v = _num(value)
            return "—" if v is None else "%.2f%%" % (v * 100)

        def _signed(value):
            v = _num(value)
            return "—" if v is None else "%+.2f%%" % v

        def _sharpe(value):
            v = _num(value)
            return "—" if v is None else "%.3f" % v

        def _passed(metric):
            if not metric:
                return "—"
            value = metric.get("passed")
            if isinstance(value, str):
                return "通过" if value.strip().lower() == "true" else "未通过"
            if value is None or pd.isna(value):
                return "—"
            return "通过" if bool(value) else "未通过"

        rows = [
            ("组合规则", "并集修复 + 动量最低 10%", "旧并集规则（不设动量下限）"),
            ("当期持仓数", str(len(five)), str(len(four))),
            ("当期权重合计", _pct(five_weight), _pct(four_weight)),
            ("当期加权下期收益", _signed(five_ret), _signed(four_ret)),
            (
                "当期上涨 / 下跌",
                "%d / %d" % ((five["下期收益(%)"] > 0).sum(), (five["下期收益(%)"] < 0).sum()),
                "%d / %d" % ((four["下期收益(%)"] > 0).sum(), (four["下期收益(%)"] < 0).sum()),
            ),
            ("与对方重叠", "%d 只" % len(overlap), "%d 只" % len(overlap)),
            ("组合独有", five_only, four_only),
            ("全期年化收益", _ratio(five_m.get("annual_return")), _ratio(four_m.get("annual_return"))),
            ("全期最大回撤", _ratio(five_m.get("max_drawdown")), _ratio(four_m.get("max_drawdown"))),
            ("基准最大回撤", _ratio(five_m.get("benchmark_max_drawdown")), _ratio(four_m.get("benchmark_max_drawdown"))),
            ("全期夏普", _sharpe(five_m.get("sharpe")), _sharpe(four_m.get("sharpe"))),
            ("判定（年化>0 且回撤≤11.95%）", _passed(five_m), _passed(four_m)),
        ]

        self.compare_table.setRowCount(len(rows))
        for row_index, (label, left, right) in enumerate(rows):
            for col_index, text in enumerate((label, left, right)):
                set_cell(self.compare_table, row_index, col_index, text)
                item = self.compare_table.item(row_index, col_index)
                if item is None:
                    continue
                # 与清单表一致：移除可编辑标志，保证只读。
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setToolTip(text)
                if label == "当期加权下期收益" and col_index > 0:
                    value = five_ret if col_index == 1 else four_ret
                    if value:
                        item.setForeground(Qt.red if value > 0 else Qt.darkGreen)

        self.compare_note.setText(
            "当期：%s → %s；全期：2024-05-31 → 2026-11-13，300 只池，等权基准，毛收益。"
            % (str(five["调仓日"].iloc[0]), str(five["收益截止日"].iloc[0]))
        )

    @staticmethod
    def _format_watch_value(column, value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        if column == "综合百分位和":
            return "%.4f" % float(value)
        if column in ("动量(%)", "年化波动(%)", "权重(%)", "下期收益(%)", "PE(TTM)"):
            return "%.2f" % float(value)
        if column == "PB(MRQ)":
            return "%.3f" % float(value)
        return str(value)

    def load_positions(self):
        """从 DB 读取持仓并刷新表格，不联网。"""
        try:
            from core.position import load_positions as _load
            df = _load()
        except Exception as exc:
            self.window.show_error("读取持仓失败", traceback.format_exc(), str(exc))
            return
        self._fill_position_table(df)

    def _fill_position_table(self, df):
        if df is None or df.empty:
            self.position_stack.setCurrentWidget(self.position_empty_label)
            return
        df = df.copy()
        # 刷新前 current_price/pnl/pnl_ratio 列不存在，补齐为空
        for col in ("current_price", "pnl", "pnl_ratio"):
            if col not in df.columns:
                df[col] = None
        self.position_table.setRowCount(len(df))
        for row_index, (_, row) in enumerate(df.iterrows()):
            for col_index, (_label, key, numeric) in enumerate(self.POS_COLUMNS):
                value = row.get(key) if key in df.columns else None
                text = self._format_cell(key, value)
                user_data = int(row["id"]) if key == "code" else None
                set_cell(
                    self.position_table, row_index, col_index, text,
                    user_data=user_data, numeric=numeric,
                )
                # 盈亏比例着色
                if key == "pnl_ratio" and value is not None and pd.notna(value):
                    item = self.position_table.item(row_index, col_index)
                    item.setForeground(Qt.red if value < 0 else Qt.darkGreen)
        self.position_stack.setCurrentWidget(self.position_table)

    @staticmethod
    def _format_cell(key, value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if key in ("cost_price", "current_price", "target_price"):
            return "%.3f" % float(value)
        if key == "pnl":
            return "%.2f" % float(value)
        if key == "pnl_ratio":
            return "%.2f%%" % (float(value) * 100)
        if key in ("stop_loss", "take_profit"):
            v = float(value)
            return "%.2f%%" % (v * 100) if v > 0 else "—"
        if key == "days_held":
            return "%d" % int(value)
        if key == "force_sell":
            return "是" if bool(value) else ""
        if key == "shares":
            return str(int(value))
        return str(value)

    def load_row_to_form(self, row, _column):
        item = self.position_table.item(row, 0)  # code 列，存了 position_id
        if item is None:
            return
        position_id = item.data(Qt.UserRole)
        if position_id is None:
            return
        try:
            from core.position import load_positions as _load
            df = _load()
            row_df = df[df["id"] == position_id]
            if row_df.empty:
                return
            r = row_df.iloc[0]
        except Exception:
            return
        self.editing_id = int(position_id)
        self.code_edit.setText(str(r["code"]))
        self.name_edit.setText(str(r["name"]))
        self.shares_spin.setValue(int(r["shares"]))
        self.cost_spin.setValue(float(r["cost_price"]))
        try:
            parts = str(r["buy_date"]).split("-")
            if len(parts) == 3:
                self.buy_date_edit.setDate(QDate(int(parts[0]), int(parts[1]), int(parts[2])))
        except Exception:
            pass
        self.stop_loss_spin.setValue(float(r["stop_loss"]))
        self.take_profit_spin.setValue(float(r["take_profit"]))
        self.target_price_spin.setValue(float(r["target_price"]))
        self.note_edit.setText(str(r["note"]))
        self.save_button.setText("保存修改")
        self.window.statusBar().showMessage("已载入持仓 %s 到编辑表单" % r["code"], 5000)

    def save_position(self):
        code_raw = self.code_edit.text().strip()
        if not code_raw:
            QMessageBox.warning(self, "无法保存", "请输入股票代码")
            return
        try:
            codes = parse_codes(code_raw)
            if len(codes) != 1:
                raise ValueError("请输入单个股票代码")
            code = codes[0]
        except ValueError as exc:
            QMessageBox.warning(self, "无法保存", str(exc))
            return
        try:
            from core.position import save_position as _save
            _save(
                code=code,
                name=self.name_edit.text().strip(),
                shares=self.shares_spin.value(),
                cost_price=self.cost_spin.value(),
                buy_date=self.buy_date_edit.date().toString("yyyy-MM-dd"),
                stop_loss=self.stop_loss_spin.value(),
                take_profit=self.take_profit_spin.value(),
                target_price=self.target_price_spin.value(),
                note=self.note_edit.text().strip(),
                position_id=self.editing_id,
            )
        except Exception as exc:
            self.window.show_error("保存持仓失败", traceback.format_exc(), str(exc))
            return
        action = "更新" if self.editing_id else "新增"
        self.reset_form()
        self.load_positions()
        self.window.statusBar().showMessage("持仓已%s：%s" % (action, code), 5000)

    def reset_form(self):
        self.editing_id = None
        self.code_edit.clear()
        self.name_edit.clear()
        self.shares_spin.setValue(0)
        self.cost_spin.setValue(0)
        self.stop_loss_spin.setValue(0)
        self.take_profit_spin.setValue(0)
        self.target_price_spin.setValue(0)
        self.note_edit.clear()
        self.save_button.setText("新增持仓")

    def delete_selected(self):
        row = self.position_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "未选择持仓", "请先在表格中选择一行")
            return
        item = self.position_table.item(row, 0)
        position_id = item.data(Qt.UserRole) if item else None
        if position_id is None:
            return
        code = item.text() if item else ""
        answer = QMessageBox.question(
            self, "确认删除", "确定删除持仓 %s 吗？" % code,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            from core.position import delete_position as _delete
            _delete(position_id)
        except Exception as exc:
            self.window.show_error("删除持仓失败", traceback.format_exc(), str(exc))
            return
        if self.editing_id == position_id:
            self.reset_form()
        self.load_positions()
        self.window.statusBar().showMessage("已删除持仓 %s" % code, 5000)

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入持仓 CSV", "", "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            from core.position import import_positions_csv as _import
            n = _import(path)
        except Exception as exc:
            self.window.show_error("导入失败", traceback.format_exc(), str(exc))
            return
        self.load_positions()
        QMessageBox.information(self, "导入完成", "已导入 %d 条持仓" % n)
        self.window.statusBar().showMessage("已导入 %d 条持仓" % n, 5000)

    def export_template(self):
        db.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        default_path = db.EXPORT_DIR / "positions_template.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出持仓 CSV 模板", str(default_path), "CSV 文件 (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            content = (
                "code,name,shares,cost_price,buy_date,stop_loss,take_profit,target_price,note\n"
                "600519,贵州茅台,100,1680.000,2024-01-01,0.08,0.15,1900,示例行可删除\n"
            )
            Path(path).write_text(content, encoding="utf-8-sig")
        except Exception as exc:
            self.window.show_error("导出模板失败", traceback.format_exc(), str(exc))
            return
        self.window.statusBar().showMessage("模板已导出：" + path, 5000)

    def refresh_positions(self):
        """后台用本地日线刷新持仓行情并生成预警，不联网。"""
        def task(progress):
            progress.emit({"total": 0, "text": "正在用本地日线刷新持仓，不联网"})
            from core.position import refresh_positions as _refresh
            return _refresh()

        self.window.start_task(task, self.refresh_finished, "正在刷新持仓行情")

    def refresh_finished(self, result):
        alerts = self._apply_refresh(result.get("frame"), result.get("alerts", []))
        msg = "刷新完成，共 %d 条预警" % len(alerts) if alerts else "刷新完成，暂无预警"
        self.window.statusBar().showMessage(msg, 10000)

    def _apply_refresh(self, frame, alerts):
        """持仓表与预警的统一入口：合并满5天/止损等预警与已载入的 SELL 信号。"""
        self._fill_position_table(frame)
        merged = list(alerts or [])
        if frame is not None and self.sell_signals is not None:
            try:
                merged += hammer_boll.sell_alerts(frame, self.sell_signals)
            except Exception:
                pass
        self._fill_alerts(merged)
        self.alert_label.setText(
            "预警（满5天强制卖出 / SELL信号 / 止损 / 止盈 / 目标价 / 无行情）：共 %d 条" % len(merged)
        )
        return merged

    def run_hammer_backtest(self):
        """用本地日线回放 HAMMER 买卖公式，生成 hammer_trades.csv 作为卖点核对清单。"""
        try:
            from core.position import load_positions as _load
            codes = [str(code).zfill(6) for code in _load()["code"].tolist()]
        except Exception as exc:
            self.window.show_error("读取持仓失败", traceback.format_exc(), str(exc))
            return
        if not codes:
            QMessageBox.warning(self, "无法回测", "当前没有持仓，请先新增或导入持仓 CSV")
            return

        def task(progress):
            from core.position import refresh_positions as _refresh
            result = hammer_boll.run_sell_backtest(codes, start="2024-01-01", progress=progress)
            result["signals"] = hammer_boll.load_sell_signals(result["path"])
            refreshed = _refresh()
            result["frame"] = refreshed["frame"]
            result["alerts"] = refreshed["alerts"]
            return result

        self.window.start_task(task, self._hammer_finished, "正在回放 HAMMER 卖点")

    def _hammer_finished(self, result):
        self.sell_signals = result.get("signals")
        alerts = self._apply_refresh(result.get("frame"), result.get("alerts", []))
        self.window.statusBar().showMessage(
            "hammer_trades.csv 已生成：%d 笔成交；预警 %d 条"
            % (len(result.get("rows", [])), len(alerts)), 12000
        )
        QMessageBox.information(
            self, "卖点回测完成",
            "成交记录：\n%s\n\n共 %d 笔（%d 只股票），期末仍持仓 %d 笔。\n"
            "仅用于卖点核对，不构成投资建议，不会下单。"
            % (result.get("path"), len(result.get("rows", [])),
               result.get("codes", 0), result.get("open_count", 0))
        )

    def import_sell_signals(self):
        """导入通达信 SELL 扫描结果或 hammer_trades.csv，只做预警核对。"""
        path, _ = QFileDialog.getOpenFileName(self, "导入 SELL 结果 CSV", "", "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            signals = hammer_boll.load_sell_signals(path)
        except Exception as exc:
            self.window.show_error("导入 SELL 结果失败", traceback.format_exc(), str(exc))
            return
        if signals.empty:
            QMessageBox.warning(self, "没有可用信号", "文件中未找到有效的代码/日期记录")
            return
        self.sell_signals = signals
        self.refresh_positions()
        self.window.statusBar().showMessage(
            "已载入 %d 条 SELL 信号（%s）；仅做预警，不自动清仓"
            % (len(signals), Path(path).name), 10000
        )

    def _fill_alerts(self, alerts):
        if not alerts:
            self.alert_stack.setCurrentWidget(self.alert_empty_label)
            return
        self.alert_table.setRowCount(len(alerts))
        for row_index, alert in enumerate(alerts):
            set_cell(self.alert_table, row_index, 0, alert.get("type"))
            set_cell(self.alert_table, row_index, 1, alert.get("code"))
            set_cell(self.alert_table, row_index, 2, alert.get("name"))
            set_cell(self.alert_table, row_index, 3, alert.get("message"))
        self.alert_stack.setCurrentWidget(self.alert_table)


class SimulationPage(QWidget):
    """纸面仿真页（I3-sim-01/02）：5 个只读标签页 + 6 个带依赖锁的操作按钮。

    单期流程：调仓日生成目标 → 次一交易日开盘买入 → 持有到收益截止日；
    阈值监控子页展示硬触发（ST/停牌/涨跌停/熔断）与权重漂移队列。
    全流程只用本地 SQLite 与本地清单：不联网、不接券商、不发送真实委托。
    产物统一写入 data/sim/。仅研究用途，不构成投资建议。
    """

    HOLD_HEADERS = [
        "代码", "名称", "持股数", "可用股数", "今日买入", "成本价", "最新价",
        "持仓市值", "浮动盈亏", "账户权重(%)", "建仓日",
    ]
    CARDS = [
        ("total", "总资产"), ("cash", "可用现金"), ("market_value", "持仓市值"),
        ("cumulative", "累计收益"), ("daily", "日盈亏"), ("holdings", "持仓数"),
        ("exposure", "策略暴露"), ("buffer", "现金缓冲"),
    ]
    DERIVED_FILES = (TARGETS_CSV, TRADES_CSV, NAV_CSV, REPORT_HTML,
                     EMERGENCY_CSV, DRIFT_CSV)
    HARD_HEADERS = {
        "code": "代码", "action": "动作", "reason": "原因", "priority": "优先级",
    }
    DRIFT_HEADERS = {
        "code": "代码/对象", "current_weight": "当前权重(%)",
        "target_weight": "目标权重(%)", "drift%": "偏离(pp)",
        "action": "动作", "shares_to_trade": "调整股数",
    }
    TRADE_HEADERS = {
        "trade_id": "编号", "date": "日期", "side": "方向", "code": "代码",
        "name": "名称", "shares": "数量", "price": "价格", "value": "金额",
        "commission": "佣金", "tax": "印花税", "transfer": "过户费",
        "slippage": "滑点", "status": "状态", "reason": "原因",
    }
    NAV_HEADERS = {
        "date": "日期", "cash": "现金", "market_value": "持仓市值",
        "total_asset": "总资产", "daily_return": "当日收益(%)",
        "cumulative_return": "累计收益(%)",
    }

    def __init__(self, window):
        super().__init__()
        self.window = window
        layout = QVBoxLayout(self)

        title = QLabel(
            "纸面仿真：读取最新五因子清单，在本地生成虚拟账户、目标持仓、"
            "单期买入成交与持有期净值。全程离线，不接券商、不发送真实委托。")
        title.setWordWrap(True)
        layout.addWidget(title)

        action_row = QHBoxLayout()
        self.init_button = QPushButton("初始化账户")
        self.init_button.setToolTip("输入初始资金并重建虚拟账户（默认 100 万）")
        self.init_button.clicked.connect(self.init_account)
        self.targets_button = QPushButton("加载目标")
        self.targets_button.clicked.connect(self.load_targets)
        self.exec_button = QPushButton("执行买入")
        self.exec_button.clicked.connect(self.execute_trades)
        self.nav_button = QPushButton("刷新净值")
        self.nav_button.clicked.connect(self.refresh_nav)
        self.threshold_button = QPushButton("扫描阈值")
        self.threshold_button.setToolTip("盘后扫描硬触发与权重漂移")
        self.threshold_button.clicked.connect(self.scan_thresholds)
        self.report_button = QPushButton("导出报告")
        self.report_button.clicked.connect(self.export_report)
        for button in (self.init_button, self.targets_button,
                       self.exec_button, self.nav_button,
                       self.threshold_button, self.report_button):
            action_row.addWidget(button)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.sim_tabs = QTabWidget()
        self.overview_tab = self._build_overview()
        self.targets_table = self._make_table(["暂无"])
        self.trades_table = self._make_table(["暂无"])
        self.nav_tab = self._build_nav_tab()
        self.threshold_tab = self._build_threshold_tab()
        self.sim_tabs.addTab(self.overview_tab, "账户概览")
        self.sim_tabs.addTab(self.targets_table, "目标持仓")
        self.sim_tabs.addTab(self.trades_table, "成交记录")
        self.sim_tabs.addTab(self.nav_tab, "净值曲线")
        self.sim_tabs.addTab(self.threshold_tab, "阈值监控")
        layout.addWidget(self.sim_tabs, 1)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #a33; font-weight: bold;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._refresh_all()

    @staticmethod
    def _make_table(headers):
        table = QTableWidget()
        prepare_table(table, headers)
        return table

    def _build_overview(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        cards_box = QWidget()
        grid = QGridLayout(cards_box)
        grid.setContentsMargins(0, 0, 0, 0)
        self.card_values = {}
        for index, (key, label) in enumerate(self.CARDS):
            card = QWidget()
            card.setObjectName("simCard")
            card.setStyleSheet(
                "QWidget#simCard { border: 1px solid #d0d0d0; border-radius: 6px;"
                " background: #fafafa; }")
            box = QVBoxLayout(card)
            box.setContentsMargins(10, 6, 10, 6)
            name = QLabel(label)
            name.setStyleSheet("border: none; color: #666; font-size: 12px;")
            value = QLabel("—")
            value.setStyleSheet("border: none; font-size: 18px; font-weight: bold;")
            box.addWidget(name)
            box.addWidget(value)
            grid.addWidget(card, index // 4, index % 4)
            self.card_values[key] = value
        layout.addWidget(cards_box)
        label = QLabel("账户持仓明细（只读）")
        label.setStyleSheet("font-weight: bold; margin-top: 6px;")
        layout.addWidget(label)
        self.holdings_table = self._make_table(self.HOLD_HEADERS)
        layout.addWidget(self.holdings_table, 1)
        return page

    def _build_nav_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.nav_figure = Figure(figsize=(6, 2.4))
        self.nav_canvas = FigureCanvasQTAgg(self.nav_figure)
        self.nav_canvas.setMinimumHeight(220)
        layout.addWidget(self.nav_canvas)
        self.nav_table = self._make_table(["暂无"])
        layout.addWidget(self.nav_table, 1)
        return page

    def _build_threshold_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        header = QHBoxLayout()
        self.hard_status_label = QLabel("硬触发：未扫描")
        self.breaker_label = QLabel("熔断：—")
        self.next_check_label = QLabel("下次检查：—")
        for label in (self.hard_status_label, self.breaker_label,
                      self.next_check_label):
            header.addWidget(label)
        header.addStretch()
        layout.addLayout(header)
        note = QLabel(
            "硬触发优先于漂移：清仓股票不再进入微调队列；同一股票多信号合并为一条。"
            "本地缺行业字段时自动跳过板块微调（阈值 5%）。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)
        layout.addWidget(QLabel("硬触发动作（emergency_actions.csv）"))
        self.hard_table = self._make_table(["暂无"])
        layout.addWidget(self.hard_table, 1)
        layout.addWidget(QLabel("权重漂移队列（rebalance_queue.csv）"))
        self.drift_table = self._make_table(["暂无"])
        layout.addWidget(self.drift_table, 1)
        return page

    @staticmethod
    def _read_csv(path):
        try:
            if not Path(path).exists() or Path(path).stat().st_size == 0:
                return pd.DataFrame()
            return pd.read_csv(path, encoding="utf-8-sig", dtype={"代码": str})
        except Exception:
            return pd.DataFrame()

    def _fill_frame(self, table, frame):
        """把 CSV 内容填入只读表格，空值显示为空。"""
        if frame is None or frame.empty:
            table.setRowCount(0)
            return
        prepare_table(table, [str(column) for column in frame.columns])
        table.setRowCount(len(frame))
        for row, record in enumerate(frame.to_dict("records")):
            for column, key in enumerate(frame.columns):
                value = record[key]
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    text = ""
                else:
                    text = str(value)
                set_cell(table, row, column, text, numeric=column >= 2)

    @staticmethod
    def _display_frame(frame, headers, percent_columns=()):
        """把英文 CSV 列转成中文表头展示（不改动落盘文件）。"""
        if frame is None or frame.empty:
            return frame
        view = frame.copy()
        if "side" in view.columns:
            view["side"] = view["side"].map(
                {"BUY": "买入", "SELL": "卖出"}).fillna(view["side"])
        for column in percent_columns:
            if column in view.columns:
                view[column] = (
                    pd.to_numeric(view[column], errors="coerce") * 100
                ).round(4)
        return view.rename(columns=headers)

    def _cutoff_date(self, nav):
        """数据截止日：优先净值末日，其次成交日，最后本地最新交易日。"""
        if nav is not None and not nav.empty and "date" in nav.columns:
            return str(nav["date"].iloc[-1])
        account = load_account()
        if account and account.get("last_exec_date"):
            return str(account["last_exec_date"])
        try:
            return latest_trade_date()
        except Exception:
            return ""

    def _refresh_all(self):
        """重新读取 data/sim 产物，刷新卡片、表格、曲线与按钮依赖锁。"""
        account = load_account()
        targets = self._read_csv(TARGETS_CSV)
        trades = self._read_csv(TRADES_CSV)
        nav = self._read_csv(NAV_CSV)
        hard = self._read_csv(EMERGENCY_CSV)
        drift = self._read_csv(DRIFT_CSV)
        self._fill_frame(self.targets_table, targets)
        self._fill_frame(
            self.trades_table,
            self._display_frame(trades, self.TRADE_HEADERS))
        self._fill_frame(
            self.nav_table,
            self._display_frame(
                nav, self.NAV_HEADERS, ("daily_return", "cumulative_return")))
        self._update_cards(account, nav)
        self._fill_holdings(account, nav)
        self._draw_curve(nav)
        self._fill_threshold(hard, drift, nav)
        self._update_status(account, nav)
        self._update_locks(account)

    def _update_cards(self, account, nav):
        if account is None:
            for label in self.card_values.values():
                label.setText("—")
            return
        if nav.empty:
            total = float(account.get("cash", 0.0) or 0.0)
            cash, market_value, cumulative, daily = total, 0.0, 0.0, 0.0
        else:
            last = nav.iloc[-1]
            total = float(last.get("total_asset", 0.0) or 0.0)
            cash = float(last.get("cash", 0.0) or 0.0)
            market_value = float(last.get("market_value", 0.0) or 0.0)
            cumulative = float(last.get("cumulative_return", 0.0) or 0.0)
            daily = (total - float(nav.iloc[-2]["total_asset"])
                     if len(nav) > 1 else 0.0)
        exposure = market_value / total if total else 0.0
        buffer = cash / total if total else 0.0
        values = {
            "total": "{:,.2f} 元".format(total),
            "cash": "{:,.2f} 元".format(cash),
            "market_value": "{:,.2f} 元".format(market_value),
            "cumulative": "{:+.2f}%".format(cumulative * 100),
            "daily": "{:+,.2f} 元".format(daily),
            "holdings": "%d 只" % len(account.get("positions", {})),
            "exposure": "%.2f%%" % (exposure * 100),
            "buffer": "%.2f%%" % (buffer * 100),
        }
        for key, value in values.items():
            self.card_values[key].setText(value)

    def _fill_threshold(self, hard, drift, nav):
        """阈值监控：状态灯、熔断状态、下次检查时间与两张只读表。"""
        self._fill_frame(
            self.hard_table, self._display_frame(hard, self.HARD_HEADERS))
        if drift is not None and not drift.empty:
            view = drift.copy()
            for column in ("current_weight", "target_weight"):
                if column in view.columns:
                    view[column] = (
                        pd.to_numeric(view[column], errors="coerce") * 100
                    ).round(2)
            view = view.rename(columns=self.DRIFT_HEADERS)
        else:
            view = drift
        self._fill_frame(self.drift_table, view)

        unscanned = not EMERGENCY_CSV.exists() and not DRIFT_CSV.exists()
        actions = set(hard["action"]) if not hard.empty else set()
        codes = set(hard["code"]) if not hard.empty else set()
        breaker = "正常"
        if "只许减仓" in actions:
            breaker = "只许减仓"
        elif "暂停买入" in actions and "ACCOUNT" in codes:
            breaker = "暂停买入"
        if unscanned:
            text, color = "硬触发：未扫描", "#7f8c8d"
        elif actions & {"清仓", "复牌清仓", "只许减仓"}:
            text, color = "硬触发：红 · 紧急", "#c0392b"
        elif not hard.empty or not drift.empty or breaker != "正常":
            text, color = "硬触发：黄 · 关注", "#e67e22"
        else:
            text, color = "硬触发：绿 · 正常", "#27ae60"
        self.hard_status_label.setText(text)
        self.hard_status_label.setStyleSheet(
            "color: #fff; background: %s; padding: 4px 10px; border-radius: 4px;"
            " font-weight: bold;" % color)
        self.breaker_label.setText(
            "熔断：%s" % ("—" if unscanned else breaker))
        try:
            check = next_check_time(self._cutoff_date(nav))
        except Exception:
            check = "—"
        self.next_check_label.setText(
            "下次检查：%s" % ("—" if unscanned else check))

    def _fill_holdings(self, account, nav):
        if account is None or not account.get("positions"):
            self.holdings_table.setRowCount(0)
            return
        date = self._cutoff_date(nav)
        price_map = load_price_map(list(account["positions"]), date, date) if date else {}
        self._fill_frame(self.holdings_table, holdings_frame(account, price_map, date))

    def _draw_curve(self, nav):
        self.nav_figure.clear()
        axes = self.nav_figure.add_subplot(111)
        if nav.empty or "total_asset" not in nav.columns or "date" not in nav.columns:
            axes.text(0.5, 0.5, "暂无净值数据", ha="center", va="center",
                      transform=axes.transAxes)
            axes.set_axis_off()
        else:
            values = pd.to_numeric(nav["total_asset"], errors="coerce")
            axes.plot(list(range(len(values))), values, color="#1f4e79",
                      linewidth=1.6, label="策略总资产")
            account = load_account()
            targets = self._read_csv(TARGETS_CSV)
            if account is not None and not targets.empty:
                benchmark = benchmark_curve(
                    [str(code).zfill(6) for code in targets["代码"]],
                    str(nav["date"].iloc[0]), str(nav["date"].iloc[-1]),
                    float(account.get("initial_cash", 0.0) or 0.0))
                if not benchmark.empty:
                    mapping = dict(zip(
                        benchmark["date"].astype(str),
                        benchmark["benchmark_value"]))
                    bench_values = [
                        mapping.get(str(date)) for date in nav["date"]]
                    axes.plot(list(range(len(bench_values))), bench_values,
                              color="#c0504d", linewidth=1.4,
                              linestyle="--", label="等权基准")
            axes.legend(loc="best", fontsize=7)
            step = max(len(values) // 8, 1)
            ticks = list(range(0, len(values), step))
            axes.set_xticks(ticks)
            axes.set_xticklabels([str(nav["date"].iloc[i]) for i in ticks],
                                 rotation=30, fontsize=7)
            axes.set_title("纸面账户总资产（元）", fontsize=10)
            axes.grid(True, linestyle="--", alpha=0.4)
            axes.tick_params(labelsize=8)
        self.nav_figure.tight_layout()
        self.nav_canvas.draw_idle()

    def _update_status(self, account, nav):
        if nav is not None and not nav.empty and "date" in nav.columns:
            period = "持有期 %s ~ %s" % (nav["date"].iloc[0], nav["date"].iloc[-1])
        elif account and account.get("nav_start"):
            period = "持有期 %s ~ %s" % (
                account.get("nav_start"), account.get("nav_end") or "—")
        else:
            period = "持有期 —"
        text = "纸面仿真 · 非真实交易 · %s" % period
        if account is None:
            text += "｜账户未初始化"
        self.status_label.setText(text + "｜" + DISCLAIMER + "。")

    def _update_locks(self, account):
        """按钮依赖锁：后一步必须等前一步产物存在才可用。"""
        has_account = account is not None
        has_targets = TARGETS_CSV.exists()
        has_trades = TRADES_CSV.exists()
        has_nav = NAV_CSV.exists()
        self.init_button.setEnabled(True)
        self.targets_button.setEnabled(has_account)
        self.exec_button.setEnabled(has_account and has_targets)
        self.nav_button.setEnabled(has_account and has_trades)
        self.threshold_button.setEnabled(has_account and has_targets)
        self.report_button.setEnabled(has_account and has_nav)
        for button, ready, tip in (
            (self.targets_button, has_account, "请先初始化账户"),
            (self.exec_button, has_account and has_targets,
             "请先初始化账户并加载目标持仓"),
            (self.nav_button, has_account and has_trades,
             "请先初始化账户并执行买入"),
            (self.threshold_button, has_account and has_targets,
             "请先初始化账户并加载目标持仓"),
            (self.report_button, has_account and has_nav,
             "请先初始化账户并刷新净值"),
        ):
            button.setToolTip("" if ready else tip)

    def init_account(self):
        """弹窗输入初始资金（默认 100 万），重建虚拟账户。"""
        if load_account() is not None:
            answer = QMessageBox.question(
                self, "初始化虚拟账户",
                "将重置虚拟账户并清理旧的仿真产物（目标持仓、成交、净值、报告）。"
                "是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        cash, accepted = QInputDialog.getDouble(
            self, "初始化账户", "初始资金（元）", 1000000.0,
            10000.0, 100000000.0, 0)
        if not accepted:
            return
        try:
            init_account(float(cash), reset=True)
            for path in self.DERIVED_FILES:
                Path(path).unlink(missing_ok=True)
        except Exception as exc:
            self.window.show_error("初始化账户失败", traceback.format_exc(), str(exc))
            return
        self.sim_tabs.setCurrentWidget(self.overview_tab)
        self._refresh_all()
        self.window.statusBar().showMessage(
            "虚拟账户已初始化：初始资金 %.2f 元" % float(cash), 8000)

    def _run_step(self, title, function, handler):
        try:
            result = function()
        except Exception as exc:
            self.window.show_error(title + "失败", traceback.format_exc(), str(exc))
            self.window.statusBar().showMessage(
                title + "失败：" + str(exc), 10000)
            return
        self._refresh_all()
        handler(result)

    def load_targets(self):
        self._run_step("加载目标持仓", run_sim_targets, self._targets_done)

    def _targets_done(self, result):
        self.sim_tabs.setCurrentWidget(self.targets_table)
        self.window.statusBar().showMessage(
            "目标持仓已生成：%d 只，权重合计 %.2f%%，成交日 %s｜%s" % (
                len(result["targets"]),
                float(result["targets"]["weight"].sum()) * 100,
                result["exec_date"], result["path"]), 10000)

    def execute_trades(self):
        self._run_step("执行买入撮合", run_sim_execution, self._execution_done)

    def _execution_done(self, result):
        summary = result["summary"]
        self.sim_tabs.setCurrentWidget(self.trades_table)
        self.window.statusBar().showMessage(
            "买入撮合完成：成交 %d 笔（部分 %d）｜拒单 %d 笔｜佣金 %.2f｜印花税 %.2f｜"
            "过户费 %.2f｜滑点 %.2f｜现金 %.2f 元" % (
                summary["filled"], summary["partial"], summary["rejected"],
                summary["commission"], summary["tax"], summary["transfer"],
                summary["slippage"], summary["cash"]), 15000)

    def refresh_nav(self):
        self._run_step("刷新净值", run_sim_nav, self._nav_done)

    def _nav_done(self, result):
        self.sim_tabs.setCurrentWidget(self.nav_tab)
        self.window.statusBar().showMessage(
            "净值已刷新：%s → %s｜总资产 %.2f 元｜累计 %+.2f%%｜最大回撤 %.2f%%｜"
            "基准 %+.2f%%｜超额 %+.2f%%" % (
                result["start_date"], result["end_date"], result["total"],
                result["cumulative"] * 100, result["max_drawdown"] * 100,
                result["benchmark_return"] * 100,
                result["excess_return"] * 100), 15000)

    def scan_thresholds(self):
        self._run_step("阈值扫描", run_threshold_scan, self._threshold_done)

    def _threshold_done(self, result):
        self.sim_tabs.setCurrentWidget(self.threshold_tab)
        self.window.statusBar().showMessage(
            "阈值扫描完成：扫描日 %s｜硬触发 %d 条｜漂移 %d 条｜熔断 %s｜"
            "下次检查 %s" % (
                result["scan_date"], len(result["hard"]), len(result["drift"]),
                result["breaker"], result["next_check"]), 15000)

    def export_report(self):
        self._run_step("导出离线报告", build_sim_report, self._report_done)

    def _report_done(self, path):
        self.window.statusBar().showMessage("离线报告已生成：%s" % path, 15000)


class RollingPage(QWidget):
    """多期滚动页（I3-sim-03）：月末重算 + 每日阈值干预的历史回放。

    输入回放区间与初始资金，输出全期净值、逐期统计、成交明细、
    阈值干预记录与成本拆解；表格全部只读，状态栏标注纸面仿真与区间。
    仅使用本地 SQLite/CSV，不联网、不接券商、不下单。
    """

    NAV_HEADERS = ["日期", "现金", "持仓市值", "总资产", "当日收益(%)",
                   "累计收益(%)", "基准"]
    PERIOD_HEADERS = ["期号", "调仓日", "成交日", "期末日", "期收益",
                      "换手率", "滑点成本", "命中率", "新调入", "调出",
                      "阈值触发", "成交笔数"]
    TRADE_HEADERS = ["编号", "期号", "日期", "方向", "代码", "名称", "数量",
                     "价格", "金额", "佣金", "印花税", "过户费", "滑点",
                     "来源", "原因"]
    THRESHOLD_HEADERS = ["期号", "日期", "类型", "代码/对象", "动作",
                         "原因", "执行"]

    def __init__(self, window):
        super().__init__()
        self.window = window
        layout = QVBoxLayout(self)

        title = QLabel(
            "多期滚动纸面回测：每月末用五因子重新生成目标持仓，次一交易日开盘撮合，"
            "持有期内每日扫描阈值。全程离线，不接券商、不发送真实委托。")
        title.setWordWrap(True)
        layout.addWidget(title)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("起始月末"))
        self.start_edit = QDateEdit()
        self.start_edit.setCalendarPopup(True)
        self.start_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_edit.setDate(QDate(2024, 5, 31))
        controls.addWidget(self.start_edit)
        controls.addWidget(QLabel("终止月末"))
        self.end_edit = QDateEdit()
        self.end_edit.setCalendarPopup(True)
        self.end_edit.setDisplayFormat("yyyy-MM-dd")
        try:
            latest = latest_trade_date()
            self.end_edit.setDate(QDate.fromString(latest, "yyyy-MM-dd"))
        except Exception:
            self.end_edit.setDate(QDate.currentDate())
        controls.addWidget(self.end_edit)
        controls.addWidget(QLabel("初始资金"))
        self.cash_spin = QDoubleSpinBox()
        self.cash_spin.setRange(10000, 100000000)
        self.cash_spin.setDecimals(0)
        self.cash_spin.setSingleStep(100000)
        self.cash_spin.setValue(1000000)
        self.cash_spin.setSuffix(" 元")
        controls.addWidget(self.cash_spin)
        self.replay_button = QPushButton("开始回放")
        self.replay_button.setToolTip("按月末重算目标持仓并逐日回放阈值干预")
        self.replay_button.clicked.connect(self.start_replay)
        controls.addWidget(self.replay_button)
        self.report_button = QPushButton("生成报告")
        self.report_button.setToolTip("根据已落盘 CSV 生成离线汇总报告")
        self.report_button.clicked.connect(self.export_report)
        controls.addWidget(self.report_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.rolling_tabs = QTabWidget()
        self.nav_tab = self._build_nav_tab()
        self.periods_table = self._make_table(self.PERIOD_HEADERS)
        self.trades_table = self._make_table(self.TRADE_HEADERS)
        self.thresholds_table = self._make_table(self.THRESHOLD_HEADERS)
        self.costs_tab = self._build_costs_tab()
        self.rolling_tabs.addTab(self.nav_tab, "净值曲线")
        self.rolling_tabs.addTab(self.periods_table, "逐期统计")
        self.rolling_tabs.addTab(self.trades_table, "成交明细")
        self.rolling_tabs.addTab(self.thresholds_table, "阈值干预记录")
        self.rolling_tabs.addTab(self.costs_tab, "成本拆解")
        layout.addWidget(self.rolling_tabs, 1)

        self.status_label = QLabel("纸面仿真·非真实交易·尚未回放")
        self.status_label.setStyleSheet("color: #a33; font-weight: bold;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self._load_existing()

    @staticmethod
    def _make_table(headers):
        table = QTableWidget()
        prepare_table(table, headers)
        return table

    def _build_nav_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.nav_figure = Figure(figsize=(7, 2.8))
        self.nav_canvas = FigureCanvasQTAgg(self.nav_figure)
        self.nav_canvas.setMinimumHeight(240)
        layout.addWidget(self.nav_canvas)
        self.nav_table = self._make_table(self.NAV_HEADERS)
        layout.addWidget(self.nav_table, 1)
        return page

    def _build_costs_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.cost_label = QLabel("尚未回放。")
        self.cost_label.setWordWrap(True)
        layout.addWidget(self.cost_label)
        self.cost_table = self._make_table(["项目", "金额(元)", "占比(%)"])
        layout.addWidget(self.cost_table, 1)
        return page

    @staticmethod
    def _read_csv(path):
        try:
            if not Path(path).exists() or Path(path).stat().st_size == 0:
                return pd.DataFrame()
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _fill(table, headers, rows):
        prepare_table(table, headers)
        table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                set_cell(table, row_index, column, value, numeric=column > 0)

    def _load_existing(self):
        nav = self._read_csv(ROLLING_NAV_CSV)
        periods = self._read_csv(ROLLING_PERIODS_CSV)
        trades = self._read_csv(ROLLING_TRADES_CSV)
        thresholds = self._read_csv(ROLLING_THRESHOLDS_CSV)
        self._render(nav, periods, trades, thresholds)
        if not nav.empty:
            self.status_label.setText(
                "纸面仿真·非真实交易·回放区间 %s ~ %s（%d 期）"
                % (nav["date"].iloc[0], nav["date"].iloc[-1], len(periods)))

    def _render(self, nav, periods, trades, thresholds):
        self._draw_nav(nav)
        nav_rows = []
        for record in nav.tail(800).to_dict("records"):
            nav_rows.append([
                record.get("date"),
                "%.2f" % record.get("cash", 0.0),
                "%.2f" % record.get("market_value", 0.0),
                "%.2f" % record.get("total_asset", 0.0),
                "%.4f" % (record.get("daily_return", 0.0) * 100),
                "%.4f" % (record.get("cumulative_return", 0.0) * 100),
                "%.2f" % record.get("benchmark", 0.0),
            ])
        self._fill(self.nav_table, self.NAV_HEADERS, nav_rows)

        period_rows = []
        for record in periods.to_dict("records"):
            def percent(key):
                value = record.get(key)
                return "" if value is None or pd.isna(value) else "%.2f%%" % (float(value) * 100)
            period_rows.append([
                int(record.get("期号", 0)), record.get("调仓日"),
                record.get("成交日"), record.get("期末日"),
                percent("期收益"), percent("换手率"),
                "%.2f" % float(record.get("滑点成本", 0.0) or 0.0),
                percent("命中率"), percent("新调入平均收益"),
                percent("调出平均收益"), int(record.get("阈值触发次数", 0) or 0),
                int(record.get("成交笔数", 0) or 0),
            ])
        self._fill(self.periods_table, self.PERIOD_HEADERS, period_rows)

        trade_rows = []
        for record in trades.to_dict("records"):
            trade_rows.append([
                record.get("trade_id"), int(record.get("period", 0)),
                record.get("date"),
                {"BUY": "买入", "SELL": "卖出"}.get(record.get("side"),
                                                  record.get("side")),
                str(record.get("code", "")).zfill(6), record.get("name"),
                int(record.get("shares", 0) or 0),
                "%.2f" % float(record.get("price", 0.0) or 0.0),
                "%.2f" % float(record.get("value", 0.0) or 0.0),
                "%.2f" % float(record.get("commission", 0.0) or 0.0),
                "%.2f" % float(record.get("tax", 0.0) or 0.0),
                "%.2f" % float(record.get("transfer", 0.0) or 0.0),
                "%.2f" % float(record.get("slippage", 0.0) or 0.0),
                record.get("source"), record.get("reason"),
            ])
        self._fill(self.trades_table, self.TRADE_HEADERS, trade_rows)

        shown = thresholds.tail(3000)
        threshold_rows = []
        for record in shown.to_dict("records"):
            threshold_rows.append([
                int(record.get("period", 0)), record.get("date"),
                record.get("kind"), record.get("code"),
                record.get("action"), record.get("reason"),
                record.get("execution"),
            ])
        self._fill(self.thresholds_table, self.THRESHOLD_HEADERS,
                   threshold_rows)

        commission = float(trades.get("commission", pd.Series(dtype=float)).sum())
        tax = float(trades.get("tax", pd.Series(dtype=float)).sum())
        transfer = float(trades.get("transfer", pd.Series(dtype=float)).sum())
        slippage = float(trades.get("slippage", pd.Series(dtype=float)).sum())
        total = commission + tax + transfer + slippage
        cost_rows = []
        for label, value in (("佣金", commission), ("印花税", tax),
                             ("过户费", transfer), ("滑点", slippage),
                             ("合计", total)):
            share = (value / total * 100) if total else 0.0
            cost_rows.append([label, "%.2f" % value, "%.2f" % share])
        self._fill(self.cost_table, ["项目", "金额(元)", "占比(%)"], cost_rows)
        self.cost_label.setText(
            "总成本 %.2f 元：佣金 %.2f、印花税 %.2f、过户费 %.2f、滑点 %.2f。"
            "滑点按开盘价与成交价差额估算，仅研究用途。"
            % (total, commission, tax, transfer, slippage))

    def _draw_nav(self, nav):
        figure = self.nav_figure
        figure.clear()
        if nav is None or nav.empty:
            self.nav_canvas.draw()
            return
        axis = figure.add_subplot(111)
        dates = list(range(len(nav)))
        axis.plot(dates, nav["total_asset"].astype(float), color="#1f4e79",
                  linewidth=1.6, label="策略总资产")
        axis.plot(dates, nav["benchmark"].astype(float), color="#c0504d",
                  linewidth=1.2, label="等权基准")
        step = max(len(nav) // 8, 1)
        ticks = list(range(0, len(nav), step))
        axis.set_xticks(ticks)
        axis.set_xticklabels([nav["date"].iloc[i] for i in ticks], fontsize=8)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, loc="upper left")
        figure.tight_layout()
        self.nav_canvas.draw()

    def start_replay(self):
        start = self.start_edit.date().toString("yyyy-MM-dd")
        end = self.end_edit.date().toString("yyyy-MM-dd")
        if start >= end:
            QMessageBox.warning(self, "区间错误", "起始月末不能晚于终止月末")
            return
        cash = float(self.cash_spin.value())

        def task(progress):
            return run_rolling_sim(
                start, end, cash, save=True,
                progress_cb=lambda current, total, text: progress.emit({
                    "current": current, "total": total, "text": text}))

        self.window.start_task(task, self._replay_done,
                               "多期滚动回放中：%s ~ %s" % (start, end))

    def _replay_done(self, result):
        self._load_existing()
        nav = result["nav"]
        self.window.statusBar().showMessage(
            "多期滚动回放完成：%s ~ %s，%d 期，期末总资产 %.2f 元"
            % (result["start"], result["end"], result["periods_count"],
               float(nav["total_asset"].iloc[-1])), 20000)

    def export_report(self):
        def task(progress):
            progress.emit({"current": 0, "total": 1,
                           "text": "读取本地滚动结果并生成报告"})
            path = build_rolling_report()
            progress.emit({"current": 1, "total": 1, "text": "报告已生成"})
            return path

        self.window.start_task(task, self._report_done, "生成多期滚动报告…")

    def _report_done(self, path):
        self.status_label.setText(
            "纸面仿真·非真实交易·报告已生成：%s" % path)
        self.window.statusBar().showMessage("多期滚动报告：%s" % path, 20000)


SHADOW_OUTPUT_DIR = Path(__file__).resolve().parent / "shadow_output"
MONITOR_HISTORY = SHADOW_OUTPUT_DIR / "monitor_history.csv"
SHADOW_LOG_DIR = Path(__file__).resolve().parent / "logs"
SHADOW_BAT = Path(__file__).resolve().parent / "run_shadow_daily.bat"
NPU_OUTPUT_DIR = Path(__file__).resolve().parent / "npu_output"
TASK_NAME = "ShadowInfer_Daily"


class ShadowPage(QWidget):
    """影子推理看板：只读 monitor_history.csv，展示指标与告警。

    锁版冻结：不提供重训、参数调优入口；仅刷新、查看日志、导出历史。
    """

    ROWS = [
        ("600519 上行捕获率", "窗口内上涨日策略收益 / 基准收益"),
        ("600519 连续<40%天数", "达到 5 天进入人工 floor 讨论"),
        ("000001 趋势日仓位", "价格>MA20 且动量>0 时的平均仓位"),
        ("300750 超额收益", "策略 NAV − 买入持有"),
        ("FP16 最大仓位误差", "NPU / IR 对 CPU FP16 参考"),
        ("FP16 最大 NAV 相对误差", "一致性阈值 1e-3"),
    ]

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.latest_as_of = "-"
        layout = QVBoxLayout(self)

        status_box = QGroupBox("锁版状态")
        status_layout = QVBoxLayout(status_box)
        self.lock_label = QLabel(
            "锁版冻结 | floor 0.20/0.20/0.25 | FP16 IR + NPU 盘后推理 | 每日16:30影子运行"
        )
        self.lock_label.setStyleSheet(
            "color:#0a6b3d; font-weight:bold; padding:4px; background:#eef8f1;"
            "border:1px solid #b9e2c8;"
        )
        status_layout.addWidget(self.lock_label)
        self.meta_label = QLabel("尚未加载 monitor_history.csv")
        self.meta_label.setStyleSheet("color:#555;")
        status_layout.addWidget(self.meta_label)
        layout.addWidget(status_box)

        board_box = QGroupBox("影子推理看板")
        board_layout = QVBoxLayout(board_box)
        self.table = QTableWidget(len(self.ROWS), 3)
        self.table.setHorizontalHeaderLabels(["指标", "数值", "说明"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for row, (name, note) in enumerate(self.ROWS):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem("-"))
            self.table.setItem(row, 2, QTableWidgetItem(note))
        board_layout.addWidget(self.table)
        layout.addWidget(board_box)

        chart_box = QGroupBox("FP16 vs NPU 误差（逐日）")
        chart_layout = QVBoxLayout(chart_box)
        self.figure = Figure(figsize=(8, 2.6))
        self.chart = FigureCanvasQTAgg(self.figure)
        self.chart.setMinimumHeight(230)
        self.chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chart_label = QLabel("尚未加载 NPU 对比数据")
        self.chart_label.setStyleSheet("color:#666;")
        chart_layout.addWidget(self.chart)
        chart_layout.addWidget(self.chart_label)
        layout.addWidget(chart_box)

        alert_box = QGroupBox("告警")
        alert_layout = QVBoxLayout(alert_box)
        self.alert_label = QLabel("尚未加载监控数据")
        self.alert_label.setWordWrap(True)
        self.alert_label.setTextFormat(Qt.RichText)
        self.alert_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.alert_label.setMinimumHeight(70)
        self.alert_label.setStyleSheet(
            "background:#f7f7f7; border:1px solid #ddd; padding:8px;"
        )
        alert_layout.addWidget(self.alert_label)
        layout.addWidget(alert_box)

        schedule_box = QGroupBox("调度")
        schedule_layout = QHBoxLayout(schedule_box)
        self.schedule_label = QLabel("正在检测任务计划…")
        self.schedule_label.setTextFormat(Qt.RichText)
        self.schedule_label.setOpenExternalLinks(False)
        self.schedule_label.linkActivated.connect(self.run_scheduled_now)
        schedule_layout.addWidget(self.schedule_label)
        schedule_layout.addStretch(1)
        layout.addWidget(schedule_box)

        action_layout = QHBoxLayout()
        self.refresh_button = QPushButton("手动刷新")
        self.refresh_button.clicked.connect(self.refresh)
        self.logs_button = QPushButton("查看日志")
        self.logs_button.clicked.connect(self.open_logs)
        self.export_button = QPushButton("导出历史")
        self.export_button.clicked.connect(self.export_history)
        self.retrain_button = QPushButton("重训模型（锁版禁用）")
        self.retrain_button.setEnabled(False)
        self.retrain_button.setToolTip("锁版冻结：不重训、不替换权重")
        self.tune_button = QPushButton("参数调优（锁版禁用）")
        self.tune_button.setEnabled(False)
        self.tune_button.setToolTip("锁版冻结：不调 floor、不改阈值")
        self.floor_button = QPushButton("调整 floor（锁版禁用）")
        self.floor_button.setEnabled(False)
        self.floor_button.setToolTip("锁版冻结：floor 维持 0.20/0.20/0.25，仅在 streak≥5 时人工评估")
        action_layout.addWidget(self.refresh_button)
        action_layout.addWidget(self.logs_button)
        action_layout.addWidget(self.export_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self.retrain_button)
        action_layout.addWidget(self.tune_button)
        action_layout.addWidget(self.floor_button)
        layout.addLayout(action_layout)
        layout.addStretch(1)

    # ---------- 数据加载 ----------
    def refresh(self):
        self.refresh_chart()
        try:
            if not MONITOR_HISTORY.exists():
                self.meta_label.setText("未找到 %s" % MONITOR_HISTORY)
                self.set_alert(["尚未运行影子推理，无监控记录。"], level="info")
                self.window.set_alert_badge(False, 0, loaded=False)
                self.update_schedule()
                return
            frame = pd.read_csv(
                MONITOR_HISTORY, dtype={"code": str, "as_of": str}, encoding="utf-8-sig"
            )
            if frame.empty:
                self.set_alert(["monitor_history.csv 为空。"], level="info")
                return
            frame["as_of"] = frame["as_of"].astype(str)
            self.latest_as_of = frame["as_of"].max()
            latest = frame[frame["as_of"] == self.latest_as_of]
            fp16 = latest[latest["precision"].astype(str).str.upper() == "FP16"]
            if not fp16.empty:
                latest = fp16
            rows = {str(code): group.iloc[-1] for code, group in latest.groupby("code")}
            self.fill_table(rows)
            self.meta_label.setText(
                "as_of=%s | 运行时间 %s | 窗口 %s 天 | 精度 %s | 记录 %d 条" % (
                    self.latest_as_of,
                    latest["run_at"].iloc[-1],
                    int(latest["window_days"].iloc[-1]),
                    latest["precision"].astype(str).iloc[-1],
                    len(frame),
                )
            )
            messages = self.build_alerts(rows, latest)
            self.set_alert(messages, level="auto")
            if "risk_level" in latest.columns:
                risk_levels = latest["risk_level"].fillna("").astype(str)
            else:
                risk_levels = pd.Series("", index=latest.index)
            self.window.set_alert_badge(
                any("<span" in item for item in messages), self.streak_of(rows),
                risk_blacklist=int((risk_levels == "BLACKLIST").sum()),
                risk_watch=int((risk_levels == "WATCH").sum()),
            )
            self.update_schedule()
        except Exception as exc:
            self.set_alert(["看板读取失败：%s" % exc], level="error")
            self.window.set_alert_badge(False, 0, loaded=False)

    def fill_table(self, rows):
        def value(code, column):
            row = rows.get(code)
            return None if row is None else row.get(column)

        capture = value("600519", "uptrend_capture")
        streak = value("600519", "capture_streak_600519")
        position = value("000001", "avg_position_uptrend")
        if position is None or pd.isna(position):
            position = value("000001", "avg_position")
        excess = value("300750", "excess")
        max_pos = max(
            [abs(float(v)) for v in [r.get("max_position_diff") for r in rows.values()]
             if v is not None and not pd.isna(v)] or [float("nan")]
        )
        max_nav = max(
            [abs(float(v)) for v in [r.get("max_nav_rel_diff") for r in rows.values()]
             if v is not None and not pd.isna(v)] or [float("nan")]
        )
        values = [
            self.pct(capture),
            "-" if streak is None or pd.isna(streak) else "%d" % int(float(streak)),
            self.pct(position),
            self.pct(excess, signed=True),
            self.sci(max_pos),
            self.sci(max_nav),
        ]
        for row, text in enumerate(values):
            self.table.setItem(row, 1, QTableWidgetItem(text))
        self.fit_table()

    def fit_table(self):
        """按内容高度撑开表格，避免最后几行被裁切。"""
        self.table.resizeRowsToContents()
        header = max(self.table.horizontalHeader().sizeHint().height(), 24)
        rows = sum(self.table.rowHeight(row) for row in range(self.table.rowCount()))
        self.table.setFixedHeight(header + rows + 4)

    def build_alerts(self, rows, latest):
        alerts = []
        row_600519 = rows.get("600519")
        capture = None if row_600519 is None else row_600519.get("uptrend_capture")
        streak = None if row_600519 is None else row_600519.get("capture_streak_600519")
        streak = 0 if streak is None or pd.isna(streak) else int(float(streak))
        if capture is not None and not pd.isna(capture):
            if capture < 0.40 and streak >= 5:
                alerts.append(
                    '<span style="color:#b00020; font-weight:bold;">'
                    "600519 捕获率 %.2f%% 且连续 %d 天 &lt;40%%：进入人工 floor 讨论</span>"
                    "（仅提示，不自动调参）" % (100 * capture, streak)
                )
            elif capture < 0.40:
                alerts.append(
                    "观察：600519 捕获率 %.2f%%（streak=%d，未达 5 天，不触发 floor 讨论）"
                    % (100 * capture, streak)
                )
        max_pos = max(
            [abs(float(v)) for v in latest["max_position_diff"].tolist()
             if not pd.isna(v)] or [0.0]
        )
        max_nav = max(
            [abs(float(v)) for v in latest["max_nav_rel_diff"].tolist()
             if not pd.isna(v)] or [0.0]
        )
        if max_pos >= 1e-3 or max_nav >= 1e-3:
            alerts.append(
                '<span style="color:#b00020; font-weight:bold;">'
                "一致性超差：FP16 仓位误差 %.3e / NAV 相对误差 %.3e（阈值 1e-3）</span>"
                % (max_pos, max_nav)
            )
        # 事件风控分级告警：拦截红色，只有预警时看板保持黄色
        if "risk_level" in latest.columns:
            risk_levels = latest["risk_level"].fillna("").astype(str)
            blocked_codes = latest.loc[risk_levels == "BLACKLIST", "code"].astype(str).tolist()
            watch_codes = latest.loc[risk_levels == "WATCH", "code"].astype(str).tolist()
            if blocked_codes:
                alerts.append(
                    '<span style="color:#b00020; font-weight:bold;">'
                    "事件风控拦截 %d 只：%s</span>"
                    % (len(blocked_codes), "、".join(blocked_codes))
                )
            if watch_codes:
                alerts.append(
                    "事件风控预警 %d 只：%s"
                    % (len(watch_codes), "、".join(watch_codes))
                )
        failures = self.failed_logs()
        if failures:
            alerts.append(
                '<span style="color:#b00020; font-weight:bold;">FAILED exit= 记录：</span>'
                + "、".join(failures)
            )
        return alerts

    @staticmethod
    def streak_of(rows):
        row = rows.get("600519")
        if row is None:
            return 0
        value = row.get("capture_streak_600519")
        if value is None or pd.isna(value):
            return 0
        return int(float(value))

    def failed_logs(self, limit=30):
        if not SHADOW_LOG_DIR.exists():
            return []
        failures = []
        for path in sorted(SHADOW_LOG_DIR.glob("shadow_*.log"))[-limit:]:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "FAILED exit=" in text:
                failures.append(path.name)
        return failures

    def set_alert(self, messages, level="auto"):
        if not messages:
            self.alert_label.setText("告警：无")
            self.alert_label.setStyleSheet(
                "background:#f0f8f2; border:1px solid #bfe3cc; padding:8px; color:#0a6b3d;"
            )
            return
        has_error = level == "error" or any("<span" in item for item in messages)
        if level == "error":
            has_error = True
        color = "#b00020" if has_error else "#7a5b00"
        background = "#fff5f5" if has_error else "#fffbe8"
        border = "#f0b8b8" if has_error else "#f0d890"
        self.alert_label.setStyleSheet(
            "background:%s; border:1px solid %s; padding:8px; color:%s;" % (
                background, border, color
            )
        )
        self.alert_label.setText("<br>".join(messages))

    # ---------- 图表 ----------
    def refresh_chart(self):
        """绘制 NPU 与 CPU FP16 参考的逐日仓位/NAV 误差。"""
        try:
            run = self.latest_npu_run()
            path = (run / "npu_daily_all.csv") if run else None
            if path is None or not path.exists():
                self.figure.clear()
                self.chart.hide()
                self.chart_label.setText(
                    "尚未运行 npu_batch_infer.py，暂无 FP16 / NPU 对比数据"
                )
                return
            frame = pd.read_csv(path, dtype={"code": str})
            if frame.empty:
                self.figure.clear()
                self.chart.hide()
                self.chart_label.setText("npu_daily_all.csv 为空")
                return
            self.chart.show()
            self.figure.clear()
            position_ax = self.figure.add_subplot(1, 2, 1)
            nav_ax = self.figure.add_subplot(1, 2, 2)
            colors = {"600519": "#c0392b", "000001": "#2c6fbb", "300750": "#e08b00"}
            for code, group in frame.groupby("code"):
                group = group.sort_values("date").reset_index(drop=True)
                x_axis = list(range(len(group)))
                color = colors.get(str(code))
                for axis, column in (
                    (position_ax, "position_diff"), (nav_ax, "nav_rel_diff")
                ):
                    # 对数轴无法画 0，精确相等的点留空（300750 两项误差均为 0）
                    series = group[column].astype(float).replace(0.0, float("nan"))
                    axis.plot(
                        x_axis, series, label=code, color=color, linewidth=1.0,
                        marker="o", markersize=2.2, alpha=0.9,
                        drawstyle="steps-post" if column == "position_diff" else "default",
                    )
            dates = frame[frame["code"] == str(frame["code"].iloc[0])]["date"].astype(str).tolist()
            count = len(dates)
            ticks = sorted(set([0, count // 4, count // 2, 3 * count // 4, count - 1]))
            for axis, column, title in (
                (position_ax, "position_diff", "逐日仓位误差 |NPU−FP16|"),
                (nav_ax, "nav_rel_diff", "逐日 NAV 相对误差"),
            ):
                values = frame[column].astype(float)
                peak = float(values.max()) if len(values) else 0.0
                axis.set_yscale("log")
                axis.set_ylim(1e-8, max(peak, 1e-3) * 1.8)
                axis.axhline(
                    1e-3, color="#b00020", linestyle="--", linewidth=0.9, label="阈值 1e-3"
                )
                axis.set_title(
                    "%s（max %.2e / mean %.2e）" % (title, peak, float(values.mean())),
                    fontsize=9,
                )
                axis.set_xticks(ticks)
                axis.set_xticklabels([dates[i] for i in ticks], rotation=25, fontsize=7)
                axis.grid(alpha=0.25, which="both")
                axis.legend(fontsize=6, loc="lower right")
            self.figure.suptitle(
                "FP16(CPU) vs NPU  %s（CPU 参考已对 FP32 校验 <1e-6）" % run.name,
                fontsize=8,
            )
            self.figure.tight_layout()
            self.chart.draw()
            self.chart_label.setText(
                "数据源：%s/%s；红线为一致性阈值 1e-3（对数纵轴，300750 误差为 0 不显示）；"
                "CPU 参考为 FP16 IR（对 FP32 残差 <1e-6）；仅展示不调参"
                % (run.name, path.name)
            )
        except Exception as exc:
            self.figure.clear()
            self.chart.hide()
            self.chart_label.setText("图表加载失败：%s" % exc)

    def latest_npu_run(self):
        runs = sorted(NPU_OUTPUT_DIR.glob("run_*"))
        return runs[-1] if runs else None

    # ---------- 调度 ----------
    def update_schedule(self):
        enabled = self.task_enabled()
        if enabled:
            self.schedule_label.setText(
                '<a href="run">%s 已启用（每日 16:30）· 点击手动运行 run_shadow_daily.bat</a>'
                % TASK_NAME
            )
            self.schedule_label.setStyleSheet("color:#0a6b3d;")
        else:
            self.schedule_label.setText(
                "%s 未检测到；请按 shadow_daily_task.md 配置（每日 16:30）" % TASK_NAME
            )
            self.schedule_label.setStyleSheet("color:#a33;")

    def task_enabled(self):
        try:
            result = subprocess.run(
                ["schtasks", "/Query", "/TN", TASK_NAME],
                capture_output=True, text=True, encoding="gbk", errors="ignore",
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return result.returncode == 0
        except Exception:
            return False

    def run_scheduled_now(self, _link=None):
        if not SHADOW_BAT.exists():
            QMessageBox.warning(self, "找不到脚本", "未找到 run_shadow_daily.bat")
            return
        answer = QMessageBox.question(
            self, "手动运行影子推理",
            "将立即在后台执行 run_shadow_daily.bat（与每日 16:30 任务相同）。\n"
            "完成后点击【手动刷新】查看 monitor_history.csv。是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        ok = QProcess.startDetached(
            "cmd.exe", ["/c", str(SHADOW_BAT)], str(SHADOW_BAT.parent)
        )
        if ok:
            self.window.statusBar().showMessage(
                "已启动 run_shadow_daily.bat（后台运行）；完成后点击【手动刷新】", 10000
            )
        else:
            QMessageBox.warning(self, "启动失败", "无法启动 run_shadow_daily.bat")

    # ---------- 操作 ----------
    def open_logs(self):
        if not SHADOW_LOG_DIR.exists():
            SHADOW_LOG_DIR.mkdir(parents=True, exist_ok=True)
        logs = sorted(SHADOW_LOG_DIR.glob("shadow_*.log"))
        target = logs[-1] if logs else SHADOW_LOG_DIR
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def export_history(self):
        if not MONITOR_HISTORY.exists():
            QMessageBox.information(self, "无历史记录", "monitor_history.csv 尚不存在")
            return
        default = SHADOW_OUTPUT_DIR / (
            "monitor_history_%s.csv" % datetime.now().strftime("%Y%m%d")
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "导出监控历史", str(default), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            Path(path).write_bytes(MONITOR_HISTORY.read_bytes())
            self.window.statusBar().showMessage("已导出监控历史：" + path, 10000)
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    # ---------- 工具 ----------
    @staticmethod
    def pct(value, signed=False):
        if value is None or pd.isna(value):
            return "-"
        return ("%+.2f%%" if signed else "%.2f%%") % (100 * float(value))

    @staticmethod
    def sci(value):
        if value is None or pd.isna(value):
            return "-"
        return "%.3e" % float(value)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        db.init_db()
        self.worker = None
        self.setWindowTitle("A 股本地量化研究工具 V1.0 - 仅研究用途")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 820)

        central = QWidget()
        root_layout = QVBoxLayout(central)
        disclaimer = QLabel(DISCLAIMER + "。本工具不连接券商、不自动下单、不上传云端。")
        disclaimer.setStyleSheet("color: #a33; font-weight: bold; padding: 4px;")
        root_layout.addWidget(disclaimer)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.data_page = DataPage(self)
        self.formula_page = FormulaPage(self)
        self.kline_page = KlinePage(self)
        self.backtest_page = BacktestPage(self)
        self.smart_beta_page = SmartBetaPage(self)
        self.position_page = PositionPage(self)
        self.simulation_page = SimulationPage(self)
        self.rolling_page = RollingPage(self)
        self.shadow_page = ShadowPage(self)
        self.tabs.addTab(self.data_page, "数据管理")
        self.tabs.addTab(self.formula_page, "公式选股")
        self.tabs.addTab(self.kline_page, "K 线看图")
        self.tabs.addTab(self.backtest_page, "回测")
        self.tabs.addTab(self.smart_beta_page, "Smart Beta")
        self.tabs.addTab(self.position_page, "持仓盯盘")
        self.tabs.addTab(self.simulation_page, "Simulation")
        self.tabs.addTab(self.rolling_page, "多期滚动")
        self.shadow_scroll = QScrollArea()
        self.shadow_scroll.setWidgetResizable(True)
        self.shadow_scroll.setWidget(self.shadow_page)
        self.tabs.addTab(self.shadow_scroll, "影子推理")
        self.task_buttons = [
            self.data_page.update_codes_button,
            self.data_page.update_all_button,
            self.formula_page.run_button,
            self.backtest_page.run_button,
            self.smart_beta_page.exposure_button,
            self.smart_beta_page.run_button,
            self.smart_beta_page.sensitivity_button,
            self.smart_beta_page.summary_report_button,
            self.position_page.refresh_button,
            self.position_page.hammer_button,
            self.rolling_page.replay_button,
            self.rolling_page.report_button,
        ]
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.hide()
        self.lock_status = QLabel(
            "锁版冻结 | floor 0.20/0.20/0.25 | FP16 IR + NPU 盘后推理 | 每日16:30影子运行"
        )
        self.lock_status.setStyleSheet("color:#0a6b3d; padding:0 8px;")
        self.lock_status.setToolTip("锁版参数冻结，不重训、不调参；详见【影子推理】页")
        self.alert_status = QLabel("告警状态：加载中")
        self.alert_status.setStyleSheet("color:#7a5b00; padding:0 8px;")
        self.alert_status.setToolTip("来自 monitor_history.csv 与 logs/shadow_*.log，仅提示不调参")
        self.statusBar().addPermanentWidget(self.alert_status)
        self.statusBar().addPermanentWidget(self.lock_status)
        self.logs_link = QLabel('<a href="logs">查看日志</a>')
        self.logs_link.setTextFormat(Qt.RichText)
        self.logs_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.logs_link.setOpenExternalLinks(False)
        self.logs_link.setToolTip("打开本地 logs 目录")
        self.logs_link.linkActivated.connect(lambda _link: self.open_logs())
        self.statusBar().addPermanentWidget(self.logs_link)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().showMessage("就绪")
        self.data_page.refresh_status()
        self.shadow_page.refresh()

    def set_alert_badge(self, has_alert, streak=0, loaded=True,
                        risk_blacklist=0, risk_watch=0):
        """状态栏告警徽标：风控拦截优先红色计数，仅预警时黄色计数。"""
        if not loaded:
            text, color, weight = "告警状态：未加载监控记录", "#7a5b00", ""
        elif risk_blacklist > 0:
            text = "风控拦截%d只" % int(risk_blacklist)
            color, weight = "#b00020", "font-weight:bold;"
        elif has_alert:
            text, color, weight = "当前有告警（详见【影子推理】）", "#b00020", "font-weight:bold;"
        elif risk_watch > 0:
            text = "预警%d只" % int(risk_watch)
            color, weight = "#b8860b", "font-weight:bold;"
        else:
            text = "当前无告警（streak=%d）" % int(streak)
            color, weight = "#0a6b3d", ""
        self.alert_status.setText(text)
        self.alert_status.setStyleSheet("color:%s; padding:0 8px; %s" % (color, weight))

    def start_task(self, function, result_handler, busy_text):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "任务进行中", "请等待当前任务结束后再启动新任务")
            return
        for button in self.task_buttons:
            button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.statusBar().showMessage(busy_text)
        self.worker = WorkerThread(function, self)
        self.worker.progress.connect(self.task_progress)
        self.worker.result_ready.connect(result_handler)
        self.worker.failed.connect(self.task_failed)
        self.worker.finished.connect(self.task_finished)
        self.worker.start()

    def task_progress(self, data):
        total = int(data.get("total", 0))
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(int(data.get("current", 0)), total))
        else:
            self.progress_bar.setRange(0, 0)
        if data.get("text"):
            self.statusBar().showMessage(data["text"])

    def task_failed(self, details):
        last_line = next((line for line in reversed(details.splitlines()) if line.strip()), "未知错误")
        self.show_error("后台任务执行失败", details, last_line)
        self.statusBar().showMessage("任务失败：" + last_line, 10000)

    def task_finished(self):
        for button in self.task_buttons:
            button.setEnabled(True)
        self.progress_bar.hide()
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def show_error(self, title, details, short_message=None):
        box = QMessageBox(QMessageBox.Critical, title, short_message or "执行失败", parent=self)
        box.setDetailedText(details)
        box.exec_()

    def open_logs(self):
        """只打开本地日志目录，不读取或修改日志内容。"""
        db.ensure_dirs()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(db.LOG_DIR)))

    def show_kline(self, code, fq="qfq", period="daily", anchor_date=None):
        self.tabs.setCurrentWidget(self.kline_page)
        self.kline_page.open_stock(code, fq, period, anchor_date=anchor_date)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "任务进行中", "为避免破坏本地数据，请等待当前任务结束后关闭程序")
            event.ignore()
        else:
            event.accept()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication([])
    app.setApplicationName("A 股本地量化研究工具")
    # 全局窗口/任务栏图标：优先使用 assets/app_icon.ico，缺失时保持系统默认。
    icon_path = Path(__file__).resolve().with_name("assets") / "app_icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
