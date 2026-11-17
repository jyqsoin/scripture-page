# -*- coding: utf-8 -*-
"""
A 股本地量化研究工具：独立本地滚动日志。

日志文件：
- logs/data_update.log：行情更新成功、无变化及失败；
- logs/formula_error.log：公式语法、未来函数和单股执行错误；
- logs/backtest.log：回测启动、完成和异常；
- logs/alert.log：研究预警及规则拦截提示。

仅使用标准 logging + RotatingFileHandler，不输出到网络、不启动服务。
模块导入时不创建文件；首次写入时才创建对应日志。单文件默认 5 MB，保留 5 份。
仅研究用途，不构成投资建议。
"""

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db

LOG_FILES = {
    "data_update": "data_update.log",
    "formula_error": "formula_error.log",
    "backtest": "backtest.log",
    "alert": "alert.log",
}
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5
_FORMAT = "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
_configured = {}
_lock = threading.Lock()


def _validate_category(category):
    if category not in LOG_FILES:
        raise ValueError("未知日志类别：%s；可选：%s" % (category, ", ".join(LOG_FILES)))


def _one_line(value):
    """字段保持单行，避免股票名称或公式中的换行伪造日志记录。"""
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


def _configure(category, log_dir, level, max_bytes, backup_count):
    _validate_category(category)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = (log_dir / LOG_FILES[category]).resolve()
    logger = logging.getLogger("aquant." + category)
    logger.setLevel(level)
    logger.propagate = False

    with _lock:
        current_handler = None
        for handler in list(logger.handlers):
            if not getattr(handler, "_aquant_handler", False):
                continue
            if Path(handler.baseFilename).resolve() == path:
                current_handler = handler
            else:
                handler.close()
                logger.removeHandler(handler)
        if current_handler is None:
            current_handler = RotatingFileHandler(
                path,
                maxBytes=int(max_bytes),
                backupCount=int(backup_count),
                encoding="utf-8",
                delay=True,
            )
            current_handler._aquant_handler = True
            logger.addHandler(current_handler)
        current_handler.setLevel(level)
        current_handler.maxBytes = int(max_bytes)
        current_handler.backupCount = int(backup_count)
        current_handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        _configured[category] = path
    return path


def setup_logging(log_dir=None, level=logging.INFO, max_bytes=DEFAULT_MAX_BYTES, backup_count=DEFAULT_BACKUP_COUNT):
    """初始化四类日志；重复调用不会添加重复 handler。"""
    log_dir = Path(log_dir) if log_dir is not None else Path(db.LOG_DIR)
    if int(max_bytes) <= 0 or int(backup_count) < 0:
        raise ValueError("max_bytes 必须大于 0，backup_count 不能为负数")
    return {
        category: str(_configure(category, log_dir, level, max_bytes, backup_count))
        for category in LOG_FILES
    }


def get_logger(category, log_dir=None):
    """获取独立 logger；未显式初始化时使用项目 logs 目录。"""
    _validate_category(category)
    if category not in _configured or log_dir is not None:
        _configure(
            category,
            log_dir if log_dir is not None else db.LOG_DIR,
            logging.INFO,
            DEFAULT_MAX_BYTES,
            DEFAULT_BACKUP_COUNT,
        )
    return logging.getLogger("aquant." + category)


def log_event(category, message, level=logging.INFO, exc_info=False, **fields):
    """写一条单行字段日志；exc_info=True 时可附加当前异常堆栈。"""
    if isinstance(level, str):
        levels = {
            "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
            "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL,
        }
        normalized = level.upper()
        if normalized not in levels:
            raise ValueError("未知日志级别：%s" % level)
        level = levels[normalized]
    field_text = " ".join("%s=%s" % (_one_line(key), _one_line(value)) for key, value in sorted(fields.items()) if value is not None)
    text = _one_line(message)
    if field_text:
        text += " | " + field_text
    get_logger(category).log(int(level), text, exc_info=bool(exc_info))


def log_data_update(code, fq, status, message="", error=None, exc_info=False):
    level = logging.ERROR if error is not None else logging.INFO
    log_event(
        "data_update", message or "行情更新", level=level, exc_info=exc_info,
        code=code, fq=fq, status=status, error=error,
    )


def log_formula_error(formula_name, code, error, formula="", exc_info=False):
    log_event(
        "formula_error", "公式执行错误", level=logging.ERROR, exc_info=exc_info,
        formula_name=formula_name, code=code, formula=formula, error=error,
    )


def log_backtest(name, status, message="", record_id=None, error=None, exc_info=False):
    level = logging.ERROR if error is not None else logging.INFO
    log_event(
        "backtest", message or "回测任务", level=level, exc_info=exc_info,
        name=name, status=status, record_id=record_id, error=error,
    )


def log_alert(code, message, level="INFO", **fields):
    log_event("alert", message, level=level, code=code, **fields)


def shutdown_logging():
    """关闭本模块创建的文件句柄，便于桌面程序退出、备份和临时目录清理。"""
    with _lock:
        for category in LOG_FILES:
            logger = logging.getLogger("aquant." + category)
            for handler in list(logger.handlers):
                if getattr(handler, "_aquant_handler", False):
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)
        _configured.clear()


if __name__ == "__main__":
    # 临时目录自测：不写入项目正式 logs。
    import tempfile

    with tempfile.TemporaryDirectory(prefix="aquant_logger_test_") as temp_dir:
        paths = setup_logging(temp_dir, max_bytes=2048, backup_count=1)
        setup_logging(temp_dir, max_bytes=2048, backup_count=1)  # 重复初始化不应重复写一条记录
        log_data_update("000001", "qfq", "ok", "自测更新完成")
        log_formula_error("测试公式", "000001", "未来函数", formula="XG:REF(C,-1);")
        log_backtest("自测回测", "ok", record_id=1)
        log_alert("000001", "自测研究预警", reason="测试")
        for category in LOG_FILES:
            for handler in get_logger(category).handlers:
                handler.flush()
        for category, path in paths.items():
            content = Path(path).read_text(encoding="utf-8")
            assert "自测" in content or "公式执行错误" in content
            assert content.count("自测更新完成") <= 1
        shutdown_logging()
        print("logger.py 自测通过：四类独立日志、单行字段和重复初始化检查正常")
        print("仅研究用途，不构成投资建议")
