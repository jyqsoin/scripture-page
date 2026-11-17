# -*- coding: utf-8 -*-
"""
英大证券自选股写入（纯本地，仅研究用途）。

落点：
- 读取英大证券客户端自选股文件 zxg.blk（通常位于安装目录 T0002\\blocknew 下）；
- 追加模式：保留原有自选股，仅把本次选股结果中不存在的代码追加到文件末尾；
- 写入前自动备份原文件为 zxg.blk.bak_YYYYmmdd_HHMMSS，原文件内容按原字节保留。

文件格式（通达信系客户端通用）：
- 每行 1 个 7 位代码：市场前缀 + 6 位股票代码；
- 沪市（6/9 开头）前缀为 1，深市（0/3 开头）前缀为 0；
- ASCII 编码，换行符为 CRLF。

安全约束：
- 必须在英大证券客户端完全退出后写入，避免客户端退出时覆盖写入结果；
- 不联网、不接券商、不下单，只读写本地文件。

仅研究用途，不构成投资建议。
"""

import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db

DISCLAIMER = "仅研究用途，不构成投资建议"
CONFIG_PATH = db.DATA_DIR / "ydzq_watchlist.json"
NAME_KEYWORDS = ("英大", "yingda", "ydzq")
CLIENT_SUBDIRS = (
    Path("T0002") / "blocknew" / "zxg.blk",
    Path("blocknew") / "zxg.blk",
    Path("zxg.blk"),
)


def market_prefix(code):
    """返回英大证券自选股市场前缀：沪市 1，深市 0。"""
    return "1" if str(code)[:1] in ("6", "9") else "0"


def load_last_path():
    """读取上次使用的 zxg.blk 路径；未配置或失效时返回 None。"""
    if not CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        path = Path(str(data.get("path", "")).strip())
        return path if str(path) and path.exists() else None
    except (OSError, ValueError):
        return None


def save_last_path(path):
    """记住本次使用的 zxg.blk 路径，便于下次直接打开。"""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"path": str(path)}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _registry_install_dirs():
    """从注册表卸载信息中查找英大证券安装目录。"""
    try:
        import winreg
    except ImportError:
        return []
    subkeys = [
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    results = []
    for hive, subkey in subkeys:
        try:
            with winreg.OpenKey(hive, subkey) as root:
                total = winreg.QueryInfoKey(root)[0]
                for index in range(total):
                    try:
                        name = winreg.EnumKey(root, index)
                        with winreg.OpenKey(root, name) as child:
                            display = _registry_value(child, "DisplayName")
                            location = _registry_value(child, "InstallLocation")
                    except OSError:
                        continue
                    if not display or not location:
                        continue
                    if any(key in display.lower() for key in NAME_KEYWORDS):
                        results.append(Path(location))
        except OSError:
            continue
    return results


def _registry_value(key, name):
    """读取注册表字符串值，缺失时返回 None。"""
    try:
        import winreg
        return str(winreg.QueryValueEx(key, name)[0]).strip()
    except (ImportError, OSError):
        return None


def _zxg_in(directory):
    """在指定目录下查找 zxg.blk（先查常见相对路径，再递归查找）。"""
    base = Path(directory)
    if not base.is_dir():
        return []
    found = [base / sub for sub in CLIENT_SUBDIRS if (base / sub).is_file()]
    try:
        found.extend(path for path in base.rglob("zxg.blk") if path.is_file())
    except OSError:
        pass
    return found


def _keyword_dirs(root, depth=2):
    """按名称关键词查找目录，限制搜索深度，避免全盘扫描。"""
    root = Path(root)
    if not root.is_dir():
        return []
    results = []
    frontier = [(root, 0)]
    while frontier:
        current, level = frontier.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            name = entry.name.lower()
            if any(key in name for key in NAME_KEYWORDS):
                results.append(Path(entry.path))
            elif level + 1 < depth:
                frontier.append((Path(entry.path), level + 1))
    return results


def _search_roots():
    """常见安装位置：各盘符顶层、Program Files 与 AppData（限制深度）。"""
    roots = []
    for letter in "CDEFGH":
        drive = Path(letter + ":\\")
        if drive.exists():
            roots.append((drive, 1))
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 os.environ.get("LOCALAPPDATA"), os.environ.get("APPDATA")):
        if base and Path(base).is_dir():
            roots.append((Path(base), 2))
    return roots


def find_default_path():
    """自动定位英大证券 zxg.blk：记忆路径 → 注册表安装目录 → 常见目录。

    返回修改时间最新的 zxg.blk；找不到时返回 None（界面再回退到手动选择）。
    """
    remembered = load_last_path()
    if remembered:
        return remembered
    candidates = []
    for directory in _registry_install_dirs():
        candidates.extend(_zxg_in(directory))
    for root, depth in _search_roots():
        for directory in _keyword_dirs(root, depth):
            candidates.extend(_zxg_in(directory))
    unique = {}
    for path in candidates:
        try:
            unique[path.resolve()] = path
        except OSError:
            continue
    if not unique:
        return None
    return max(unique.values(), key=lambda item: item.stat().st_mtime)


def _normalize_codes(codes):
    """校验并去重 6 位代码，保持输入顺序。"""
    result = []
    for code in codes:
        text = str(code).strip().zfill(6)
        if re.fullmatch(r"\d{6}", text) and text not in result:
            result.append(text)
    return result


def _existing_codes(raw):
    """从原文件字节中提取已存在的 6 位代码（ASCII，忽略非法字节）。

    自选股每行为“市场前缀 + 6 位代码”（7 位），取末 6 位；
    也兼容只有 6 位代码的行。
    """
    text = raw.decode("ascii", errors="ignore")
    codes = set()
    for line in text.splitlines():
        digits = re.sub(r"\D", "", line)
        if len(digits) >= 7:
            codes.add(digits[-6:])
        elif len(digits) == 6:
            codes.add(digits)
    return codes


def _backup(path):
    """把原文件复制为带时间戳的备份，返回备份路径。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name("%s.bak_%s" % (path.name, stamp))
    counter = 1
    while backup.exists():
        backup = path.with_name("%s.bak_%s_%d" % (path.name, stamp, counter))
        counter += 1
    shutil.copy2(path, backup)
    return backup


def append_codes(path, codes):
    """追加自选股代码到 zxg.blk，保留原内容。

    返回 dict：path（目标文件）、backup（备份路径，新建文件时为 None）、
    added（新增数量）、skipped（已存在跳过数量）、codes（实际写入代码）。
    """
    target = Path(path)
    normalized = _normalize_codes(codes)
    if not normalized:
        raise ValueError("没有可写入的 6 位股票代码")

    raw = b""
    backup = None
    if target.exists():
        raw = target.read_bytes()
        if raw:
            backup = _backup(target)
    existing = _existing_codes(raw)
    new_codes = [code for code in normalized if code not in existing]
    skipped = len(normalized) - len(new_codes)

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = raw
    if new_codes:
        if payload and not payload.endswith(b"\n"):
            payload += b"\r\n"  # 原文件末尾缺少换行时补齐，避免与新代码粘连
        payload += "".join(
            "%s%s\r\n" % (market_prefix(code), code) for code in new_codes
        ).encode("ascii")
    target.write_bytes(payload)  # 原文件内容按原字节保留，只在末尾追加
    return {
        "path": target,
        "backup": backup,
        "added": len(new_codes),
        "skipped": skipped,
        "codes": new_codes,
    }


if __name__ == "__main__":
    # 临时目录自测：不触碰正式自选股文件。
    print(DISCLAIMER)
    with tempfile.TemporaryDirectory() as folder:
        sample = Path(folder) / "zxg.blk"
        original = b"1600519\r\n0000001\r\n"
        sample.write_bytes(original)
        result = append_codes(sample, ["300750", "600519", "000001", "300750"])
        assert result["added"] == 1, result
        assert result["skipped"] == 2, result
        assert result["backup"] and result["backup"].exists()
        assert result["backup"].read_bytes() == original
        assert sample.read_bytes() == original + b"0300750\r\n"
        print("追加自测通过：", result["added"], "新增，", result["skipped"], "跳过")

        # 原文件末尾无换行时补 CRLF，避免粘连。
        no_newline = Path(folder) / "no_newline.blk"
        no_newline.write_bytes(b"1600519")
        result = append_codes(no_newline, ["000001"])
        assert no_newline.read_bytes() == b"1600519\r\n0000001\r\n"
        print("无换行补全自测通过")

        # 文件不存在时新建。
        fresh = Path(folder) / "new.blk"
        result = append_codes(fresh, ["600519"])
        assert fresh.read_bytes() == b"1600519\r\n" and result["backup"] is None
        print("新建文件自测通过")
    print("ydzq_watchlist.py 自测通过")
