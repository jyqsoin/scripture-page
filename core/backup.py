# -*- coding: utf-8 -*-
"""
SQLite + parquet 本地备份、完整性校验与安全恢复。

- SQLite 使用在线 backup API 生成一致性快照，不直接复制正在使用的数据库；
- parquet 复制前后检查尺寸和修改时间，发现更新中的文件会中止备份；
- ZIP 内 manifest.json 记录路径、尺寸和 SHA-256，恢复前逐项校验并检查 SQLite integrity_check；
- 恢复必须 confirm=True，建议先关闭桌面程序；恢复前自动生成 pre_restore ZIP；
- 替换失败时尝试回滚，若回滚不完整会保留回滚目录并明确报错；
- 不联网、不压缩上传、不删除备份目录中的历史文件。

仅研究用途，不构成投资建议。
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime
from pathlib import Path, PurePosixPath

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db

FORMAT_VERSION = 1
DISCLAIMER = "仅研究用途，不构成投资建议"


class BackupError(RuntimeError):
    pass


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _safe_archive_path(path):
    """只允许 aquant.db 和 parquet/**/*.parquet，阻止绝对路径及目录穿越。"""
    if not isinstance(path, str) or not path or "\\" in path or ":" in path:
        raise BackupError("备份包含非法路径：%r" % path)
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise BackupError("备份包含不安全路径：%s" % path)
    if path == "aquant.db":
        return pure
    if path.startswith("parquet/") and len(pure.parts) >= 2 and pure.suffix == ".parquet":
        return pure
    raise BackupError("备份包含未允许的文件：%s" % path)


def _check_sqlite(path):
    path = Path(path)
    if not path.exists():
        raise BackupError("SQLite 文件不存在：%s" % path)
    try:
        with closing(sqlite3.connect(str(path))) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.Error as exc:
        raise BackupError("SQLite 完整性检查失败：%s" % exc) from exc
    if not result or any(str(row[0]).lower() != "ok" for row in result):
        raise BackupError("SQLite integrity_check 未通过：%s" % result)
    return True


def _snapshot_database(source, destination):
    source = Path(source)
    if not source.exists():
        raise BackupError("本地数据库不存在，无法备份：%s" % source)
    try:
        with closing(sqlite3.connect(str(source))) as source_conn:
            source_conn.execute("PRAGMA busy_timeout = 5000")
            with closing(sqlite3.connect(str(destination))) as target_conn:
                source_conn.backup(target_conn)
    except sqlite3.Error as exc:
        raise BackupError("SQLite 快照失败：%s" % exc) from exc
    _check_sqlite(destination)


def _copy_stable(source, destination):
    source = Path(source)
    destination = Path(destination)
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise BackupError("备份期间 parquet 文件发生变化，请停止数据更新后重试：%s" % source)


def _file_entry(stage_path, archive_path):
    return {
        "path": archive_path,
        "size": stage_path.stat().st_size,
        "sha256": _sha256_file(stage_path),
    }


def _build_archive(database_path, parquet_root, destination, include_parquet=True):
    database_path = Path(database_path)
    parquet_root = Path(parquet_root)
    destination = Path(destination)
    if destination.suffix.lower() != ".zip":
        destination = destination.with_suffix(".zip")
    if destination.exists():
        raise FileExistsError("备份文件已存在，拒绝覆盖：%s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="aquant_backup_stage_") as temp_dir:
        stage = Path(temp_dir)
        staged_db = stage / "aquant.db"
        _snapshot_database(database_path, staged_db)
        staged_files = [(staged_db, "aquant.db")]
        if include_parquet and parquet_root.exists():
            for source in sorted(parquet_root.rglob("*.parquet")):
                if not source.is_file():
                    continue
                archive_path = "parquet/" + source.relative_to(parquet_root).as_posix()
                _safe_archive_path(archive_path)
                target = stage / PurePosixPath(archive_path)
                _copy_stable(source, target)
                staged_files.append((target, archive_path))

        manifest = {
            "format_version": FORMAT_VERSION,
            "application": "aquant",
            "created_at": db.now_str(),
            "includes_parquet": bool(include_parquet),
            "database_integrity": "ok",
            "files": [_file_entry(path, archive_path) for path, archive_path in staged_files],
            "disclaimer": DISCLAIMER,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        descriptor, temp_name = tempfile.mkstemp(
            prefix=destination.stem + "_", suffix=".tmp", dir=str(destination.parent)
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path, archive_path in staged_files:
                    archive.write(path, arcname=archive_path)
                archive.write(manifest_path, arcname="manifest.json")
            temp_path.replace(destination)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    return {
        "path": str(destination),
        "size": destination.stat().st_size,
        "files": len(manifest["files"]),
        "parquet_files": sum(item["path"].startswith("parquet/") for item in manifest["files"]),
        "manifest": manifest,
    }


def create_backup(destination=None, data_dir=None, include_parquet=True):
    """创建 ZIP 备份；默认备份 data/aquant.db 和 data/parquet，不覆盖已有文件。"""
    data_dir = Path(data_dir) if data_dir is not None else Path(db.DATA_DIR)
    if destination is None:
        filename = "aquant_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".zip"
        destination = data_dir / "backup" / filename
    result = _build_archive(
        data_dir / "aquant.db", data_dir / "parquet", destination, include_parquet
    )
    return result


def verify_backup(path):
    """验证路径、文件清单、尺寸、SHA-256、ZIP CRC 和 SQLite 完整性。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("备份不存在：%s" % path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            actual = [info.filename for info in archive.infolist() if not info.is_dir()]
            if len(actual) != len(set(actual)):
                raise BackupError("备份 ZIP 包含重复路径")
            if "manifest.json" not in actual:
                raise BackupError("备份缺少 manifest.json")
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            if not isinstance(manifest, dict):
                raise BackupError("manifest 必须是 JSON 对象")
            if manifest.get("format_version") != FORMAT_VERSION:
                raise BackupError("不支持的备份格式版本：%s" % manifest.get("format_version"))
            if not isinstance(manifest.get("includes_parquet"), bool):
                raise BackupError("manifest 缺少有效的 includes_parquet")
            entries = manifest.get("files")
            if not isinstance(entries, list) or not entries:
                raise BackupError("manifest 文件清单为空")
            expected = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise BackupError("manifest 文件项必须是 JSON 对象")
                archive_path = entry.get("path")
                _safe_archive_path(archive_path)
                expected.append(archive_path)
            if len(expected) != len(set(expected)):
                raise BackupError("manifest 包含重复路径")
            if "aquant.db" not in expected:
                raise BackupError("备份缺少 aquant.db")
            if not manifest["includes_parquet"] and any(item.startswith("parquet/") for item in expected):
                raise BackupError("manifest 的 includes_parquet 与文件清单不符")
            if set(actual) != set(expected) | {"manifest.json"}:
                raise BackupError("ZIP 实际文件与 manifest 清单不一致")

            for entry in entries:
                archive_path = entry["path"]
                actual_size = archive.getinfo(archive_path).file_size
                with archive.open(archive_path, "r") as handle:
                    digest = _sha256_stream(handle)
                if actual_size != int(entry.get("size", -1)):
                    raise BackupError("文件尺寸校验失败：%s" % archive_path)
                if digest != entry.get("sha256"):
                    raise BackupError("SHA-256 校验失败：%s" % archive_path)

            with tempfile.TemporaryDirectory(prefix="aquant_verify_") as temp_dir:
                database_path = Path(temp_dir) / "aquant.db"
                with archive.open("aquant.db", "r") as source, database_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                _check_sqlite(database_path)
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise BackupError("备份格式或内容无效：%s" % exc) from exc
    return manifest


def restore_backup(path, target_dir=None, confirm=False, create_safety_backup=True):
    """
    恢复本地数据。必须 confirm=True，且应先关闭桌面程序和其他占用数据库的进程。
    includes_parquet=False 的备份只恢复 SQLite，不改动现有 parquet。
    """
    if confirm is not True:
        raise PermissionError("恢复会替换本地数据，必须显式传入 confirm=True")
    path = Path(path)
    manifest = verify_backup(path)
    target_dir = Path(target_dir) if target_dir is not None else Path(db.DATA_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_db = target_dir / "aquant.db"
    target_parquet = target_dir / "parquet"
    stage = Path(tempfile.mkdtemp(prefix=".aquant_restore_stage_", dir=str(target_dir.parent)))
    rollback = Path(tempfile.mkdtemp(prefix=".aquant_restore_rollback_", dir=str(target_dir.parent)))
    moved_db = moved_parquet = installed_db = installed_parquet = False
    commit_completed = rollback_completed = False
    safety_path = ""

    try:
        with zipfile.ZipFile(path, "r") as archive:
            for entry in manifest["files"]:
                archive_path = entry["path"]
                _safe_archive_path(archive_path)
                destination = stage / PurePosixPath(archive_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(archive_path, "r") as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if destination.stat().st_size != int(entry["size"]) or _sha256_file(destination) != entry["sha256"]:
                    raise BackupError("恢复前二次校验失败：%s" % archive_path)
        if manifest["includes_parquet"]:
            (stage / "parquet").mkdir(parents=True, exist_ok=True)
        _check_sqlite(stage / "aquant.db")

        if create_safety_backup and target_db.exists():
            safety_path = target_dir / "backup" / (
                "pre_restore_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".zip"
            )
            _build_archive(target_db, target_parquet, safety_path, include_parquet=True)
            safety_path = str(safety_path)

        try:
            if target_db.exists():
                shutil.move(str(target_db), str(rollback / "aquant.db"))
                moved_db = True
            if manifest["includes_parquet"] and target_parquet.exists():
                shutil.move(str(target_parquet), str(rollback / "parquet"))
                moved_parquet = True
            shutil.move(str(stage / "aquant.db"), str(target_db))
            installed_db = True
            if manifest["includes_parquet"]:
                shutil.move(str(stage / "parquet"), str(target_parquet))
                installed_parquet = True
            commit_completed = True
        except Exception as original_error:
            rollback_errors = []
            for installed, target, is_directory in [
                (installed_db, target_db, False),
                (installed_parquet, target_parquet, True),
            ]:
                if installed and target.exists():
                    try:
                        shutil.rmtree(target) if is_directory else target.unlink()
                    except Exception as exc:
                        rollback_errors.append(str(exc))
            for moved, old_path, target in [
                (moved_db, rollback / "aquant.db", target_db),
                (moved_parquet, rollback / "parquet", target_parquet),
            ]:
                if moved and old_path.exists():
                    try:
                        shutil.move(str(old_path), str(target))
                    except Exception as exc:
                        rollback_errors.append(str(exc))
            if rollback_errors:
                raise BackupError(
                    "恢复失败且自动回滚未完全成功。回滚目录已保留：%s；错误：%s"
                    % (rollback, "；".join(rollback_errors))
                ) from original_error
            rollback_completed = True
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if commit_completed or rollback_completed or not any(rollback.iterdir()):
            shutil.rmtree(rollback, ignore_errors=True)

    return {
        "database": str(target_db),
        "parquet": str(target_parquet) if manifest["includes_parquet"] else "",
        "safety_backup": safety_path,
        "restored_files": len(manifest["files"]),
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    # 临时目录完整自测：备份、校验、恢复、二次恢复安全备份；不触碰正式数据。
    with tempfile.TemporaryDirectory(prefix="aquant_backup_test_") as temp_dir:
        root = Path(temp_dir)
        source_dir = root / "source"
        parquet_dir = source_dir / "parquet" / "daily" / "qfq"
        parquet_dir.mkdir(parents=True)
        (parquet_dir / "000001.parquet").write_bytes(b"PAR1 local demo")
        with closing(sqlite3.connect(str(source_dir / "aquant.db"))) as connection:
            connection.execute("CREATE TABLE demo (value TEXT)")
            connection.execute("INSERT INTO demo VALUES ('backup-ok')")
            connection.commit()

        archive_path = root / "backup.zip"
        created = create_backup(destination=archive_path, data_dir=source_dir)
        verify_backup(archive_path)
        target_dir = root / "target"
        restored = restore_backup(archive_path, target_dir=target_dir, confirm=True)
        with closing(sqlite3.connect(str(target_dir / "aquant.db"))) as connection:
            assert connection.execute("SELECT value FROM demo").fetchone()[0] == "backup-ok"
        assert (target_dir / "parquet" / "daily" / "qfq" / "000001.parquet").exists()

        with closing(sqlite3.connect(str(target_dir / "aquant.db"))) as connection:
            connection.execute("INSERT INTO demo VALUES ('before-second-restore')")
            connection.commit()
        second = restore_backup(archive_path, target_dir=target_dir, confirm=True)
        assert second["safety_backup"] and Path(second["safety_backup"]).exists()
        verify_backup(second["safety_backup"])
        with closing(sqlite3.connect(str(target_dir / "aquant.db"))) as connection:
            assert connection.execute("SELECT COUNT(*) FROM demo").fetchone()[0] == 1
        assert created["parquet_files"] == 1 and restored["restored_files"] == 2
        print("backup.py 自测通过：SQLite/parquet 备份、校验、恢复及恢复前安全备份正常")
        print(DISCLAIMER)
