# -*- coding: utf-8 -*-
"""在用户桌面创建「AQuant 量化研究」快捷方式（PowerShell COM，离线）。

只是创建本地 .lnk，不安装程序、不打包、不修改注册表、不联网。
仅研究用途，不构成投资建议。
"""

import base64
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "launch_aquant.bat"
ICON = ROOT / "assets" / "app_icon.ico"
SHORTCUT_NAME = "AQuant 量化研究.lnk"


def _quote(value):
    """PowerShell 单引号字符串转义。"""
    return str(value).replace("'", "''")


def build_powershell(shortcut_name=SHORTCUT_NAME, target=TARGET,
                     working_dir=ROOT, icon=ICON):
    return (
        "$desktop = [Environment]::GetFolderPath('Desktop')\n"
        "$shell = New-Object -ComObject WScript.Shell\n"
        "$path = Join-Path $desktop '%s'\n"
        "$shortcut = $shell.CreateShortcut($path)\n"
        "$shortcut.TargetPath = '%s'\n"
        "$shortcut.WorkingDirectory = '%s'\n"
        "$shortcut.IconLocation = '%s'\n"
        "$shortcut.Description = 'AQuant 量化研究（本地离线，仅研究用途，不构成投资建议）'\n"
        "$shortcut.WindowStyle = 7\n"
        "$shortcut.Save()\n"
        "Write-Output $path\n"
        % (_quote(shortcut_name), _quote(target), _quote(working_dir),
           _quote(icon))
    )


def create_shortcut():
    """调用 PowerShell 生成快捷方式，返回 (路径, 是否存在)。"""
    if not TARGET.exists():
        raise FileNotFoundError("缺少启动脚本：%s" % TARGET)
    if not ICON.exists():
        raise FileNotFoundError("缺少图标文件：%s（先运行 tools/make_app_icon.py）" % ICON)
    script = build_powershell()
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError("PowerShell 创建快捷方式失败：%s" % result.stderr.strip())
    path = Path(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else None
    if path is None:
        raise RuntimeError("PowerShell 未返回快捷方式路径")
    return path, path.exists()


def main():
    try:
        path, exists = create_shortcut()
    except Exception as exc:
        print("创建失败：%s" % exc)
        return 1
    print("快捷方式：", path)
    print("目标：", TARGET)
    print("工作目录：", ROOT)
    print("图标：", ICON)
    print("状态：", "已生成" if exists else "未找到（请检查桌面权限）")
    print("仅研究用途，不构成投资建议。")
    return 0 if exists else 1


if __name__ == "__main__":
    sys.exit(main())
