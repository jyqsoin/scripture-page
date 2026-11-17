@echo off
rem AQuant launch wrapper (ASCII-safe for cmd codepage 936)
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "ui_main.py"
    exit /b 0
)
if exist ".venv\Scripts\python.exe" (
    start "" ".venv\Scripts\python.exe" "ui_main.py"
    exit /b 0
)
python ui_main.py
