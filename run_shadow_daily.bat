@echo off
REM =====================================================================
REM AQuant daily post-close shadow inference (FP16 IR, locked v3_fc_trend020)
REM Log only: never auto-tune parameters, never write UPGRADE_SUCCESS.flag
REM Task Scheduler setup: see shadow_daily_task.md
REM =====================================================================
setlocal
cd /d "%~dp0"

set "LOG_DIR=%~dp0logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "STAMP=%%i"
set "LOG=%LOG_DIR%\shadow_%STAMP%.log"

echo [%date% %time%] shadow start>>"%LOG%"

REM Skip weekends; holidays still run, streak de-duplicates by data end date.
for /f %%d in ('powershell -NoProfile -Command "(Get-Date).DayOfWeek.value__"') do set "DOW=%%d"
if %DOW% GEQ 6 goto weekend

".\.venv\Scripts\python.exe" shadow_infer.py --days 63 --output shadow_output --refresh >>"%LOG%" 2>&1
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" ".\.venv\Scripts\python.exe" monitor_dashboard.py --limit 10 >>"%LOG%" 2>&1
echo [%date% %time%] shadow end exit=%CODE% >>"%LOG%"
if not "%CODE%"=="0" echo [%date% %time%] FAILED exit=%CODE% >>"%LOG%"
exit /b %CODE%

:weekend
echo [%date% %time%] weekend, skip>>"%LOG%"
exit /b 0
