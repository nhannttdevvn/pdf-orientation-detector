@echo off
chcp 65001 >nul
setlocal
REM ============================================================================
REM  CHAY TOOL KIEM TRA HO SO TREN WINDOWS (tu ma nguon)
REM  Lan dau: tu tao moi truong .venv va cai thu vien (can Internet, khoang 3-5 phut).
REM  Cac lan sau: mo giao dien ngay.
REM  Can co: Python 3.11 64-bit (https://www.python.org/downloads/ , tick "Add python.exe to PATH")
REM ============================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" goto :run

set "PY="
py -3.11 --version >nul 2>&1 && set "PY=py -3.11"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [LOI] Khong tim thay Python. Hay cai Python 3.11 64-bit tu https://www.python.org/downloads/
    echo       va tick "Add python.exe to PATH" khi cai, roi chay lai file nay.
    pause
    exit /b 1
)
echo [*] Lan dau chay: dang cai dat thu vien, vui long doi...
%PY% -m venv .venv || goto :fail
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirement.txt || goto :fail

:run
start "" ".venv\Scripts\pythonw.exe" chay_tool.py
exit /b 0

:fail
echo.
echo [LOI] Cai dat that bai. Kiem tra ket noi Internet roi chay lai.
pause
exit /b 1
