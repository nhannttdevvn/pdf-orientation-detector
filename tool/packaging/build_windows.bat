@echo off
chcp 65001 >nul
setlocal
REM ============================================================================
REM  BUILD BAN CHAY DOC LAP CHO WINDOWS (chay 1 lan tren may Windows de dong goi)
REM  Can co san:
REM    - Python 3.11 (64-bit) tu python.org (khi cai nho tick "Add python.exe to PATH")
REM    - (Nen co) Tesseract OCR tai C:\Program Files\Tesseract-OCR  -> se duoc dong goi kem
REM      Tai bo cai: https://github.com/UB-Mannheim/tesseract/wiki
REM  Ket qua: packaging\_build\KiemTraHoSo-windows-x64.zip
REM ============================================================================
cd /d "%~dp0.."

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
echo [*] Dung Python: %PY%

if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo [*] Tim thay Tesseract, se dong goi kem vao ban build.
) else (
    if defined TESSERACT_DIR (
        echo [*] Dung Tesseract tai TESSERACT_DIR=%TESSERACT_DIR%
    ) else (
        echo [!] Khong thay Tesseract tai C:\Program Files\Tesseract-OCR.
        echo     Ban build van chay duoc ^(kiem tra trang xoay bang RapidOCR^),
        echo     nhung nen cai Tesseract truoc de dong goi kem: https://github.com/UB-Mannheim/tesseract/wiki
    )
)

if not exist packaging\_build mkdir packaging\_build
if not exist packaging\_build\venv (
    echo [*] Tao moi truong build...
    %PY% -m venv packaging\_build\venv || goto :fail
)
call packaging\_build\venv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip || goto :fail
python -m pip install -r packaging\requirements-build.txt || goto :fail
python packaging\build.py || goto :fail

echo.
echo [OK] Da tao: %CD%\packaging\_build\KiemTraHoSo-windows-x64.zip
explorer "%CD%\packaging\_build"
pause
exit /b 0

:fail
echo.
echo [LOI] Build that bai. Xem thong bao loi o tren.
pause
exit /b 1
