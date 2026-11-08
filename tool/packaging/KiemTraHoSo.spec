# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: build bản chạy độc lập cho Windows / macOS / Linux (chạy bằng packaging/build.py).
# Kết quả: dist/KiemTraHoSo/ gồm 2 file chạy dùng chung thư viện:
#   - KiemTraHoSo      : Giao diện (GUI)
#   - KiemTraHoSo_CLI  : Dòng lệnh (CLI, có cửa sổ console)
# Trên macOS có thêm dist/KiemTraHoSo.app (ứng dụng Mac: bấm đúp là mở, không hiện Terminal).
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
TESS_STAGE = os.environ.get("KTHS_TESSERACT_STAGE", "")

datas = [(os.path.join(ROOT, "templates"), "templates")]
datas += collect_data_files("rapidocr_onnxruntime")          # config.yaml + model .onnx
if TESS_STAGE and os.path.isdir(TESS_STAGE):
    datas.append((TESS_STAGE, "Tesseract-OCR"))              # Tesseract đóng gói kèm (Windows)

hiddenimports = collect_submodules("rapidocr_onnxruntime") + [
    "gui", "kiem_tra_ho_so", "xoay_trang", "doi_chieu_muc_luc", "app_paths",
    "hough_line_detector", "detect_confidential_docs_fast",
]

a = Analysis(
    [os.path.join(ROOT, "chay_tool.py")],
    pathex=[ROOT],
    binaries=collect_dynamic_libs("onnxruntime"),
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pandas", "matplotlib", "scipy", "IPython", "pytest", "notebook", "PyQt5", "PyQt6", "PySide6", "shiboken6"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe_gui = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="KiemTraHoSo",
    console=False,
    upx=False,
)
exe_cli = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="KiemTraHoSo_CLI",
    console=True,
    upx=False,
)
coll = COLLECT(
    exe_gui, exe_cli,
    a.binaries, a.datas,
    upx=False,
    name="KiemTraHoSo",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="KiemTraHoSo.app",
        bundle_identifier="vn.kiemtrahoso.tool",
        info_plist={
            "CFBundleDisplayName": "Kiểm Tra Hồ Sơ",
            "CFBundleName": "KiemTraHoSo",
            "CFBundleShortVersionString": "1.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
