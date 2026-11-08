#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Đường dẫn & thao tác hệ điều hành dùng chung cho Windows / macOS / Linux,
chạy đúng cả khi chạy từ mã nguồn lẫn khi đã đóng gói bằng PyInstaller.

- resource_dir(): nơi chứa tài nguyên đi kèm (templates/, Tesseract-OCR/).
    + Mã nguồn : thư mục chứa file .py
    + Đóng gói : thư mục nội bộ của PyInstaller (sys._MEIPASS)
- user_dir(): nơi người dùng nhìn thấy (data/pdf, reports/).
    + Mã nguồn : thư mục chứa file .py
    + Đóng gói : thư mục chứa file chạy (KiemTraHoSo / KiemTraHoSo.exe)
- open_path(): mở file / thư mục bằng ứng dụng mặc định của hệ điều hành.
"""

import os
import subprocess
import sys

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def resource_dir():
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return SOURCE_DIR


def user_dir():
    if is_frozen():
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        if ".app/Contents/MacOS" in exe_dir:
            # Ứng dụng macOS (.app): không ghi vào bên trong gói ứng dụng -> dùng ~/Documents/KiemTraHoSo
            docs = os.path.join(os.path.expanduser("~"), "Documents", "KiemTraHoSo")
            os.makedirs(os.path.join(docs, "data", "pdf"), exist_ok=True)
            return docs
        return exe_dir
    return SOURCE_DIR


def patch_opencv_unicode_paths():
    """
    Trên Windows, cv2.imread / cv2.imwrite không đọc/ghi được đường dẫn có ký tự tiếng Việt
    (vd. 'D:\\Hồ sơ\\...', hoặc tên ảnh '..._MẬT.png') và thất bại mà không báo lỗi.
    Thay bằng bản đọc/ghi qua numpy (np.fromfile / ndarray.tofile) cho mọi module dùng cv2.
    """
    import cv2
    import numpy as np
    if getattr(cv2, "_kths_unicode_patched", False):
        return
    _orig_imread, _orig_imwrite = cv2.imread, cv2.imwrite

    def imread(path, flags=cv2.IMREAD_COLOR):
        if not os.path.isfile(str(path)):
            return None
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, flags)
        except Exception:
            return _orig_imread(str(path), flags)

    def imwrite(path, img, params=None):
        try:
            ext = os.path.splitext(str(path))[1] or ".png"
            ok, buf = cv2.imencode(ext, img, params if params is not None else [])
            if ok:
                buf.tofile(str(path))
            return bool(ok)
        except Exception:
            return _orig_imwrite(str(path), img) if params is None else _orig_imwrite(str(path), img, params)

    cv2.imread, cv2.imwrite = imread, imwrite
    cv2._kths_unicode_patched = True


def open_path(path):
    """Mở file/thư mục bằng ứng dụng mặc định (Explorer / Finder / trình quản lý file Linux)."""
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
