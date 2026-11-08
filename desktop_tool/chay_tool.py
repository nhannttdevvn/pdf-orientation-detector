#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Điểm khởi chạy thống nhất (dùng cho cả mã nguồn và bản đóng gói Windows / macOS / Linux):
- Không có tham số       -> mở Giao diện (GUI).
- Có tham số (--input…)  -> chạy Dòng lệnh (CLI), ví dụ:  KiemTraHoSo_CLI --input data/pdf --output reports
- File chạy tên *_CLI    -> luôn chạy Dòng lệnh.
- Máy không có màn hình đồ họa / thiếu Tkinter -> tự chuyển sang Dòng lệnh.
"""

import os
import sys

# Bản đóng gói không có console (GUI) có sys.stdout = None -> các module gọi print()/stdout.encoding sẽ lỗi.
for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def _want_cli():
    exe_name = os.path.splitext(os.path.basename(sys.executable if getattr(sys, "frozen", False) else sys.argv[0]))[0]
    return len(sys.argv) > 1 or exe_name.upper().endswith("_CLI")


def run_cli():
    from kiem_tra_ho_so import main
    main()


def run_gui():
    # Giao diện Tkinter ban đầu (gui.py); máy không có màn hình / thiếu Tkinter thì chạy CLI
    try:
        import gui
    except Exception as exc:
        print(f"[!] Không mở được giao diện ({exc}). Chuyển sang chế độ dòng lệnh.")
        run_cli()
        return
    gui.main()


if __name__ == "__main__":
    if _want_cli():
        run_cli()
    else:
        run_gui()
