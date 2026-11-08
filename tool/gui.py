#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HỆ THỐNG KIỂM TRA & BÓC TÁCH HỒ SƠ LƯU TRỮ VĂN BẢN (GUI DASHBOARD v4.0)
Hỗ trợ:
1. Đa ngôn ngữ: Phân tách & Nhận diện Văn bản Tiếng Anh & Văn bản Tiếng Việt.
2. Kiểm tra tính khép kín từng văn bản [START -> END].
3. Kiểm tra chất lượng: Tài liệu Mật (>= 0.80) & Sọc Scan.
4. Giao diện 2 Tab: Tổng hợp theo File PDF & Chi tiết từng văn bản bóc tách.
5. Xuất báo cáo Excel 6 Sheet & 1-Click mở nhanh mọi thư mục ảnh minh chứng.
================================================================================
"""

import os
import sys
import glob
import re
import threading
import time
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import pymupdf

# Cấu hình UTF-8 cho console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from kiem_tra_ho_so import BiLingualDocumentChecker, natural_sort_key
from app_paths import open_path, user_dir

USER_DIR = user_dir()


class ModernBiLingualGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Hệ Thống Kiểm Tra & Bóc Tách Hồ Sơ Văn Bản Đa Ngôn Ngữ (Anh - Việt v4.0)")
        self.geometry("1200x820")
        self.minsize(1050, 700)
        
        # Đường dẫn mặc định
        default_input = os.path.join(USER_DIR, "data", "pdf")
        if not glob.glob(os.path.join(default_input, "*.pdf")):
            candidates = [
                os.path.join(USER_DIR, "data", "pdf"),
                os.path.join(os.path.dirname(USER_DIR), "tool_tailieu", "data", "pdf"),
                r"C:\tool_tailieu\data\pdf",
                r"C:\Tool\data\pdf",
                r"C:\Tool\tách sai\tách sai",
                CURRENT_DIR
            ]
            for c in candidates:
                if os.path.exists(c) and len(glob.glob(os.path.join(c, "*.pdf"))) > 0:
                    default_input = c
                    break
                    
        default_output = os.path.join(USER_DIR, "reports")
        
        self.input_dir_var = tk.StringVar(value=default_input)
        self.output_dir_var = tk.StringVar(value=default_output)
        self.status_var = tk.StringVar(value="Sẵn sàng kiểm tra và bóc tách hồ sơ văn bản.")
        self.progress_val = tk.DoubleVar(value=0.0)
        
        # Thống kê KPI
        self.kpi_total_files = tk.StringVar(value="0")
        self.kpi_total_docs = tk.StringVar(value="0")
        self.kpi_en_docs = tk.StringVar(value="0")
        self.kpi_vn_docs = tk.StringVar(value="0")
        self.kpi_mat_pages = tk.StringVar(value="0 trang")
        self.kpi_soc_pages = tk.StringVar(value="0 trang")
        self.kpi_xoay_pages = tk.StringVar(value="0 trang")
        self.kpi_overall_status = tk.StringVar(value="Chờ quét")
        
        self.is_running = False
        self.last_excel_path = None
        self.last_evidence_dir = os.path.join(default_output, "evidence_crops")
        self.last_mat_dir = os.path.join(default_output, "tai_lieu_mat")
        self.last_soc_dir = os.path.join(default_output, "anh_co_soc")
        self.last_xoay_dir = os.path.join(default_output, "trang_bi_xoay")
        self.last_pdf_xoay_dir = os.path.join(default_output, "pdf_da_xoay")
        self.current_results = None
        
        self._build_ui()
        
    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except Exception:
            pass
            
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#E2E8F0", foreground="#1E293B")
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=25)
        style.map("Treeview", background=[('selected', '#3B82F6')])
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 6])
        
        # =====================================================================
        # 1. HEADER BANNER HIỆN ĐẠI
        # =====================================================================
        header_frame = tk.Frame(self, bg="#0F172A", height=76)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        title_box = tk.Frame(header_frame, bg="#0F172A")
        title_box.pack(side=tk.LEFT, padx=18, pady=10)
        
        lbl_title = tk.Label(
            title_box,
            text="HỆ THỐNG KIỂM TRA & BÓC TÁCH HỒ SƠ LƯU TRỮ ĐA NGÔN NGỮ",
            font=("Segoe UI", 15, "bold"),
            fg="#FFFFFF",
            bg="#0F172A"
        )
        lbl_title.pack(anchor="w")
        
        lbl_sub = tk.Label(
            title_box,
            text="Phân loại ranh giới Đầu/Cuối Tiếng Anh & Tiếng Việt • Khép kín văn bản • Dấu Mật >= 0.80 • Sọc Scan",
            font=("Segoe UI", 9),
            fg="#94A3B8",
            bg="#0F172A"
        )
        lbl_sub.pack(anchor="w")
        
        lbl_badge = tk.Label(
            header_frame,
            text="v4.0 BI-LINGUAL",
            font=("Segoe UI", 9, "bold"),
            fg="#0F172A",
            bg="#38BDF8",
            padx=12,
            pady=4,
            relief=tk.FLAT
        )
        lbl_badge.pack(side=tk.RIGHT, padx=18, pady=20)
        
        # =====================================================================
        # MAIN CONTAINER
        # =====================================================================
        main_container = tk.Frame(self, bg="#F8FAFC", padx=12, pady=6)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # =====================================================================
        # 2. KHUNG CHỌN ĐƯỜNG DẪN (FOLDER CONFIG)
        # =====================================================================
        cfg_frame = tk.LabelFrame(main_container, text=" Cấu hình thư mục dữ liệu ", font=("Segoe UI", 9, "bold"), bg="#FFFFFF", fg="#1E293B", padx=10, pady=5)
        cfg_frame.pack(fill=tk.X, pady=(0, 6))
        
        tk.Label(cfg_frame, text="Thư mục PDF đầu vào:", font=("Segoe UI", 9, "bold"), bg="#FFFFFF", fg="#334155").grid(row=0, column=0, sticky="w", pady=2)
        ent_in = ttk.Entry(cfg_frame, textvariable=self.input_dir_var, font=("Segoe UI", 9))
        ent_in.grid(row=0, column=1, sticky="ew", padx=8, pady=2)
        btn_bin = tk.Button(cfg_frame, text="Chọn...", font=("Segoe UI", 9), bg="#E2E8F0", relief=tk.GROOVE, command=self._browse_input)
        btn_bin.grid(row=0, column=2, padx=2, pady=2)
        
        tk.Label(cfg_frame, text="Thư mục xuất báo cáo:", font=("Segoe UI", 9, "bold"), bg="#FFFFFF", fg="#334155").grid(row=1, column=0, sticky="w", pady=2)
        ent_out = ttk.Entry(cfg_frame, textvariable=self.output_dir_var, font=("Segoe UI", 9))
        ent_out.grid(row=1, column=1, sticky="ew", padx=8, pady=2)
        btn_bout = tk.Button(cfg_frame, text="Chọn...", font=("Segoe UI", 9), bg="#E2E8F0", relief=tk.GROOVE, command=self._browse_output)
        btn_bout.grid(row=1, column=2, padx=2, pady=2)
        
        cfg_frame.columnconfigure(1, weight=1)
        
        # =====================================================================
        # 3. THẺ THỐNG KÊ KPI NHANH (8 CARDS)
        # =====================================================================
        kpi_frame = tk.Frame(main_container, bg="#F8FAFC")
        kpi_frame.pack(fill=tk.X, pady=(0, 6))
        
        def create_kpi_card(parent, title, var, bg_color, fg_color):
            card = tk.Frame(parent, bg=bg_color, padx=8, pady=5, relief=tk.GROOVE, bd=1)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            tk.Label(card, text=title, font=("Segoe UI", 8, "bold"), fg=fg_color, bg=bg_color).pack(anchor="w")
            tk.Label(card, textvariable=var, font=("Segoe UI", 13, "bold"), fg=fg_color, bg=bg_color).pack(anchor="w", pady=(1, 0))
            return card
            
        create_kpi_card(kpi_frame, "TỔNG SỐ FILE", self.kpi_total_files, "#F1F5F9", "#0F172A")
        create_kpi_card(kpi_frame, "TỔNG VĂN BẢN", self.kpi_total_docs, "#E0E7FF", "#3730A3")
        create_kpi_card(kpi_frame, "VB TIẾNG ANH", self.kpi_en_docs, "#EFF6FF", "#1D4ED8")
        create_kpi_card(kpi_frame, "VB TIẾNG VIỆT", self.kpi_vn_docs, "#ECFDF5", "#047857")
        create_kpi_card(kpi_frame, "TÀI LIỆU MẬT", self.kpi_mat_pages, "#FEF3C7", "#B45309")
        create_kpi_card(kpi_frame, "SỌC SCAN", self.kpi_soc_pages, "#F1F5F9", "#475569")
        create_kpi_card(kpi_frame, "TRANG BỊ XOAY", self.kpi_xoay_pages, "#F5F3FF", "#6D28D9")
        create_kpi_card(kpi_frame, "KẾT LUẬN CHUNG", self.kpi_overall_status, "#F8FAFC", "#0F172A")
        
        # =====================================================================
        # 4. NÚT CHẠY 1-CLICK & CÁC NÚT MỞ NHANH
        # =====================================================================
        action_bar = tk.Frame(main_container, bg="#F8FAFC")
        action_bar.pack(fill=tk.X, pady=(0, 6))
        
        self.btn_run = tk.Button(
            action_bar,
            text="🚀  BẮT ĐẦU KIỂM TRA & BÓC TÁCH HỒ SƠ (1-CLICK)",
            font=("Segoe UI", 10, "bold"),
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            padx=14,
            pady=7,
            command=self._start_pipeline
        )
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.btn_excel = tk.Button(
            action_bar,
            text="📊  Báo Cáo Excel (6 Sheet)",
            font=("Segoe UI", 9, "bold"),
            bg="#059669",
            fg="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=7,
            command=self._open_excel
        )
        self.btn_excel.pack(side=tk.LEFT, padx=(6, 0))
        
        self.btn_mat = tk.Button(
            action_bar,
            text="🔒  Dấu Mật",
            font=("Segoe UI", 9, "bold"),
            bg="#D97706",
            fg="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=7,
            command=self._open_mat_dir
        )
        self.btn_mat.pack(side=tk.LEFT, padx=(6, 0))
        
        self.btn_soc = tk.Button(
            action_bar,
            text="⚡  Sọc Scan",
            font=("Segoe UI", 9, "bold"),
            bg="#475569",
            fg="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=7,
            command=self._open_soc_dir
        )
        self.btn_soc.pack(side=tk.LEFT, padx=(6, 0))

        self.btn_xoay = tk.Button(
            action_bar,
            text="🔄  Trang Xoay",
            font=("Segoe UI", 9, "bold"),
            bg="#7C3AED",
            fg="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=7,
            command=self._open_xoay_dir
        )
        self.btn_xoay.pack(side=tk.LEFT, padx=(6, 0))

        self.btn_pdf_xoay = tk.Button(
            action_bar,
            text="📄  PDF Đã Xoay",
            font=("Segoe UI", 9, "bold"),
            bg="#6D28D9",
            fg="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=7,
            command=self._open_pdf_xoay_dir
        )
        self.btn_pdf_xoay.pack(side=tk.LEFT, padx=(6, 0))

        self.btn_evidence = tk.Button(
            action_bar,
            text="📁  Ảnh Minh Chứng",
            font=("Segoe UI", 9, "bold"),
            bg="#DC2626",
            fg="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=7,
            command=self._open_evidence
        )
        self.btn_evidence.pack(side=tk.LEFT, padx=(6, 0))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(main_container, orient=tk.HORIZONTAL, mode='determinate', variable=self.progress_val)
        self.progress_bar.pack(fill=tk.X, pady=(0, 6))
        
        # =====================================================================
        # 5. NOTEBOOK TABS: TAB 1 (THEO FILE) & TAB 2 (BÓC TÁCH VĂN BẢN)
        # =====================================================================
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        
        # --- TAB 1: TỔNG QUAN THEO FILE ---
        tab1 = tk.Frame(self.notebook, bg="#FFFFFF")
        self.notebook.add(tab1, text="  📁  Tổng Quan Theo File PDF  ")
        
        cols1 = ("stt", "file", "pages", "total_docs", "en", "vn", "valid", "error", "mat", "soc", "xoay", "status", "detail")
        self.tree_files = ttk.Treeview(tab1, columns=cols1, show="headings", selectmode="browse")
        
        self.tree_files.heading("stt", text="STT")
        self.tree_files.heading("file", text="Tên File PDF")
        self.tree_files.heading("pages", text="Tổng Trang")
        self.tree_files.heading("total_docs", text="Tổng VB")
        self.tree_files.heading("en", text="VB Tiếng Anh")
        self.tree_files.heading("vn", text="VB Tiếng Việt")
        self.tree_files.heading("valid", text="VB Hợp Lệ")
        self.tree_files.heading("error", text="VB Lỗi")
        self.tree_files.heading("mat", text="Dấu Mật")
        self.tree_files.heading("soc", text="Sọc Scan")
        self.tree_files.heading("xoay", text="Trang Xoay")
        self.tree_files.heading("status", text="Kết Luận")
        self.tree_files.heading("detail", text="Mô Tả Chi Tiết")
        
        self.tree_files.column("stt", width=38, anchor="center")
        self.tree_files.column("file", width=110, anchor="w")
        self.tree_files.column("pages", width=65, anchor="center")
        self.tree_files.column("total_docs", width=65, anchor="center")
        self.tree_files.column("en", width=80, anchor="center")
        self.tree_files.column("vn", width=80, anchor="center")
        self.tree_files.column("valid", width=65, anchor="center")
        self.tree_files.column("error", width=60, anchor="center")
        self.tree_files.column("mat", width=85, anchor="center")
        self.tree_files.column("soc", width=80, anchor="center")
        self.tree_files.column("xoay", width=240, anchor="w")
        self.tree_files.column("status", width=85, anchor="center")
        self.tree_files.column("detail", width=280, anchor="w")
        
        self.tree_files.tag_configure("error_row", background="#FEE2E2", foreground="#991B1B")
        self.tree_files.tag_configure("valid_row", background="#DCFCE7", foreground="#166534")
        self.tree_files.tag_configure("review_row", background="#FEF3C7", foreground="#92400E")
        
        scr_y1 = ttk.Scrollbar(tab1, orient=tk.VERTICAL, command=self.tree_files.yview)
        scr_x1 = ttk.Scrollbar(tab1, orient=tk.HORIZONTAL, command=self.tree_files.xview)
        self.tree_files.configure(yscrollcommand=scr_y1.set, xscrollcommand=scr_x1.set)
        
        self.tree_files.grid(row=0, column=0, sticky="nsew")
        scr_y1.grid(row=0, column=1, sticky="ns")
        scr_x1.grid(row=1, column=0, sticky="ew")
        tab1.rowconfigure(0, weight=1)
        tab1.columnconfigure(0, weight=1)
        
        # --- TAB 2: BÓC TÁCH CHI TIẾT TỪNG VĂN BẢN ---
        tab2 = tk.Frame(self.notebook, bg="#FFFFFF")
        self.notebook.add(tab2, text="  📑  Bóc Tách Chi Tiết Từng Văn Bản (Anh / Việt)  ")
        
        cols2 = ("stt", "file", "doc_id", "lang", "type", "start", "end", "count", "status", "summary")
        self.tree_docs = ttk.Treeview(tab2, columns=cols2, show="headings", selectmode="browse")
        
        self.tree_docs.heading("stt", text="STT")
        self.tree_docs.heading("file", text="Tên File PDF")
        self.tree_docs.heading("doc_id", text="VB Số")
        self.tree_docs.heading("lang", text="Ngôn Ngữ")
        self.tree_docs.heading("type", text="Loại Văn Bản")
        self.tree_docs.heading("start", text="Trang Đầu")
        self.tree_docs.heading("end", text="Trang Cuối")
        self.tree_docs.heading("count", text="Số Trang")
        self.tree_docs.heading("status", text="Trạng Thái Khép Kín")
        self.tree_docs.heading("summary", text="Trích Yếu / Tiêu Đề")
        
        self.tree_docs.column("stt", width=40, anchor="center")
        self.tree_docs.column("file", width=110, anchor="w")
        self.tree_docs.column("doc_id", width=50, anchor="center")
        self.tree_docs.column("lang", width=75, anchor="center")
        self.tree_docs.column("type", width=190, anchor="w")
        self.tree_docs.column("start", width=70, anchor="center")
        self.tree_docs.column("end", width=70, anchor="center")
        self.tree_docs.column("count", width=65, anchor="center")
        self.tree_docs.column("status", width=150, anchor="center")
        self.tree_docs.column("summary", width=260, anchor="w")
        
        self.tree_docs.tag_configure("doc_error", background="#FEE2E2", foreground="#991B1B")
        self.tree_docs.tag_configure("doc_valid", background="#DCFCE7", foreground="#166534")
        self.tree_docs.tag_configure("doc_review", background="#FEF3C7", foreground="#92400E")
        
        scr_y2 = ttk.Scrollbar(tab2, orient=tk.VERTICAL, command=self.tree_docs.yview)
        scr_x2 = ttk.Scrollbar(tab2, orient=tk.HORIZONTAL, command=self.tree_docs.xview)
        self.tree_docs.configure(yscrollcommand=scr_y2.set, xscrollcommand=scr_x2.set)
        
        self.tree_docs.grid(row=0, column=0, sticky="nsew")
        scr_y2.grid(row=0, column=1, sticky="ns")
        scr_x2.grid(row=1, column=0, sticky="ew")
        tab2.rowconfigure(0, weight=1)
        tab2.columnconfigure(0, weight=1)
        
        # =====================================================================
        # 6. KHU VỰC LOG CONSOLE
        # =====================================================================
        log_frame = tk.LabelFrame(main_container, text=" Nhật ký kiểm tra thời gian thực (Console Log) ", font=("Segoe UI", 8, "bold"), bg="#FFFFFF", fg="#475569", padx=6, pady=3)
        log_frame.pack(fill=tk.X)
        
        self.txt_log = tk.Text(log_frame, height=4, font=("Consolas", 8), bg="#0F172A", fg="#E2E8F0", insertbackground="#FFFFFF", wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=log_scroll.set)
        
        self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # =====================================================================
        # 7. STATUS BAR
        # =====================================================================
        status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg="#E2E8F0",
            fg="#334155",
            anchor="w",
            padx=10,
            pady=3
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _browse_input(self):
        d = filedialog.askdirectory(initialdir=self.input_dir_var.get(), title="Chọn thư mục chứa file PDF")
        if d:
            self.input_dir_var.set(d)

    def _browse_output(self):
        d = filedialog.askdirectory(initialdir=self.output_dir_var.get(), title="Chọn thư mục xuất báo cáo")
        if d:
            self.output_dir_var.set(d)

    def _log(self, msg):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)

    def _open_folder_safely(self, path, name):
        if not path or not os.path.exists(path):
            out = self.output_dir_var.get().strip()
            path = os.path.join(out, name)
        if os.path.exists(path):
            try:
                open_path(path)
            except Exception as e:
                messagebox.showerror("Lỗi mở thư mục", f"Không thể mở thư mục {name}: {e}")
        else:
            messagebox.showinfo("Thông báo", f"Thư mục '{name}' chưa tồn tại hoặc chưa có ảnh được tạo.")

    def _open_excel(self):
        out = self.output_dir_var.get().strip()
        candidates = [
            self.last_excel_path,
            os.path.join(out, "bao_cao_kiem_tra.xlsx"),
            os.path.join(out, "bao_cao_sai_lech_muc_luc.xlsx"),
            os.path.join(out, "bao_cao_kiem_tra_ket_qua.xlsx")
        ]
        target = None
        for c in candidates:
            if c and os.path.exists(c):
                target = c
                break
                
        if target:
            try:
                open_path(target)
            except Exception as e:
                messagebox.showerror("Lỗi mở file", f"Không thể mở file Excel: {e}")
        else:
            messagebox.showinfo("Thông báo", "Chưa có file báo cáo Excel được tạo.")

    def _open_evidence(self):
        self._open_folder_safely(self.last_evidence_dir, "evidence_crops")

    def _open_mat_dir(self):
        self._open_folder_safely(self.last_mat_dir, "tai_lieu_mat")

    def _open_soc_dir(self):
        self._open_folder_safely(self.last_soc_dir, "anh_co_soc")

    def _open_xoay_dir(self):
        self._open_folder_safely(self.last_xoay_dir, "trang_bi_xoay")

    def _open_pdf_xoay_dir(self):
        self._open_folder_safely(self.last_pdf_xoay_dir, "pdf_da_xoay")

    def _start_pipeline(self):
        if self.is_running:
            return
            
        in_dir = self.input_dir_var.get().strip()
        out_dir = self.output_dir_var.get().strip()
        
        if not os.path.exists(in_dir):
            messagebox.showerror("Lỗi thư mục", f"Thư mục đầu vào không tồn tại:\n{in_dir}")
            return
            
        self.is_running = True
        self.btn_run.configure(state=tk.DISABLED, text="⏳  ĐANG KIỂM TRA & BÓC TÁCH HỒ SƠ...")
        
        # Xóa các bảng cũ
        for item in self.tree_files.get_children():
            self.tree_files.delete(item)
        for item in self.tree_docs.get_children():
            self.tree_docs.delete(item)
            
        self.txt_log.delete("1.0", tk.END)
        self.progress_val.set(0.0)
        self.status_var.set("Đang khởi tạo engine bóc tách đa ngôn ngữ...")
        
        t = threading.Thread(target=self._run_worker, args=(in_dir, out_dir), daemon=True)
        t.start()

    def _run_worker(self, in_dir, out_dir):
        try:
            checker = BiLingualDocumentChecker(target_dir=in_dir, output_dir=out_dir)
            self.last_evidence_dir = checker.evidence_dir
            self.last_mat_dir = checker.mat_dir
            self.last_soc_dir = checker.soc_dir
            self.last_xoay_dir = checker.xoay_dir
            self.last_pdf_xoay_dir = checker.pdf_xoay_dir

            def on_progress(pct, text):
                self.after(0, lambda: self.progress_val.set(pct))
                self.after(0, lambda: self.status_var.set(text))
                
            def on_log(msg):
                self.after(0, lambda: self._log(msg))
                
            def on_row(r):
                self.after(0, lambda: self._insert_file_row(r))
                
            code, results, saved_excel = checker.run(
                progress_callback=on_progress,
                log_callback=on_log,
                row_callback=on_row
            )
            
            self.last_excel_path = saved_excel
            self.current_results = results
            
            # Cập nhật danh sách từng văn bản sang Tab 2
            if results and results.get('segmented_docs'):
                self.after(0, lambda: self._populate_segmented_docs(results['segmented_docs']))
                
            self.after(0, lambda: self._finish_callback(results, None))
            
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self._log(f"[LỖI NGOẠI LỆ] {err_msg}"))
            self.after(0, lambda: self._finish_callback(None, err_msg))

    def _insert_file_row(self, r):
        tag = {"Lỗi": "error_row", "Cần kiểm tra": "review_row"}.get(r['ket_luan'], "valid_row")
        mat_text = f"{r['mat_count']} trang" if r['mat_count'] > 0 else "-"
        soc_text = f"{r['soc_count']} trang" if r['soc_count'] > 0 else "-"
        # Ghi rõ trang nào cần xoay, ví dụ "Trang 2 (cần xoay 90°), Trang 5 (cần kiểm tra)"
        xoay_text = r.get('xoay_str') or "-"

        self.tree_files.insert("", tk.END, values=(
            r['stt'],
            r['file_name'],
            r['total_pages'],
            r['total_docs'],
            r['en_docs'],
            r['vn_docs'],
            r['valid_docs'],
            r['error_docs'],
            mat_text,
            soc_text,
            xoay_text,
            r['ket_luan'],
            r['chi_tiet']
        ), tags=(tag,))

    def _populate_segmented_docs(self, segmented_docs):
        for idx, d in enumerate(segmented_docs, start=1):
            tag = "doc_error" if "Lỗi" in d['status'] else ("doc_review" if "Cần kiểm tra" in d['status'] else "doc_valid")
            self.tree_docs.insert("", tk.END, values=(
                idx,
                d['file'],
                d['doc_id'],
                d['lang'],
                d['type'],
                d['start_page'],
                d['end_page'],
                d['page_count'],
                d['status'],
                d['summary']
            ), tags=(tag,))

    def _finish_callback(self, results, err):
        self.is_running = False
        self.btn_run.configure(state=tk.NORMAL, text="🚀  BẮT ĐẦU KIỂM TRA & BÓC TÁCH HỒ SƠ (1-CLICK)")
        
        if err:
            self.status_var.set(f"Đã dừng do lỗi: {err}")
            messagebox.showerror("Thông báo lỗi", f"Đã xảy ra lỗi trong quá trình quét:\n{err}")
            return
            
        if not results:
            self.status_var.set("Không có dữ liệu kết quả.")
            return
            
        file_reports = results['file_reports']
        segmented = results.get('segmented_docs', [])
        total_files = len(file_reports)
        error_files = sum(1 for r in file_reports if r['ket_luan'] == "Lỗi")
        valid_files = total_files - error_files
        
        total_en = sum(r['en_docs'] for r in file_reports)
        total_vn = sum(r['vn_docs'] for r in file_reports)
        total_docs = len(segmented)
        
        mat_total = len(results.get('mat_records', []))
        soc_total = len(results.get('soc_records', []))
        xoay_records = results.get('xoay_records', [])
        xoay_total = sum(1 for x in xoay_records if x['status'] == "ROTATED")
        xoay_review_total = sum(1 for x in xoay_records if x['status'] == "REVIEW")

        self.kpi_total_files.set(str(total_files))
        self.kpi_total_docs.set(str(total_docs))
        self.kpi_en_docs.set(f"{total_en} VB")
        self.kpi_vn_docs.set(f"{total_vn} VB")
        self.kpi_mat_pages.set(f"{mat_total} trang")
        self.kpi_soc_pages.set(f"{soc_total} trang")
        self.kpi_xoay_pages.set(f"{xoay_total} trang" + (f" (+{xoay_review_total} KT)" if xoay_review_total else ""))

        review_files = sum(1 for r in file_reports if r['ket_luan'] == "Cần kiểm tra")
        overall = "LỖI" if error_files > 0 else ("CẦN KIỂM TRA" if review_files > 0 else "KHÔNG LỖI")
        self.kpi_overall_status.set(overall)

        # Danh sách trang cần xoay theo từng file
        xoay_lines = []
        for r in file_reports:
            pages = [f"{x['page']} ({x['angle']}°)" for x in xoay_records if x['file'] == r['file_name'] and x['status'] == "ROTATED"]
            review = [str(x['page']) for x in xoay_records if x['file'] == r['file_name'] and x['status'] == "REVIEW"]
            if pages or review:
                line = f"{r['file_name']}: "
                if pages:
                    line += "cần xoay trang " + ", ".join(pages)
                if review:
                    line += ("; " if pages else "") + "cần kiểm tra hướng trang " + ", ".join(review)
                xoay_lines.append(line)
            
        self.progress_val.set(100.0)
        self.status_var.set(f"Hoàn tất kiểm tra {total_files} file. Bóc tách {total_docs} văn bản ({total_en} Anh, {total_vn} Việt).")
        
        self._log("\n" + "=" * 65)
        self._log(f"[✓] HOÀN TẤT: {error_files} file lỗi, {review_files} file cần kiểm tra, {total_files - error_files - review_files} file không lỗi.")
        self._log(f"[✓] Bóc tách: {total_docs} văn bản ({total_en} Tiếng Anh, {total_vn} Tiếng Việt).")
        self._log(f"[✓] Trang bị xoay: {xoay_total} trang (đã xuất PDF sửa vào pdf_da_xoay/), cần kiểm tra thủ công: {xoay_review_total} trang.")
        for line in xoay_lines:
            self._log(f"      - {line}")
        self._log(f"[✓] File Excel 6 Sheet: {self.last_excel_path}")
        self._log(f"[✓] Kết luận chung: {overall}")
        self._log("=" * 65)
        
        messagebox.showinfo(
            "Kiểm Tra Hoàn Tất",
            f"Đã hoàn thành kiểm tra và bóc tách {total_files} file PDF!\n\n"
            f"- Tổng số văn bản bóc tách: {total_docs}\n"
            f"  + Văn bản Tiếng Anh: {total_en}\n"
            f"  + Văn bản Tiếng Việt: {total_vn}\n\n"
            f"- File lỗi: {error_files}  |  Cần kiểm tra: {review_files}  |  Không lỗi: {total_files - error_files - review_files}\n"
            f"- Trang bị xoay: {xoay_total} (cần kiểm tra thủ công: {xoay_review_total})\n"
            + "".join(f"    • {line}\n" for line in xoay_lines[:12])
            + (f"    • ... và {len(xoay_lines) - 12} file khác (xem Excel, sheet 'Trang Bị Xoay')\n" if len(xoay_lines) > 12 else "")
            + f"\n>>> KẾT LUẬN TOÀN BỘ HỒ SƠ: {overall}\n\n"
            f"Báo cáo chi tiết 6 Sheet đã xuất tại:\n{self.last_excel_path}"
        )


def main():
    app = ModernBiLingualGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
