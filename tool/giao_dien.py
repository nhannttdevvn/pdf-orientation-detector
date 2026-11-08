#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GIAO DIỆN MỚI (PySide6 / Qt) — chạy trên Windows, macOS, Linux.
- Kéo thả file PDF / thư mục, hoặc chọn từng file / cả thư mục.
- Chạy kiểm tra trong luồng nền, xem tiến độ và nhật ký trực tiếp.
- Xem kết quả theo tab: Tổng quan, Văn bản, Mục lục (kèm ảnh minh chứng), Trang xoay, Dấu mật, Sọc scan.
Giao diện Tkinter cũ (gui.py) vẫn giữ nguyên để dự phòng.
================================================================================
"""

import glob
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

from app_paths import open_path, user_dir

APP_TITLE = "Kiểm Tra Hồ Sơ Lưu Trữ"
USER_DIR = user_dir()

# Bảng màu (dùng chung cho QSS và tô màu ô trạng thái)
C_BG = "#F4F6F8"
C_CARD = "#FFFFFF"
C_BORDER = "#DDE3EA"
C_TEXT = "#1B2733"
C_MUTED = "#5E6E7E"
C_ACCENT = "#1F6FB2"
C_ACCENT_DARK = "#185A91"
STATUS_COLORS = {
    "error": ("#FDE7E7", "#A61B1B"),
    "review": ("#FFF4D6", "#8A5A00"),
    "ok": ("#E3F4EA", "#1E6B3A"),
    "info": ("#E8F0FA", "#1F4F80"),
}

QSS = f"""
QMainWindow, QWidget#root {{ background: {C_BG}; }}
QWidget {{ color: {C_TEXT}; font-size: 13px; }}
QFrame#card {{ background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 10px; }}
QLabel#title {{ font-size: 20px; font-weight: 700; }}
QLabel#subtitle {{ color: {C_MUTED}; }}
QLabel#section {{ font-size: 12px; font-weight: 700; color: {C_MUTED}; letter-spacing: 0.5px; }}
QLabel#kpiValue {{ font-size: 20px; font-weight: 700; }}
QLabel#kpiLabel {{ color: {C_MUTED}; font-size: 11px; font-weight: 600; }}
QLabel#drop {{
    border: 2px dashed #A9B8C7; border-radius: 10px; background: #F8FAFC;
    color: {C_MUTED}; padding: 14px; font-size: 13px;
}}
QLabel#drop[active="true"] {{ border-color: {C_ACCENT}; background: #EAF2FB; color: {C_ACCENT}; }}
QPushButton {{
    background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 7px; padding: 6px 12px;
}}
QPushButton:hover {{ border-color: #A9B8C7; background: #F8FAFC; }}
QPushButton:disabled {{ color: #9AA7B4; }}
QPushButton#primary {{
    background: {C_ACCENT}; color: white; border: none; font-size: 14px; font-weight: 700; padding: 11px;
}}
QPushButton#primary:hover {{ background: {C_ACCENT_DARK}; }}
QPushButton#primary:disabled {{ background: #8FB3D6; color: #EEF4FA; }}
QListWidget, QTableWidget, QPlainTextEdit, QLineEdit {{
    background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 7px;
}}
QLineEdit {{ padding: 5px 8px; }}
QTableWidget {{ gridline-color: #EDF1F5; alternate-background-color: #FAFBFD; selection-background-color: #D6E6F7;
    selection-color: {C_TEXT}; }}
QHeaderView::section {{ background: #EEF2F6; border: none; border-bottom: 1px solid {C_BORDER};
    padding: 6px; font-weight: 700; color: #3A4A5A; }}
QTabWidget::pane {{ border: 1px solid {C_BORDER}; border-radius: 8px; background: {C_CARD}; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 8px 14px; color: {C_MUTED}; font-weight: 600; }}
QTabBar::tab:selected {{ color: {C_ACCENT}; border-bottom: 2px solid {C_ACCENT}; }}
QProgressBar {{ border: none; background: #E4E9EF; border-radius: 4px; height: 8px; }}
QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 4px; }}
QPlainTextEdit#log {{ background: #0F1A24; color: #D7E1EA; font-family: Menlo, Consolas, monospace; font-size: 12px; }}
"""


def status_kind(text):
    t = str(text or "")
    if t.startswith("Lỗi") or t.startswith("LỖI"):
        return "error"
    if t.startswith("Cần kiểm tra") or t.startswith("Cảnh báo") or t.startswith("REVIEW"):
        return "review"
    if t in ("Không lỗi", "Khớp") or t.startswith("Hợp lệ"):
        return "ok"
    return None


def pdf_page_count(path):
    try:
        import pymupdf
        with pymupdf.open(path) as d:
            return d.page_count
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Luồng nền chạy kiểm tra
# -----------------------------------------------------------------------------
class Worker(QObject):
    progress = Signal(float, str)
    log = Signal(str)
    finished = Signal(object, object, int, object)   # results, excel_path, exit_code, checker
    failed = Signal(str)

    def __init__(self, files, out_dir, fix_rotation, check_toc):
        super().__init__()
        self.files, self.out_dir = files, out_dir
        self.fix_rotation, self.check_toc = fix_rotation, check_toc

    def run(self):
        try:
            from kiem_tra_ho_so import BiLingualDocumentChecker
            checker = BiLingualDocumentChecker(output_dir=self.out_dir, pdf_files=self.files,
                                               fix_rotation=self.fix_rotation, check_toc=self.check_toc)
            code, results, excel = checker.run(
                progress_callback=lambda p, t: self.progress.emit(float(p), str(t)),
                log_callback=lambda m: self.log.emit(str(m)),
            )
            self.finished.emit(results, excel, int(code), checker)
        except Exception as exc:
            import traceback
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(exc))


# -----------------------------------------------------------------------------
# Thành phần giao diện
# -----------------------------------------------------------------------------
def card(parent=None):
    f = QFrame(parent)
    f.setObjectName("card")
    return f


def section_label(text):
    lab = QLabel(text.upper())
    lab.setObjectName("section")
    return lab


class DropLabel(QLabel):
    """Vùng kéo thả file PDF / thư mục."""
    dropped = Signal(list)

    def __init__(self):
        super().__init__("Kéo thả file PDF hoặc thư mục vào đây")
        self.setObjectName("drop")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(70)

    def _set_active(self, on):
        self.setProperty("active", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._set_active(True)

    def dragLeaveEvent(self, e):
        self._set_active(False)

    def dropEvent(self, e):
        self._set_active(False)
        self.dropped.emit([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])


class KpiCard(QFrame):
    def __init__(self, label, accent):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        bar = QFrame()
        bar.setFixedHeight(3)
        bar.setStyleSheet(f"background:{accent}; border-radius:1px;")
        self.value = QLabel("–")
        self.value.setObjectName("kpiValue")
        cap = QLabel(label)
        cap.setObjectName("kpiLabel")
        lay.addWidget(bar)
        lay.addWidget(self.value)
        lay.addWidget(cap)

    def set(self, text, color=None):
        self.value.setText(str(text))
        self.value.setStyleSheet(f"color:{color};" if color else "")


class ImagePreview(QFrame):
    """Ô xem ảnh minh chứng (tự co theo khung)."""

    def __init__(self, caption):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        self.caption = QLabel(caption)
        self.caption.setObjectName("kpiLabel")
        self.img = QLabel("—")
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setMinimumHeight(70)
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img.setStyleSheet(f"color:{C_MUTED};")
        lay.addWidget(self.caption)
        lay.addWidget(self.img, 1)
        self._pix = None

    def show_file(self, path):
        if path and os.path.isfile(path):
            self._pix = QPixmap(path)
            self._fit()
        else:
            self._pix = None
            self.img.setPixmap(QPixmap())
            self.img.setText("Không có ảnh")

    def _fit(self):
        if self._pix and not self._pix.isNull():
            self.img.setPixmap(self._pix.scaled(self.img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()


def make_table(headers, stretch_col=None):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setAlternatingRowColors(True)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.verticalHeader().setVisible(False)
    t.setWordWrap(False)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeToContents)
    if stretch_col is not None:
        hh.setSectionResizeMode(stretch_col, QHeaderView.Stretch)
    return t


def fill_table(table, rows, status_cols=()):
    table.setSortingEnabled(False)
    table.setRowCount(len(rows))
    for r, values in enumerate(rows):
        for c, v in enumerate(values):
            text = "" if v is None else str(v)
            item = QTableWidgetItem(text)
            if len(text) > 28:
                item.setToolTip(text)
            if c in status_cols:
                kind = status_kind(v)
                if kind:
                    bg, fg = STATUS_COLORS[kind]
                    item.setBackground(QColor(bg))
                    item.setForeground(QColor(fg))
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                item.setTextAlignment(Qt.AlignCenter)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()
    # Cột chữ dài: giới hạn độ rộng (xem đầy đủ khi rê chuột), cột co giãn vẫn chiếm phần còn lại
    hh = table.horizontalHeader()
    for c in range(table.columnCount()):
        if hh.sectionResizeMode(c) != QHeaderView.Stretch and table.columnWidth(c) > 240:
            hh.setSectionResizeMode(c, QHeaderView.Interactive)
            table.setColumnWidth(c, 240)
    table.resizeRowsToContents()


# -----------------------------------------------------------------------------
# Cửa sổ chính
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1320, 840)
        self.setMinimumSize(1080, 700)
        self.files = []
        self.thread = None
        self.results = None
        self.checker = None
        self.excel_path = None
        self.toc_records = []
        self.xoay_records = []

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(12)

        # Tiêu đề
        head = QHBoxLayout()
        tbox = QVBoxLayout()
        tbox.setSpacing(0)
        title = QLabel(APP_TITLE)
        title.setObjectName("title")
        sub = QLabel("Bóc tách văn bản  ·  Đối chiếu mục lục  ·  Trang bị xoay  ·  Dấu mật  ·  Sọc scan")
        sub.setObjectName("subtitle")
        tbox.addWidget(title)
        tbox.addWidget(sub)
        head.addLayout(tbox)
        head.addStretch(1)
        outer.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(14)
        outer.addLayout(body, 1)
        body.addWidget(self._build_left(), 0)
        body.addWidget(self._build_right(), 1)

        default_in = os.path.join(USER_DIR, "data", "pdf")
        if os.path.isdir(default_in):
            self.add_paths([default_in])

    # ---------------- Cột trái: đầu vào & chạy ----------------
    def _build_left(self):
        panel = QWidget()
        panel.setFixedWidth(350)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        c1 = card()
        l1 = QVBoxLayout(c1)
        l1.setContentsMargins(14, 12, 14, 12)
        l1.setSpacing(8)
        l1.addWidget(section_label("Tài liệu đầu vào"))
        self.drop = DropLabel()
        self.drop.dropped.connect(self.add_paths)
        l1.addWidget(self.drop)
        row = QHBoxLayout()
        b_files = QPushButton("+ Thêm file")
        b_files.clicked.connect(self.pick_files)
        b_dir = QPushButton("+ Thêm thư mục")
        b_dir.clicked.connect(self.pick_dir)
        self.b_remove = QPushButton("Bỏ chọn")
        self.b_remove.clicked.connect(self.remove_selected)
        b_clear = QPushButton("Xóa hết")
        b_clear.clicked.connect(self.clear_files)
        for b in (b_files, b_dir):
            row.addWidget(b)
        l1.addLayout(row)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setMinimumHeight(150)
        l1.addWidget(self.list, 1)
        row2 = QHBoxLayout()
        self.files_info = QLabel("Chưa có file nào")
        self.files_info.setObjectName("subtitle")
        row2.addWidget(self.files_info, 1)
        row2.addWidget(self.b_remove)
        row2.addWidget(b_clear)
        l1.addLayout(row2)
        lay.addWidget(c1, 1)

        c2 = card()
        l2 = QVBoxLayout(c2)
        l2.setContentsMargins(14, 12, 14, 12)
        l2.setSpacing(8)
        l2.addWidget(section_label("Thư mục kết quả"))
        orow = QHBoxLayout()
        self.out_edit = QLineEdit(os.path.join(USER_DIR, "reports"))
        b_out = QPushButton("Chọn…")
        b_out.clicked.connect(self.pick_out)
        orow.addWidget(self.out_edit, 1)
        orow.addWidget(b_out)
        l2.addLayout(orow)
        l2.addWidget(section_label("Tùy chọn"))
        self.chk_fix = QCheckBox("Xuất bản PDF đã sửa trang bị xoay")
        self.chk_fix.setChecked(True)
        self.chk_toc = QCheckBox("Đối chiếu Mục lục với văn bản gốc")
        self.chk_toc.setChecked(True)
        l2.addWidget(self.chk_fix)
        l2.addWidget(self.chk_toc)
        lay.addWidget(c2)

        c3 = card()
        l3 = QVBoxLayout(c3)
        l3.setContentsMargins(14, 12, 14, 12)
        l3.setSpacing(8)
        self.b_run = QPushButton("Bắt đầu kiểm tra")
        self.b_run.setObjectName("primary")
        self.b_run.setCursor(Qt.PointingHandCursor)
        self.b_run.clicked.connect(self.start)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1000)
        self.status = QLabel("Sẵn sàng.")
        self.status.setObjectName("subtitle")
        self.status.setWordWrap(True)
        l3.addWidget(self.b_run)
        l3.addWidget(self.progress)
        l3.addWidget(self.status)
        lay.addWidget(c3)
        return panel

    # ---------------- Cột phải: kết quả ----------------
    def _build_right(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.kpi = {
            "files": KpiCard("FILE PDF", "#1F6FB2"),
            "docs": KpiCard("VĂN BẢN BÓC TÁCH", "#5B6CC2"),
            "toc": KpiCard("MỤC LỤC (KHỚP/LỖI/KT)", "#0F766E"),
            "xoay": KpiCard("TRANG BỊ XOAY", "#7C3AED"),
            "soc": KpiCard("TRANG CÓ SỌC", "#475569"),
            "mat": KpiCard("TRANG CÓ DẤU MẬT", "#B45309"),
            "kl": KpiCard("KẾT LUẬN CHUNG", "#1B2733"),
        }
        for i, key in enumerate(["files", "docs", "toc", "xoay", "soc", "mat", "kl"]):
            grid.addWidget(self.kpi[key], 0, i)
            grid.setColumnStretch(i, 2 if key == "toc" else 1)
        lay.addLayout(grid)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        self.t_files = make_table(["STT", "File PDF", "Kết luận", "Trang", "Văn bản", "Hợp lệ", "Cần KT", "Mục lục",
                                   "Trang xoay", "Sọc", "Dấu mật", "Chi tiết"], stretch_col=11)
        self.tabs.addTab(self.t_files, "Tổng quan")

        self.t_docs = make_table(["#", "File PDF", "VB", "Ngôn ngữ", "Loại văn bản", "Từ trang", "Đến trang", "Số trang",
                                  "Trạng thái", "Trích yếu"], stretch_col=9)
        self.tabs.addTab(self.t_docs, "Văn bản")

        toc_box = QWidget()
        tl = QVBoxLayout(toc_box)
        tl.setContentsMargins(8, 8, 8, 8)
        self.toc_warn = QLabel("")
        self.toc_warn.setWordWrap(True)
        self.toc_warn.setVisible(False)
        tl.addWidget(self.toc_warn)
        split = QSplitter(Qt.Vertical)
        self.t_toc = make_table(["Hồ sơ", "STT", "File", "Trang", "Tờ", "Số KH (Mục lục)", "Số KH (Gốc)",
                                 "Ngày (Mục lục)", "Ngày (Gốc)", "Đánh giá", "Chi tiết"], stretch_col=10)
        self.t_toc.itemSelectionChanged.connect(self.on_toc_select)
        split.addWidget(self.t_toc)
        prev = QWidget()
        pl = QHBoxLayout(prev)
        pl.setContentsMargins(0, 6, 0, 0)
        self.prev_sokh = ImagePreview("SỐ KÝ HIỆU TRÊN VĂN BẢN GỐC")
        self.prev_ngay = ImagePreview("NGÀY THÁNG TRÊN VĂN BẢN GỐC")
        self.prev_trang = ImagePreview("SỐ TỜ (GÓC TRÊN PHẢI)")
        pl.addWidget(self.prev_sokh, 3)
        pl.addWidget(self.prev_ngay, 4)
        pl.addWidget(self.prev_trang, 1)
        split.addWidget(prev)
        split.setSizes([420, 160])
        tl.addWidget(split, 1)
        self.toc_hint = QLabel("Chọn một dòng để xem ảnh minh chứng cắt từ văn bản gốc.")
        self.toc_hint.setObjectName("subtitle")
        tl.addWidget(self.toc_hint)
        self.tabs.addTab(toc_box, "Mục lục")

        xo_box = QWidget()
        xl = QHBoxLayout(xo_box)
        xl.setContentsMargins(8, 8, 8, 8)
        self.t_xoay = make_table(["File PDF", "Trang", "Trạng thái", "Góc cần xoay", "Độ tin cậy", "Phương pháp",
                                  "Đã sửa", "Ghi chú"], stretch_col=7)
        self.t_xoay.itemSelectionChanged.connect(self.on_xoay_select)
        self.prev_xoay = ImagePreview("ẢNH TRANG GỐC")
        self.prev_xoay.setMinimumWidth(260)
        xl.addWidget(self.t_xoay, 3)
        xl.addWidget(self.prev_xoay, 1)
        self.tabs.addTab(xo_box, "Trang xoay")

        self.t_mat = make_table(["File PDF", "Trang", "Cấp độ mật", "Độ tin cậy"], stretch_col=0)
        self.tabs.addTab(self.t_mat, "Dấu mật")
        self.t_soc = make_table(["File PDF", "Trang", "Số đường sọc"], stretch_col=0)
        self.tabs.addTab(self.t_soc, "Sọc scan")

        self.log = QPlainTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "Nhật ký")

        actions = QHBoxLayout()
        self.b_excel = QPushButton("Mở báo cáo Excel")
        self.b_excel.clicked.connect(lambda: self.open_target(self.excel_path, "báo cáo Excel"))
        self.b_outdir = QPushButton("Thư mục kết quả")
        self.b_outdir.clicked.connect(lambda: self.open_target(self.out_edit.text().strip(), "thư mục kết quả"))
        self.b_pdfx = QPushButton("PDF đã sửa xoay")
        self.b_pdfx.clicked.connect(lambda: self.open_target(os.path.join(self.out_edit.text().strip(), "pdf_da_xoay"), "thư mục PDF đã sửa"))
        self.b_evid = QPushButton("Ảnh minh chứng")
        self.b_evid.clicked.connect(lambda: self.open_target(os.path.join(self.out_edit.text().strip(), "evidence_crops"), "thư mục ảnh minh chứng"))
        for b in (self.b_excel, self.b_outdir, self.b_pdfx, self.b_evid):
            actions.addWidget(b)
        actions.addStretch(1)
        lay.addLayout(actions)
        return panel

    # ---------------- Quản lý danh sách file ----------------
    def add_paths(self, paths):
        added = 0
        for p in paths:
            if os.path.isdir(p):
                cands = glob.glob(os.path.join(p, "**", "*.pdf"), recursive=True)
            else:
                cands = [p]
            for f in cands:
                f = os.path.abspath(f)
                if not f.lower().endswith(".pdf") or os.path.basename(f).startswith(("._", "~$")) or f in self.files:
                    continue
                self.files.append(f)
                n = pdf_page_count(f)
                item = QListWidgetItem(f"{os.path.basename(f)}" + (f"   ·  {n} trang" if n else ""))
                item.setToolTip(f)
                item.setData(Qt.UserRole, f)
                self.list.addItem(item)
                added += 1
        self._update_files_info()
        return added

    def _update_files_info(self):
        if not self.files:
            self.files_info.setText("Chưa có file nào")
        else:
            self.files_info.setText(f"{len(self.files)} file PDF")

    def pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn file PDF", USER_DIR, "PDF (*.pdf)")
        if files:
            self.add_paths(files)

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa PDF", USER_DIR)
        if d:
            if not self.add_paths([d]):
                QMessageBox.information(self, APP_TITLE, "Thư mục không có file PDF nào mới.")

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    def remove_selected(self):
        for item in self.list.selectedItems():
            self.files.remove(item.data(Qt.UserRole))
            self.list.takeItem(self.list.row(item))
        self._update_files_info()

    def clear_files(self):
        self.files.clear()
        self.list.clear()
        self._update_files_info()

    # ---------------- Chạy kiểm tra ----------------
    def start(self):
        if self.thread is not None:
            return
        if not self.files:
            QMessageBox.warning(self, APP_TITLE, "Hãy thêm ít nhất một file PDF (kéo thả, '+ Thêm file' hoặc '+ Thêm thư mục').")
            return
        out = self.out_edit.text().strip() or os.path.join(USER_DIR, "reports")
        os.makedirs(out, exist_ok=True)
        self.log.clear()
        self.progress.setValue(0)
        self.b_run.setEnabled(False)
        self.b_run.setText("Đang kiểm tra…")
        self.status.setText("Đang khởi tạo bộ máy nhận diện (lần đầu có thể mất vài giây)…")
        self.thread = QThread(self)
        self.worker = Worker(list(self.files), out, self.chk_fix.isChecked(), self.chk_toc.isChecked())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.thread.start()

    def _stop_thread(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.b_run.setEnabled(True)
        self.b_run.setText("Bắt đầu kiểm tra")

    def on_progress(self, pct, text):
        self.progress.setValue(int(pct * 10))
        self.status.setText(text)

    def on_failed(self, msg):
        self._stop_thread()
        self.status.setText(f"Đã dừng do lỗi: {msg}")
        QMessageBox.critical(self, APP_TITLE, f"Có lỗi trong quá trình kiểm tra:\n{msg}\n\nXem tab Nhật ký để biết chi tiết.")

    def on_finished(self, results, excel, code, checker):
        self._stop_thread()
        self.checker = checker
        self.excel_path = excel
        if not results:
            self.status.setText("Không có kết quả (không tìm thấy file PDF hợp lệ).")
            return
        self.show_results(results)
        self.progress.setValue(1000)
        kl = {0: "Không lỗi", 1: "Lỗi", 2: "Cần kiểm tra"}.get(code, "")
        self.status.setText(f"Hoàn tất. Kết luận chung: {kl}. Báo cáo: {os.path.basename(excel or '')}")

    # ---------------- Hiển thị kết quả ----------------
    def show_results(self, results):
        self.results = results
        reports = results.get("file_reports", [])
        docs = results.get("segmented_docs", [])
        self.toc_records = results.get("toc_records", [])
        toc_warn = results.get("toc_warnings", [])
        self.xoay_records = results.get("xoay_records", [])
        mat = results.get("mat_records", [])
        soc = results.get("soc_records", [])

        fill_table(self.t_files, [[
            r["stt"], r["file_name"], r["ket_luan"], r["total_pages"], r["total_docs"], r["valid_docs"], r["error_docs"],
            r.get("toc_str", "-"), r.get("xoay_str", "-"), r["soc_str"], r["mat_str"], r["chi_tiet"],
        ] for r in reports], status_cols=(2,))
        fill_table(self.t_docs, [[
            i, d["file"], d["doc_id"], d["lang"], d["type"], d["start_page"], d["end_page"], d["page_count"],
            d["status"], d["summary"],
        ] for i, d in enumerate(docs, start=1)], status_cols=(8,))
        fill_table(self.t_toc, [[
            r["dossier"], r["stt"], r["file"], r["page"], r["sheet"], r["so_kh_ml"], r["so_kh_goc"] or "(không đọc được)",
            r["ngay_ml"], r["ngay_goc"] or "(không đọc được)", r["status"],
            "; ".join(txt for _, _, txt in r["issues"]),
        ] for r in self.toc_records], status_cols=(9,))
        if toc_warn:
            bg, fg = STATUS_COLORS["error"]
            self.toc_warn.setText("Lỗi bảng Mục lục:  " + ";  ".join(f"Hồ sơ {w['dossier']}: {w['detail']}" for w in toc_warn))
            self.toc_warn.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:6px; padding:8px; font-weight:600;")
            self.toc_warn.setVisible(True)
        else:
            self.toc_warn.setVisible(False)
        fill_table(self.t_xoay, [[
            x["file"], x["page"], "Bị xoay" if x["status"] == "ROTATED" else "Cần kiểm tra",
            f"{x['angle']}°" if x["status"] == "ROTATED" else "-",
            f"{x['conf']:.2f}" if x.get("conf") is not None else "-", x.get("method", "-"),
            "Có" if x.get("fixed") else "Không", x.get("reason") or "",
        ] for x in self.xoay_records], status_cols=())
        for r in range(self.t_xoay.rowCount()):
            item = self.t_xoay.item(r, 2)
            bg, fg = STATUS_COLORS["error" if item.text() == "Bị xoay" else "review"]
            item.setBackground(QColor(bg))
            item.setForeground(QColor(fg))
        fill_table(self.t_mat, [[m["file"], m["page"], m["type"], f"{m['conf']:.2f}"] for m in mat])
        fill_table(self.t_soc, [[s["file"], s["page"], s["count"]] for s in soc])

        # KPI
        from collections import Counter
        c = Counter(r["status"] for r in self.toc_records)
        n_rot = sum(1 for x in self.xoay_records if x["status"] == "ROTATED")
        n_rot_rv = sum(1 for x in self.xoay_records if x["status"] == "REVIEW")
        kls = [r["ket_luan"] for r in reports]
        overall = "Lỗi" if "Lỗi" in kls else ("Cần kiểm tra" if "Cần kiểm tra" in kls else "Không lỗi")
        self.kpi["files"].set(len(reports))
        self.kpi["docs"].set(len(docs))
        if self.toc_records or toc_warn:
            self.kpi["toc"].set(f"{c.get('Khớp', 0)} / {c.get('Lỗi', 0) + len(toc_warn)} / {c.get('Cần kiểm tra', 0)}")
        else:
            self.kpi["toc"].set("Không có")
        self.kpi["xoay"].set(f"{n_rot}" + (f"  (+{n_rot_rv} KT)" if n_rot_rv else ""))
        self.kpi["soc"].set(len(soc))
        self.kpi["mat"].set(len(mat))
        self.kpi["kl"].set(overall, STATUS_COLORS[status_kind(overall)][1])
        for i in range(self.tabs.count()):
            name = self.tabs.tabText(i).split("  (")[0]
            counts = {"Văn bản": len(docs), "Mục lục": len(self.toc_records), "Trang xoay": len(self.xoay_records),
                      "Dấu mật": len(mat), "Sọc scan": len(soc), "Tổng quan": len(reports)}
            if name in counts:
                self.tabs.setTabText(i, f"{name}  ({counts[name]})")
        if self.toc_records:
            self.t_toc.selectRow(next((i for i, r in enumerate(self.toc_records) if r["status"] != "Khớp"), 0))

    def _evidence_dir(self):
        return self.checker.evidence_dir if self.checker else os.path.join(self.out_edit.text().strip(), "evidence_crops")

    def on_toc_select(self):
        rows = self.t_toc.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self.toc_records):
            return
        rec = self.toc_records[rows[0].row()]
        ev = {name.rsplit("_", 1)[-1].replace(".png", ""): os.path.join(self._evidence_dir(), name) for name in rec["evidence"]}
        if not ev:
            self.toc_hint.setText("Văn bản này khớp Mục lục nên không lưu ảnh minh chứng.")
        else:
            self.toc_hint.setText(f"STT {rec['stt']} · {rec['file']} trang {rec['page']} (tờ {rec['sheet']}) · "
                                  f"Mục lục: {rec['so_kh_ml']}, ngày {rec['ngay_ml']}")
        self.prev_sokh.show_file(ev.get("kh"))
        self.prev_ngay.show_file(ev.get("ngay"))
        self.prev_trang.show_file(ev.get("trang"))

    def on_xoay_select(self):
        rows = self.t_xoay.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self.xoay_records):
            return
        x = self.xoay_records[rows[0].row()]
        xoay_dir = self.checker.xoay_dir if self.checker else os.path.join(self.out_edit.text().strip(), "trang_bi_xoay")
        stem = os.path.splitext(x["file"])[0]
        suffix = f"xoay{x['angle']}" if x["status"] == "ROTATED" else "can_kiem_tra"
        self.prev_xoay.show_file(os.path.join(xoay_dir, f"{stem}_p{x['page']}_{suffix}.png"))

    def open_target(self, path, name):
        if path and os.path.exists(path):
            try:
                open_path(path)
            except Exception as exc:
                QMessageBox.warning(self, APP_TITLE, f"Không mở được {name}: {exc}")
        else:
            QMessageBox.information(self, APP_TITLE, f"Chưa có {name}. Hãy chạy kiểm tra trước.")

    def closeEvent(self, e):
        if self.thread is not None:
            if QMessageBox.question(self, APP_TITLE, "Đang kiểm tra. Thoát sẽ dừng giữa chừng. Vẫn thoát?") != QMessageBox.Yes:
                e.ignore()
                return
            os._exit(0)
        e.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
