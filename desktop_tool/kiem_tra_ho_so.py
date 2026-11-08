#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HỆ THỐNG KIỂM TRA & BÓC TÁCH HỒ SƠ LƯU TRỮ VĂN BẢN ĐA NGÔN NGỮ (v4.0)
Tích hợp:
1. Phân loại & Xác định ranh giới Đầu - Cuối:
   - Văn bản Tiếng Anh: Fax, Thư thương mại, Chiến lược hợp đồng (PVEP / TLJOC).
   - Văn bản Tiếng Việt: Văn bản Doanh nghiệp (Tờ trình, Báo cáo, Phê duyệt TGĐ)
                         & Văn bản Đảng bộ (Tiêu ngữ ĐCSVN có gạch chân, Dấu đỏ Bí thư).
2. Tự động bóc tách từng văn bản độc lập (Document Segmentation) & Kiểm tra khép kín:
   - Phát hiện văn bản hoàn chỉnh [START -> END].
   - Phát hiện văn bản bị cắt giữa 2 file PDF hoặc mồ côi dấu kết thúc.
3. Kiểm tra chất lượng:
   - Phát hiện Tài liệu Mật (MẬT, TỐI MẬT, TUYỆT MẬT, GIẢI MẬT) với độ tin cậy >= 0.80.
   - Phát hiện Sọc Scan (Vertical Streaks) bằng Hough Lines Transform.
   - Phát hiện Trang bị xoay 90° / 180° / 270° bằng Tesseract OSD (module `xoay_trang.py`),
     tự xoay ảnh trang về đúng chiều trước khi nhận diện Đầu/Cuối, Dấu mật, Sọc scan,
     và xuất bản PDF đã sửa xoay riêng (không ghi đè file gốc, không tự xoay trang REVIEW).
4. Xuất Báo cáo Excel 5 Sheet chuyên nghiệp & Thư mục ảnh minh chứng.
================================================================================
"""

import os
import sys
import glob
import re
import argparse
import time
import cv2
import numpy as np
import pymupdf
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Cấu hình UTF-8 cho Windows Console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app_paths import patch_opencv_unicode_paths, resource_dir, user_dir

# Windows: cho cv2.imread / cv2.imwrite đọc/ghi được đường dẫn và tên file tiếng Việt (vd. ảnh '..._MẬT.png')
patch_opencv_unicode_paths()

# Thư mục người dùng nhìn thấy (data/pdf, reports). Khi đóng gói: thư mục chứa file chạy.
USER_DIR = user_dir()

# Import module phát hiện sọc
try:
    from hough_line_detector import detect_vertical_streaks
except ImportError:
    try:
        from xu_ly_soc.hough_line_detector import detect_vertical_streaks
    except ImportError:
        detect_vertical_streaks = None

# Import module phát hiện tài liệu mật
try:
    from detect_confidential_docs_fast import FastConfidentialDocDetector
except ImportError:
    FastConfidentialDocDetector = None

# Import module phát hiện trang bị xoay
try:
    import xoay_trang
except ImportError:
    xoay_trang = None

# Import module đối chiếu Mục lục với văn bản gốc
try:
    import doi_chieu_muc_luc
except ImportError:
    doi_chieu_muc_luc = None

# Kết luận 3 mức
KL_LOI = "Lỗi"
KL_CAN_KIEM_TRA = "Cần kiểm tra"
KL_KHONG_LOI = "Không lỗi"

# Từ đặc trưng (đã bỏ dấu) để nhận biết ngôn ngữ văn bản; bỏ các từ trùng nghĩa 2 ngôn ngữ (to, do, so, an, ten...)
_VI_WORDS = {
    "cua", "va", "cac", "nhung", "trong", "duoc", "theo", "voi", "ngay", "thang", "nam", "dang", "huyen",
    "uy", "bao", "cao", "nhan", "dan", "cong", "viet", "tong", "giam", "quyet", "dinh", "viec", "dong",
    "chi", "khong", "nhiem", "thuong", "truc", "xa", "tinh", "ve", "cho", "nay", "tai", "la", "mot",
    "nguoi", "hoi", "nghi", "ket", "luan", "chinh", "tri", "thuc", "hien", "can", "cu", "phong", "thon",
}
_EN_WORDS = {
    "the", "of", "and", "for", "with", "is", "on", "by", "this", "that", "be", "are", "from", "shall",
    "will", "as", "or", "which", "contract", "company", "page", "agreement", "party", "parties", "operator",
    "joint", "operating", "letter", "dear", "sincerely", "regards", "please", "subject", "reference",
    "draft", "schedule", "article", "section", "accordance", "provided", "work", "services", "price",
}
_VI_CHARS = set("ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")


def detect_language(text):
    """Nhận biết ngôn ngữ theo nội dung chữ: 'Tiếng Việt' / 'Tiếng Anh' / None (không đủ căn cứ)."""
    text = str(text or "")
    if sum(1 for ch in text.lower() if ch in _VI_CHARS) >= 5:
        return "Tiếng Việt"
    plain = doi_chieu_muc_luc.unaccent(text) if doi_chieu_muc_luc else text
    tokens = re.findall(r"[a-z]+", plain.lower())
    vi = sum(1 for t in tokens if t in _VI_WORDS)
    en = sum(1 for t in tokens if t in _EN_WORDS)
    if max(vi, en) < 3:
        return None
    if vi >= en * 1.5:
        return "Tiếng Việt"
    if en >= vi * 1.5:
        return "Tiếng Anh"
    return None

# Import RapidOCR
try:
    from rapidocr_onnxruntime import RapidOCR
    _GLOBAL_OCR = RapidOCR()
except Exception:
    _GLOBAL_OCR = None


def natural_sort_key(s):
    """Sắp xếp chuỗi chứa số theo thứ tự tự nhiên (0001 -> 0002 -> 0056_1...)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def get_template_dir():
    """Tự động xác định thư mục templates mẫu dấu mật."""
    candidates = [
        os.path.join(resource_dir(), "templates"),
        os.path.join(CURRENT_DIR, "templates"),
        r"C:\tool_tailieu\templates",
        r"C:\Tool\templates",
        r"C:\Users\Admin\Downloads\tool_phathienmat\tailieumat\templates"
    ]
    for c in candidates:
        if os.path.exists(c) and len(os.listdir(c)) > 0:
            return c
    return "templates"


class BiLingualDocumentChecker:
    def __init__(self, target_dir=None, output_dir=None, fix_rotation=True, pdf_files=None, check_toc=True):
        # Danh sách file PDF cụ thể (chọn từng file trên giao diện); nếu không có thì quét cả thư mục target_dir
        self.pdf_files = [os.path.abspath(f) for f in pdf_files] if pdf_files else None
        self.check_toc = check_toc
        if target_dir is None and self.pdf_files:
            target_dir = os.path.dirname(self.pdf_files[0])
        if target_dir is None:
            candidates = [
                os.path.join(USER_DIR, "data", "pdf"),
                os.path.join(os.path.dirname(USER_DIR), "tool_tailieu", "data", "pdf"),
                r"C:\tool_tailieu\data\pdf",
                r"C:\Tool\data\pdf",
                r"C:\Tool\tách sai\tách sai",
                USER_DIR
            ]
            for c in candidates:
                if os.path.exists(c) and len(glob.glob(os.path.join(c, "*.pdf"))) > 0:
                    target_dir = c
                    break
            if target_dir is None:
                target_dir = os.path.join(USER_DIR, "data", "pdf")

        if output_dir is None:
            output_dir = os.path.join(USER_DIR, "reports")
            
        self.target_dir = target_dir
        self.output_dir = output_dir
        
        # Thư mục xuất ảnh minh chứng
        self.evidence_dir = os.path.join(output_dir, "evidence_crops")
        self.mat_dir = os.path.join(output_dir, "tai_lieu_mat")
        self.soc_dir = os.path.join(output_dir, "anh_co_soc")
        self.xoay_dir = os.path.join(output_dir, "trang_bi_xoay")
        self.pdf_xoay_dir = os.path.join(output_dir, "pdf_da_xoay")

        os.makedirs(self.evidence_dir, exist_ok=True)
        os.makedirs(self.mat_dir, exist_ok=True)
        os.makedirs(self.soc_dir, exist_ok=True)
        os.makedirs(self.xoay_dir, exist_ok=True)
        os.makedirs(self.pdf_xoay_dir, exist_ok=True)
        
        self.excel_path = os.path.join(output_dir, "bao_cao_kiem_tra.xlsx")
        self.alt_excel_path = os.path.join(output_dir, "bao_cao_sai_lech_muc_luc.xlsx")
        
        # Khởi tạo detector tài liệu mật với ngưỡng nghiêm ngặt >= 0.80
        tpl_dir = get_template_dir()
        if FastConfidentialDocDetector is not None:
            self.mat_detector = FastConfidentialDocDetector(template_dir=tpl_dir, threshold=0.80, dpi=200)
        else:
            self.mat_detector = None

        # Khởi tạo detector trang bị xoay (Tesseract OSD, ngưỡng độ tin cậy 10.0)
        self.fix_rotation = fix_rotation
        if xoay_trang is not None:
            self.orientation_detector = xoay_trang.OrientationDetector(
                conf_threshold=xoay_trang.ORIENTATION_CONF_THRESHOLD
            )
        else:
            self.orientation_detector = None

        self.ocr = _GLOBAL_OCR

    def find_pdf_files(self):
        """Tìm toàn bộ các file PDF trong thư mục được chỉ định (hoặc dùng danh sách file đã chọn)."""
        if self.pdf_files:
            files = [f for f in self.pdf_files if f.lower().endswith(".pdf") and os.path.isfile(f)]
            return sorted(dict.fromkeys(files), key=natural_sort_key)
        patterns = [
            os.path.join(self.target_dir, "*.pdf"),
            os.path.join(self.target_dir, "**", "*.pdf")
        ]
        all_found = []
        for pat in patterns:
            all_found.extend(glob.glob(pat, recursive=True))
            
        all_found = [f for f in all_found if not os.path.basename(f).startswith("~$")]
        seen = set()
        unique_files = []
        for f in all_found:
            fname = os.path.basename(f)
            if fname not in seen:
                seen.add(fname)
                unique_files.append(f)
                
        unique_files.sort(key=natural_sort_key)
        return unique_files

    def is_toc_page(self, img):
        """Kiểm tra trang có phải bảng Mục lục hay không (lưới dày đặc)."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.05), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.02)))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        grid = cv2.add(h_lines, v_lines)
        return np.count_nonzero(grid) > 20000

    def detect_red_stamp(self, img):
        """Nhận diện con dấu tròn màu đỏ (HSV Mask kép)."""
        h, w = img.shape[:2]
        sub = img[int(h * 0.15):, int(w * 0.35):]
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, np.array([0, 40, 40], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
        mask2 = cv2.inRange(hsv, np.array([165, 40, 40], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_stamp = False
        best_box = None
        max_area = 0
        
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 1000:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh) if bh > 0 else 0
                if 0.5 < aspect < 2.0:
                    has_stamp = True
                    if area > max_area:
                        max_area = area
                        best_box = (bx + int(w * 0.35), by + int(h * 0.15), bw, bh)
                        
        stamp_crop = None
        if has_stamp and best_box is not None:
            bx, by, bw, bh = best_box
            pad = 20
            y1 = max(0, by - pad)
            y2 = min(h, by + bh + pad)
            x1 = max(0, bx - pad)
            x2 = min(w, bx + bw + pad)
            stamp_crop = img[y1:y2, x1:x2]
            
        return has_stamp, {'area': max_area, 'box': best_box, 'crop': stamp_crop}

    def detect_party_header(self, img):
        """Nhận diện Header văn bản Đảng: Đường kẻ liền nét bên phải + Tiêu ngữ ĐCSVN."""
        h, w = img.shape[:2]
        top_h = int(h * 0.22)
        top_roi = img[:top_h, :]
        gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        right_roi = thresh[int(h * 0.04):int(h * 0.18), int(w * 0.40):int(w * 0.95)]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        h_lines = cv2.morphologyEx(right_roi, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_line = any(cv2.boundingRect(c)[2] >= int(w * 0.08) and cv2.boundingRect(c)[3] <= 35 and (cv2.boundingRect(c)[2] / max(1, cv2.boundingRect(c)[3])) >= 4.0 for c in cnts)
        if not has_line:
            return False, None
            
        left_box = thresh[int(h * 0.04):int(h * 0.14), int(w * 0.08):int(w * 0.45)]
        right_box = thresh[int(h * 0.04):int(h * 0.12), int(w * 0.50):int(w * 0.92)]
        
        is_title = (np.mean(left_box > 0) > 0.015 and np.mean(right_box > 0) > 0.015)
        crop = top_roi[0:int(top_h * 0.9), :] if is_title else None
        return is_title, crop

    def extract_header_footer_text(self, page, img, use_text_layer=True):
        """
        Trích xuất văn bản vùng Header (top 28%) và Footer (bottom 32%).
        Ưu tiên text layer có sẵn trong PDF, nếu không có thì chạy RapidOCR siêu tốc.
        Trang bị xoay (use_text_layer=False): vùng Header/Footer của text layer nằm theo hướng cũ (sai),
        nên đọc bằng OCR trên ảnh đã xoay về đúng chiều.
        """
        h, w = img.shape[:2]
        t_text, b_text = "", ""
        if use_text_layer:
            rect = page.rect
            top_rect = pymupdf.Rect(0, 0, rect.width, rect.height * 0.32)
            bot_rect = pymupdf.Rect(0, rect.height * 0.65, rect.width, rect.height)
            t_text = page.get_text("text", clip=top_rect).strip()
            b_text = page.get_text("text", clip=bot_rect).strip()
        
        # Nếu trang scan không có text layer thì dùng RapidOCR
        if (len(t_text) < 10 or len(b_text) < 10) and self.ocr is not None:
            top_roi = img[:int(h * 0.30), :]
            bot_roi = img[int(h * 0.68):, :]
            
            res_t, _ = self.ocr(top_roi)
            res_b, _ = self.ocr(bot_roi)
            
            if res_t:
                t_text = " ".join([r[1] for r in res_t])
            if res_b:
                b_text = " ".join([r[1] for r in res_b])
                
        return t_text, b_text

    def classify_page_boundaries(self, page, img, pno, total_pages, use_text_layer=True):
        """
        Phân loại đặc trưng Đầu & Cuối của trang:
        - Tiếng Anh: EN_START, EN_END
        - Tiếng Việt: VN_START, VN_END
        """
        h, w = img.shape[:2]
        has_red_stamp, s_info = self.detect_red_stamp(img)
        has_party_header, party_crop = self.detect_party_header(img)

        t_text, b_text = self.extract_header_footer_text(page, img, use_text_layer=use_text_layer)
        t_clean = re.sub(r'[\s\-_.:/]+', '', t_text.lower())
        b_clean = re.sub(r'[\s\-_.:/]+', '', b_text.lower())
        t_raw = t_text.lower()
        b_raw = b_text.lower()
        
        # -------------------------------------------------------------
        # 1. NHẬN DIỆN VĂN BẢN TIẾNG ANH (ENGLISH)
        # -------------------------------------------------------------
        # Dấu hiệu bắt đầu Tiếng Anh
        en_start_kw = [
            'petrovietnam', 'thanglong', 'jointoperating', 'tljoc',
            'facsimiletransmission', 'faxtransmission', 'contractstrategy',
            'recommendationforaward', 'invitationtobid', 'scopeofwork',
            'letteroftransmittal', 'requestforproposal', 'page1of'
        ]
        is_en_start = any(k in t_clean for k in en_start_kw)
        
        # Dấu hiệu kết thúc Tiếng Anh
        en_end_kw = [
            'yourssincerely', 'yourstruly', 'sincerelyyours', 'bestregards',
            'thankyouforyourcooperation', 'generalmanager', 'deputygeneralmanager',
            'mcmember', 'presidentceo', 'vicepresident'
        ]
        is_en_end = any(k in b_clean for k in en_end_kw)
        if not is_en_end:
            # Kiểm tra trang cuối hợp đồng: page N of N (ví dụ Page 7 of 7)
            m = re.search(r'page\s*(\d+)\s*of\s*(\d+)', b_raw)
            if m and m.group(1) == m.group(2):
                is_en_end = True
                
        # -------------------------------------------------------------
        # 2. NHẬN DIỆN VĂN BẢN TIẾNG VIỆT (VIETNAMESE)
        # -------------------------------------------------------------
        # Dấu hiệu bắt đầu Tiếng Việt
        vn_corp_start_kw = [
            'conghoaxahoichunghia', 'doclaptudohanhphuc', 'tongcongtythamdo',
            'banqlda', 'totrinh', 'baocaoketqua', 'kehoachluachon', 'hequanhhedieuhanh'
        ]
        is_vn_corp_start = any(k in t_clean for k in vn_corp_start_kw)
        is_vn_party_start = has_party_header or ('dangcongsan' in t_clean or 'dangbo' in t_clean or 'danguy' in t_clean)
        is_vn_start = is_vn_party_start or is_vn_corp_start
        
        # Dấu hiệu kết thúc Tiếng Việt
        vn_corp_end_kw = [
            'pheduyetcua', 'tonggiamdoc', 'truongdaidien', 'phocongty', 'giamdoc'
        ]
        is_vn_corp_end = any(k in b_clean for k in vn_corp_end_kw)
        is_vn_party_end = has_red_stamp and any(k in b_clean for k in ['banthuongvu', 'bithu', 'danguy'])
        is_vn_end = is_vn_party_end or is_vn_corp_end or (has_red_stamp and not is_en_end)
        
        # Cắt ảnh trích yếu nếu là trang Start/End
        crop_start = party_crop if party_crop is not None else img[:int(h * 0.25), :]
        crop_end = s_info['crop'] if s_info.get('crop') is not None else img[int(h * 0.70):, :]
        
        return {
            'is_en_start': is_en_start,
            'is_en_end': is_en_end,
            'is_vn_start': is_vn_start,
            'is_vn_end': is_vn_end,
            'has_red_stamp': has_red_stamp,
            'crop_start': crop_start,
            'crop_end': crop_end,
            'top_text': t_text[:120].strip(),
            'bot_text': b_text[-120:].strip(),
            'lang_hint': detect_language(t_text + " " + b_text)
        }

    def render_rotated_header_roi(self, page, rotate_angle):
        """
        Render vùng Header cho trang bị xoay: render cả trang ở DPI của detector mật,
        xoay về đúng chiều rồi cắt cùng vùng với `render_header_roi` (60% ngang, 35% dọc).
        """
        try:
            dpi = self.mat_detector.dpi if self.mat_detector is not None else 200
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
            gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w).copy()
            gray = xoay_trang.rotate_image_cw(gray, rotate_angle)
            h, w = gray.shape[:2]
            return gray[:int(h * 0.35), :int(w * 0.60)].copy()
        except Exception:
            return None

    def detect_orientation(self, page, pno):
        """
        Phát hiện hướng trang (NORMAL / ROTATED / REVIEW).
        Trả về None nếu chưa cài Tesseract (bỏ qua kiểm tra, không coi là trang bình thường).
        """
        if self.orientation_detector is None or not self.orientation_detector.is_available:
            return None
        try:
            return xoay_trang.inspect_page_orientation(
                self.orientation_detector, page, pno + 1,
                conf_threshold=self.orientation_detector.conf_threshold
            )
        except Exception as e:
            return {
                'page_number': pno + 1, 'status': xoay_trang.STATUS_REVIEW, 'correction_angle': 0,
                'rotate': None, 'orientation_conf': None, 'detection_method': '-',
                'reason': f"Lỗi kiểm tra hướng trang: {e}"
            }

    def detect_confidential(self, pdf_path, pno, rotate_angle=0, page=None):
        """Phát hiện tài liệu mật trên trang (chỉ ghi nhận khi độ tin cậy >= 0.80)."""
        if self.mat_detector is None:
            return False, "BÌNH THƯỜNG", 0.0, None

        if rotate_angle and page is not None:
            roi_gray = self.render_rotated_header_roi(page, rotate_angle)
        else:
            roi_gray, _, _ = self.mat_detector.render_header_roi(pdf_path, pno)
        if roi_gray is None:
            return False, "BÌNH THƯỜNG", 0.0, None
            
        res = self.mat_detector.detect_stamp(roi_gray)
        conf = float(res.get('confidence', 0.0))
        is_mat = (res.get('status', "BÌNH THƯỜNG") != "BÌNH THƯỜNG") and (conf >= 0.80)
        
        crop_hl = None
        if is_mat and res.get('box') is not None:
            bx, by, bw, bh = res['box']
            hl_img = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(hl_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
            cv2.putText(hl_img, f"[{res['status']}] {conf:.2f}", (bx, max(20, by - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            crop_hl = hl_img
            
        return is_mat, (res.get('status', "BÌNH THƯỜNG") if is_mat else "BÌNH THƯỜNG"), conf, crop_hl

    def detect_streaks(self, img, min_height_ratio=0.95):
        """Phát hiện sọc scan (vertical streaks) với ngưỡng nghiêm ngặt >= 0.95 chiều cao trang."""
        if detect_vertical_streaks is None:
            return False, [], None
            
        annotated_img, streaks = detect_vertical_streaks(img, min_height_ratio=min_height_ratio)
        has_streaks = len(streaks) > 0
        return has_streaks, streaks, annotated_img if has_streaks else None

    def segment_file_documents(self, fname, page_classifications):
        """
        Bóc tách luồng văn bản độc lập trong file PDF dựa vào các điểm START và END:
        - Văn bản Tiếng Anh
        - Văn bản Tiếng Việt
        - Xác định trạng thái khép kín: Hợp lệ / Cần kiểm tra (thiếu đầu / thiếu cuối / bị cắt ngang).
          Ranh giới văn bản dựa trên quy tắc nhận diện (từ khóa, tiêu ngữ, dấu đỏ) nên không đủ chắc chắn để kết luận Lỗi.
        """
        documents = []
        open_doc = None
        doc_count = 0
        
        for pno_idx, p_info in enumerate(page_classifications, start=1):
            is_en_s = p_info['is_en_start']
            is_en_e = p_info['is_en_end']
            is_vn_s = p_info['is_vn_start']
            is_vn_e = p_info['is_vn_end']
            
            # 1. Nếu trang đồng thời là Start và End (Văn bản 1 trang)
            if (is_en_s and is_en_e) or (is_vn_s and is_vn_e):
                # Nếu trước đó đang có văn bản mở chưa đóng, đóng nó dạng thiếu trang cuối
                if open_doc is not None:
                    open_doc['end_page'] = pno_idx - 1
                    open_doc['status'] = "Cần kiểm tra (Bị cắt ngang hoặc thiếu trang cuối)"
                    documents.append(open_doc)
                    open_doc = None
                    
                lang = p_info.get('lang_hint') or ("Tiếng Anh" if is_en_s else "Tiếng Việt")
                doc_type = "Thư Fax / Thư thương mại (1 trang)" if lang == "Tiếng Anh" else "Văn bản hành chính / Quyết định (1 trang)"
                doc_count += 1
                documents.append({
                    'doc_id': doc_count,
                    'file': fname,
                    'lang': lang,
                    'type': doc_type,
                    'start_page': pno_idx,
                    'end_page': pno_idx,
                    'page_count': 1,
                    'status': "Hợp lệ (Khép kín)",
                    'summary': p_info['top_text'] or p_info['bot_text']
                })
                continue
                
            # 2. Bắt đầu văn bản mới
            if is_en_s or is_vn_s:
                if open_doc is not None:
                    open_doc['end_page'] = pno_idx - 1
                    open_doc['status'] = "Cần kiểm tra (Thiếu trang cuối - văn bản trước bị ngắt)"
                    documents.append(open_doc)
                    open_doc = None
                    
                lang = p_info.get('lang_hint') or ("Tiếng Anh" if is_en_s else "Tiếng Việt")
                doc_type = "Hồ sơ Tiếng Anh (Fax / Hợp đồng / Đề xuất thầu)" if lang == "Tiếng Anh" else "Hồ sơ Tiếng Việt (Tờ trình / Báo cáo Đảng)"
                doc_count += 1
                open_doc = {
                    'doc_id': doc_count,
                    'file': fname,
                    'lang': lang,
                    'type': doc_type,
                    'start_page': pno_idx,
                    'end_page': None,
                    'page_count': 0,
                    'status': "Đang xử lý",
                    'summary': p_info['top_text']
                }
                
            # 3. Kết thúc văn bản
            elif is_en_e or is_vn_e:
                if open_doc is not None:
                    open_doc['end_page'] = pno_idx
                    open_doc['page_count'] = pno_idx - open_doc['start_page'] + 1
                    open_doc['status'] = "Hợp lệ (Khép kín)"
                    documents.append(open_doc)
                    open_doc = None
                else:
                    # Xuất hiện trang kết thúc mồ côi (thuộc văn bản bị ngắt từ file trước)
                    doc_count += 1
                    documents.append({
                        'doc_id': doc_count,
                        'file': fname,
                        'lang': p_info.get('lang_hint') or ("Tiếng Anh" if is_en_e else "Tiếng Việt"),
                        'type': "Trang kết thúc mồ côi (Chữ ký / Dấu đỏ)",
                        'start_page': "-",
                        'end_page': pno_idx,
                        'page_count': 1,
                        'status': "Cần kiểm tra (Thiếu trang đầu - thuộc văn bản file trước)",
                        'summary': p_info['bot_text']
                    })
                    
        # Nếu cuối file còn văn bản mở chưa có trang kết thúc
        if open_doc is not None:
            open_doc['end_page'] = len(page_classifications)
            open_doc['page_count'] = len(page_classifications) - open_doc['start_page'] + 1
            open_doc['status'] = "Cần kiểm tra (Thiếu trang cuối - bị ngắt qua file sau)"
            documents.append(open_doc)
            
        return documents

    @staticmethod
    def _conclude_file(r):
        """
        Kết luận 3 mức cho một file:
        - Lỗi          : sọc scan, trang bị xoay, sai lệch Mục lục đọc được chắc chắn / lỗi của bảng Mục lục.
        - Cần kiểm tra : văn bản bóc tách chưa khép kín, trang xoay chưa chắc chắn, Mục lục cần đối chiếu thủ công.
        - Không lỗi    : còn lại.
        """
        errors, reviews = [], []
        if r['soc_count']:
            errors.append(f"LỖI SỌC SCAN (ngưỡng 0.95): {r['soc_count']} trang ({r['soc_str']})")
        if r['xoay_count']:
            errors.append(f"LỖI TRANG BỊ XOAY: {r['xoay_count']} trang ("
                          + ", ".join(f"Trang {p} ({a}°)" for p, a in r.get('xoay_pages_detail', [])) + ")")
        if r['toc_err']:
            errors.append(f"LỖI MỤC LỤC: {r['toc_err']} sai lệch" + (f" ({'; '.join(r['toc_notes'])})" if r['toc_notes'] else ""))
        if r['error_docs']:
            reviews.append(f"CẦN KIỂM TRA: {r['error_docs']} văn bản chưa xác định được đủ đầu/cuối")
        if r['xoay_review_count']:
            reviews.append(f"CẦN KIỂM TRA hướng trang thủ công: {r['xoay_review_count']} trang")
        if r['toc_review']:
            reviews.append(f"CẦN KIỂM TRA đối chiếu Mục lục: {r['toc_review']} văn bản (số/ngày viết tay hoặc OCR chưa chắc chắn)")
        if errors:
            r['ket_luan'] = KL_LOI
        elif reviews:
            r['ket_luan'] = KL_CAN_KIEM_TRA
        else:
            r['ket_luan'] = KL_KHONG_LOI
        details = errors + reviews
        if not details:
            details.append(f"Khép kín hoàn hảo ({r['valid_docs']}/{r['total_docs']} văn bản đạt chuẩn, không có sọc scan, không có trang bị xoay)")
        if r['mat_count']:
            details.append(f"Mật: {r['mat_count']} trang")
        r['chi_tiet'] = " | ".join(details)

    def cross_check_toc(self, pdf_files, page_info, rotations, log, progress_callback=None):
        """Đối chiếu Mục lục cho từng hồ sơ có trang Mục lục. Trả về (records, warnings)."""
        records, warnings = [], []
        if doi_chieu_muc_luc is None or self.ocr is None:
            log("[!] Thiếu RapidOCR / module doi_chieu_muc_luc: bỏ qua đối chiếu Mục lục.")
            return records, warnings
        groups = {}
        for f in pdf_files:
            groups.setdefault(doi_chieu_muc_luc.group_key(f), []).append(f)
        log("\n[*] Đối chiếu Mục lục với văn bản gốc...")
        for key, files in groups.items():
            files = sorted(files, key=natural_sort_key)
            if progress_callback:
                progress_callback(99.0, f"Đang đối chiếu Mục lục hồ sơ {key}...")
            try:
                res = doi_chieu_muc_luc.cross_check_dossier(
                    self.ocr, files, page_classifications=page_info, rotations=rotations,
                    evidence_dir=self.evidence_dir, log=log)
            except Exception as e:
                log(f"    [LỖI] Không đối chiếu được Mục lục hồ sơ {key}: {e}")
                continue
            if res is None:
                continue
            records.extend(res['records'])
            for code, detail in res['toc_warnings']:
                warnings.append({'dossier': key, 'file': res['toc_file'], 'code': code, 'detail': detail})
            from collections import Counter
            c = Counter(r['status'] for r in res['records'])
            log(f"    -> Hồ sơ {key}: {c.get('Khớp', 0)} khớp, {c.get('Lỗi', 0)} lỗi, {c.get('Cần kiểm tra', 0)} cần kiểm tra"
                + (f"; Mục lục: {'; '.join(d for _, d in res['toc_warnings'])}" if res['toc_warnings'] else ""))
        if records:
            self.export_toc_txt(records, warnings)
        return records, warnings

    def export_toc_txt(self, records, warnings):
        """Báo cáo văn bản tổng hợp kết quả đối chiếu Mục lục (reports/ket_qua_kiem_tra_van_ban.txt)."""
        path = os.path.join(self.output_dir, "ket_qua_kiem_tra_van_ban.txt")
        from collections import Counter
        c = Counter(r['status'] for r in records)
        lines = ["=" * 100, "     BÁO CÁO ĐỐI CHIẾU MỤC LỤC VĂN BẢN VỚI VĂN BẢN GỐC", "=" * 100, "",
                 f"Tổng số văn bản trong Mục lục: {len(records)}",
                 f"  - Khớp hoàn toàn : {c.get('Khớp', 0)}",
                 f"  - Lỗi            : {c.get('Lỗi', 0)}",
                 f"  - Cần kiểm tra   : {c.get('Cần kiểm tra', 0)}  (số/ngày viết tay hoặc OCR chưa chắc chắn - xem ảnh minh chứng)",
                 ""]
        if warnings:
            lines.append("CẢNH BÁO BẢNG MỤC LỤC:")
            lines += [f"  [!] Hồ sơ {w['dossier']}: {w['detail']}" for w in warnings]
            lines.append("")
        lines.append("-" * 100)
        for r in records:
            lines.append(f"STT {r['stt']} | Hồ sơ {r['dossier']} | {r['file']} trang {r['page']} (tờ {r['sheet']}) | "
                         f"Số KH: {r['so_kh_ml']} / {r['so_kh_goc'] or '?'} | Ngày: {r['ngay_ml']} / {r['ngay_goc'] or '?'} | {r['status'].upper()}")
            for code, lvl, txt in r['issues']:
                lines.append(f"      - [{lvl}] {txt}")
            if r['evidence']:
                lines.append(f"      - Ảnh minh chứng: {', '.join(r['evidence'])}")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception:
            pass

    def process_all_files(self, progress_callback=None, log_callback=None, row_callback=None):
        """
        Quét và phân tích toàn bộ file PDF trong hồ sơ.
        Hỗ trợ các callbacks để cập nhật giao diện thời gian thực.
        """
        def log(msg):
            print(msg)
            if log_callback:
                log_callback(msg)

        pdf_files = self.find_pdf_files()
        if not pdf_files:
            log("[LỖI] Không tìm thấy file PDF nào trong thư mục!")
            return None
            
        log("=" * 85)
        log("HỆ THỐNG KIỂM TRA & BÓC TÁCH VĂN BẢN ĐA NGÔN NGỮ (ANH - VIỆT) v4.0")
        log("=" * 85)
        log(f"[*] Tìm thấy {len(pdf_files)} file PDF cần xử lý.")
        
        file_reports = []
        all_segmented_docs = []
        all_mat_records = []
        all_soc_records = []
        all_xoay_records = []
        all_page_info = {}      # (file, trang) -> đặc trưng trang, dùng cho đối chiếu Mục lục
        all_rotations = {}      # (file, trang) -> góc cần xoay
        total_files = len(pdf_files)

        if self.orientation_detector is None or not self.orientation_detector.is_available:
            note = self.orientation_detector.engine_note if self.orientation_detector is not None else "Thiếu file xoay_trang.py."
            log(f"[!] Không có bộ máy kiểm tra hướng trang ({note}): BỎ QUA kiểm tra trang bị xoay (ghi 'Chưa kiểm tra').")
        else:
            log(f"[*] Kiểm tra trang bị xoay bằng: {self.orientation_detector.engine_note}")
        
        for file_idx, pdf_path in enumerate(pdf_files, start=1):
            fname = os.path.basename(pdf_path)
            doc = pymupdf.open(pdf_path)
            num_pages = len(doc)
            log(f"\n[+] Phân tích file {file_idx}/{total_files}: {fname} ({num_pages} trang)...")
            
            page_classifications = []
            mat_pages = []
            soc_pages = []
            xoay_pages = []          # trang ROTATED (được sửa)
            xoay_review_pages = []   # trang REVIEW (cần người kiểm tra, KHÔNG tự xoay)
            orientation_results = []
            
            for pno in range(num_pages):
                p_display_no = pno + 1
                if progress_callback:
                    pct = ((file_idx - 1) + (pno + 1) / max(1, num_pages)) / total_files * 100.0
                    progress_callback(pct, f"Đang phân tích {fname} (trang {p_display_no}/{num_pages})...")
                    
                page = doc[pno]
                pix = page.get_pixmap(dpi=100)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # 0. Phát hiện Trang bị xoay -> xoay ảnh về đúng chiều trước các bước kiểm tra sau
                rotate_angle = 0
                o_res = self.detect_orientation(page, pno)
                if o_res is not None:
                    orientation_results.append(o_res)
                    o_status = o_res['status']
                    if o_status == xoay_trang.STATUS_ROTATED:
                        rotate_angle = o_res['correction_angle']
                        xoay_pages.append((p_display_no, rotate_angle))
                        all_rotations[(fname, p_display_no)] = rotate_angle
                        cv2.imwrite(os.path.join(self.xoay_dir, f"{os.path.splitext(fname)[0]}_p{p_display_no}_xoay{rotate_angle}.png"), img)
                        img = xoay_trang.rotate_image_cw(img, rotate_angle)
                    elif o_status == xoay_trang.STATUS_REVIEW:
                        xoay_review_pages.append(p_display_no)
                        cv2.imwrite(os.path.join(self.xoay_dir, f"{os.path.splitext(fname)[0]}_p{p_display_no}_can_kiem_tra.png"), img)
                    if o_status != xoay_trang.STATUS_NORMAL:
                        all_xoay_records.append({
                            'file': fname,
                            'page': p_display_no,
                            'status': o_status,
                            'angle': o_res.get('correction_angle', 0),
                            'conf': o_res.get('orientation_conf'),
                            'method': o_res.get('detection_method', '-'),
                            'reason': o_res.get('reason') or '',
                            'fixed': False
                        })

                # Bỏ qua 3 trang đầu mục lục ở file 0056_1.pdf
                is_toc = (fname == "0056_1.pdf" and pno < 3) or (pno == 0 and self.is_toc_page(img))
                
                # 1. Phân loại Đầu - Cuối
                if not is_toc:
                    p_info = self.classify_page_boundaries(page, img, pno, num_pages, use_text_layer=(rotate_angle == 0))
                else:
                    p_info = {
                        'is_en_start': False, 'is_en_end': False,
                        'is_vn_start': False, 'is_vn_end': False,
                        'has_red_stamp': False, 'crop_start': None, 'crop_end': None,
                        'top_text': 'MỤC LỤC VĂN BẢN', 'bot_text': ''
                    }
                page_classifications.append(p_info)
                all_page_info[(fname, p_display_no)] = {
                    'is_vn_start': p_info['is_vn_start'], 'is_en_start': p_info['is_en_start'],
                    'has_red_stamp': p_info['has_red_stamp'], 'is_toc': is_toc}
                
                # Lưu ảnh minh chứng nếu có Start / End
                if p_info['is_en_start'] or p_info['is_vn_start']:
                    if p_info.get('crop_start') is not None:
                        cv2.imwrite(os.path.join(self.evidence_dir, f"{os.path.splitext(fname)[0]}_p{p_display_no}_start.png"), p_info['crop_start'])
                if p_info['is_en_end'] or p_info['is_vn_end']:
                    if p_info.get('crop_end') is not None:
                        cv2.imwrite(os.path.join(self.evidence_dir, f"{os.path.splitext(fname)[0]}_p{p_display_no}_end.png"), p_info['crop_end'])
                        
                # 2. Phát hiện Tài liệu Mật (ngưỡng >= 0.80)
                is_mat, mat_type, mat_conf, mat_img = self.detect_confidential(pdf_path, pno, rotate_angle=rotate_angle, page=page)
                if is_mat:
                    mat_pages.append((p_display_no, mat_type, mat_conf))
                    all_mat_records.append({
                        'file': fname,
                        'page': p_display_no,
                        'type': mat_type,
                        'conf': mat_conf
                    })
                    if mat_img is not None:
                        cv2.imwrite(os.path.join(self.mat_dir, f"{os.path.splitext(fname)[0]}_p{p_display_no}_{mat_type}.png"), mat_img)
                        
                # 3. Phát hiện Sọc Scan
                has_soc, streaks, soc_img = self.detect_streaks(img)
                if has_soc:
                    soc_pages.append((p_display_no, len(streaks)))
                    all_soc_records.append({
                        'file': fname,
                        'page': p_display_no,
                        'count': len(streaks)
                    })
                    if soc_img is not None:
                        cv2.imwrite(os.path.join(self.soc_dir, f"{os.path.splitext(fname)[0]}_p{p_display_no}_soc.png"), soc_img)
                        
            doc.close()

            # Xuất bản PDF đã sửa xoay (chỉ trang ROTATED, giữ nguyên file gốc)
            pdf_xoay_path = None
            if xoay_pages and self.fix_rotation:
                try:
                    out_pdf = os.path.join(self.pdf_xoay_dir, f"{os.path.splitext(fname)[0]}_da_xoay.pdf")
                    fix_info = xoay_trang.save_rotated_pdf(pdf_path, out_pdf, orientation_results)
                    pdf_xoay_path = fix_info['output_path']
                    fixed_set = set(fix_info['corrected_pages'])
                    for rec in all_xoay_records:
                        if rec['file'] == fname and rec['page'] in fixed_set:
                            rec['fixed'] = True
                    log(f"    -> Đã xuất PDF sửa xoay ({len(fixed_set)} trang): {pdf_xoay_path}")
                    for w in fix_info['warnings']:
                        log(f"    [!] {w}")
                except Exception as e:
                    log(f"    [LỖI] Không xuất được PDF sửa xoay cho {fname}: {e}")

            # Bóc tách các văn bản trong file
            segmented_docs = self.segment_file_documents(fname, page_classifications)
            for d in segmented_docs:
                all_segmented_docs.append(d)
                
            en_count = sum(1 for d in segmented_docs if d['lang'] == "Tiếng Anh")
            vn_count = sum(1 for d in segmented_docs if d['lang'] == "Tiếng Việt")
            valid_count = sum(1 for d in segmented_docs if d['status'] == "Hợp lệ (Khép kín)")
            error_count = len(segmented_docs) - valid_count
            
            mat_summary = ", ".join([f"Trang {p} ({t})" for p, t, _ in mat_pages]) if mat_pages else "-"
            soc_summary = ", ".join([f"Trang {p} ({c} sọc)" for p, c in soc_pages]) if soc_pages else "-"
            if self.orientation_detector is None or not self.orientation_detector.is_available:
                xoay_summary = "Chưa kiểm tra"
            else:
                xoay_parts = [f"Trang {p} (cần xoay {a}°)" for p, a in xoay_pages]
                xoay_parts += [f"Trang {p} (cần kiểm tra)" for p in xoay_review_pages]
                xoay_summary = ", ".join(xoay_parts) if xoay_parts else "-"

            file_report = {
                'stt': file_idx,
                'file_name': fname,
                'total_pages': num_pages,
                'total_docs': len(segmented_docs),
                'en_docs': en_count,
                'vn_docs': vn_count,
                'valid_docs': valid_count,
                'error_docs': error_count,
                'mat_count': len(mat_pages),
                'mat_str': mat_summary,
                'soc_count': len(soc_pages),
                'soc_str': soc_summary,
                'xoay_count': len(xoay_pages),
                'xoay_review_count': len(xoay_review_pages),
                'xoay_str': xoay_summary,
                'pdf_da_xoay': pdf_xoay_path or "-",
                'xoay_pages_detail': xoay_pages,
                'toc_err': 0,
                'toc_review': 0,
                'toc_str': "-",
                'toc_notes': [],
            }
            self._conclude_file(file_report)
            file_reports.append(file_report)
            
            log(f"    -> Tổng văn bản bóc tách: {len(segmented_docs)} (Tiếng Anh: {en_count}, Tiếng Việt: {vn_count})")
            log(f"    -> Hợp lệ: {valid_count} | Cần kiểm tra: {error_count} => KẾT LUẬN: {file_report['ket_luan']}")
            
            if row_callback:
                row_callback(file_report)
                
        # Đối chiếu Mục lục với văn bản gốc (theo từng hồ sơ: các file nối tiếp 0056_1, 0056_2, ...)
        toc_records, toc_warnings = [], []
        if self.check_toc:
            toc_records, toc_warnings = self.cross_check_toc(pdf_files, all_page_info, all_rotations, log, progress_callback)
            by_file = {r['file_name']: r for r in file_reports}
            for rec in toc_records:
                r = by_file.get(rec['file'])
                if r is None:
                    continue
                if rec['status'] == doi_chieu_muc_luc.STATUS_ERROR:
                    r['toc_err'] += 1
                elif rec['status'] == doi_chieu_muc_luc.STATUS_REVIEW:
                    r['toc_review'] += 1
            for w in toc_warnings:
                r = by_file.get(w['file'])
                if r is not None:
                    r['toc_err'] += 1
                    r['toc_notes'].append(w['detail'])
            for r in file_reports:
                if r['toc_err'] or r['toc_review'] or any(rec['file'] == r['file_name'] for rec in toc_records):
                    n_match = sum(1 for rec in toc_records if rec['file'] == r['file_name'] and rec['status'] == doi_chieu_muc_luc.STATUS_MATCH)
                    r['toc_str'] = f"{n_match} khớp / {r['toc_err']} lỗi / {r['toc_review']} cần KT"
                self._conclude_file(r)

        return {
            'file_reports': file_reports,
            'toc_records': toc_records,
            'toc_warnings': toc_warnings,
            'segmented_docs': all_segmented_docs,
            'mat_records': all_mat_records,
            'soc_records': all_soc_records,
            'xoay_records': all_xoay_records
        }

    def export_excel(self, results):
        """Xuất file Excel chuyên nghiệp với đầy đủ 6 Sheet."""
        file_reports = results['file_reports']
        segmented_docs = results['segmented_docs']
        mat_records = [m for m in results['mat_records'] if float(m.get('conf', 0.0)) >= 0.80]
        soc_records = results['soc_records']
        xoay_records = results.get('xoay_records', [])
        
        wb = openpyxl.Workbook()
        
        h_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        border = Border(
            left=Side(style='thin', color='CBD5E1'),
            right=Side(style='thin', color='CBD5E1'),
            top=Side(style='thin', color='CBD5E1'),
            bottom=Side(style='thin', color='CBD5E1')
        )
        err_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
        err_font = Font(name="Calibri", size=10, bold=True, color="991B1B")
        ok_fill = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
        ok_font = Font(name="Calibri", size=10, bold=True, color="166534")
        rv_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
        rv_font = Font(name="Calibri", size=10, bold=True, color="92400E")

        def paint(cell, text):
            """Tô màu 3 mức: Lỗi (đỏ) / Cần kiểm tra (vàng) / Không lỗi, Khớp, Hợp lệ (xanh)."""
            t = str(text)
            if t.startswith("Lỗi"):
                cell.fill, cell.font = err_fill, err_font
            elif t.startswith("Cần kiểm tra") or t.startswith("Cảnh báo"):
                cell.fill, cell.font = rv_fill, rv_font
            else:
                cell.fill, cell.font = ok_fill, ok_font
        
        # -------------------------------------------------------------
        # SHEET 1: TỔNG QUAN THEO FILE PDF
        # -------------------------------------------------------------
        ws1 = wb.active
        ws1.title = "Kiểm Tra Theo File"
        h_fill_1 = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
        
        headers1 = [
            "STT", "Tên File PDF", "Tổng Trang", "Tổng Văn Bản",
            "VB Tiếng Anh", "VB Tiếng Việt", "VB Hợp Lệ", "VB Cần Kiểm Tra",
            "Tài Liệu Mật", "Sọc Scan", "Trang Bị Xoay", "Đối Chiếu Mục Lục", "Kết Luận", "Mô Tả Chi Tiết"
        ]
        ws1.append(headers1)
        for c in range(1, len(headers1) + 1):
            cell = ws1.cell(row=1, column=c)
            cell.font = h_font
            cell.fill = h_fill_1
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border
            
        for r_idx, r in enumerate(file_reports, start=2):
            vals = [
                r['stt'], r['file_name'], r['total_pages'], r['total_docs'],
                r['en_docs'], r['vn_docs'], r['valid_docs'], r['error_docs'],
                r['mat_str'], r['soc_str'], r.get('xoay_str', '-'), r.get('toc_str', '-'), r['ket_luan'], r['chi_tiet']
            ]
            ws1.append(vals)
            for c_idx in range(1, len(vals) + 1):
                cell = ws1.cell(row=r_idx, column=c_idx)
                cell.border = border
                if c_idx in [1, 3, 4, 5, 6, 7, 8, 13]:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")
                if c_idx == 13:
                    paint(cell, r['ket_luan'])
                        
        for col in ws1.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            ws1.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 12)
            
        # -------------------------------------------------------------
        # SHEET 2: BÓC TÁCH CHI TIẾT TỪNG VĂN BẢN (ANH / VIỆT)
        # -------------------------------------------------------------
        ws2 = wb.create_sheet(title="Bóc Tách Chi Tiết Văn Bản")
        h_fill_2 = PatternFill(start_color="0D9488", end_color="0D9488", fill_type="solid")
        headers2 = [
            "STT", "Tên File PDF", "Văn Bản Số", "Ngôn Ngữ", "Loại Văn Bản",
            "Trang Bắt Đầu", "Trang Kết Thúc", "Số Trang", "Trạng Thái Khép Kín", "Trích Yếu Nội Dung"
        ]
        ws2.append(headers2)
        for c in range(1, len(headers2) + 1):
            cell = ws2.cell(row=1, column=c)
            cell.font = h_font
            cell.fill = h_fill_2
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border
            
        for d_idx, d in enumerate(segmented_docs, start=2):
            vals = [
                d_idx - 1, d['file'], d['doc_id'], d['lang'], d['type'],
                d['start_page'], d['end_page'], d['page_count'], d['status'], d['summary']
            ]
            ws2.append(vals)
            for c_idx in range(1, len(vals) + 1):
                cell = ws2.cell(row=d_idx, column=c_idx)
                cell.border = border
                if c_idx in [1, 3, 4, 6, 7, 8, 9]:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")
                    
                if c_idx == 9:
                    paint(cell, d['status'])
                        
        for col in ws2.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            ws2.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 14)
            
        # -------------------------------------------------------------
        # SHEET 3: CHI TIẾT TÀI LIỆU MẬT (>= 0.80)
        # -------------------------------------------------------------
        ws3 = wb.create_sheet(title="Tài Liệu Mật")
        h_fill_3 = PatternFill(start_color="B45309", end_color="B45309", fill_type="solid")
        headers3 = ["STT", "Tên File PDF", "Trang Số", "Cấp Độ Mật", "Độ Tin Cậy Template"]
        ws3.append(headers3)
        for c in range(1, len(headers3) + 1):
            cell = ws3.cell(row=1, column=c)
            cell.font = h_font
            cell.fill = h_fill_3
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border
            
        for m_idx, m in enumerate(mat_records, start=1):
            vals = [m_idx, m['file'], m['page'], m['type'], f"{m['conf']:.2f}"]
            ws3.append(vals)
            for c_idx in range(1, len(vals) + 1):
                cell = ws3.cell(row=m_idx + 1, column=c_idx)
                cell.border = border
                cell.alignment = Alignment(horizontal="center", vertical="center")
                
        for col in ws3.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            ws3.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 14)
            
        # -------------------------------------------------------------
        # SHEET 4: CHI TIẾT TRANG CÓ SỌC SCAN
        # -------------------------------------------------------------
        ws4 = wb.create_sheet(title="Trang Có Sọc Scan")
        h_fill_4 = PatternFill(start_color="475569", end_color="475569", fill_type="solid")
        headers4 = ["STT", "Tên File PDF", "Trang Số", "Số Lượng Đường Sọc Phát Hiện"]
        ws4.append(headers4)
        for c in range(1, len(headers4) + 1):
            cell = ws4.cell(row=1, column=c)
            cell.font = h_font
            cell.fill = h_fill_4
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border
            
        for s_idx, s in enumerate(soc_records, start=1):
            vals = [s_idx, s['file'], s['page'], s['count']]
            ws4.append(vals)
            for c_idx in range(1, len(vals) + 1):
                cell = ws4.cell(row=s_idx + 1, column=c_idx)
                cell.border = border
                cell.alignment = Alignment(horizontal="center", vertical="center")
                
        for col in ws4.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            ws4.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 14)

        # -------------------------------------------------------------
        # SHEET 5: CHI TIẾT TRANG BỊ XOAY
        # -------------------------------------------------------------
        ws5 = wb.create_sheet(title="Trang Bị Xoay")
        h_fill_5 = PatternFill(start_color="7C3AED", end_color="7C3AED", fill_type="solid")
        warn_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
        warn_font = Font(name="Calibri", size=10, bold=True, color="92400E")
        headers5 = [
            "STT", "Tên File PDF", "Trang Số", "Trạng Thái", "Góc Cần Xoay (chiều kim đồng hồ)",
            "Độ Tin Cậy OSD", "Phương Pháp", "Đã Sửa Trong PDF Mới", "Ghi Chú"
        ]
        ws5.append(headers5)
        for c in range(1, len(headers5) + 1):
            cell = ws5.cell(row=1, column=c)
            cell.font = h_font
            cell.fill = h_fill_5
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border

        for x_idx, x in enumerate(xoay_records, start=1):
            is_rotated = x['status'] == "ROTATED"
            vals = [
                x_idx, x['file'], x['page'],
                "Bị xoay" if is_rotated else "Cần kiểm tra",
                f"{x['angle']}°" if is_rotated else "-",
                f"{x['conf']:.2f}" if x.get('conf') is not None else "-",
                x.get('method', '-'),
                "Có" if x.get('fixed') else "Không",
                x.get('reason') or ("" if is_rotated else "Không tự xoay trang cần kiểm tra")
            ]
            ws5.append(vals)
            for c_idx in range(1, len(vals) + 1):
                cell = ws5.cell(row=x_idx + 1, column=c_idx)
                cell.border = border
                cell.alignment = Alignment(horizontal="left" if c_idx in [2, 9] else "center", vertical="center")
                if c_idx == 4:
                    cell.fill = err_fill if is_rotated else warn_fill
                    cell.font = err_font if is_rotated else warn_font

        for col in ws5.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            ws5.column_dimensions[get_column_letter(col[0].column)].width = min(max(max_len + 4, 14), 80)

        # -------------------------------------------------------------
        # SHEET 6: ĐỐI CHIẾU MỤC LỤC VỚI VĂN BẢN GỐC
        # -------------------------------------------------------------
        toc_records = results.get('toc_records', [])
        toc_warnings = results.get('toc_warnings', [])
        ws6 = wb.create_sheet(title="Đối Chiếu Mục Lục")
        h_fill_6 = PatternFill(start_color="0F766E", end_color="0F766E", fill_type="solid")
        headers6 = [
            "Hồ Sơ", "STT (Mục Lục)", "File PDF", "Trang Trong File", "Số Tờ", "Số KH (Mục Lục)", "Số KH (Văn Bản Gốc)",
            "Ngày (Mục Lục)", "Ngày (Văn Bản Gốc)", "Đánh Giá", "Chi Tiết", "Ảnh Minh Chứng (evidence_crops/)"
        ]
        ws6.append(headers6)
        for c in range(1, len(headers6) + 1):
            cell = ws6.cell(row=1, column=c)
            cell.font = h_font
            cell.fill = h_fill_6
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border
        row_i = 1
        for w in toc_warnings:
            row_i += 1
            ws6.append([w['dossier'], "-", w['file'], "Mục lục", "-", "-", "-", "-", "-", "Lỗi", w['detail'], ""])
        for rec in toc_records:
            row_i += 1
            details = "; ".join(f"[{lvl}] {txt}" for _, lvl, txt in rec['issues'])
            ws6.append([rec['dossier'], rec['stt'], rec['file'], rec['page'], rec['sheet'], rec['so_kh_ml'],
                        rec['so_kh_goc'] or "(không đọc được)", rec['ngay_ml'], rec['ngay_goc'] or "(không đọc được)",
                        rec['status'], details, ", ".join(rec['evidence'])])
        for r_i in range(2, row_i + 1):
            for c_idx in range(1, len(headers6) + 1):
                cell = ws6.cell(row=r_i, column=c_idx)
                cell.border = border
                cell.alignment = Alignment(horizontal="left" if c_idx in (11, 12) else "center", vertical="center", wrap_text=c_idx == 11)
            paint(ws6.cell(row=r_i, column=10), ws6.cell(row=r_i, column=10).value)
        for i, wdt in enumerate([10, 12, 14, 12, 8, 16, 18, 14, 16, 14, 80, 50], start=1):
            ws6.column_dimensions[get_column_letter(i)].width = wdt
        if not toc_records and not toc_warnings:
            ws6.append(["Không tìm thấy trang Mục lục (bảng 'MỤC LỤC VĂN BẢN') trong các hồ sơ đã quét."])

        # Lưu file
        saved_paths = []
        for target_p in [self.excel_path, self.alt_excel_path]:
            try:
                wb.save(target_p)
                saved_paths.append(target_p)
            except PermissionError:
                base, ext = os.path.splitext(target_p)
                alt = f"{base}_ket_qua{ext}"
                try:
                    wb.save(alt)
                    saved_paths.append(alt)
                except Exception:
                    pass
            except Exception:
                pass
                
        primary_saved = saved_paths[0] if saved_paths else self.excel_path
        print(f"\n[✓] Đã xuất file báo cáo Excel 6 Sheet: {primary_saved}")
        return primary_saved

    def run(self, progress_callback=None, log_callback=None, row_callback=None):
        """Thực thi kiểm tra toàn bộ."""
        results = self.process_all_files(
            progress_callback=progress_callback,
            log_callback=log_callback,
            row_callback=row_callback
        )
        if not results:
            return 1, None, None
            
        saved_excel = self.export_excel(results)
        
        file_reports = results['file_reports']
        segmented = results['segmented_docs']
        print("\n" + "=" * 95)
        print("BẢNG TỔNG HỢP KIỂM TRA & BÓC TÁCH HỒ SƠ ĐA NGÔN NGỮ:")
        print("=" * 95)
        print(f"{'STT':<4} | {'Tên File':<12} | {'Trang':<6} | {'Tổng VB':<8} | {'VB Tiếng Anh':<13} | {'VB Tiếng Việt':<14} | {'Hợp Lệ':<8} | {'Kết Luận':<9}")
        print("-" * 95)
        
        total_err_files = 0
        total_review_files = 0
        total_en = 0
        total_vn = 0
        total_valid = 0
        
        for r in file_reports:
            if r['ket_luan'] == KL_LOI:
                total_err_files += 1
            elif r['ket_luan'] == KL_CAN_KIEM_TRA:
                total_review_files += 1
            total_en += r['en_docs']
            total_vn += r['vn_docs']
            total_valid += r['valid_docs']
            print(f"{r['stt']:<4} | {r['file_name']:<12} | {r['total_pages']:<6} | {r['total_docs']:<8} | {r['en_docs']:<13} | {r['vn_docs']:<14} | {r['valid_docs']:<8} | {r['ket_luan']:<9}")
            
        print("=" * 95)
        print("TỔNG KẾT TOÀN BỘ HỒ SƠ:")
        print(f"- Tổng số file PDF kiểm tra:          {len(file_reports)}")
        print(f"- Tổng số văn bản bóc tách được:      {len(segmented)}")
        print(f"  + Văn bản Tiếng Anh:                {total_en}")
        print(f"  + Văn bản Tiếng Việt:               {total_vn}")
        print(f"  + Văn bản Hợp lệ (Khép kín):        {total_valid}")
        print(f"  + VB Cần kiểm tra (thiếu đầu/cuối): {len(segmented) - total_valid}")
        print(f"- Số file phát hiện Lỗi:              {total_err_files}")
        print(f"- Số file Cần kiểm tra:               {total_review_files}")
        print(f"- Số file Không lỗi:                  {len(file_reports) - total_err_files - total_review_files}")
        toc_records = results.get('toc_records', [])
        if toc_records or results.get('toc_warnings'):
            from collections import Counter
            c = Counter(r['status'] for r in toc_records)
            print(f"- Đối chiếu Mục lục:                  {c.get('Khớp', 0)} khớp, {c.get('Lỗi', 0)} lỗi, "
                  f"{c.get('Cần kiểm tra', 0)} cần kiểm tra, {len(results.get('toc_warnings', []))} lỗi bảng Mục lục")
        print(f"- Tổng số trang có dấu Mật (>= 0.80): {len(results['mat_records'])}")
        print(f"- Tổng số trang có Sọc scan:          {len(results['soc_records'])}")
        xoay_records = results.get('xoay_records', [])
        print(f"- Tổng số trang bị xoay (đã sửa):     {sum(1 for x in xoay_records if x['status'] == 'ROTATED')}")
        print(f"- Trang cần kiểm tra hướng thủ công:  {sum(1 for x in xoay_records if x['status'] == 'REVIEW')}")
        overall_status = KL_LOI if total_err_files else (KL_CAN_KIEM_TRA if total_review_files else KL_KHONG_LOI)
        print(f"\n>>> ĐẦU RA CHUNG: {overall_status}")
        print("=" * 95)
        # Mã thoát: 0 = Không lỗi, 1 = Lỗi, 2 = Cần kiểm tra
        exit_code = {KL_KHONG_LOI: 0, KL_LOI: 1, KL_CAN_KIEM_TRA: 2}[overall_status]
        return exit_code, results, saved_excel


# Backward-compatible alias
DocumentIntegrityChecker = BiLingualDocumentChecker


def main():
    parser = argparse.ArgumentParser(description="Toolkit bóc tách và kiểm tra hồ sơ lưu trữ đa ngôn ngữ Anh - Việt")
    parser.add_argument("--input", "-i", default=None, help="Thư mục chứa file PDF")
    parser.add_argument("--output", "-o", default=None, help="Thư mục xuất báo cáo")
    parser.add_argument("--khong-sua-xoay", action="store_true",
                        help="Chỉ phát hiện trang bị xoay, không xuất bản PDF đã sửa xoay")
    parser.add_argument("--khong-doi-chieu-muc-luc", action="store_true",
                        help="Bỏ qua bước đối chiếu Mục lục với văn bản gốc")
    args, _ = parser.parse_known_args()

    checker = BiLingualDocumentChecker(target_dir=args.input, output_dir=args.output,
                                       fix_rotation=not args.khong_sua_xoay,
                                       check_toc=not args.khong_doi_chieu_muc_luc)
    code, _, _ = checker.run()
    sys.exit(code)


if __name__ == "__main__":
    main()
