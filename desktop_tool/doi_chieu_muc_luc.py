#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ĐỐI CHIẾU MỤC LỤC VĂN BẢN VỚI VĂN BẢN GỐC (Nhóm 2)

Luồng xử lý cho mỗi hồ sơ (nhóm file nối tiếp: 0056_1.pdf, 0056_2.pdf, ... -> hồ sơ "0056"):
1. Tìm các trang Mục lục ở đầu file đầu tiên (bảng kẻ ô + tiêu đề "MỤC LỤC" / "STT").
2. Bóc tách bảng: OpenCV tìm đường kẻ ngang/dọc -> lưới ô; RapidOCR đọc chữ cả trang rồi gán vào từng ô.
   Ánh xạ cột theo tiêu đề (STT, Số ký hiệu, Ngày tháng, Trích yếu, Tác giả, Trang số, Ghi chú).
3. Đánh số tờ liên tục qua các file của hồ sơ (bỏ qua trang Mục lục) -> biết chính xác cột "Trang số"
   của Mục lục chỉ tới trang nào của file nào (không phụ thuộc OCR số trang viết tay).
4. Tại trang đầu mỗi văn bản: OCR vùng đầu trang, cắt 3 vùng ROI (Số ký hiệu, Ngày tháng, Số tờ viết tay)
   làm ảnh minh chứng và so khớp với Mục lục.

Mức độ đánh giá:
- "Khớp"          : các trường đọc được đều trùng Mục lục.
- "Lỗi"           : đọc RÕ RÀNG (độ tin cậy OCR cao) mà vẫn khác Mục lục, hoặc lỗi của chính bảng Mục lục.
- "Cần kiểm tra"  : OCR không chắc chắn / không đọc được (thường do số, ngày viết tay) -> xem ảnh minh chứng.
================================================================================
"""

import os
import re
import unicodedata

import cv2
import numpy as np
import pymupdf

TOC_DPI = 200
HEADER_DPI = 200
CONFIDENT_OCR = 0.90          # điểm OCR tối thiểu để coi một giá trị đọc được là chắc chắn

STATUS_MATCH = "Khớp"
STATUS_ERROR = "Lỗi"
STATUS_REVIEW = "Cần kiểm tra"

ERROR_TEXT = {
    "LECH_SO_KY_HIEU": "Lệch Số/ký hiệu giữa Mục lục và văn bản gốc",
    "KHUYET_SO_KY_HIEU": "Văn bản gốc khuyết / không đọc được số ký hiệu",
    "LECH_NGAY_THANG": "Lệch Ngày tháng ban hành",
    "KHUYET_NGAY_THANG": "Văn bản gốc khuyết / không đọc được ngày tháng",
    "LECH_SO_TRANG": "Trang Mục lục chỉ tới không khớp trang đầu văn bản",
    "ML_KHUYET_TRANG": "Mục lục bỏ trống cột Trang số",
    "TRUNG_STT_MUC_LUC": "Đánh trùng STT trong Mục lục",
    "VB_CHIA_CAT_QUA_FILE": "Văn bản bị chia cắt qua nhiều file PDF",
    "THIEU_DAU_DO_END_DOC": "Không thấy dấu đỏ kết thúc văn bản",
}


# -----------------------------------------------------------------------------
# Tiện ích chuỗi
# -----------------------------------------------------------------------------
def unaccent(text):
    """Bỏ dấu tiếng Việt (RapidOCR trả về chữ không dấu, Mục lục / text layer có dấu)."""
    text = str(text or "").replace("Đ", "D").replace("đ", "d").replace("Ð", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def normalize_so_kh(text):
    """'39-BC/ĐU', 'Số: 39 -BC/DU' -> ('39', 'BC', 'DU'). Trả về None nếu không nhận ra cấu trúc."""
    t = unaccent(text).upper()
    t = t.replace("–", "-").replace("—", "-").replace("_", "-")
    t = re.sub(r"\bS[O0Ô6]\s*[:.]?", " ", t)           # bỏ chữ "Số:" (OCR hay đọc thành "S6")
    m = re.search(r"(\d{1,4})?\s*-?\s*([A-Z]{1,4})\s*/\s*([A-Z]{1,4})", t)
    if not m:
        return None
    return (m.group(1) or "", m.group(2), m.group(3))


def format_so_kh(parts):
    if not parts:
        return ""
    num, kind, org = parts
    return f"{num}-{kind}/{org}" if num else f"-{kind}/{org}"


def parse_date(text):
    """Đọc ngày: '30/12/2015', '2/6/2016', 'ngày 30 tháng 12 năm 2015' -> (d, m, y) (có thể thiếu d/m)."""
    t = unaccent(text).lower()
    m = re.search(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})", t)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"ngay\s*(\d{1,2})?\D{0,12}?thang\s*(\d{1,2})?\D{0,12}?nam\s*(\d{4})", t)
    if m:
        return (int(m.group(1)) if m.group(1) else None,
                int(m.group(2)) if m.group(2) else None,
                int(m.group(3)))
    return None


def format_date(d):
    if not d:
        return ""
    day, month, year = d
    return f"{day if day else '..'}/{month if month else '..'}/{year}"


def first_int(text):
    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def group_key(file_name):
    """0056_1.pdf -> '0056' (các file nối tiếp của cùng hồ sơ); 0001.pdf -> '0001'."""
    stem = os.path.splitext(os.path.basename(file_name))[0]
    m = re.match(r"^(.+?)[_\-](\d+)$", stem)
    return m.group(1) if m else stem


# -----------------------------------------------------------------------------
# Render & OCR
# -----------------------------------------------------------------------------
def render(page, dpi, clip=None, rotate_cw=0, color=False):
    pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=pymupdf.csRGB if color else pymupdf.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).copy()
    if not color:
        img = img[:, :, 0]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if rotate_cw == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_cw == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotate_cw == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def ocr_boxes(ocr, img):
    """Chạy RapidOCR -> danh sách dict(text, score, x0, y0, x1, y1, cx, cy)."""
    res, _ = ocr(img)
    boxes = []
    for r in res or []:
        pts = np.array(r[0], dtype=float)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        boxes.append({"text": str(r[1]), "score": float(r[2]), "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                      "cx": (x0 + x1) / 2.0, "cy": (y0 + y1) / 2.0})
    return boxes


# -----------------------------------------------------------------------------
# 1-2. Nhận diện & bóc tách bảng Mục lục
# -----------------------------------------------------------------------------
def _line_positions(mask, axis, min_fill):
    """Vị trí các đường kẻ (gộp các hàng/cột pixel liền nhau) có độ phủ >= min_fill."""
    proj = mask.sum(axis=axis) / 255.0
    idx = np.where(proj >= min_fill)[0]
    lines = []
    for i in idx:
        if lines and i - lines[-1][-1] <= 3:
            lines[-1].append(i)
        else:
            lines.append([i])
    return [int(np.mean(g)) for g in lines]


def detect_table_grid(gray):
    """Trả về (y_lines, x_lines) của bảng kẻ ô, hoặc ([], []) nếu trang không có bảng."""
    h, w = gray.shape[:2]
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 12), 1)))
    v_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 60))))
    y_lines = _line_positions(h_mask, axis=1, min_fill=w * 0.45)
    x_lines = _line_positions(v_mask, axis=0, min_fill=h * 0.04)
    if len(y_lines) < 3 or len(x_lines) < 4:
        return [], []
    return y_lines, x_lines


HEADER_KEYS = [
    ("so_kh", ("ky hieu", "so, ky", "so ky", "hieu")),
    ("ngay", ("ngay", "thang")),
    ("trich_yeu", ("trich yeu", "ten loai", "trich")),
    ("tac_gia", ("tac gia",)),
    ("trang", ("trang",)),
    ("ghi_chu", ("ghi chu", "ghi", "chu")),
    ("stt", ("stt", "st")),
]


def _assign_cells(boxes, y_lines, x_lines):
    """Gán từng dòng chữ OCR vào ô (hàng, cột) theo tâm hộp chữ."""
    cells = {}
    for b in boxes:
        r = int(np.searchsorted(y_lines, b["cy"])) - 1
        c = int(np.searchsorted(x_lines, b["cx"])) - 1
        if r < 0 or c < 0 or r >= len(y_lines) - 1 or c >= len(x_lines) - 1:
            continue
        cells.setdefault((r, c), []).append(b)
    out = {}
    for key, items in cells.items():
        items.sort(key=lambda b: (round(b["cy"] / 12.0), b["cx"]))
        text = ""
        for b in items:
            piece = b["text"].strip()
            text = (text + piece) if text.endswith("-") else (f"{text} {piece}".strip())
        out[key] = {"text": text, "score": min(b["score"] for b in items)}
    return out


def _map_header(cells, row, n_cols):
    """Đọc hàng tiêu đề -> {chỉ số cột: tên trường}."""
    mapping = {}
    for c in range(n_cols):
        txt = unaccent(cells.get((row, c), {}).get("text", "")).lower()
        for field, keys in HEADER_KEYS:
            if field in mapping.values():
                continue
            if any(k in txt for k in keys):
                mapping[c] = field
                break
    return mapping


def extract_toc_page(ocr, page, layout=None):
    """
    Bóc tách một trang Mục lục. `layout` (từ trang Mục lục trước) dùng cho trang tiếp theo không có tiêu đề.
    Trả về (rows, layout) hoặc (None, layout) nếu trang không phải bảng Mục lục.
    """
    gray = render(page, TOC_DPI)
    h, w = gray.shape[:2]
    y_lines, x_lines = detect_table_grid(gray)
    if not y_lines:
        return None, layout
    boxes = ocr_boxes(ocr, gray)
    cells = _assign_cells(boxes, y_lines, x_lines)
    n_rows, n_cols = len(y_lines) - 1, len(x_lines) - 1

    header_row, mapping = None, {}
    for r in range(min(3, n_rows)):
        m = _map_header(cells, r, n_cols)
        if len(m) >= 3 and ("stt" in m.values() or "trang" in m.values()):
            header_row, mapping = r, m
            break

    if header_row is not None:
        centers = {f: ((x_lines[c] + x_lines[c + 1]) / 2.0) / w for c, f in mapping.items()}
        layout = {"centers": centers}
    elif layout is not None:
        # Trang Mục lục tiếp theo: ánh xạ cột theo vị trí tương đối của trang có tiêu đề
        for c in range(n_cols):
            cx = ((x_lines[c] + x_lines[c + 1]) / 2.0) / w
            field, center = min(layout["centers"].items(), key=lambda kv: abs(kv[1] - cx))
            if abs(center - cx) < 0.04 and field not in mapping.values():
                mapping[c] = field
    else:
        return None, layout

    if "stt" not in mapping.values() or "so_kh" not in mapping.values():
        return None, layout

    rows = []
    for r in range((header_row + 1) if header_row is not None else 0, n_rows):
        rec = {"y0": y_lines[r], "y1": y_lines[r + 1]}
        for c, field in mapping.items():
            cell = cells.get((r, c))
            if cell is None and field in ("stt", "so_kh", "ngay", "trang"):
                # OCR cả trang hay bỏ sót ô chỉ có 1 ký tự (vd. "1") -> OCR lại riêng ô nếu ô có mực
                crop = gray[y_lines[r] + 4:y_lines[r + 1] - 4, x_lines[c] + 4:x_lines[c + 1] - 4]
                if crop.size and np.count_nonzero(crop < 128) > 15:
                    sub = ocr_boxes(ocr, crop)
                    if sub:
                        cell = {"text": " ".join(b["text"] for b in sorted(sub, key=lambda b: (b["cy"], b["cx"]))),
                                "score": min(b["score"] for b in sub)}
            cell = cell or {"text": "", "score": 1.0}
            rec[field] = cell["text"]
            rec[field + "_score"] = cell["score"]
        stt = first_int(rec.get("stt"))
        if stt is None and not rec.get("so_kh"):
            continue
        rec["stt_num"] = stt
        rec["so_kh_parts"] = normalize_so_kh(rec.get("so_kh", ""))
        rec["ngay_parsed"] = parse_date(rec.get("ngay", ""))
        rec["trang_num"] = first_int(rec.get("trang"))
        rows.append(rec)
    return rows, layout


def page_looks_like_toc_title(ocr, page):
    """Trang đầu Mục lục: có dòng 'MỤC LỤC' ở phần trên trang."""
    r = page.rect
    gray = render(page, 120, clip=pymupdf.Rect(0, 0, r.width, r.height * 0.18))
    txt = unaccent(" ".join(b["text"] for b in ocr_boxes(ocr, gray))).lower().replace(" ", "")
    return "mucluc" in txt


# -----------------------------------------------------------------------------
# 4. Đọc vùng đầu trang văn bản gốc (Số ký hiệu, Ngày tháng, Số tờ viết tay)
# -----------------------------------------------------------------------------
def _same_line(boxes, anchor, x_min, x_max):
    line = [b for b in boxes if abs(b["cy"] - anchor["cy"]) <= max(12.0, 0.6 * (anchor["y1"] - anchor["y0"]))
            and x_min <= b["cx"] <= x_max]
    return sorted(line, key=lambda b: b["x0"])


def has_pen_ink(crop_bgr):
    """Vùng chữ có mực màu (bút bi xanh/tím/đỏ) -> nhiều khả năng số/ngày được viết tay, OCR không đáng tin."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim != 3:
        return False
    b, g, r = [crop_bgr[:, :, i].astype(np.int16) for i in range(3)]
    chroma = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    colored = (chroma > 45) & (np.minimum(np.minimum(b, g), r) < 170)
    return int(np.count_nonzero(colored)) > 40


def _partial_read(doc_digits, ml_digits):
    """OCR đọc thiếu chữ số viết tay ('36' so với '136', '65' so với '135', '1' so với '17'): ít chữ số hơn Mục lục."""
    doc_digits, ml_digits = str(doc_digits), str(ml_digits)
    return doc_digits != ml_digits and len(doc_digits) < len(ml_digits)


def _crop(img, x0, y0, x1, y1, pad=14):
    h, w = img.shape[:2]
    return img[max(0, int(y0) - pad):min(h, int(y1) + pad), max(0, int(x0) - pad):min(w, int(x1) + pad)].copy()


def read_document_header(ocr, page, rotate_cw=0):
    """
    OCR 32% đầu trang văn bản gốc, trả về dict:
    so_kh_text, so_kh_parts, so_kh_score, ngay_text, ngay_parsed, ngay_score, trang_viet_tay,
    crop_so_kh, crop_ngay, crop_trang (ảnh BGR để lưu minh chứng).
    """
    color = render(page, HEADER_DPI, rotate_cw=rotate_cw, color=True)
    h_full, w = color.shape[:2]
    head = color[:int(h_full * 0.32)]
    boxes = ocr_boxes(ocr, cv2.cvtColor(head, cv2.COLOR_BGR2GRAY))
    out = {"so_kh_text": "", "so_kh_parts": None, "so_kh_score": 0.0,
           "ngay_text": "", "ngay_parsed": None, "ngay_score": 0.0, "trang_viet_tay": None,
           "crop_so_kh": None, "crop_ngay": None, "crop_trang": None}

    # Số ký hiệu: dòng bắt đầu bằng "Số" ở nửa trái
    so_anchor = None
    for b in boxes:
        t = unaccent(b["text"]).upper().replace(" ", "")
        if b["cx"] < w * 0.58 and re.match(r"^S[O0Ô6][:.]?", t) and ("/" in t or "-" in t or len(t) <= 4):
            so_anchor = b
            break
    if so_anchor is None:
        for b in boxes:
            if b["cx"] < w * 0.58 and normalize_so_kh(b["text"]):
                so_anchor = b
                break
    if so_anchor is not None:
        line = _same_line(boxes, so_anchor, 0, w * 0.58)
        text = " ".join(b["text"] for b in line)
        out["so_kh_text"] = text
        out["so_kh_parts"] = normalize_so_kh(text)
        out["so_kh_score"] = min(b["score"] for b in line)
        out["crop_so_kh"] = _crop(head, min(b["x0"] for b in line), min(b["y0"] for b in line),
                                  max(b["x1"] for b in line), max(b["y1"] for b in line))

    # Ngày tháng: dòng chứa "ngày ... tháng ... năm" ở nửa phải
    for b in boxes:
        t = unaccent(b["text"]).lower()
        if b["cx"] > w * 0.42 and ("ngay" in t or "thang" in t) and re.search(r"\d{4}", t + "".join(
                x["text"] for x in _same_line(boxes, b, w * 0.42, w))):
            line = _same_line(boxes, b, w * 0.42, w)
            text = " ".join(x["text"] for x in line)
            out["ngay_text"] = text
            out["ngay_parsed"] = parse_date(text)
            out["ngay_score"] = min(x["score"] for x in line)
            out["crop_ngay"] = _crop(head, min(x["x0"] for x in line), min(x["y0"] for x in line),
                                     max(x["x1"] for x in line), max(x["y1"] for x in line))
            break

    # Số tờ viết tay: góc trên bên phải
    corner_y1, corner_x0 = int(h_full * 0.07), int(w * 0.82)
    out["crop_trang"] = head[:corner_y1, corner_x0:].copy()
    corner = [b for b in boxes if b["cy"] < corner_y1 and b["cx"] > corner_x0 and re.fullmatch(r"\s*\d{1,4}\s*", b["text"])]
    if corner:
        best = max(corner, key=lambda b: b["score"])
        out["trang_viet_tay"] = (int(best["text"].strip()), best["score"])
    return out


# -----------------------------------------------------------------------------
# 3 + 4. Đối chiếu cho một hồ sơ
# -----------------------------------------------------------------------------
def _compare_entry(entry, header, sheet, page_info, span_files, span_has_stamp, missing_files=()):
    """So khớp một dòng Mục lục với trang đầu văn bản. Trả về (status, [(mã lỗi, mức, chi tiết)])."""
    issues = []

    # Trang số
    if entry.get("trang_num") is None:
        issues.append(("ML_KHUYET_TRANG", STATUS_ERROR, "Mục lục bỏ trống cột Trang số"))
    elif page_info is None and missing_files:
        issues.append(("LECH_SO_TRANG", STATUS_REVIEW,
                       f"Tờ {entry['trang_num']} nằm ngoài các file đã chọn - chưa chọn đủ hồ sơ "
                       f"(còn {', '.join(missing_files)} cùng thư mục)"))
    elif page_info is None:
        issues.append(("LECH_SO_TRANG", STATUS_ERROR,
                       f"Mục lục ghi trang {entry['trang_num']} nhưng hồ sơ không có tờ số {entry['trang_num']}"))
    else:
        if not page_info.get("is_start"):
            issues.append(("LECH_SO_TRANG", STATUS_REVIEW,
                           f"Tờ {sheet} (Mục lục ghi) không có dấu hiệu trang đầu văn bản (tiêu ngữ / số ký hiệu)"))
        hw = header.get("trang_viet_tay") if header else None
        if hw and hw[1] >= CONFIDENT_OCR and hw[0] != entry["trang_num"]:
            issues.append(("LECH_SO_TRANG", STATUS_REVIEW,
                           f"Số tờ viết tay đọc được là {hw[0]}, Mục lục ghi {entry['trang_num']}"))

    if header is not None:
        # Số ký hiệu
        ml, doc = entry.get("so_kh_parts"), header.get("so_kh_parts")
        sure = (header.get("so_kh_score", 0) >= CONFIDENT_OCR and entry.get("so_kh_score", 0) >= 0.80
                and not has_pen_ink(header.get("crop_so_kh")))
        if doc and ml and doc[0] and _partial_read(doc[0].lstrip("0"), ml[0].lstrip("0")):
            sure = False
        if doc is None:
            issues.append(("KHUYET_SO_KY_HIEU", STATUS_REVIEW, "Không đọc được dòng Số ký hiệu trên văn bản gốc"))
        elif not doc[0]:
            issues.append(("KHUYET_SO_KY_HIEU", STATUS_REVIEW,
                           f"Văn bản gốc để trống / viết tay phần số (đọc được '{format_so_kh(doc)}'), Mục lục ghi '{format_so_kh(ml)}'"))
        elif ml and (doc[0].lstrip("0") != ml[0].lstrip("0") or doc[1] != ml[1]):
            issues.append(("LECH_SO_KY_HIEU", STATUS_ERROR if sure else STATUS_REVIEW,
                           f"Mục lục ghi '{format_so_kh(ml)}', văn bản gốc đọc được '{format_so_kh(doc)}'"
                           + ("" if sure else " (OCR chưa chắc chắn)")))

        # Ngày tháng
        ml_d, doc_d = entry.get("ngay_parsed"), header.get("ngay_parsed")
        sure_d = header.get("ngay_score", 0) >= CONFIDENT_OCR and not has_pen_ink(header.get("crop_ngay"))
        if ml_d and doc_d and doc_d[0] is not None and doc_d[1] is not None and (
                _partial_read(doc_d[0], ml_d[0]) or _partial_read(doc_d[1], ml_d[1])):
            sure_d = False
        if ml_d:
            if doc_d is None:
                issues.append(("KHUYET_NGAY_THANG", STATUS_REVIEW, "Không đọc được dòng ngày tháng trên văn bản gốc"))
            elif doc_d[0] is None or doc_d[1] is None:
                issues.append(("KHUYET_NGAY_THANG", STATUS_REVIEW,
                               f"Văn bản gốc để trống / viết tay ngày tháng (đọc được '{format_date(doc_d)}'), Mục lục ghi '{format_date(ml_d)}'"))
            elif tuple(doc_d) != tuple(ml_d):
                issues.append(("LECH_NGAY_THANG", STATUS_ERROR if sure_d else STATUS_REVIEW,
                               f"Mục lục ghi '{format_date(ml_d)}', văn bản gốc đọc được '{format_date(doc_d)}'"
                               + ("" if sure_d else " (OCR chưa chắc chắn)")))

    if len(span_files) > 1:
        issues.append(("VB_CHIA_CAT_QUA_FILE", "Cảnh báo", "Văn bản trải qua các file: " + " -> ".join(span_files)))
    if page_info is not None and not span_has_stamp:
        issues.append(("THIEU_DAU_DO_END_DOC", STATUS_REVIEW, "Không thấy dấu đỏ trên các trang của văn bản"))

    levels = [lvl for _, lvl, _ in issues]
    if STATUS_ERROR in levels:
        status = STATUS_ERROR
    elif STATUS_REVIEW in levels:
        status = STATUS_REVIEW
    else:
        status = STATUS_MATCH
    return status, issues


def cross_check_dossier(ocr, files, page_classifications=None, rotations=None, evidence_dir=None, log=print):
    """
    Đối chiếu Mục lục cho một hồ sơ (danh sách file PDF theo thứ tự).
    - page_classifications: {(tên file, số trang): dict có is_vn_start / is_en_start / has_red_stamp} từ bước quét chính.
    - rotations: {(tên file, số trang): góc cần xoay} cho các trang bị xoay.
    Trả về None nếu hồ sơ không có Mục lục, ngược lại dict(toc_pages, records, toc_warnings).
    """
    page_classifications = page_classifications or {}
    rotations = rotations or {}
    docs = [pymupdf.open(f) for f in files]
    try:
        first = docs[0]
        fname0 = os.path.basename(files[0])
        if first.page_count == 0 or not page_looks_like_toc_title(ocr, first[0]):
            return None

        # Bóc tách bảng Mục lục (trang 1 và các trang bảng liền sau)
        toc_rows, layout, toc_pages = [], None, []
        for pno in range(first.page_count):
            rows, layout = extract_toc_page(ocr, first[pno], layout)
            if rows is None:
                break
            for r in rows:
                r["toc_page"] = pno + 1
            toc_rows.extend(rows)
            toc_pages.append(pno + 1)
        if not toc_rows:
            log(f"    [!] Thấy tiêu đề Mục lục trong {fname0} nhưng không bóc tách được bảng.")
            return None
        log(f"    -> Mục lục: {len(toc_pages)} trang ({fname0}), {len(toc_rows)} dòng văn bản.")

        # Đánh số tờ liên tục qua các file (bỏ trang Mục lục)
        sheets = {}
        n = 0
        for f, d in zip(files, docs):
            fname = os.path.basename(f)
            for pno in range(d.page_count):
                if fname == fname0 and (pno + 1) in toc_pages:
                    continue
                n += 1
                sheets[n] = (fname, pno + 1, d)
        total_sheets = n

        # File cùng hồ sơ trong cùng thư mục nhưng không được chọn quét (người dùng chọn thiếu file)
        chosen = {os.path.abspath(f) for f in files}
        key0 = group_key(fname0)
        folder = os.path.dirname(os.path.abspath(files[0]))
        missing_files = sorted(
            os.path.basename(f) for f in os.listdir(folder)
            if f.lower().endswith(".pdf") and not f.startswith("._") and group_key(f) == key0
            and os.path.abspath(os.path.join(folder, f)) not in chosen)

        # Lỗi của chính bảng Mục lục
        toc_warnings = []
        seen = {}
        for r in toc_rows:
            if r["stt_num"] is not None:
                seen.setdefault(r["stt_num"], []).append(r)
        for stt, items in sorted(seen.items()):
            if len(items) > 1:
                toc_warnings.append(("TRUNG_STT_MUC_LUC", f"STT {stt} xuất hiện {len(items)} lần trong bảng Mục lục"))

        starts = sorted(r["trang_num"] for r in toc_rows if r.get("trang_num"))
        records = []
        for idx, entry in enumerate(toc_rows, start=1):
            sheet = entry.get("trang_num")
            loc = sheets.get(sheet) if sheet else None
            header, page_info, span_files, span_has_stamp = None, None, [], True
            if loc is not None:
                fname, pno, d = loc
                key = (fname, pno)
                info = page_classifications.get(key, {})
                header = read_document_header(ocr, d[pno - 1], rotate_cw=rotations.get(key, 0))
                has_title = bool(header.get("so_kh_parts")) or bool(info.get("is_vn_start") or info.get("is_en_start"))
                page_info = {"is_start": has_title}
                nxt = [s for s in starts if s > sheet]
                end_sheet = (nxt[0] - 1) if nxt else total_sheets
                span = [sheets[s] for s in range(sheet, end_sheet + 1) if s in sheets]
                span_files = list(dict.fromkeys(s[0] for s in span))
                if page_classifications:
                    span_has_stamp = any(page_classifications.get((s[0], s[1]), {}).get("has_red_stamp") for s in span)

            status, issues = _compare_entry(entry, header, sheet, page_info, span_files, span_has_stamp, missing_files)

            evid = []
            if evidence_dir and header is not None and status != STATUS_MATCH:
                os.makedirs(evidence_dir, exist_ok=True)
                tag = f"{group_key(fname0)}_STT_{entry.get('stt_num') or idx}_To_{sheet}"
                for part in ("so_kh", "ngay", "trang"):
                    crop = header.get(f"crop_{part}")
                    if crop is not None and crop.size > 0:
                        name = f"{tag}_{part}.png"
                        cv2.imwrite(os.path.join(evidence_dir, name), crop)
                        evid.append(name)

            records.append({
                "dossier": group_key(fname0),
                "stt": entry.get("stt_num") if entry.get("stt_num") is not None else entry.get("stt", ""),
                "file": loc[0] if loc else "-",
                "page": loc[1] if loc else "-",
                "sheet": sheet if sheet else "",
                "so_kh_ml": format_so_kh(entry.get("so_kh_parts")) or entry.get("so_kh", ""),
                "so_kh_goc": format_so_kh(header.get("so_kh_parts")) if header else "",
                "so_kh_goc_raw": header.get("so_kh_text", "") if header else "",
                "ngay_ml": format_date(entry.get("ngay_parsed")) or entry.get("ngay", ""),
                "ngay_goc": format_date(header.get("ngay_parsed")) if header else "",
                "trich_yeu": entry.get("trich_yeu", ""),
                "status": status,
                "issues": issues,
                "evidence": evid,
            })
        return {"toc_file": fname0, "toc_pages": toc_pages, "records": records, "toc_warnings": toc_warnings}
    finally:
        for d in docs:
            d.close()
