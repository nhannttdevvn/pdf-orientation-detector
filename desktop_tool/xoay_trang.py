#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MODULE PHÁT HIỆN & SỬA TRANG BỊ XOAY (0° / 90° / 180° / 270°)
Chuyển nguyên thuật toán từ dự án pdf-orientation-detector (`src/pdf_inspector/`):
- detectors/orientation/detector.py   -> OrientationDetector (Tesseract OSD 3 lần thử)
- processing/image_preprocessor.py    -> Tiền xử lý OpenCV & trích xuất dải dòng chữ
- services/orientation_service.py     -> Quy tắc phân loại NORMAL / ROTATED / REVIEW
- services/correction_service.py      -> Sửa xoay bằng metadata /Rotate (không rasterize)
Module độc lập, không phụ thuộc package `pdf_inspector`.

Nguyên tắc an toàn (giữ nguyên từ bản gốc):
- Không bao giờ tự suy đoán trang là 0° khi OCR lỗi -> trả về UNKNOWN -> REVIEW.
- Không bao giờ tự xoay trang REVIEW.
- Không bao giờ ghi đè file PDF gốc.
================================================================================
"""

import os
import re
import shutil

import cv2
import numpy as np
import pymupdf

try:
    import pytesseract
    from pytesseract import Output, TesseractError, TesseractNotFoundError
except ImportError:
    pytesseract = None

from app_paths import resource_dir, user_dir

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# Thư mục Tesseract đóng gói kèm tool: trong gói build (resource_dir) hoặc người dùng tự chép cạnh file chạy (user_dir)
BUNDLED_TESSERACT_DIRS = [os.path.join(resource_dir(), "Tesseract-OCR"), os.path.join(user_dir(), "Tesseract-OCR")]
BUNDLED_TESSERACT_DIR = BUNDLED_TESSERACT_DIRS[0]

# -----------------------------------------------------------------------------
# CẤU HÌNH (giống `src/pdf_inspector/core/config.py`)
# -----------------------------------------------------------------------------
RENDER_DPI = 160
ORIENTATION_CONF_THRESHOLD = 10.0
MAX_OCR_DIMENSION = 2400
MIN_OCR_DIMENSION = 800
MEDIAN_BLUR_KSIZE = 3
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
TESSERACT_OSD_CONFIG = "--psm 0"
VALID_ANGLES = (0, 90, 180, 270)

DETECTION_UNKNOWN = "UNKNOWN"
STATUS_NORMAL = "NORMAL"
STATUS_ROTATED = "ROTATED"
STATUS_REVIEW = "REVIEW"

COMMON_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Tool\Tesseract-OCR\tesseract.exe",
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/local/bin/tesseract",
    "/usr/bin/tesseract",
)


def find_tesseract_binary():
    """
    Tìm Tesseract theo thứ tự: bản đóng gói kèm tool (Tesseract-OCR/) -> biến môi trường TESSERACT_CMD
    -> PATH -> các đường dẫn cài đặt phổ biến (Windows/macOS/Linux).
    Đặt biến môi trường KTHS_NO_TESSERACT=1 để bỏ qua Tesseract (dùng RapidOCR), ví dụ khi Tesseract trên máy bị lỗi.
    """
    if os.getenv("KTHS_NO_TESSERACT", "").strip() in ("1", "true", "yes"):
        return None
    for tess_dir in BUNDLED_TESSERACT_DIRS:
        for name in ("tesseract.exe", "tesseract", os.path.join("bin", "tesseract")):
            bundled = os.path.join(tess_dir, name)
            if os.path.isfile(bundled):
                tessdata = os.path.join(tess_dir, "tessdata")
                if os.path.isdir(tessdata) and not os.getenv("TESSDATA_PREFIX"):
                    os.environ["TESSDATA_PREFIX"] = tessdata
                return bundled
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and os.path.isfile(env_cmd):
        return env_cmd
    in_path = shutil.which("tesseract")
    if in_path:
        return in_path
    for candidate in COMMON_TESSERACT_PATHS:
        if os.path.isfile(candidate):
            return candidate
    return None


def normalize_angle(angle):
    """Chuẩn hóa góc xoay về 0 / 90 / 180 / 270 (ném ValueError nếu không hợp lệ)."""
    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        raise ValueError(f"Góc xoay không hợp lệ: {angle!r}")
    normalized = int(round(angle)) % 360
    if normalized not in VALID_ANGLES:
        raise ValueError(f"Góc xoay {angle} (chuẩn hóa: {normalized}) không thuộc {VALID_ANGLES}.")
    return normalized


def rotate_image_cw(img, angle_cw):
    """Xoay ảnh theo chiều kim đồng hồ đúng góc cần sửa (0 / 90 / 180 / 270)."""
    angle_cw = normalize_angle(angle_cw)
    if angle_cw == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle_cw == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle_cw == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# -----------------------------------------------------------------------------
# TIỀN XỬ LÝ ẢNH (giống `src/pdf_inspector/processing/image_preprocessor.py`)
# -----------------------------------------------------------------------------
def to_grayscale(image):
    """Chuyển ảnh đầu vào (GRAY / RGB / RGBA) sang ảnh xám 2D uint8."""
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc None.")
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0].astype(np.uint8, copy=False)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Định dạng mảng ảnh không được hỗ trợ: shape={image.shape}")


def normalize_dimensions(gray, max_dim=MAX_OCR_DIMENSION, min_dim=MIN_OCR_DIMENSION):
    """Thu nhỏ (INTER_AREA) nếu cạnh dài > max_dim, phóng to (INTER_CUBIC) nếu cạnh dài < min_dim."""
    height, width = gray.shape[:2]
    longest_side = max(height, width)
    if longest_side == 0:
        return gray
    if longest_side > max_dim:
        scale = max_dim / float(longest_side)
        return cv2.resize(gray, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                          interpolation=cv2.INTER_AREA)
    if longest_side < min_dim:
        scale = min_dim / float(longest_side)
        return cv2.resize(gray, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                          interpolation=cv2.INTER_CUBIC)
    return gray


def prepare_original_grayscale(image, max_dim=MAX_OCR_DIMENSION, min_dim=MIN_OCR_DIMENSION):
    """Phương án 1: ảnh xám gốc đã chuẩn hóa kích thước."""
    return normalize_dimensions(to_grayscale(image), max_dim=max_dim, min_dim=min_dim)


def preprocess_for_ocr(image, ksize=MEDIAN_BLUR_KSIZE, max_dim=MAX_OCR_DIMENSION, min_dim=MIN_OCR_DIMENSION):
    """Phương án 2: ảnh xám -> chuẩn hóa kích thước -> Median Blur (giữ dấu tiếng Việt) -> CLAHE."""
    normalized = normalize_dimensions(to_grayscale(image), max_dim=max_dim, min_dim=min_dim)
    if ksize > 1:
        if ksize % 2 == 0:
            ksize += 1
        normalized = cv2.medianBlur(normalized, ksize)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(normalized)


def extract_text_line_canvas(image, max_strips=10):
    """
    Phân tích hình thái học dòng chữ:
    1. Xác định trục dòng chữ chính: ngang -> ứng viên (0, 180); dọc -> ứng viên (90, 270).
    2. Cắt các dải dòng chữ rõ nhất, ghép thành một ảnh gọn để OCR.
    Trả về (canvas, candidates) hoặc (None, candidates) nếu trang không đủ chữ.
    """
    gray = to_grayscale(image)
    h, w = gray.shape[:2]
    if h < 64 or w < 64:
        return None, (0, 180)

    # Bỏ 5% viền ngoài (bóng đen mép trang, lỗ ghim bìa hồ sơ)
    roi = gray[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]
    if roi.size == 0 or float(np.std(roi)) < 1.5:
        return None, (0, 180)

    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    h_conn = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, hk)
    v_conn = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, vk)

    _, _, h_stats, _ = cv2.connectedComponentsWithStats(h_conn)
    _, _, v_stats, _ = cv2.connectedComponentsWithStats(v_conn)

    h_score = sum(int(s[4]) for s in h_stats[1:] if s[2] > s[3] * 2.2 and 8 <= s[3] <= 85 and s[2] >= 40)
    v_score = sum(int(s[4]) for s in v_stats[1:] if s[3] > s[2] * 2.2 and 8 <= s[2] <= 85 and s[3] >= 40)

    is_horizontal = h_score >= v_score
    candidates = (0, 180) if is_horizontal else (90, 270)
    stats = h_stats if is_horizontal else v_stats

    boxes = []
    for s in stats[1:]:
        x, y, cw, ch, area = int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4])
        if is_horizontal:
            if cw >= 45 and 8 <= ch <= 80 and cw > ch * 2.2:
                boxes.append((area, x, y, cw, ch))
        else:
            if ch >= 45 and 8 <= cw <= 80 and ch > cw * 2.2:
                boxes.append((area, x, y, cw, ch))
    if not boxes:
        return None, candidates

    boxes.sort(key=lambda item: item[0], reverse=True)
    strips = []
    for _, x, y, cw, ch in boxes[:max_strips]:
        pad = 6
        strip = roi[max(0, y - pad):min(roi.shape[0], y + ch + pad), max(0, x - pad):min(roi.shape[1], x + cw + pad)]
        if strip.size > 0:
            strips.append(strip)
    if not strips:
        return None, candidates

    if is_horizontal:
        canvas = np.full((sum(s.shape[0] + 10 for s in strips) + 20, max(s.shape[1] for s in strips) + 20), 255, dtype=np.uint8)
        cy = 10
        for s in strips:
            sh, sw = s.shape[:2]
            canvas[cy:cy + sh, 10:10 + sw] = s
            cy += sh + 10
    else:
        canvas = np.full((max(s.shape[0] for s in strips) + 20, sum(s.shape[1] + 10 for s in strips) + 20), 255, dtype=np.uint8)
        cx = 10
        for s in strips:
            sh, sw = s.shape[:2]
            canvas[10:10 + sh, cx:cx + sw] = s
            cx += sw + 10
    return canvas, candidates


# -----------------------------------------------------------------------------
# NHẬN DIỆN HƯỚNG TRANG (giống `src/pdf_inspector/detectors/orientation/detector.py`)
# -----------------------------------------------------------------------------
def _attempt(method, status, orientation=None, rotate=None, orientation_conf=None,
             script=None, script_conf=None, error=None):
    return {
        'method': method, 'status': status, 'orientation': orientation, 'rotate': rotate,
        'orientation_conf': orientation_conf, 'script': script, 'script_conf': script_conf, 'error': error,
    }


def parse_osd_text(raw_osd):
    """Phân tích chuỗi OSD thô của Tesseract (`--psm 0`) khi output DICT gặp định dạng lạ."""
    if not raw_osd or not raw_osd.strip():
        raise ValueError("Kết quả OSD rỗng.")
    patterns = {
        "orientation": r"Orientation in degrees:\s*(\d+)",
        "rotate": r"Rotate:\s*(\d+)",
        "orientation_conf": r"Orientation confidence:\s*([+-]?\d+(?:\.\d+)?)",
        "script": r"Script:\s*([^\r\n]+)",
        "script_conf": r"Script confidence:\s*([+-]?\d+(?:\.\d+)?)",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, raw_osd, flags=re.IGNORECASE)
        if match:
            val_str = match.group(1).strip()
            if key in ("orientation", "rotate"):
                result[key] = int(val_str)
            elif key in ("orientation_conf", "script_conf"):
                result[key] = float(val_str)
            else:
                result[key] = val_str
    if "orientation" not in result or "rotate" not in result:
        raise ValueError(f"Không tìm thấy trường orientation/rotate trong OSD: {raw_osd!r}")
    return result


def tesseract_has_osd():
    """Tesseract có gói `osd` (bắt buộc cho --psm 0) hay không. Thiếu gói này thì mọi trang OSD đều lỗi."""
    try:
        return "osd" in pytesseract.get_languages(config="")
    except Exception:
        return False


class RapidOCRDirectionChecker:
    """
    Phương án dự phòng khi máy không có Tesseract: dùng RapidOCR (thư viện bắt buộc sẵn có của tool).
    1. Hình thái học xác định trục dòng chữ -> ứng viên (0°, 180°) hoặc (90°, 270°)  [extract_text_line_canvas].
    2. Xoay dải dòng chữ theo từng ứng viên, đọc bằng RapidOCR với bộ tự xoay chữ (angle cls) TẮT.
    3. Chiều đúng đọc được nhiều ký tự với điểm cao (~0.9), chiều sai chỉ đọc ra rác (~0.2).
       Chênh lệch không đủ rõ -> UNKNOWN -> REVIEW (không tự xoay).
    """

    METHOD = "rapidocr_line_direction"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr = RapidOCR()
        for attr in ("use_angle_cls", "use_cls"):
            if hasattr(self.ocr, attr):
                setattr(self.ocr, attr, False)

    def _score(self, img):
        res, _ = self.ocr(img, use_cls=False, text_score=0.0)
        total = 0.0
        chars = 0
        for r in res or []:
            n = sum(ch.isalnum() for ch in str(r[1]))
            if n >= 2:
                total += float(r[2]) * n
                chars += n
        return {"chars": chars, "total": total, "avg": (total / chars) if chars else 0.0}

    def detect(self, image):
        try:
            canvas, candidates = extract_text_line_canvas(image, max_strips=10)
            if canvas is None or canvas.size == 0:
                return _attempt(self.METHOD, DETECTION_UNKNOWN,
                                error="Trang không có đủ dòng văn bản để xác định hướng.")
            scores = {rot: self._score(rotate_image_cw(canvas, rot)) for rot in candidates}
            rot_a, rot_b = candidates
            best_rot, other_rot = (rot_a, rot_b) if scores[rot_a]["total"] >= scores[rot_b]["total"] else (rot_b, rot_a)
            best, other = scores[best_rot], scores[other_rot]

            if best["chars"] < 20 or best["avg"] < 0.60:
                return _attempt(self.METHOD, DETECTION_UNKNOWN, error=(
                    f"RapidOCR đọc được quá ít chữ ({best['chars']} ký tự, điểm TB {best['avg']:.2f}) "
                    f"để kết luận hướng an toàn."))
            ratio = best["total"] / max(1.0, other["total"])
            avg_diff = best["avg"] - other["avg"]
            if ratio < 2.0 or avg_diff < 0.30:
                return _attempt(self.METHOD, DETECTION_UNKNOWN,
                                error="Chênh lệch giữa 2 chiều đối diện không đủ rõ ràng (RapidOCR).")
            # Quy đổi về thang tương đương độ tin cậy OSD (ngưỡng 10) để dùng chung quy tắc phân loại
            return _attempt(self.METHOD, "OK", orientation=(360 - best_rot) % 360, rotate=best_rot,
                            orientation_conf=round(min(35.0, avg_diff * 40.0), 2),
                            script_conf=round(best["avg"], 2))
        except Exception as exc:
            return _attempt(self.METHOD, DETECTION_UNKNOWN, error=f"Lỗi kiểm tra hướng bằng RapidOCR: {exc}")


class OrientationDetector:
    """
    Phát hiện hướng trang. Chọn bộ máy theo thứ tự ưu tiên:
    A. Tesseract OSD (`--psm 0`) — thuật toán gốc đã kiểm chứng, tối đa 3 lần thử:
       1. OSD trên ảnh xám gốc.
       2. Nếu UNKNOWN / độ tin cậy thấp: OSD trên ảnh tiền xử lý (Median Blur + CLAHE).
       3. Nếu vẫn chưa chắc: kiểm định theo dải dòng chữ (Otsu + Morph Close + OCR --psm 6 từng góc).
    B. RapidOCR (RapidOCRDirectionChecker) — khi máy không có Tesseract hoặc thiếu gói osd.
    """

    def __init__(self, tesseract_cmd=None, conf_threshold=ORIENTATION_CONF_THRESHOLD, retry_with_preprocessed=True,
                 allow_rapidocr_fallback=True):
        self.conf_threshold = float(conf_threshold)
        self.retry_with_preprocessed = bool(retry_with_preprocessed)
        self.tesseract_cmd = tesseract_cmd or find_tesseract_binary()
        self.engine = None
        self.engine_note = ""
        self.rapid_checker = None

        if pytesseract is None:
            self.engine_note = "Chưa cài thư viện pytesseract. "
        elif not self.tesseract_cmd or not os.path.isfile(self.tesseract_cmd):
            self.engine_note = "Không tìm thấy Tesseract OCR trên máy. "
        else:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            if tesseract_has_osd():
                self.engine = "tesseract"
                cmd_norm = os.path.normcase(os.path.abspath(self.tesseract_cmd))
                bundled = any(cmd_norm.startswith(os.path.normcase(d)) for d in BUNDLED_TESSERACT_DIRS)
                self.engine_note = f"Tesseract OSD ({'đóng gói kèm tool' if bundled else self.tesseract_cmd})"
            else:
                self.engine_note = f"Tesseract tại {self.tesseract_cmd} thiếu gói 'osd'. "

        if self.engine is None and allow_rapidocr_fallback:
            try:
                self.rapid_checker = RapidOCRDirectionChecker()
                self.engine = "rapidocr"
                self.engine_note += "RapidOCR (dự phòng, không cần Tesseract)"
            except Exception as exc:
                self.engine_note += f"Không khởi tạo được RapidOCR: {exc}"

    @property
    def is_available(self):
        """Có ít nhất một bộ máy (Tesseract có osd, hoặc RapidOCR) để kiểm tra hướng trang hay không."""
        return self.engine is not None

    def _run_single_osd(self, image, method_name):
        """Chạy Tesseract OSD trên một ảnh. Không bao giờ tự suy đoán 0° nếu OCR lỗi."""
        if image is None or image.size == 0:
            return _attempt(method_name, DETECTION_UNKNOWN, error="Ảnh đầu vào rỗng.")

        std_val = float(np.std(image))
        if std_val < 1.5:
            return _attempt(method_name, DETECTION_UNKNOWN,
                            error=f"Trang trắng hoặc độ tương phản quá thấp (std={std_val:.2f}).")

        try:
            osd_data = pytesseract.image_to_osd(image, config=TESSERACT_OSD_CONFIG, output_type=Output.DICT)
            if not isinstance(osd_data, dict) or "rotate" not in osd_data:
                raw_text = pytesseract.image_to_osd(image, config=TESSERACT_OSD_CONFIG, output_type=Output.STRING)
                osd_data = parse_osd_text(raw_text)

            raw_orientation = osd_data.get("orientation")
            raw_rotate = osd_data.get("rotate")
            if raw_orientation is None or raw_rotate is None:
                return _attempt(method_name, DETECTION_UNKNOWN,
                                error="Tesseract OSD không trả về giá trị orientation hoặc rotate.")

            raw_orient_conf = osd_data.get("orientation_conf", 0.0)
            script_val = osd_data.get("script")
            raw_script_conf = osd_data.get("script_conf")
            return _attempt(
                method_name, "OK",
                orientation=normalize_angle(int(raw_orientation)),
                rotate=normalize_angle(int(raw_rotate)),
                orientation_conf=round(float(raw_orient_conf), 2) if raw_orient_conf is not None else 0.0,
                script=str(script_val).strip() if script_val is not None else None,
                script_conf=round(float(raw_script_conf), 2) if raw_script_conf is not None else None,
            )
        except TesseractNotFoundError as exc:
            return _attempt(method_name, DETECTION_UNKNOWN,
                            error=f"Không tìm thấy bộ thực thi Tesseract OCR trên hệ thống. Chi tiết: {exc}")
        except TesseractError as exc:
            clean_err = str(exc).strip().replace("\n", " ")
            return _attempt(method_name, DETECTION_UNKNOWN, error=f"Tesseract OSD không thể nhận diện: {clean_err}")
        except Exception as exc:
            clean_err = str(exc).strip().replace("\n", " ")
            return _attempt(method_name, DETECTION_UNKNOWN, error=f"Lỗi xử lý OSD: {clean_err}")

    def _run_strip_ocr_verification(self, image, method_name="tesseract_osd_preprocessed"):
        """
        Kiểm định theo dải dòng chữ cho trang A3 2-up, trang nhiều gạch chân hoặc trang ít chữ.
        Nếu nhận diện < 3 từ hoặc chênh lệch giữa 2 chiều đối diện không rõ ràng -> UNKNOWN (đưa vào REVIEW).
        """
        try:
            canvas, candidates = extract_text_line_canvas(image, max_strips=10)
            if canvas is None or canvas.size == 0:
                return _attempt(method_name, DETECTION_UNKNOWN,
                                error="Trang không có đủ dòng văn bản để xác định hướng.")

            scores = {}
            for rot in candidates:
                if rot == 0:
                    test_img = canvas
                elif rot == 90:
                    test_img = np.rot90(canvas, k=-1)
                elif rot == 180:
                    test_img = np.rot90(canvas, k=2)
                elif rot == 270:
                    test_img = np.rot90(canvas, k=1)
                else:
                    continue

                upscaled = cv2.resize(test_img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                data = pytesseract.image_to_data(upscaled, config="--psm 6", output_type=Output.DICT)
                valid_confs = [
                    float(c)
                    for c, txt in zip(data.get("conf", []), data.get("text", []))
                    if float(c) > 0 and len(str(txt).strip()) >= 2 and any(ch.isalpha() for ch in str(txt))
                ]
                scores[rot] = {
                    "count": float(len(valid_confs)),
                    "avg_conf": float(np.mean(valid_confs)) if valid_confs else 0.0,
                    "total_conf": float(np.sum(valid_confs)) if valid_confs else 0.0,
                }

            rot_a, rot_b = candidates[0], candidates[1]
            info_a, info_b = scores[rot_a], scores[rot_b]
            if info_a["total_conf"] >= info_b["total_conf"]:
                best_rot, best_info, other_info = rot_a, info_a, info_b
            else:
                best_rot, best_info, other_info = rot_b, info_b, info_a

            if best_info["count"] < 3 or best_info["avg_conf"] < 60.0:
                return _attempt(method_name, DETECTION_UNKNOWN, error=(
                    f"Số lượng từ nhận diện được quá ít ({int(best_info['count'])} từ, "
                    f"avg_conf={best_info['avg_conf']:.1f}%) để kết luận hướng an toàn."))

            ratio = best_info["total_conf"] / max(1.0, other_info["total_conf"])
            conf_diff = max(0.0, best_info["avg_conf"] - other_info["avg_conf"])
            if ratio < 1.35 or conf_diff < 15.0:
                return _attempt(method_name, DETECTION_UNKNOWN,
                                error="Chênh lệch độ tin cậy giữa 2 chiều đối diện không đủ rõ ràng.")

            return _attempt(
                method_name, "OK",
                orientation=(360 - best_rot) % 360,
                rotate=best_rot,
                orientation_conf=round(min(35.0, conf_diff * 0.4), 2),
                script="Latin",
                script_conf=round(best_info["avg_conf"], 2),
            )
        except Exception as exc:
            return _attempt(method_name, DETECTION_UNKNOWN, error=f"Lỗi kiểm định dải dòng văn bản: {exc}")

    @staticmethod
    def _result(page_number, attempt, existing_rot, attempts, has_conflict=False, error=None):
        return {
            'page_number': page_number,
            'status': attempt['status'],
            'orientation': attempt['orientation'],
            'rotate': attempt['rotate'],
            'orientation_conf': attempt['orientation_conf'],
            'script': attempt['script'],
            'existing_pdf_rotation': existing_rot,
            'detection_method': attempt['method'],
            'has_conflict': has_conflict,
            'error': error,
            'attempts': attempts,
        }

    def detect_page(self, image, page_number=1, existing_pdf_rotation=0):
        """
        Phát hiện hướng một trang. So sánh kết quả các lần thử mà KHÔNG cộng hay lấy trung bình độ tin cậy.
        Trả về dict: status (OK/UNKNOWN), rotate (góc cần xoay theo chiều kim đồng hồ), orientation_conf,
        has_conflict, error, attempts.
        """
        threshold = self.conf_threshold
        try:
            norm_existing_rot = normalize_angle(existing_pdf_rotation)
        except ValueError:
            norm_existing_rot = 0
        attempts = []

        if not self.is_available:
            att = _attempt("none", DETECTION_UNKNOWN,
                           error="Không có Tesseract (kèm gói osd) lẫn RapidOCR nên không kiểm tra được hướng trang.")
            return self._result(page_number, att, norm_existing_rot, [att], error=att['error'])

        if self.engine == "rapidocr":
            att = self.rapid_checker.detect(image)
            return self._result(page_number, att, norm_existing_rot, [att], error=att['error'])

        # Lần thử 1: Ảnh xám gốc
        try:
            attempt1 = self._run_single_osd(prepare_original_grayscale(image), "tesseract_osd")
        except Exception as exc:
            attempt1 = _attempt("tesseract_osd", DETECTION_UNKNOWN, error=f"Lỗi chuẩn bị ảnh gốc: {exc}")
        attempts.append(attempt1)

        need_retry = self.retry_with_preprocessed and (
            attempt1['status'] == DETECTION_UNKNOWN
            or attempt1['orientation_conf'] is None
            or attempt1['orientation_conf'] < threshold
        )
        if not need_retry:
            return self._result(page_number, attempt1, norm_existing_rot, attempts, error=attempt1['error'])

        # Lần thử 2: Ảnh tiền xử lý, rồi kiểm định dải dòng chữ nếu vẫn chưa đủ tin cậy
        try:
            attempt2 = self._run_single_osd(preprocess_for_ocr(image), "tesseract_osd_preprocessed")
            if (attempt2['status'] == DETECTION_UNKNOWN
                    or attempt2['orientation_conf'] is None
                    or attempt2['orientation_conf'] < threshold):
                strip_attempt = self._run_strip_ocr_verification(image)
                if strip_attempt['status'] == "OK" or attempt2['status'] == DETECTION_UNKNOWN:
                    attempt2 = strip_attempt
        except Exception as exc:
            attempt2 = _attempt("tesseract_osd_preprocessed", DETECTION_UNKNOWN, error=f"Lỗi tiền xử lý ảnh: {exc}")
        attempts.append(attempt2)

        s1, s2 = attempt1['status'], attempt2['status']
        if s1 == DETECTION_UNKNOWN and s2 == DETECTION_UNKNOWN:
            err = attempt1['error'] or attempt2['error'] or "OCR không nhận diện được hướng."
            unknown = _attempt(attempt1['method'], DETECTION_UNKNOWN)
            return self._result(page_number, unknown, norm_existing_rot, attempts, error=err)
        if s1 == DETECTION_UNKNOWN and s2 == "OK":
            return self._result(page_number, attempt2, norm_existing_rot, attempts)
        if s1 == "OK" and s2 == DETECTION_UNKNOWN:
            return self._result(page_number, attempt1, norm_existing_rot, attempts, error=attempt1['error'])

        # Cả hai lần thử đều OK
        conf1 = attempt1['orientation_conf'] or 0.0
        conf2 = attempt2['orientation_conf'] or 0.0
        if attempt1['rotate'] != attempt2['rotate']:
            min_meaningful_conf = max(3.0, threshold * 0.5)
            if conf1 < min_meaningful_conf and conf2 >= threshold:
                return self._result(page_number, attempt2, norm_existing_rot, attempts)
            if conf2 < min_meaningful_conf and conf1 >= threshold:
                return self._result(page_number, attempt1, norm_existing_rot, attempts)
            conflict_msg = (
                f"Mâu thuẫn kết quả giữa ảnh gốc (rotate={attempt1['rotate']}°, conf={attempt1['orientation_conf']}) "
                f"và ảnh tiền xử lý (rotate={attempt2['rotate']}°, conf={attempt2['orientation_conf']})."
            )
            chosen = attempt2 if conf2 > conf1 else attempt1
            return self._result(page_number, chosen, norm_existing_rot, attempts, has_conflict=True, error=conflict_msg)

        chosen = attempt2 if conf2 > conf1 else attempt1
        return self._result(page_number, chosen, norm_existing_rot, attempts)


# -----------------------------------------------------------------------------
# QUY TẮC PHÂN LOẠI (giống `UnifiedQualityEngine.evaluate_page`)
# -----------------------------------------------------------------------------
def classify_orientation(detection, conf_threshold=ORIENTATION_CONF_THRESHOLD):
    """
    - REVIEW  : UNKNOWN, hoặc độ tin cậy < ngưỡng, hoặc mâu thuẫn giữa các lần thử (KHÔNG tự xoay).
    - NORMAL  : góc cần xoay = 0°.
    - ROTATED : góc cần xoay 90° / 180° / 270° (được phép sửa).
    Trả về dict: status, correction_angle, reason.
    """
    is_unknown = (
        detection['status'] == DETECTION_UNKNOWN
        or detection['rotate'] is None
        or detection['orientation_conf'] is None
    )
    is_low_conf = not is_unknown and detection['orientation_conf'] < conf_threshold
    has_conflict = bool(detection.get('has_conflict'))

    if is_unknown or is_low_conf or has_conflict:
        reason = detection.get('error')
        if not reason:
            if is_unknown:
                reason = "OCR không nhận diện được hướng văn bản."
            elif has_conflict:
                reason = "Kết quả giữa các lần thử OCR mâu thuẫn nhau."
            else:
                reason = (f"Độ tin cậy OCR ({detection['orientation_conf']:.2f}) "
                          f"thấp hơn ngưỡng ({conf_threshold:.2f}).")
        return {'status': STATUS_REVIEW, 'correction_angle': 0, 'reason': reason}

    rot_angle = normalize_angle(detection['rotate'])
    if rot_angle == 0:
        return {'status': STATUS_NORMAL, 'correction_angle': 0, 'reason': None}
    return {'status': STATUS_ROTATED, 'correction_angle': rot_angle, 'reason': None}


# -----------------------------------------------------------------------------
# RENDER & KIỂM TRA MỘT TRANG
# -----------------------------------------------------------------------------
def render_page_gray(page, dpi=RENDER_DPI):
    """Render trang PDF sang ảnh xám (đã áp dụng /Rotate hiện có, giống cách người xem nhìn thấy)."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w).copy()
    del pix
    return img


def inspect_page_orientation(detector, page, page_number, conf_threshold=ORIENTATION_CONF_THRESHOLD):
    """Render + nhận diện + phân loại hướng của một trang. Trả về dict gộp kết quả nhận diện và quyết định."""
    try:
        existing_rot = normalize_angle(page.rotation)
    except ValueError:
        existing_rot = 0
    detection = detector.detect_page(render_page_gray(page), page_number=page_number, existing_pdf_rotation=existing_rot)
    decision = classify_orientation(detection, conf_threshold=conf_threshold)
    detection.update(decision)
    return detection


# -----------------------------------------------------------------------------
# SỬA XOAY & LƯU PDF MỚI (giống `RotationCorrector.apply_corrections`)
# -----------------------------------------------------------------------------
def has_digital_signature(doc):
    """Kiểm tra PDF có trường chữ ký số hay không (xoay trang có thể làm chữ ký trên file mới mất hiệu lực)."""
    try:
        sig_widget_type = getattr(pymupdf, "PDF_WIDGET_TYPE_SIGNATURE", 6)
        for page in doc:
            for widget in page.widgets() or []:
                if (getattr(widget, "field_type", None) == sig_widget_type
                        or getattr(widget, "field_type_string", "").lower() == "signature"):
                    return True
        catalog_xref = doc.pdf_catalog()
        if catalog_xref > 0:
            acroform = doc.xref_get_key(catalog_xref, "AcroForm")
            if acroform and "SigFlags" in str(acroform):
                return True
    except Exception:
        pass
    return False


def save_rotated_pdf(input_pdf_path, output_pdf_path, page_results):
    """
    Lưu bản PDF mới đã sửa xoay. Chỉ xoay trang ROTATED, KHÔNG xoay trang REVIEW.
    Dùng page.set_rotation() cập nhật /Rotate (modulo 360), không rasterize lại -> giữ nguyên chất lượng.
    Không bao giờ ghi đè file gốc.
    Trả về dict: output_path, corrected_pages, warnings.
    """
    in_abs = os.path.abspath(input_pdf_path)
    out_abs = os.path.abspath(output_pdf_path)
    if os.path.normcase(in_abs) == os.path.normcase(out_abs):
        raise ValueError(f"Không được phép ghi đè lên file PDF gốc ('{in_abs}').")

    to_fix = {r['page_number']: r['correction_angle'] for r in page_results
              if r.get('status') == STATUS_ROTATED and r.get('correction_angle')}
    warnings = []
    corrected_pages = []

    doc = pymupdf.open(in_abs)
    try:
        if has_digital_signature(doc):
            warnings.append("File PDF gốc có chữ ký số: bản đã xoay có thể làm chữ ký số không còn hợp lệ. "
                            "File gốc vẫn được giữ nguyên.")
        for idx in range(doc.page_count):
            page_number = idx + 1
            if page_number not in to_fix:
                continue
            page = doc.load_page(idx)
            try:
                current_rot = normalize_angle(page.rotation)
            except ValueError:
                current_rot = 0
            page.set_rotation((current_rot + normalize_angle(to_fix[page_number])) % 360)
            corrected_pages.append(page_number)

        os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
        doc.save(out_abs)
    finally:
        doc.close()

    return {'output_path': out_abs, 'corrected_pages': corrected_pages, 'warnings': warnings}
