"""
Xử lý ảnh trang PDF bằng OpenCV:
- Chuyển đổi sang grayscale.
- Chuẩn hóa kích thước ảnh phục vụ OCR.
- Giảm nhiễu nhẹ bằng Median Blur (giữ nguyên dấu tiếng Việt và ký tự nhỏ).
- Tăng tương phản cục bộ bằng CLAHE.
- Trích xuất dải dòng văn bản (text-line strips) và xác định trục văn bản chính (ngang/dọc)
  để hỗ trợ nhận diện chính xác cả tài liệu scan A3/2-up, hợp đồng có gạch chân/track-changes.
Cung cấp các phương án ảnh đầu vào cho OCR:
1. Ảnh grayscale gốc (original grayscale).
2. Ảnh sau tiền xử lý (preprocessed grayscale & text-line strip canvas).
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from pdf_inspector.core.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    MAX_OCR_DIMENSION,
    MEDIAN_BLUR_KSIZE,
    MIN_OCR_DIMENSION,
)
from pdf_inspector.core.logging_config import setup_logger

logger = setup_logger(__name__)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh đầu vào sang ảnh đa mức xám (grayscale 2D uint8).
    """
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


def normalize_dimensions(
    gray: np.ndarray,
    max_dim: int = MAX_OCR_DIMENSION,
    min_dim: int = MIN_OCR_DIMENSION,
) -> np.ndarray:
    """
    Chuẩn hóa kích thước ảnh phục vụ OCR:
    - Thu nhỏ bằng INTER_AREA nếu cạnh lớn nhất vượt quá `max_dim`.
    - Phóng to bằng INTER_CUBIC nếu cạnh lớn nhất nhỏ hơn `min_dim`.
    """
    height, width = gray.shape[:2]
    longest_side = max(height, width)
    if longest_side == 0:
        return gray

    if longest_side > max_dim:
        scale = max_dim / float(longest_side)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if longest_side < min_dim:
        scale = min_dim / float(longest_side)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return gray


def denoise_image(gray: np.ndarray, ksize: int = MEDIAN_BLUR_KSIZE) -> np.ndarray:
    """
    Giảm nhiễu hạt (salt-and-pepper noise) trên ảnh scan bằng Median Blur.
    Đảm bảo kernel size là số lẻ >= 3 và không quá lớn để bảo toàn dấu tiếng Việt.
    """
    if ksize <= 1:
        return gray
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(gray, ksize)


def enhance_contrast_clahe(
    gray: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Tăng tương phản cục bộ bằng thuật toán CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Giúp làm rõ chữ mờ hoặc trang scan bị bóng tối mà không làm cháy sáng nền.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def prepare_original_grayscale(
    image: np.ndarray,
    max_dim: int = MAX_OCR_DIMENSION,
    min_dim: int = MIN_OCR_DIMENSION,
) -> np.ndarray:
    """
    Phương án 1: Ảnh grayscale gốc (chỉ chuyển sang grayscale và chuẩn hóa kích thước).
    """
    gray = to_grayscale(image)
    return normalize_dimensions(gray, max_dim=max_dim, min_dim=min_dim)


def preprocess_for_ocr(
    image: np.ndarray,
    ksize: int = MEDIAN_BLUR_KSIZE,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
    max_dim: int = MAX_OCR_DIMENSION,
    min_dim: int = MIN_OCR_DIMENSION,
) -> np.ndarray:
    """
    Phương án 2: Ảnh sau tiền xử lý phục vụ thử lại (retry) khi ảnh gốc thất bại hoặc confidence thấp.
    Quy trình:
    1. Chuyển sang grayscale.
    2. Chuẩn hóa kích thước.
    3. Giảm nhiễu nhẹ bằng Median Blur (ksize=3).
    4. Tăng tương phản bằng CLAHE.
    """
    gray = to_grayscale(image)
    normalized = normalize_dimensions(gray, max_dim=max_dim, min_dim=min_dim)
    denoised = denoise_image(normalized, ksize=ksize)
    enhanced = enhance_contrast_clahe(
        denoised, clip_limit=clip_limit, tile_grid_size=tile_grid_size
    )
    return enhanced


def extract_text_line_canvas(
    image: np.ndarray,
    max_strips: int = 10,
) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    """
    Phân tích hình thái học (Morphological Text-Line Analysis) bằng OpenCV để:
    1. Xác định trục dòng văn bản chiếm ưu thế:
       - Dòng nằm ngang -> ứng viên góc xoay là `(0, 180)`.
       - Dòng nằm dọc   -> ứng viên góc xoay là `(90, 270)`.
    2. Trích xuất các dải dòng văn bản rõ nhất (bỏ qua lỗ bấm ghim, viền đen máy scan,
       khoảng trắng lớn trên trang A3/2-up) và ghép thành một ảnh compact phục vụ OCR.
    Trả về `(canvas_image, candidate_angles)` hoặc `(None, (0, 180))` nếu trang không đủ chữ.
    """
    gray = to_grayscale(image)
    h, w = gray.shape[:2]
    if h < 64 or w < 64:
        return None, (0, 180)

    # Cắt bỏ 5% viền ngoài để loại bỏ bóng đen mép trang và lỗ đục ghim bìa hồ sơ
    y0_roi, y1_roi = int(h * 0.05), int(h * 0.95)
    x0_roi, x1_roi = int(w * 0.05), int(w * 0.95)
    roi = gray[y0_roi:y1_roi, x0_roi:x1_roi]
    if roi.size == 0 or float(np.std(roi)) < 1.5:
        return None, (0, 180)

    # Nhị phân hóa Otsu đảo ngược (chữ trắng = 255, nền đen = 0)
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Liên kết các ký tự theo phương ngang (25x3) và phương dọc (3x25)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    h_conn = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, hk)
    v_conn = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, vk)

    _, _, h_stats, _ = cv2.connectedComponentsWithStats(h_conn)
    _, _, v_stats, _ = cv2.connectedComponentsWithStats(v_conn)

    h_score = sum(
        int(s[4])
        for s in h_stats[1:]
        if s[2] > s[3] * 2.2 and 8 <= s[3] <= 85 and s[2] >= 40
    )
    v_score = sum(
        int(s[4])
        for s in v_stats[1:]
        if s[3] > s[2] * 2.2 and 8 <= s[2] <= 85 and s[3] >= 40
    )

    is_horizontal = h_score >= v_score
    candidates: Tuple[int, int] = (0, 180) if is_horizontal else (90, 270)
    stats = h_stats if is_horizontal else v_stats

    boxes: List[Tuple[int, int, int, int, int]] = []
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
    selected = boxes[:max_strips]

    strips: List[np.ndarray] = []
    for _, x, y, cw, ch in selected:
        pad = 6
        sy0 = max(0, y - pad)
        sy1 = min(roi.shape[0], y + ch + pad)
        sx0 = max(0, x - pad)
        sx1 = min(roi.shape[1], x + cw + pad)
        strip = roi[sy0:sy1, sx0:sx1]
        if strip.size > 0:
            strips.append(strip)

    if not strips:
        return None, candidates

    if is_horizontal:
        max_w = max(s.shape[1] for s in strips) + 20
        total_h = sum(s.shape[0] + 10 for s in strips) + 20
        canvas = np.full((total_h, max_w), 255, dtype=np.uint8)
        cy = 10
        for s in strips:
            sh, sw = s.shape[:2]
            canvas[cy : cy + sh, 10 : 10 + sw] = s
            cy += sh + 10
    else:
        max_h = max(s.shape[0] for s in strips) + 20
        total_w = sum(s.shape[1] + 10 for s in strips) + 20
        canvas = np.full((max_h, total_w), 255, dtype=np.uint8)
        cx = 10
        for s in strips:
            sh, sw = s.shape[:2]
            canvas[10 : 10 + sh, cx : cx + sw] = s
            cx += sw + 10

    return canvas, candidates

