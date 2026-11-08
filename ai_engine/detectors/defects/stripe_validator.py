"""
Module kiểm định ứng viên vệt sọc (Stripe Validator — `src/pdf_inspector/detectors/defects/stripe_validator.py`).
Đánh giá toàn diện các đặc trưng hình học, quang học và cấu trúc bố cục trang của từng `LineCandidate`:
- Ước lượng vùng lề nội dung thực tế (`ContentMargins`: `content_left`, `content_right`, `content_width`, `page_width`)
  một cách bền vững (robust), loại trừ ứng viên đường thẳng dài và nhiễu mép giấy.
- Phân tích tỷ lệ chiều dài theo trang (`page_length_ratio`) và độ vượt lề chuẩn hóa (`left_extension_ratio`,
  `right_extension_ratio`) cho Horizontal Stripe Detection (Phase 4 — Part A).
- Kiểm định bằng chứng cấu trúc bảng biểu (`table_structure_evidence`, bao gồm cả bảng chỉ có đường ngang và khung
  biểu mẫu đóng góc vuông, đồng thời giữ nguyên khả năng bắt sọc scan đè xuyên qua bảng ra ngoài lề).
- Kiểm định đường hợp lệ thuộc Header, Quốc hiệu/Tiêu ngữ, Footer, Gạch chân tiêu đề, Đường phân cách mục và Chữ ký.
- Hệ thống chấm điểm có giải thích (`HorizontalStripeScore`) kèm `reason_codes` tường minh và phân loại 3 mức:
  `SCAN_STRIPE`, `LIKELY_DOCUMENT_LINE`, `REVIEW` (không coi heuristic score là xác suất đã hiệu chỉnh).
- Bảo toàn 100% thuật toán phát hiện sọc dọc (`VERTICAL_STRIPE`, `BAND_STRIPE`) và sọc nghiêng (`DIAGONAL_STRIPE`).
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ai_engine.core.config import (
    HORIZONTAL_CLASS_DOCUMENT_LINE,
    HORIZONTAL_CLASS_REVIEW,
    HORIZONTAL_CLASS_SCAN_STRIPE,
    HORIZONTAL_STRIPE_MARGIN_EXTENSION,
    HORIZONTAL_STRIPE_MIN_PAGE_RATIO,
    STRIPE_CONFIRM_THRESHOLD,
    STRIPE_MAX_THICKNESS_PX,
    STRIPE_MIN_CONTRAST,
    STRIPE_MIN_RELATIVE_LENGTH,
    STRIPE_REVIEW_THRESHOLD,
)
from ai_engine.detectors.defects.line_candidate_detector import (
    LineCandidate,
    PreparedStripeMaps,
    _angle_diff_deg,
)


@dataclass
class ContentMargins:
    """
    Kết quả ước lượng vùng nội dung chính theo phương ngang của trang PDF (Phase 4 — Part A3).
    """

    page_width: int
    page_height: int
    content_left: float
    content_right: float
    content_width: float
    reliable: bool
    method: str  # "text_blocks_and_projection" hoặc "fallback_proportional"
    component_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StripeFeatures:
    """Bộ đặc trưng định lượng thực tế của một ứng viên sọc trên trang PDF."""

    relative_length: float
    length_px: float
    thickness_px: float
    angle_deg: float
    continuity: float
    contrast: float
    background_intensity: float
    stripe_intensity: float
    intensity_std: float
    position_norm: float
    edge_distance_norm: float
    margin_penetration: float
    content_overlap: float
    endpoint_junctions: int
    perpendicular_crossings: int
    parallel_grid_siblings: int
    text_above_density: float
    repeat_page_count: int = 1
    detected_by: List[str] = field(default_factory=list)
    # Các đặc trưng chuyên sâu cho Horizontal Stripe Detection (Phase 4 — Part A)
    page_length_ratio: float = 0.0
    content_left_px: float = 0.0
    content_right_px: float = 0.0
    content_width_px: float = 0.0
    content_margins_reliable: bool = False
    left_extension_px: float = 0.0
    right_extension_px: float = 0.0
    left_extension_ratio: float = 0.0
    right_extension_ratio: float = 0.0
    table_structure_evidence: float = 0.0
    header_footer_evidence: float = 0.0
    text_below_density: float = 0.0
    text_cut_through_ratio: float = 0.0
    extends_past_table_bounds: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidatedStripe:
    """Kết quả sau khi kiểm định một ứng viên vệt sọc."""

    stripe_type: str  # "VERTICAL", "HORIZONTAL", "DIAGONAL", "BAND"
    defect_type: str  # "VERTICAL_STRIPE", "HORIZONTAL_STRIPE", "DIAGONAL_STRIPE", "BAND_STRIPE"
    status: str  # "CONFIRMED", "SUSPECTED", hoặc "REJECTED"
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) theo pixel
    endpoints: Tuple[float, float, float, float]
    length_px: float
    width_px: float
    relative_length: float
    angle_deg: float
    suspicion_score: float
    confidence: float
    severity: str  # "LOW", "MEDIUM", "HIGH"
    reason: str
    rejection_category: Optional[str] = None
    features: Optional[StripeFeatures] = None
    horizontal_classification: Optional[str] = None  # "SCAN_STRIPE", "LIKELY_DOCUMENT_LINE", "REVIEW"
    horizontal_score: Optional[float] = None
    reason_codes: List[str] = field(default_factory=list)


def estimate_content_margins(maps: PreparedStripeMaps) -> ContentMargins:
    """
    Ước lượng bền vững (robust) vùng lề nội dung chính `content_left`, `content_right`, `content_width`
    từ mặt nạ nhị phân của trang (`maps.otsu_binary`), loại trừ hoàn toàn:
    - Các đường thẳng ngang dài / ứng viên sọc ngang (`cw > 0.26 * w` và dẹt)
    - Các đường thẳng dọc dài / ứng viên sọc dọc (`ch > 0.12 * h` và mảnh)
    - Viền đen sát mép giấy (`1.8%` quanh khung trang) và nhiễu hạt bụi nhỏ (`area < 12`).

    Nhờ loại bỏ ứng viên đường thẳng trước khi gom cụm dòng chữ (text-line blocks) và tính phân vị kết hợp
    hồ sơ chiếu dọc của chữ, một vệt sọc dài chạy xuyên trang không bao giờ làm lệch `content_left` / `content_right`.
    Nếu trang trắng hoặc quá ít chữ không đủ độ tin cậy, trả về fallback tỷ lệ `10%..90%` kèm `reliable = False`.
    """
    h, w = maps.gray.shape[:2]
    edge_x = max(4, int(round(w * 0.018)))
    edge_y = max(4, int(round(h * 0.018)))

    fg = maps.otsu_binary.copy()
    fg[:edge_y, :] = 0
    fg[h - edge_y :, :] = 0
    fg[:, :edge_x] = 0
    fg[:, w - edge_x :] = 0

    # Loại trừ trực tiếp các pixel thuộc cấu trúc đường thẳng dài mạnh trên v_residual / h_residual
    # để chữ dính vào vệt sọc hoặc khung bảng lớn không kéo giãn biên chữ ra sát mép giấy
    strong_h_lines = cv2.morphologyEx(
        ((maps.h_residual >= 14).astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(45, int(round(w * 0.22))), 1)),
    )
    strong_v_lines = cv2.morphologyEx(
        ((maps.v_residual >= 14).astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(45, int(round(h * 0.14))))),
    )
    fg_no_lines = cv2.bitwise_and(
        fg, cv2.bitwise_not(cv2.bitwise_or(strong_h_lines, strong_v_lines))
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_no_lines)
    text_mask = np.zeros_like(fg_no_lines)
    valid_glyph_count = 0

    min_glyph_h = max(4, int(round(h * 0.0035)))
    max_glyph_h = max(30, int(round(h * 0.095)))
    min_glyph_w = max(2, int(round(w * 0.002)))
    max_glyph_w = max(40, int(round(w * 0.26)))

    for idx in range(1, num_labels):
        cx = int(stats[idx, cv2.CC_STAT_LEFT])
        cy = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])

        if area < 12:
            continue
        if ch < min_glyph_h or ch > max_glyph_h:
            continue
        if cw < min_glyph_w or cw > max_glyph_w:
            continue
        if cw > ch * 9.0 or ch > cw * 9.0:
            continue
        if cx <= edge_x or (cx + cw) >= (w - edge_x) or cy <= edge_y or (cy + ch) >= (h - edge_y):
            if area < 45:
                continue

        text_mask[labels == idx] = 255
        valid_glyph_count += 1

    if valid_glyph_count >= 12:
        # Gom các ký tự trên cùng hàng thành cụm từ / dòng văn bản để lấy biên trái/phải thực của dòng
        group_kw = max(18, int(round(w * 0.032)))
        group_kh = max(4, int(round(h * 0.006)))
        grouped = cv2.morphologyEx(
            text_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (group_kw, group_kh)),
        )
        n_blk, _, blk_stats, _ = cv2.connectedComponentsWithStats(grouped)
        block_x0s: List[float] = []
        block_x1s: List[float] = []
        strong_word_x0s: List[float] = []
        strong_word_x1s: List[float] = []
        min_blk_w = max(12, int(round(w * 0.015)))
        strong_blk_w = max(48, int(round(w * 0.042)))
        for b_idx in range(1, n_blk):
            bx = int(blk_stats[b_idx, cv2.CC_STAT_LEFT])
            bw = int(blk_stats[b_idx, cv2.CC_STAT_WIDTH])
            bh = int(blk_stats[b_idx, cv2.CC_STAT_HEIGHT])
            b_area = int(blk_stats[b_idx, cv2.CC_STAT_AREA])
            if bw >= min_blk_w and b_area >= 40:
                block_x0s.append(float(bx))
                block_x1s.append(float(bx + bw))
            # Cụm nhiều từ rõ ràng (ví dụ ngày tháng góc phải, mã biểu mẫu chân trang, tiêu ngữ) không thể là hạt bụi
            if bw >= strong_blk_w and bh >= max(6, int(round(h * 0.005))) and b_area >= 95:
                strong_word_x0s.append(float(bx))
                strong_word_x1s.append(float(bx + bw))

        # Kết hợp với hồ sơ chiếu dọc của mặt nạ chữ sạch để bắt chính xác mép lề ngay cả khi có nhiều cụm từ ngắn ở giữa dòng
        col_fg_px = np.sum(text_mask > 0, axis=0)
        # Làm mượt nhẹ cửa sổ 5px theo cột
        col_fg_smooth = np.convolve(col_fg_px.astype(np.float32), np.ones(5, dtype=np.float32) / 5.0, mode="same")
        min_col_support = max(8.0, float(h) * 0.0065)
        supported_cols = np.flatnonzero(col_fg_smooth >= min_col_support)

        estimates_left: List[float] = []
        estimates_right: List[float] = []

        if len(block_x0s) >= 4:
            estimates_left.append(float(np.percentile(block_x0s, 5)))
            estimates_right.append(float(np.percentile(block_x1s, 95)))
        if strong_word_x0s:
            estimates_left.append(float(min(strong_word_x0s)))
            estimates_right.append(float(max(strong_word_x1s)))
        if supported_cols.size >= 20:
            estimates_left.append(float(supported_cols[0]))
            estimates_right.append(float(supported_cols[-1]))

        if estimates_left and estimates_right:
            c_left = float(min(estimates_left))
            c_right = float(max(estimates_right))
            c_width = c_right - c_left
            if c_width >= float(w) * 0.22:
                return ContentMargins(
                    page_width=w,
                    page_height=h,
                    content_left=round(c_left, 2),
                    content_right=round(c_right, 2),
                    content_width=round(c_width, 2),
                    reliable=True,
                    method="text_blocks_and_projection",
                    component_count=valid_glyph_count,
                )

    # Fallback an toàn khi trang quá ít chữ / trang biểu mẫu trống / nhiễu nặng (ghi nhận rõ reliable=False)
    fb_left = round(float(w) * 0.10, 2)
    fb_right = round(float(w) * 0.90, 2)
    return ContentMargins(
        page_width=w,
        page_height=h,
        content_left=fb_left,
        content_right=fb_right,
        content_width=round(fb_right - fb_left, 2),
        reliable=False,
        method="fallback_proportional",
        component_count=valid_glyph_count,
    )


def _sample_diagonal_features(
    cand: LineCandidate,
    maps: PreparedStripeMaps,
    min_contrast: float,
) -> Tuple[float, float, float, float, float]:
    """
    Đo đạc `continuity`, `contrast`, `background_intensity`, `stripe_intensity`, `intensity_std`
    dọc theo đoạn thẳng nghiêng (`DIAGONAL`).
    """
    h, w = maps.gray.shape[:2]
    x1, y1, x2, y2 = cand.endpoints
    num_samples = max(40, int(round(cand.length_px)))
    xs = np.linspace(x1, x2, num_samples)
    ys = np.linspace(y1, y2, num_samples)

    ang_rad = math.radians(cand.angle_deg)
    nx = -math.sin(ang_rad)
    ny = math.cos(ang_rad)
    offset = max(6.0, cand.width_px * 0.8 + 5.0)

    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)

    # Lấy giá trị tối nhất trong lân cận vuông góc [-2..+2] px quanh đường tâm để bù sai số lượng tử hóa
    line_vals = maps.gray[yi, xi].astype(np.float32)
    for d in (-2.0, -1.0, 1.0, 2.0):
        dxi = np.clip(np.round(xs + d * nx).astype(int), 0, w - 1)
        dyi = np.clip(np.round(ys + d * ny).astype(int), 0, h - 1)
        line_vals = np.minimum(line_vals, maps.gray[dyi, dxi].astype(np.float32))

    # Lấy mẫu nền hai bên ở khoảng cách vuông góc +/- offset
    bg1_x = np.clip(np.round(xs + offset * nx).astype(int), 0, w - 1)
    bg1_y = np.clip(np.round(ys + offset * ny).astype(int), 0, h - 1)
    bg2_x = np.clip(np.round(xs - offset * nx).astype(int), 0, w - 1)
    bg2_y = np.clip(np.round(ys - offset * ny).astype(int), 0, h - 1)

    bg_vals = 0.5 * (
        maps.gray[bg1_y, bg1_x].astype(np.float32)
        + maps.gray[bg2_y, bg2_x].astype(np.float32)
    )
    diffs = np.maximum(0.0, bg_vals - line_vals)

    active_thresh = max(5.5, min_contrast * 0.75)
    active_mask = diffs >= active_thresh
    continuity = float(np.mean(active_mask))

    if np.any(active_mask):
        contrast = float(np.mean(diffs[active_mask]))
        stripe_int = float(np.mean(line_vals[active_mask]))
        bg_int = float(np.mean(bg_vals[active_mask]))
        int_std = float(np.std(line_vals[active_mask]))
    else:
        contrast = float(np.mean(diffs))
        stripe_int = float(np.mean(line_vals))
        bg_int = float(np.mean(bg_vals))
        int_std = float(np.std(line_vals))

    return continuity, contrast, bg_int, stripe_int, int_std


def _has_vertical_corner_at_x(
    maps: PreparedStripeMaps,
    x_target: int,
    yc: float,
    contrast: float,
) -> bool:
    """
    Kiểm tra xem đầu mút `x_target` của đoạn thẳng ngang tại tung độ `yc` có tạo thành góc vuông khép kín
    (góc chữ L ở cạnh trên/dưới của khung biểu mẫu/bảng hoặc giao chữ T ở hàng giữa bảng) với một đường dọc hay không.
    """
    h, w = maps.gray.shape[:2]
    yc_i = int(round(yc))
    sx0 = max(0, x_target - 7)
    sx1 = min(w, x_target + 8)
    if sx1 <= sx0:
        return False

    arm_len = max(22, int(round(h * 0.022)))
    v_thresh = max(9.0, min(28.0, contrast * 0.35))

    # Nhánh dọc đi lên trên yc (góc L cạnh dưới hoặc chữ T)
    up_y0 = max(0, yc_i - arm_len)
    up_y1 = max(0, yc_i - 2)
    has_up = False
    if up_y1 - up_y0 >= 14:
        up_res = np.max(maps.v_residual[up_y0:up_y1, sx0:sx1], axis=1)
        up_bin = np.max(maps.otsu_binary[up_y0:up_y1, sx0:sx1], axis=1) > 0
        if float(np.mean(up_res >= v_thresh)) >= 0.58 or float(np.mean(up_bin)) >= 0.78:
            has_up = True

    # Nhánh dọc đi xuống dưới yc (góc L cạnh trên hoặc chữ T)
    dn_y0 = min(h, yc_i + 2)
    dn_y1 = min(h, yc_i + arm_len)
    has_dn = False
    if dn_y1 - dn_y0 >= 14:
        dn_res = np.max(maps.v_residual[dn_y0:dn_y1, sx0:sx1], axis=1)
        dn_bin = np.max(maps.otsu_binary[dn_y0:dn_y1, sx0:sx1], axis=1) > 0
        if float(np.mean(dn_res >= v_thresh)) >= 0.58 or float(np.mean(dn_bin)) >= 0.78:
            has_dn = True

    return has_up or has_dn


def _extract_candidate_features(
    cand: LineCandidate,
    all_candidates: List[LineCandidate],
    maps: PreparedStripeMaps,
    min_contrast: float,
    repeat_page_count: int = 1,
    content_margins: Optional[ContentMargins] = None,
) -> StripeFeatures:
    """
    Trích xuất đầy đủ các đặc trưng của ứng viên `cand` trên trang hiện tại.
    """
    h, w = maps.gray.shape[:2]
    margins = content_margins or estimate_content_margins(maps)

    x0, y0, x1, y1 = cand.bbox
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))

    is_diagonal_type = cand.stripe_type == "DIAGONAL"
    is_vertical = (not is_diagonal_type) and (_angle_diff_deg(cand.angle_deg, 90.0) <= 5.0)
    is_horizontal = (not is_diagonal_type) and (_angle_diff_deg(cand.angle_deg, 0.0) <= 5.0)

    endpoint_junctions = 0
    perp_crossings = 0
    parallel_siblings = 0
    text_above_density = 0.0
    text_below_density = 0.0
    text_cut_through_ratio = 0.0
    page_length_ratio = 0.0
    left_ext_px = 0.0
    right_ext_px = 0.0
    left_ext_ratio = 0.0
    right_ext_ratio = 0.0
    table_evidence = 0.0
    header_footer_evidence = 0.0
    extends_past_table_bounds = False

    if is_vertical:
        roi_gray = maps.gray[y0:y1, x0:x1]
        roi_res = maps.v_residual[y0:y1, x0:x1]
        row_max_res = np.max(roi_res, axis=1).astype(np.float32)
        row_min_gray = np.min(roi_gray, axis=1).astype(np.float32)

        active_thresh = max(5.5, min_contrast * 0.75)
        active_rows = row_max_res >= active_thresh
        continuity = float(np.mean(active_rows))

        flank_pad = max(6, int(round(cand.width_px * 0.5)) + 4)
        flank_w = 12
        lx0 = max(0, x0 - flank_pad - flank_w)
        lx1 = max(0, x0 - flank_pad)
        rx0 = min(w, x1 + flank_pad)
        rx1 = min(w, x1 + flank_pad + flank_w)

        flank_pixels: List[np.ndarray] = []
        if lx1 > lx0:
            flank_pixels.append(maps.gray[y0:y1, lx0:lx1].reshape(-1))
        if rx1 > rx0:
            flank_pixels.append(maps.gray[y0:y1, rx0:rx1].reshape(-1))
        bg_int = (
            float(np.median(np.concatenate(flank_pixels)))
            if flank_pixels
            else float(np.median(maps.gray))
        )

        if np.any(active_rows):
            stripe_int = float(np.mean(row_min_gray[active_rows]))
            contrast = float(
                max(np.mean(row_max_res[active_rows]), max(0.0, bg_int - stripe_int))
            )
            s_vals = row_min_gray[active_rows]
            q25, q75 = np.percentile(s_vals, [25, 75])
            int_std = float((q75 - q25) * 0.7413)
        else:
            stripe_int = float(np.mean(row_min_gray))
            contrast = float(np.mean(row_max_res))
            int_std = float(np.std(row_min_gray))

        xc = 0.5 * (x0 + x1)
        pos_norm = xc / float(w)
        edge_dist_norm = min(xc / float(w), (w - xc) / float(w))

        top_margin_px = max(1.0, h * 0.07)
        bot_margin_px = max(1.0, h * 0.07)
        top_pen = max(0.0, (top_margin_px - y0) / top_margin_px)
        bot_pen = max(0.0, (y1 - (h - bot_margin_px)) / bot_margin_px)
        margin_penetration = min(1.0, 0.5 * (top_pen + bot_pen))

        if lx1 > lx0 and rx1 > rx0:
            left_fg = np.mean(maps.otsu_binary[y0:y1, lx0:lx1] > 0, axis=1) > 0.25
            right_fg = np.mean(maps.otsu_binary[y0:y1, rx0:rx1] > 0, axis=1) > 0.25
            content_overlap = float(np.mean(left_fg | right_fg))
        else:
            content_overlap = 0.0

        j_span_x0 = max(0, int(xc - 25))
        j_span_x1 = min(w, int(xc + 25))
        if j_span_x1 - j_span_x0 >= 16:
            top_y0, top_y1 = max(0, y0 - 5), min(h, y0 + 6)
            bot_y0, bot_y1 = max(0, y1 - 6), min(h, y1 + 5)
            top_h_line = np.max(maps.h_residual[top_y0:top_y1, j_span_x0:j_span_x1], axis=0)
            bot_h_line = np.max(maps.h_residual[bot_y0:bot_y1, j_span_x0:j_span_x1], axis=0)
            if float(np.mean(top_h_line >= max(10.0, contrast * 0.45))) >= 0.32:
                endpoint_junctions += 1
            if float(np.mean(bot_h_line >= max(10.0, contrast * 0.45))) >= 0.32:
                endpoint_junctions += 1

            cross_band = np.mean(
                maps.h_residual[y0:y1, j_span_x0:j_span_x1] >= max(10.0, contrast * 0.45),
                axis=1,
            )
            cross_rows = np.flatnonzero(cross_band >= 0.32)
            if cross_rows.size > 0:
                perp_crossings = 1 + int(np.sum(np.diff(cross_rows) > 8))

        for other in all_candidates:
            if other is cand:
                continue
            if _angle_diff_deg(other.angle_deg, 90.0) <= 5.0:
                oy0, oy1 = other.bbox[1], other.bbox[3]
                overlap_y = min(y1, oy1) - max(y0, oy0)
                max_len = max(1, max(y1 - y0, oy1 - oy0))
                if (
                    abs(oy0 - y0) <= max(28, int(h * 0.035))
                    and abs(oy1 - y1) <= max(28, int(h * 0.035))
                ) or (overlap_y / float(max_len) >= 0.78 and max_len < h * 0.85):
                    parallel_siblings += 1

    elif is_horizontal:
        roi_gray = maps.gray[y0:y1, x0:x1]
        roi_res = maps.h_residual[y0:y1, x0:x1]
        col_max_res = np.max(roi_res, axis=0).astype(np.float32)
        col_min_gray = np.min(roi_gray, axis=0).astype(np.float32)

        active_thresh = max(5.5, min_contrast * 0.75)
        active_cols = col_max_res >= active_thresh
        continuity = float(np.mean(active_cols))

        flank_pad = max(6, int(round(cand.width_px * 0.5)) + 4)
        flank_h = 12
        ty0 = max(0, y0 - flank_pad - flank_h)
        ty1 = max(0, y0 - flank_pad)
        by0 = min(h, y1 + flank_pad)
        by1 = min(h, y1 + flank_pad + flank_h)

        flank_pixels = []
        if ty1 > ty0:
            flank_pixels.append(maps.gray[ty0:ty1, x0:x1].reshape(-1))
        if by1 > by0:
            flank_pixels.append(maps.gray[by0:by1, x0:x1].reshape(-1))
        bg_int = (
            float(np.median(np.concatenate(flank_pixels)))
            if flank_pixels
            else float(np.median(maps.gray))
        )

        if np.any(active_cols):
            stripe_int = float(np.mean(col_min_gray[active_cols]))
            contrast = float(
                max(np.mean(col_max_res[active_cols]), max(0.0, bg_int - stripe_int))
            )
            s_vals = col_min_gray[active_cols]
            q25, q75 = np.percentile(s_vals, [25, 75])
            int_std = float((q75 - q25) * 0.7413)
        else:
            stripe_int = float(np.mean(col_min_gray))
            contrast = float(np.mean(col_max_res))
            int_std = float(np.std(col_min_gray))

        yc = 0.5 * (y0 + y1)
        pos_norm = yc / float(h)
        edge_dist_norm = min(yc / float(h), (h - yc) / float(h))

        # Phase 4 — Part A4: Tính toán chính xác tỷ lệ chiều dài theo trang và độ vượt lề nội dung
        ep_x1, _, ep_x2, _ = cand.endpoints
        seg_x_min = float(min(ep_x1, ep_x2))
        seg_x_max = float(max(ep_x1, ep_x2))
        line_length = abs(ep_x2 - ep_x1)
        page_length_ratio = min(1.0, line_length / float(max(1, w)))

        left_ext_px = max(0.0, margins.content_left - seg_x_min)
        right_ext_px = max(0.0, seg_x_max - margins.content_right)
        left_ext_ratio = left_ext_px / float(max(1, w))
        right_ext_ratio = right_ext_px / float(max(1, w))

        # Chuẩn hóa margin_penetration dựa trên vùng lề nội dung thực tế
        left_margin_span = max(1.0, margins.content_left)
        right_margin_span = max(1.0, float(w) - margins.content_right)
        l_pen = min(1.0, left_ext_px / left_margin_span)
        r_pen = min(1.0, right_ext_px / right_margin_span)
        margin_penetration = min(1.0, 0.5 * (l_pen + r_pen))

        # Đo mật độ chữ phía trên (cho đường gạch chân / tiêu ngữ) và phía dưới (cho header/footer)
        band_h_px = max(32, int(round(h * 0.032)))
        above_y0 = max(0, y0 - band_h_px)
        above_y1 = max(0, y0 - 1)
        if above_y1 > above_y0:
            text_above_density = float(np.mean(maps.otsu_binary[above_y0:above_y1, x0:x1] > 0))

        below_y0 = min(h, y1 + 1)
        below_y1 = min(h, y1 + band_h_px)
        if below_y1 > below_y0:
            text_below_density = float(np.mean(maps.otsu_binary[below_y0:below_y1, x0:x1] > 0))

        content_overlap = text_above_density

        # Đo tỷ lệ đường ngang cắt đè trực tiếp qua giữa thân các ký tự chữ (loại trừ giao điểm với kẻ dọc bảng)
        imm_up0, imm_up1 = max(0, y0 - 7), max(0, y0 - 1)
        imm_dn0, imm_dn1 = min(h, y1 + 1), min(h, y1 + 7)
        if imm_up1 > imm_up0 and imm_dn1 > imm_dn0 and (x1 - x0) >= 20:
            not_vline_col = np.max(maps.v_residual[imm_up0:imm_dn1, x0:x1], axis=0) < 12
            up_fg = np.any(maps.otsu_binary[imm_up0:imm_up1, x0:x1] > 0, axis=0) & not_vline_col
            dn_fg = np.any(maps.otsu_binary[imm_dn0:imm_dn1, x0:x1] > 0, axis=0) & not_vline_col
            text_cut_through_ratio = float(np.mean(up_fg & dn_fg))

        # Phase 4 — Part A6: Phân tích giao điểm góc vuông (L/T corners) & giao điểm cột dọc bảng biểu
        left_has_corner = _has_vertical_corner_at_x(maps, x0, yc, contrast)
        right_has_corner = _has_vertical_corner_at_x(maps, x1, yc, contrast)
        endpoint_junctions = int(left_has_corner) + int(right_has_corner)

        j_span_y0 = max(0, int(yc - 25))
        j_span_y1 = min(h, int(yc + 25))
        cross_x_coords: List[int] = []
        if j_span_y1 - j_span_y0 >= 16:
            v_thresh = max(10.0, min(30.0, contrast * 0.40))
            cross_band = np.mean(
                maps.v_residual[j_span_y0:j_span_y1, x0:x1] >= v_thresh,
                axis=0,
            )
            cross_cols = np.flatnonzero(cross_band >= 0.32)
            if cross_cols.size > 0:
                # Nhóm các cột giao cắt liền kề để lấy tọa độ x thực tế của từng đường kẻ dọc
                cur_grp = [int(cross_cols[0])]
                for col_i in cross_cols[1:]:
                    ci = int(col_i)
                    if ci - cur_grp[-1] <= 8:
                        cur_grp.append(ci)
                    else:
                        cross_x_coords.append(x0 + int(round(float(np.mean(cur_grp)))))
                        cur_grp = [ci]
                cross_x_coords.append(x0 + int(round(float(np.mean(cur_grp)))))
                perp_crossings = len(cross_x_coords)

        # Đếm số đường ngang song song có cùng căn lề trái/phải [x0, x1] (bắt cả bảng hiện đại KHÔNG có đường dọc!)
        aligned_tol = max(20, int(round(w * 0.022)))
        sibling_ys: List[float] = []
        for other in all_candidates:
            if other is cand:
                continue
            if _angle_diff_deg(other.angle_deg, 0.0) <= 5.0 and other.stripe_type != "DIAGONAL":
                ox0, ox1 = other.bbox[0], other.bbox[2]
                oyc = 0.5 * (other.bbox[1] + other.bbox[3])
                if abs(ox0 - x0) <= aligned_tol and abs(ox1 - x1) <= aligned_tol and abs(oyc - yc) >= 16.0:
                    parallel_siblings += 1
                    sibling_ys.append(oyc)

        # Kiểm tra xem ứng viên ngang có phải là sọc scan đè xuyên qua bảng rồi thò dài ra ngoài biên bảng không (Case 11!)
        protrude_thresh_px = max(24.0, float(w) * 0.028)
        if len(cross_x_coords) >= 2:
            first_vline_x = float(min(cross_x_coords))
            last_vline_x = float(max(cross_x_coords))
            if (
                not left_has_corner
                and not right_has_corner
                and ((first_vline_x - seg_x_min) >= protrude_thresh_px or (seg_x_max - last_vline_x) >= protrude_thresh_px)
            ):
                extends_past_table_bounds = True

        # Tính điểm bằng chứng cấu trúc bảng biểu (`table_structure_evidence` trong [0.0 .. 1.0])
        if extends_past_table_bounds:
            table_evidence = 0.0
        else:
            if left_has_corner and right_has_corner:
                # Cả hai đầu trái/phải đều đóng góc vuông L/T khép kín với cạnh dọc -> khung biểu mẫu/bảng chắc chắn (kể cả rộng 95%!)
                table_evidence = 0.92 if (perp_crossings >= 2 or parallel_siblings >= 1) else 0.82
            elif endpoint_junctions >= 1 and (perp_crossings >= 2 or parallel_siblings >= 1):
                table_evidence = 0.80
            elif (
                parallel_siblings >= 2
                and left_ext_ratio < 0.03
                and right_ext_ratio < 0.03
                and (text_above_density < 0.045 or (seg_x_max - seg_x_min) >= margins.content_width * 0.90)
            ):
                # Bảng hiện đại chỉ có các đường kẻ ngang song song cùng lề trái/phải
                table_evidence = 0.85
            elif parallel_siblings == 1 and perp_crossings >= 2 and left_ext_ratio < 0.03 and right_ext_ratio < 0.03:
                table_evidence = 0.78
            elif perp_crossings >= 4 and left_ext_ratio < 0.03 and right_ext_ratio < 0.03:
                table_evidence = 0.72
            elif parallel_siblings == 1 and left_ext_ratio < 0.025 and right_ext_ratio < 0.025 and cand.width_px <= 8.0:
                table_evidence = 0.55

        # Phase 4 — Part A7: Tính bằng chứng thuộc Header / Quốc hiệu / Footer / Tiêu đề / Chữ ký
        center_sym_err = abs(seg_x_min - (float(w) - seg_x_max)) / float(max(1, w))
        is_header_zone = pos_norm <= 0.18
        is_footer_zone = pos_norm >= 0.84
        is_signature_zone = 0.62 <= pos_norm < 0.94

        if is_header_zone and (text_above_density >= 0.015 or text_below_density >= 0.015 or center_sym_err <= 0.06):
            header_footer_evidence = max(header_footer_evidence, 0.82)
        if is_footer_zone and cand.width_px <= 12.0 and (
            text_below_density >= 0.006 or text_above_density >= 0.008 or center_sym_err <= 0.06
        ):
            header_footer_evidence = max(header_footer_evidence, 0.85)
        if (
            not is_header_zone
            and not is_footer_zone
            and text_above_density >= 0.024
            and text_cut_through_ratio < 0.06
            and cand.width_px <= 16.0
        ):
            # Đường gạch chân tiêu đề / gạch chân dòng chữ
            header_footer_evidence = max(header_footer_evidence, 0.76)
        if is_signature_zone and page_length_ratio < 0.85 and left_ext_ratio < 0.03 and right_ext_ratio < 0.03:
            header_footer_evidence = max(header_footer_evidence, 0.70)

    else:
        # Sọc nghiêng (DIAGONAL)
        continuity, contrast, bg_int, stripe_int, int_std = _sample_diagonal_features(
            cand, maps, min_contrast
        )
        ep_x1, ep_y1, ep_x2, ep_y2 = cand.endpoints
        xc = 0.5 * (ep_x1 + ep_x2)
        yc = 0.5 * (ep_y1 + ep_y2)
        pos_norm = xc / float(w)
        edge_dist_norm = min(
            xc / float(w),
            (w - xc) / float(w),
            yc / float(h),
            (h - yc) / float(h),
        )
        # Nếu là đoạn nghiêng gần phương ngang, ghi nhận thêm độ phủ ngang và vượt lề
        if abs(ep_x2 - ep_x1) > abs(ep_y2 - ep_y1):
            page_length_ratio = min(1.0, abs(ep_x2 - ep_x1) / float(max(1, w)))
            left_ext_px = max(0.0, margins.content_left - min(ep_x1, ep_x2))
            right_ext_px = max(0.0, max(ep_x1, ep_x2) - margins.content_right)
            left_ext_ratio = left_ext_px / float(max(1, w))
            right_ext_ratio = right_ext_px / float(max(1, w))
        margin_penetration = 0.5 if cand.relative_length >= 0.5 else 0.1
        content_overlap = 0.15

    return StripeFeatures(
        relative_length=round(float(cand.relative_length), 4),
        length_px=round(float(cand.length_px), 2),
        thickness_px=round(float(cand.width_px), 2),
        angle_deg=round(float(cand.angle_deg), 2),
        continuity=round(float(continuity), 4),
        contrast=round(float(contrast), 2),
        background_intensity=round(float(bg_int), 2),
        stripe_intensity=round(float(stripe_int), 2),
        intensity_std=round(float(int_std), 2),
        position_norm=round(float(pos_norm), 4),
        edge_distance_norm=round(float(edge_dist_norm), 4),
        margin_penetration=round(float(margin_penetration), 4),
        content_overlap=round(float(content_overlap), 4),
        endpoint_junctions=endpoint_junctions,
        perpendicular_crossings=perp_crossings,
        parallel_grid_siblings=parallel_siblings,
        text_above_density=round(float(text_above_density), 4),
        repeat_page_count=int(repeat_page_count),
        detected_by=list(cand.detected_by),
        page_length_ratio=round(float(page_length_ratio), 4),
        content_left_px=round(float(margins.content_left), 2),
        content_right_px=round(float(margins.content_right), 2),
        content_width_px=round(float(margins.content_width), 2),
        content_margins_reliable=bool(margins.reliable),
        left_extension_px=round(float(left_ext_px), 2),
        right_extension_px=round(float(right_ext_px), 2),
        left_extension_ratio=round(float(left_ext_ratio), 4),
        right_extension_ratio=round(float(right_ext_ratio), 4),
        table_structure_evidence=round(float(table_evidence), 4),
        header_footer_evidence=round(float(header_footer_evidence), 4),
        text_below_density=round(float(text_below_density), 4),
        text_cut_through_ratio=round(float(text_cut_through_ratio), 4),
        extends_past_table_bounds=bool(extends_past_table_bounds),
    )


def _evaluate_horizontal_candidate(
    cand: LineCandidate,
    feat: StripeFeatures,
    page_width: int,
    min_relative_length: float,
    max_thickness_px: int,
    min_contrast: float,
    horizontal_min_page_ratio: float,
    horizontal_margin_extension: float,
    confirm_threshold: float,
    review_threshold: float,
) -> ValidatedStripe:
    """
    Hệ thống đánh giá rule-based có giải thích (`HorizontalStripeScore` — Phase 4 Part A8 & A9)
    dành riêng cho ứng viên đường ngang (`HORIZONTAL` và dải ngang `BAND` góc ~0°).

    Phân loại mỗi ứng viên thành một trong 3 nhãn:
      - `SCAN_STRIPE` (`status = "CONFIRMED"`): Đủ bằng chứng là sọc lỗi máy scan.
      - `REVIEW` (`status = "SUSPECTED"`): Có dấu hiệu bất thường nhưng chưa đủ chắc chắn, cần người dùng kiểm tra.
      - `LIKELY_DOCUMENT_LINE` (`status = "REJECTED"`): Có đặc trưng của đường kẻ hợp lệ trong tài liệu
        (TUYỆT ĐỐI không đưa vào bảng trang lỗi như sọc đã xác nhận).
    """
    defect_type = "BAND_STRIPE" if cand.stripe_type == "BAND" else "HORIZONTAL_STRIPE"
    reasons: List[str] = []

    if not feat.content_margins_reliable:
        reasons.append("FALLBACK_CONTENT_MARGINS")

    # 1. Lọc viền giấy / bóng mép trên hoặc dưới cùng của trang
    if feat.edge_distance_norm < 0.02:
        return ValidatedStripe(
            stripe_type=cand.stripe_type,
            defect_type=defect_type,
            status="REJECTED",
            bbox=cand.bbox,
            endpoints=cand.endpoints,
            length_px=feat.length_px,
            width_px=feat.thickness_px,
            relative_length=feat.relative_length,
            angle_deg=feat.angle_deg,
            suspicion_score=0.0,
            confidence=0.0,
            severity="LOW",
            reason="Loại bỏ: Đường ngang sát mép trên/dưới trang (viền giấy / bóng mép máy scan).",
            rejection_category="page_border",
            features=feat,
            horizontal_classification=HORIZONTAL_CLASS_DOCUMENT_LINE,
            horizontal_score=0.0,
            reason_codes=["PAGE_EDGE_BORDER"],
        )

    # 2. Lọc nét quá ngắn / độ liên tục hoặc độ tương phản dưới ngưỡng vật lý
    if (
        feat.page_length_ratio < min_relative_length
        or feat.thickness_px > max_thickness_px
        or feat.continuity < 0.28
        or feat.contrast < min_contrast
    ):
        return ValidatedStripe(
            stripe_type=cand.stripe_type,
            defect_type=defect_type,
            status="REJECTED",
            bbox=cand.bbox,
            endpoints=cand.endpoints,
            length_px=feat.length_px,
            width_px=feat.thickness_px,
            relative_length=feat.relative_length,
            angle_deg=feat.angle_deg,
            suspicion_score=0.0,
            confidence=0.0,
            severity="LOW",
            reason="Loại bỏ: Nét ngang ngắn, độ liên tục hoặc độ tương phản dưới ngưỡng sọc scan.",
            rejection_category="short_or_weak_stroke",
            features=feat,
            horizontal_classification=HORIZONTAL_CLASS_DOCUMENT_LINE,
            horizontal_score=0.0,
            reason_codes=["INSUFFICIENT_EVIDENCE"],
        )

    ep_x1, _, ep_x2, _ = cand.endpoints
    min_x = min(ep_x1, ep_x2)
    max_x = max(ep_x1, ep_x2)
    touches_left_edge = min_x <= max(14.0, float(page_width) * 0.015)
    touches_right_edge = max_x >= float(page_width) - max(14.0, float(page_width) * 0.015)
    touches_any_edge = touches_left_edge or touches_right_edge
    touches_both_edges = touches_left_edge and touches_right_edge

    ext_left = feat.left_extension_ratio >= horizontal_margin_extension
    ext_right = feat.right_extension_ratio >= horizontal_margin_extension
    ext_both = ext_left and ext_right
    ext_any = ext_left or ext_right
    near_full_width = feat.page_length_ratio >= horizontal_min_page_ratio

    if near_full_width:
        reasons.append("NEAR_FULL_PAGE_WIDTH")
    else:
        reasons.append("BELOW_PAGE_WIDTH_RATIO")

    if ext_both:
        reasons.append("EXTENDS_BOTH_MARGINS")
    elif ext_left:
        reasons.append("EXTENDS_LEFT_MARGIN")
    elif ext_right:
        reasons.append("EXTENDS_RIGHT_MARGIN")
    else:
        reasons.append("WITHIN_CONTENT_MARGINS")

    if touches_any_edge:
        reasons.append("TOUCHES_PAGE_EDGE")
    if feat.continuity >= 0.80:
        reasons.append("HIGH_CONTINUITY")
    if feat.extends_past_table_bounds:
        reasons.append("EXTENDS_PAST_TABLE_BOUNDS")
    if feat.text_cut_through_ratio >= 0.08:
        reasons.append("CUTS_THROUGH_TEXT_LINE")
    if feat.repeat_page_count >= 2:
        reasons.append("REPEATED_ACROSS_PAGES")

    # 3. Kiểm tra bằng chứng Bảng biểu / Khung biểu mẫu khép kín (A6)
    # Lưu ý: Nếu là sọc đè xuyên qua bảng thò ra ngoài biên bảng (`extends_past_table_bounds = True`),
    # `table_structure_evidence` đã bằng 0.0 nên KHÔNG bao giờ xóa nhầm sọc thật!
    if feat.table_structure_evidence >= 0.72 and not touches_both_edges:
        reasons.append("LIKELY_TABLE_BORDER")
        return ValidatedStripe(
            stripe_type=cand.stripe_type,
            defect_type=defect_type,
            status="REJECTED",
            bbox=cand.bbox,
            endpoints=cand.endpoints,
            length_px=feat.length_px,
            width_px=feat.thickness_px,
            relative_length=feat.relative_length,
            angle_deg=feat.angle_deg,
            suspicion_score=0.08,
            confidence=0.08,
            severity="LOW",
            reason=(
                f"Đường kẻ bảng biểu / khung biểu mẫu hợp lệ "
                f"(table_evidence={feat.table_structure_evidence:.2f}, "
                f"corners={feat.endpoint_junctions}, parallel={feat.parallel_grid_siblings}, "
                f"crossings={feat.perpendicular_crossings})."
            ),
            rejection_category="table_grid_line",
            features=feat,
            horizontal_classification=HORIZONTAL_CLASS_DOCUMENT_LINE,
            horizontal_score=0.08,
            reason_codes=reasons,
        )

    # 4. Kiểm tra đường Footer ở mép dưới trang (A7):
    # - Nếu nằm gọn trong lề: LIKELY_DOCUMENT_LINE (LIKELY_FOOTER_SEPARATOR).
    # - Nếu có đặc điểm vượt lề bất thường nhưng vẫn giống đường phân cách footer (mảnh, không tràn sát cả 2 mép giấy):
    #   ưu tiên REVIEW thay vì xác nhận lỗi tự động (đúng quy tắc mục A7).
    is_footer_candidate = (
        feat.position_norm >= 0.84
        and feat.thickness_px <= 10.0
        and feat.text_cut_through_ratio < 0.06
        and (feat.header_footer_evidence >= 0.70 or feat.position_norm >= 0.86)
    )
    if is_footer_candidate:
        reasons.append("LIKELY_FOOTER_SEPARATOR")
        footer_sym_err = abs(min_x - (float(page_width) - max_x)) / float(max(1, page_width))
        ragged_right_normal_footer = (
            not near_full_width
            and not touches_any_edge
            and not ext_both
            and max(feat.left_extension_ratio, feat.right_extension_ratio) < 0.055
            and footer_sym_err <= 0.05
        )
        if (not ext_any and not near_full_width) or ragged_right_normal_footer:
            return ValidatedStripe(
                stripe_type=cand.stripe_type,
                defect_type=defect_type,
                status="REJECTED",
                bbox=cand.bbox,
                endpoints=cand.endpoints,
                length_px=feat.length_px,
                width_px=feat.thickness_px,
                relative_length=feat.relative_length,
                angle_deg=feat.angle_deg,
                suspicion_score=0.10,
                confidence=0.10,
                severity="LOW",
                reason=(
                    f"Đường phân cách chân trang (footer) hợp lệ trong vùng lề "
                    f"(page_ratio={feat.page_length_ratio:.2f}, y_norm={feat.position_norm:.2f})."
                ),
                rejection_category="footer_separator",
                features=feat,
                horizontal_classification=HORIZONTAL_CLASS_DOCUMENT_LINE,
                horizontal_score=0.10,
                reason_codes=reasons,
            )
        if ext_any and not touches_both_edges and feat.page_length_ratio < 0.97:
            score_rev = max(review_threshold, min(confirm_threshold - 0.04, 0.54))
            return ValidatedStripe(
                stripe_type=cand.stripe_type,
                defect_type=defect_type,
                status="SUSPECTED",
                bbox=cand.bbox,
                endpoints=cand.endpoints,
                length_px=feat.length_px,
                width_px=feat.thickness_px,
                relative_length=feat.relative_length,
                angle_deg=feat.angle_deg,
                suspicion_score=round(score_rev, 3),
                confidence=round(score_rev, 3),
                severity="LOW",
                reason=(
                    f"Đường ngang vùng chân trang nhưng vượt ra ngoài lề nội dung — cần kiểm tra "
                    f"(page_ratio={feat.page_length_ratio:.2f}, left_ext={feat.left_extension_ratio:.3f}, "
                    f"right_ext={feat.right_extension_ratio:.3f})."
                ),
                rejection_category=None,
                features=feat,
                horizontal_classification=HORIZONTAL_CLASS_REVIEW,
                horizontal_score=round(score_rev, 3),
                reason_codes=reasons,
            )

    # 5. Kiểm tra các đường ngang ngắn hơn ngưỡng `horizontal_min_page_ratio`:
    # Chỉ giữ lại nếu có BẰNG CHỨNG LỖI SCAN RÕ RÀNG (tràn sát mép giấy vật lý + vượt sâu ngoài lề + cắt ngang chữ hoặc dải mực dày).
    # Mọi đường ngang nằm trong vùng lề nội dung (Quốc hiệu, gạch chân tiêu đề, phân cách mục, dòng ký tên/biểu mẫu) -> LIKELY_DOCUMENT_LINE.
    if not near_full_width:
        abnormal_edge_bleed = (
            touches_any_edge
            and max(feat.left_extension_ratio, feat.right_extension_ratio) >= max(0.045, horizontal_margin_extension * 1.5)
            and feat.continuity >= 0.65
            and (
                feat.text_cut_through_ratio >= 0.06
                or feat.thickness_px >= 7.0
                or feat.page_length_ratio >= max(0.86, horizontal_min_page_ratio - 0.04)
            )
        )
        abnormal_interior_smudge = (
            feat.thickness_px >= 10.0
            and feat.text_cut_through_ratio >= 0.14
            and feat.contrast >= 35.0
            and feat.table_structure_evidence < 0.30
        )

        if abnormal_edge_bleed or abnormal_interior_smudge:
            reasons.append("ABNORMAL_SHORT_STRIPE_EVIDENCE")
            # Nếu vừa tràn sát mép giấy + vượt lề mạnh + cắt chữ hoặc rất dài (>= 87%) -> xác nhận SCAN_STRIPE, ngược lại -> REVIEW
            if touches_any_edge and (
                feat.page_length_ratio >= max(0.87, horizontal_min_page_ratio - 0.03)
                or (feat.thickness_px >= 8.0 and feat.text_cut_through_ratio >= 0.06)
            ):
                h_score = round(min(0.92, max(confirm_threshold + 0.08, 0.74)), 3)
                return ValidatedStripe(
                    stripe_type=cand.stripe_type,
                    defect_type=defect_type,
                    status="CONFIRMED",
                    bbox=cand.bbox,
                    endpoints=cand.endpoints,
                    length_px=feat.length_px,
                    width_px=feat.thickness_px,
                    relative_length=feat.relative_length,
                    angle_deg=feat.angle_deg,
                    suspicion_score=h_score,
                    confidence=h_score,
                    severity="MEDIUM" if feat.page_length_ratio < 0.85 else "HIGH",
                    reason=(
                        f"Phát hiện sọc ngang bất thường tràn mép giấy / đè nội dung "
                        f"(page_ratio={feat.page_length_ratio:.2f}, left_ext={feat.left_extension_ratio:.3f}, "
                        f"right_ext={feat.right_extension_ratio:.3f}, width={feat.thickness_px:.1f}px)."
                    ),
                    rejection_category=None,
                    features=feat,
                    horizontal_classification=HORIZONTAL_CLASS_SCAN_STRIPE,
                    horizontal_score=h_score,
                    reason_codes=reasons,
                )

            h_score = round(max(review_threshold + 0.06, min(confirm_threshold - 0.03, 0.56)), 3)
            return ValidatedStripe(
                stripe_type=cand.stripe_type,
                defect_type=defect_type,
                status="SUSPECTED",
                bbox=cand.bbox,
                endpoints=cand.endpoints,
                length_px=feat.length_px,
                width_px=feat.thickness_px,
                relative_length=feat.relative_length,
                angle_deg=feat.angle_deg,
                suspicion_score=h_score,
                confidence=h_score,
                severity="MEDIUM",
                reason=(
                    f"Vệt ngang ngắn hơn ngưỡng trang nhưng có dấu hiệu lỗi scan bất thường cần kiểm tra "
                    f"(page_ratio={feat.page_length_ratio:.2f}, cut_text={feat.text_cut_through_ratio:.2f})."
                ),
                rejection_category=None,
                features=feat,
                horizontal_classification=HORIZONTAL_CLASS_REVIEW,
                horizontal_score=h_score,
                reason_codes=reasons,
            )

        # Phân loại chi tiết lý do đường tài liệu hợp lệ để phục vụ kiểm thử và giải thích kết quả
        if feat.table_structure_evidence >= 0.50:
            reasons.append("LIKELY_TABLE_BORDER")
            rej_cat = "table_grid_line"
            desc = "Đường kẻ ngang bảng biểu hợp lệ"
        elif feat.position_norm <= 0.18 and feat.header_footer_evidence >= 0.65:
            reasons.append("LIKELY_HEADER_RULE")
            rej_cat = "header_rule"
            desc = "Đường kẻ dưới Quốc hiệu / Tiêu ngữ / Header hợp lệ"
        elif feat.text_above_density >= 0.024 and feat.text_cut_through_ratio < 0.06:
            reasons.append("LIKELY_TEXT_UNDERLINE")
            rej_cat = "text_underline"
            desc = "Đường gạch chân văn bản / tiêu đề hợp lệ"
        elif feat.position_norm >= 0.62:
            reasons.append("LIKELY_SIGNATURE_OR_FORM_LINE")
            rej_cat = "signature_or_form_line"
            desc = "Đường kẻ ký tên / điền thông tin biểu mẫu hợp lệ"
        else:
            reasons.append("LIKELY_SECTION_DIVIDER")
            reasons.append("INSUFFICIENT_EVIDENCE")
            rej_cat = "likely_document_line"
            desc = "Đường kẻ phân cách nội dung nằm trong vùng lề tài liệu"

        return ValidatedStripe(
            stripe_type=cand.stripe_type,
            defect_type=defect_type,
            status="REJECTED",
            bbox=cand.bbox,
            endpoints=cand.endpoints,
            length_px=feat.length_px,
            width_px=feat.thickness_px,
            relative_length=feat.relative_length,
            angle_deg=feat.angle_deg,
            suspicion_score=0.05,
            confidence=0.05,
            severity="LOW",
            reason=(
                f"Loại bỏ: {desc} (page_ratio={feat.page_length_ratio:.2f} < {horizontal_min_page_ratio:.2f}, "
                f"left_ext={feat.left_extension_ratio:.3f}, right_ext={feat.right_extension_ratio:.3f})."
            ),
            rejection_category=rej_cat,
            features=feat,
            horizontal_classification=HORIZONTAL_CLASS_DOCUMENT_LINE,
            horizontal_score=0.05,
            reason_codes=reasons,
        )

    # 6. Ứng viên dài gần hết trang (`page_length_ratio >= horizontal_min_page_ratio`):
    # Chấm điểm rule-based kết hợp độ vượt hai lề, chạm mép giấy, độ liên tục, độ tương phản và bằng chứng cấu trúc.
    base_score = 0.44
    if feat.page_length_ratio >= 0.96:
        base_score += 0.12
    elif feat.page_length_ratio >= 0.92:
        base_score += 0.07

    if ext_both:
        base_score += 0.24
    elif ext_any:
        base_score += 0.12
    else:
        # Dài >= ngưỡng nhưng không vượt lề (ví dụ trang in rất sát lề và chưa đóng góc bảng): thận trọng đưa vào REVIEW/REJECTED
        base_score -= 0.10

    if touches_both_edges:
        base_score += 0.12
    elif touches_any_edge:
        base_score += 0.07

    if feat.continuity >= 0.85:
        base_score += 0.08
    elif feat.continuity >= 0.60:
        base_score += 0.04
    else:
        base_score -= 0.08

    if feat.contrast >= 25.0:
        base_score += 0.05
    elif feat.contrast < 11.0:
        base_score -= 0.06

    if feat.repeat_page_count >= 2:
        base_score += 0.10

    # Khấu trừ nếu có bằng chứng bảng biểu hoặc header/footer một phần
    base_score -= 0.35 * feat.table_structure_evidence
    if not touches_both_edges:
        base_score -= 0.20 * feat.header_footer_evidence

    h_score = round(min(1.0, max(0.0, base_score)), 3)

    if h_score >= confirm_threshold and (ext_any or touches_any_edge):
        severity = "HIGH" if (h_score >= 0.78 or feat.thickness_px >= 8.0 or ext_both) else "MEDIUM"
        return ValidatedStripe(
            stripe_type=cand.stripe_type,
            defect_type=defect_type,
            status="CONFIRMED",
            bbox=cand.bbox,
            endpoints=cand.endpoints,
            length_px=feat.length_px,
            width_px=feat.thickness_px,
            relative_length=feat.relative_length,
            angle_deg=feat.angle_deg,
            suspicion_score=h_score,
            confidence=h_score,
            severity=severity,
            reason=(
                f"Phát hiện sọc ngang máy scan ({cand.stripe_type}) vượt lề nội dung "
                f"(page_ratio={feat.page_length_ratio:.2f}, left_ext={feat.left_extension_ratio:.3f}, "
                f"right_ext={feat.right_extension_ratio:.3f}, continuity={feat.continuity:.2f}, "
                f"contrast={feat.contrast:.1f}, reasons={','.join(reasons)})."
            ),
            rejection_category=None,
            features=feat,
            horizontal_classification=HORIZONTAL_CLASS_SCAN_STRIPE,
            horizontal_score=h_score,
            reason_codes=reasons,
        )

    if h_score >= review_threshold:
        return ValidatedStripe(
            stripe_type=cand.stripe_type,
            defect_type=defect_type,
            status="SUSPECTED",
            bbox=cand.bbox,
            endpoints=cand.endpoints,
            length_px=feat.length_px,
            width_px=feat.thickness_px,
            relative_length=feat.relative_length,
            angle_deg=feat.angle_deg,
            suspicion_score=h_score,
            confidence=h_score,
            severity="MEDIUM" if h_score >= 0.52 else "LOW",
            reason=(
                f"Đường ngang dài nghi ngờ sọc scan cần kiểm tra "
                f"(page_ratio={feat.page_length_ratio:.2f}, left_ext={feat.left_extension_ratio:.3f}, "
                f"right_ext={feat.right_extension_ratio:.3f}, score={h_score:.2f})."
            ),
            rejection_category=None,
            features=feat,
            horizontal_classification=HORIZONTAL_CLASS_REVIEW,
            horizontal_score=h_score,
            reason_codes=reasons,
        )

    reasons.append("INSUFFICIENT_EVIDENCE")
    return ValidatedStripe(
        stripe_type=cand.stripe_type,
        defect_type=defect_type,
        status="REJECTED",
        bbox=cand.bbox,
        endpoints=cand.endpoints,
        length_px=feat.length_px,
        width_px=feat.thickness_px,
        relative_length=feat.relative_length,
        angle_deg=feat.angle_deg,
        suspicion_score=h_score,
        confidence=h_score,
        severity="LOW",
        reason=f"Đường ngang không đủ bằng chứng lỗi sọc scan (score={h_score:.3f} < {review_threshold:.3f}).",
        rejection_category="likely_document_line",
        features=feat,
        horizontal_classification=HORIZONTAL_CLASS_DOCUMENT_LINE,
        horizontal_score=h_score,
        reason_codes=reasons,
    )


def validate_stripe_candidates(
    candidates: List[LineCandidate],
    maps: PreparedStripeMaps,
    min_relative_length: float = STRIPE_MIN_RELATIVE_LENGTH,
    max_thickness_px: int = STRIPE_MAX_THICKNESS_PX,
    min_contrast: float = STRIPE_MIN_CONTRAST,
    confirm_threshold: float = STRIPE_CONFIRM_THRESHOLD,
    review_threshold: float = STRIPE_REVIEW_THRESHOLD,
    repeat_counts_by_candidate: Optional[Dict[int, int]] = None,
    horizontal_min_page_ratio: float = HORIZONTAL_STRIPE_MIN_PAGE_RATIO,
    horizontal_margin_extension: float = HORIZONTAL_STRIPE_MARGIN_EXTENSION,
    content_margins: Optional[ContentMargins] = None,
) -> List[ValidatedStripe]:
    """
    Kiểm định danh sách `LineCandidate`:
    - Ước lượng `ContentMargins` một lần duy nhất cho toàn trang.
    - Với ứng viên ngang (`HORIZONTAL` hoặc dải `BAND` phương ngang ~0°): áp dụng bộ kiểm định chuyên sâu Phase 4
      (`_evaluate_horizontal_candidate`).
    - Với ứng viên dọc (`VERTICAL` / `BAND` dọc) và sọc nghiêng (`DIAGONAL`): bảo toàn nguyên vẹn logic kiểm định
      đã được kiểm chứng ở Phase 2 (không gây regression).
    """
    validated_list: List[ValidatedStripe] = []
    repeat_map = repeat_counts_by_candidate or {}
    h, w = maps.gray.shape[:2]
    margins = content_margins or estimate_content_margins(maps)

    for idx, cand in enumerate(candidates):
        rep_count = repeat_map.get(idx, 1)
        feat = _extract_candidate_features(
            cand=cand,
            all_candidates=candidates,
            maps=maps,
            min_contrast=min_contrast,
            repeat_page_count=rep_count,
            content_margins=margins,
        )

        is_horizontal_cand = (cand.stripe_type != "DIAGONAL") and (
            _angle_diff_deg(feat.angle_deg, 0.0) <= 5.0
        )
        if is_horizontal_cand:
            validated_list.append(
                _evaluate_horizontal_candidate(
                    cand=cand,
                    feat=feat,
                    page_width=w,
                    min_relative_length=min_relative_length,
                    max_thickness_px=max_thickness_px,
                    min_contrast=min_contrast,
                    horizontal_min_page_ratio=horizontal_min_page_ratio,
                    horizontal_margin_extension=horizontal_margin_extension,
                    confirm_threshold=confirm_threshold,
                    review_threshold=review_threshold,
                )
            )
            continue

        defect_type_map = {
            "VERTICAL": "VERTICAL_STRIPE",
            "HORIZONTAL": "HORIZONTAL_STRIPE",
            "DIAGONAL": "DIAGONAL_STRIPE",
            "BAND": "BAND_STRIPE",
        }
        defect_type = defect_type_map.get(cand.stripe_type, "VERTICAL_STRIPE")

        # 1. Lọc viền giấy / bóng mép trang (Page Border)
        if feat.edge_distance_norm < 0.02:
            validated_list.append(
                ValidatedStripe(
                    stripe_type=cand.stripe_type,
                    defect_type=defect_type,
                    status="REJECTED",
                    bbox=cand.bbox,
                    endpoints=cand.endpoints,
                    length_px=feat.length_px,
                    width_px=feat.thickness_px,
                    relative_length=feat.relative_length,
                    angle_deg=feat.angle_deg,
                    suspicion_score=0.0,
                    confidence=0.0,
                    severity="LOW",
                    reason="Loại bỏ: Đường sát mép trang (viền giấy / bóng mép máy scan).",
                    rejection_category="page_border",
                    features=feat,
                )
            )
            continue

        # 2. Lọc đường kẻ dọc bảng biểu (Table Grid Line)
        is_table_grid = (
            feat.margin_penetration < 0.15
            and feat.relative_length < 0.88
            and (
                (feat.endpoint_junctions >= 1 and feat.perpendicular_crossings >= 2)
                or (feat.parallel_grid_siblings >= 1 and feat.perpendicular_crossings >= 2)
                or (feat.endpoint_junctions >= 1 and feat.parallel_grid_siblings >= 1)
                or (feat.perpendicular_crossings >= 4)
            )
        )
        # Bổ sung cho cạnh dọc đóng khung biểu mẫu rộng (như Case 12 khung 95% chiều rộng nhưng chiều cao < 90% trang và có >=4 đường ngang giao cắt)
        if not is_table_grid and _angle_diff_deg(feat.angle_deg, 90.0) <= 5.0:
            if (
                feat.relative_length < 0.82
                and feat.endpoint_junctions >= 1
                and (feat.perpendicular_crossings >= 2 or feat.parallel_grid_siblings >= 1)
            ):
                is_table_grid = True

        if is_table_grid:
            validated_list.append(
                ValidatedStripe(
                    stripe_type=cand.stripe_type,
                    defect_type=defect_type,
                    status="REJECTED",
                    bbox=cand.bbox,
                    endpoints=cand.endpoints,
                    length_px=feat.length_px,
                    width_px=feat.thickness_px,
                    relative_length=feat.relative_length,
                    angle_deg=feat.angle_deg,
                    suspicion_score=0.0,
                    confidence=0.0,
                    severity="LOW",
                    reason=(
                        f"Loại bỏ: Đường kẻ bảng biểu hợp lệ (giao điểm đầu mút={feat.endpoint_junctions}, "
                        f"đường song song={feat.parallel_grid_siblings}, đường vuông góc={feat.perpendicular_crossings})."
                    ),
                    rejection_category="table_grid_line",
                    features=feat,
                )
            )
            continue

        # 3. Lọc thanh đánh dấu sửa đổi ở lề dọc
        is_margin_revision_or_text_line = (
            feat.margin_penetration == 0.0
            and (
                (
                    _angle_diff_deg(feat.angle_deg, 90.0) <= 5.0
                    and feat.content_overlap == 0.0
                    and feat.relative_length < 0.68
                    and feat.thickness_px <= 8.0
                )
                or (
                    feat.intensity_std >= 28.0
                    and (feat.content_overlap >= 0.02 or feat.parallel_grid_siblings >= 1)
                )
            )
        )
        if is_margin_revision_or_text_line:
            validated_list.append(
                ValidatedStripe(
                    stripe_type=cand.stripe_type,
                    defect_type=defect_type,
                    status="REJECTED",
                    bbox=cand.bbox,
                    endpoints=cand.endpoints,
                    length_px=feat.length_px,
                    width_px=feat.thickness_px,
                    relative_length=feat.relative_length,
                    angle_deg=feat.angle_deg,
                    suspicion_score=0.0,
                    confidence=0.0,
                    severity="LOW",
                    reason=(
                        f"Loại bỏ: Đường kẻ lề nội bộ / dòng văn bản nằm trong lề "
                        f"(rel_len={feat.relative_length:.2f}, margin_pen=0.0, std={feat.intensity_std:.1f})."
                    ),
                    rejection_category="margin_revision_bar",
                    features=feat,
                )
            )
            continue

        # 3b. Nếu là đoạn DIAGONAL hơi nghiêng theo phương ngang nhưng ngắn và nằm hoàn toàn trong lề nội dung (ví dụ nét gạch dưới chữ ký nghiêng)
        if (
            cand.stripe_type == "DIAGONAL"
            and feat.page_length_ratio > 0.0
            and feat.page_length_ratio < horizontal_min_page_ratio
            and feat.left_extension_ratio < horizontal_margin_extension
            and feat.right_extension_ratio < horizontal_margin_extension
        ):
            validated_list.append(
                ValidatedStripe(
                    stripe_type=cand.stripe_type,
                    defect_type=defect_type,
                    status="REJECTED",
                    bbox=cand.bbox,
                    endpoints=cand.endpoints,
                    length_px=feat.length_px,
                    width_px=feat.thickness_px,
                    relative_length=feat.relative_length,
                    angle_deg=feat.angle_deg,
                    suspicion_score=0.05,
                    confidence=0.05,
                    severity="LOW",
                    reason="Loại bỏ: Nét nghiêng ngang ngắn nằm gọn trong vùng lề nội dung.",
                    rejection_category="likely_document_line",
                    features=feat,
                )
            )
            continue

        # 4. Lọc chữ ký hoặc nét vẽ ngắn / không liên tục / độ tương phản quá yếu
        if (
            feat.relative_length < min_relative_length
            or feat.thickness_px > max_thickness_px
            or feat.continuity < 0.28
            or feat.contrast < min_contrast
        ):
            validated_list.append(
                ValidatedStripe(
                    stripe_type=cand.stripe_type,
                    defect_type=defect_type,
                    status="REJECTED",
                    bbox=cand.bbox,
                    endpoints=cand.endpoints,
                    length_px=feat.length_px,
                    width_px=feat.thickness_px,
                    relative_length=feat.relative_length,
                    angle_deg=feat.angle_deg,
                    suspicion_score=0.0,
                    confidence=0.0,
                    severity="LOW",
                    reason="Loại bỏ: Nét ngắn, độ liên tục hoặc độ tương phản dưới ngưỡng sọc scan.",
                    rejection_category="short_or_weak_stroke",
                    features=feat,
                )
            )
            continue

        # 5. Tính toán điểm nghi vấn cho VERTICAL / BAND dọc / DIAGONAL
        len_score = min(1.0, max(0.0, (feat.relative_length - 0.25) / 0.58))
        cont_score = min(1.0, max(0.0, (feat.continuity - 0.22) / 0.62))
        contrast_score = min(1.0, max(0.25, (feat.contrast - 4.0) / 28.0))
        uniformity_score = min(1.0, max(0.25, 1.0 - (feat.intensity_std / 65.0)))

        method_bonus = min(0.14, 0.07 * max(0, len(feat.detected_by) - 1))
        margin_bonus = 0.08 * feat.margin_penetration
        repeat_bonus = 0.12 if feat.repeat_page_count >= 2 else 0.0

        raw_score = (
            0.34 * len_score
            + 0.26 * cont_score
            + 0.16 * contrast_score
            + 0.14 * uniformity_score
            + method_bonus
            + margin_bonus
            + repeat_bonus
        )
        suspicion_score = round(min(1.0, max(0.0, raw_score)), 3)

        if suspicion_score >= confirm_threshold:
            status = "CONFIRMED"
            severity = (
                "HIGH"
                if (
                    suspicion_score >= 0.78
                    or feat.relative_length >= 0.75
                    or cand.stripe_type == "BAND"
                )
                else "MEDIUM"
            )
            reason = (
                f"Phát hiện sọc {cand.stripe_type} rõ ràng "
                f"(rel_len={feat.relative_length:.2f}, width={feat.thickness_px:.1f}px, "
                f"continuity={feat.continuity:.2f}, contrast={feat.contrast:.1f}, "
                f"methods={'+'.join(feat.detected_by)})."
            )
        elif suspicion_score >= review_threshold:
            status = "SUSPECTED"
            severity = "MEDIUM" if suspicion_score >= 0.52 else "LOW"
            reason = (
                f"Vệt nghi ngờ sọc {cand.stripe_type} cần kiểm tra "
                f"(rel_len={feat.relative_length:.2f}, width={feat.thickness_px:.1f}px, "
                f"continuity={feat.continuity:.2f}, contrast={feat.contrast:.1f})."
            )
        else:
            status = "REJECTED"
            severity = "LOW"
            reason = f"Điểm nghi vấn thấp ({suspicion_score:.3f} < {review_threshold:.3f})."

        validated_list.append(
            ValidatedStripe(
                stripe_type=cand.stripe_type,
                defect_type=defect_type,
                status=status,
                bbox=cand.bbox,
                endpoints=cand.endpoints,
                length_px=feat.length_px,
                width_px=feat.thickness_px,
                relative_length=feat.relative_length,
                angle_deg=feat.angle_deg,
                suspicion_score=suspicion_score,
                confidence=suspicion_score,
                severity=severity,
                reason=reason,
                rejection_category="below_threshold" if status == "REJECTED" else None,
                features=feat,
            )
        )

    return validated_list
