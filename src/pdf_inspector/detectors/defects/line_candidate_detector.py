"""
Module phát hiện ứng viên vệt sọc (Line Candidate Detector — `src/pdf_inspector/detectors/defects/line_candidate_detector.py`).
Triển khai kết hợp 3 kỹ thuật Computer Vision trên ảnh trang PDF từ `PageContext`:
1. Method 1 — Morphological Operations (Kernel định hướng dọc/ngang tỷ lệ theo kích thước trang & DPI).
2. Method 2 — Probabilistic Hough Line Transform (`cv2.HoughLinesP` kèm thuật toán hợp nhất đoạn thẳng đồng tuyến có kiểm soát).
3. Method 3 — Projection Profile / Intensity Analysis (Phân tích hồ sơ cường độ và độ lệch nền cục bộ theo hàng/cột).
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreparedStripeMaps:
    """Các bản đồ ảnh dẫn xuất phục vụ phát hiện và kiểm định sọc mà không làm hỏng sọc mảnh."""

    gray: np.ndarray  # Ảnh xám gốc 2D uint8
    clahe: np.ndarray  # Ảnh tăng tương phản cục bộ CLAHE
    v_residual: np.ndarray  # Độ lệch tối theo phương dọc (đã lọc bỏ nét chữ ngắn)
    h_residual: np.ndarray  # Độ lệch tối theo phương ngang (đã lọc bỏ nét chữ ngắn)
    diag_residual: np.ndarray  # Độ lệch tối đa hướng phục vụ HoughLinesP phát hiện sọc nghiêng
    otsu_binary: np.ndarray  # Mặt nạ nhị phân Otsu (chữ/nội dung đậm = 255)
    dpi: int = 160


@dataclass
class LineCandidate:
    """Ứng viên vệt sọc phát hiện được từ một hoặc nhiều phương pháp."""

    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) theo pixel
    endpoints: Tuple[float, float, float, float]  # (x_start, y_start, x_end, y_end)
    stripe_type: str  # "VERTICAL", "HORIZONTAL", "DIAGONAL", "BAND"
    length_px: float
    width_px: float
    relative_length: float
    angle_deg: float  # Góc trong [0, 180): ~90° là dọc, ~0° là ngang
    detected_by: List[str] = field(default_factory=list)


def prepare_stripe_maps(
    gray_image: np.ndarray,
    clahe_image: Optional[np.ndarray] = None,
    dpi: int = 160,
) -> PreparedStripeMaps:
    """
    Chuẩn bị các bản đồ ảnh xám gốc, ảnh CLAHE và bản đồ độ lệch nền định hướng (`v_residual`, `h_residual`).
    Sử dụng phép đóng hình thái học (morphological closing) theo phương vuông góc với sọc để ước lượng
    nền giấy cục bộ (kể cả khi trang bị bóng mờ hoặc nền giấy không đều) mà KHÔNG xóa mất sọc xám mảnh 1-2px.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Ảnh đầu vào cho Stripe Detector bị rỗng.")

    if gray_image.ndim == 3:
        gray = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = gray_image.astype(np.uint8, copy=False)

    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        raise ValueError(f"Kích thước ảnh quá nhỏ để phân tích sọc: {w}x{h}")

    if clahe_image is not None and clahe_image.shape == gray.shape:
        clahe = clahe_image
    else:
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe = clahe_op.apply(gray)

    # Ước lượng nền cục bộ theo phương ngang (để làm nổi bật sọc dọc / hơi nghiêng dọc)
    # Kích thước cửa sổ ngang đủ rộng để vượt qua độ dày của sọc/dải sọc (tối đa ~45px)
    bg_win_w = max(35, min(95, int(round(w * 0.045)) | 1))
    bg_win_h = max(35, min(95, int(round(h * 0.045)) | 1))

    # Không làm mờ dọc theo hướng dòng chữ trước khi trừ nền để tránh nối liền khoảng trắng 2-3px giữa các ký tự
    kernel_bg_h = cv2.getStructuringElement(cv2.MORPH_RECT, (bg_win_w, 1))
    kernel_bg_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bg_win_h))

    local_bg_h = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_bg_h)
    local_bg_v = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_bg_v)

    raw_v_res = np.clip(
        local_bg_h.astype(np.int16) - gray.astype(np.int16), 0, 255
    ).astype(np.uint8)
    raw_h_res = np.clip(
        local_bg_v.astype(np.int16) - gray.astype(np.int16), 0, 255
    ).astype(np.uint8)
    diag_residual = np.maximum(raw_v_res, raw_h_res)

    # Lọc bỏ các nét chữ ngắn rời rạc khỏi bản đồ residual dọc/ngang
    # nhưng bảo toàn 100% các vệt sọc dọc/ngang/dải sọc/sọc đứt quãng (vốn có đoạn liên tục >= 42px)
    min_dim_seg = max(40, int(round(max(h, w) * 0.024)))
    v_seg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_dim_seg))
    h_seg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_dim_seg, 1))
    v_residual = cv2.morphologyEx(raw_v_res, cv2.MORPH_OPEN, v_seg_kernel)
    h_residual = cv2.morphologyEx(raw_h_res, cv2.MORPH_OPEN, h_seg_kernel)

    _, otsu_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return PreparedStripeMaps(
        gray=gray,
        clahe=clahe,
        v_residual=v_residual,
        h_residual=h_residual,
        diag_residual=diag_residual,
        otsu_binary=otsu_binary,
        dpi=dpi,
    )


def detect_candidates_morphological(
    maps: PreparedStripeMaps,
    min_relative_length: float = 0.30,
    max_thickness_px: int = 45,
    min_contrast: float = 6.5,
) -> List[LineCandidate]:
    """
    Method 1 — Morphological Operations:
    Sử dụng kernel định hướng dọc và ngang tỷ lệ theo kích thước trang để phát hiện:
    - Sọc dọc đậm, sọc xám dọc mảnh, dải sọc dày (BAND), và sọc bị đứt quãng.
    - Sọc ngang kéo dài.
    """
    h, w = maps.gray.shape[:2]
    candidates: List[LineCandidate] = []

    # Cắt bỏ 1.5% viền sát mép để tránh lấy mép khung ảnh
    margin_x = max(4, int(round(w * 0.015)))
    margin_y = max(4, int(round(h * 0.015)))

    # 1. Phát hiện cấu trúc dọc trên v_residual ở 2 mức nhạy (nhạy cho sọc xám mảnh & chuẩn cho sọc đậm/dải sọc)
    v_open_len = max(25, int(round(h * 0.075)))
    v_bridge_len = max(20, int(round(h * 0.065)))
    v_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_open_len))
    v_bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_bridge_len))

    for thresh_val in (max(6, int(round(min_contrast))), 18):
        v_mask = (maps.v_residual >= thresh_val).astype(np.uint8) * 255
        v_mask[:, :margin_x] = 0
        v_mask[:, w - margin_x :] = 0

        opened = cv2.morphologyEx(v_mask, cv2.MORPH_OPEN, v_open_kernel)
        bridged = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, v_bridge_kernel)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bridged)
        for i in range(1, num_labels):
            x, y, cw, ch, _ = (
                int(stats[i, 0]),
                int(stats[i, 1]),
                int(stats[i, 2]),
                int(stats[i, 3]),
                int(stats[i, 4]),
            )
            rel_len = ch / float(h)
            if rel_len < min_relative_length or cw > max_thickness_px:
                continue
            if ch < cw * 4.0:
                continue

            band_thresh = max(8, int(round(w * 0.008)))
            s_type = "BAND" if cw >= band_thresh else "VERTICAL"
            xc = x + cw * 0.5
            candidates.append(
                LineCandidate(
                    bbox=(x, y, x + cw, y + ch),
                    endpoints=(xc, float(y), xc, float(y + ch)),
                    stripe_type=s_type,
                    length_px=float(ch),
                    width_px=float(cw),
                    relative_length=round(rel_len, 4),
                    angle_deg=90.0,
                    detected_by=["morphological"],
                )
            )

    # 2. Phát hiện cấu trúc ngang trên h_residual
    h_open_len = max(25, int(round(w * 0.085)))
    h_bridge_len = max(20, int(round(w * 0.06)))
    h_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_open_len, 1))
    h_bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_bridge_len, 1))

    for thresh_val in (max(7, int(round(min_contrast))), 18):
        h_mask = (maps.h_residual >= thresh_val).astype(np.uint8) * 255
        h_mask[:margin_y, :] = 0
        h_mask[h - margin_y :, :] = 0

        opened = cv2.morphologyEx(h_mask, cv2.MORPH_OPEN, h_open_kernel)
        # Đệm biên 0 trước khi MORPH_CLOSE để tránh hiệu ứng biên OpenCV kéo giãn giả tạo đầu mút đường 94-95% ra sát mép 0..w
        pad_x = h_bridge_len
        opened_padded = cv2.copyMakeBorder(
            opened, 0, 0, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0
        )
        bridged_padded = cv2.morphologyEx(
            opened_padded, cv2.MORPH_CLOSE, h_bridge_kernel
        )
        bridged = bridged_padded[:, pad_x : pad_x + w]

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bridged)
        for i in range(1, num_labels):
            x, y, cw, ch, _ = (
                int(stats[i, 0]),
                int(stats[i, 1]),
                int(stats[i, 2]),
                int(stats[i, 3]),
                int(stats[i, 4]),
            )
            rel_len = cw / float(w)
            if rel_len < min_relative_length or ch > max_thickness_px:
                continue
            if cw < ch * 4.0:
                continue

            band_thresh = max(8, int(round(h * 0.008)))
            s_type = "BAND" if ch >= band_thresh else "HORIZONTAL"
            yc = y + ch * 0.5
            candidates.append(
                LineCandidate(
                    bbox=(x, y, x + cw, y + ch),
                    endpoints=(float(x), yc, float(x + cw), yc),
                    stripe_type=s_type,
                    length_px=float(cw),
                    width_px=float(ch),
                    relative_length=round(rel_len, 4),
                    angle_deg=0.0,
                    detected_by=["morphological"],
                )
            )

    return candidates


def _segment_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Tính góc của đoạn thẳng trong khoảng [0, 180) độ (0° = ngang, 90° = dọc)."""
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return ang


def _angle_diff_deg(a: float, b: float) -> float:
    """Khoảng cách góc nhỏ nhất trên chu kỳ 180°."""
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _merge_collinear_segments(
    segments: List[Tuple[float, float, float, float]],
    max_angle_diff: float = 2.5,
    max_perp_dist: float = 4.5,
    max_gap_px: float = 75.0,
) -> List[Tuple[float, float, float, float]]:
    """
    Hợp nhất có kiểm soát các đoạn thẳng gần thẳng hàng (collinear) từ HoughLinesP:
    Chỉ ghép khi:
    - Chênh lệch góc <= `max_angle_diff` (2.5°).
    - Khoảng cách vuông góc giữa hai đường <= `max_perp_dist` (4.5 px).
    - Khoảng cách dọc theo phương đường thẳng <= `max_gap_px`.
    """
    if not segments:
        return []

    # Chuẩn hóa thứ tự đầu mút của từng đoạn theo chiều tăng của trục chính
    norm_segs: List[Tuple[float, float, float, float, float]] = []
    for x1, y1, x2, y2 in segments:
        ang = _segment_angle_deg(x1, y1, x2, y2)
        if abs(y2 - y1) >= abs(x2 - x1):
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
        else:
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
        norm_segs.append((x1, y1, x2, y2, ang))

    # Sắp xếp theo chiều dài giảm dần
    norm_segs.sort(key=lambda s: math.hypot(s[2] - s[0], s[3] - s[1]), reverse=True)
    used = [False] * len(norm_segs)
    merged: List[Tuple[float, float, float, float]] = []

    for i, seg_i in enumerate(norm_segs):
        if used[i]:
            continue
        used[i] = True
        cx1, cy1, cx2, cy2, c_ang = seg_i

        changed = True
        while changed:
            changed = False
            length_c = max(1.0, math.hypot(cx2 - cx1, cy2 - cy1))
            ux = (cx2 - cx1) / length_c
            uy = (cy2 - cy1) / length_c
            nx = -uy
            ny = ux

            for j, seg_j in enumerate(norm_segs):
                if used[j]:
                    continue
                jx1, jy1, jx2, jy2, j_ang = seg_j
                if _angle_diff_deg(c_ang, j_ang) > max_angle_diff:
                    continue

                # Kiểm tra khoảng cách vuông góc từ trung điểm đoạn j tới đường thẳng c
                jmx = 0.5 * (jx1 + jx2)
                jmy = 0.5 * (jy1 + jy2)
                perp_dist = abs((jmx - cx1) * nx + (jmy - cy1) * ny)
                if perp_dist > max_perp_dist:
                    continue

                # Chiếu 4 điểm đầu mút lên trục u
                p_c1 = 0.0
                p_c2 = length_c
                p_j1 = (jx1 - cx1) * ux + (jy1 - cy1) * uy
                p_j2 = (jx2 - cx1) * ux + (jy2 - cy1) * uy
                j_min, j_max = min(p_j1, p_j2), max(p_j1, p_j2)

                # Khoảng hở dọc theo trục
                gap = max(0.0, max(p_c1 - j_max, j_min - p_c2))
                if gap <= max_gap_px:
                    all_pts = [
                        (p_c1, cx1, cy1),
                        (p_c2, cx2, cy2),
                        (p_j1, jx1, jy1),
                        (p_j2, jx2, jy2),
                    ]
                    min_pt = min(all_pts, key=lambda item: item[0])
                    max_pt = max(all_pts, key=lambda item: item[0])
                    cx1, cy1 = min_pt[1], min_pt[2]
                    cx2, cy2 = max_pt[1], max_pt[2]
                    c_ang = _segment_angle_deg(cx1, cy1, cx2, cy2)
                    used[j] = True
                    changed = True

        merged.append((cx1, cy1, cx2, cy2))

    return merged


def detect_candidates_hough(
    maps: PreparedStripeMaps,
    min_relative_length: float = 0.30,
    min_contrast: float = 7.5,
) -> List[LineCandidate]:
    """
    Method 2 — Hough Line Transform (`cv2.HoughLinesP`):
    Tìm các đoạn thẳng dài (bao gồm cả sọc hơi nghiêng `DIAGONAL`, sọc dọc và sọc ngang).
    Để tránh ghép nhầm các ký tự văn bản dày đặc thành đường nghiêng giả, hàm tạo mặt nạ
    loại bỏ các thành phần chữ nhỏ/ngắn trước khi chạy HoughLinesP.
    """
    h, w = maps.gray.shape[:2]
    min_dim = float(min(h, w))

    # Sử dụng diag_residual để bắt cả sọc nghiêng, sọc dọc và sọc ngang
    combined_res = maps.diag_residual
    thresh_val = max(8, int(round(min_contrast)))
    raw_mask = (combined_res >= thresh_val).astype(np.uint8) * 255

    margin_x = max(4, int(round(w * 0.015)))
    margin_y = max(4, int(round(h * 0.015)))
    raw_mask[:margin_y, :] = 0
    raw_mask[h - margin_y :, :] = 0
    raw_mask[:, :margin_x] = 0
    raw_mask[:, w - margin_x :] = 0

    # Giữ lại các thành phần liên thông có kích thước lớn (loại bỏ toàn bộ các ký tự chữ rời rạc)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
    filtered_mask = np.zeros_like(raw_mask)
    min_comp_span = max(48, int(round(min_dim * 0.065)))
    for idx in range(1, num_labels):
        cw, ch, area = int(stats[idx, 2]), int(stats[idx, 3]), int(stats[idx, 4])
        span = max(cw, ch)
        if span >= min_comp_span and area >= 30:
            filtered_mask[labels == idx] = 255

    # Thu mảnh nhẹ các dải quá dày bằng Canny/skeleton hoặc giữ nguyên nếu mảnh
    lines = cv2.HoughLinesP(
        filtered_mask,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=max(40, int(round(min_dim * 0.08))),
        minLineLength=max(40, int(round(min_dim * 0.14))),
        maxLineGap=max(12, int(round(min_dim * 0.04))),
    )

    if lines is None or len(lines) == 0:
        return []

    raw_segments: List[Tuple[float, float, float, float]] = [
        (float(l[0][0]), float(l[0][1]), float(l[0][2]), float(l[0][3]))
        for l in lines
    ]
    merged_segments = _merge_collinear_segments(
        raw_segments,
        max_angle_diff=2.2,
        max_perp_dist=4.0,
        max_gap_px=max(20.0, min_dim * 0.065),
    )

    candidates: List[LineCandidate] = []
    for x1, y1, x2, y2 in merged_segments:
        length_px = math.hypot(x2 - x1, y2 - y1)
        ang = _segment_angle_deg(x1, y1, x2, y2)

        # Quy chiếu chiều dài tương đối theo trục chính
        if _angle_diff_deg(ang, 90.0) <= 3.0:
            stripe_type = "VERTICAL"
            ref_dim = float(h)
        elif _angle_diff_deg(ang, 0.0) <= 3.0:
            stripe_type = "HORIZONTAL"
            ref_dim = float(w)
        else:
            stripe_type = "DIAGONAL"
            # Với sọc hơi nghiêng, so sánh với chiều cao hoặc chiều rộng tùy hướng nghiêng
            ref_dim = float(h) if abs(y2 - y1) >= abs(x2 - x1) else float(w)

        rel_len = length_px / max(1.0, ref_dim)
        if rel_len < min_relative_length:
            continue

        # Ước lượng độ dày của sọc quanh đoạn thẳng
        pad = 2
        bx0 = max(0, int(math.floor(min(x1, x2))) - pad)
        by0 = max(0, int(math.floor(min(y1, y2))) - pad)
        bx1 = min(w, int(math.ceil(max(x1, x2))) + pad)
        by1 = min(h, int(math.ceil(max(y1, y2))) + pad)

        width_est = 3.0
        if stripe_type == "VERTICAL":
            width_est = float(max(1, bx1 - bx0 - 2 * pad + 1))
        elif stripe_type == "HORIZONTAL":
            width_est = float(max(1, by1 - by0 - 2 * pad + 1))

        candidates.append(
            LineCandidate(
                bbox=(bx0, by0, bx1, by1),
                endpoints=(x1, y1, x2, y2),
                stripe_type=stripe_type,
                length_px=round(length_px, 2),
                width_px=round(width_est, 2),
                relative_length=round(min(1.0, rel_len), 4),
                angle_deg=round(ang, 2),
                detected_by=["hough_lines_p"],
            )
        )

    return candidates


def _longest_active_run(
    active_1d: np.ndarray, max_gap: int
) -> Tuple[int, int, float]:
    """
    Tìm đoạn liên tục dài nhất trên mảng boolean 1D cho phép khoảng đứt quãng nhỏ `<= max_gap`.
    Trả về `(start_idx, end_idx, active_density_in_span)`.
    """
    indices = np.flatnonzero(active_1d)
    if indices.size == 0:
        return 0, 0, 0.0

    best_s = int(indices[0])
    best_e = int(indices[0])
    cur_s = int(indices[0])
    cur_e = int(indices[0])

    for idx in indices[1:]:
        idx_int = int(idx)
        if idx_int - cur_e <= max_gap:
            cur_e = idx_int
        else:
            if (cur_e - cur_s) > (best_e - best_s):
                best_s, best_e = cur_s, cur_e
            cur_s = idx_int
            cur_e = idx_int

    if (cur_e - cur_s) > (best_e - best_s):
        best_s, best_e = cur_s, cur_e

    span_len = max(1, best_e - best_s + 1)
    density = float(np.sum(active_1d[best_s : best_e + 1])) / float(span_len)
    return best_s, best_e, density


def detect_candidates_projection(
    maps: PreparedStripeMaps,
    min_relative_length: float = 0.35,
    max_thickness_px: int = 45,
    min_contrast: float = 6.5,
) -> List[LineCandidate]:
    """
    Method 3 — Projection Profile / Intensity Analysis:
    Phân tích hồ sơ cường độ theo từng cột và từng hàng trên bản đồ `v_residual` và `h_residual`
    để phát hiện cả các vệt sọc xám rất mảnh (1-2px) hoặc dải sọc (`BAND`) có tính liên tục cao.
    """
    h, w = maps.gray.shape[:2]
    candidates: List[LineCandidate] = []

    margin_x = max(5, int(round(w * 0.018)))
    margin_y = max(5, int(round(h * 0.018)))
    thresh = max(6.0, float(min_contrast))

    # 1. Phân tích theo cột (Vertical Projection Profile)
    active_v = maps.v_residual >= thresh
    col_ratios = np.mean(active_v[margin_y : h - margin_y, :], axis=0)
    col_candidates = np.flatnonzero(col_ratios >= 0.25)
    col_candidates = col_candidates[
        (col_candidates >= margin_x) & (col_candidates < w - margin_x)
    ]

    if col_candidates.size > 0:
        # Nhóm các cột liền kề (cách nhau <= 2px)
        groups: List[List[int]] = [[int(col_candidates[0])]]
        for c in col_candidates[1:]:
            c_int = int(c)
            if c_int - groups[-1][-1] <= 2:
                groups[-1].append(c_int)
            else:
                groups.append([c_int])

        max_gap_y = max(18, int(round(h * 0.055)))
        for grp in groups:
            x0 = grp[0]
            x1 = grp[-1] + 1
            cw = x1 - x0
            if cw > max_thickness_px:
                continue

            band_active = np.any(active_v[:, x0:x1], axis=1)
            y0, y1_incl, density = _longest_active_run(band_active, max_gap=max_gap_y)
            ch = y1_incl - y0 + 1
            rel_len = ch / float(h)
            if rel_len < min_relative_length or density < 0.30:
                continue

            band_thresh = max(8, int(round(w * 0.008)))
            s_type = "BAND" if cw >= band_thresh else "VERTICAL"
            xc = x0 + cw * 0.5
            candidates.append(
                LineCandidate(
                    bbox=(x0, y0, x1, y1_incl + 1),
                    endpoints=(xc, float(y0), xc, float(y1_incl + 1)),
                    stripe_type=s_type,
                    length_px=float(ch),
                    width_px=float(cw),
                    relative_length=round(rel_len, 4),
                    angle_deg=90.0,
                    detected_by=["projection_profile"],
                )
            )

    # 2. Phân tích theo hàng (Horizontal Projection Profile)
    active_h = maps.h_residual >= thresh
    row_ratios = np.mean(active_h[:, margin_x : w - margin_x], axis=1)
    row_candidates = np.flatnonzero(row_ratios >= 0.25)
    row_candidates = row_candidates[
        (row_candidates >= margin_y) & (row_candidates < h - margin_y)
    ]

    if row_candidates.size > 0:
        r_groups: List[List[int]] = [[int(row_candidates[0])]]
        for r in row_candidates[1:]:
            r_int = int(r)
            if r_int - r_groups[-1][-1] <= 2:
                r_groups[-1].append(r_int)
            else:
                r_groups.append([r_int])

        max_gap_x = max(18, int(round(w * 0.055)))
        for grp in r_groups:
            y0 = grp[0]
            y1 = grp[-1] + 1
            ch = y1 - y0
            if ch > max_thickness_px:
                continue

            band_active = np.any(active_h[y0:y1, :], axis=0)
            x0, x1_incl, density = _longest_active_run(band_active, max_gap=max_gap_x)
            cw = x1_incl - x0 + 1
            rel_len = cw / float(w)
            if rel_len < min_relative_length or density < 0.30:
                continue

            band_thresh = max(8, int(round(h * 0.008)))
            s_type = "BAND" if ch >= band_thresh else "HORIZONTAL"
            yc = y0 + ch * 0.5
            candidates.append(
                LineCandidate(
                    bbox=(x0, y0, x1_incl + 1, y1),
                    endpoints=(float(x0), yc, float(x1_incl + 1), yc),
                    stripe_type=s_type,
                    length_px=float(cw),
                    width_px=float(ch),
                    relative_length=round(rel_len, 4),
                    angle_deg=0.0,
                    detected_by=["projection_profile"],
                )
            )

    return candidates


def _candidates_overlap(a: LineCandidate, b: LineCandidate) -> bool:
    """
    Kiểm tra xem hai ứng viên có biểu diễn cùng một vệt sọc vật lý hay không.
    """
    if _angle_diff_deg(a.angle_deg, b.angle_deg) > 5.0:
        return False

    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox

    # Đối với sọc dọc / band dọc
    if _angle_diff_deg(a.angle_deg, 90.0) <= 5.0:
        acx = 0.5 * (ax0 + ax1)
        bcx = 0.5 * (bx0 + bx1)
        max_w = max(a.width_px, b.width_px, 6.0)
        if abs(acx - bcx) <= max_w * 0.85:
            overlap_y = min(ay1, by1) - max(ay0, by0)
            min_h = max(1, min(ay1 - ay0, by1 - by0))
            if overlap_y >= -25 or (overlap_y / float(min_h)) >= 0.25:
                return True
        return False

    # Đối với sọc ngang / band ngang
    if _angle_diff_deg(a.angle_deg, 0.0) <= 5.0:
        acy = 0.5 * (ay0 + ay1)
        bcy = 0.5 * (by0 + by1)
        max_h = max(a.width_px, b.width_px, 6.0)
        if abs(acy - bcy) <= max_h * 0.85:
            overlap_x = min(ax1, bx1) - max(ax0, bx0)
            min_w = max(1, min(ax1 - ax0, bx1 - bx0))
            if overlap_x >= -25 or (overlap_x / float(min_w)) >= 0.25:
                return True
        return False

    # Đối với sọc nghiêng (DIAGONAL): kiểm tra khoảng cách giữa các trung điểm và IoU bbox
    amx = 0.5 * (a.endpoints[0] + a.endpoints[2])
    amy = 0.5 * (a.endpoints[1] + a.endpoints[3])
    bmx = 0.5 * (b.endpoints[0] + b.endpoints[2])
    bmy = 0.5 * (b.endpoints[1] + b.endpoints[3])
    ang_rad = math.radians(a.angle_deg)
    nx = -math.sin(ang_rad)
    ny = math.cos(ang_rad)
    perp_dist = abs((bmx - amx) * nx + (bmy - amy) * ny)
    return perp_dist <= 8.0


def merge_and_deduplicate_candidates(
    candidates: List[LineCandidate],
    image_shape: Tuple[int, int],
) -> List[LineCandidate]:
    """
    Kết hợp kết quả từ cả 3 phương pháp (Morphological, HoughLinesP, Projection Profile),
    loại bỏ các vùng ứng viên trùng lặp và giữ đầy đủ danh sách phương pháp đã phát hiện (`detected_by`).
    """
    if not candidates:
        return []

    h, w = image_shape[:2]
    sorted_cands = sorted(candidates, key=lambda c: c.relative_length, reverse=True)
    merged_list: List[LineCandidate] = []

    for cand in sorted_cands:
        matched_idx: Optional[int] = None
        for idx, existing in enumerate(merged_list):
            if _candidates_overlap(existing, cand):
                matched_idx = idx
                break

        if matched_idx is None:
            merged_list.append(
                LineCandidate(
                    bbox=cand.bbox,
                    endpoints=cand.endpoints,
                    stripe_type=cand.stripe_type,
                    length_px=cand.length_px,
                    width_px=cand.width_px,
                    relative_length=cand.relative_length,
                    angle_deg=cand.angle_deg,
                    detected_by=list(dict.fromkeys(cand.detected_by)),
                )
            )
        else:
            ex = merged_list[matched_idx]
            combined_methods = list(dict.fromkeys(ex.detected_by + cand.detected_by))
            # Hợp nhất bao hình chữ nhật theo chiều dài lớn hơn, nhưng giữ độ dày chính xác từ morphological/projection
            nx0 = min(ex.bbox[0], cand.bbox[0])
            ny0 = min(ex.bbox[1], cand.bbox[1])
            nx1 = max(ex.bbox[2], cand.bbox[2])
            ny1 = max(ex.bbox[3], cand.bbox[3])

            if ex.stripe_type == "DIAGONAL" or cand.stripe_type == "DIAGONAL":
                chosen = ex if ex.length_px >= cand.length_px else cand
                merged_list[matched_idx] = LineCandidate(
                    bbox=(nx0, ny0, nx1, ny1),
                    endpoints=chosen.endpoints,
                    stripe_type="DIAGONAL",
                    length_px=chosen.length_px,
                    width_px=max(ex.width_px, cand.width_px),
                    relative_length=chosen.relative_length,
                    angle_deg=chosen.angle_deg,
                    detected_by=combined_methods,
                )
            elif _angle_diff_deg(ex.angle_deg, 90.0) <= 5.0:
                # Sọc dọc / Band dọc: ưu tiên chiều rộng của morphological/projection thay vì bbox đệm của Hough
                non_hough = [
                    c for c in (ex, cand) if "hough_lines_p" not in c.detected_by or len(c.detected_by) > 1
                ]
                ref_w_cand = max(non_hough, key=lambda c: c.width_px) if non_hough else ex
                bx0 = ref_w_cand.bbox[0]
                bx1 = ref_w_cand.bbox[2]
                cw = float(max(1, bx1 - bx0))
                ch = float(max(1, ny1 - ny0))
                rel_len = min(1.0, ch / float(h))
                band_thresh = max(8, int(round(w * 0.008)))
                s_type = "BAND" if cw >= band_thresh else "VERTICAL"
                xc = bx0 + cw * 0.5
                merged_list[matched_idx] = LineCandidate(
                    bbox=(bx0, ny0, bx1, ny1),
                    endpoints=(xc, float(ny0), xc, float(ny1)),
                    stripe_type=s_type,
                    length_px=round(ch, 2),
                    width_px=round(cw, 2),
                    relative_length=round(rel_len, 4),
                    angle_deg=90.0,
                    detected_by=combined_methods,
                )
            else:
                non_hough = [
                    c for c in (ex, cand) if "hough_lines_p" not in c.detected_by or len(c.detected_by) > 1
                ]
                ref_h_cand = max(non_hough, key=lambda c: c.width_px) if non_hough else ex
                by0 = ref_h_cand.bbox[1]
                by1 = ref_h_cand.bbox[3]
                ch = float(max(1, by1 - by0))
                cw = float(max(1, nx1 - nx0))
                rel_len = min(1.0, cw / float(w))
                band_thresh = max(8, int(round(h * 0.008)))
                s_type = "BAND" if ch >= band_thresh else "HORIZONTAL"
                yc = by0 + ch * 0.5
                merged_list[matched_idx] = LineCandidate(
                    bbox=(nx0, by0, nx1, by1),
                    endpoints=(float(nx0), yc, float(nx1), yc),
                    stripe_type=s_type,
                    length_px=round(cw, 2),
                    width_px=round(ch, 2),
                    relative_length=round(rel_len, 4),
                    angle_deg=0.0,
                    detected_by=combined_methods,
                )

    return merged_list
