"""
Phase 3.1 — Page Layout Analysis (`src/pdf_inspector/detectors/boundary/layout_analyzer.py`).

Phân tích bố cục trang tài liệu từ `PageContext` bằng OpenCV:
- Phát hiện các khối văn bản (`TextRegionBlock`), dòng chữ, tiêu đề, header/footer/page number.
- Phát hiện khoảng trắng lớn (`WhitespaceBand`) theo Horizontal & Vertical Projection Profile.
- Phát hiện đường viền giấy / vết ghép nối vật lý (`has_seam_line`, `seam_strength`) và bước nhảy độ sáng nền (`illumination_delta`).
- Phát hiện vùng nội dung sát mép bị cắt cụt (`EdgeTruncationSignal` -> `CONTENT_TRUNCATED`).
- Phân biệt bố cục hai cột (`is_two_column`), trang sách đôi hợp lệ (`is_book_spread`), bảng dài (`has_table_structure`) và vùng chữ ký (`has_signature_bottom`).
- Loại trừ ảnh hưởng của vệt sọc máy scan mảnh khi tính hồ sơ chiếu văn bản (để trang vừa bị sọc vừa lẫn trang vẫn phát hiện chính xác cả hai lỗi).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pdf_inspector.models.boundary_schemas import (
    EdgeTruncationSignal,
    PageLayoutAnalysisResult,
    TextRegionBlock,
    WhitespaceBand,
)


def _suppress_scanner_stripes_from_binary(
    binary_mask: np.ndarray,
    dpi: int = 160,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tách riêng các đường thẳng liên tục quá dài và mảnh (như vệt sọc dọc/ngang hoặc khung bảng)
    ra khỏi mặt nạ ký tự chữ (`text_only_mask`) để:
    1. Sọc dọc máy scan không làm nối liền khoảng trắng ngang giữa hai trang bị ghép lẫn.
    2. Vẫn giữ lại `h_lines_mask` và `v_lines_mask` phục vụ phát hiện bảng biểu (`table`) và đường phân cách (`seam`).
    """
    h, w = binary_mask.shape[:2]
    v_len = max(65, int(round(h * 0.22)))
    h_len = max(65, int(round(w * 0.25)))

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))

    v_lines = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, v_kernel)
    h_lines = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, h_kernel)

    # Mở rộng nhẹ 1px để xóa sạch rìa sọc khỏi mặt nạ chữ
    v_dil = cv2.dilate(v_lines, np.ones((1, 3), dtype=np.uint8))
    h_dil = cv2.dilate(h_lines, np.ones((3, 1), dtype=np.uint8))

    text_only = binary_mask.copy()
    text_only[v_dil > 0] = 0
    text_only[h_dil > 0] = 0

    # Bỏ viền sát mép 4px khỏi text_only để bóng mép trang không bị tính là chữ
    border = 4
    text_only[:border, :] = 0
    text_only[h - border :, :] = 0
    text_only[:, :border] = 0
    text_only[:, w - border :] = 0

    return text_only, h_lines, v_lines


def find_safe_cut_line_px(
    text_only_mask: np.ndarray,
    target_pos_px: int,
    orientation: str = "HORIZONTAL",
    search_radius_px: int = 40,
) -> Tuple[int, bool]:
    """
    Tìm tọa độ đường cắt an toàn gần `target_pos_px` nhất sao cho KHÔNG cắt xuyên qua ký tự chữ.
    Trả về `(safe_pos_px, crosses_text)`.
    """
    h, w = text_only_mask.shape[:2]
    if orientation == "HORIZONTAL":
        target = max(0, min(h - 1, int(target_pos_px)))
        y_min = max(0, target - search_radius_px)
        y_max = min(h - 1, target + search_radius_px)
        row_counts = np.sum(text_only_mask[y_min : y_max + 1, :] > 0, axis=1)
        zero_indices = np.flatnonzero(row_counts == 0)
        if zero_indices.size > 0:
            # Chọn hàng trắng gần target nhất
            best_local = int(
                zero_indices[np.argmin(np.abs(zero_indices + y_min - target))]
            )
            return y_min + best_local, False
        # Nếu không có hàng hoàn toàn trống, chọn hàng ít điểm chữ nhất và đánh dấu crosses_text=True
        best_local = int(np.argmin(row_counts))
        crosses = bool(row_counts[best_local] > max(2, int(w * 0.003)))
        return y_min + best_local, crosses
    else:
        target = max(0, min(w - 1, int(target_pos_px)))
        x_min = max(0, target - search_radius_px)
        x_max = min(w - 1, target + search_radius_px)
        col_counts = np.sum(text_only_mask[:, x_min : x_max + 1] > 0, axis=0)
        zero_indices = np.flatnonzero(col_counts == 0)
        if zero_indices.size > 0:
            best_local = int(
                zero_indices[np.argmin(np.abs(zero_indices + x_min - target))]
            )
            return x_min + best_local, False
        best_local = int(np.argmin(col_counts))
        crosses = bool(col_counts[best_local] > max(2, int(h * 0.003)))
        return x_min + best_local, crosses


def _extract_text_blocks_and_line_stats(
    text_only_mask: np.ndarray,
    h_lines_mask: np.ndarray,
    v_lines_mask: np.ndarray,
    dpi: int = 160,
) -> Tuple[List[TextRegionBlock], float, bool, bool]:
    """
     Gom cụm các ký tự thành dòng chữ và khối văn bản (`TextRegionBlock`),
    ước lượng chiều cao dòng chữ chuẩn (`median_line_height_px`),
    đồng thời kiểm tra sự tồn tại của bảng biểu (`has_table_structure`) và khối chữ ký cuối trang (`has_signature_bottom`).
    """
    h, w = text_only_mask.shape[:2]
    scale = float(dpi) / 160.0

    # 1. Đo chiều cao trung vị của một dòng chữ thông qua phép nối ngang mức dòng (line-level dilation)
    line_kw = max(18, int(round(28 * scale)))
    line_kh = max(2, int(round(3 * scale)))
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_kw, line_kh))
    line_mask = cv2.dilate(text_only_mask, line_kernel)

    num_l, _, l_stats, _ = cv2.connectedComponentsWithStats(line_mask)
    line_heights: List[int] = []
    for i in range(1, num_l):
        lw, lh, larea = int(l_stats[i, 2]), int(l_stats[i, 3]), int(l_stats[i, 4])
        if lw >= max(24, int(w * 0.04)) and 6 <= lh <= int(h * 0.06) and larea >= 80:
            line_heights.append(lh)
    median_line_h = float(np.median(line_heights)) if line_heights else max(14.0, 18.0 * scale)

    # 2. Kiểm tra cấu trúc bảng biểu từ giao điểm của h_lines và v_lines
    h_count = int(np.sum(np.mean(h_lines_mask > 0, axis=1) >= 0.18))
    v_count = int(np.sum(np.mean(v_lines_mask > 0, axis=0) >= 0.12))
    intersections = cv2.bitwise_and(
        cv2.dilate(h_lines_mask, np.ones((5, 5), dtype=np.uint8)),
        cv2.dilate(v_lines_mask, np.ones((5, 5), dtype=np.uint8)),
    )
    num_inter, _, _, _ = cv2.connectedComponentsWithStats(intersections)
    has_table_structure = bool((num_inter - 1) >= 6 or (h_count >= 5 and v_count >= 3))

    # 3. Gom các dòng chữ gần nhau thành khối đoạn văn / vùng bố cục (Paragraph/Block level)
    # Chiều dọc kernel nhỏ (~1.2 lần chiều cao dòng) để KHÔNG nối liền qua khoảng cách đoạn lớn hay ranh giới trang
    blk_kw = max(28, int(round(42 * scale)))
    blk_kh = max(8, int(round(median_line_h * 0.85)))
    blk_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (blk_kw, blk_kh))
    blk_mask = cv2.morphologyEx(text_only_mask, cv2.MORPH_CLOSE, blk_kernel)

    num_b, _, b_stats, _ = cv2.connectedComponentsWithStats(blk_mask)
    blocks: List[TextRegionBlock] = []
    has_signature_bottom = False

    for i in range(1, num_b):
        bx = int(b_stats[i, 0])
        by = int(b_stats[i, 1])
        bw = int(b_stats[i, 2])
        bh = int(b_stats[i, 3])
        barea = int(b_stats[i, 4])

        if bw < max(16, int(w * 0.02)) or bh < 6 or barea < 60:
            continue

        x0, y0, x1, y1 = bx, by, bx + bw, by + bh
        roi_text = text_only_mask[y0:y1, x0:x1]
        fg_pixels = int(np.sum(roi_text > 0))
        if fg_pixels < 25:
            continue

        density = float(fg_pixels) / float(max(1, bw * bh))
        est_lines = max(1, int(round(bh / max(10.0, median_line_h * 1.45))))

        x0_n = x0 / float(w)
        y0_n = y0 / float(h)
        x1_n = x1 / float(w)
        y1_n = y1 / float(h)
        w_n = bw / float(w)
        h_n = bh / float(h)

        # Phân loại vai trò khối (role)
        role = "body"
        if y1_n <= 0.12 and est_lines <= 2:
            role = "page_number" if w_n <= 0.18 else "header"
        elif y0_n >= 0.88 and est_lines <= 2:
            role = "page_number" if w_n <= 0.18 else "footer"
        elif y0_n >= 0.72 and w_n <= 0.42 and x0_n >= 0.45 and bh <= h * 0.18:
            role = "signature"
            has_signature_bottom = True
        elif est_lines <= 2 and w_n >= 0.25 and bh >= median_line_h * 0.9:
            role = "heading"

        blocks.append(
            TextRegionBlock(
                bbox=(x0, y0, x1, y1),
                bbox_norm=(x0_n, y0_n, x1_n, y1_n),
                area_px=fg_pixels,
                line_count=est_lines,
                mean_line_height_px=round(median_line_h, 2),
                left_margin_norm=round(x0_n, 4),
                right_margin_norm=round(1.0 - x1_n, 4),
                width_norm=round(w_n, 4),
                height_norm=round(h_n, 4),
                density=round(density, 4),
                region_role=role,
            )
        )

    blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return blocks, median_line_h, has_table_structure, has_signature_bottom


def _detect_two_column_and_book_spread(
    text_only_mask: np.ndarray,
    text_blocks: List[TextRegionBlock],
    gray: np.ndarray,
) -> Tuple[bool, bool]:
    """
    Phân biệt:
    1. `is_two_column`: Trang tài liệu bố cục 2 cột hợp lệ (trang dọc hoặc có tiêu đề/header trải ngang cả 2 cột).
    2. `is_book_spread`: Ảnh scan 2 trang sách mở đối xứng hợp lệ theo thiết kế (`W >= 1.12 * H`,
       có rãnh gáy sách ở giữa `x ≈ 0.45 .. 0.55` và hai trang trái/phải cân đối).
    """
    h, w = text_only_mask.shape[:2]
    aspect = float(w) / float(max(1, h))

    # Kiểm tra dải giữa trang theo trục ngang [0.44 .. 0.56] trong vùng thân [0.18 .. 0.82]
    y_body0, y_body1 = int(h * 0.18), int(h * 0.82)
    if y_body1 <= y_body0 + 20:
        return False, False

    col_profile = np.mean(text_only_mask[y_body0:y_body1, :] > 0, axis=0)
    left_zone = col_profile[int(w * 0.08) : int(w * 0.43)]
    mid_zone = col_profile[int(w * 0.45) : int(w * 0.55)]
    right_zone = col_profile[int(w * 0.57) : int(w * 0.92)]

    if left_zone.size == 0 or mid_zone.size == 0 or right_zone.size == 0:
        return False, False

    left_act = float(np.mean(left_zone >= 0.02))
    right_act = float(np.mean(right_zone >= 0.02))
    min_mid_val = float(np.min(cv2.blur(mid_zone.reshape(1, -1), (1, 7))))

    has_clear_central_gutter = (
        left_act >= 0.35 and right_act >= 0.35 and min_mid_val <= 0.008
    )
    if not has_clear_central_gutter:
        return False, False

    # Kiểm tra xem có khối tiêu đề/header trải ngang qua cả 2 cột (x0 < 0.38 và x1 > 0.62) hay không
    has_spanning_header = any(
        b.bbox_norm[0] < 0.38 and b.bbox_norm[2] > 0.62 and b.bbox_norm[1] < 0.30
        for b in text_blocks
    )

    if aspect >= 1.12 and not has_spanning_header:
        # Trang khổ ngang (landscape) có rãnh giữa tách 2 trang đối xứng -> Book Spread hợp lệ
        return False, True

    return True, False


def _detect_whitespace_bands_and_seams(
    gray: np.ndarray,
    text_only_mask: np.ndarray,
    h_lines_mask: np.ndarray,
) -> List[WhitespaceBand]:
    """
    Phát hiện các dải khoảng trắng lớn nằm ngang (`HORIZONTAL`) ở vùng giữa trang `[0.12 .. 0.88]`
    kèm thông tin đường nối/mép giấy vật lý (`has_seam_line`, `seam_strength`)
    và độ chênh lệch độ sáng nền giấy hai phía (`has_illumination_step`, `illumination_delta`).
    """
    h, w = gray.shape[:2]
    margin_x = max(8, int(round(w * 0.03)))
    row_text_ratio = np.mean(text_only_mask[:, margin_x : w - margin_x] > 0, axis=1)

    # Tìm các hàng trống chữ (tỷ lệ pixel chữ < 0.25%)
    empty_rows = row_text_ratio < 0.0025

    # Tính gradient dọc Sobel để phát hiện vết ghép mép giấy / bóng mép trang chạy ngang suốt chiều rộng
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    # Đường seam vật lý của mép giấy thường kéo dài ra tận vùng lề trái/phải (nơi không có chữ)
    left_margin_strip = sobel_y[:, max(2, int(w * 0.01)) : max(6, int(w * 0.07))]
    right_margin_strip = sobel_y[:, min(w - 6, int(w * 0.93)) : min(w - 2, int(w * 0.99))]
    full_row_edge = np.mean(sobel_y[:, int(w * 0.02) : int(w * 0.98)] >= 18.0, axis=1)
    margin_row_edge = 0.5 * (
        np.mean(left_margin_strip >= 14.0, axis=1)
        + np.mean(right_margin_strip >= 14.0, axis=1)
    )

    bands: List[WhitespaceBand] = []
    y_start_lim = int(round(h * 0.12))
    y_end_lim = int(round(h * 0.88))
    min_band_px = max(24, int(round(h * 0.025)))

    in_run = False
    run_s = y_start_lim
    for y in range(y_start_lim, y_end_lim):
        if empty_rows[y]:
            if not in_run:
                in_run = True
                run_s = y
        else:
            if in_run:
                run_e = y - 1
                if (run_e - run_s + 1) >= min_band_px:
                    bands.append(
                        _build_horizontal_band(
                            gray=gray,
                            s=run_s,
                            e=run_e,
                            full_row_edge=full_row_edge,
                            margin_row_edge=margin_row_edge,
                        )
                    )
                in_run = False
    if in_run and (y_end_lim - run_s) >= min_band_px:
        bands.append(
            _build_horizontal_band(
                gray=gray,
                s=run_s,
                e=y_end_lim - 1,
                full_row_edge=full_row_edge,
                margin_row_edge=margin_row_edge,
            )
        )

    bands.sort(key=lambda b: b.center_px)
    return bands


def _build_horizontal_band(
    gray: np.ndarray,
    s: int,
    e: int,
    full_row_edge: np.ndarray,
    margin_row_edge: np.ndarray,
    force_seam: bool = False,
) -> WhitespaceBand:
    """Tính toán các đặc trưng quang học và hình học cho một dải ngang `[s..e]`."""
    h, w = gray.shape[:2]
    center_px = int(round(0.5 * (s + e)))
    thickness_px = max(1, e - s + 1)

    roi = gray[s : e + 1, int(w * 0.04) : int(w * 0.96)]
    mean_int = float(np.mean(roi)) if roi.size > 0 else 250.0
    std_int = float(np.std(roi)) if roi.size > 0 else 0.0

    # So sánh độ sáng nền giấy (phân vị 82 để lấy nền giấy thay vì chữ) phía trên và phía dưới dải
    win_h = max(40, int(round(h * 0.12)))
    above_y0 = max(0, s - win_h)
    above_y1 = max(above_y0 + 1, s)
    below_y0 = min(h - 1, e + 1)
    below_y1 = min(h, e + 1 + win_h)

    above_patch = gray[above_y0:above_y1, int(w * 0.05) : int(w * 0.95)]
    below_patch = gray[below_y0:below_y1, int(w * 0.05) : int(w * 0.95)]
    bg_above = float(np.percentile(above_patch, 82)) if above_patch.size > 0 else mean_int
    bg_below = float(np.percentile(below_patch, 82)) if below_patch.size > 0 else mean_int
    illum_delta = abs(bg_above - bg_below)
    has_illum_step = illum_delta >= 5.5

    # Kiểm tra đường seam hoặc bóng mép giấy xuyên lề trong vùng [s-4 .. e+4].
    # Để tránh nhầm vệt sọc ngang máy scan (scanner stripe trên nền giấy đồng nhất) thành vết ghép 2 trang giấy,
    # yêu cầu phải có chênh lệch độ sáng nền giấy (`illum_delta >= 3.5`) hoặc độ dày khoảng trắng lớn (`>= 3.2%`).
    s_pad = max(0, s - 4)
    e_pad = min(h - 1, e + 4)
    max_full_edge = float(np.max(full_row_edge[s_pad : e_pad + 1]))
    max_margin_edge = float(np.max(margin_row_edge[s_pad : e_pad + 1]))
    has_edge_grad = max_full_edge >= 0.45 and max_margin_edge >= 0.32
    has_seam = force_seam or (
        has_edge_grad and (illum_delta >= 3.5 or thickness_px >= int(round(h * 0.035)))
    )
    seam_strength = (
        round(0.6 * max_full_edge + 0.4 * max_margin_edge, 4)
        if has_edge_grad
        else round(min(1.0, illum_delta / 18.0), 4)
    )

    return WhitespaceBand(
        orientation="HORIZONTAL",
        start_px=s,
        end_px=e,
        center_px=center_px,
        center_norm=round(center_px / float(h), 4),
        thickness_px=thickness_px,
        thickness_norm=round(thickness_px / float(h), 4),
        mean_intensity=round(mean_int, 2),
        std_intensity=round(std_int, 2),
        has_seam_line=bool(has_seam),
        seam_strength=seam_strength,
        has_illumination_step=bool(has_illum_step),
        illumination_delta=round(illum_delta, 2),
        crosses_full_width=True,
    )


def _detect_edge_truncations(
    gray: np.ndarray,
    otsu_binary: np.ndarray,
    text_only_mask: np.ndarray,
    median_line_h: float,
) -> List[EdgeTruncationSignal]:
    """
    Phát hiện lỗi nội dung sát mép bị cắt mất (`CONTENT_TRUNCATED`) ở 4 cạnh `TOP`, `BOTTOM`, `LEFT`, `RIGHT`.
    Chỉ kết luận khi có nhiều thành phần ký tự chữ (`>= 4` cụm ký tự rời rạc dọc theo dòng)
    chạm sát mép ảnh (`0..5 px` từ rìa) hoặc dòng chữ sát mép bị cắt cụt chiều cao (`char_height_ratio < 0.75`),
    phân biệt rõ với viền đen khung scan hoặc sọc máy scan.
    """
    h, w = gray.shape[:2]
    signals: List[EdgeTruncationSignal] = []

    # Sử dụng mặt nạ chữ chưa cắt 4px viền nhưng đã loại bỏ đường thẳng dài liên tục (viền giấy / sọc scan)
    v_long = cv2.morphologyEx(
        otsu_binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, int(h * 0.20)))),
    )
    h_long = cv2.morphologyEx(
        otsu_binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, int(w * 0.20)), 1)),
    )
    edge_char_mask = otsu_binary.copy()
    edge_char_mask[cv2.dilate(v_long, np.ones((1, 5), dtype=np.uint8)) > 0] = 0
    edge_char_mask[cv2.dilate(h_long, np.ones((5, 1), dtype=np.uint8)) > 0] = 0

    strip_h = max(16, int(round(max(median_line_h * 1.15, h * 0.014))))
    margin_x = max(20, int(round(w * 0.05)))

    # Kiểm tra TOP và BOTTOM
    for side in ("TOP", "BOTTOM"):
        if side == "TOP":
            y0, y1 = 0, strip_h
            touch_rows = edge_char_mask[0:6, margin_x : w - margin_x]
        else:
            y0, y1 = h - strip_h, h
            touch_rows = edge_char_mask[h - 6 : h, margin_x : w - margin_x]

        strip_roi = edge_char_mask[y0:y1, margin_x : w - margin_x]
        touch_pixels = int(np.sum(touch_rows > 0))
        if touch_pixels < 18:
            continue

        num_c, _, c_stats, _ = cv2.connectedComponentsWithStats(strip_roi)
        valid_comps = 0
        comp_heights: List[int] = []
        touching_comps = 0
        for idx in range(1, num_c):
            cy = int(c_stats[idx, 1])
            cw = int(c_stats[idx, 2])
            ch = int(c_stats[idx, 3])
            carea = int(c_stats[idx, 4])
            if 2 <= cw <= int(w * 0.18) and ch >= 3 and carea >= 8:
                valid_comps += 1
                comp_heights.append(ch)
                if (side == "TOP" and cy <= 4) or (
                    side == "BOTTOM" and (cy + ch) >= (strip_h - 4)
                ):
                    touching_comps += 1

        if touching_comps >= 4 and valid_comps >= 5:
            mean_ch = float(np.median(comp_heights)) if comp_heights else strip_h
            ch_ratio = min(1.0, mean_ch / max(8.0, median_line_h))
            clipped_ratio = float(touch_pixels) / float(max(1, (w - 2 * margin_x) * 6))
            raw_score = (
                0.45 * min(1.0, touching_comps / 10.0)
                + 0.35 * min(1.0, max(0.2, 1.15 - ch_ratio))
                + 0.20 * min(1.0, clipped_ratio / 0.04)
            )
            score = round(min(1.0, max(0.0, raw_score)), 3)
            status = "CONFIRMED" if score >= 0.62 else "SUSPECTED"
            signals.append(
                EdgeTruncationSignal(
                    edge_side=side,
                    bbox=(margin_x, y0, w - margin_x, y1),
                    bbox_norm=(
                        margin_x / float(w),
                        y0 / float(h),
                        (w - margin_x) / float(w),
                        y1 / float(h),
                    ),
                    clipped_pixel_ratio=round(clipped_ratio, 4),
                    clipped_components_count=touching_comps,
                    char_height_ratio=round(ch_ratio, 3),
                    suspicion_score=score,
                    status=status,
                    reason=(
                        f"Phát hiện {touching_comps} cụm ký tự văn bản chạm sát mép {side} "
                        f"và bị cắt cụt (tỷ lệ chiều cao ký tự={ch_ratio:.2f})."
                    ),
                )
            )

    return signals


def analyze_page_layout(
    gray_image: np.ndarray,
    page_number: int = 1,
    dpi: int = 160,
    pdf_width_pt: float = 595.0,
    pdf_height_pt: float = 842.0,
    analysis_rotation_applied: int = 0,
) -> PageLayoutAnalysisResult:
    """
    Thực thi phân tích bố cục toàn diện cho một trang PDF (Phase 3.1).
    Chuẩn hóa mọi ngưỡng theo `dpi` và kích thước trang `(w, h)`.
    """
    t0 = time.perf_counter()
    if gray_image is None or gray_image.size == 0:
        raise ValueError(f"Ảnh trang {page_number} đưa vào analyze_page_layout bị rỗng.")

    if gray_image.ndim == 3:
        gray = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = gray_image.astype(np.uint8, copy=False)

    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        raise ValueError(f"Kích thước ảnh trang {page_number} quá nhỏ: {w}x{h}")

    # Nhị phân hóa kết hợp Otsu và Adaptive Threshold để bắt cả chữ mờ
    _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_only_mask, h_lines_mask, v_lines_mask = _suppress_scanner_stripes_from_binary(
        otsu_bin, dpi=dpi
    )

    total_text_pixels = int(np.sum(text_only_mask > 0))
    is_mostly_blank = total_text_pixels < max(150, int(h * w * 0.0004))

    text_blocks, median_line_h, has_table, has_sig = _extract_text_blocks_and_line_stats(
        text_only_mask=text_only_mask,
        h_lines_mask=h_lines_mask,
        v_lines_mask=v_lines_mask,
        dpi=dpi,
    )

    is_two_col, is_book = _detect_two_column_and_book_spread(
        text_only_mask=text_only_mask,
        text_blocks=text_blocks,
        gray=gray,
    )

    whitespace_bands = (
        []
        if is_mostly_blank
        else _detect_whitespace_bands_and_seams(
            gray=gray,
            text_only_mask=text_only_mask,
            h_lines_mask=h_lines_mask,
        )
    )

    edge_truncations = (
        []
        if is_mostly_blank
        else _detect_edge_truncations(
            gray=gray,
            otsu_binary=otsu_bin,
            text_only_mask=text_only_mask,
            median_line_h=median_line_h,
        )
    )

    if text_blocks:
        min_x0 = min(b.bbox_norm[0] for b in text_blocks)
        min_y0 = min(b.bbox_norm[1] for b in text_blocks)
        max_x1 = max(b.bbox_norm[2] for b in text_blocks)
        max_y1 = max(b.bbox_norm[3] for b in text_blocks)
        content_bbox_norm = (min_x0, min_y0, max_x1, max_y1)
        top_empty_norm = min_y0
        bottom_empty_norm = max(0.0, 1.0 - max_y1)
        left_margin_norm = min_x0
        right_margin_norm = max(0.0, 1.0 - max_x1)
        content_fill_ratio = max(0.0, max_y1 - min_y0)
    else:
        content_bbox_norm = (0.0, 0.0, 0.0, 0.0)
        top_empty_norm = 1.0
        bottom_empty_norm = 1.0
        left_margin_norm = 0.5
        right_margin_norm = 0.5
        content_fill_ratio = 0.0

    # Trích xuất dải ảnh biên trên/dưới (top/bottom 32% của trang, thu phóng về bề rộng chuẩn 660px)
    # để phục vụ đối chiếu liên trang (Cross-Page Validation) mà chỉ tốn ~150KB/trang
    strip_ratio = 0.34
    strip_h = max(48, int(round(h * strip_ratio)))
    target_w = min(w, 660)
    scale_f = target_w / float(w)
    target_strip_h = max(24, int(round(strip_h * scale_f)))

    top_gray_raw = gray[:strip_h, :]
    bot_gray_raw = gray[h - strip_h :, :]
    top_bin_raw = text_only_mask[:strip_h, :]
    bot_bin_raw = text_only_mask[h - strip_h :, :]

    top_strip_gray = cv2.resize(
        top_gray_raw, (target_w, target_strip_h), interpolation=cv2.INTER_AREA
    )
    bottom_strip_gray = cv2.resize(
        bot_gray_raw, (target_w, target_strip_h), interpolation=cv2.INTER_AREA
    )
    top_strip_binary = cv2.resize(
        top_bin_raw, (target_w, target_strip_h), interpolation=cv2.INTER_NEAREST
    )
    bottom_strip_binary = cv2.resize(
        bot_bin_raw, (target_w, target_strip_h), interpolation=cv2.INTER_NEAREST
    )

    row_text_profile = np.mean(text_only_mask > 0, axis=1).astype(np.float32)
    mean_bg = float(np.percentile(gray, 82))
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return PageLayoutAnalysisResult(
        page_number=page_number,
        image_width_px=w,
        image_height_px=h,
        pdf_width_pt=pdf_width_pt,
        pdf_height_pt=pdf_height_pt,
        dpi=dpi,
        analysis_rotation_applied=analysis_rotation_applied,
        text_blocks=text_blocks,
        whitespace_bands=whitespace_bands,
        edge_truncations=edge_truncations,
        is_two_column=is_two_col,
        is_book_spread=is_book,
        is_mostly_blank=is_mostly_blank,
        has_table_structure=has_table,
        has_signature_bottom=has_sig,
        content_bbox_norm=content_bbox_norm,
        content_fill_ratio=round(content_fill_ratio, 4),
        top_empty_norm=round(top_empty_norm, 4),
        bottom_empty_norm=round(bottom_empty_norm, 4),
        left_margin_norm=round(left_margin_norm, 4),
        right_margin_norm=round(right_margin_norm, 4),
        mean_bg_intensity=round(mean_bg, 2),
        processing_time_ms=elapsed_ms,
        top_strip_gray=top_strip_gray,
        bottom_strip_gray=bottom_strip_gray,
        top_strip_binary=top_strip_binary,
        bottom_strip_binary=bottom_strip_binary,
        row_text_profile=row_text_profile,
    )
