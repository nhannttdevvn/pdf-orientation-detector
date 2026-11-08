"""
Phase 3.4 — Cross-Page Validation (`src/pdf_inspector/detectors/boundary/cross_page_validator.py`).

Kiểm tra quan hệ liên trang giữa trang `N` với trang liền trước (`N-1`) và trang liền sau (`N+1`)
thông qua cửa sổ trượt của `PageLayoutAnalysisResult`:
1. Phát hiện trùng lặp / chồng lấp nội dung (`CONTENT_OVERLAP`) giữa phần cuối trang `N` và phần đầu trang `N+1`
   bằng kết hợp:
   - Normalized Cross-Correlation (`cv2.matchTemplate` với `TM_CCOEFF_NORMED`)
   - Tương quan hồ sơ chiếu ngang/dọc (`Projection Profile Correlation`)
   - Đối sánh đặc trưng cục bộ ORB (`cv2.ORB_create`) kiểm tra độ nhất quán dịch chuyển hình học
2. Phát hiện một trang vật lý bị chia đôi qua hai trang PDF liền kề (`SPLIT_CONTINUATION` -> `SPLIT_PAGE`).
3. Xác minh hiện tượng tràn/dính trang giữa hai trang liền kề (`MIXED_SPILLOVER` -> `MIXED_PAGE` liên kết trang `N` và `N+1`/`N-1`).
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ai_engine.core.config import BOUNDARY_OVERLAP_THRESHOLD
from backend.schemas.boundary_schemas import (
    CrossPageMatchResult,
    PageLayoutAnalysisResult,
)


def _compute_orb_translation_inlier_ratio(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
) -> float:
    """
    Tính tỷ lệ cặp đặc trưng ORB khớp nhau có cùng vector dịch chuyển phẳng `(dx, dy)`
    giữa hai dải ảnh `patch_a` và `patch_b`.
    """
    if patch_a is None or patch_b is None or patch_a.size == 0 or patch_b.size == 0:
        return 0.0
    if min(patch_a.shape[:2]) < 24 or min(patch_b.shape[:2]) < 24:
        return 0.0

    try:
        orb = cv2.ORB_create(nfeatures=180, scaleFactor=1.2, nlevels=4, edgeThreshold=8)
        kp1, des1 = orb.detectAndCompute(patch_a, None)
        kp2, des2 = orb.detectAndCompute(patch_b, None)
        if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 5:
            return 0.0

        good_matches = [m for m in matches if m.distance <= 48]
        if len(good_matches) < 4:
            return 0.0

        dxs = np.array(
            [kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0] for m in good_matches],
            dtype=np.float32,
        )
        dys = np.array(
            [kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1] for m in good_matches],
            dtype=np.float32,
        )
        med_dx = float(np.median(dxs))
        med_dy = float(np.median(dys))
        inliers = int(np.sum((np.abs(dxs - med_dx) <= 6.0) & (np.abs(dys - med_dy) <= 6.0)))
        return round(float(inliers) / float(max(8, len(good_matches))), 4)
    except Exception:
        return 0.0


def _compute_profile_correlation(bin_a: np.ndarray, bin_b: np.ndarray) -> float:
    """Tính hệ số tương quan Pearson trung bình của hồ sơ chiếu ngang và dọc giữa hai dải nhị phân cùng kích thước."""
    if bin_a.shape != bin_b.shape or bin_a.size == 0:
        return 0.0

    row_a = np.mean(bin_a > 0, axis=1).astype(np.float32)
    row_b = np.mean(bin_b > 0, axis=1).astype(np.float32)
    col_a = np.mean(bin_a > 0, axis=0).astype(np.float32)
    col_b = np.mean(bin_b > 0, axis=0).astype(np.float32)

    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        sx = float(np.std(x))
        sy = float(np.std(y))
        if sx < 1e-5 or sy < 1e-5:
            return 0.0
        c = float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (sx * sy))
        return max(0.0, min(1.0, c))

    return round(0.5 * (_corr(row_a, row_b) + _corr(col_a, col_b)), 4)


def detect_overlap_between_adjacent_pages(
    page_curr: PageLayoutAnalysisResult,
    page_next: PageLayoutAnalysisResult,
    overlap_threshold: float = BOUNDARY_OVERLAP_THRESHOLD,
) -> Optional[CrossPageMatchResult]:
    """
    Kiểm tra xem phần cuối của `page_curr` (trang N) có bị lặp lại/chồng lấp ở phần đầu
    của `page_next` (trang N+1) hay không (`CONTENT_OVERLAP`).
    """
    bot_gray = page_curr.bottom_strip_gray
    top_gray = page_next.top_strip_gray
    bot_bin = page_curr.bottom_strip_binary
    top_bin = page_next.top_strip_binary

    if (
        bot_gray is None
        or top_gray is None
        or bot_bin is None
        or top_bin is None
        or bot_gray.shape != top_gray.shape
    ):
        return None

    sh, sw = bot_gray.shape[:2]
    # Tìm vùng có chữ thực tế trong dải dưới của trang N và dải trên của trang N+1
    bot_row_act = np.mean(bot_bin > 0, axis=1)
    top_row_act = np.mean(top_bin > 0, axis=1)

    bot_nz = np.flatnonzero(bot_row_act >= 0.008)
    top_nz = np.flatnonzero(top_row_act >= 0.008)
    if bot_nz.size < 8 or top_nz.size < 8:
        return None

    # Lấy cửa sổ mẫu từ phần nội dung cuối của trang N (cao khoảng 35%..65% chiều cao strip, tức ~12%..22% chiều cao trang)
    bot_end = int(bot_nz[-1])
    best_ncc = 0.0
    best_prof = 0.0
    best_orb = 0.0
    best_tpl_y0 = 0
    best_tpl_y1 = 0
    best_match_y0 = 0
    best_match_y1 = 0

    margin_x = int(sw * 0.06)
    search_img = top_gray[:, margin_x : sw - margin_x]

    for win_ratio in (0.38, 0.50, 0.64):
        win_h = max(24, int(round(sh * win_ratio)))
        tpl_y1 = min(sh, bot_end + 4)
        tpl_y0 = max(0, tpl_y1 - win_h)
        if tpl_y1 - tpl_y0 < 20 or (tpl_y1 - tpl_y0) >= sh - 4:
            continue

        tpl_bin = bot_bin[tpl_y0:tpl_y1, margin_x : sw - margin_x]
        if float(np.mean(tpl_bin > 0)) < 0.015:
            continue

        tpl_gray = bot_gray[tpl_y0:tpl_y1, margin_x : sw - margin_x]
        if float(np.std(tpl_gray)) < 12.0:
            continue

        res = cv2.matchTemplate(search_img, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        ncc = float(max_val)

        if ncc > best_ncc:
            my0 = int(max_loc[1])
            my1 = my0 + (tpl_y1 - tpl_y0)
            matched_bin = top_bin[my0:my1, margin_x : sw - margin_x]
            matched_gray = top_gray[my0:my1, margin_x : sw - margin_x]
            prof_c = _compute_profile_correlation(tpl_bin, matched_bin)
            orb_r = (
                _compute_orb_translation_inlier_ratio(tpl_gray, matched_gray)
                if ncc >= 0.45
                else 0.0
            )
            best_ncc = ncc
            best_prof = prof_c
            best_orb = orb_r
            best_tpl_y0, best_tpl_y1 = tpl_y0, tpl_y1
            best_match_y0, best_match_y1 = my0, my1

    combined = round(0.55 * best_ncc + 0.25 * best_prof + 0.20 * max(best_ncc, best_orb), 3)
    if combined < max(0.56, overlap_threshold - 0.08) or best_ncc < 0.55:
        return None

    # Quy đổi tọa độ từ strip về ảnh trang gốc của page_curr và page_next
    strip_ratio = 0.34
    curr_h, curr_w = page_curr.image_height_px, page_curr.image_width_px
    next_h, next_w = page_next.image_height_px, page_next.image_width_px

    curr_strip_top_px = int(round(curr_h * (1.0 - strip_ratio)))
    curr_strip_h_px = curr_h - curr_strip_top_px
    src_y0 = curr_strip_top_px + int(round((best_tpl_y0 / float(sh)) * curr_strip_h_px))
    src_y1 = curr_strip_top_px + int(round((best_tpl_y1 / float(sh)) * curr_strip_h_px))

    next_strip_h_px = int(round(next_h * strip_ratio))
    dst_y0 = int(round((best_match_y0 / float(sh)) * next_strip_h_px))
    dst_y1 = int(round((best_match_y1 / float(sh)) * next_strip_h_px))

    overlap_h_norm = round((src_y1 - src_y0) / float(max(1, curr_h)), 4)
    confirmed = bool(combined >= overlap_threshold and best_ncc >= 0.65)

    return CrossPageMatchResult(
        source_page=page_curr.page_number,
        target_page=page_next.page_number,
        relation_type="CONTENT_OVERLAP",
        ncc_score=round(best_ncc, 4),
        orb_inlier_ratio=round(best_orb, 4),
        profile_correlation=round(best_prof, 4),
        width_consistency_score=1.0,
        complementary_height_score=overlap_h_norm,
        overlap_bbox_source_px=(0, src_y0, curr_w, src_y1),
        overlap_bbox_target_px=(0, dst_y0, next_w, dst_y1),
        overlap_height_norm=overlap_h_norm,
        combined_score=combined,
        confirmed=confirmed,
        reason=(
            f"Phát hiện vùng nội dung trùng lặp (CONTENT_OVERLAP) giữa cuối trang {page_curr.page_number} "
            f"và đầu trang {page_next.page_number} (NCC={best_ncc:.2f}, profile_corr={best_prof:.2f}, "
            f"ORB_inliers={best_orb:.2f}, chiều cao lặp={overlap_h_norm:.2f})."
        ),
    )


def _compute_margin_width_consistency(
    page_a: PageLayoutAnalysisResult,
    page_b: PageLayoutAnalysisResult,
) -> float:
    """Đánh giá mức độ tương thích về lề trái/phải và bề rộng cột chữ giữa hai trang liền kề."""
    if page_a.is_mostly_blank or page_b.is_mostly_blank:
        return 0.0
    left_diff = abs(page_a.left_margin_norm - page_b.left_margin_norm)
    right_diff = abs(page_a.right_margin_norm - page_b.right_margin_norm)
    w_a = max(0.0, page_a.content_bbox_norm[2] - page_a.content_bbox_norm[0])
    w_b = max(0.0, page_b.content_bbox_norm[2] - page_b.content_bbox_norm[0])
    w_diff = abs(w_a - w_b)
    score = 1.0 - min(1.0, (left_diff + right_diff + w_diff) / 0.28)
    return round(max(0.0, score), 4)


def detect_split_or_spillover_between_adjacent_pages(
    page_curr: PageLayoutAnalysisResult,
    page_next: PageLayoutAnalysisResult,
) -> Optional[CrossPageMatchResult]:
    """
    Kiểm tra hai kịch bản liên trang giữa `page_curr` (trang N) và `page_next` (trang N+1):
    1. `MIXED_SPILLOVER`: Trang N có ứng viên ranh giới ngang (`BoundaryCandidate` ở phần dưới `y >= 0.52`)
       hoặc phần đầu trang N+1 bị thiếu/bắt đầu lưng chừng (phần đầu của trang N+1 đã bị kéo sang cuối trang N,
       hoặc phần cuối của trang N bị đẩy sang đầu trang N+1).
    2. `SPLIT_CONTINUATION` (`SPLIT_PAGE`): Một trang vật lý bị cắt làm hai phần nằm trên trang N và trang N+1.
    """
    if page_curr.is_mostly_blank or page_next.is_mostly_blank:
        return None
    if page_curr.is_book_spread or page_next.is_book_spread:
        return None

    width_cons = _compute_margin_width_consistency(page_curr, page_next)
    if width_cons < 0.65:
        return None

    # Kịch bản A: MIXED_SPILLOVER — Trang N có ranh giới trang ngang hoặc trang N+1 có ranh giới trang ngang ở phần đầu
    curr_lower_cands = [
        c
        for c in page_curr.validated_candidates
        if c.orientation == "HORIZONTAL" and 0.48 <= c.position <= 0.88
    ]
    next_upper_cands = [
        c
        for c in page_next.validated_candidates
        if c.orientation == "HORIZONTAL" and 0.12 <= c.position <= 0.48
    ]
    next_has_top_trunc = any(t.edge_side == "TOP" for t in page_next.edge_truncations)
    curr_has_bot_trunc = any(t.edge_side == "BOTTOM" for t in page_curr.edge_truncations)

    if curr_lower_cands and (
        next_has_top_trunc
        or page_next.content_fill_ratio <= 0.68
        or page_next.top_empty_norm >= 0.14
        or curr_lower_cands[0].suspicion_score >= 0.65
    ):
        top_cand = curr_lower_cands[0]
        spilled_h_norm = max(0.05, 1.0 - top_cand.position - page_curr.bottom_empty_norm)
        comp_sum = spilled_h_norm + page_next.content_fill_ratio
        comp_score = (
            1.0
            if 0.45 <= comp_sum <= 1.15
            else max(0.4, 1.0 - abs(comp_sum - 0.80))
        )
        comb = round(
            min(
                1.0,
                0.50 * top_cand.suspicion_score
                + 0.25 * width_cons
                + 0.25 * comp_score
                + (0.10 if next_has_top_trunc else 0.0),
            ),
            3,
        )
        return CrossPageMatchResult(
            source_page=page_curr.page_number,
            target_page=page_next.page_number,
            relation_type="MIXED_SPILLOVER",
            width_consistency_score=width_cons,
            complementary_height_score=round(comp_score, 4),
            combined_score=comb,
            confirmed=bool(comb >= 0.62),
            reason=(
                f"Phần cuối trang {page_curr.page_number} (dưới y={top_cand.position:.2f}) "
                f"chứa nội dung tràn từ đầu trang {page_next.page_number} "
                f"(width_consistency={width_cons:.2f}, complementary_score={comp_score:.2f})."
            ),
        )

    if next_upper_cands and (
        curr_has_bot_trunc
        or page_curr.content_fill_ratio <= 0.68
        or page_curr.bottom_empty_norm >= 0.18
        or next_upper_cands[0].suspicion_score >= 0.65
    ):
        top_cand = next_upper_cands[0]
        spilled_h_norm = max(0.05, top_cand.position - page_next.top_empty_norm)
        comp_sum = page_curr.content_fill_ratio + spilled_h_norm
        comp_score = (
            1.0
            if 0.45 <= comp_sum <= 1.15
            else max(0.4, 1.0 - abs(comp_sum - 0.80))
        )
        comb = round(
            min(
                1.0,
                0.50 * top_cand.suspicion_score
                + 0.25 * width_cons
                + 0.25 * comp_score
                + (0.10 if curr_has_bot_trunc else 0.0),
            ),
            3,
        )
        return CrossPageMatchResult(
            source_page=page_curr.page_number,
            target_page=page_next.page_number,
            relation_type="MIXED_SPILLOVER",
            width_consistency_score=width_cons,
            complementary_height_score=round(comp_score, 4),
            combined_score=comb,
            confirmed=bool(comb >= 0.62),
            reason=(
                f"Phần đầu trang {page_next.page_number} (trên y={top_cand.position:.2f}) "
                f"chứa phần nội dung bị cắt tràn từ cuối trang {page_curr.page_number} "
                f"(width_consistency={width_cons:.2f}, complementary_score={comp_score:.2f})."
            ),
        )

    # Kịch bản B: SPLIT_CONTINUATION (SPLIT_PAGE) — Một trang tài liệu vật lý bị chia thành 2 trang PDF liên tiếp
    # Điều kiện: Cả 2 trang đều chỉ chứa một phần trang (content_fill_ratio <= 0.64) VÀ có dấu hiệu bị cắt ở ranh giới
    # (hoặc trang N trống hẳn nửa dưới + trang N+1 trống hẳn nửa trên / sát mép trên mà không có chữ ký cuối trang)
    fill_curr = page_curr.content_fill_ratio
    fill_next = page_next.content_fill_ratio
    combined_fill = fill_curr + fill_next

    is_partial_pair = (
        0.18 <= fill_curr <= 0.62
        and 0.16 <= fill_next <= 0.62
        and 0.48 <= combined_fill <= 1.12
        and not page_curr.has_signature_bottom
    )
    has_split_boundary_cue = (
        curr_has_bot_trunc
        or next_has_top_trunc
        or (page_curr.bottom_empty_norm >= 0.36 and page_next.top_empty_norm >= 0.32)
        or (page_curr.bottom_empty_norm >= 0.38 and page_next.top_empty_norm <= 0.04)
        or (
            fill_curr <= 0.42
            and fill_next <= 0.42
            and combined_fill <= 0.84
            and page_curr.bottom_empty_norm >= 0.48
            and page_next.bottom_empty_norm >= 0.48
        )
    )

    if is_partial_pair and has_split_boundary_cue:
        comp_score = max(0.5, 1.0 - abs(combined_fill - 0.82))
        trunc_bonus = (0.12 if curr_has_bot_trunc else 0.0) + (
            0.12 if next_has_top_trunc else 0.0
        )
        comb = round(
            min(1.0, 0.42 * width_cons + 0.42 * comp_score + trunc_bonus), 3
        )
        if comb >= 0.60:
            return CrossPageMatchResult(
                source_page=page_curr.page_number,
                target_page=page_next.page_number,
                relation_type="SPLIT_CONTINUATION",
                width_consistency_score=width_cons,
                complementary_height_score=round(comp_score, 4),
                combined_score=comb,
                confirmed=bool(comb >= 0.64),
                reason=(
                    f"Trang {page_curr.page_number} (fill={fill_curr:.2f}) và trang {page_next.page_number} "
                    f"(fill={fill_next:.2f}) là hai phần bị tách ra từ cùng một trang tài liệu vật lý "
                    f"(tổng tỷ lệ lấp đầy={combined_fill:.2f}, đồng nhất lề={width_cons:.2f})."
                ),
            )

    return None


def run_cross_page_validation(
    layouts_by_page: Dict[int, PageLayoutAnalysisResult],
    overlap_threshold: float = BOUNDARY_OVERLAP_THRESHOLD,
) -> Dict[int, List[CrossPageMatchResult]]:
    """
    Thực thi kiểm tra quan hệ liên trang cho mọi cặp trang liền kề `(N, N+1)`
    và trả về ánh xạ `page_number -> List[CrossPageMatchResult]`.
    """
    matches_by_page: Dict[int, List[CrossPageMatchResult]] = {
        pno: [] for pno in layouts_by_page
    }
    sorted_pages = sorted(layouts_by_page.keys())

    for i in range(len(sorted_pages) - 1):
        p_curr = sorted_pages[i]
        p_next = sorted_pages[i + 1]
        if p_next != p_curr + 1:
            continue

        lay_curr = layouts_by_page[p_curr]
        lay_next = layouts_by_page[p_next]

        # 1. Kiểm tra CONTENT_OVERLAP giữa trang p_curr và p_next
        ov_match = detect_overlap_between_adjacent_pages(
            lay_curr, lay_next, overlap_threshold=overlap_threshold
        )
        if ov_match is not None:
            matches_by_page[p_curr].append(ov_match)
            # Ghi nhận đối xứng cho trang p_next liên kết ngược về p_curr
            rev_ov = CrossPageMatchResult(
                source_page=p_next,
                target_page=p_curr,
                relation_type="CONTENT_OVERLAP",
                ncc_score=ov_match.ncc_score,
                orb_inlier_ratio=ov_match.orb_inlier_ratio,
                profile_correlation=ov_match.profile_correlation,
                width_consistency_score=ov_match.width_consistency_score,
                complementary_height_score=ov_match.complementary_height_score,
                overlap_bbox_source_px=ov_match.overlap_bbox_target_px,
                overlap_bbox_target_px=ov_match.overlap_bbox_source_px,
                overlap_height_norm=ov_match.overlap_height_norm,
                combined_score=ov_match.combined_score,
                confirmed=ov_match.confirmed,
                reason=ov_match.reason,
            )
            matches_by_page[p_next].append(rev_ov)

        # 2. Kiểm tra SPLIT_CONTINUATION / MIXED_SPILLOVER giữa trang p_curr và p_next
        sp_match = detect_split_or_spillover_between_adjacent_pages(
            lay_curr, lay_next
        )
        if sp_match is not None:
            matches_by_page[p_curr].append(sp_match)
            rev_sp = CrossPageMatchResult(
                source_page=p_next,
                target_page=p_curr,
                relation_type=sp_match.relation_type,
                width_consistency_score=sp_match.width_consistency_score,
                complementary_height_score=sp_match.complementary_height_score,
                combined_score=sp_match.combined_score,
                confirmed=sp_match.confirmed,
                reason=sp_match.reason,
            )
            matches_by_page[p_next].append(rev_sp)

    return matches_by_page
