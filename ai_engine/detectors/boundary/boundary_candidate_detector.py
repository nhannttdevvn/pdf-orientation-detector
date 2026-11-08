"""
Phase 3.2 & 3.3 — Boundary Candidate Detection, Validation & Optional Layout Model
(`src/pdf_inspector/detectors/boundary/boundary_candidate_detector.py`).

Đảm nhiệm:
1. Sinh danh sách `BoundaryCandidate` từ kết quả phân tích bố cục (`PageLayoutAnalysisResult`).
2. Áp dụng bộ kiểm định đa đặc trưng để loại bỏ False Positive:
   - `paragraph_spacing`: Khoảng trắng bình thường giữa các đoạn văn.
   - `section_gap`: Khoảng trắng lớn giữa hai mục văn bản nhưng cùng nền giấy, cùng lề, không có vết ghép trang.
   - `table_grid_border`: Đường kẻ ngang thuộc bảng biểu.
   - `content_divider_rule`: Đường kẻ trang trí / phân cách nội dung nằm trong lề văn bản.
   - `text_underline`: Đường gạch chân văn bản.
   - `signature_block`: Vùng chữ ký / con dấu ở cuối trang.
   - `two_column_layout`: Rãnh dọc giữa hai cột văn bản.
   - `valid_book_spread`: Ảnh scan hai trang sách mở đối xứng theo thiết kế.
3. Cung cấp `OptionalLayoutModelVerifier` (Phase 3.3): chỉ chạy khi có ứng viên nghi vấn và được bật,
   hỗ trợ kiểm tra dấu hiệu header/footer/page number giữa trang mà không tạo dữ liệu giả khi tắt.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ai_engine.core.config import BOUNDARY_CONFIRM_THRESHOLD, BOUNDARY_REVIEW_THRESHOLD
from ai_engine.detectors.boundary.layout_analyzer import find_safe_cut_line_px
from backend.schemas.boundary_schemas import (
    BoundaryCandidate,
    PageLayoutAnalysisResult,
    TextRegionBlock,
)

_PAGE_NUM_PATTERN = re.compile(
    r"(?:\bpage\s*\d+\b|\btrang\s*\d+\b|^\s*-\s*\d+\s*-\s*$|^\s*\[\s*\d+\s*\]\s*$)",
    re.IGNORECASE,
)
_DOC_HEADER_PATTERN = re.compile(
    r"(?:\bchapter\s+\d+|\bphần\s+[ivx\d]+|\bcộng\s+hòa\s+xã\s+hội|\bofficial\s+report|\bdocument\s+id\b)",
    re.IGNORECASE,
)


class OptionalLayoutModelVerifier:
    """
    Phase 3.3 — Optional Layout / Semantic Verifier.
    Ưu tiên OpenCV screening trước; chỉ gọi phân tích OCR/Layout nhẹ trên vùng lân cận
    của `BoundaryCandidate` nghi vấn khi `enabled=True`.
    Nếu tắt hoặc môi trường không có engine tương ứng, trả về trạng thái trung thực (`DISABLED` / `UNAVAILABLE`).
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled: bool = bool(enabled)

    def verify_candidate_region(
        self,
        gray_image: np.ndarray,
        band_start_px: int,
        band_end_px: int,
    ) -> Dict[str, Any]:
        """
        Kiểm tra vùng ngay phía trên (`above_roi`) và ngay phía dưới (`below_roi`) ứng viên ranh giới
        để tìm tín hiệu số trang (`page_number_detected`) hoặc tiêu đề đầu trang mới (`header_restart_detected`).
        """
        if not self.enabled:
            return {
                "model_status": "DISABLED_BY_CONFIG",
                "page_number_above": False,
                "header_restart_below": False,
                "bonus_score": 0.0,
            }

        try:
            import pytesseract
        except ImportError:
            return {
                "model_status": "UNAVAILABLE",
                "page_number_above": False,
                "header_restart_below": False,
                "bonus_score": 0.0,
            }

        h, w = gray_image.shape[:2]
        roi_h = max(45, int(round(h * 0.08)))
        ay0 = max(0, band_start_px - roi_h)
        ay1 = max(ay0 + 10, band_start_px)
        by0 = min(h - 10, band_end_px)
        by1 = min(h, band_end_px + roi_h)

        above_crop = gray_image[ay0:ay1, :]
        below_crop = gray_image[by0:by1, :]

        try:
            txt_above = pytesseract.image_to_string(
                above_crop, config="--psm 6"
            ).strip()
            txt_below = pytesseract.image_to_string(
                below_crop, config="--psm 6"
            ).strip()
        except Exception as exc:
            return {
                "model_status": f"ERROR: {exc}",
                "page_number_above": False,
                "header_restart_below": False,
                "bonus_score": 0.0,
            }

        has_pnum = bool(_PAGE_NUM_PATTERN.search(txt_above))
        has_hdr = bool(_DOC_HEADER_PATTERN.search(txt_below))
        bonus = (0.12 if has_pnum else 0.0) + (0.10 if has_hdr else 0.0)
        return {
            "model_status": "ACTIVE_TESSERACT_LAYOUT",
            "page_number_above": has_pnum,
            "header_restart_below": has_hdr,
            "bonus_score": round(bonus, 3),
        }

    def verify_candidate(
        self,
        layout: PageLayoutAnalysisResult,
        candidate: BoundaryCandidate,
        gray_image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Kiểm định một `BoundaryCandidate` dựa trên đặc trưng hình học/ngữ nghĩa của `layout`
        (và ảnh xám nếu cung cấp). Khi `enabled=False`, tuyệt đối không tự bịa kết quả.
        """
        if not self.enabled:
            return {
                "enabled": False,
                "verified_by_model": False,
                "model_status": "DISABLED_BY_CONFIG",
                "adjusted_score": round(float(candidate.suspicion_score), 3),
            }

        bonus = 0.0
        if gray_image is not None:
            reg = self.verify_candidate_region(
                gray_image=gray_image,
                band_start_px=candidate.band_bbox_px[1],
                band_end_px=candidate.band_bbox_px[3],
            )
            bonus += float(reg.get("bonus_score", 0.0))
        else:
            if candidate.features.get("has_midpage_footer_above"):
                bonus += 0.06
            if candidate.features.get("has_midpage_header_below"):
                bonus += 0.06
            if candidate.features.get("has_illumination_step") or candidate.features.get("has_seam_line"):
                bonus += 0.04

        adj = round(min(1.0, float(candidate.suspicion_score) + bonus), 3)
        return {
            "enabled": True,
            "verified_by_model": True,
            "model_status": "ACTIVE_LAYOUT_VERIFIER",
            "adjusted_score": adj,
        }


def _compute_above_below_block_features(
    band_start_px: int,
    band_end_px: int,
    text_blocks: List[TextRegionBlock],
    image_height_px: int,
) -> Dict[str, Any]:
    """
    Tính toán quan hệ hình học giữa nhóm khối văn bản phía trên dải (`above_blocks`)
    và nhóm khối văn bản phía dưới dải (`below_blocks`).
    """
    above_blocks = [b for b in text_blocks if b.bbox[3] <= band_start_px + 8]
    below_blocks = [b for b in text_blocks if b.bbox[1] >= band_end_px - 8]

    if not above_blocks or not below_blocks:
        return {
            "has_both_sides": False,
            "above_count": len(above_blocks),
            "below_count": len(below_blocks),
            "margin_shift_norm": 0.0,
            "width_diff_norm": 0.0,
            "has_midpage_footer_above": False,
            "has_midpage_header_below": False,
            "below_ only_signature": False,
            "above_span_norm": 0.0,
            "below_span_norm": 0.0,
        }

    # Lề trái/phải chủ đạo của phần thân trên và phần thân dưới
    above_body = [b for b in above_blocks if b.width_norm >= 0.25] or above_blocks
    below_body = [b for b in below_blocks if b.width_norm >= 0.25] or below_blocks

    left_above = float(np.median([b.left_margin_norm for b in above_body]))
    left_below = float(np.median([b.left_margin_norm for b in below_body]))
    right_above = float(np.median([b.right_margin_norm for b in above_body]))
    right_below = float(np.median([b.right_margin_norm for b in below_body]))
    width_above = float(np.median([b.width_norm for b in above_body]))
    width_below = float(np.median([b.width_norm for b in below_body]))

    margin_shift = max(abs(left_above - left_below), abs(right_above - right_below))
    width_diff = abs(width_above - width_below)

    # Kiểm tra khối sát ngay phía trên dải (cách <= 6% chiều cao trang) có dáng dấp số trang / footer ngắn hay không
    last_above = max(above_blocks, key=lambda b: b.bbox[3])
    dist_above_norm = (band_start_px - last_above.bbox[3]) / float(max(1, image_height_px))
    has_midpage_footer = bool(
        dist_above_norm <= 0.065
        and last_above.line_count <= 2
        and last_above.width_norm <= 0.24
        and (last_above.left_margin_norm >= 0.32 or last_above.right_margin_norm <= 0.16)
    )

    # Kiểm tra khối đầu tiên ngay phía dưới dải có phải là tiêu đề trang mới / header căn giữa hay không
    first_below = min(below_blocks, key=lambda b: b.bbox[1])
    dist_below_norm = (first_below.bbox[1] - band_end_px) / float(max(1, image_height_px))
    is_centered_below = (
        abs(first_below.left_margin_norm - first_below.right_margin_norm) <= 0.08
        and first_below.left_margin_norm >= 0.12
        and first_below.width_norm <= 0.76
    )
    has_midpage_header = bool(
        dist_below_norm <= 0.08
        and first_below.line_count <= 3
        and (is_centered_below or first_below.region_role in ("header", "heading"))
    )

    below_only_signature = all(
        b.region_role == "signature" or b.width_norm <= 0.38 for b in below_blocks
    ) and len(below_blocks) <= 2

    above_y0 = min(b.bbox_norm[1] for b in above_blocks)
    above_y1 = max(b.bbox_norm[3] for b in above_blocks)
    below_y0 = min(b.bbox_norm[1] for b in below_blocks)
    below_y1 = max(b.bbox_norm[3] for b in below_blocks)

    return {
        "has_both_sides": True,
        "above_count": len(above_blocks),
        "below_count": len(below_blocks),
        "margin_shift_norm": round(margin_shift, 4),
        "width_diff_norm": round(width_diff, 4),
        "has_midpage_footer_above": has_midpage_footer,
        "has_midpage_header_below": has_midpage_header,
        "below_only_signature": below_only_signature,
        "above_span_norm": round(max(0.0, above_y1 - above_y0), 4),
        "below_span_norm": round(max(0.0, below_y1 - below_y0), 4),
    }


def detect_and_validate_boundary_candidates(
    layout: PageLayoutAnalysisResult,
    gray_image: np.ndarray,
    confirm_threshold: float = BOUNDARY_CONFIRM_THRESHOLD,
    review_threshold: float = BOUNDARY_REVIEW_THRESHOLD,
    optional_verifier: Optional[OptionalLayoutModelVerifier] = None,
) -> Tuple[List[BoundaryCandidate], List[BoundaryCandidate]]:
    """
    Phát hiện và kiểm định toàn bộ các ứng viên ranh giới trang (`BoundaryCandidate`) trên một trang:
    Trả về `(raw_candidates, validated_non_rejected_candidates)`.
    Đảm bảo nguyên tắc Phase 3.2:
    - KHÔNG kết luận lỗi chỉ vì có khoảng trắng lớn hoặc có một đường kẻ ngang đơn lẻ.
    - Ghi nhận rõ `rejection_category` và `reason` cho mọi ứng viên bị loại bỏ.
    """
    h, w = gray_image.shape[:2]
    _, otsu_bin = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    raw_candidates: List[BoundaryCandidate] = []
    validated_candidates: List[BoundaryCandidate] = []

    # 1. Nếu phát hiện rãnh dọc ở giữa trang (2 cột hoặc 2 trang sách), ghi nhận một ứng viên VERTICAL
    #    và kiểm định theo quy tắc two_column_layout / valid_book_spread
    if layout.is_two_column or layout.is_book_spread:
        vx_px = w // 2
        vx_norm = 0.5
        vx_pt = round(vx_norm * layout.pdf_width_pt, 2)
        rej_cat = "valid_book_spread" if layout.is_book_spread else "two_column_layout"
        reason_txt = (
            "Loại bỏ: Ảnh scan hai trang sách mở đối xứng theo thiết kế (valid_book_spread)."
            if layout.is_book_spread
            else "Loại bỏ: Bố cục văn bản hai cột hợp lệ trên cùng một trang (two_column_layout)."
        )
        v_cand = BoundaryCandidate(
            candidate_id=f"p{layout.page_number}_v_{vx_px}",
            orientation="VERTICAL",
            position=vx_norm,
            position_px=vx_px,
            position_pdf_pt=vx_pt,
            band_bbox_px=(int(w * 0.46), 0, int(w * 0.54), h),
            band_bbox_norm=(0.46, 0.0, 0.54, 1.0),
            coordinate_system="normalized_page",
            detected_by=["vertical_projection"],
            suspicion_score=0.0,
            status="REJECTED",
            rejection_category=rej_cat,
            safe_cut_line_px=vx_px,
            safe_cut_line_norm=vx_norm,
            crosses_text=False,
            reason=reason_txt,
            features={
                "is_two_column": layout.is_two_column,
                "is_book_spread": layout.is_book_spread,
                "aspect_ratio": round(float(w) / float(max(1, h)), 3),
            },
        )
        raw_candidates.append(v_cand)

    # 2. Đánh giá từng dải khoảng trắng / đường seam ngang (`WhitespaceBand`)
    for idx, band in enumerate(layout.whitespace_bands):
        pos_px = band.center_px
        pos_norm = band.center_norm
        pos_pt = round(pos_norm * layout.pdf_height_pt, 2)

        sources: List[str] = ["horizontal_projection"]
        if band.has_seam_line:
            sources.append("seam_edge_profile")
        if band.has_illumination_step:
            sources.append("illumination_step")

        blk_feat = _compute_above_below_block_features(
            band_start_px=band.start_px,
            band_end_px=band.end_px,
            text_blocks=layout.text_blocks,
            image_height_px=h,
        )
        if (
            blk_feat["margin_shift_norm"] >= 0.04
            or blk_feat["has_midpage_footer_above"]
            or blk_feat["has_midpage_header_below"]
        ):
            sources.append("layout_transition")

        safe_y, crosses_txt = find_safe_cut_line_px(
            text_only_mask=otsu_bin,
            target_pos_px=pos_px,
            orientation="HORIZONTAL",
            search_radius_px=max(16, band.thickness_px // 2 + 8),
        )
        safe_norm = round(safe_y / float(h), 4)

        cand_features: Dict[str, Any] = {
            "thickness_px": band.thickness_px,
            "thickness_norm": band.thickness_norm,
            "has_seam_line": band.has_seam_line,
            "seam_strength": band.seam_strength,
            "has_illumination_step": band.has_illumination_step,
            "illumination_delta": band.illumination_delta,
            "has_table_structure": layout.has_table_structure,
            **blk_feat,
        }

        # --- KIỂM ĐỊNH FALSE POSITIVE (Phase 3.2) ---
        # Rule A: Không có nội dung ở cả hai phía trên và dưới -> chỉ là lề trên/dưới rộng (`large_margin`)
        if not blk_feat["has_both_sides"]:
            cand = BoundaryCandidate(
                candidate_id=f"p{layout.page_number}_h_{idx + 1}",
                orientation="HORIZONTAL",
                position=pos_norm,
                position_px=pos_px,
                position_pdf_pt=pos_pt,
                band_bbox_px=(0, band.start_px, w, band.end_px),
                band_bbox_norm=(0.0, round(band.start_px / float(h), 4), 1.0, round(band.end_px / float(h), 4)),
                detected_by=sources,
                suspicion_score=0.0,
                status="REJECTED",
                rejection_category="large_margin",
                safe_cut_line_px=safe_y,
                safe_cut_line_norm=safe_norm,
                crosses_text=crosses_txt,
                reason="Loại bỏ: Khoảng trắng nằm ở vùng lề trang, không chia cắt hai khối nội dung.",
                features=cand_features,
            )
            raw_candidates.append(cand)
            continue

        # Rule B: Phần bên dưới chỉ là khối chữ ký / con dấu ngắn ở cuối trang (`signature_block`)
        if (
            blk_feat["below_only_signature"]
            and not band.has_illumination_step
            and (pos_norm >= 0.60 or blk_feat["below_span_norm"] <= 0.08)
        ):
            cand = BoundaryCandidate(
                candidate_id=f"p{layout.page_number}_h_{idx + 1}",
                orientation="HORIZONTAL",
                position=pos_norm,
                position_px=pos_px,
                position_pdf_pt=pos_pt,
                band_bbox_px=(0, band.start_px, w, band.end_px),
                band_bbox_norm=(0.0, round(band.start_px / float(h), 4), 1.0, round(band.end_px / float(h), 4)),
                detected_by=sources,
                suspicion_score=0.0,
                status="REJECTED",
                rejection_category="signature_block",
                safe_cut_line_px=safe_y,
                safe_cut_line_norm=safe_norm,
                crosses_text=crosses_txt,
                reason="Loại bỏ: Khoảng trắng trước vùng chữ ký / con dấu cuối trang hợp lệ (signature_block).",
                features=cand_features,
            )
            raw_candidates.append(cand)
            continue

        # Rule B2: Trang sách mở đôi hợp lệ (`valid_book_spread`) có nền giấy đồng nhất
        if layout.is_book_spread and not band.has_illumination_step:
            cand = BoundaryCandidate(
                candidate_id=f"p{layout.page_number}_h_{idx + 1}",
                orientation="HORIZONTAL",
                position=pos_norm,
                position_px=pos_px,
                position_pdf_pt=pos_pt,
                band_bbox_px=(0, band.start_px, w, band.end_px),
                band_bbox_norm=(0.0, round(band.start_px / float(h), 4), 1.0, round(band.end_px / float(h), 4)),
                detected_by=sources,
                suspicion_score=0.0,
                status="REJECTED",
                rejection_category="valid_book_spread",
                safe_cut_line_px=safe_y,
                safe_cut_line_norm=safe_norm,
                crosses_text=crosses_txt,
                reason="Loại bỏ: Khoảng cách dòng trên trang sách mở đôi hợp lệ (valid_book_spread).",
                features=cand_features,
            )
            raw_candidates.append(cand)
            continue

        # Rule C: Trang có cấu trúc bảng biểu và không có vết ghép giấy xuyên lề hay chênh lệch nền
        if (
            layout.has_table_structure
            and not band.has_seam_line
            and not band.has_illumination_step
        ):
            cand = BoundaryCandidate(
                candidate_id=f"p{layout.page_number}_h_{idx + 1}",
                orientation="HORIZONTAL",
                position=pos_norm,
                position_px=pos_px,
                position_pdf_pt=pos_pt,
                band_bbox_px=(0, band.start_px, w, band.end_px),
                band_bbox_norm=(0.0, round(band.start_px / float(h), 4), 1.0, round(band.end_px / float(h), 4)),
                detected_by=sources,
                suspicion_score=0.0,
                status="REJECTED",
                rejection_category="table_grid_border",
                safe_cut_line_px=safe_y,
                safe_cut_line_norm=safe_norm,
                crosses_text=crosses_txt,
                reason="Loại bỏ: Khoảng cách/đường kẻ thuộc cấu trúc bảng biểu hợp lệ (table_grid_border).",
                features=cand_features,
            )
            raw_candidates.append(cand)
            continue

        # Rule D: Chỉ là khoảng cách đoạn văn (`paragraph_spacing`) hoặc khoảng trắng lớn giữa 2 mục (`section_gap`)
        # mà KHÔNG có bằng chứng vật lý/bố cục của ranh giới trang (không seam xuyên lề, không lệch nền, không nhảy lề/header)
        has_physical_evidence = band.has_seam_line or band.has_illumination_step
        has_strong_layout_reset = (
            blk_feat["has_midpage_footer_above"] and blk_feat["has_midpage_header_below"]
        ) or (
            blk_feat["margin_shift_norm"] >= 0.055
            and (blk_feat["has_midpage_footer_above"] or blk_feat["has_midpage_header_below"])
        )

        if not has_physical_evidence and not has_strong_layout_reset:
            rej_cat = (
                "paragraph_spacing" if band.thickness_norm < 0.055 else "section_gap"
            )
            reason_msg = (
                "Loại bỏ: Khoảng cách bình thường giữa các đoạn văn (paragraph_spacing)."
                if rej_cat == "paragraph_spacing"
                else (
                    f"Loại bỏ: Khoảng trắng lớn giữa hai mục văn bản (thickness={band.thickness_norm:.2f}) "
                    "nhưng nền giấy đồng nhất, lề thẳng hàng và không có vết ghép trang (section_gap)."
                )
            )
            cand = BoundaryCandidate(
                candidate_id=f"p{layout.page_number}_h_{idx + 1}",
                orientation="HORIZONTAL",
                position=pos_norm,
                position_px=pos_px,
                position_pdf_pt=pos_pt,
                band_bbox_px=(0, band.start_px, w, band.end_px),
                band_bbox_norm=(0.0, round(band.start_px / float(h), 4), 1.0, round(band.end_px / float(h), 4)),
                detected_by=sources,
                suspicion_score=0.12,
                status="REJECTED",
                rejection_category=rej_cat,
                safe_cut_line_px=safe_y,
                safe_cut_line_norm=safe_norm,
                crosses_text=crosses_txt,
                reason=reason_msg,
                features=cand_features,
            )
            raw_candidates.append(cand)
            continue

        # --- TÍNH ĐIỂM NGHI VẤN ĐA ĐẶC TRƯNG (Multi-feature Suspicion Score) ---
        seam_score = (
            min(1.0, band.seam_strength / 0.72) if band.has_seam_line else 0.0
        )
        illum_score = (
            min(1.0, band.illumination_delta / 14.0)
            if band.has_illumination_step
            else 0.0
        )
        physical_score = max(seam_score, illum_score)
        secondary_physical = min(seam_score, illum_score)
        gap_score = min(1.0, max(0.15, band.thickness_norm / 0.07))
        layout_reset_score = (
            (0.45 if blk_feat["has_midpage_footer_above"] else 0.0)
            + (0.45 if blk_feat["has_midpage_header_below"] else 0.0)
            + min(0.45, blk_feat["margin_shift_norm"] * 5.0)
        )
        layout_reset_score = min(1.0, layout_reset_score)

        # Kết hợp đa tín hiệu: không cho phép chỉ 1 tín hiệu yếu vượt ngưỡng CONFIRMED
        raw_score = (
            0.42 * physical_score
            + 0.16 * secondary_physical
            + 0.24 * layout_reset_score
            + 0.18 * gap_score
        )
        if physical_score >= 0.55 and (
            layout_reset_score >= 0.30 or band.thickness_norm >= 0.032
        ):
            raw_score += 0.18

        # Nếu bật Optional Layout Model (Phase 3.3), xác minh thêm tín hiệu số trang / tiêu đề
        if optional_verifier is not None and optional_verifier.enabled and raw_score >= 0.32:
            opt_res = optional_verifier.verify_candidate_region(
                gray_image=gray_image,
                band_start_px=band.start_px,
                band_end_px=band.end_px,
            )
            cand_features["optional_layout_model"] = opt_res
            raw_score += float(opt_res.get("bonus_score", 0.0))

        suspicion_score = round(min(1.0, max(0.0, raw_score)), 3)
        if suspicion_score >= confirm_threshold:
            c_status = "CONFIRMED"
        elif suspicion_score >= review_threshold:
            c_status = "SUSPECTED"
        else:
            c_status = "REJECTED"

        reasons_parts: List[str] = []
        if band.has_seam_line:
            reasons_parts.append(f"vết ghép mép giấy xuyên lề (seam={band.seam_strength:.2f})")
        if band.has_illumination_step:
            reasons_parts.append(f"bước nhảy độ sáng nền (delta={band.illumination_delta:.1f})")
        if blk_feat["has_midpage_footer_above"]:
            reasons_parts.append("dấu hiệu số trang/footer giữa trang")
        if blk_feat["has_midpage_header_below"]:
            reasons_parts.append("dấu hiệu tiêu đề đầu trang mới bên dưới")
        if blk_feat["margin_shift_norm"] >= 0.04:
            reasons_parts.append(f"lệch lề văn bản ({blk_feat['margin_shift_norm']:.2f})")

        reason_str = (
            f"Phát hiện ranh giới chuyển trang ngang tại y={pos_norm:.2f} "
            f"({', '.join(reasons_parts) or 'khoảng cách bất thường'})."
        )

        cand = BoundaryCandidate(
            candidate_id=f"p{layout.page_number}_h_{idx + 1}",
            orientation="HORIZONTAL",
            position=pos_norm,
            position_px=pos_px,
            position_pdf_pt=pos_pt,
            band_bbox_px=(0, band.start_px, w, band.end_px),
            band_bbox_norm=(
                0.0,
                round(band.start_px / float(h), 4),
                1.0,
                round(band.end_px / float(h), 4),
            ),
            coordinate_system="normalized_page",
            detected_by=sources,
            suspicion_score=suspicion_score,
            status=c_status,
            rejection_category="below_threshold" if c_status == "REJECTED" else None,
            safe_cut_line_px=safe_y,
            safe_cut_line_norm=safe_norm,
            crosses_text=crosses_txt,
            reason=reason_str,
            features=cand_features,
        )
        raw_candidates.append(cand)
        if c_status in ("CONFIRMED", "SUSPECTED"):
            validated_candidates.append(cand)

    validated_candidates.sort(key=lambda c: c.suspicion_score, reverse=True)
    # Nếu đã có ứng viên CONFIRMED chính trên trang, loại bỏ các ứng viên SUSPECTED yếu hơn
    if any(c.status == "CONFIRMED" for c in validated_candidates):
        validated_candidates = [
            c for c in validated_candidates if c.status == "CONFIRMED"
        ]

    layout.raw_candidates = raw_candidates
    layout.validated_candidates = validated_candidates
    return raw_candidates, validated_candidates


def detect_boundary_candidates(
    layout: PageLayoutAnalysisResult,
    gray_image: Optional[np.ndarray] = None,
    confirm_threshold: float = BOUNDARY_CONFIRM_THRESHOLD,
    review_threshold: float = BOUNDARY_REVIEW_THRESHOLD,
    optional_verifier: Optional[OptionalLayoutModelVerifier] = None,
) -> List[BoundaryCandidate]:
    """
    Hàm tiện ích phát hiện và trả về danh sách toàn bộ ứng viên ranh giới (`raw_candidates`,
    trong đó các ứng viên hợp lệ có `is_false_positive == False`).
    """
    if gray_image is None:
        gray_image = np.full(
            (max(1, layout.image_height_px), max(1, layout.image_width_px)),
            245,
            dtype=np.uint8,
        )
    raw_cands, _ = detect_and_validate_boundary_candidates(
        layout=layout,
        gray_image=gray_image,
        confirm_threshold=confirm_threshold,
        review_threshold=review_threshold,
        optional_verifier=optional_verifier,
    )
    return raw_cands
