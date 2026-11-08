"""
Phase 3.5, 3.6 & 3.7 — PageBoundaryDetector & Boundary Decision Engine
(`src/pdf_inspector/detectors/boundary/boundary_detector.py`).

Triển khai `PageBoundaryDetector(DocumentLevelDetector)`:
- `detector_id = "boundary"` (hỗ trợ bí danh `"page_boundary"`).
- `scope = "CROSS_PAGE"`.
- `supported_defect_types = ("BOUNDARY", "MIXED_PAGE", "SPLIT_PAGE", "CONTENT_OVERLAP", "CONTENT_TRUNCATED", "PAGE_BOUNDARY_SUSPECTED")`.
- Hỗ trợ nhiều lỗi đồng thời trên cùng một trang (Multi-label Defect Detection).
- Tích hợp chặt chẽ với `PageContext`, `CoordinateTransform` (chuyển đổi tọa độ giữa ảnh render, ảnh đã xoay và PDF gốc),
  `DetectorRegistry` và `DocumentPipeline`.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ai_engine.core.config import (
    BOUNDARY_CONFIRM_THRESHOLD,
    BOUNDARY_OVERLAP_THRESHOLD,
    BOUNDARY_REVIEW_THRESHOLD,
    BOUNDARY_STATUS_ERROR,
    BOUNDARY_STATUS_MISSING_CONTENT,
    BOUNDARY_STATUS_MIXED_DETECTED,
    BOUNDARY_STATUS_NORMAL,
    BOUNDARY_STATUS_RECONSTRUCTABLE,
    BOUNDARY_STATUS_REVIEW,
    BOUNDARY_STATUS_SUSPECTED,
    BOUNDARY_USE_OPTIONAL_LAYOUT_MODEL,
)
from ai_engine.detectors.base_detector import DocumentLevelDetector
from ai_engine.detectors.boundary.boundary_candidate_detector import (
    OptionalLayoutModelVerifier,
    detect_and_validate_boundary_candidates,
)
from ai_engine.detectors.boundary.cross_page_validator import run_cross_page_validation
from ai_engine.detectors.boundary.layout_analyzer import analyze_page_layout
from backend.schemas.boundary_schemas import (
    BoundaryCandidate,
    BoundaryDefectCode,
    BoundaryPageDecisionResult,
    CrossPageMatchResult,
    PageLayoutAnalysisResult,
    ReconstructionProposal,
)
from backend.schemas.context import CoordinateSpace, CoordinateTransform, PageContext
from backend.schemas.schemas import DefectResult, DefectStatus, DefectType
from ai_engine.reconstruction.page_reconstructor import render_boundary_overlay


class PageBoundaryDetector(DocumentLevelDetector):
    """
    Detector phát hiện lỗi ranh giới trang, lẫn trang, chia trang, chồng lấp nội dung
    và cắt mất mép nội dung (`detector_id = "boundary"`).
    Hoạt động theo kiến trúc 2 pha tiết kiệm RAM:
    - Pha 1 (`extract_page_layout`): Phân tích bố cục từng trang ngay khi `PageContext` đang mở
      và chỉ giữ lại đặc trưng hình học + dải biên thu gọn (`~150KB/trang`).
    - Pha 2 (`evaluate_document_boundaries`): Đối chiếu liên trang `(N-1, N, N+1)`,
      ra quyết định trạng thái (`BoundaryStatus`) và sinh đề xuất tái cấu trúc (`ReconstructionProposal`).
    """

    def __init__(
        self,
        confirm_threshold: float = BOUNDARY_CONFIRM_THRESHOLD,
        review_threshold: float = BOUNDARY_REVIEW_THRESHOLD,
        overlap_threshold: float = BOUNDARY_OVERLAP_THRESHOLD,
        use_optional_layout_model: bool = BOUNDARY_USE_OPTIONAL_LAYOUT_MODEL,
    ) -> None:
        self.confirm_threshold: float = float(confirm_threshold)
        self.review_threshold: float = float(review_threshold)
        self.overlap_threshold: float = float(overlap_threshold)
        self.use_optional_layout_model: bool = bool(use_optional_layout_model)
        self._optional_verifier = OptionalLayoutModelVerifier(
            enabled=self.use_optional_layout_model
        )

    @property
    def detector_id(self) -> str:
        return "boundary"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_defect_types(self) -> Tuple[str, ...]:
        return (DefectType.BOUNDARY.value,)

    def extract_page_layout(
        self,
        page_ctx: PageContext,
        recommended_rotation_cw: int = 0,
    ) -> PageLayoutAnalysisResult:
        """
        Phân tích bố cục và ứng viên ranh giới trên một `PageContext`.
        Nếu trang đã được OrientationDetector phát hiện bị xoay (`90°`, `180°`, `270°`),
        hàm tự động phân tích trên ảnh đã xoay về chiều đọc chuẩn (`page_ctx.get_rotated_image(recommended_rotation_cw)`)
        mà không sửa đổi `page_ctx.original_image`.
        """
        rot_cw = int(recommended_rotation_cw or 0) % 360
        analysis_img = (
            page_ctx.get_rotated_image(rot_cw)
            if rot_cw != 0
            else page_ctx.original_image
        )
        pdf_w, pdf_h = page_ctx.pdf_page_size
        if rot_cw in (90, 270):
            pdf_w, pdf_h = pdf_h, pdf_w

        layout = analyze_page_layout(
            gray_image=analysis_img,
            page_number=page_ctx.page_number,
            dpi=page_ctx.dpi,
            pdf_width_pt=pdf_w,
            pdf_height_pt=pdf_h,
            analysis_rotation_applied=rot_cw,
        )
        self._optional_verifier.enabled = self.use_optional_layout_model
        detect_and_validate_boundary_candidates(
            layout=layout,
            gray_image=analysis_img,
            confirm_threshold=self.confirm_threshold,
            review_threshold=self.review_threshold,
            optional_verifier=self._optional_verifier,
        )
        return layout

    def _map_rot_bbox_to_rendered_and_pdf(
        self,
        bbox_rot: Tuple[int, int, int, int],
        rot_cw: int,
        transform: Optional[CoordinateTransform],
        rot_w: int,
        rot_h: int,
    ) -> Tuple[Tuple[int, int, int, int], Optional[Tuple[float, float, float, float]]]:
        """
        Ánh xạ `bbox = (x0, y0, x1, y1)` từ ảnh phân tích (đã xoay `rot_cw`) về:
        1. Hệ tọa độ pixel của ảnh render gốc (`RENDERED_IMAGE_PX`).
        2. Hệ tọa độ point của trang PDF gốc (`PDF_POINT`).
        """
        x0, y0, x1, y1 = bbox_rot
        if transform is None:
            return (x0, y0, x1, y1), None

        if rot_cw % 360 == 0:
            ren_bbox = (int(x0), int(y0), int(x1), int(y1))
        else:
            p1 = transform.rotated_pixel_to_rendered_pixel(x0, y0, rot_cw)
            p2 = transform.rotated_pixel_to_rendered_pixel(x1, y1, rot_cw)
            rx0 = int(round(min(p1[0], p2[0])))
            ry0 = int(round(min(p1[1], p2[1])))
            rx1 = max(rx0 + 1, int(round(max(p1[0], p2[0]))))
            ry1 = max(ry0 + 1, int(round(max(p1[1], p2[1]))))
            ren_bbox = (rx0, ry0, rx1, ry1)

        pdf_rect = transform.bbox_px_to_pdf(
            (
                float(ren_bbox[0]),
                float(ren_bbox[1]),
                float(ren_bbox[2]),
                float(ren_bbox[3]),
            )
        )
        return ren_bbox, pdf_rect

    def evaluate_document_boundaries(
        self,
        layouts_by_page: Any,
        transforms_by_page: Optional[Dict[int, CoordinateTransform]] = None,
        page_errors: Optional[Dict[int, str]] = None,
    ) -> Tuple[Dict[int, BoundaryPageDecisionResult], Dict[int, List[DefectResult]]]:
        """
        Thực thi kiểm tra liên trang (`Cross-Page Validation`) và `Boundary Decision Engine` (Phase 3.7)
        cho toàn bộ các trang của tài liệu.
        Giải phóng các dải ảnh biên (`release_strips()`) ngay sau khi hoàn tất.
        """
        if not isinstance(layouts_by_page, dict):
            layouts_by_page = {
                int(lay.page_number): lay for lay in layouts_by_page
            }
        t_map = transforms_by_page or {}
        err_map = page_errors or {}

        cross_matches_by_page = run_cross_page_validation(
            layouts_by_page=layouts_by_page,
            overlap_threshold=self.overlap_threshold,
        )

        page_decisions: Dict[int, BoundaryPageDecisionResult] = {}
        defects_by_page: Dict[int, List[DefectResult]] = {}

        all_pages = sorted(set(layouts_by_page.keys()) | set(err_map.keys()))
        for pno in all_pages:
            if pno in err_map and pno not in layouts_by_page:
                err_msg = err_map[pno]
                fail_def = DefectResult(
                    defect_type=DefectType.BOUNDARY.value,
                    detector_id=self.detector_id,
                    status=DefectStatus.DETECTOR_FAILED.value,
                    requires_review=True,
                    error=err_msg,
                    reason=err_msg,
                )
                page_decisions[pno] = BoundaryPageDecisionResult(
                    page_number=pno,
                    boundary_status=BOUNDARY_STATUS_ERROR,
                    defects=[fail_def.to_dict()],
                    requires_review=True,
                    reason=err_msg,
                    error=err_msg,
                )
                defects_by_page[pno] = [fail_def]
                continue

            layout = layouts_by_page[pno]
            transform = t_map.get(pno)
            cross_matches = cross_matches_by_page.get(pno, [])

            page_defects: List[DefectResult] = []
            related_pages_set = set()
            for cm in cross_matches:
                related_pages_set.add(cm.target_page)

            spillover_matches = [
                cm for cm in cross_matches if cm.relation_type == "MIXED_SPILLOVER"
            ]
            split_matches = [
                cm for cm in cross_matches if cm.relation_type == "SPLIT_CONTINUATION"
            ]
            overlap_matches = [
                cm for cm in cross_matches if cm.relation_type == "CONTENT_OVERLAP"
            ]

            # 1. Đánh giá các ứng viên ranh giới nội trang (`MIXED_PAGE` hoặc `PAGE_BOUNDARY_SUSPECTED`)
            active_candidates: List[BoundaryCandidate] = []
            for cand in layout.validated_candidates:
                # Nếu có bằng chứng liên trang MIXED_SPILLOVER thì tăng điểm tin cậy cho ứng viên
                boosted_score = cand.suspicion_score
                if spillover_matches:
                    boosted_score = min(
                        1.0,
                        round(
                            max(
                                cand.suspicion_score + 0.14,
                                spillover_matches[0].combined_score,
                            ),
                            3,
                        ),
                    )
                    cand.suspicion_score = boosted_score
                    if boosted_score >= self.confirm_threshold:
                        cand.status = "CONFIRMED"

                active_candidates.append(cand)
                ren_bbox, pdf_bbox = self._map_rot_bbox_to_rendered_and_pdf(
                    cand.band_bbox_px,
                    layout.analysis_rotation_applied,
                    transform,
                    layout.image_width_px,
                    layout.image_height_px,
                )
                rel_pages = sorted(
                    {cm.target_page for cm in cross_matches}
                )

                if cand.status == "CONFIRMED":
                    d_type = BoundaryDefectCode.MIXED_PAGE.value
                    d_status = DefectStatus.CONFIRMED.value
                    sev = "HIGH"
                else:
                    d_type = BoundaryDefectCode.PAGE_BOUNDARY_SUSPECTED.value
                    d_status = DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
                    sev = "MEDIUM"

                page_defects.append(
                    DefectResult(
                        defect_type=d_type,
                        detector_id=self.detector_id,
                        status=d_status,
                        requires_review=True,
                        confidence=cand.suspicion_score,
                        suspicion_score=cand.suspicion_score,
                        confidence_threshold=self.confirm_threshold,
                        detection_method="+".join(cand.detected_by),
                        bbox=ren_bbox,
                        bbox_pdf_pt=pdf_bbox,
                        severity=sev,
                        reason=cand.reason,
                        coordinate_space=CoordinateSpace.RENDERED_IMAGE_PX.value,
                        dpi=layout.dpi,
                        boundary_candidates=[cand.to_dict()],
                        related_pages=rel_pages,
                        features=dict(cand.features),
                        details={
                            "analysis_rotation_applied": layout.analysis_rotation_applied,
                            "suspected_boundary_norm": cand.position,
                            "suspected_boundary_px": cand.position_px,
                            "safe_cut_line_px": cand.safe_cut_line_px,
                            "safe_cut_line_norm": cand.safe_cut_line_norm,
                            "crosses_text": cand.crosses_text,
                            "related_pages": rel_pages,
                        },
                    )
                )

            # 2. Đánh giá lỗi chia tách trang vật lý qua 2 trang PDF (`SPLIT_PAGE`)
            for sp_m in split_matches:
                d_status = (
                    DefectStatus.CONFIRMED.value
                    if sp_m.confirmed
                    else DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
                )
                sp_bnd_norm = round(float(layout.content_bbox_norm[3]), 4)
                sp_bnd_px = int(round(sp_bnd_norm * layout.image_height_px))
                page_defects.append(
                    DefectResult(
                        defect_type=BoundaryDefectCode.SPLIT_PAGE.value,
                        detector_id=self.detector_id,
                        status=d_status,
                        requires_review=True,
                        confidence=sp_m.combined_score,
                        suspicion_score=sp_m.combined_score,
                        confidence_threshold=self.confirm_threshold,
                        detection_method="cross_page_split_validator",
                        severity="HIGH" if sp_m.confirmed else "MEDIUM",
                        reason=sp_m.reason,
                        coordinate_space=CoordinateSpace.NORMALIZED.value,
                        dpi=layout.dpi,
                        related_pages=[sp_m.target_page],
                        features={
                            "width_consistency_score": sp_m.width_consistency_score,
                            "complementary_height_score": sp_m.complementary_height_score,
                            "content_fill_ratio": layout.content_fill_ratio,
                        },
                        details={
                            "related_pages": [sp_m.target_page],
                            "suspected_boundary_norm": sp_bnd_norm,
                            "suspected_boundary_px": sp_bnd_px,
                        },
                    )
                )

            # 3. Đánh giá lỗi chồng lấp / lặp lại nội dung (`CONTENT_OVERLAP`)
            for ov_m in overlap_matches:
                d_status = (
                    DefectStatus.CONFIRMED.value
                    if ov_m.confirmed
                    else DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
                )
                ov_bbox_ren: Optional[Tuple[int, int, int, int]] = None
                ov_bbox_pdf: Optional[Tuple[float, float, float, float]] = None
                if ov_m.overlap_bbox_source_px is not None:
                    ov_bbox_ren, ov_bbox_pdf = self._map_rot_bbox_to_rendered_and_pdf(
                        ov_m.overlap_bbox_source_px,
                        layout.analysis_rotation_applied,
                        transform,
                        layout.image_width_px,
                        layout.image_height_px,
                    )
                ov_bnd_norm = (
                    round(max(0.05, 1.0 - ov_m.overlap_height_norm), 4)
                    if ov_m.target_page > pno
                    else round(min(0.95, ov_m.overlap_height_norm), 4)
                )
                ov_bnd_px = int(round(ov_bnd_norm * layout.image_height_px))
                page_defects.append(
                    DefectResult(
                        defect_type=BoundaryDefectCode.CONTENT_OVERLAP.value,
                        detector_id=self.detector_id,
                        status=d_status,
                        requires_review=True,
                        confidence=ov_m.combined_score,
                        suspicion_score=ov_m.combined_score,
                        confidence_threshold=self.overlap_threshold,
                        detection_method="ncc_template+profile_corr+orb",
                        bbox=ov_bbox_ren,
                        bbox_pdf_pt=ov_bbox_pdf,
                        severity="HIGH" if ov_m.confirmed else "MEDIUM",
                        reason=ov_m.reason,
                        coordinate_space=CoordinateSpace.RENDERED_IMAGE_PX.value,
                        dpi=layout.dpi,
                        related_pages=[ov_m.target_page],
                        features={
                            "ncc_score": ov_m.ncc_score,
                            "profile_correlation": ov_m.profile_correlation,
                            "orb_inlier_ratio": ov_m.orb_inlier_ratio,
                            "overlap_height_norm": ov_m.overlap_height_norm,
                        },
                        details={
                            "related_pages": [ov_m.target_page],
                            "overlap_ratio": ov_m.overlap_height_norm,
                            "suspected_boundary_norm": ov_bnd_norm,
                            "suspected_boundary_px": ov_bnd_px,
                        },
                    )
                )

            # 4. Đánh giá lỗi nội dung sát mép bị cắt cụt (`CONTENT_TRUNCATED`)
            has_unrecoverable_truncation = False
            for trunc in layout.edge_truncations:
                ren_bbox, pdf_bbox = self._map_rot_bbox_to_rendered_and_pdf(
                    trunc.bbox,
                    layout.analysis_rotation_applied,
                    transform,
                    layout.image_width_px,
                    layout.image_height_px,
                )
                # Kiểm tra xem phần bị cắt ở mép TOP/BOTTOM có nằm ở trang liền kề (N-1 / N+1) hay không
                partner_page = (
                    (pno - 1) if trunc.edge_side == "TOP" else ((pno + 1) if trunc.edge_side == "BOTTOM" else None)
                )
                recoverable_from_neighbor = bool(
                    partner_page is not None
                    and any(
                        cm.target_page == partner_page
                        and cm.relation_type in ("SPLIT_CONTINUATION", "MIXED_SPILLOVER", "CONTENT_OVERLAP")
                        for cm in cross_matches
                    )
                )
                if not recoverable_from_neighbor and trunc.status == "CONFIRMED":
                    has_unrecoverable_truncation = True

                d_status = (
                    DefectStatus.CONFIRMED.value
                    if trunc.status == "CONFIRMED"
                    else DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
                )
                rel_p = [partner_page] if (recoverable_from_neighbor and partner_page) else []
                trunc_norm = (
                    0.015
                    if trunc.edge_side == "TOP"
                    else (0.985 if trunc.edge_side == "BOTTOM" else 0.50)
                )
                trunc_px = int(round(trunc_norm * layout.image_height_px))
                page_defects.append(
                    DefectResult(
                        defect_type=BoundaryDefectCode.CONTENT_TRUNCATED.value,
                        detector_id=self.detector_id,
                        status=d_status,
                        requires_review=True,
                        confidence=trunc.suspicion_score,
                        suspicion_score=trunc.suspicion_score,
                        confidence_threshold=self.confirm_threshold,
                        detection_method="edge_component_clipping_analysis",
                        bbox=ren_bbox,
                        bbox_pdf_pt=pdf_bbox,
                        severity="HIGH" if not recoverable_from_neighbor else "MEDIUM",
                        reason=trunc.reason,
                        coordinate_space=CoordinateSpace.RENDERED_IMAGE_PX.value,
                        dpi=layout.dpi,
                        related_pages=rel_p,
                        features=trunc.to_dict(),
                        details={
                            "related_pages": rel_p,
                            "edge_side": trunc.edge_side,
                            "suspected_boundary_norm": trunc_norm,
                            "suspected_boundary_px": trunc_px,
                        },
                    )
                )

            # 5. Xây dựng Đề xuất Tái cấu trúc (`ReconstructionProposal` — Phase 3.8) và Quyết định Trạng thái (`BoundaryStatus` — Phase 3.7)
            related_pages_list = sorted(related_pages_set)
            proposal = self._build_reconstruction_proposal(
                page_number=pno,
                layout=layout,
                active_candidates=active_candidates,
                split_matches=split_matches,
                spillover_matches=spillover_matches,
                overlap_matches=overlap_matches,
                has_unrecoverable_truncation=has_unrecoverable_truncation,
            )
            for pdef in page_defects:
                pdef.details["reconstruction_proposal"] = proposal.to_dict()

            has_confirmed_mixed = any(
                d.defect_type == BoundaryDefectCode.MIXED_PAGE.value
                and d.status == DefectStatus.CONFIRMED.value
                for d in page_defects
            )
            has_confirmed_split_or_overlap = any(
                d.defect_type
                in (
                    BoundaryDefectCode.SPLIT_PAGE.value,
                    BoundaryDefectCode.CONTENT_OVERLAP.value,
                )
                and d.status == DefectStatus.CONFIRMED.value
                for d in page_defects
            )
            has_suspected_boundary = any(
                d.defect_type
                in (
                    BoundaryDefectCode.PAGE_BOUNDARY_SUSPECTED.value,
                    BoundaryDefectCode.CONTENT_TRUNCATED.value,
                )
                for d in page_defects
            )

            if has_confirmed_mixed:
                boundary_status = BOUNDARY_STATUS_MIXED_DETECTED
            elif has_unrecoverable_truncation:
                boundary_status = BOUNDARY_STATUS_MISSING_CONTENT
            elif has_confirmed_split_or_overlap and proposal and proposal.reconstructable:
                boundary_status = BOUNDARY_STATUS_RECONSTRUCTABLE
            elif has_suspected_boundary:
                boundary_status = BOUNDARY_STATUS_SUSPECTED
            elif page_defects:
                boundary_status = BOUNDARY_STATUS_REVIEW
            else:
                boundary_status = BOUNDARY_STATUS_NORMAL

            if proposal is not None and page_defects:
                for d in page_defects:
                    d.reconstruction_proposal = proposal.to_dict()
                    d.details["boundary_status"] = boundary_status
                    d.details["all_related_pages"] = related_pages_list
            elif page_defects:
                for d in page_defects:
                    d.details["boundary_status"] = boundary_status
                    d.details["all_related_pages"] = related_pages_list

            max_score = (
                max((d.suspicion_score or 0.0) for d in page_defects)
                if page_defects
                else 0.0
            )
            summary_reason = (
                " | ".join(d.reason for d in page_defects if d.reason)
                if page_defects
                else "Bố cục và ranh giới trang bình thường (NORMAL)."
            )

            if not page_defects:
                # Ghi nhận rõ 1 kết quả NO_DEFECT để UnifiedQualityEngine biết BOUNDARY đã được kiểm tra
                no_def = DefectResult(
                    defect_type=DefectType.BOUNDARY.value,
                    detector_id=self.detector_id,
                    status=DefectStatus.NO_DEFECT.value,
                    requires_review=False,
                    confidence=1.0,
                    suspicion_score=0.0,
                    detection_method="layout+cross_page_validation",
                    coordinate_space=CoordinateSpace.NORMALIZED.value,
                    dpi=layout.dpi,
                    features=layout.to_summary_dict(),
                    details={"boundary_status": BOUNDARY_STATUS_NORMAL},
                )
                defects_by_page[pno] = [no_def]
            else:
                defects_by_page[pno] = page_defects

            page_decisions[pno] = BoundaryPageDecisionResult(
                page_number=pno,
                boundary_status=boundary_status,
                defects=[d.to_dict() for d in page_defects],
                boundary_candidates=active_candidates,
                edge_truncations=list(layout.edge_truncations),
                cross_page_matches=cross_matches,
                related_pages=related_pages_list,
                reconstruction_proposal=proposal,
                suspicion_score=round(max_score, 3),
                requires_review=bool(page_defects),
                reason=summary_reason,
                layout_summary=layout.to_summary_dict(),
            )

        # Giải phóng toàn bộ các dải ảnh biên khỏi RAM sau khi kiểm tra liên trang hoàn tất
        for lay in layouts_by_page.values():
            lay.release_strips()

        return page_decisions, defects_by_page

    def _build_reconstruction_proposal(
        self,
        page_number: int,
        layout: PageLayoutAnalysisResult,
        active_candidates: List[BoundaryCandidate],
        split_matches: List[CrossPageMatchResult],
        spillover_matches: List[CrossPageMatchResult],
        overlap_matches: List[CrossPageMatchResult],
        has_unrecoverable_truncation: bool,
    ) -> Optional[ReconstructionProposal]:
        """
        Tạo đề xuất phục hồi trang (`ReconstructionProposal`) an toàn theo đúng nguyên tắc Phase 3.8:
        - Nếu nội dung bị cắt mất ở mép mà không có ở trang kề (`has_unrecoverable_truncation`),
          đánh dấu `reconstructable=False`, `action_type="REQUIRES_RESCAN"`, tuyệt đối không bịa nội dung.
        - Nếu trang là `MIXED_PAGE` có đường cắt an toàn (`not crosses_text`), đề xuất tọa độ tách trang
          và ghép phần tràn sang trang liên quan (nếu có).
        - Nếu là `SPLIT_PAGE` hoặc `CONTENT_OVERLAP`, đề xuất ghép trang liên quan kèm khử vùng chồng lặp.
        """
        if has_unrecoverable_truncation and not active_candidates:
            return ReconstructionProposal(
                proposal_id=f"rec_p{page_number}_rescan",
                page_number=page_number,
                related_pages=[],
                action_type="REQUIRES_RESCAN",
                reconstructable=False,
                safe_to_cut=False,
                source_regions=[
                    {
                        "source_page": page_number,
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                        "note": "Truncated edge content missing from adjacent pages",
                    }
                ],
                explanation=(
                    "Không thể tự động khôi phục vì phần nội dung bị cắt mất ở mép trang không tồn tại "
                    "trên các trang liền kề. Vui lòng quét (scan) lại trang tài liệu gốc."
                ),
            )

        if active_candidates:
            top_cand = active_candidates[0]
            cut_px = top_cand.safe_cut_line_px
            cut_norm = top_cand.safe_cut_line_norm
            cut_pt = round(cut_norm * layout.pdf_height_pt, 2)
            safe = not top_cand.crosses_text
            rel_pages = sorted({cm.target_page for cm in spillover_matches})

            if rel_pages:
                partner = rel_pages[0]
                return ReconstructionProposal(
                    proposal_id=f"rec_p{page_number}_spillover_{partner}",
                    page_number=page_number,
                    related_pages=rel_pages,
                    action_type="SPLIT_AND_MERGE_SPILLOVER",
                    reconstructable=safe,
                    cut_orientation=top_cand.orientation,
                    cut_position_px=cut_px,
                    cut_position_norm=cut_norm,
                    cut_position_pdf_pt=cut_pt,
                    safe_to_cut=safe,
                    source_regions=[
                        {
                            "region_id": "part_A_keep",
                            "source_page": page_number,
                            "bbox_norm": [0.0, 0.0, 1.0, cut_norm],
                            "rotation_applied": layout.analysis_rotation_applied,
                        },
                        {
                            "region_id": "part_B_spillover",
                            "source_page": page_number,
                            "bbox_norm": [0.0, cut_norm, 1.0, 1.0],
                            "target_merge_page": partner,
                            "rotation_applied": layout.analysis_rotation_applied,
                        },
                    ],
                    explanation=(
                        f"Đề xuất tách trang {page_number} tại đường cắt an toàn y={cut_norm:.3f} ({cut_px}px) "
                        f"và ghép phần tràn với trang {partner}."
                    ),
                )

            return ReconstructionProposal(
                proposal_id=f"rec_p{page_number}_split",
                page_number=page_number,
                related_pages=[],
                action_type="SPLIT_SINGLE_PAGE",
                reconstructable=safe,
                cut_orientation=top_cand.orientation,
                cut_position_px=cut_px,
                cut_position_norm=cut_norm,
                cut_position_pdf_pt=cut_pt,
                safe_to_cut=safe,
                source_regions=[
                    {
                        "region_id": "upper_page",
                        "source_page": page_number,
                        "bbox_norm": [0.0, 0.0, 1.0, cut_norm],
                        "rotation_applied": layout.analysis_rotation_applied,
                    },
                    {
                        "region_id": "lower_page",
                        "source_page": page_number,
                        "bbox_norm": [0.0, cut_norm, 1.0, 1.0],
                        "rotation_applied": layout.analysis_rotation_applied,
                    },
                ],
                explanation=(
                    f"Đề xuất tách trang {page_number} thành 2 trang độc lập tại đường cắt an toàn "
                    f"y={cut_norm:.3f} ({cut_px}px, không cắt vào chữ)."
                ),
            )

        if overlap_matches:
            ov = overlap_matches[0]
            return ReconstructionProposal(
                proposal_id=f"rec_p{page_number}_overlap_{ov.target_page}",
                page_number=page_number,
                related_pages=[ov.target_page],
                action_type="REMOVE_OVERLAP_AND_MERGE",
                reconstructable=True,
                cut_orientation="HORIZONTAL",
                cut_position_norm=round(1.0 - ov.overlap_height_norm, 4),
                safe_to_cut=True,
                overlap_strip_norm=ov.overlap_height_norm,
                source_regions=[
                    {
                        "source_page": min(page_number, ov.target_page),
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                    },
                    {
                        "source_page": max(page_number, ov.target_page),
                        "bbox_norm": [0.0, ov.overlap_height_norm, 1.0, 1.0],
                        "deduplicated_overlap_norm": ov.overlap_height_norm,
                    },
                ],
                explanation=(
                    f"Đề xuất đối chiếu và loại bỏ dải nội dung chồng lặp ({ov.overlap_height_norm:.2f} chiều cao trang) "
                    f"giữa trang {page_number} và trang {ov.target_page}."
                ),
            )

        if split_matches:
            sp = split_matches[0]
            first_p = min(page_number, sp.target_page)
            second_p = max(page_number, sp.target_page)
            return ReconstructionProposal(
                proposal_id=f"rec_p{first_p}_merge_{second_p}",
                page_number=page_number,
                related_pages=[sp.target_page],
                action_type="MERGE_SPLIT_PAGES",
                reconstructable=True,
                cut_orientation="HORIZONTAL",
                cut_position_norm=round(layout.content_bbox_norm[3], 4),
                safe_to_cut=True,
                source_regions=[
                    {
                        "source_page": first_p,
                        "bbox_norm": list(layout.content_bbox_norm),
                    },
                    {
                        "source_page": second_p,
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                    },
                ],
                explanation=(
                    f"Đề xuất ghép phần nội dung của trang {first_p} và trang {second_p} "
                    f"để khôi phục trang tài liệu vật lý bị chia đôi."
                ),
            )

        return None

    def detect(self, context: Sequence[Any]) -> List[DefectResult]:
        """
        Thực thi giao thức chuẩn `DocumentLevelDetector.detect(context)`.
        Hỗ trợ cả danh sách `PageContext` hoặc danh sách `PageLayoutAnalysisResult`.
        """
        layouts_map: Dict[int, PageLayoutAnalysisResult] = {}
        transforms_map: Dict[int, CoordinateTransform] = {}
        for item in context:
            if isinstance(item, PageLayoutAnalysisResult):
                layouts_map[item.page_number] = item
            elif isinstance(item, PageContext):
                lay = self.extract_page_layout(item)
                layouts_map[item.page_number] = lay
                transforms_map[item.page_number] = item.coordinate_transform

        if not layouts_map:
            return []

        _, defects_by_page = self.evaluate_document_boundaries(
            layouts_by_page=layouts_map,
            transforms_by_page=transforms_map,
        )
        all_defects: List[DefectResult] = []
        for pno in sorted(defects_by_page.keys()):
            for d in defects_by_page[pno]:
                d.details["page_number"] = pno
                all_defects.append(d)
        return all_defects
