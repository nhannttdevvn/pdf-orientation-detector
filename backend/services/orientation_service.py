"""
Unified Quality Engine (`DecisionEngine`) và dịch vụ tổng hợp kết quả kiểm tra chất lượng trang PDF
(`src/pdf_inspector/services/orientation_service.py`).

Đảm nhiệm:
1. Tổng hợp kết quả từ `OrientationDetector` và bất kỳ `DefectResult` nào khác từ các Detector đang bật.
2. Phân loại trạng thái Orientation (`NORMAL`, `ROTATED`, `REVIEW`) giữ nguyên 100% quy tắc an toàn.
3. Tạo cấu trúc `Unified Quality Result` (`PageDecision` / `PageQualityResult`):
   - `inspection_status`: `COMPLETED`, `COMPLETED_WITH_REVIEW`, `PARTIAL_FAILURE`, `FAILED`.
   - `defects`: Danh sách các lỗi phát hiện hoặc nghi ngờ trên trang (Multi-label Defect Detection).
   - `detector_outcomes`: Phân biệt rõ `NO_DEFECT`, `CONFIRMED`, `SUSPECTED_LOW_CONFIDENCE`,
     `DETECTOR_FAILED` và `NOT_CHECKED`.
   - `requires_review`: Có cần người dùng kiểm tra hay không.
   - `enabled_detectors` & `unsupported_checks`.
"""

from typing import Dict, List, Optional

from ai_engine.core.config import (
    AUTO_FIX,
    BOUNDARY_STATUS_ERROR,
    BOUNDARY_STATUS_NORMAL,
    BOUNDARY_STATUS_SUSPECTED,
    DETECTION_UNKNOWN,
    ORIENTATION_CONF_THRESHOLD,
    STATUS_NORMAL,
    STATUS_REVIEW,
    STATUS_ROTATED,
    STRIPE_STATUS_CLEAN,
    STRIPE_STATUS_DETECTED,
    STRIPE_STATUS_ERROR,
    STRIPE_STATUS_REVIEW,
)
from ai_engine.core.logging_config import setup_logger
from backend.schemas.schemas import (
    DefectResult,
    DefectStatus,
    DefectType,
    InspectionStatus,
    OrientationDetectionResult,
    PageDecision,
)
from backend.repositories.file_utils import normalize_angle

logger = setup_logger(__name__)


class UnifiedQualityEngine:
    """
    Bộ tổng hợp chất lượng trang tài liệu đa lỗi (Unified Quality Engine / DecisionEngine).
    Kết hợp kết quả từ Orientation Detector và các Detector khác (nếu được kích hoạt)
    để sinh `PageDecision` (`PageQualityResult`).
    """

    def __init__(
        self,
        conf_threshold: float = ORIENTATION_CONF_THRESHOLD,
        auto_fix: bool = AUTO_FIX,
        enabled_detectors: Optional[List[str]] = None,
        unsupported_checks: Optional[List[str]] = None,
    ) -> None:
        self.conf_threshold: float = float(conf_threshold)
        self.auto_fix: bool = bool(auto_fix)
        self.enabled_detectors: List[str] = (
            list(enabled_detectors)
            if enabled_detectors is not None
            else ["orientation"]
        )
        self.unsupported_checks: List[str] = (
            list(unsupported_checks)
            if unsupported_checks is not None
            else [DefectType.STRIPE.value, DefectType.BOUNDARY.value]
        )

    def evaluate_page(
        self,
        detection: OrientationDetectionResult,
        manual_angle: Optional[int] = None,
        additional_defects: Optional[List[DefectResult]] = None,
    ) -> PageDecision:
        """
        Đánh giá kết quả phát hiện của một trang và trả về `PageDecision` (`PageQualityResult`).
        - Nếu `manual_angle` được người dùng chỉ định (0, 90, 180, 270), ưu tiên áp dụng góc thủ công.
        - Nếu không có `manual_angle`, áp dụng quy tắc phân loại `NORMAL` / `ROTATED` / `REVIEW`
          và nguyên tắc an toàn: KHÔNG tự động xoay trang `REVIEW`.
        - Ghi nhận đầy đủ `defects`, `detector_outcomes`, `inspection_status`, `requires_review`,
          `enabled_detectors` và `unsupported_checks`.
        """
        existing_rot = normalize_angle(detection.existing_pdf_rotation)

        is_unknown = (
            detection.status == DETECTION_UNKNOWN
            or detection.rotate is None
            or detection.orientation_conf is None
        )
        is_low_conf = (
            not is_unknown
            and detection.orientation_conf is not None
            and detection.orientation_conf < self.conf_threshold
        )
        has_conflict = bool(detection.has_conflict)

        if is_unknown or is_low_conf or has_conflict:
            status = STATUS_REVIEW
            reason = detection.error
            if not reason:
                if is_unknown:
                    reason = "OCR không nhận diện được hướng văn bản."
                elif has_conflict:
                    reason = "Kết quả giữa các lần thử OCR mâu thuẫn nhau."
                elif is_low_conf:
                    reason = (
                        f"Độ tin cậy OCR ({detection.orientation_conf:.2f}) "
                        f"thấp hơn ngưỡng ({self.conf_threshold:.2f})."
                    )
            rot_defect_status = (
                DefectStatus.DETECTOR_FAILED.value
                if is_unknown
                else DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
            )
        else:
            rot_angle = normalize_angle(detection.rotate)  # type: ignore[arg-type]
            if rot_angle == 0:
                status = STATUS_NORMAL
                reason = None
                rot_defect_status = DefectStatus.NO_DEFECT.value
            else:
                status = STATUS_ROTATED
                reason = None
                rot_defect_status = DefectStatus.CONFIRMED.value

        # Xây dựng danh sách defects trên trang (Multi-label Defect Detection)
        page_defects: List[DefectResult] = []
        if rot_defect_status != DefectStatus.NO_DEFECT.value:
            page_defects.append(
                DefectResult(
                    defect_type=DefectType.ROTATION.value,
                    detector_id="orientation",
                    status=rot_defect_status,
                    requires_review=(status == STATUS_REVIEW),
                    correction_angle=(
                        normalize_angle(detection.rotate)
                        if (status == STATUS_ROTATED and detection.rotate is not None)
                        else 0
                    ),
                    detected_rotation=detection.rotate,
                    detected_orientation=detection.orientation,
                    raw_confidence=detection.orientation_conf,
                    confidence_threshold=self.conf_threshold,
                    detection_method=detection.detection_method,
                    error=reason,
                )
            )

        stripe_defects: List[DefectResult] = []
        stripe_raw_results: List[DefectResult] = []
        boundary_defects: List[DefectResult] = []
        boundary_raw_results: List[DefectResult] = []
        boundary_candidates_list: List[ dict ] = []
        related_pages_set = set()
        reconstruction_proposal_dict = None

        for extra_def in additional_defects or []:
            if extra_def.detector_id == "stripe":
                stripe_raw_results.append(extra_def)
            elif extra_def.detector_id in ("boundary", "page_boundary"):
                boundary_raw_results.append(extra_def)
                for cand in extra_def.boundary_candidates:
                    boundary_candidates_list.append(cand)
                for rp in extra_def.related_pages:
                    related_pages_set.add(int(rp))
                if extra_def.details and "all_related_pages" in extra_def.details:
                    for rp in extra_def.details["all_related_pages"]:
                        related_pages_set.add(int(rp))
                if (
                    extra_def.reconstruction_proposal is not None
                    and reconstruction_proposal_dict is None
                ):
                    reconstruction_proposal_dict = dict(
                        extra_def.reconstruction_proposal
                    )

            if extra_def.status != DefectStatus.NO_DEFECT.value:
                page_defects.append(extra_def)
                if extra_def.detector_id == "stripe":
                    stripe_defects.append(extra_def)
                elif extra_def.detector_id in ("boundary", "page_boundary"):
                    boundary_defects.append(extra_def)

        # Ghi nhận rõ kết quả từng loại lỗi để phân biệt NO_DEFECT vs NOT_CHECKED vs DETECTOR_FAILED
        detector_outcomes: Dict[str, str] = {
            DefectType.ROTATION.value: rot_defect_status,
        }
        for extra_def in additional_defects or []:
            if extra_def.detector_id not in ("stripe", "boundary", "page_boundary"):
                detector_outcomes[extra_def.defect_type] = extra_def.status

        if stripe_raw_results:
            if any(
                d.status == DefectStatus.DETECTOR_FAILED.value
                for d in stripe_raw_results
            ):
                detector_outcomes[DefectType.STRIPE.value] = (
                    DefectStatus.DETECTOR_FAILED.value
                )
                stripe_status = STRIPE_STATUS_ERROR
            elif any(
                d.status == DefectStatus.CONFIRMED.value for d in stripe_raw_results
            ):
                detector_outcomes[DefectType.STRIPE.value] = (
                    DefectStatus.CONFIRMED.value
                )
                stripe_status = STRIPE_STATUS_DETECTED
            elif any(
                d.status == DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
                for d in stripe_raw_results
            ):
                detector_outcomes[DefectType.STRIPE.value] = (
                    DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
                )
                stripe_status = STRIPE_STATUS_REVIEW
            else:
                detector_outcomes[DefectType.STRIPE.value] = (
                    DefectStatus.NO_DEFECT.value
                )
                stripe_status = STRIPE_STATUS_CLEAN
        else:
            detector_outcomes[DefectType.STRIPE.value] = DefectStatus.NOT_CHECKED.value
            stripe_status = STRIPE_STATUS_CLEAN

        if boundary_raw_results:
            explicit_b_status = next(
                (
                    str(d.details["boundary_status"])
                    for d in boundary_raw_results
                    if d.details and "boundary_status" in d.details
                ),
                None,
            )
            if any(
                d.status == DefectStatus.DETECTOR_FAILED.value
                for d in boundary_raw_results
            ):
                detector_outcomes[DefectType.BOUNDARY.value] = (
                    DefectStatus.DETECTOR_FAILED.value
                )
                boundary_status = BOUNDARY_STATUS_ERROR
            elif any(
                d.status == DefectStatus.CONFIRMED.value for d in boundary_raw_results
            ):
                detector_outcomes[DefectType.BOUNDARY.value] = (
                    DefectStatus.CONFIRMED.value
                )
                boundary_status = explicit_b_status or "MIXED_PAGE_DETECTED"
            elif any(
                d.status == DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
                for d in boundary_raw_results
            ):
                detector_outcomes[DefectType.BOUNDARY.value] = (
                    DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
                )
                boundary_status = explicit_b_status or BOUNDARY_STATUS_SUSPECTED
            else:
                detector_outcomes[DefectType.BOUNDARY.value] = (
                    DefectStatus.NO_DEFECT.value
                )
                boundary_status = explicit_b_status or BOUNDARY_STATUS_NORMAL
        else:
            detector_outcomes[DefectType.BOUNDARY.value] = (
                DefectStatus.NOT_CHECKED.value
            )
            boundary_status = BOUNDARY_STATUS_NORMAL

        for unchecked in self.unsupported_checks:
            if unchecked not in detector_outcomes:
                detector_outcomes[unchecked] = DefectStatus.NOT_CHECKED.value

        requires_review = (status == STATUS_REVIEW) or any(
            d.requires_review for d in page_defects
        )
        has_detector_failure = (
            rot_defect_status == DefectStatus.DETECTOR_FAILED.value
            or any(d.status == DefectStatus.DETECTOR_FAILED.value for d in page_defects)
        )
        if has_detector_failure:
            inspection_status = InspectionStatus.PARTIAL_FAILURE.value
        elif requires_review:
            inspection_status = InspectionStatus.COMPLETED_WITH_REVIEW.value
        else:
            inspection_status = InspectionStatus.COMPLETED.value

        if manual_angle is not None:
            norm_manual = normalize_angle(manual_angle)
            correction_angle = norm_manual
            should_rotate = norm_manual != 0
            method = f"{detection.detection_method}+manual_override"
            return PageDecision(
                page_number=detection.page_number,
                status=status,
                detected_rotation=detection.rotate,
                orientation_conf=detection.orientation_conf,
                existing_pdf_rotation=existing_rot,
                correction_angle=correction_angle,
                corrected=False,
                detection_method=method,
                error=reason,
                detected_orientation=detection.orientation,
                script=detection.script,
                script_conf=detection.script_conf,
                has_conflict=has_conflict,
                manual_override=True,
                should_rotate=should_rotate,
                inspection_status=inspection_status,
                defects=page_defects,
                stripe_status=stripe_status,
                stripe_defects=stripe_defects,
                boundary_status=boundary_status,
                boundary_defects=boundary_defects,
                boundary_candidates=boundary_candidates_list,
                related_pages=sorted(related_pages_set),
                reconstruction_proposal=reconstruction_proposal_dict,
                requires_review=requires_review,
                enabled_detectors=list(self.enabled_detectors),
                unsupported_checks=list(self.unsupported_checks),
                detector_outcomes=detector_outcomes,
            )

        if status == STATUS_ROTATED and detection.rotate is not None:
            correction_angle = normalize_angle(detection.rotate)
            should_rotate = bool(self.auto_fix and correction_angle != 0)
        else:
            correction_angle = 0
            should_rotate = False

        return PageDecision(
            page_number=detection.page_number,
            status=status,
            detected_rotation=detection.rotate,
            orientation_conf=detection.orientation_conf,
            existing_pdf_rotation=existing_rot,
            correction_angle=correction_angle,
            corrected=False,
            detection_method=detection.detection_method,
            error=reason,
            detected_orientation=detection.orientation,
            script=detection.script,
            script_conf=detection.script_conf,
            has_conflict=has_conflict,
            manual_override=False,
            should_rotate=should_rotate,
            inspection_status=inspection_status,
            defects=page_defects,
            stripe_status=stripe_status,
            stripe_defects=stripe_defects,
            boundary_status=boundary_status,
            boundary_defects=boundary_defects,
            boundary_candidates=boundary_candidates_list,
            related_pages=sorted(related_pages_set),
            reconstruction_proposal=reconstruction_proposal_dict,
            requires_review=requires_review,
            enabled_detectors=list(self.enabled_detectors),
            unsupported_checks=list(self.unsupported_checks),
            detector_outcomes=detector_outcomes,
        )

    def evaluate_all(
        self,
        detections: List[OrientationDetectionResult],
        manual_overrides: Optional[Dict[int, int]] = None,
        additional_defects_by_page: Optional[Dict[int, List[DefectResult]]] = None,
    ) -> List[PageDecision]:
        """
        Đánh giá danh sách kết quả phát hiện cho toàn bộ tài liệu PDF.
        """
        overrides = manual_overrides or {}
        extra_map = additional_defects_by_page or {}
        decisions: List[PageDecision] = []
        for det in detections:
            manual_angle = overrides.get(det.page_number)
            extra_defects = extra_map.get(det.page_number)
            decisions.append(
                self.evaluate_page(
                    det,
                    manual_angle=manual_angle,
                    additional_defects=extra_defects,
                )
            )
        return decisions


# Bí danh tương thích ngược
DecisionEngine = UnifiedQualityEngine


def __getattr__(name: str):
    """Hỗ trợ import trễ `PDFOrientationPipeline` và `DocumentPipeline` tránh vòng lặp import."""
    if name in ("PDFOrientationPipeline", "DocumentPipeline"):
        from ai_engine.pipelines.document_pipeline import DocumentPipeline

        return DocumentPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
