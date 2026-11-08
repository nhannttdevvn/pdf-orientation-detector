"""
StripeDetector — Bộ phát hiện lỗi vệt sọc máy scan (`src/pdf_inspector/detectors/defects/stripe_detector.py`).
Kế thừa `PageLevelDetector` (`BaseDetector`) trong kiến trúc Multi-Defect PDF Quality Inspection.

Trách nhiệm:
1. Tái sử dụng ảnh trang từ `PageContext` (`original_image`, `clahe_image`, `dpi`, `coordinate_transform`)
   mà KHÔNG render lại PDF và KHÔNG làm thay đổi `original_image`.
2. Gọi `line_candidate_detector` (kết hợp Morphological, HoughLinesP, Projection Profile) để tìm ứng viên.
3. Gọi `stripe_validator` để trích xuất đặc trưng, lọc False Positive (bảng biểu, gạch chân, chữ ký, viền giấy)
   và chấm điểm `suspicion_score` / `confidence`.
4. Trả về trạng thái trang (`CLEAN`, `STRIPE_DETECTED`, `REVIEW`, `PROCESSING_ERROR`) cùng danh sách `DefectResult`
   chuẩn hóa (gồm cả tọa độ pixel `bbox` và tọa độ PDF point `bbox_pdf_pt`).
5. Cung cấp hàm `render_stripe_overlay` để vẽ khung đánh dấu sọc phục vụ Preview trên Streamlit UI.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from pdf_inspector.core.config import (
    HORIZONTAL_CLASS_SCAN_STRIPE,
    HORIZONTAL_STRIPE_MARGIN_EXTENSION,
    HORIZONTAL_STRIPE_MIN_PAGE_RATIO,
    STRIPE_CONFIRM_THRESHOLD,
    STRIPE_MAX_THICKNESS_PX,
    STRIPE_MIN_CONTRAST,
    STRIPE_MIN_RELATIVE_LENGTH,
    STRIPE_REVIEW_THRESHOLD,
    STRIPE_STATUS_CLEAN,
    STRIPE_STATUS_DETECTED,
    STRIPE_STATUS_ERROR,
    STRIPE_STATUS_REVIEW,
)
from pdf_inspector.core.logging_config import setup_logger
from pdf_inspector.detectors.base_detector import PageLevelDetector
from pdf_inspector.detectors.defects.line_candidate_detector import (
    LineCandidate,
    detect_candidates_hough,
    detect_candidates_morphological,
    detect_candidates_projection,
    merge_and_deduplicate_candidates,
    prepare_stripe_maps,
)
from pdf_inspector.detectors.defects.stripe_validator import (
    ContentMargins,
    ValidatedStripe,
    estimate_content_margins,
    validate_stripe_candidates,
)
from pdf_inspector.models.context import CoordinateSpace, CoordinateTransform, PageContext
from pdf_inspector.models.schemas import DefectResult, DefectStatus, DefectType

logger = setup_logger(__name__)


@dataclass
class StripePageResult:
    """
    Kết quả phân tích vệt sọc tổng hợp trên một trang PDF:
    - `status`: `CLEAN`, `STRIPE_DETECTED`, `REVIEW`, hoặc `PROCESSING_ERROR`.
    - `has_stripe`: `True` nếu có ít nhất một vệt sọc `CONFIRMED` hoặc `SUSPECTED`.
    - `stripes`: Danh sách các vệt sọc đáng chú ý (`CONFIRMED` hoặc `SUSPECTED`).
    - `rejected_candidates`: Danh sách các ứng viên đã bị bộ lọc False Positive loại bỏ.
    """

    page_number: int
    status: str  # CLEAN, STRIPE_DETECTED, REVIEW, PROCESSING_ERROR
    has_stripe: bool = False
    candidates_count: int = 0
    confirmed_count: int = 0
    suspected_count: int = 0
    stripes: List[ValidatedStripe] = field(default_factory=list)
    rejected_candidates: List[ValidatedStripe] = field(default_factory=list)
    raw_candidates: List[LineCandidate] = field(default_factory=list)
    content_margins: Optional[ContentMargins] = None
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class StripeDetector(PageLevelDetector):
    """
    Detector phát hiện lỗi vệt sọc đen/xám/dải sọc (Vertical, Horizontal, Diagonal, Band Stripes)
    trong tài liệu PDF scan (`detector_id = "stripe"`).
    """

    def __init__(
        self,
        confirm_threshold: float = STRIPE_CONFIRM_THRESHOLD,
        review_threshold: float = STRIPE_REVIEW_THRESHOLD,
        min_relative_length: float = STRIPE_MIN_RELATIVE_LENGTH,
        max_thickness_px: int = STRIPE_MAX_THICKNESS_PX,
        min_contrast: float = STRIPE_MIN_CONTRAST,
        horizontal_min_page_ratio: float = HORIZONTAL_STRIPE_MIN_PAGE_RATIO,
        horizontal_margin_extension: float = HORIZONTAL_STRIPE_MARGIN_EXTENSION,
    ) -> None:
        self.confirm_threshold: float = float(confirm_threshold)
        self.review_threshold: float = float(review_threshold)
        self.min_relative_length: float = float(min_relative_length)
        self.max_thickness_px: int = int(max_thickness_px)
        self.min_contrast: float = float(min_contrast)
        self.horizontal_min_page_ratio: float = float(horizontal_min_page_ratio)
        self.horizontal_margin_extension: float = float(horizontal_margin_extension)

    @property
    def detector_id(self) -> str:
        return "stripe"

    @property
    def version(self) -> str:
        return "1.1.0"

    @property
    def supported_defect_types(self) -> Tuple[str, ...]:
        return (DefectType.STRIPE.value,)

    def extract_candidates(
        self,
        image: np.ndarray,
        clahe_image: Optional[np.ndarray] = None,
        dpi: int = 160,
    ) -> Tuple[Any, List[LineCandidate]]:
        """
        Chạy cả 3 kỹ thuật phát hiện ứng viên (Morphological, HoughLinesP, Projection Profile)
        và hợp nhất danh sách `LineCandidate`.
        """
        maps = prepare_stripe_maps(
            gray_image=image,
            clahe_image=clahe_image,
            dpi=dpi,
        )
        cand_morph = detect_candidates_morphological(
            maps=maps,
            min_relative_length=max(0.28, self.min_relative_length - 0.05),
            max_thickness_px=self.max_thickness_px,
            min_contrast=max(6.0, self.min_contrast - 1.0),
        )
        cand_hough = detect_candidates_hough(
            maps=maps,
            min_relative_length=max(0.28, self.min_relative_length - 0.05),
            min_contrast=self.min_contrast,
        )
        cand_proj = detect_candidates_projection(
            maps=maps,
            min_relative_length=self.min_relative_length,
            max_thickness_px=self.max_thickness_px,
            min_contrast=max(6.0, self.min_contrast - 1.0),
        )
        merged = merge_and_deduplicate_candidates(
            cand_morph + cand_hough + cand_proj,
            image_shape=maps.gray.shape[:2],
        )
        return maps, merged

    def detect_page(
        self,
        image: np.ndarray,
        page_number: int = 1,
        dpi: int = 160,
        clahe_image: Optional[np.ndarray] = None,
        repeat_counts_by_candidate: Optional[Dict[int, int]] = None,
    ) -> StripePageResult:
        """
        Thực thi phát hiện và kiểm định vệt sọc trên mảng ảnh NumPy của một trang.
        Nếu ảnh lỗi hoặc gặp ngoại lệ, trả về `status = PROCESSING_ERROR` (tuyệt đối không gán `CLEAN`).
        """
        t0 = time.perf_counter()
        try:
            maps, merged_candidates = self.extract_candidates(
                image=image,
                clahe_image=clahe_image,
                dpi=dpi,
            )
            margins = estimate_content_margins(maps)
            validated = validate_stripe_candidates(
                candidates=merged_candidates,
                maps=maps,
                min_relative_length=self.min_relative_length,
                max_thickness_px=self.max_thickness_px,
                min_contrast=self.min_contrast,
                confirm_threshold=self.confirm_threshold,
                review_threshold=self.review_threshold,
                repeat_counts_by_candidate=repeat_counts_by_candidate,
                horizontal_min_page_ratio=self.horizontal_min_page_ratio,
                horizontal_margin_extension=self.horizontal_margin_extension,
                content_margins=margins,
            )

            confirmed = [v for v in validated if v.status == "CONFIRMED"]
            suspected = [v for v in validated if v.status == "SUSPECTED"]
            rejected = [v for v in validated if v.status == "REJECTED"]
            active_stripes = confirmed + suspected

            if confirmed:
                page_status = STRIPE_STATUS_DETECTED
            elif suspected:
                page_status = STRIPE_STATUS_REVIEW
            else:
                page_status = STRIPE_STATUS_CLEAN

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return StripePageResult(
                page_number=page_number,
                status=page_status,
                has_stripe=bool(active_stripes),
                candidates_count=len(merged_candidates),
                confirmed_count=len(confirmed),
                suspected_count=len(suspected),
                stripes=active_stripes,
                rejected_candidates=rejected,
                raw_candidates=merged_candidates,
                content_margins=margins,
                processing_time_ms=elapsed_ms,
                error=None,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            err_msg = f"StripeDetector lỗi trên trang {page_number}: {exc}"
            logger.error(err_msg)
            return StripePageResult(
                page_number=page_number,
                status=STRIPE_STATUS_ERROR,
                has_stripe=False,
                candidates_count=0,
                confirmed_count=0,
                suspected_count=0,
                stripes=[],
                rejected_candidates=[],
                raw_candidates=[],
                processing_time_ms=elapsed_ms,
                error=err_msg,
            )

    def build_defect_results(
        self,
        page_result: StripePageResult,
        dpi: int = 160,
        coordinate_transform: Optional[CoordinateTransform] = None,
    ) -> List[DefectResult]:
        """
        Chuyển đổi `StripePageResult` sang danh sách `DefectResult` chuẩn của `DocumentPipeline`.
        """
        if page_result.status == STRIPE_STATUS_ERROR:
            return [
                DefectResult(
                    defect_type=DefectType.STRIPE.value,
                    detector_id=self.detector_id,
                    status=DefectStatus.DETECTOR_FAILED.value,
                    requires_review=True,
                    error=page_result.error or "Lỗi xử lý StripeDetector.",
                    coordinate_space=CoordinateSpace.RENDERED_IMAGE_PX.value,
                    dpi=dpi,
                    details={
                        "page_number": page_result.page_number,
                        "stripe_page_status": STRIPE_STATUS_ERROR,
                        "processing_time_ms": page_result.processing_time_ms,
                    },
                )
            ]

        if not page_result.stripes:
            return [
                DefectResult(
                    defect_type=DefectType.STRIPE.value,
                    detector_id=self.detector_id,
                    status=DefectStatus.NO_DEFECT.value,
                    requires_review=False,
                    raw_confidence=1.0,
                    confidence=1.0,
                    suspicion_score=0.0,
                    confidence_threshold=self.confirm_threshold,
                    coordinate_space=CoordinateSpace.RENDERED_IMAGE_PX.value,
                    dpi=dpi,
                    reason="Không phát hiện vệt sọc bất thường trên trang.",
                    details={
                        "page_number": page_result.page_number,
                        "stripe_page_status": STRIPE_STATUS_CLEAN,
                        "candidates_evaluated": page_result.candidates_count,
                        "rejected_false_positives": len(page_result.rejected_candidates),
                        "processing_time_ms": page_result.processing_time_ms,
                    },
                )
            ]

        defects: List[DefectResult] = []
        for s in page_result.stripes:
            is_confirmed = s.status == "CONFIRMED"
            d_status = (
                DefectStatus.CONFIRMED.value
                if is_confirmed
                else DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
            )
            requires_rev = not is_confirmed

            bbox_pdf_pt: Optional[Tuple[float, float, float, float]] = None
            if coordinate_transform is not None:
                try:
                    bbox_pdf_pt = coordinate_transform.bbox_px_to_pdf(s.bbox)
                except Exception:
                    bbox_pdf_pt = None

            feat_dict = s.features.to_dict() if s.features is not None else {}
            methods_str = (
                "+".join(s.features.detected_by)
                if (s.features and s.features.detected_by)
                else "cv_hybrid"
            )

            defects.append(
                DefectResult(
                    defect_type=s.defect_type,  # VERTICAL_STRIPE, HORIZONTAL_STRIPE, DIAGONAL_STRIPE, BAND_STRIPE
                    detector_id=self.detector_id,
                    status=d_status,
                    requires_review=requires_rev,
                    raw_confidence=s.confidence,
                    confidence=s.confidence,
                    suspicion_score=s.suspicion_score,
                    confidence_threshold=self.confirm_threshold,
                    detection_method=methods_str,
                    bbox=s.bbox,
                    bbox_pdf_pt=bbox_pdf_pt,
                    stripe_type=s.stripe_type,
                    angle_deg=s.angle_deg,
                    length_px=s.length_px,
                    width_px=s.width_px,
                    relative_length=s.relative_length,
                    severity=s.severity,
                    reason=s.reason,
                    message=s.reason,
                    coordinate_space=CoordinateSpace.RENDERED_IMAGE_PX.value,
                    dpi=dpi,
                    features=feat_dict,
                    details={
                        "page_number": page_result.page_number,
                        "stripe_page_status": page_result.status,
                        "endpoints_px": list(s.endpoints),
                        "horizontal_classification": s.horizontal_classification,
                        "reason_codes": list(s.reason_codes),
                        "content_margins": (
                            page_result.content_margins.to_dict()
                            if page_result.content_margins is not None
                            else None
                        ),
                        "processing_time_ms": page_result.processing_time_ms,
                    },
                )
            )
        return defects

    def detect(self, context: PageContext) -> List[DefectResult]:
        """
        Thực thi phát hiện sọc từ `PageContext` dùng chung trong `DocumentPipeline`.
        Đảm bảo KHÔNG chỉnh sửa `context.original_image`.
        """
        clahe_img: Optional[np.ndarray] = None
        try:
            clahe_img = context.ensure_clahe()
        except Exception:
            clahe_img = None

        page_result = self.detect_page(
            image=context.original_image,
            page_number=context.page_number,
            dpi=context.dpi,
            clahe_image=clahe_img,
        )
        return self.build_defect_results(
            page_result=page_result,
            dpi=context.dpi,
            coordinate_transform=context.coordinate_transform,
        )

    def apply_cross_page_repetition_evidence(
        self,
        page_results: Dict[int, StripePageResult],
    ) -> None:
        """
        Phân tích bằng chứng lặp lại vị trí sọc (`position_norm` ± 0.012) giữa nhiều trang
        trong cùng một tài liệu PDF mà không cần giữ ảnh các trang trong RAM.
        """
        if len(page_results) < 2:
            return

        # Thu thập tất cả các ứng viên không bị loại bởi bộ lọc False Positive cứng
        eligible_by_page: Dict[int, List[ValidatedStripe]] = {}
        for pno, res in page_results.items():
            if res.status == STRIPE_STATUS_ERROR:
                continue
            all_v = list(res.stripes) + [
                r for r in res.rejected_candidates if r.rejection_category == "below_threshold"
            ]
            eligible = [
                v
                for v in all_v
                if v.features is not None
                and v.stripe_type in ("VERTICAL", "HORIZONTAL", "BAND")
            ]
            if eligible:
                eligible_by_page[pno] = eligible

        if len(eligible_by_page) < 2:
            return

        for pno, candidates in eligible_by_page.items():
            res = page_results[pno]
            updated_any = False
            for cand in candidates:
                feat = cand.features
                if feat is None:
                    continue
                matched_pages = {pno}
                for other_pno, other_list in eligible_by_page.items():
                    if other_pno == pno:
                        continue
                    for other_c in other_list:
                        ofeat = other_c.features
                        if ofeat is None:
                            continue
                        if abs(feat.angle_deg - ofeat.angle_deg) <= 5.0 and abs(
                            feat.position_norm - ofeat.position_norm
                        ) <= 0.012:
                            matched_pages.add(other_pno)
                            break

                rep_count = len(matched_pages)
                if (
                    rep_count >= 2
                    and feat.repeat_page_count < rep_count
                    and "LIKELY_FOOTER_SEPARATOR" not in cand.reason_codes
                ):
                    old_bonus = 0.12 if feat.repeat_page_count >= 2 else 0.0
                    feat.repeat_page_count = rep_count
                    new_score = round(
                        min(1.0, max(0.0, cand.suspicion_score - old_bonus + 0.12)), 3
                    )
                    cand.suspicion_score = new_score
                    cand.confidence = new_score
                    if cand.horizontal_score is not None:
                        cand.horizontal_score = new_score
                    if "REPEATED_ACROSS_PAGES" not in cand.reason_codes:
                        cand.reason_codes.append("REPEATED_ACROSS_PAGES")
                    if new_score >= self.confirm_threshold:
                        cand.status = "CONFIRMED"
                        cand.rejection_category = None
                        if cand.horizontal_classification is not None:
                            cand.horizontal_classification = HORIZONTAL_CLASS_SCAN_STRIPE
                        if cand.severity == "LOW":
                            cand.severity = "MEDIUM"
                    elif new_score >= self.review_threshold:
                        cand.status = "SUSPECTED"
                        cand.rejection_category = None
                    if "Lặp lại tại cùng vị trí" not in cand.reason:
                        cand.reason = (
                            f"{cand.reason} [Lặp lại tại cùng vị trí trên {rep_count} trang]"
                        )
                    updated_any = True

            if updated_any:
                all_items = list(res.stripes) + list(res.rejected_candidates)
                # Khử trùng lặp theo đối tượng
                seen_ids = set()
                unique_items: List[ValidatedStripe] = []
                for item in all_items:
                    if id(item) not in seen_ids:
                        seen_ids.add(id(item))
                        unique_items.append(item)

                confirmed = [v for v in unique_items if v.status == "CONFIRMED"]
                suspected = [v for v in unique_items if v.status == "SUSPECTED"]
                rejected = [v for v in unique_items if v.status == "REJECTED"]
                res.stripes = confirmed + suspected
                res.rejected_candidates = rejected
                res.confirmed_count = len(confirmed)
                res.suspected_count = len(suspected)
                res.has_stripe = bool(res.stripes)
                if confirmed:
                    res.status = STRIPE_STATUS_DETECTED
                elif suspected:
                    res.status = STRIPE_STATUS_REVIEW
                else:
                    res.status = STRIPE_STATUS_CLEAN



def render_stripe_overlay(
    image: np.ndarray,
    stripes: Sequence[Union[DefectResult, ValidatedStripe, Dict[str, Any]]],
) -> np.ndarray:
    """
    Vẽ lớp phủ (overlay) trực quan hóa vị trí các vệt sọc phát hiện được lên bản sao ảnh trang PDF:
    - Sọc `CONFIRMED` (`STRIPE_DETECTED`): khung bao và nhãn màu Đỏ (`(220, 38, 38)`).
    - Sọc `SUSPECTED_LOW_CONFIDENCE` (`REVIEW`): khung bao và nhãn màu Cam (`(234, 88, 12)`).
    - Tuyệt đối KHÔNG làm thay đổi mảng `image` gốc truyền vào.
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào của render_stripe_overlay bị rỗng.")

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        canvas = image.copy()

    overlay = canvas.copy()
    h, w = canvas.shape[:2]

    for idx, item in enumerate(stripes, start=1):
        if isinstance(item, DefectResult):
            if item.status in (DefectStatus.NO_DEFECT.value, DefectStatus.NOT_CHECKED.value):
                continue
            bbox = item.bbox
            status = item.status
            s_type = item.stripe_type or item.defect_type
            score = item.suspicion_score if item.suspicion_score is not None else (item.confidence or 0.0)
            endpoints = item.details.get("endpoints_px") if item.details else None
        elif isinstance(item, ValidatedStripe):
            if item.status == "REJECTED":
                continue
            bbox = item.bbox
            status = item.status
            s_type = item.stripe_type
            score = item.suspicion_score
            endpoints = list(item.endpoints)
        elif isinstance(item, dict):
            status = str(item.get("status", ""))
            if status in ("NO_DEFECT", "REJECTED", "NOT_CHECKED"):
                continue
            raw_bbox = item.get("bbox")
            bbox = tuple(raw_bbox) if raw_bbox and len(raw_bbox) == 4 else None  # type: ignore[assignment]
            s_type = str(item.get("stripe_type") or item.get("defect_type") or "STRIPE")
            score = float(item.get("suspicion_score") or item.get("confidence") or 0.0)
            endpoints = item.get("details", {}).get("endpoints_px") if isinstance(item.get("details"), dict) else None
        else:
            continue

        if bbox is None or len(bbox) != 4:
            continue

        is_confirmed = status in (DefectStatus.CONFIRMED.value, "CONFIRMED", "STRIPE_DETECTED")
        color = (220, 38, 38) if is_confirmed else (234, 88, 12)

        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        pad = 4
        rx0 = max(0, x0 - pad)
        ry0 = max(0, y0 - pad)
        rx1 = min(w - 1, x1 + pad)
        ry1 = min(h - 1, y1 + pad)

        cv2.rectangle(overlay, (rx0, ry0), (rx1, ry1), color, -1)
        cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), color, 2)

        if endpoints and len(endpoints) == 4:
            pt1 = (int(round(endpoints[0])), int(round(endpoints[1])))
            pt2 = (int(round(endpoints[2])), int(round(endpoints[3])))
            cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

        label = f"#{idx} {s_type} ({score:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        lx = max(4, min(w - tw - 10, rx0))
        ly = max(th + 8, ry0 - 6) if ry0 > th + 14 else min(h - 6, ry0 + th + 10)
        cv2.rectangle(
            canvas,
            (lx - 3, ly - th - 4),
            (lx + tw + 4, ly + 4),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (lx, ly),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    blended = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0)
    return blended
