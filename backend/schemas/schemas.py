"""
Định nghĩa toàn bộ các Data Models / Schemas (`src/pdf_inspector/models/schemas.py`).
Hỗ trợ kiến trúc Multi-Defect PDF Quality Inspection (Multi-label Defect Detection),
đồng thời giữ tương thích ngược 100% với báo cáo Orientation hiện có.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ai_engine.core.config import STATUS_REVIEW, STATUS_ROTATED, DetectorConfig
from backend.schemas.context import CoordinateTransform, PageContext
from backend.repositories.file_utils import combine_rotation_angles, normalize_angle


class DetectorScope(str, Enum):
    """Phạm vi hoạt động của một Detector."""

    PAGE_LEVEL = "PAGE_LEVEL"
    DOCUMENT_LEVEL = "DOCUMENT_LEVEL"
    CROSS_PAGE = "CROSS_PAGE"


class DetectorStatus(str, Enum):
    """Trạng thái của một Detector trong DetectorRegistry."""

    AVAILABLE = "AVAILABLE"
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class DefectType(str, Enum):
    """Các loại lỗi chất lượng tài liệu PDF trong kiến trúc tổng quát."""

    ROTATION = "ROTATION"
    STRIPE = "STRIPE"
    BOUNDARY = "BOUNDARY"
    MIXED_PAGE = "MIXED_PAGE"
    SPLIT_PAGE = "SPLIT_PAGE"
    CONTENT_OVERLAP = "CONTENT_OVERLAP"
    CONTENT_TRUNCATED = "CONTENT_TRUNCATED"
    PAGE_BOUNDARY_SUSPECTED = "PAGE_BOUNDARY_SUSPECTED"


class DefectStatus(str, Enum):
    """
    Trạng thái kết luận cụ thể cho từng loại lỗi trên một trang.
    Phân biệt rõ 5 trường hợp:
    - NO_DEFECT: Đã kiểm tra với độ tin cậy đủ cao và xác nhận không có lỗi này.
    - CONFIRMED: Đã kiểm tra và xác nhận có lỗi với độ tin cậy >= ngưỡng.
    - SUSPECTED_LOW_CONFIDENCE: Phát hiện dấu hiệu lỗi hoặc kết quả có độ tin cậy < ngưỡng / mâu thuẫn, cần người dùng kiểm tra.
    - DETECTOR_FAILED: Detector không thể phân tích trang (ví dụ trang trắng, quá ít chữ, lỗi OCR), cần người dùng kiểm tra.
    - NOT_CHECKED: Loại lỗi này chưa được kiểm tra (do detector đang DISABLED hoặc NOT_IMPLEMENTED).
    """

    NO_DEFECT = "NO_DEFECT"
    CONFIRMED = "CONFIRMED"
    SUSPECTED_LOW_CONFIDENCE = "SUSPECTED_LOW_CONFIDENCE"
    DETECTOR_FAILED = "DETECTOR_FAILED"
    NOT_CHECKED = "NOT_CHECKED"


class InspectionStatus(str, Enum):
    """Trạng thái thực thi kiểm tra chất lượng của một trang PDF."""

    COMPLETED = "COMPLETED"
    COMPLETED_WITH_REVIEW = "COMPLETED_WITH_REVIEW"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    FAILED = "FAILED"


@dataclass
class PDFDocumentInfo:
    """Thông tin tổng quan và metadata của tài liệu PDF."""

    file_path: str
    file_name: str
    file_size_bytes: int
    page_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_rotations: Dict[int, int] = field(default_factory=dict)
    has_digital_signature: bool = False
    is_encrypted: bool = False


@dataclass
class OSDAttemptResult:
    """Kết quả của một lần chạy nhận diện hướng đơn lẻ."""

    method: str  # Ví dụ: "tesseract_osd", "tesseract_osd_preprocessed"
    status: str  # "OK" hoặc "UNKNOWN"
    orientation: Optional[int] = None
    rotate: Optional[int] = None
    orientation_conf: Optional[float] = None
    script: Optional[str] = None
    script_conf: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrientationDetectionResult:
    """
    Schema chuẩn hóa kết quả trả về từ bộ phát hiện hướng (Orientation Detector):
    - `rotate`: Góc dự đoán cần xoay theo chiều kim đồng hồ để trang về đúng chiều (0, 90, 180, 270).
    - `orientation`: Góc định hướng hiện tại của trang (0, 90, 180, 270).
    - `orientation_conf`: Điểm tin cậy.
    - `detection_method`: Phương pháp nhận diện.
    - `status`: Trạng thái ("OK" hoặc "UNKNOWN").
    - `error`: Thông tin lỗi nếu có.
    """

    page_number: int
    status: str  # "OK" hoặc "UNKNOWN"
    orientation: Optional[int] = None
    rotate: Optional[int] = None
    orientation_conf: Optional[float] = None
    script: Optional[str] = None
    script_conf: Optional[float] = None
    existing_pdf_rotation: int = 0
    detection_method: str = "tesseract_osd"
    has_conflict: bool = False
    error: Optional[str] = None
    attempts: List[OSDAttemptResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DefectResult:
    """
    Kết quả chuẩn hóa của một loại lỗi trên trang PDF (Multi-label Defect Detection).
    Một trang có thể chứa 0, 1 hoặc nhiều `DefectResult` từ các detector khác nhau
    (ví dụ: Orientation và Stripe).
    """

    defect_type: str  # Ví dụ: "ROTATION", "STRIPE", "VERTICAL_STRIPE", "HORIZONTAL_STRIPE", "DIAGONAL_STRIPE", "BAND_STRIPE"
    detector_id: str  # Ví dụ: "orientation", "stripe"
    status: str  # Giá trị từ DefectStatus
    requires_review: bool = False
    correction_angle: Optional[int] = None
    detected_rotation: Optional[int] = None
    detected_orientation: Optional[int] = None
    raw_confidence: Optional[float] = None
    confidence: Optional[float] = None
    suspicion_score: Optional[float] = None
    confidence_threshold: Optional[float] = None
    detection_method: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    bbox_pdf_pt: Optional[Tuple[float, float, float, float]] = None
    stripe_type: Optional[str] = None  # "VERTICAL", "HORIZONTAL", "DIAGONAL", "BAND"
    angle_deg: Optional[float] = None
    length_px: Optional[float] = None
    width_px: Optional[float] = None
    relative_length: Optional[float] = None
    severity: Optional[str] = None
    reason: Optional[str] = None
    coordinate_space: Optional[str] = None
    dpi: Optional[int] = None
    boundary_candidates: List[Dict[str, Any]] = field(default_factory=list)
    related_pages: List[int] = field(default_factory=list)
    reconstruction_proposal: Optional[Dict[str, Any]] = None
    features: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "defect_type": self.defect_type,
            "detector_id": self.detector_id,
            "status": self.status,
            "requires_review": self.requires_review,
        }
        if self.correction_angle is not None:
            data["correction_angle"] = self.correction_angle
        if self.detected_rotation is not None:
            data["detected_rotation"] = self.detected_rotation
        if self.detected_orientation is not None:
            data["detected_orientation"] = self.detected_orientation
        if self.raw_confidence is not None:
            data["raw_confidence"] = self.raw_confidence
        if self.confidence is not None:
            data["confidence"] = self.confidence
        if self.suspicion_score is not None:
            data["suspicion_score"] = self.suspicion_score
        if self.confidence_threshold is not None:
            data["confidence_threshold"] = self.confidence_threshold
        if self.detection_method is not None:
            data["detection_method"] = self.detection_method
        if self.bbox is not None:
            data["bbox"] = list(self.bbox)
        if self.bbox_pdf_pt is not None:
            data["bbox_pdf_pt"] = list(self.bbox_pdf_pt)
        if self.stripe_type is not None:
            data["stripe_type"] = self.stripe_type
        if self.angle_deg is not None:
            data["angle_deg"] = self.angle_deg
        if self.length_px is not None:
            data["length_px"] = self.length_px
        if self.width_px is not None:
            data["width_px"] = self.width_px
        if self.relative_length is not None:
            data["relative_length"] = self.relative_length
        if self.severity is not None:
            data["severity"] = self.severity
        if self.reason is not None:
            data["reason"] = self.reason
        if self.coordinate_space is not None:
            data["coordinate_space"] = self.coordinate_space
        if self.dpi is not None:
            data["dpi"] = self.dpi
        if self.boundary_candidates:
            data["boundary_candidates"] = list(self.boundary_candidates)
        if self.related_pages:
            data["related_pages"] = list(self.related_pages)
        if self.reconstruction_proposal is not None:
            data["reconstruction_proposal"] = dict(self.reconstruction_proposal)
        if self.features:
            data["features"] = self.features
        if self.message:
            data["message"] = self.message
        if self.error:
            data["error"] = self.error
        if self.details:
            data["details"] = self.details
        return data


@dataclass
class PageDecision:
    """
    Kết quả kiểm tra chất lượng hợp nhất (Unified Quality Result) và quyết định xử lý cho một trang PDF.
    Hỗ trợ đồng thời:
    - Multi-Defect Quality Inspection (`inspection_status`, `defects`, `requires_review`,
      `stripe_status`, `stripe_defects`, `enabled_detectors`, `unsupported_checks`, `detector_outcomes`).
    - Tương thích ngược hoàn toàn với báo cáo Orientation (`status`, `detected_rotation`,
      `orientation_conf`, `existing_pdf_rotation`, `correction_angle`, `corrected`, `detection_method`, `error`).
    """

    page_number: int
    status: str  # "NORMAL", "ROTATED", "REVIEW" (Trạng thái của kiểm tra Orientation)
    detected_rotation: Optional[int]
    orientation_conf: Optional[float]
    existing_pdf_rotation: int
    correction_angle: int  # Góc sẽ áp dụng để xoay trang (0, 90, 180, 270)
    corrected: bool  # Trang đã thực sự được xoay trong file xuất hay chưa
    detection_method: str
    error: Optional[str] = None
    detected_orientation: Optional[int] = None
    script: Optional[str] = None
    script_conf: Optional[float] = None
    has_conflict: bool = False
    manual_override: bool = False
    should_rotate: bool = False
    # Các trường của Unified Quality Result (Multi-Defect Architecture)
    inspection_status: str = InspectionStatus.COMPLETED.value
    defects: List[DefectResult] = field(default_factory=list)
    stripe_status: str = "CLEAN"  # "CLEAN", "STRIPE_DETECTED", "REVIEW", "PROCESSING_ERROR", "NOT_CHECKED"
    stripe_defects: List[DefectResult] = field(default_factory=list)
    boundary_status: str = "NORMAL"  # "NORMAL", "BOUNDARY_SUSPECTED", "MIXED_PAGE_DETECTED", "RECONSTRUCTABLE", "MISSING_CONTENT", "REVIEW", "PROCESSING_ERROR", "NOT_CHECKED"
    boundary_defects: List[DefectResult] = field(default_factory=list)
    boundary_candidates: List[Dict[str, Any]] = field(default_factory=list)
    related_pages: List[int] = field(default_factory=list)
    reconstruction_proposal: Optional[Dict[str, Any]] = None
    requires_review: bool = False
    enabled_detectors: List[str] = field(
        default_factory=lambda: ["orientation", "stripe", "boundary"]
    )
    unsupported_checks: List[str] = field(default_factory=list)
    detector_outcomes: Dict[str, str] = field(default_factory=dict)

    @property
    def target_pdf_rotation(self) -> int:
        """Góc xoay metadata PDF mới sau khi cộng góc sửa theo modulo 360."""
        if self.correction_angle == 0:
            return normalize_angle(self.existing_pdf_rotation)
        return combine_rotation_angles(self.existing_pdf_rotation, self.correction_angle)

    @property
    def action_label(self) -> str:
        """Nhãn hành động hiển thị trên bảng kết quả UI (Rotate / Review / None)."""
        if self.manual_override:
            return (
                f"Manual ({self.correction_angle}°)"
                if self.correction_angle != 0
                else "Keep (0°)"
            )
        if self.status == STATUS_ROTATED:
            return "Rotate"
        if self.status == STATUS_REVIEW or self.requires_review:
            return "Review"
        return "None"

    def to_report_dict(self) -> Dict[str, Any]:
        """
        Xuất dict phục vụ báo cáo JSON/CSV, bao gồm cả schema Unified Quality Result
        lẫn các trường Orientation, Stripe & Boundary.
        """
        return {
            "page_number": self.page_number,
            "status": self.status,
            "stripe_status": self.stripe_status,
            "boundary_status": self.boundary_status,
            "inspection_status": self.inspection_status,
            "requires_review": self.requires_review,
            "detected_rotation": self.detected_rotation,
            "orientation_conf": self.orientation_conf,
            "existing_pdf_rotation": self.existing_pdf_rotation,
            "correction_angle": self.correction_angle,
            "corrected": self.corrected,
            "detection_method": self.detection_method,
            "error": self.error,
            "defects": [d.to_dict() for d in self.defects],
            "stripe_defects": [d.to_dict() for d in self.stripe_defects],
            "boundary_defects": [d.to_dict() for d in self.boundary_defects],
            "boundary_candidates": list(self.boundary_candidates),
            "related_pages": list(self.related_pages),
            "reconstruction_proposal": (
                dict(self.reconstruction_proposal)
                if self.reconstruction_proposal is not None
                else None
            ),
            "detector_outcomes": dict(self.detector_outcomes),
            "enabled_detectors": list(self.enabled_detectors),
            "unsupported_checks": list(self.unsupported_checks),
        }


# Bí danh ngữ nghĩa cho kiến trúc Multi-Defect Quality Inspection
PageQualityResult = PageDecision


@dataclass
class CorrectionSummary:
    """Kết quả tóm tắt sau khi thực hiện chỉnh sửa góc xoay PDF."""

    input_path: str
    output_path: str
    total_pages: int
    corrected_pages: List[int] = field(default_factory=list)
    corrected_count: int = 0
    has_digital_signature: bool = False
    warnings: List[str] = field(default_factory=list)
    supported_corrections: List[str] = field(
        default_factory=lambda: [DefectType.ROTATION.value]
    )
    unsupported_corrections: List[str] = field(
        default_factory=lambda: [DefectType.STRIPE.value, DefectType.BOUNDARY.value]
    )


@dataclass
class PipelineResult:
    """
    Kết quả toàn diện của một phiên kiểm tra chất lượng và chỉnh sửa PDF
    (dùng chung cho cả CLI và Streamlit).
    """

    doc_info: PDFDocumentInfo
    config: DetectorConfig
    detections: List[OrientationDetectionResult]
    decisions: List[PageDecision]
    processing_time_seconds: float
    output_pdf_path: Optional[str] = None
    json_report_path: Optional[str] = None
    csv_report_path: Optional[str] = None
    report_data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    system_errors: List[str] = field(default_factory=list)
    enabled_detectors: List[str] = field(
        default_factory=lambda: ["orientation", "stripe", "boundary"]
    )
    unsupported_checks: List[str] = field(default_factory=list)
    detector_registry_info: List[Dict[str, Any]] = field(default_factory=list)
    additional_defects_by_page: Dict[int, List[DefectResult]] = field(
        default_factory=dict
    )


ProgressCallback = Callable[[int, int, PageDecision], None]
