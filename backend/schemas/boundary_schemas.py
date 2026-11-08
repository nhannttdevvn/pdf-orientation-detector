"""
Schemas dữ liệu cho Phase 3 — Page Boundary & Mixed-Page Detection (`src/pdf_inspector/models/boundary_schemas.py`).

Bao gồm:
- Phân loại lỗi ranh giới trang (`BoundaryDefectCode`: `MIXED_PAGE`, `SPLIT_PAGE`, `CONTENT_OVERLAP`, `CONTENT_TRUNCATED`, `PAGE_BOUNDARY_SUSPECTED`).
- Trạng thái quyết định Boundary trên từng trang (`BoundaryStatus`: `NORMAL`, `BOUNDARY_SUSPECTED`, `MIXED_PAGE_DETECTED`, `RECONSTRUCTABLE`, `MISSING_CONTENT`, `REVIEW`, `PROCESSING_ERROR`, `NOT_CHECKED`).
- Các cấu trúc đặc trưng bố cục (`TextRegionBlock`, `WhitespaceBand`, `EdgeTruncationSignal`).
- Ứng viên ranh giới trang (`BoundaryCandidate`) hỗ trợ đầy đủ hệ tọa độ `RENDERED_IMAGE_PX`, `NORMALIZED` (`normalized_page`) và `PDF_POINT`.
- Kết quả đối chiếu liên trang (`CrossPageMatchResult`) giữa trang `N` với `N-1` và `N+1`.
- Đề xuất tái cấu trúc tách/ghép trang (`ReconstructionProposal`) kèm truy vết nguồn gốc vùng ảnh (`source_regions`).
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BoundaryDefectCode(str, Enum):
    """Các mã lỗi cụ thể thuộc nhóm Page Boundary & Mixed-Page Detection."""

    MIXED_PAGE = "MIXED_PAGE"
    SPLIT_PAGE = "SPLIT_PAGE"
    CONTENT_OVERLAP = "CONTENT_OVERLAP"
    CONTENT_TRUNCATED = "CONTENT_TRUNCATED"
    PAGE_BOUNDARY_SUSPECTED = "PAGE_BOUNDARY_SUSPECTED"


class BoundaryStatus(str, Enum):
    """Trạng thái phân loại chất lượng Boundary của một trang PDF (Phase 3.7)."""

    NORMAL = "NORMAL"
    BOUNDARY_SUSPECTED = "BOUNDARY_SUSPECTED"
    MIXED_PAGE_DETECTED = "MIXED_PAGE_DETECTED"
    RECONSTRUCTABLE = "RECONSTRUCTABLE"
    MISSING_CONTENT = "MISSING_CONTENT"
    REVIEW = "REVIEW"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    NOT_CHECKED = "NOT_CHECKED"


@dataclass
class TextRegionBlock:
    """Một khối nội dung văn bản/bảng biểu phát hiện được trên trang."""

    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) theo pixel
    bbox_norm: Tuple[float, float, float, float]  # (x0/W, y0/H, x1/W, y1/H)
    area_px: int
    line_count: int
    mean_line_height_px: float
    left_margin_norm: float
    right_margin_norm: float
    width_norm: float
    height_norm: float
    density: float
    region_role: str = "body"  # "header", "footer", "page_number", "heading", "body", "table", "signature", "column"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["bbox_norm"] = [round(v, 4) for v in self.bbox_norm]
        return data


@dataclass
class WhitespaceBand:
    """Khoảng trắng lớn theo chiều ngang hoặc dọc có thể là ranh giới giữa hai khối/trang."""

    orientation: str  # "HORIZONTAL" hoặc "VERTICAL"
    start_px: int
    end_px: int
    center_px: int
    center_norm: float
    thickness_px: int
    thickness_norm: float
    mean_intensity: float
    std_intensity: float
    has_seam_line: bool = False
    seam_strength: float = 0.0
    has_illumination_step: bool = False
    illumination_delta: float = 0.0
    crosses_full_width: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeTruncationSignal:
    """Tín hiệu phát hiện nội dung nằm sát mép ảnh và có dấu hiệu bị cắt mất (`CONTENT_TRUNCATED`)."""

    edge_side: str  # "TOP", "BOTTOM", "LEFT", "RIGHT"
    bbox: Tuple[int, int, int, int]
    bbox_norm: Tuple[float, float, float, float]
    clipped_pixel_ratio: float
    clipped_components_count: int
    char_height_ratio: float
    suspicion_score: float
    status: str  # "CONFIRMED", "SUSPECTED", "REJECTED"
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["bbox_norm"] = [round(v, 4) for v in self.bbox_norm]
        return data


@dataclass
class BoundaryCandidate:
    """
    Ứng viên ranh giới trang bất thường trên một trang PDF (Phase 3.1, 3.2 & 3.6).
    Hỗ trợ chuyển đổi tọa độ giữa ảnh render (`position_px`), tọa độ chuẩn hóa (`position`)
    và tọa độ PDF gốc (`position_pdf_pt`).
    """

    candidate_id: str
    orientation: str  # "HORIZONTAL" hoặc "VERTICAL"
    position: float  # Tọa độ chuẩn hóa [0.0 .. 1.0] trong hệ "normalized_page"
    position_px: int  # Tọa độ pixel trên ảnh phân tích
    position_pdf_pt: float  # Tọa độ point trên trang PDF
    band_bbox_px: Tuple[int, int, int, int]  # (x0, y0, x1, y1) của vùng ranh giới
    band_bbox_norm: Tuple[float, float, float, float]
    coordinate_system: str = "normalized_page"
    detected_by: List[str] = field(default_factory=list)
    suspicion_score: float = 0.0
    status: str = "SUSPECTED"  # "CONFIRMED", "SUSPECTED", "REJECTED"
    rejection_category: Optional[str] = None
    safe_cut_line_px: int = 0
    safe_cut_line_norm: float = 0.0
    crosses_text: bool = False
    reason: str = ""
    features: Dict[str, Any] = field(default_factory=dict)

    @property
    def boundary_norm(self) -> float:
        """Tọa độ chuẩn hóa [0.0 .. 1.0] của ứng viên ranh giới."""
        return float(self.position)

    @property
    def is_false_positive(self) -> bool:
        """Trả về True nếu ứng viên đã bị bộ lọc loại bỏ là False Positive (REJECTED)."""
        return self.status == "REJECTED" or self.rejection_category is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "orientation": self.orientation,
            "position": round(float(self.position), 4),
            "position_px": int(self.position_px),
            "position_pdf_pt": round(float(self.position_pdf_pt), 2),
            "band_bbox_px": list(self.band_bbox_px),
            "band_bbox_norm": [round(float(v), 4) for v in self.band_bbox_norm],
            "coordinate_system": self.coordinate_system,
            "detected_by": list(self.detected_by),
            "suspicion_score": round(float(self.suspicion_score), 3),
            "status": self.status,
            "rejection_category": self.rejection_category,
            "safe_cut_line_px": int(self.safe_cut_line_px),
            "safe_cut_line_norm": round(float(self.safe_cut_line_norm), 4),
            "crosses_text": bool(self.crosses_text),
            "reason": self.reason,
            "features": dict(self.features),
        }


@dataclass
class CrossPageMatchResult:
    """Kết quả đối chiếu liên trang giữa trang `source_page` và trang liền kề `target_page` (Phase 3.4)."""

    source_page: int
    target_page: int
    relation_type: str  # "CONTENT_OVERLAP", "SPLIT_CONTINUATION", "MIXED_SPILLOVER", "NONE"
    ncc_score: float = 0.0
    orb_inlier_ratio: float = 0.0
    profile_correlation: float = 0.0
    width_consistency_score: float = 0.0
    complementary_height_score: float = 0.0
    ocr_overlap_ratio: Optional[float] = None
    overlap_bbox_source_px: Optional[Tuple[int, int, int, int]] = None
    overlap_bbox_target_px: Optional[Tuple[int, int, int, int]] = None
    overlap_height_norm: float = 0.0
    combined_score: float = 0.0
    confirmed: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.overlap_bbox_source_px is not None:
            data["overlap_bbox_source_px"] = list(self.overlap_bbox_source_px)
        if self.overlap_bbox_target_px is not None:
            data["overlap_bbox_target_px"] = list(self.overlap_bbox_target_px)
        return data


@dataclass
class ReconstructionProposal:
    """
    Đề xuất cắt/ghép phục hồi trang (Phase 3.8).
    Lưu rõ nguồn gốc từng vùng ảnh (`source_regions`) và đảm bảo không cắt xuyên qua ký tự.
    """

    proposal_id: str
    page_number: int
    related_pages: List[int]
    action_type: str  # "SPLIT_SINGLE_PAGE", "MERGE_SPLIT_PAGES", "SPLIT_AND_MERGE_SPILLOVER", "REMOVE_OVERLAP_AND_MERGE", "REQUIRES_RESCAN"
    reconstructable: bool
    cut_orientation: str = "HORIZONTAL"
    cut_position_px: int = 0
    cut_position_norm: float = 0.0
    cut_position_pdf_pt: float = 0.0
    safe_to_cut: bool = True
    overlap_strip_norm: float = 0.0
    source_regions: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""

    @property
    def action(self) -> str:
        """Bí danh tương thích cho `action_type`."""
        return self.action_type

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageLayoutAnalysisResult:
    """
    Kết quả phân tích bố cục của một trang đơn lẻ (Phase 3.1) kèm dữ liệu dải biên
    nhỏ gọn phục vụ kiểm tra liên trang theo cửa sổ trượt mà không giữ toàn bộ ảnh gốc trong RAM.
    """

    page_number: int
    image_width_px: int
    image_height_px: int
    pdf_width_pt: float
    pdf_height_pt: float
    dpi: int
    analysis_rotation_applied: int = 0
    text_blocks: List[TextRegionBlock] = field(default_factory=list)
    whitespace_bands: List[WhitespaceBand] = field(default_factory=list)
    edge_truncations: List[EdgeTruncationSignal] = field(default_factory=list)
    raw_candidates: List[BoundaryCandidate] = field(default_factory=list)
    validated_candidates: List[BoundaryCandidate] = field(default_factory=list)
    is_two_column: bool = False
    is_book_spread: bool = False
    is_mostly_blank: bool = False
    has_table_structure: bool = False
    has_signature_bottom: bool = False
    content_bbox_norm: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    content_fill_ratio: float = 0.0
    top_empty_norm: float = 0.0
    bottom_empty_norm: float = 0.0
    left_margin_norm: float = 0.0
    right_margin_norm: float = 0.0
    mean_bg_intensity: float = 250.0
    optional_layout_cues: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    # Dải ảnh biên thu gọn để đối chiếu với trang N-1 và N+1 (được giải phóng sau khi cross-page hoàn tất)
    top_strip_gray: Optional[np.ndarray] = field(default=None, repr=False)
    bottom_strip_gray: Optional[np.ndarray] = field(default=None, repr=False)
    top_strip_binary: Optional[np.ndarray] = field(default=None, repr=False)
    bottom_strip_binary: Optional[np.ndarray] = field(default=None, repr=False)
    row_text_profile: Optional[np.ndarray] = field(default=None, repr=False)

    def release_strips(self) -> None:
        """Giải phóng các mảng ảnh dải biên sau khi kiểm tra liên trang hoàn tất."""
        self.top_strip_gray = None
        self.bottom_strip_gray = None
        self.top_strip_binary = None
        self.bottom_strip_binary = None
        self.row_text_profile = None

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "pdf_width_pt": self.pdf_width_pt,
            "pdf_height_pt": self.pdf_height_pt,
            "dpi": self.dpi,
            "analysis_rotation_applied": self.analysis_rotation_applied,
            "text_blocks_count": len(self.text_blocks),
            "is_two_column": self.is_two_column,
            "is_book_spread": self.is_book_spread,
            "is_mostly_blank": self.is_mostly_blank,
            "has_table_structure": self.has_table_structure,
            "has_signature_bottom": self.has_signature_bottom,
            "content_bbox_norm": [round(v, 4) for v in self.content_bbox_norm],
            "content_fill_ratio": round(self.content_fill_ratio, 4),
            "top_empty_norm": round(self.top_empty_norm, 4),
            "bottom_empty_norm": round(self.bottom_empty_norm, 4),
            "left_margin_norm": round(self.left_margin_norm, 4),
            "right_margin_norm": round(self.right_margin_norm, 4),
            "mean_bg_intensity": round(self.mean_bg_intensity, 2),
            "optional_layout_cues": dict(self.optional_layout_cues),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class BoundaryPageDecisionResult:
    """Kết quả tổng hợp của Boundary Detector cho một trang PDF sau khi kết hợp Cross-Page Validation."""

    page_number: int
    boundary_status: str  # Giá trị từ BoundaryStatus
    defects: List[Dict[str, Any]] = field(default_factory=list)
    boundary_candidates: List[BoundaryCandidate] = field(default_factory=list)
    edge_truncations: List[EdgeTruncationSignal] = field(default_factory=list)
    cross_page_matches: List[CrossPageMatchResult] = field(default_factory=list)
    related_pages: List[int] = field(default_factory=list)
    reconstruction_proposal: Optional[ReconstructionProposal] = None
    suspicion_score: float = 0.0
    requires_review: bool = False
    reason: str = ""
    layout_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def status(self) -> str:
        """Bí danh tương thích cho `boundary_status`."""
        return self.boundary_status
