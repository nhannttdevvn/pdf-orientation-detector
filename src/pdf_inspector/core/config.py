"""
Cấu hình trung tâm cho hệ thống PDF Quality Inspection (`src/pdf_inspector/core/config.py`).
Mọi đường dẫn đều sử dụng `pathlib.Path` tính từ gốc dự án, không phụ thuộc thư mục làm việc hiện tại.
"""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
from typing import Optional, Tuple

# Đường dẫn gốc của dự án (pdf-orientation-detector/ — config.py nằm tại src/pdf_inspector/core/)
BASE_DIR: Path = Path(__file__).resolve().parents[3]

# Thư mục dữ liệu (data/samples, data/uploads, data/outputs) và logs
DATA_DIR: Path = BASE_DIR / "data"
SAMPLES_DIR: Path = DATA_DIR / "samples"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
OUTPUTS_DIR: Path = DATA_DIR / "outputs"
LOGS_DIR: Path = BASE_DIR / "logs"
STRIPE_FEEDBACK_PATH: Path = OUTPUTS_DIR / "stripe_review_feedback.json"
BOUNDARY_FEEDBACK_PATH: Path = OUTPUTS_DIR / "boundary_review_feedback.json"
RECONSTRUCTED_PDF_PATH: Path = OUTPUTS_DIR / "reconstructed.pdf"
RECONSTRUCTION_PROVENANCE_PATH: Path = OUTPUTS_DIR / "reconstruction_provenance.json"

# Các hằng số cấu hình cốt lõi (hỗ trợ ghi đè qua biến môi trường nếu có)
ORIENTATION_CONF_THRESHOLD: float = float(
    os.getenv("ORIENTATION_CONF_THRESHOLD", "10.0")
)
RENDER_DPI: int = int(os.getenv("RENDER_DPI", "160"))
PREVIEW_DPI: int = int(os.getenv("PREVIEW_DPI", "120"))
AUTO_FIX: bool = os.getenv("AUTO_FIX", "false").strip().lower() in ("1", "true", "yes")
DEFAULT_DETECTOR_NAME: str = os.getenv("DETECTOR_BACKEND", "tesseract")

# Các hằng số cấu hình cho Stripe Defect Detection (Phase 2 & Phase 4)
STRIPE_CONFIRM_THRESHOLD: float = float(
    os.getenv("STRIPE_CONFIRM_THRESHOLD", "0.62")
)
STRIPE_REVIEW_THRESHOLD: float = float(
    os.getenv("STRIPE_REVIEW_THRESHOLD", "0.42")
)
STRIPE_MIN_RELATIVE_LENGTH: float = float(
    os.getenv("STRIPE_MIN_RELATIVE_LENGTH", "0.35")
)
STRIPE_MAX_THICKNESS_PX: int = int(os.getenv("STRIPE_MAX_THICKNESS_PX", "45"))
STRIPE_MIN_CONTRAST: float = float(os.getenv("STRIPE_MIN_CONTRAST", "7.5"))

# Ngưỡng chuyên biệt cho Horizontal Stripe Detection dựa trên vùng lề nội dung (Phase 4 — Part A5)
HORIZONTAL_STRIPE_MIN_PAGE_RATIO: float = float(
    os.getenv("HORIZONTAL_STRIPE_MIN_PAGE_RATIO", "0.90")
)
HORIZONTAL_STRIPE_MARGIN_EXTENSION: float = float(
    os.getenv("HORIZONTAL_STRIPE_MARGIN_EXTENSION", "0.03")
)

# Phân loại ứng viên đường ngang (Phase 4 — Part A9)
HORIZONTAL_CLASS_SCAN_STRIPE: str = "SCAN_STRIPE"
HORIZONTAL_CLASS_DOCUMENT_LINE: str = "LIKELY_DOCUMENT_LINE"
HORIZONTAL_CLASS_REVIEW: str = "REVIEW"

# Trạng thái phân loại của Stripe Detector trên từng trang
STRIPE_STATUS_CLEAN: str = "CLEAN"
STRIPE_STATUS_DETECTED: str = "STRIPE_DETECTED"
STRIPE_STATUS_REVIEW: str = "REVIEW"
STRIPE_STATUS_ERROR: str = "PROCESSING_ERROR"

# Cấu hình cho Secret Marking Detection (Phase 4 — Part C)
SECRET_TEMPLATES_DIR: Path = (
    Path(__file__).resolve().parents[1] / "detectors" / "secret" / "templates"
)
SECRET_FEEDBACK_PATH: Path = OUTPUTS_DIR / "secret_review_feedback.json"
SECRET_RENDER_DPI: int = int(os.getenv("SECRET_RENDER_DPI", "200"))
SECRET_CONFIRM_THRESHOLD: float = float(
    os.getenv("SECRET_CONFIRM_THRESHOLD", "0.70")
)
SECRET_REVIEW_THRESHOLD: float = float(
    os.getenv("SECRET_REVIEW_THRESHOLD", "0.56")
)
SECRET_STATUS_DETECTED: str = "DETECTED"
SECRET_STATUS_NO_MARK: str = "NO_MARK_DETECTED"
SECRET_STATUS_REVIEW: str = "REVIEW"
SECRET_STATUS_ERROR: str = "PROCESSING_ERROR"

# Các hằng số cấu hình cho Page Boundary & Mixed-Page Detection (Phase 3)
BOUNDARY_CONFIRM_THRESHOLD: float = float(
    os.getenv("BOUNDARY_CONFIRM_THRESHOLD", "0.64")
)
BOUNDARY_REVIEW_THRESHOLD: float = float(
    os.getenv("BOUNDARY_REVIEW_THRESHOLD", "0.42")
)
BOUNDARY_OVERLAP_THRESHOLD: float = float(
    os.getenv("BOUNDARY_OVERLAP_THRESHOLD", "0.68")
)
BOUNDARY_USE_OPTIONAL_LAYOUT_MODEL: bool = (
    os.getenv("BOUNDARY_USE_OPTIONAL_LAYOUT_MODEL", "false").strip().lower()
    in ("1", "true", "yes")
)

# Trạng thái phân loại của Boundary Decision Engine (Phase 3.7)
BOUNDARY_STATUS_NORMAL: str = "NORMAL"
BOUNDARY_STATUS_SUSPECTED: str = "BOUNDARY_SUSPECTED"
BOUNDARY_STATUS_MIXED_DETECTED: str = "MIXED_PAGE_DETECTED"
BOUNDARY_STATUS_MIXED: str = BOUNDARY_STATUS_MIXED_DETECTED
BOUNDARY_STATUS_RECONSTRUCTABLE: str = "RECONSTRUCTABLE"
BOUNDARY_STATUS_MISSING_CONTENT: str = "MISSING_CONTENT"
BOUNDARY_STATUS_MISSING: str = BOUNDARY_STATUS_MISSING_CONTENT
BOUNDARY_STATUS_REVIEW: str = "REVIEW"
BOUNDARY_STATUS_ERROR: str = "PROCESSING_ERROR"

# Các góc xoay hợp lệ (độ theo chiều kim đồng hồ)
VALID_ANGLES: Tuple[int, ...] = (0, 90, 180, 270)

# Trạng thái phân loại trang của Orientation Decision Engine
STATUS_NORMAL: str = "NORMAL"
STATUS_ROTATED: str = "ROTATED"
STATUS_REVIEW: str = "REVIEW"

# Giá trị trả về khi OCR không nhận diện được hướng
DETECTION_UNKNOWN: str = "UNKNOWN"

# Cấu hình tiền xử lý ảnh (Image Preprocessing)
MEDIAN_BLUR_KSIZE: int = 3
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)
MAX_OCR_DIMENSION: int = 2400
MIN_OCR_DIMENSION: int = 800

# Cấu hình Tesseract OSD
TESSERACT_OSD_CONFIG: str = "--psm 0"

# Các đường dẫn phổ biến của Tesseract trên macOS (Apple Silicon & Intel) và Linux
COMMON_TESSERACT_PATHS: Tuple[str, ...] = (
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/local/bin/tesseract",
    "/usr/bin/tesseract",
)


def find_tesseract_binary() -> Optional[str]:
    """
    Tự động tìm đường dẫn thực thi của Tesseract trên hệ thống
    (ưu tiên biến môi trường TESSERACT_CMD, sau đó đến PATH và các đường dẫn chuẩn).
    """
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).is_file():
        return env_cmd

    in_path = shutil.which("tesseract")
    if in_path:
        return in_path

    for candidate in COMMON_TESSERACT_PATHS:
        if Path(candidate).is_file():
            return candidate
    return None


@dataclass
class DetectorConfig:
    """
    Cấu hình truyền xuyên suốt pipeline (tái sử dụng chung cho cả CLI và Streamlit).
    """

    render_dpi: int = RENDER_DPI
    orientation_conf_threshold: float = ORIENTATION_CONF_THRESHOLD
    auto_fix: bool = AUTO_FIX
    detector_backend: str = DEFAULT_DETECTOR_NAME
    retry_with_preprocessed: bool = True
    check_conflict_on_borderline: bool = True
    borderline_conf_multiplier: float = 1.5
    max_ocr_dimension: int = MAX_OCR_DIMENSION
    min_ocr_dimension: int = MIN_OCR_DIMENSION
    median_blur_ksize: int = MEDIAN_BLUR_KSIZE
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT
    tesseract_cmd: Optional[str] = None
    # Cấu hình Stripe Defect Detection (Phase 2 & Phase 4)
    enable_stripe_detection: bool = True
    stripe_confirm_threshold: float = STRIPE_CONFIRM_THRESHOLD
    stripe_review_threshold: float = STRIPE_REVIEW_THRESHOLD
    stripe_min_relative_length: float = STRIPE_MIN_RELATIVE_LENGTH
    stripe_max_thickness_px: int = STRIPE_MAX_THICKNESS_PX
    stripe_min_contrast: float = STRIPE_MIN_CONTRAST
    horizontal_min_page_ratio: float = HORIZONTAL_STRIPE_MIN_PAGE_RATIO
    horizontal_margin_extension: float = HORIZONTAL_STRIPE_MARGIN_EXTENSION
    # Cấu hình Page Boundary & Mixed-Page Detection (Phase 3)
    enable_boundary_detection: bool = True
    boundary_confirm_threshold: float = BOUNDARY_CONFIRM_THRESHOLD
    boundary_review_threshold: float = BOUNDARY_REVIEW_THRESHOLD
    boundary_overlap_threshold: float = BOUNDARY_OVERLAP_THRESHOLD
    boundary_use_optional_layout_model: bool = BOUNDARY_USE_OPTIONAL_LAYOUT_MODEL

    def __post_init__(self) -> None:
        if self.tesseract_cmd is None:
            self.tesseract_cmd = find_tesseract_binary()

    def to_dict(self) -> dict:
        """Chuyển cấu hình sang dict để ghi vào báo cáo JSON."""
        return asdict(self)
