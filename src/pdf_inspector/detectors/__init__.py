"""
Các bộ phát hiện chất lượng trang tài liệu (`pdf_inspector.detectors`).
"""

from pdf_inspector.detectors.base_detector import (
    BaseDetector,
    BaseOrientationDetector,
    DocumentLevelDetector,
    PageLevelDetector,
)
from pdf_inspector.detectors.defects import (
    StripeDetector,
    StripePageResult,
    render_stripe_overlay,
)
from pdf_inspector.detectors.orientation import (
    OrientationDetector,
    TesseractOrientationDetector,
    parse_osd_text,
)
from pdf_inspector.detectors.registry import (
    DetectorDescriptor,
    DetectorRegistry,
    create_default_registry,
)
from pdf_inspector.models.schemas import OSDAttemptResult, OrientationDetectionResult

__all__ = [
    "BaseDetector",
    "PageLevelDetector",
    "DocumentLevelDetector",
    "BaseOrientationDetector",
    "DetectorDescriptor",
    "DetectorRegistry",
    "create_default_registry",
    "TesseractOrientationDetector",
    "OrientationDetector",
    "StripeDetector",
    "StripePageResult",
    "render_stripe_overlay",
    "parse_osd_text",
    "OSDAttemptResult",
    "OrientationDetectionResult",
]
