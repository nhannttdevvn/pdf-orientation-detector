"""
Các bộ phát hiện chất lượng trang tài liệu (`ai_engine.detectors`).
"""

from ai_engine.detectors.base_detector import (
    BaseDetector,
    BaseOrientationDetector,
    DocumentLevelDetector,
    PageLevelDetector,
)
from ai_engine.detectors.defects import (
    StripeDetector,
    StripePageResult,
    render_stripe_overlay,
)
from ai_engine.detectors.orientation import (
    OrientationDetector,
    TesseractOrientationDetector,
    parse_osd_text,
)
from ai_engine.detectors.registry import (
    DetectorDescriptor,
    DetectorRegistry,
    create_default_registry,
)
from backend.schemas.schemas import OSDAttemptResult, OrientationDetectionResult

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
