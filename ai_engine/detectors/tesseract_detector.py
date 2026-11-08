"""
Module tương thích ngược `ai_engine.detectors.tesseract_detector`,
chuyển tiếp sang `ai_engine.detectors.orientation.detector`.
"""

from ai_engine.detectors.orientation.detector import (
    OrientationDetector,
    TesseractOrientationDetector,
    parse_osd_text,
)
from backend.schemas.schemas import OSDAttemptResult, OrientationDetectionResult

__all__ = [
    "TesseractOrientationDetector",
    "OrientationDetector",
    "parse_osd_text",
    "OSDAttemptResult",
    "OrientationDetectionResult",
]
