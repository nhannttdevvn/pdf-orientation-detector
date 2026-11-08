"""
Module tương thích ngược `pdf_inspector.detectors.tesseract_detector`,
chuyển tiếp sang `pdf_inspector.detectors.orientation.detector`.
"""

from pdf_inspector.detectors.orientation.detector import (
    OrientationDetector,
    TesseractOrientationDetector,
    parse_osd_text,
)
from pdf_inspector.models.schemas import OSDAttemptResult, OrientationDetectionResult

__all__ = [
    "TesseractOrientationDetector",
    "OrientationDetector",
    "parse_osd_text",
    "OSDAttemptResult",
    "OrientationDetectionResult",
]
