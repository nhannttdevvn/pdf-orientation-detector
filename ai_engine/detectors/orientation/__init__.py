"""
Package `ai_engine.detectors.orientation` — Bộ phát hiện lỗi hướng trang PDF (Orientation Detector).
"""

from ai_engine.detectors.orientation.detector import (
    OrientationDetector,
    TesseractOrientationDetector,
    parse_osd_text,
)

__all__ = [
    "TesseractOrientationDetector",
    "OrientationDetector",
    "parse_osd_text",
]

