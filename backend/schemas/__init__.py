"""
Package `backend.schemas` chứa các data schemas và `PageContext` dùng chung cho toàn hệ thống.
"""

from backend.schemas.context import CoordinateTransform, PageContext
from backend.schemas.schemas import (
    CorrectionSummary,
    DefectResult,
    DefectStatus,
    DefectType,
    DetectorScope,
    DetectorStatus,
    InspectionStatus,
    OSDAttemptResult,
    OrientationDetectionResult,
    PDFDocumentInfo,
    PageDecision,
    PageQualityResult,
    PipelineResult,
    ProgressCallback,
)

__all__ = [
    "CoordinateTransform",
    "PageContext",
    "DetectorScope",
    "DetectorStatus",
    "DefectType",
    "DefectStatus",
    "InspectionStatus",
    "PDFDocumentInfo",
    "OSDAttemptResult",
    "OrientationDetectionResult",
    "DefectResult",
    "PageDecision",
    "PageQualityResult",
    "CorrectionSummary",
    "PipelineResult",
    "ProgressCallback",
]
