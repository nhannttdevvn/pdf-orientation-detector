"""
Các dịch vụ điều phối nghiệp vụ (`backend.services`).
"""

import ai_engine.detectors.boundary.boundary_detector  # noqa: F401
from backend.services.correction_service import PDFCorrector, RotationCorrector
from backend.services.orientation_service import DecisionEngine, UnifiedQualityEngine
from backend.services.pdf_service import PDFLoader
from backend.services.report_service import CSV_FIELDNAMES, ReportGenerator
from backend.services.secret_scan_service import SecretDocumentResult, SecretScanService

__all__ = [
    "PDFLoader",
    "PDFCorrector",
    "RotationCorrector",
    "DecisionEngine",
    "UnifiedQualityEngine",
    "DocumentPipeline",
    "PDFOrientationPipeline",
    "ReportGenerator",
    "CSV_FIELDNAMES",
    "SecretScanService",
    "SecretDocumentResult",
]


def __getattr__(name: str):
    if name in ("DocumentPipeline", "PDFOrientationPipeline"):
        from ai_engine.pipelines.document_pipeline import DocumentPipeline

        return DocumentPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
