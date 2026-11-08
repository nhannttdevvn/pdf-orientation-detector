"""
Các dịch vụ điều phối nghiệp vụ (`pdf_inspector.services`).
"""

import pdf_inspector.detectors.boundary.boundary_detector  # noqa: F401
from pdf_inspector.services.correction_service import PDFCorrector, RotationCorrector
from pdf_inspector.services.orientation_service import DecisionEngine, UnifiedQualityEngine
from pdf_inspector.services.pdf_service import PDFLoader
from pdf_inspector.services.report_service import CSV_FIELDNAMES, ReportGenerator
from pdf_inspector.services.secret_scan_service import SecretDocumentResult, SecretScanService

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
        from pdf_inspector.pipeline.document_pipeline import DocumentPipeline

        return DocumentPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
