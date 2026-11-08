"""
Package `pdf_inspector.pipeline` — Điều phối toàn bộ luồng kiểm tra chất lượng tài liệu PDF (`DocumentPipeline`).
"""

from pdf_inspector.pipeline.document_pipeline import DocumentPipeline, PDFOrientationPipeline

__all__ = [
    "DocumentPipeline",
    "PDFOrientationPipeline",
]

