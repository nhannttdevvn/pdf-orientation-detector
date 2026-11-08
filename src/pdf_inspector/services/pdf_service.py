"""
Dịch vụ quản lý và đọc tài liệu PDF (`pdf_inspector.services.pdf_service`).
Tái xuất `PDFLoader` từ `pdf_inspector.processing.pdf_loader` để bảo toàn tương thích ngược.
"""

from pdf_inspector.core.exceptions import (
    InvalidPDFError,
    PDFLoaderError,
    PasswordProtectedPDFError,
)
from pdf_inspector.models.context import PageContext
from pdf_inspector.models.schemas import PDFDocumentInfo
from pdf_inspector.processing.pdf_loader import PDFLoader

__all__ = [
    "PDFLoader",
    "PDFDocumentInfo",
    "PageContext",
    "PDFLoaderError",
    "InvalidPDFError",
    "PasswordProtectedPDFError",
]
