"""
Dịch vụ quản lý và đọc tài liệu PDF (`backend.services.pdf_service`).
Tái xuất `PDFLoader` từ `ai_engine.preprocessing.pdf_loader` để bảo toàn tương thích ngược.
"""

from ai_engine.core.exceptions import (
    InvalidPDFError,
    PDFLoaderError,
    PasswordProtectedPDFError,
)
from backend.schemas.context import PageContext
from backend.schemas.schemas import PDFDocumentInfo
from ai_engine.preprocessing.pdf_loader import PDFLoader

__all__ = [
    "PDFLoader",
    "PDFDocumentInfo",
    "PageContext",
    "PDFLoaderError",
    "InvalidPDFError",
    "PasswordProtectedPDFError",
]
