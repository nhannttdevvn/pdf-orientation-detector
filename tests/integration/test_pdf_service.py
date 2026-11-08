"""
Kiểm thử đơn vị và tích hợp cho `backend.services.pdf_service` và `ai_engine.preprocessing.page_renderer`.
"""

from pathlib import Path

import pytest

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore[no-redef]

from ai_engine.core.config import SAMPLES_DIR
from ai_engine.core.exceptions import InvalidPDFError, PasswordProtectedPDFError
from backend.services.pdf_service import PDFLoader


def test_pdf_loader_metadata_and_sequential_iteration() -> None:
    """Kiểm tra mở file PDF hợp lệ, đọc metadata, render trang và duyệt tuần tự."""
    sample_pdf = SAMPLES_DIR / "11_metadata_rotation_nonzero.pdf"
    with PDFLoader(sample_pdf) as loader:
        info = loader.get_document_info()
        assert info.page_count == 2
        assert loader.get_page_rotation(1) == 180
        assert loader.get_page_rotation(2) == 90

        pages = list(loader.iter_pages(dpi=120, grayscale=True))
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[0].image.ndim == 2
        assert pages[1].page_number == 2

        preview_rgb = loader.render_preview(page_number=1, dpi=96, extra_rotation=90)
        assert preview_rgb.ndim == 3


def test_invalid_and_corrupted_pdf_handling(tmp_path: Path) -> None:
    """Case 10: Kiểm tra phát hiện file PDF bị hỏng, rỗng hoặc không tồn tại."""
    corrupted_pdf = SAMPLES_DIR / "10_invalid_corrupted.pdf"
    with pytest.raises(InvalidPDFError):
        with PDFLoader(corrupted_pdf):
            pass

    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(InvalidPDFError):
        with PDFLoader(empty_pdf):
            pass

    missing_pdf = tmp_path / "does_not_exist.pdf"
    with pytest.raises(InvalidPDFError):
        with PDFLoader(missing_pdf):
            pass


def test_password_protected_pdf_handling(tmp_path: Path) -> None:
    """Kiểm tra phát hiện file PDF bị khóa bằng mật khẩu."""
    locked_pdf = tmp_path / "locked.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 72), "Confidential content")
    doc.save(
        str(locked_pdf),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="secret123",
        owner_pw="owner123",
    )
    doc.close()

    with pytest.raises(PasswordProtectedPDFError):
        with PDFLoader(locked_pdf):
            pass

