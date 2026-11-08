"""
Module đọc và kiểm tra tài liệu PDF (`src/pdf_inspector/processing/pdf_loader.py`).
Sử dụng PyMuPDF (fitz) để mở, kiểm tra tính hợp lệ, đọc metadata, góc xoay hiện tại,
kiểm tra chữ ký số, và khởi tạo `PageContext` tuần tự từng trang PDF để tiết kiệm RAM.
"""

from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24 / 2.0
except ImportError:
    import fitz  # type: ignore[no-redef]
import numpy as np

from ai_engine.core.config import PREVIEW_DPI, RENDER_DPI
from ai_engine.core.exceptions import (
    InvalidPDFError,
    PDFLoaderError,
    PasswordProtectedPDFError,
)
from ai_engine.core.logging_config import setup_logger
from backend.schemas.context import PageContext
from backend.schemas.schemas import PDFDocumentInfo
from ai_engine.preprocessing.page_renderer import render_page_pixmap, render_preview_image
from backend.repositories.file_utils import normalize_angle

logger = setup_logger(__name__)


class PDFLoader:
    """
    Quản lý vòng đời tài liệu PDF an toàn bằng Context Manager.
    Hỗ trợ đọc tuần tự từng trang và khởi tạo `PageContext` để xử lý tài liệu lớn
    (100 - 1.000+ trang) mà không lưu toàn bộ ảnh độ phân giải cao trong RAM.
    """

    def __init__(self, pdf_path: Union[str, Path]) -> None:
        self.pdf_path: Path = Path(pdf_path).expanduser().resolve()
        self._doc: Optional[fitz.Document] = None
        self._doc_info: Optional[PDFDocumentInfo] = None

    def __enter__(self) -> "PDFLoader":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def open(self) -> PDFDocumentInfo:
        """
        Mở và kiểm tra tính hợp lệ của file PDF.
        Ném InvalidPDFError hoặc PasswordProtectedPDFError nếu file không hợp lệ.
        """
        if self._doc is not None:
            return self.get_document_info()

        if not self.pdf_path.exists() or not self.pdf_path.is_file():
            raise InvalidPDFError(f"File PDF không tồn tại: {self.pdf_path}")

        file_size = self.pdf_path.stat().st_size
        if file_size == 0:
            raise InvalidPDFError(f"File PDF rỗng (0 bytes): {self.pdf_path.name}")

        try:
            doc = fitz.open(str(self.pdf_path))
        except Exception as exc:
            logger.error("Không thể mở file PDF '%s': %s", self.pdf_path.name, exc)
            raise InvalidPDFError(
                f"File không hợp lệ hoặc bị hỏng ('{self.pdf_path.name}'): {exc}"
            ) from exc

        try:
            if not getattr(doc, "is_pdf", True):
                raise InvalidPDFError(
                    f"File '{self.pdf_path.name}' không phải là tài liệu PDF hợp lệ."
                )

            if doc.needs_pass:
                raise PasswordProtectedPDFError(
                    f"File PDF '{self.pdf_path.name}' được bảo vệ bằng mật khẩu. "
                    "Vui lòng mở khóa trước khi xử lý."
                )

            page_count = doc.page_count
            if page_count <= 0:
                raise InvalidPDFError(
                    f"File PDF '{self.pdf_path.name}' không có trang nào."
                )

            _ = doc.load_page(0)

            self._doc = doc
            self._doc_info = self._extract_document_info(file_size)
            logger.info(
                "Đã mở PDF '%s' (%d trang, %d bytes, chữ ký số=%s)",
                self.pdf_path.name,
                page_count,
                file_size,
                self._doc_info.has_digital_signature,
            )
            return self._doc_info
        except PDFLoaderError:
            doc.close()
            raise
        except Exception as exc:
            doc.close()
            logger.error("Lỗi khi kiểm tra cấu trúc PDF '%s': %s", self.pdf_path.name, exc)
            raise InvalidPDFError(
                f"Cấu trúc PDF bị lỗi ('{self.pdf_path.name}'): {exc}"
            ) from exc

    def _extract_document_info(self, file_size: int) -> PDFDocumentInfo:
        """Trích xuất metadata, góc xoay từng trang và kiểm tra chữ ký số."""
        assert self._doc is not None
        raw_meta = self._doc.metadata or {}
        clean_meta = {k: v for k, v in raw_meta.items() if v is not None}

        page_rotations: Dict[int, int] = {}
        has_sig = self._detect_digital_signature()

        for idx in range(self._doc.page_count):
            page = self._doc.load_page(idx)
            try:
                rot = normalize_angle(page.rotation)
            except ValueError:
                rot = 0
            page_rotations[idx + 1] = rot

        return PDFDocumentInfo(
            file_path=str(self.pdf_path),
            file_name=self.pdf_path.name,
            file_size_bytes=file_size,
            page_count=self._doc.page_count,
            metadata=clean_meta,
            page_rotations=page_rotations,
            has_digital_signature=has_sig,
            is_encrypted=bool(self._doc.is_encrypted),
        )

    def _detect_digital_signature(self) -> bool:
        """
        Phát hiện xem PDF có chứa trường chữ ký số (Digital Signature) hay không.
        """
        assert self._doc is not None
        try:
            sig_widget_type = getattr(fitz, "PDF_WIDGET_TYPE_SIGNATURE", 6)
            for idx in range(self._doc.page_count):
                page = self._doc.load_page(idx)
                for widget in page.widgets() or []:
                    if (
                        getattr(widget, "field_type", None) == sig_widget_type
                        or getattr(widget, "field_type_string", "").lower() == "signature"
                    ):
                        return True

            catalog_xref = self._doc.pdf_catalog()
            if catalog_xref > 0:
                acroform = self._doc.xref_get_key(catalog_xref, "AcroForm")
                if acroform and "SigFlags" in str(acroform):
                    return True
        except Exception as exc:
            logger.debug("Không thể kiểm tra chữ ký số chi tiết: %s", exc)
        return False

    def get_document_info(self) -> PDFDocumentInfo:
        """Trả về thông tin tài liệu PDF đã mở."""
        if self._doc is None or self._doc_info is None:
            return self.open()
        return self._doc_info

    @property
    def page_count(self) -> int:
        """Tổng số trang của PDF."""
        return self.get_document_info().page_count

    def get_page_rotation(self, page_number: int) -> int:
        """
        Trả về góc xoay hiện tại trong metadata PDF của trang `page_number` (1-based).
        """
        info = self.get_document_info()
        if page_number < 1 or page_number > info.page_count:
            raise IndexError(
                f"Số trang {page_number} nằm ngoài phạm vi (1..{info.page_count})."
            )
        assert self._doc is not None
        page = self._doc.load_page(page_number - 1)
        try:
            return normalize_angle(page.rotation)
        except ValueError:
            return 0

    def render_page(
        self,
        page_number: int,
        dpi: int = RENDER_DPI,
        grayscale: bool = True,
    ) -> np.ndarray:
        """
        Render một trang cụ thể (1-based) thành mảng ảnh NumPy.
        Giải phóng Pixmap ngay sau khi chuyển sang NumPy array để tránh rò rỉ RAM.
        """
        info = self.get_document_info()
        if page_number < 1 or page_number > info.page_count:
            raise IndexError(
                f"Số trang {page_number} nằm ngoài phạm vi (1..{info.page_count})."
            )
        assert self._doc is not None
        page = self._doc.load_page(page_number - 1)
        return render_page_pixmap(page=page, dpi=dpi, grayscale=grayscale)

    def build_page_context(
        self,
        page_number: int,
        dpi: int = RENDER_DPI,
        grayscale: bool = True,
        max_ocr_dimension: Optional[int] = None,
        min_ocr_dimension: Optional[int] = None,
        median_blur_ksize: Optional[int] = None,
    ) -> PageContext:
        """
        Render trang `page_number` và khởi tạo đối tượng `PageContext` dùng chung cho các Detector.
        """
        info = self.get_document_info()
        if page_number < 1 or page_number > info.page_count:
            raise IndexError(
                f"Số trang {page_number} nằm ngoài phạm vi (1..{info.page_count})."
            )
        assert self._doc is not None
        idx = page_number - 1
        page = self._doc.load_page(idx)
        try:
            existing_rot = normalize_angle(page.rotation)
        except ValueError:
            existing_rot = 0

        rect = page.rect
        width_pt = float(rect.width)
        height_pt = float(rect.height)
        image = render_page_pixmap(page=page, dpi=dpi, grayscale=grayscale)

        kwargs: Dict[str, Any] = {}
        if max_ocr_dimension is not None:
            kwargs["max_ocr_dimension"] = max_ocr_dimension
        if min_ocr_dimension is not None:
            kwargs["min_ocr_dimension"] = min_ocr_dimension
        if median_blur_ksize is not None:
            kwargs["median_blur_ksize"] = median_blur_ksize

        return PageContext(
            document_id=info.file_name,
            page_number=page_number,
            page_index=idx,
            pdf_page_size=(width_pt, height_pt),
            original_rotation=existing_rot,
            original_image=image,
            dpi=dpi,
            metadata=dict(info.metadata),
            **kwargs,
        )

    def render_preview(
        self,
        page_number: int,
        dpi: int = PREVIEW_DPI,
        extra_rotation: int = 0,
    ) -> np.ndarray:
        """
        Render ảnh xem trước (RGB) cho giao diện Streamlit, hỗ trợ xoay thử góc `extra_rotation`
        mà không sửa file gốc.
        """
        info = self.get_document_info()
        if page_number < 1 or page_number > info.page_count:
            raise IndexError(
                f"Số trang {page_number} nằm ngoài phạm vi (1..{info.page_count})."
            )
        assert self._doc is not None
        page = self._doc.load_page(page_number - 1)
        return render_preview_image(page=page, dpi=dpi, extra_rotation=extra_rotation)

    def iter_pages(
        self,
        dpi: int = RENDER_DPI,
        grayscale: bool = True,
        auto_release: bool = False,
    ) -> Generator[PageContext, None, None]:
        """
        Generator duyệt tuần tự từng trang PDF từ 1 đến N.
        Nếu `auto_release=True`, tự động giải phóng ảnh trong `PageContext` ngay sau mỗi vòng lặp.
        """
        info = self.get_document_info()
        for idx in range(info.page_count):
            page_num = idx + 1
            ctx = self.build_page_context(
                page_number=page_num, dpi=dpi, grayscale=grayscale
            )
            try:
                yield ctx
            finally:
                if auto_release:
                    ctx.release()

    def close(self) -> None:
        """Đóng tài liệu PDF và giải phóng tài nguyên hệ thống."""
        if self._doc is not None:
            try:
                self._doc.close()
                logger.debug("Đã đóng file PDF '%s'", self.pdf_path.name)
            except Exception as exc:
                logger.warning("Lỗi khi đóng PDF '%s': %s", self.pdf_path.name, exc)
            finally:
                self._doc = None

