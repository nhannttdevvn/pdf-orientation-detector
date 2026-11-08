"""
Render từng trang PDF thành mảng ảnh NumPy phục vụ OCR và xem trước giao diện.
Giải phóng bộ nhớ Pixmap và cache MuPDF ngay sau mỗi trang để xử lý tài liệu lớn.
"""

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24 / 2.0
except ImportError:
    import fitz  # type: ignore[no-redef]
import numpy as np

from ai_engine.core.config import PREVIEW_DPI, RENDER_DPI
from backend.repositories.file_utils import normalize_angle


def render_page_pixmap(
    page: fitz.Page,
    dpi: int = RENDER_DPI,
    grayscale: bool = True,
) -> np.ndarray:
    """
    Render một đối tượng `fitz.Page` thành mảng ảnh NumPy (grayscale 2D hoặc RGB 3D).
    Giải phóng Pixmap và bộ nhớ đệm MuPDF ngay sau khi copy sang NumPy array.
    """
    colorspace = fitz.csGRAY if grayscale else fitz.csRGB
    pix = page.get_pixmap(dpi=dpi, colorspace=colorspace, alpha=False)
    try:
        if grayscale or pix.n == 1:
            img = (
                np.frombuffer(pix.samples, dtype=np.uint8)
                .reshape(pix.height, pix.width)
                .copy()
            )
        else:
            img = (
                np.frombuffer(pix.samples, dtype=np.uint8)
                .reshape(pix.height, pix.width, pix.n)
                .copy()
            )
        return img
    finally:
        del pix
        if hasattr(fitz, "TOOLS") and hasattr(fitz.TOOLS, "store_shrink"):
            fitz.TOOLS.store_shrink(100)


def render_preview_image(
    page: fitz.Page,
    dpi: int = PREVIEW_DPI,
    extra_rotation: int = 0,
) -> np.ndarray:
    """
    Render ảnh xem trước màu (RGB) cho giao diện, hỗ trợ xoay thử góc `extra_rotation`
    (theo chiều kim đồng hồ) mà không sửa đổi tài liệu gốc.
    """
    img = render_page_pixmap(page=page, dpi=dpi, grayscale=False)
    rot = normalize_angle(extra_rotation)
    if rot == 90:
        img = np.rot90(img, k=-1)
    elif rot == 180:
        img = np.rot90(img, k=2)
    elif rot == 270:
        img = np.rot90(img, k=1)
    return img

