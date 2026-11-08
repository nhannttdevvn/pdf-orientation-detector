"""
Module đọc PDF, render trang và xử lý ảnh (`pdf_inspector.processing`).
"""

from pdf_inspector.processing.image_preprocessor import (
    denoise_image,
    enhance_contrast_clahe,
    extract_text_line_canvas,
    normalize_dimensions,
    prepare_original_grayscale,
    preprocess_for_ocr,
    to_grayscale,
)
from pdf_inspector.processing.page_renderer import render_page_pixmap, render_preview_image
from pdf_inspector.processing.pdf_loader import PDFLoader

__all__ = [
    "PDFLoader",
    "to_grayscale",
    "normalize_dimensions",
    "denoise_image",
    "enhance_contrast_clahe",
    "prepare_original_grayscale",
    "preprocess_for_ocr",
    "extract_text_line_canvas",
    "render_page_pixmap",
    "render_preview_image",
]
