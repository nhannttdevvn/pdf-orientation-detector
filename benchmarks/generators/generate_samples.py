#!/usr/bin/env python3
"""
Script khởi tạo bộ 12 file PDF mẫu kiểm thử (có ground truth rõ ràng) trong thư mục `data/samples/`.
Có thể chạy độc lập:
    python scripts/generate_samples.py
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cv2
import numpy as np

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore[no-redef]

from ai_engine.core.config import SAMPLES_DIR

RICH_TEXT_CONTENT = (
    "COMMUNITY FINANCIAL & DOCUMENT PROCESSING SYSTEM - OFFICIAL REPORT\n\n"
    "1. Executive Summary and Technical Architecture\n"
    "This document serves as a standard multi-paragraph benchmark page for testing "
    "Optical Character Recognition (OCR) Orientation and Script Detection (OSD). "
    "Modern document digitization pipelines must accurately identify pages that were "
    "accidentally scanned upside down (180 degrees) or sideways (90 or 270 degrees).\n\n"
    "2. Key Operational Principles\n"
    "First, every page in a multi-page PDF document is rendered sequentially at 160 DPI "
    "to maintain low memory consumption even for large documents exceeding 500 pages. "
    "Second, OpenCV preprocessing applies gentle median filtering and CLAHE local contrast "
    "enhancement whenever initial detection confidence falls below the configured threshold.\n\n"
    "3. Safety Guarantees in Automated Correction\n"
    "Pages with insufficient text, blank pages, or ambiguous orientation confidence scores "
    "must be classified as REVIEW rather than NORMAL or ROTATED. Under no circumstances "
    "should a REVIEW page be automatically rotated without explicit human verification.\n\n"
    "4. Metadata Integrity and Non-Destructive Output\n"
    "When applying rotation corrections, the system updates the PDF page rotation attribute "
    "modulo 360 degrees without re-rasterizing vector or image streams, preserving original "
    "archival quality while writing to a separate output file."
)


def _render_upright_scan_image(dpi: int = 160) -> np.ndarray:
    """Tạo một ảnh trang văn bản chuẩn (0°) dưới dạng mảng grayscale uint8."""
    doc = fitz.open()
    try:
        page = doc.new_page(width=595, height=842)
        rect = fitz.Rect(45, 50, 550, 790)
        page.insert_textbox(
            rect,
            RICH_TEXT_CONTENT,
            fontsize=11.5,
            fontname="helv",
            align=fitz.TEXT_ALIGN_LEFT,
        )
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
        img = (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width)
            .copy()
        )
        return img
    finally:
        doc.close()


def _rotate_image_for_required_correction(
    img: np.ndarray, required_correction_cw: int
) -> np.ndarray:
    """
    Xoay ảnh gốc sao cho để đưa trang về đúng chiều (0°), hệ thống cần xoay thêm
    một góc `required_correction_cw` độ theo chiều kim đồng hồ.
    """
    norm = required_correction_cw % 360
    if norm == 0:
        return img.copy()
    if norm == 90:
        return np.rot90(img, k=1).copy()
    if norm == 180:
        return np.rot90(img, k=2).copy()
    if norm == 270:
        return np.rot90(img, k=-1).copy()
    raise ValueError(f"Góc không hợp lệ: {required_correction_cw}")


def _insert_scan_image_page(doc: fitz.Document, gray_img: np.ndarray) -> fitz.Page:
    """Nhúng một mảng ảnh grayscale thành một trang PDF scan."""
    h, w = gray_img.shape[:2]
    width_pt = w * 72.0 / 160.0
    height_pt = h * 72.0 / 160.0
    ok, png_buf = cv2.imencode(".png", gray_img)
    if not ok:
        raise RuntimeError("Không thể mã hóa PNG cho trang PDF mẫu.")
    page = doc.new_page(width=width_pt, height=height_pt)
    page.insert_image(fitz.Rect(0, 0, width_pt, height_pt), stream=png_buf.tobytes())
    return page


def generate_all_sample_pdfs(
    samples_dir: Path = SAMPLES_DIR,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Khởi tạo đủ 12 trường hợp PDF mẫu kiểm thử trong `data/samples/` kèm file `ground_truth.json`.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)
    gt_path = samples_dir / "ground_truth.json"
    if not force and gt_path.exists():
        try:
            existing_gt = json.loads(gt_path.read_text(encoding="utf-8"))
            if len(existing_gt) == 12 and all(
                (samples_dir / fname).exists() for fname in existing_gt
            ):
                return existing_gt
        except Exception:
            pass

    base_scan = _render_upright_scan_image(dpi=160)
    ground_truth: Dict[str, Any] = {}

    # 1. PDF tất cả trang đúng chiều (3 trang 0°)
    f1 = samples_dir / "01_all_normal.pdf"
    doc = fitz.open()
    for _ in range(3):
        _insert_scan_image_page(doc, base_scan)
    doc.save(str(f1))
    doc.close()
    ground_truth[f1.name] = {
        "description": "PDF tất cả trang đúng chiều",
        "valid_pdf": True,
        "page_count": 3,
        "pages": [
            {"page_number": 1, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 2, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 3, "expected_status": "NORMAL", "expected_correction_angle": 0},
        ],
    }

    # 2. PDF có một trang đảo ngược 180° (trang 2)
    f2 = samples_dir / "02_single_upside_down_180.pdf"
    doc = fitz.open()
    _insert_scan_image_page(doc, base_scan)
    _insert_scan_image_page(doc, _rotate_image_for_required_correction(base_scan, 180))
    _insert_scan_image_page(doc, base_scan)
    doc.save(str(f2))
    doc.close()
    ground_truth[f2.name] = {
        "description": "PDF có một trang đảo ngược 180°",
        "valid_pdf": True,
        "page_count": 3,
        "pages": [
            {"page_number": 1, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 2, "expected_status": "ROTATED", "expected_correction_angle": 180},
            {"page_number": 3, "expected_status": "NORMAL", "expected_correction_angle": 0},
        ],
    }

    # 3. PDF có nhiều trang đảo ngược (trang 1 và trang 3 đảo ngược 180°)
    f3 = samples_dir / "03_multiple_upside_down.pdf"
    doc = fitz.open()
    _insert_scan_image_page(doc, _rotate_image_for_required_correction(base_scan, 180))
    _insert_scan_image_page(doc, base_scan)
    _insert_scan_image_page(doc, _rotate_image_for_required_correction(base_scan, 180))
    _insert_scan_image_page(doc, base_scan)
    doc.save(str(f3))
    doc.close()
    ground_truth[f3.name] = {
        "description": "PDF có nhiều trang đảo ngược 180°",
        "valid_pdf": True,
        "page_count": 4,
        "pages": [
            {"page_number": 1, "expected_status": "ROTATED", "expected_correction_angle": 180},
            {"page_number": 2, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 3, "expected_status": "ROTATED", "expected_correction_angle": 180},
            {"page_number": 4, "expected_status": "NORMAL", "expected_correction_angle": 0},
        ],
    }

    # 4. PDF có trang cần xoay 90° (trang 2)
    f4 = samples_dir / "04_rotated_90.pdf"
    doc = fitz.open()
    _insert_scan_image_page(doc, base_scan)
    _insert_scan_image_page(doc, _rotate_image_for_required_correction(base_scan, 90))
    doc.save(str(f4))
    doc.close()
    ground_truth[f4.name] = {
        "description": "PDF có trang cần xoay 90°",
        "valid_pdf": True,
        "page_count": 2,
        "pages": [
            {"page_number": 1, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 2, "expected_status": "ROTATED", "expected_correction_angle": 90},
        ],
    }

    # 5. PDF có trang cần xoay 270° (trang 2)
    f5 = samples_dir / "05_rotated_270.pdf"
    doc = fitz.open()
    _insert_scan_image_page(doc, base_scan)
    _insert_scan_image_page(doc, _rotate_image_for_required_correction(base_scan, 270))
    doc.save(str(f5))
    doc.close()
    ground_truth[f5.name] = {
        "description": "PDF có trang cần xoay 270°",
        "valid_pdf": True,
        "page_count": 2,
        "pages": [
            {"page_number": 1, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 2, "expected_status": "ROTATED", "expected_correction_angle": 270},
        ],
    }

    # 6. PDF có cả 4 hướng (0°, 90°, 180°, 270°)
    f6 = samples_dir / "06_all_four_orientations.pdf"
    doc = fitz.open()
    for req_angle in (0, 90, 180, 270):
        _insert_scan_image_page(
            doc, _rotate_image_for_required_correction(base_scan, req_angle)
        )
    doc.save(str(f6))
    doc.close()
    ground_truth[f6.name] = {
        "description": "PDF có cả 4 hướng (0°, 90°, 180°, 270°)",
        "valid_pdf": True,
        "page_count": 4,
        "pages": [
            {"page_number": 1, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 2, "expected_status": "ROTATED", "expected_correction_angle": 90},
            {"page_number": 3, "expected_status": "ROTATED", "expected_correction_angle": 180},
            {"page_number": 4, "expected_status": "ROTATED", "expected_correction_angle": 270},
        ],
    }

    # 7. PDF có trang trắng (trang 1 bình thường, trang 2 trắng tinh)
    f7 = samples_dir / "07_blank_page.pdf"
    doc = fitz.open()
    _insert_scan_image_page(doc, base_scan)
    blank_img = np.full_like(base_scan, 255)
    _insert_scan_image_page(doc, blank_img)
    doc.save(str(f7))
    doc.close()
    ground_truth[f7.name] = {
        "description": "PDF có trang trắng",
        "valid_pdf": True,
        "page_count": 2,
        "pages": [
            {"page_number": 1, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 2, "expected_status": "REVIEW", "expected_correction_angle": 0},
        ],
    }

    # 8. PDF có rất ít chữ (trang 1 bình thường, trang 2 chỉ có "OK")
    f8 = samples_dir / "08_very_few_chars.pdf"
    doc = fitz.open()
    _insert_scan_image_page(doc, base_scan)
    page_sparse = doc.new_page(width=595, height=842)
    page_sparse.insert_text(fitz.Point(280, 420), "OK", fontsize=10)
    doc.save(str(f8))
    doc.close()
    ground_truth[f8.name] = {
        "description": "PDF có trang rất ít chữ",
        "valid_pdf": True,
        "page_count": 2,
        "pages": [
            {"page_number": 1, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 2, "expected_status": "REVIEW", "expected_correction_angle": 0},
        ],
    }

    # 9. PDF scan nhiễu hoặc mờ
    f9 = samples_dir / "09_noisy_blurry_scan.pdf"
    rng = np.random.default_rng(42)
    low_contrast = np.clip(
        base_scan.astype(np.float32) * 0.45 + 110.0, 0, 255
    ).astype(np.uint8)
    noise_mask = rng.random(low_contrast.shape)
    noisy_scan = low_contrast.copy()
    noisy_scan[noise_mask < 0.015] = 30
    noisy_scan[noise_mask > 0.985] = 245
    noisy_blurry = cv2.GaussianBlur(noisy_scan, (3, 3), 0.7)

    doc = fitz.open()
    _insert_scan_image_page(doc, noisy_blurry)
    _insert_scan_image_page(
        doc, _rotate_image_for_required_correction(noisy_blurry, 180)
    )
    doc.save(str(f9))
    doc.close()
    ground_truth[f9.name] = {
        "description": "PDF scan nhiễu hoặc mờ",
        "valid_pdf": True,
        "page_count": 2,
        "pages": [
            {"page_number": 1, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 2, "expected_status": "ROTATED", "expected_correction_angle": 180},
        ],
    }

    # 10. PDF không hợp lệ (file bị hỏng cấu trúc)
    f10 = samples_dir / "10_invalid_corrupted.pdf"
    f10.write_bytes(b"%PDF-1.7\nTHIS_IS_CORRUPTED_NOT_A_VALID_PDF_STREAM\n%%EOF")
    ground_truth[f10.name] = {
        "description": "PDF không hợp lệ / bị hỏng",
        "valid_pdf": False,
        "page_count": 0,
        "pages": [],
    }

    # 11. PDF có metadata rotation khác 0°
    f11 = samples_dir / "11_metadata_rotation_nonzero.pdf"
    doc = fitz.open()
    p1 = _insert_scan_image_page(doc, base_scan)
    p1.set_rotation(180)
    p2 = _insert_scan_image_page(doc, base_scan)
    p2.set_rotation(90)
    doc.save(str(f11))
    doc.close()
    ground_truth[f11.name] = {
        "description": "PDF có metadata rotation khác 0°",
        "valid_pdf": True,
        "page_count": 2,
        "pages": [
            {
                "page_number": 1,
                "existing_pdf_rotation": 180,
                "expected_status": "ROTATED",
                "expected_correction_angle": 180,
                "expected_final_pdf_rotation": 0,
            },
            {
                "page_number": 2,
                "existing_pdf_rotation": 90,
                "expected_status": "ROTATED",
                "expected_correction_angle": 270,
                "expected_final_pdf_rotation": 0,
            },
        ],
    }

    # 12. PDF có cả trang văn bản thông thường (vector text) và trang scan (bitmap 180°)
    f12 = samples_dir / "12_mixed_vector_and_scan.pdf"
    doc = fitz.open()
    vec_page = doc.new_page(width=595, height=842)
    vec_page.insert_textbox(
        fitz.Rect(45, 50, 550, 790),
        RICH_TEXT_CONTENT,
        fontsize=11.5,
        fontname="helv",
    )
    _insert_scan_image_page(doc, _rotate_image_for_required_correction(base_scan, 180))
    doc.save(str(f12))
    doc.close()
    ground_truth[f12.name] = {
        "description": "PDF có cả trang văn bản thông thường (vector) và trang scan (180°)",
        "valid_pdf": True,
        "page_count": 2,
        "pages": [
            {"page_number": 1, "expected_status": "NORMAL", "expected_correction_angle": 0},
            {"page_number": 2, "expected_status": "ROTATED", "expected_correction_angle": 180},
        ],
    }

    gt_path.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ground_truth


if __name__ == "__main__":
    gt = generate_all_sample_pdfs(SAMPLES_DIR, force=True)
    print(f"Đã tạo thành công {len(gt)} file PDF mẫu trong '{SAMPLES_DIR}'.")

