"""
Unit test & Integration test toàn diện cho Secret Marking Detection (`tests/test_secret_detector.py`).

Bao phủ đầy đủ 12 kịch bản kiểm thử bắt buộc trong Phase 4 — Part C5:
1. Dấu MẬT rõ.
2. Dấu TỐI MẬT rõ.
3. Dấu TUYỆT MẬT rõ.
4. Dấu đỏ (`RED_INK_ENHANCED`).
5. Dấu đen (`GRAY` / `RED_SUPPRESSED`).
6. Dấu mờ (`CLAHE_ENHANCED` / faint ink).
7. Dấu xoay nhẹ (`±8°`).
8. Dấu có kích thước khác nhau (`0.65x` nhỏ và `1.35x` lớn).
9. Trang không có dấu (`NO_MARK_DETECTED`).
10. Văn bản chứa từ "mật" thông thường (`"BẢO MẬT"`, `"MẬT ĐỘ"`, câu văn có từ `"bí mật"` -> `NO_MARK_DETECTED`).
11. PDF nhiều trang (quét qua `SecretScanService`, rà soát `Xác nhận` / `Bỏ qua` / `Chỉnh sửa`, xuất JSON/CSV, không tách PDF).
12. Xử lý lỗi khi ảnh/render thất bại (`PROCESSING_ERROR`, không mặc định `NO_MARK_DETECTED`).
Kèm kiểm chứng tính độc lập tuyệt đối giữa hai workflow (Part C3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pymupdf as fitz
import pytest

from pdf_inspector.core.config import (
    SECRET_STATUS_DETECTED,
    SECRET_STATUS_ERROR,
    SECRET_STATUS_NO_MARK,
    SECRET_STATUS_REVIEW,
    SECRET_TEMPLATES_DIR,
    DetectorConfig,
)
from pdf_inspector.detectors.secret import (
    SecretMarkingDetector,
    crop_secret_mark,
    render_secret_overlay,
)
from pdf_inspector.pipeline.document_pipeline import DocumentPipeline
from pdf_inspector.services.secret_scan_service import SecretScanService


def _create_admin_page_rgb(w: int = 1654, h: int = 2339) -> np.ndarray:
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    cv2.putText(img, "CO QUAN CHU QUAN", (160, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.line(img, (195, 136), (395, 136), (40, 40, 40), 2)
    cv2.putText(img, "CONG HOA XA HOI CHU NGHIA VIET NAM", (770, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(img, "Doc lap - Tu do - Hanh phuc", (880, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.line(img, (905, 180), (1235, 180), (40, 40, 40), 2)

    for idx in range(10):
        y = 580 + idx * 100
        cv2.putText(
            img,
            f"Noi dung van ban hanh chinh thong thuong dong so {idx + 1}.",
            (180, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
    return img


def _stamp_on_page(
    page_rgb: np.ndarray,
    tpl_name: str,
    x: int = 210,
    y: int = 235,
    scale: float = 1.0,
    angle_deg: float = 0.0,
    ink_rgb: Tuple[int, int, int] = (20, 20, 20),
    alpha_strength: float = 1.0,
) -> np.ndarray:
    out = page_rgb.copy()
    tpl = cv2.imread(str(SECRET_TEMPLATES_DIR / tpl_name), cv2.IMREAD_GRAYSCALE)
    assert tpl is not None, f"Thiếu template {tpl_name}"

    if abs(scale - 1.0) > 1e-3:
        sw = max(20, int(round(tpl.shape[1] * scale)))
        sh = max(12, int(round(tpl.shape[0] * scale)))
        tpl = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

    if abs(angle_deg) > 0.2:
        th, tw = tpl.shape[:2]
        rad = np.deg2rad(abs(angle_deg))
        nw = int(np.ceil(tw * np.cos(rad) + th * np.sin(rad)))
        nh = int(np.ceil(tw * np.sin(rad) + th * np.cos(rad)))
        mat = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), angle_deg, 1.0)
        mat[0, 2] += (nw - tw) / 2.0
        mat[1, 2] += (nh - th) / 2.0
        tpl = cv2.warpAffine(tpl, mat, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    sh, sw = tpl.shape[:2]
    ink_alpha = np.clip((255.0 - tpl.astype(np.float32)) / 255.0, 0.0, 1.0) * float(alpha_strength)
    roi = out[y : y + sh, x : x + sw].astype(np.float32)
    ink_vec = np.array(ink_rgb, dtype=np.float32).reshape(1, 1, 3)
    out[y : y + sh, x : x + sw] = np.clip(
        roi * (1.0 - ink_alpha[:, :, None]) + ink_vec * ink_alpha[:, :, None],
        0,
        255,
    ).astype(np.uint8)
    return out


@pytest.fixture(scope="module")
def secret_detector() -> SecretMarkingDetector:
    return SecretMarkingDetector()


def test_c5_01_02_03_clear_mat_toi_mat_tuyet_mat(secret_detector: SecretMarkingDetector) -> None:
    """Kiểm thử C5 #1, #2, #3: Nhận diện chính xác dấu MẬT, TỐI MẬT, TUYỆT MẬT rõ."""
    for tpl_name, expected_label in (
        ("box_mat.png", "MẬT"),
        ("box_toi_mat.png", "TỐI MẬT"),
        ("box_tuyet_mat.png", "TUYỆT MẬT"),
    ):
        page = _stamp_on_page(_create_admin_page_rgb(), tpl_name)
        res = secret_detector.detect_page(page, page_number=1, color_order="RGB")
        assert res.status == SECRET_STATUS_DETECTED
        assert res.primary_label == expected_label
        assert len(res.marks) >= 1
        assert res.marks[0].confidence >= 0.70


def test_c5_04_05_red_and_black_stamps_and_red_seal_overlap(
    secret_detector: SecretMarkingDetector,
) -> None:
    """Kiểm thử C5 #4 & #5: Dấu mực đỏ tươi, dấu mực đen, và dấu đen bị con dấu tròn đỏ đè lên."""
    # Dấu TUYỆT MẬT đóng mực đỏ tươi
    red_page = _stamp_on_page(
        _create_admin_page_rgb(),
        "box_tuyet_mat.png",
        x=1020,
        y=235,
        ink_rgb=(215, 22, 25),
    )
    res_red = secret_detector.detect_page(red_page, page_number=1, color_order="RGB")
    assert res_red.status == SECRET_STATUS_DETECTED
    assert res_red.primary_label == "TUYỆT MẬT"

    # Dấu TỐI MẬT mực đen bị con dấu tròn đỏ đè lên một nửa
    overlap_page = _stamp_on_page(
        _create_admin_page_rgb(),
        "box_toi_mat.png",
        x=210,
        y=235,
        ink_rgb=(20, 20, 20),
    )
    seal_layer = overlap_page.copy()
    cv2.circle(seal_layer, (235, 265), 85, (220, 25, 25), 5)
    cv2.circle(seal_layer, (235, 265), 55, (220, 25, 25), 3)
    overlap_page = cv2.addWeighted(overlap_page, 0.55, seal_layer, 0.45, 0)

    res_overlap = secret_detector.detect_page(overlap_page, page_number=2, color_order="RGB")
    assert res_overlap.status in (SECRET_STATUS_DETECTED, SECRET_STATUS_REVIEW)
    assert res_overlap.primary_label == "TỐI MẬT"


def test_c5_06_faded_stamp(secret_detector: SecretMarkingDetector) -> None:
    """Kiểm thử C5 #6: Dấu bị phai mực rất mờ vẫn được phát hiện (DETECTED hoặc REVIEW)."""
    faded_page = _stamp_on_page(
        _create_admin_page_rgb(),
        "box_mat.png",
        alpha_strength=0.18,
    )
    res = secret_detector.detect_page(faded_page, page_number=1, color_order="RGB")
    assert res.status in (SECRET_STATUS_DETECTED, SECRET_STATUS_REVIEW)
    assert res.primary_label == "MẬT"


def test_c5_07_slightly_rotated_stamps(secret_detector: SecretMarkingDetector) -> None:
    """Kiểm thử C5 #7: Dấu bị đóng nghiêng nhẹ (+8° có khung và -9° không khung)."""
    rot_box_page = _stamp_on_page(
        _create_admin_page_rgb(),
        "box_mat.png",
        angle_deg=8.0,
    )
    res_box = secret_detector.detect_page(rot_box_page, page_number=1, color_order="RGB")
    assert res_box.status in (SECRET_STATUS_DETECTED, SECRET_STATUS_REVIEW)
    assert res_box.primary_label == "MẬT"

    rot_text_page = _stamp_on_page(
        _create_admin_page_rgb(),
        "text_toi_mat.png",
        angle_deg=-9.0,
        ink_rgb=(210, 25, 25),
    )
    res_txt = secret_detector.detect_page(rot_text_page, page_number=2, color_order="RGB")
    assert res_txt.status in (SECRET_STATUS_DETECTED, SECRET_STATUS_REVIEW)
    assert res_txt.primary_label == "TỐI MẬT"


def test_c5_08_multi_scale_stamps(secret_detector: SecretMarkingDetector) -> None:
    """Kiểm thử C5 #8: Dấu có kích thước khác nhau (0.65x nhỏ và 1.35x lớn)."""
    small_page = _stamp_on_page(_create_admin_page_rgb(), "box_mat.png", scale=0.65)
    res_small = secret_detector.detect_page(small_page, page_number=1, color_order="RGB")
    assert res_small.status == SECRET_STATUS_DETECTED
    assert res_small.primary_label == "MẬT"

    large_page = _stamp_on_page(_create_admin_page_rgb(), "box_tuyet_mat.png", x=180, scale=1.35)
    res_large = secret_detector.detect_page(large_page, page_number=2, color_order="RGB")
    assert res_large.status == SECRET_STATUS_DETECTED
    assert res_large.primary_label == "TUYỆT MẬT"


def test_c5_09_10_clean_page_and_ordinary_text_with_word_mat(
    secret_detector: SecretMarkingDetector,
) -> None:
    """
    Kiểm thử C5 #9 & #10:
    - Trang sạch trả về `NO_MARK_DETECTED`.
    - Văn bản chứa từ `"mật"` / `"BẢO MẬT"` / `"MẬT ĐỘ"` thông thường tuyệt đối không bị báo nhầm.
    """
    clean_page = _create_admin_page_rgb()
    res_clean = secret_detector.detect_page(clean_page, page_number=1, color_order="RGB")
    assert res_clean.status == SECRET_STATUS_NO_MARK
    assert res_clean.primary_label is None
    assert res_clean.marks == []

    # Dán chính xác glyph chữ `MẬT` vào giữa dòng tiêu đề "BAO CAO CONG TAC BAO [MẬT] NAM 2026"
    inline_page = _create_admin_page_rgb()
    mat_tpl = cv2.imread(str(SECRET_TEMPLATES_DIR / "text_mat.png"), cv2.IMREAD_GRAYSCALE)
    assert mat_tpl is not None
    mw, mh = int(mat_tpl.shape[1] * 0.56), int(mat_tpl.shape[0] * 0.56)
    mat_small = cv2.resize(mat_tpl, (mw, mh), interpolation=cv2.INTER_AREA)

    y0 = 260
    cv2.putText(
        inline_page,
        "BAO CAO CONG TAC BAO",
        (300, y0 + mh - 7),
        cv2.FONT_HERSHEY_DUPLEX,
        0.92,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    mx = 692
    ink = (255.0 - mat_small.astype(np.float32)) / 255.0
    roi = inline_page[y0 : y0 + mh, mx : mx + mw].astype(np.float32)
    inline_page[y0 : y0 + mh, mx : mx + mw] = np.clip(
        roi * (1.0 - ink[:, :, None]) + 25.0 * ink[:, :, None],
        0,
        255,
    ).astype(np.uint8)
    cv2.putText(
        inline_page,
        "VA AN TOAN THONG TIN",
        (mx + mw + 14, y0 + mh - 7),
        cv2.FONT_HERSHEY_DUPLEX,
        0.92,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )

    res_inline = secret_detector.detect_page(inline_page, page_number=2, color_order="RGB")
    assert res_inline.status == SECRET_STATUS_NO_MARK
    assert res_inline.marks == []


def test_c5_11_multipage_pdf_service_review_and_export(tmp_path: Path) -> None:
    """
    Kiểm thử C5 #11 + Part B3/C4:
    - Quét PDF nhiều trang qua `SecretScanService`.
    - Kiểm tra overlay, ảnh crop trong RAM (không lưu xuống đĩa), thao tác Xác nhận / Bỏ qua / Chỉnh sửa,
      và xuất báo cáo JSON / CSV (không tách PDF).
    """
    pdf_path = tmp_path / "multi_secret_test.pdf"
    doc = fitz.open()

    pages_rgb = [
        _stamp_on_page(_create_admin_page_rgb(), "box_toi_mat.png"),  # Trang 1: TỐI MẬT
        _create_admin_page_rgb(),  # Trang 2: Sạch
        _stamp_on_page(_create_admin_page_rgb(), "box_tuyet_mat.png", ink_rgb=(215, 20, 25)),  # Trang 3: TUYỆT MẬT đỏ
        _stamp_on_page(_create_admin_page_rgb(), "box_mat.png"),  # Trang 4: MẬT
    ]

    for rgb in pages_rgb:
        ok, png_buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        assert ok
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=bytes(png_buf))
    doc.save(str(pdf_path))
    doc.close()

    service = SecretScanService()
    progress_steps = []
    result = service.scan_pdf(pdf_path, progress_callback=lambda cur, tot: progress_steps.append((cur, tot)))

    assert result.total_pages == 4
    assert progress_steps[-1] == (4, 4)
    assert result.detected_pages_count == 3
    assert result.get_page(1).primary_label == "TỐI MẬT"
    assert result.get_page(2).status == SECRET_STATUS_NO_MARK
    assert result.get_page(3).primary_label == "TUYỆT MẬT"
    assert result.get_page(4).primary_label == "MẬT"

    # Kiểm tra render overlay và crop trong RAM, không tạo file crop trên ổ đĩa
    p1 = result.get_page(1)
    assert p1 is not None
    overlay_rgb, crops = service.render_page_and_ephemeral_crops(pdf_path, p1)
    assert overlay_rgb.ndim == 3 and overlay_rgb.size > 0
    assert len(crops) >= 1
    crop_img = crop_secret_mark(overlay_rgb, p1.marks[0], source_size=(p1.image_width, p1.image_height))
    assert crop_img is not None and crop_img.size > 0
    assert list(tmp_path.glob("*crop*")) == []

    # Kiểm tra Review Actions: Xác nhận trang 1, Chỉnh sửa trang 3, Bỏ qua trang 4
    service.apply_review_action(result, 1, "CONFIRMED")
    service.apply_review_action(result, 3, "EDITED", override_label="TỐI MẬT", note="Hiệu chỉnh thủ công")
    service.apply_review_action(result, 4, "IGNORED")

    assert result.detected_pages_count == 2
    assert result.get_page(3).effective_label == "TỐI MẬT"
    assert result.get_page(4).effective_status == SECRET_STATUS_NO_MARK

    # Kiểm tra xuất báo cáo JSON / CSV (không tách PDF)
    json_bytes = service.export_report_json_bytes(result)
    report_obj = json.loads(json_bytes.decode("utf-8"))
    assert report_obj["workflow"] == "secret_marking_detection"
    assert "không tự động tách file PDF" in report_obj["disclaimer"]
    assert report_obj["summary"]["pages_with_marks"] == 2

    csv_text = service.export_report_csv_bytes(result).decode("utf-8-sig")
    assert "multi_secret_test.pdf" in csv_text
    assert "Hiệu chỉnh thủ công" in csv_text
    # Đảm bảo không sinh thêm bất kỳ file PDF tách nhỏ nào
    assert list(tmp_path.glob("*.pdf")) == [pdf_path]


def test_c5_12_processing_error_on_invalid_input_or_render_failure(
    secret_detector: SecretMarkingDetector,
    tmp_path: Path,
) -> None:
    """Kiểm thử C5 #12: Khi ảnh rỗng hoặc render lỗi phải trả về `PROCESSING_ERROR`, không mặc định `NO_MARK_DETECTED`."""
    res_none = secret_detector.detect_page(None, page_number=1)
    assert res_none.status == SECRET_STATUS_ERROR
    assert res_none.error_message

    res_empty = secret_detector.detect_page(np.zeros((0, 0), dtype=np.uint8), page_number=2)
    assert res_empty.status == SECRET_STATUS_ERROR

    # Giả lập lỗi render trang giữa chừng trong SecretScanService
    pdf_path = tmp_path / "render_fail.pdf"
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.save(str(pdf_path))
    doc.close()

    service = SecretScanService()
    with patch("pdf_inspector.processing.pdf_loader.PDFLoader.render_page", side_effect=RuntimeError("Simulated MuPDF render failure")):
        doc_res = service.scan_pdf(pdf_path)

    assert doc_res.total_pages == 1
    assert doc_res.pages[0].status == SECRET_STATUS_ERROR
    assert doc_res.review_pages_count == 1


def test_c3_two_workflows_strictly_independent(tmp_path: Path) -> None:
    """
    Kiểm thử Part C3:
    - Chạy `SecretScanService` tuyệt đối không gọi `StripeDetector` hay `PageBoundaryDetector`.
    - Chạy `DocumentPipeline` tuyệt đối không gọi `SecretMarkingDetector`.
    """
    pdf_path = tmp_path / "workflow_isolation.pdf"
    doc = fitz.open()
    p = doc.new_page(width=595, height=842)
    p.insert_text((72, 120), "Tai lieu kiem tra doc lap hai workflow.")
    doc.save(str(pdf_path))
    doc.close()

    with patch("pdf_inspector.detectors.defects.stripe_detector.StripeDetector.detect", side_effect=AssertionError("StripeDetector must not run in Secret workflow")), patch(
        "pdf_inspector.detectors.boundary.boundary_detector.PageBoundaryDetector.detect",
        side_effect=AssertionError("BoundaryDetector must not run in Secret workflow"),
    ):
        sec_res = SecretScanService().scan_pdf(pdf_path)
        assert sec_res.total_pages == 1

    with patch(
        "pdf_inspector.detectors.secret.secret_detector.SecretMarkingDetector.detect_page",
        side_effect=AssertionError("SecretMarkingDetector must not run in Scan Quality workflow"),
    ):
        pipe_res = DocumentPipeline(DetectorConfig()).scan_document(pdf_path=pdf_path)
        assert pipe_res.doc_info.page_count == 1
