"""
Bộ kiểm thử đơn vị (Unit Tests), kiểm thử tích hợp (Integration Tests) và kiểm thử hồi quy (Regression Tests)
cho Phase 4 — Part A: Horizontal Stripe Detection Improvement (`tests/test_horizontal_stripe_detector.py`).

Bao phủ đầy đủ:
1. Ước lượng bền vững `ContentMargins` (`estimate_content_margins`): không bị lệch bởi sọc ngang/dọc xuyên trang,
   và fallback rõ ràng (`reliable=False`) trên trang trắng.
2. Toàn bộ 15 kịch bản bắt buộc mục A11 (`01` .. `15` trong `horizontal_stripe_benchmark`).
3. Phân loại 3 mức ứng viên đường ngang (`SCAN_STRIPE`, `LIKELY_DOCUMENT_LINE`, `REVIEW`) kèm `reason_codes` tường minh.
4. Quy tắc mục A7: đường có vị trí/đặc điểm giống footer nhưng vượt lề bất thường -> ưu tiên `REVIEW` thay vì `CONFIRMED` tự động.
5. Không đưa `LIKELY_DOCUMENT_LINE` vào danh sách lỗi sọc xác nhận trên `DocumentPipeline`.
6. Kiểm tra xử lý ngoại lệ trả về `PROCESSING_ERROR` (tuyệt đối không gán `CLEAN`).
"""

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytest

from ai_engine.core.config import (
    HORIZONTAL_CLASS_DOCUMENT_LINE,
    HORIZONTAL_CLASS_REVIEW,
    HORIZONTAL_CLASS_SCAN_STRIPE,
    STRIPE_STATUS_CLEAN,
    STRIPE_STATUS_DETECTED,
    STRIPE_STATUS_ERROR,
    STRIPE_STATUS_REVIEW,
)
from ai_engine.detectors.defects import (
    StripeDetector,
    StripePageResult,
    estimate_content_margins,
    prepare_stripe_maps,
)
from ai_engine.pipelines.document_pipeline import DocumentPipeline
from ai_engine.preprocessing.pdf_loader import PDFLoader
from benchmarks.generators.generate_horizontal_stripe_samples import (
    H_GROUND_TRUTH_PATH,
    H_STRIPE_BENCHMARK_DIR,
    PAGE_H_PX,
    PAGE_W_PX,
    _create_document_canvas,
    generate_horizontal_stripe_benchmark,
)


@pytest.fixture(scope="session")
def horizontal_benchmark_gt() -> Dict[str, Any]:
    """Đảm bảo bộ 23 PDF kiểm thử Horizontal Stripe đã được sinh đầy đủ."""
    if not H_GROUND_TRUTH_PATH.exists():
        return generate_horizontal_stripe_benchmark()
    data = json.loads(H_GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    for case in data.get("cases", []):
        if not (H_STRIPE_BENCHMARK_DIR / case["file_name"]).exists():
            return generate_horizontal_stripe_benchmark()
    return data


def _inspect_h_case(file_name: str) -> StripePageResult:
    pdf_path = H_STRIPE_BENCHMARK_DIR / file_name
    detector = StripeDetector()
    with PDFLoader(pdf_path) as loader:
        ctx = loader.build_page_context(1, dpi=160, grayscale=True)
        res = detector.detect_page(
            image=ctx.original_image,
            page_number=1,
            dpi=160,
            clahe_image=ctx.ensure_clahe(),
        )
        ctx.release()
    return res


def test_content_margins_robust_against_full_width_stripe_and_blank_fallback() -> None:
    """
    A3: Phép ước lượng `content_left` và `content_right` không được bị kéo lệch bởi vệt sọc dài xuyên trang,
    và phải trả về fallback `reliable=False` khi trang trắng không có chữ.
    """
    clean_img = _create_document_canvas(seed=501, left_margin=175, right_margin=1145)
    maps_clean = prepare_stripe_maps(clean_img, dpi=160)
    m_clean = estimate_content_margins(maps_clean)
    assert m_clean.reliable is True
    assert 160.0 <= m_clean.content_left <= 190.0

    # Kẻ thêm một sọc ngang chạy suốt từ x=0..1320 và một sọc dọc ở lề trái x=40..44
    striped_img = clean_img.copy()
    striped_img[620:625, 0:PAGE_W_PX] = 25
    striped_img[10 : PAGE_H_PX - 10, 40:44] = 25
    maps_striped = prepare_stripe_maps(striped_img, dpi=160)
    m_striped = estimate_content_margins(maps_striped)
    assert m_striped.reliable is True
    assert abs(m_striped.content_left - m_clean.content_left) <= 8.0
    assert abs(m_striped.content_right - m_clean.content_right) <= 12.0

    # Trang trắng hoàn toàn -> fallback an toàn với reliable=False
    blank_img = np.full((PAGE_H_PX, PAGE_W_PX), 248, dtype=np.uint8)
    maps_blank = prepare_stripe_maps(blank_img, dpi=160)
    m_blank = estimate_content_margins(maps_blank)
    assert m_blank.reliable is False
    assert m_blank.method == "fallback_proportional"


@pytest.mark.parametrize(
    "file_name",
    [
        "01_quoc_hieu_header_rule.pdf",
        "02_horizontal_tables_page.pdf",
        "03_underlines_and_section_dividers.pdf",
        "04_footer_separator_line.pdf",
        "05_signature_and_form_lines.pdf",
        "12_valid_95pct_wide_form_frame.pdf",
        "14_clean_document_page.pdf",
    ],
)
def test_a11_legitimate_horizontal_lines_classified_clean(
    horizontal_benchmark_gt: Dict[str, Any],
    file_name: str,
) -> None:
    """
    A11 Cases 1, 2, 3, 4, 5, 12, 14: Mọi đường kẻ ngang hợp lệ (Quốc hiệu, bảng không kẻ dọc,
    gạch chân tiêu đề, phân cách mục, chân trang, đường ký tên/biểu mẫu và khung biểu mẫu rộng 95%)
    phải trả về CLEAN và được gán `LIKELY_DOCUMENT_LINE` trong rejected_candidates.
    """
    res = _inspect_h_case(file_name)
    assert res.status == STRIPE_STATUS_CLEAN
    assert res.has_stripe is False
    assert res.stripes == []
    if file_name != "14_clean_document_page.pdf":
        assert any(
            r.horizontal_classification == HORIZONTAL_CLASS_DOCUMENT_LINE
            for r in res.rejected_candidates
        )


@pytest.mark.parametrize(
    "file_name,expected_statuses",
    [
        ("06_near_full_width_horizontal_stripe.pdf", (STRIPE_STATUS_DETECTED,)),
        ("07_faint_gray_horizontal_stripe.pdf", (STRIPE_STATUS_DETECTED, STRIPE_STATUS_REVIEW)),
        ("08_thick_dark_horizontal_stripe.pdf", (STRIPE_STATUS_DETECTED,)),
        ("09_extends_both_margins_stripe.pdf", (STRIPE_STATUS_DETECTED,)),
        ("10_slightly_tilted_horizontal_stripe.pdf", (STRIPE_STATUS_DETECTED,)),
        ("11_stripe_crossing_through_table.pdf", (STRIPE_STATUS_DETECTED,)),
        ("13_short_abnormal_edge_bleed_stripe.pdf", (STRIPE_STATUS_DETECTED, STRIPE_STATUS_REVIEW)),
    ],
)
def test_a11_real_horizontal_scan_stripes_detected(
    horizontal_benchmark_gt: Dict[str, Any],
    file_name: str,
    expected_statuses: tuple[str, ...],
) -> None:
    """
    A11 Cases 6, 7, 8, 9, 10, 11, 13: Các vệt sọc ngang thật (gần hết chiều rộng, xám nhạt, đen dày,
    vượt hai lề, hơi nghiêng, đè xuyên qua bảng, hoặc ngắn nhưng tràn mép cắt chữ bất thường) phải được bắt đầy đủ.
    """
    res = _inspect_h_case(file_name)
    assert res.status in expected_statuses
    assert res.has_stripe is True
    assert len(res.stripes) >= 1


def test_a11_case_15_both_vertical_and_horizontal_stripes(
    horizontal_benchmark_gt: Dict[str, Any],
) -> None:
    """
    A11 Case 15: Trang vừa có sọc dọc vừa có sọc ngang phải phát hiện đồng thời cả VERTICAL và HORIZONTAL.
    """
    res = _inspect_h_case("15_both_vertical_and_horizontal_stripes.pdf")
    assert res.status == STRIPE_STATUS_DETECTED
    detected_types = {s.stripe_type for s in res.stripes}
    assert "VERTICAL" in detected_types
    assert "HORIZONTAL" in detected_types

    h_stripes = [s for s in res.stripes if s.stripe_type == "HORIZONTAL"]
    assert h_stripes[0].horizontal_classification == HORIZONTAL_CLASS_SCAN_STRIPE
    assert "EXTENDS_BOTH_MARGINS" in h_stripes[0].reason_codes
    assert "NEAR_FULL_PAGE_WIDTH" in h_stripes[0].reason_codes


def test_a7_footer_like_line_extending_margin_prefers_review_over_confirmed() -> None:
    """
    A7: Nếu đường ở vùng chân trang giống footer rule nhưng có phần kéo dài bất thường ra ngoài một lề
    (mà không tràn sát cả hai mép giấy), hệ thống ưu tiên trả về REVIEW thay vì xác nhận lỗi tự động.
    """
    img = _create_document_canvas(seed=601, left_margin=160, right_margin=1160)
    # Vẽ đường giống footer mảnh ở y=1700 nhưng lệch vượt sâu ra ngoài lề trái (x=45..1160, dài 84.5%)
    cv2.line(img, (45, 1700), (1160, 1700), (35,), 2)
    cv2.putText(
        img,
        "Trang 01/05",
        (160, 1745),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (45,),
        1,
        cv2.LINE_AA,
    )

    det = StripeDetector()
    res = det.detect_page(img, page_number=1, dpi=160)
    assert res.status == STRIPE_STATUS_REVIEW
    assert len(res.stripes) == 1
    assert res.stripes[0].horizontal_classification == HORIZONTAL_CLASS_REVIEW
    assert "LIKELY_FOOTER_SEPARATOR" in res.stripes[0].reason_codes


def test_pipeline_excludes_likely_document_line_from_error_decisions(
    horizontal_benchmark_gt: Dict[str, Any],
) -> None:
    """
    A9 & B2: DocumentPipeline chạy trên trang có bảng ngang / đường phân cách hợp lệ không được đưa
    `LIKELY_DOCUMENT_LINE` vào `stripe_defects` hay trạng thái lỗi của trang.
    """
    pdf_path = H_STRIPE_BENCHMARK_DIR / "02_horizontal_tables_page.pdf"
    pipeline = DocumentPipeline()
    pipe_res = pipeline.scan_document(pdf_path)
    dec = pipe_res.decisions[0]
    assert dec.stripe_status == STRIPE_STATUS_CLEAN
    assert dec.stripe_defects == []


def test_detector_failure_returns_processing_error() -> None:
    """A9: Detector thất bại phải trả về PROCESSING_ERROR, không mặc định CLEAN."""
    det = StripeDetector()
    bad_img = np.zeros((10, 10), dtype=np.uint8)
    res = det.detect_page(bad_img, page_number=1, dpi=160)
    assert res.status == STRIPE_STATUS_ERROR
    assert res.has_stripe is False

