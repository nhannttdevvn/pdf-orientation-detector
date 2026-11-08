"""
Bộ kiểm thử đơn vị (Unit Tests) và kiểm thử tích hợp (Integration Tests) cho Phase 2:
Stripe Defect Detection (`tests/test_stripe_detector.py`).

Bao phủ đầy đủ 14 kịch bản bắt buộc trong Mục 9 (Phase 2.6 — Testing):
1. Clean page -> CLEAN
2. Dark vertical stripe -> STRIPE_DETECTED (VERTICAL_STRIPE)
3. Thin low-contrast stripe -> STRIPE_DETECTED (VERTICAL_STRIPE)
4. Horizontal stripe -> STRIPE_DETECTED (HORIZONTAL_STRIPE)
5. Diagonal stripe -> STRIPE_DETECTED (DIAGONAL_STRIPE)
6. Thick band stripe -> STRIPE_DETECTED (BAND_STRIPE)
7. Table page -> CLEAN (Không báo nhầm đường kẻ bảng biểu)
8. Underline page -> CLEAN (Không báo nhầm đường gạch chân văn bản)
9. Signature page -> CLEAN (Không báo nhầm chữ ký tay)
10. Rotated page with stripe -> Phát hiện đồng thời cả ROTATION (180°) và STRIPE (VERTICAL_STRIPE)
11. DetectorRegistry enables StripeDetector (ENABLED & AVAILABLE; Boundary giữ NOT_IMPLEMENTED)
12. DocumentPipeline runs Orientation + Stripe together trên cùng PageContext (không làm hỏng original_image)
13. Error handling returns PROCESSING_ERROR / DETECTOR_FAILED (không bao giờ gán nhầm là CLEAN)
14. Cross-page repeated-position evidence, Overlay visualization & Review feedback JSON persistence
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from pdf_inspector.core.config import (
    SAMPLES_DIR,
    STATUS_NORMAL,
    STATUS_ROTATED,
    STRIPE_STATUS_CLEAN,
    STRIPE_STATUS_DETECTED,
    STRIPE_STATUS_ERROR,
    STRIPE_STATUS_REVIEW,
)
from pdf_inspector.detectors.defects import (
    StripeDetector,
    StripePageResult,
    ValidatedStripe,
    render_stripe_overlay,
)
from pdf_inspector.detectors.defects.stripe_validator import StripeFeatures
from pdf_inspector.detectors.registry import create_default_registry
from pdf_inspector.models.schemas import DefectStatus, DefectType, DetectorStatus
from pdf_inspector.pipeline.document_pipeline import DocumentPipeline
from pdf_inspector.processing.pdf_loader import PDFLoader
from pdf_inspector.ui.components.preview_panel import save_stripe_review_feedback
from scripts.generate_stripe_samples import (
    GROUND_TRUTH_PATH,
    STRIPE_BENCHMARK_DIR,
    generate_all_stripe_samples,
)


@pytest.fixture(scope="session")
def stripe_benchmark_gt() -> Dict[str, Any]:
    """Đảm bảo bộ 12 PDF benchmark Stripe đã được sinh đầy đủ trước khi chạy test."""
    if not GROUND_TRUTH_PATH.exists():
        return generate_all_stripe_samples()
    data = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    for case in data.get("cases", []):
        if not (STRIPE_BENCHMARK_DIR / case["file_name"]).exists():
            return generate_all_stripe_samples()
    return data


def _inspect_benchmark_case(file_name: str) -> StripePageResult:
    pdf_path = STRIPE_BENCHMARK_DIR / file_name
    detector = StripeDetector()
    with PDFLoader(pdf_path) as loader:
        ctx = loader.build_page_context(1, dpi=160, grayscale=True)
        orig_copy = ctx.original_image.copy()
        res = detector.detect_page(
            image=ctx.original_image,
            page_number=1,
            dpi=160,
            clahe_image=ctx.ensure_clahe(),
        )
        # Đảm bảo StripeDetector tuyệt đối không làm thay đổi original_image trong PageContext
        assert np.array_equal(ctx.original_image, orig_copy)
        ctx.release()
    return res


def test_01_clean_page(stripe_benchmark_gt: Dict[str, Any]) -> None:
    """Case 1: Trang văn bản sạch không có sọc phải trả về CLEAN."""
    res = _inspect_benchmark_case("01_clean_page.pdf")
    assert res.status == STRIPE_STATUS_CLEAN
    assert res.has_stripe is False
    assert res.stripes == []


def test_02_dark_vertical_stripe(stripe_benchmark_gt: Dict[str, Any]) -> None:
    """Case 2: Sọc dọc đậm kéo dài phải được phát hiện là STRIPE_DETECTED (VERTICAL_STRIPE)."""
    res = _inspect_benchmark_case("02_dark_vertical_stripe.pdf")
    assert res.status == STRIPE_STATUS_DETECTED
    assert res.has_stripe is True
    assert len(res.stripes) >= 1
    top = res.stripes[0]
    assert top.stripe_type == "VERTICAL"
    assert top.defect_type == "VERTICAL_STRIPE"
    assert top.status == "CONFIRMED"
    assert top.suspicion_score >= 0.75
    assert top.relative_length >= 0.90
    assert abs(top.bbox[0] - 380) <= 4
    assert top.features is not None
    assert len(top.features.detected_by) >= 2


def test_03_thin_low_contrast_gray_stripe(stripe_benchmark_gt: Dict[str, Any]) -> None:
    """Case 3: Sọc dọc xám mảnh (2px, độ tương phản thấp/vừa) vẫn phải được phát hiện chính xác."""
    res = _inspect_benchmark_case("03_thin_gray_vertical_stripe.pdf")
    assert res.status == STRIPE_STATUS_DETECTED
    assert res.has_stripe is True
    top = res.stripes[0]
    assert top.stripe_type == "VERTICAL"
    assert top.defect_type == "VERTICAL_STRIPE"
    assert top.width_px <= 5.0
    assert abs(top.bbox[0] - 820) <= 4


def test_04_horizontal_stripe(stripe_benchmark_gt: Dict[str, Any]) -> None:
    """Case 4: Sọc ngang kéo dài xuyên lề phải được phát hiện là HORIZONTAL_STRIPE."""
    res = _inspect_benchmark_case("04_horizontal_stripe.pdf")
    assert res.status == STRIPE_STATUS_DETECTED
    assert res.has_stripe is True
    top = res.stripes[0]
    assert top.stripe_type == "HORIZONTAL"
    assert top.defect_type == "HORIZONTAL_STRIPE"
    assert top.relative_length >= 0.90
    assert abs(top.bbox[1] - 625) <= 5


def test_05_diagonal_stripe(stripe_benchmark_gt: Dict[str, Any]) -> None:
    """Case 5: Sọc hơi nghiêng (~85.4°) phải được phát hiện bởi HoughLinesP và phân loại DIAGONAL_STRIPE."""
    res = _inspect_benchmark_case("05_diagonal_stripe.pdf")
    assert res.status == STRIPE_STATUS_DETECTED
    assert res.has_stripe is True
    top = res.stripes[0]
    assert top.stripe_type == "DIAGONAL"
    assert top.defect_type == "DIAGONAL_STRIPE"
    assert 80.0 <= top.angle_deg <= 88.5
    assert top.relative_length >= 0.85


def test_06_thick_band_stripe(stripe_benchmark_gt: Dict[str, Any]) -> None:
    """Case 6: Dải sọc dày (~16px) phải được phân loại là BAND / BAND_STRIPE."""
    res = _inspect_benchmark_case("06_thick_band_stripe.pdf")
    assert res.status == STRIPE_STATUS_DETECTED
    assert res.has_stripe is True
    top = res.stripes[0]
    assert top.stripe_type == "BAND"
    assert top.defect_type == "BAND_STRIPE"
    assert top.width_px >= 10.0
    assert abs(top.bbox[0] - 960) <= 5


def test_06b_dashed_and_uneven_background_stripes(
    stripe_benchmark_gt: Dict[str, Any],
) -> None:
    """Kiểm tra thêm Case 7 (sọc đứt quãng) và Case 8 (sọc trên nền giấy sáng tối không đều)."""
    res_dashed = _inspect_benchmark_case("07_dashed_interrupted_stripe.pdf")
    assert res_dashed.status in (STRIPE_STATUS_DETECTED, STRIPE_STATUS_REVIEW)
    assert res_dashed.has_stripe is True
    assert abs(res_dashed.stripes[0].bbox[0] - 445) <= 5

    res_uneven = _inspect_benchmark_case("08_uneven_background_stripe.pdf")
    assert res_uneven.status == STRIPE_STATUS_DETECTED
    assert res_uneven.has_stripe is True
    assert abs(res_uneven.stripes[0].bbox[0] - 710) <= 5


def test_07_table_page_no_false_positive(stripe_benchmark_gt: Dict[str, Any]) -> None:
    """Case 7: Trang chứa bảng biểu nhiều hàng nhiều cột không được báo nhầm là sọc scan."""
    res = _inspect_benchmark_case("09_table_page.pdf")
    assert res.status == STRIPE_STATUS_CLEAN
    assert res.has_stripe is False
    assert res.stripes == []
    assert any(
        r.rejection_category == "table_grid_line" for r in res.rejected_candidates
    )


def test_08_underline_page_no_false_positive(
    stripe_benchmark_gt: Dict[str, Any],
) -> None:
    """Case 8: Trang chứa đường gạch chân tiêu đề / câu văn không được báo nhầm là sọc ngang."""
    res = _inspect_benchmark_case("10_underline_page.pdf")
    assert res.status == STRIPE_STATUS_CLEAN
    assert res.has_stripe is False
    assert res.stripes == []
    assert any(
        r.rejection_category == "text_underline" for r in res.rejected_candidates
    )


def test_09_signature_page_no_false_positive(
    stripe_benchmark_gt: Dict[str, Any],
) -> None:
    """Case 9: Trang chứa nét chữ ký tay không được báo nhầm là sọc scan."""
    res = _inspect_benchmark_case("11_signature_page.pdf")
    assert res.status == STRIPE_STATUS_CLEAN
    assert res.has_stripe is False
    assert res.stripes == []


def test_10_rotated_page_with_stripe_detects_both_defects(
    stripe_benchmark_gt: Dict[str, Any],
) -> None:
    """
    Case 10: Trang vừa bị xoay ngược 180° vừa có vệt sọc dọc phải phát hiện đồng thời cả 2 lỗi
    (Orientation = ROTATED 180° và Stripe = STRIPE_DETECTED) mà không ghi đè nhau.
    """
    pdf_path = STRIPE_BENCHMARK_DIR / "12_rotated_180_with_stripe.pdf"
    pipeline = DocumentPipeline()
    result = pipeline.scan_document(pdf_path)

    assert len(result.decisions) == 1
    dec = result.decisions[0]
    assert dec.status == STATUS_ROTATED
    assert dec.detected_rotation == 180
    assert dec.correction_angle == 180
    assert dec.stripe_status == STRIPE_STATUS_DETECTED
    assert len(dec.stripe_defects) >= 1
    assert dec.detector_outcomes[DefectType.ROTATION.value] == DefectStatus.CONFIRMED.value
    assert dec.detector_outcomes[DefectType.STRIPE.value] == DefectStatus.CONFIRMED.value

    defect_types = {d.defect_type for d in dec.defects}
    assert DefectType.ROTATION.value in defect_types
    assert "VERTICAL_STRIPE" in defect_types


def test_11_detector_registry_enables_stripe_detector() -> None:
    """
    Case 11: DetectorRegistry mặc định phải đăng ký OrientationDetector, StripeDetector và PageBoundaryDetector
    ở trạng thái ENABLED & AVAILABLE.
    """
    registry = create_default_registry()
    assert registry.get_status("orientation") == DetectorStatus.ENABLED.value
    assert registry.get_status("stripe") == DetectorStatus.ENABLED.value
    assert registry.is_available("stripe") is True
    assert registry.get_status("boundary") == DetectorStatus.ENABLED.value
    assert registry.is_available("boundary") is True
    assert registry.get_enabled_detector_ids() == ["orientation", "stripe", "boundary"]
    assert registry.get_unsupported_checks() == []


def test_12_document_pipeline_runs_orientation_and_stripe_together(
    tmp_path: Path,
) -> None:
    """
    Case 12: DocumentPipeline chạy đồng thời Orientation + Stripe + Boundary trên tài liệu nhiều trang,
    bảo toàn kết quả khi gọi update_decisions_and_artifacts và xuất đầy đủ JSON/CSV.
    """
    pdf_path = SAMPLES_DIR / "02_single_upside_down_180.pdf"
    pipeline = DocumentPipeline()
    result = pipeline.scan_document(pdf_path)

    assert result.enabled_detectors == ["orientation", "stripe", "boundary"]
    assert result.unsupported_checks == []
    assert len(result.decisions) == 3
    assert [d.status for d in result.decisions] == [
        STATUS_NORMAL,
        STATUS_ROTATED,
        STATUS_NORMAL,
    ]
    assert all(d.stripe_status == STRIPE_STATUS_CLEAN for d in result.decisions)

    json_path = tmp_path / "combined_report.json"
    csv_path = tmp_path / "combined_report.csv"
    updated = DocumentPipeline.update_decisions_and_artifacts(
        result=result,
        json_report_path=json_path,
        csv_report_path=csv_path,
    )
    assert json_path.exists()
    assert csv_path.exists()
    report_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert "stripe_summary" in report_json["summary"]
    assert report_json["pages"][1]["detector_outcomes"]["STRIPE"] == "NO_DEFECT"
    assert updated.decisions[1].detector_outcomes["STRIPE"] == "NO_DEFECT"


def test_13_error_handling_returns_processing_error_never_clean() -> None:
    """
    Case 13: Khi ảnh đầu vào bị rỗng hoặc quá nhỏ/lỗi, StripeDetector phải trả về
    PROCESSING_ERROR và DefectResult(status=DETECTOR_FAILED, requires_review=True),
    tuyệt đối không được trả về CLEAN hay NO_DEFECT.
    """
    detector = StripeDetector()
    empty_img = np.array([], dtype=np.uint8)
    page_res = detector.detect_page(empty_img, page_number=1, dpi=160)

    assert page_res.status == STRIPE_STATUS_ERROR
    assert page_res.has_stripe is False
    assert page_res.error is not None

    defects = detector.build_defect_results(page_res, dpi=160)
    assert len(defects) == 1
    assert defects[0].status == DefectStatus.DETECTOR_FAILED.value
    assert defects[0].requires_review is True


def test_14_cross_page_repetition_overlay_and_feedback(tmp_path: Path) -> None:
    """
    Case 14: Kiểm tra:
    - Bằng chứng lặp lại vị trí sọc liên trang (`apply_cross_page_repetition_evidence`).
    - Hàm `render_stripe_overlay` không làm thay đổi mảng ảnh gốc.
    - Hàm `save_stripe_review_feedback` lưu JSON phản hồi với `verified_ground_truth=False`.
    """
    detector = StripeDetector(confirm_threshold=0.62, review_threshold=0.42)

    def _make_borderline_res(pno: int) -> StripePageResult:
        feat = StripeFeatures(
            relative_length=0.52,
            length_px=950.0,
            thickness_px=2.0,
            angle_deg=90.0,
            continuity=0.65,
            contrast=14.0,
            background_intensity=245.0,
            stripe_intensity=231.0,
            intensity_std=4.0,
            position_norm=0.3125,
            edge_distance_norm=0.3125,
            margin_penetration=0.2,
            content_overlap=0.1,
            endpoint_junctions=0,
            perpendicular_crossings=0,
            parallel_grid_siblings=0,
            text_above_density=0.0,
            repeat_page_count=1,
            detected_by=["projection_profile"],
        )
        stripe = ValidatedStripe(
            stripe_type="VERTICAL",
            defect_type="VERTICAL_STRIPE",
            status="SUSPECTED",
            bbox=(412, 100, 414, 1050),
            endpoints=(413.0, 100.0, 413.0, 1050.0),
            length_px=950.0,
            width_px=2.0,
            relative_length=0.52,
            angle_deg=90.0,
            suspicion_score=0.55,
            confidence=0.55,
            severity="MEDIUM",
            reason="Borderline vertical candidate",
            features=feat,
        )
        return StripePageResult(
            page_number=pno,
            status=STRIPE_STATUS_REVIEW,
            has_stripe=True,
            candidates_count=1,
            confirmed_count=0,
            suspected_count=1,
            stripes=[stripe],
        )

    page_map = {1: _make_borderline_res(1), 2: _make_borderline_res(2)}
    detector.apply_cross_page_repetition_evidence(page_map)

    assert page_map[1].stripes[0].features is not None
    assert page_map[1].stripes[0].features.repeat_page_count == 2
    assert page_map[1].stripes[0].suspicion_score == pytest.approx(0.67)
    assert page_map[1].status == STRIPE_STATUS_DETECTED

    # Kiểm tra render_stripe_overlay không thay đổi ảnh gốc
    orig_img = np.full((300, 200, 3), 240, dtype=np.uint8)
    orig_backup = orig_img.copy()
    defects = detector.build_defect_results(page_map[1], dpi=160)
    overlay_img = render_stripe_overlay(orig_img, defects)
    assert np.array_equal(orig_img, orig_backup)
    assert overlay_img.shape == orig_img.shape
    assert not np.array_equal(overlay_img, orig_img)

    # Kiểm tra lưu feedback người dùng
    fb_file = tmp_path / "stripe_review_feedback.json"
    save_stripe_review_feedback(
        file_name="sample_test.pdf",
        page_number=1,
        stripe_status=STRIPE_STATUS_DETECTED,
        feedback_action="Confirm defect",
        stripe_defects=defects,
        feedback_path=fb_file,
    )
    assert fb_file.exists()
    fb_data = json.loads(fb_file.read_text(encoding="utf-8"))
    assert fb_data["verified_ground_truth"] is False
    assert len(fb_data["entries"]) == 1
    assert fb_data["entries"][0]["user_feedback"] == "Confirm defect"

