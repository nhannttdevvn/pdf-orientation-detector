"""
Kiểm thử đơn vị và tích hợp cho Page Boundary & Mixed-Page Detection (`tests/test_boundary_detection.py`):
- Phase 3.1: `analyze_page_layout` (TextRegionBlock, WhitespaceBand, seam lines, illumination step, edge truncation, two-column, book spread).
- Phase 3.2 & 3.3: `detect_boundary_candidates` & False Positive Filters + `OptionalLayoutModelVerifier`.
- Phase 3.5 & 3.7: `PageBoundaryDetector`, `CoordinateTransform` (RENDERED_IMAGE_PX <-> NORMALIZED <-> PDF_POINT, rotated mapping).
- Multi-Defect Independence (Cases 13 & 14: Rotated 180° + Mixed Page, Striped + Mixed Page).
- Toàn bộ 15 kịch bản chuẩn trong `data/samples/boundary_benchmark/`.
"""

import json
from pathlib import Path
import pytest

from pdf_inspector.core.config import (
    BOUNDARY_STATUS_MISSING,
    BOUNDARY_STATUS_MIXED,
    BOUNDARY_STATUS_NORMAL,
    BOUNDARY_STATUS_RECONSTRUCTABLE,
    STATUS_NORMAL,
    STATUS_ROTATED,
    STRIPE_STATUS_CLEAN,
    STRIPE_STATUS_DETECTED,
)
from pdf_inspector.detectors.boundary import (
    OptionalLayoutModelVerifier,
    PageBoundaryDetector,
    analyze_page_layout,
    detect_boundary_candidates,
)
from pdf_inspector.models.context import CoordinateTransform
from pdf_inspector.pipeline.document_pipeline import DocumentPipeline
from pdf_inspector.processing.pdf_loader import PDFLoader
from scripts.generate_boundary_samples import (
    BOUNDARY_BENCHMARK_DIR,
    GROUND_TRUTH_PATH,
    generate_all_boundary_samples,
)


@pytest.fixture(scope="module", autouse=True)
def ensure_boundary_samples() -> None:
    """Đảm bảo 15 file PDF mẫu kiểm thử Boundary đã được sinh trước khi chạy test."""
    if not GROUND_TRUTH_PATH.exists():
        generate_all_boundary_samples()


def test_coordinate_transform_normalized_and_rotated_mapping() -> None:
    """
    Kiểm tra quy đổi tọa độ giữa RENDERED_IMAGE_PX, NORMALIZED (0..1), PDF_POINT và ảnh xoay (90°/180°/270°).
    """
    ct = CoordinateTransform(
        pdf_width_pt=594.0,
        pdf_height_pt=841.5,
        image_width_px=1320,
        image_height_px=1870,
        dpi=160,
        original_rotation=0,
    )
    nx, ny = ct.pixel_to_normalized(660, 935)
    assert nx == pytest.approx(0.5, abs=1e-3)
    assert ny == pytest.approx(0.5, abs=1e-3)

    px, py = ct.normalized_to_pixel(0.5, 0.5)
    assert px == 660
    assert py == 935

    ptx, pty = ct.normalized_to_pdf_point(0.5, 0.5)
    assert ptx == pytest.approx(297.0, abs=0.5)
    assert pty == pytest.approx(420.75, abs=0.5)

    # Kiểm tra quy đổi tọa độ trên ảnh đã xoay 180° về ảnh gốc
    rx, ry = ct.rotated_pixel_to_rendered_pixel(300, 400, applied_rotation_cw=180)
    back_x, back_y = ct.rendered_pixel_to_rotated_pixel(rx, ry, applied_rotation_cw=180)
    assert back_x == 300
    assert back_y == 400


def test_layout_analysis_and_candidate_detection_on_mixed_vs_normal() -> None:
    """
    Kiểm tra `analyze_page_layout` và `detect_boundary_candidates`:
    - Phát hiện chính xác `MIXED_PAGE` tại `y_norm ~ 0.50` trên `02_two_pages_in_one_image.pdf`.
    - Bác bỏ nhiễu dương tính giả (False Positive) trên trang khoảng trắng lớn (`08`), bảng dài (`09`),
      chữ ký + kẻ ngang (`10`), 2 cột (`11`), và sách mở đôi (`12`).
    """
    mixed_pdf = BOUNDARY_BENCHMARK_DIR / "02_two_pages_in_one_image.pdf"
    with PDFLoader(mixed_pdf) as loader:
        ctx = loader.build_page_context(1, dpi=160, grayscale=True)
        layout = analyze_page_layout(
            gray_image=ctx.original_image,
            page_number=1,
            dpi=160,
        )
        cands = detect_boundary_candidates(layout, gray_image=ctx.original_image)
        ctx.release()

    valid_cands = [c for c in cands if not c.is_false_positive]
    assert len(valid_cands) >= 1
    top_cand = valid_cands[0]
    assert top_cand.boundary_norm == pytest.approx(0.50, abs=0.05)
    assert top_cand.suspicion_score >= 0.56

    # Kiểm tra các mẫu False Positive Control (08..12)
    fp_files = [
        "08_large_whitespace_normal.pdf",
        "09_long_table_normal.pdf",
        "10_signature_and_rule_normal.pdf",
        "11_two_column_normal.pdf",
        "12_valid_book_spread.pdf",
    ]
    detector = PageBoundaryDetector()
    for fname in fp_files:
        pdf_path = BOUNDARY_BENCHMARK_DIR / fname
        with PDFLoader(pdf_path) as loader:
            ctx = loader.build_page_context(1, dpi=160, grayscale=True)
            lay = detector.extract_page_layout(ctx)
            ctx.release()
        decisions, _ = detector.evaluate_document_boundaries({1: lay})
        assert decisions[1].status == BOUNDARY_STATUS_NORMAL, f"False positive trên {fname}"


def test_optional_layout_model_verifier_disabled_and_enabled() -> None:
    """
    Kiểm tra `OptionalLayoutModelVerifier`:
    - Khi tắt (`enabled=False`), không bao giờ tự bịa kết quả (`verified_by_model=False`).
    - Khi bật (`enabled=True`), cung cấp đánh giá thứ cấp hợp lệ từ dữ liệu bố cục thật.
    """
    mixed_pdf = BOUNDARY_BENCHMARK_DIR / "02_two_pages_in_one_image.pdf"
    with PDFLoader(mixed_pdf) as loader:
        ctx = loader.build_page_context(1, dpi=160, grayscale=True)
        layout = analyze_page_layout(
            gray_image=ctx.original_image,
            page_number=1,
            dpi=160,
        )
        cands = detect_boundary_candidates(layout, gray_image=ctx.original_image)
        ctx.release()

    cand = [c for c in cands if not c.is_false_positive][0]

    verifier_off = OptionalLayoutModelVerifier(enabled=False)
    res_off = verifier_off.verify_candidate(layout, cand)
    assert res_off["enabled"] is False
    assert res_off["verified_by_model"] is False
    assert res_off["adjusted_score"] == pytest.approx(cand.suspicion_score)

    verifier_on = OptionalLayoutModelVerifier(enabled=True)
    res_on = verifier_on.verify_candidate(layout, cand)
    assert res_on["enabled"] is True
    assert res_on["verified_by_model"] is True
    assert res_on["adjusted_score"] >= cand.suspicion_score


def test_multi_defect_cases_13_and_14() -> None:
    """
    Kiểm tra 2 kịch bản phối hợp đa lỗi quan trọng:
    - Case 13 (`13_rotated_180_and_mixed_page.pdf`): Vừa xoay 180° (`ROTATED`) vừa bị `MIXED_PAGE`.
    - Case 14 (`14_striped_and_mixed_page.pdf`): Vừa có sọc dọc (`STRIPE_DETECTED`) vừa bị `MIXED_PAGE`.
    """
    pipeline = DocumentPipeline()

    res13 = pipeline.scan_document(
        BOUNDARY_BENCHMARK_DIR / "13_rotated_180_and_mixed_page.pdf"
    )
    dec13 = res13.decisions[0]
    assert dec13.status == STATUS_ROTATED
    assert dec13.detected_rotation == 180
    assert dec13.boundary_status in (
        BOUNDARY_STATUS_MIXED,
        BOUNDARY_STATUS_RECONSTRUCTABLE,
    )
    assert any(b.defect_type == "MIXED_PAGE" for b in dec13.boundary_defects)

    res14 = pipeline.scan_document(
        BOUNDARY_BENCHMARK_DIR / "14_striped_and_mixed_page.pdf"
    )
    dec14 = res14.decisions[0]
    assert dec14.status == STATUS_NORMAL
    assert dec14.stripe_status == STRIPE_STATUS_DETECTED
    assert dec14.boundary_status in (
        BOUNDARY_STATUS_MIXED,
        BOUNDARY_STATUS_RECONSTRUCTABLE,
    )
    assert any(b.defect_type == "MIXED_PAGE" for b in dec14.boundary_defects)


def test_all_15_boundary_benchmark_cases_match_ground_truth() -> None:
    """
    Kiểm tra toàn bộ 15 file PDF trong `data/samples/boundary_benchmark/` khớp với `boundary_ground_truth.json`.
    """
    gt_data = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    pipeline = DocumentPipeline()

    for case in gt_data["cases"]:
        pdf_path = BOUNDARY_BENCHMARK_DIR / case["file_name"]
        res = pipeline.scan_document(pdf_path=pdf_path)
        for page_gt in case["pages"]:
            if page_gt.get("optional_companion_page"):
                continue
            pno = int(page_gt["page_number"])
            dec = res.decisions[pno - 1]
            exp_statuses = page_gt["expected_boundary_status"]
            assert dec.boundary_status in exp_statuses, (
                f"Case {case['file_name']} page {pno}: expected {exp_statuses}, "
                f"got {dec.boundary_status}"
            )
            exp_types = page_gt["expected_defect_types"]
            if exp_types:
                pred_types = [b.defect_type for b in dec.boundary_defects]
                assert any(t in pred_types for t in exp_types), (
                    f"Case {case['file_name']} page {pno}: expected defect types {exp_types}, "
                    f"got {pred_types}"
                )
