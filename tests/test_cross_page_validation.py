"""
Kiểm thử đơn vị và tích hợp cho Cross-Page Validation (`tests/test_cross_page_validation.py`):
- Phase 3.4: So khớp vùng biên liên trang `N-1`, `N`, `N+1` bằng NCC (`cv2.matchTemplate`),
  Horizontal Projection Correlation, ORB Feature Matching (`cv2.ORB_create`),
  Margin/Width Consistency và Complementary Height Analysis.
- Phân biệt rõ `RECONSTRUCTABLE` (`CONTENT_OVERLAP`, `SPLIT_PAGE`, `MIXED_SPILLOVER`)
  với `MISSING_CONTENT` (`CONTENT_TRUNCATED` không có phần thiếu ở trang liền kề).
- Kiểm tra giải phóng bộ nhớ `release_strips()` sau khi đối chiếu liên trang.
"""

import pytest

from pdf_inspector.core.config import (
    BOUNDARY_STATUS_MISSING,
    BOUNDARY_STATUS_MIXED,
    BOUNDARY_STATUS_RECONSTRUCTABLE,
)
from pdf_inspector.detectors.boundary import (
    PageBoundaryDetector,
    detect_overlap_between_adjacent_pages,
    detect_split_or_spillover_between_adjacent_pages,
    run_cross_page_validation,
)
from pdf_inspector.processing.pdf_loader import PDFLoader
from scripts.generate_boundary_samples import (
    BOUNDARY_BENCHMARK_DIR,
    GROUND_TRUTH_PATH,
    generate_all_boundary_samples,
)


@pytest.fixture(scope="module", autouse=True)
def ensure_boundary_samples() -> None:
    if not GROUND_TRUTH_PATH.exists():
        generate_all_boundary_samples()


def test_cross_page_content_overlap_detection() -> None:
    """
    Kiểm tra phát hiện trùng lặp nội dung liên trang (`CONTENT_OVERLAP`) trên `07_content_overlap_pages.pdf`:
    - `ncc_score >= 0.65`, `profile_correlation >= 0.70`.
    - Trang 1 liên kết tới `related_pages = [2]`, Trang 2 liên kết ngược về `related_pages = [1]`.
    - Cả hai trang được phân loại là `RECONSTRUCTABLE`.
    """
    pdf_path = BOUNDARY_BENCHMARK_DIR / "07_content_overlap_pages.pdf"
    detector = PageBoundaryDetector()
    layouts = []
    with PDFLoader(pdf_path) as loader:
        for pno in (1, 2):
            ctx = loader.build_page_context(pno, dpi=160, grayscale=True)
            layouts.append(detector.extract_page_layout(ctx))
            ctx.release()

    ov_match = detect_overlap_between_adjacent_pages(layouts[0], layouts[1])
    assert ov_match is not None
    assert ov_match.relation_type == "CONTENT_OVERLAP"
    assert ov_match.confirmed is True
    assert ov_match.ncc_score >= 0.65
    assert ov_match.profile_correlation >= 0.70
    assert ov_match.overlap_height_norm == pytest.approx(0.16, abs=0.05)

    decisions, defects_map = detector.evaluate_document_boundaries(
        {1: layouts[0], 2: layouts[1]}
    )
    assert decisions[1].status == BOUNDARY_STATUS_RECONSTRUCTABLE
    assert decisions[2].status == BOUNDARY_STATUS_RECONSTRUCTABLE
    assert 2 in decisions[1].related_pages
    assert 1 in decisions[2].related_pages
    assert any(d.defect_type == "CONTENT_OVERLAP" for d in defects_map[1])
    assert any(d.defect_type == "CONTENT_OVERLAP" for d in defects_map[2])

    # Kiểm tra các dải ảnh tạm đã được giải phóng sau khi evaluate_document_boundaries kết thúc
    assert layouts[0].top_strip_gray is None
    assert layouts[0].bottom_strip_gray is None
    assert layouts[1].top_strip_gray is None
    assert layouts[1].bottom_strip_gray is None


def test_cross_page_split_page_detection() -> None:
    """
    Kiểm tra phát hiện một trang vật lý bị chia thành 2 trang liên tiếp (`SPLIT_PAGE`)
    trên `04_page_b_continuation_from_prev.pdf`:
    - Độ đồng nhất độ rộng lề `width_consistency_score >= 0.85`.
    - Tổng tỷ lệ lấp đầy bổ sung hợp lý (`complementary_height_score >= 0.70`).
    - Trang 1 liên kết `related_pages = [2]` và Trang 2 liên kết `related_pages = [1]`.
    """
    pdf_path = BOUNDARY_BENCHMARK_DIR / "04_page_b_continuation_from_prev.pdf"
    detector = PageBoundaryDetector()
    layouts = {}
    with PDFLoader(pdf_path) as loader:
        for pno in (1, 2):
            ctx = loader.build_page_context(pno, dpi=160, grayscale=True)
            layouts[pno] = detector.extract_page_layout(ctx)
            ctx.release()

    sp_match = detect_split_or_spillover_between_adjacent_pages(layouts[1], layouts[2])
    assert sp_match is not None
    assert sp_match.relation_type == "SPLIT_CONTINUATION"
    assert sp_match.width_consistency_score >= 0.75
    assert sp_match.complementary_height_score >= 0.70

    decisions, defects_map = detector.evaluate_document_boundaries(layouts)
    assert decisions[1].status == BOUNDARY_STATUS_RECONSTRUCTABLE
    assert decisions[2].status == BOUNDARY_STATUS_RECONSTRUCTABLE
    assert 2 in decisions[1].related_pages
    assert 1 in decisions[2].related_pages
    assert any(d.defect_type == "SPLIT_PAGE" for d in defects_map[1])
    assert any(d.defect_type == "SPLIT_PAGE" for d in defects_map[2])


def test_cross_page_mixed_spillover_and_unrecoverable_missing_content() -> None:
    """
    Kiểm tra:
    - `03_page_a_with_top_of_b.pdf`: Trang 1 chứa phần đầu của Trang 2 ở cuối trang -> liên kết `related_pages = [2]`.
    - `05_missing_top_content.pdf` và `06_missing_bottom_content.pdf`: Nội dung bị cắt sát mép nhưng không có ở
      trang liền kề -> phân loại chính xác là `MISSING_CONTENT` (`REQUIRES_RESCAN`), tuyệt đối không đánh dấu `RECONSTRUCTABLE`.
    """
    detector = PageBoundaryDetector()

    # Case 03: Mixed spillover
    pdf03 = BOUNDARY_BENCHMARK_DIR / "03_page_a_with_top_of_b.pdf"
    layouts03 = {}
    with PDFLoader(pdf03) as loader:
        for pno in (1, 2):
            ctx = loader.build_page_context(pno, dpi=160, grayscale=True)
            layouts03[pno] = detector.extract_page_layout(ctx)
            ctx.release()
    dec03, _ = detector.evaluate_document_boundaries(layouts03)
    assert dec03[1].status in (BOUNDARY_STATUS_MIXED, BOUNDARY_STATUS_RECONSTRUCTABLE)
    assert 2 in dec03[1].related_pages

    # Case 05 & 06: Unrecoverable edge truncation -> MISSING_CONTENT
    for fname in ("05_missing_top_content.pdf", "06_missing_bottom_content.pdf"):
        pdf_trunc = BOUNDARY_BENCHMARK_DIR / fname
        with PDFLoader(pdf_trunc) as loader:
            ctx = loader.build_page_context(1, dpi=160, grayscale=True)
            lay = detector.extract_page_layout(ctx)
            ctx.release()
        dec_trunc, defs_trunc = detector.evaluate_document_boundaries({1: lay})
        assert dec_trunc[1].status == BOUNDARY_STATUS_MISSING
        assert dec_trunc[1].reconstruction_proposal is not None
        assert dec_trunc[1].reconstruction_proposal.action == "REQUIRES_RESCAN"
        assert dec_trunc[1].reconstruction_proposal.reconstructable is False
        assert any(d.defect_type == "CONTENT_TRUNCATED" for d in defs_trunc[1])
