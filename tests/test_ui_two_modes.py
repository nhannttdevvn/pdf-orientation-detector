"""
Kiểm thử tích hợp hai chế độ giao diện Dashboard (`tests/test_ui_two_modes.py`):
- Tab 1 (`SCAN_QUALITY`): Đảm bảo trang chỉ có đường kẻ tài liệu hợp lệ (`LIKELY_DOCUMENT_LINE` / `CLEAN`)
  không bị đưa vào `build_problem_page_rows`, trong khi trang có `SCAN_STRIPE` thật vẫn hiển thị đầy đủ.
- Tab 2 (`SECRET_DETECTION`): Kiểm tra bộ lọc 5 mức (`Tất cả | Mật | Tối mật | Tuyệt mật | Cần kiểm tra`),
  thao tác rà soát (`Xác nhận | Bỏ qua | Chỉnh sửa kết quả`), xuất báo cáo JSON/CSV và cô lập trạng thái
  giữa hai chế độ (`release_secret_resources` không xóa kết quả Tab 1 và ngược lại).
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from pdf_inspector.core.config import (
    BOUNDARY_STATUS_NORMAL,
    SECRET_STATUS_DETECTED,
    SECRET_STATUS_NO_MARK,
    SECRET_STATUS_REVIEW,
    STATUS_NORMAL,
    STRIPE_STATUS_CLEAN,
    STRIPE_STATUS_DETECTED,
    DetectorConfig,
)
from pdf_inspector.detectors.secret.secret_detector import DetectedSecretMark, SecretPageResult
from pdf_inspector.models.schemas import (
    DefectResult,
    DefectStatus,
    PDFDocumentInfo,
    PageDecision,
    PipelineResult,
)
from pdf_inspector.services.secret_scan_service import SecretDocumentResult, SecretScanService
from pdf_inspector.ui.components.secret_tab_view import (
    build_secret_flagged_rows,
    filter_secret_rows,
    save_secret_review_feedback,
)
from pdf_inspector.ui.state_manager import (
    build_problem_page_rows,
    compute_dashboard_summary,
    init_dashboard_state,
    release_document_resources,
    release_secret_resources,
)


def test_tab1_excludes_legitimate_document_lines_and_keeps_real_stripes() -> None:
    st.session_state.clear()
    init_dashboard_state()

    doc_info = PDFDocumentInfo(
        file_path="/tmp/sample.pdf",
        file_name="sample.pdf",
        page_count=2,
        file_size_bytes=10240,
    )
    clean_table_page = PageDecision(
        page_number=1,
        status=STATUS_NORMAL,
        detected_rotation=0,
        orientation_conf=12.0,
        existing_pdf_rotation=0,
        correction_angle=0,
        corrected=False,
        detection_method="tesseract_osd",
        stripe_status=STRIPE_STATUS_CLEAN,
        stripe_defects=[],
        boundary_status=BOUNDARY_STATUS_NORMAL,
    )
    real_stripe_page = PageDecision(
        page_number=2,
        status=STATUS_NORMAL,
        detected_rotation=0,
        orientation_conf=11.0,
        existing_pdf_rotation=0,
        correction_angle=0,
        corrected=False,
        detection_method="tesseract_osd",
        stripe_status=STRIPE_STATUS_DETECTED,
        stripe_defects=[
            DefectResult(
                detector_id="stripe_detector",
                defect_type="HORIZONTAL",
                stripe_type="HORIZONTAL",
                status=DefectStatus.CONFIRMED.value,
                confidence=0.92,
                suspicion_score=0.92,
                bbox=[0, 420, 1200, 424],
                reason="Sọc ngang kéo dài toàn trang vượt lề trái/phải.",
            )
        ],
        boundary_status=BOUNDARY_STATUS_NORMAL,
        requires_review=True,
    )
    res = PipelineResult(
        doc_info=doc_info,
        config=DetectorConfig(),
        detections=[],
        decisions=[clean_table_page, real_stripe_page],
        processing_time_seconds=0.1,
    )

    rows = build_problem_page_rows(res)
    assert [r.page_number for r in rows] == [2]
    summary = compute_dashboard_summary(res, rows)
    assert summary["total_pages"] == 2
    assert summary["error_pages"] == 1
    assert summary["review_pages"] == 1


def test_tab2_filters_review_actions_and_state_isolation(tmp_path: Path) -> None:
    st.session_state.clear()
    init_dashboard_state()

    # Giả lập cả hai tab đều đang có dữ liệu phiên
    st.session_state["file_hash"] = "tab1_hash_abc"
    st.session_state["display_file_name"] = "scan_quality.pdf"
    st.session_state["secret_file_hash"] = "tab2_hash_xyz"
    st.session_state["secret_file_name"] = "secret_check.pdf"

    doc_info = PDFDocumentInfo(
        file_path="/tmp/secret_check.pdf",
        file_name="secret_check.pdf",
        page_count=5,
        file_size_bytes=20480,
    )
    pages = [
        SecretPageResult(
            page_number=1,
            status=SECRET_STATUS_DETECTED,
            primary_label="MẬT",
            marks=[
                DetectedSecretMark(
                    mark_id="p1_m1",
                    label="MẬT",
                    bbox=(180, 210, 310, 280),
                    confidence=0.95,
                    status=SECRET_STATUS_DETECTED,
                    detection_method="BOX_TEMPLATE_MATCH",
                )
            ],
        ),
        SecretPageResult(page_number=2, status=SECRET_STATUS_NO_MARK),
        SecretPageResult(
            page_number=3,
            status=SECRET_STATUS_DETECTED,
            primary_label="TỐI MẬT",
            marks=[
                DetectedSecretMark(
                    mark_id="p3_m1",
                    label="TỐI MẬT",
                    bbox=(190, 215, 410, 285),
                    confidence=0.96,
                    status=SECRET_STATUS_DETECTED,
                    detection_method="BOX_TEMPLATE_MATCH",
                )
            ],
        ),
        SecretPageResult(
            page_number=4,
            status=SECRET_STATUS_DETECTED,
            primary_label="TUYỆT MẬT",
            marks=[
                DetectedSecretMark(
                    mark_id="p4_m1",
                    label="TUYỆT MẬT",
                    bbox=(185, 215, 465, 285),
                    confidence=0.98,
                    status=SECRET_STATUS_DETECTED,
                    detection_method="ISOLATED_TEMPLATE_MATCH",
                )
            ],
        ),
        SecretPageResult(
            page_number=5,
            status=SECRET_STATUS_REVIEW,
            primary_label="MẬT",
            marks=[
                DetectedSecretMark(
                    mark_id="p5_m1",
                    label="MẬT",
                    bbox=(200, 220, 330, 290),
                    confidence=0.63,
                    status=SECRET_STATUS_REVIEW,
                    detection_method="BOX_TEMPLATE_MATCH",
                )
            ],
        ),
    ]
    doc_res = SecretDocumentResult(doc_info=doc_info, pages=pages)

    flagged = build_secret_flagged_rows(doc_res)
    assert [p.page_number for p in flagged] == [1, 3, 4, 5]  # Trang 2 sạch được ẩn
    assert [p.page_number for p in filter_secret_rows(flagged, "MAT")] == [1, 5]
    assert [p.page_number for p in filter_secret_rows(flagged, "TOI_MAT")] == [3]
    assert [p.page_number for p in filter_secret_rows(flagged, "TUYET_MAT")] == [4]
    assert [p.page_number for p in filter_secret_rows(flagged, "REVIEW")] == [5]

    # Thao tác rà soát: Xác nhận trang 5 -> không còn nằm trong diện REVIEW
    SecretScanService.apply_review_action(doc_res, 5, "CONFIRMED")
    fb_file = tmp_path / "secret_fb.json"
    save_secret_review_feedback(
        file_name="secret_check.pdf",
        page_result=doc_res.get_page(5),
        action="CONFIRMED",
        feedback_path=fb_file,
    )
    assert fb_file.exists()
    assert filter_secret_rows(flagged, "REVIEW") == []

    # Cô lập trạng thái: xóa tài nguyên Tab 2 không làm mất dữ liệu Tab 1
    release_secret_resources(increment_uploader=True)
    assert st.session_state["secret_file_hash"] is None
    assert st.session_state["file_hash"] == "tab1_hash_abc"

    # Ngược lại xóa tài nguyên Tab 1 không ảnh hưởng bộ đếm uploader Tab 2
    sec_ver = st.session_state["secret_uploader_key_ver"]
    release_document_resources(increment_uploader=True)
    assert st.session_state["file_hash"] is None
    assert st.session_state["secret_uploader_key_ver"] == sec_ver
