"""
Kiểm thử cho `pdf_inspector.services.correction_service` (`PDFCorrector`).
"""

import hashlib
from pathlib import Path

import pytest

from pdf_inspector.core.config import SAMPLES_DIR, STATUS_NORMAL, STATUS_REVIEW, DetectorConfig
from pdf_inspector.core.exceptions import PDFCorrectionError
from pdf_inspector.models.schemas import PageDecision
from pdf_inspector.services.correction_service import PDFCorrector
from pdf_inspector.services.orientation_service import PDFOrientationPipeline
from pdf_inspector.services.pdf_service import PDFLoader


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pdf_corrector_refuses_to_overwrite_input() -> None:
    """Nguyên tắc an toàn: PDFCorrector không bao giờ được ghi đè lên chính file đầu vào."""
    sample_pdf = SAMPLES_DIR / "02_single_upside_down_180.pdf"
    corrector = PDFCorrector()
    decisions = [
        PageDecision(
            page_number=1,
            status=STATUS_NORMAL,
            detected_rotation=0,
            orientation_conf=15.0,
            existing_pdf_rotation=0,
            correction_angle=0,
            corrected=False,
            detection_method="tesseract_osd",
        )
    ]
    with pytest.raises(PDFCorrectionError, match="ghi đè"):
        corrector.apply_corrections(
            input_pdf_path=sample_pdf,
            output_pdf_path=sample_pdf,
            decisions=decisions,
            auto_fix=True,
        )


def test_end_to_end_correction_preserves_original_and_fixes_all_orientations(
    tmp_path: Path,
) -> None:
    """
    Kiểm tra trên `06_all_four_orientations.pdf` (có đủ 4 hướng 0°, 90°, 180°, 270°):
    - File gốc không bị thay đổi (SHA-256 giữ nguyên).
    - Số trang của file xuất giữ nguyên (4 trang).
    - File đầu ra mở được bình thường.
    - Khi quét lại file `corrected.pdf`, cả 4 trang đều trở về trạng thái NORMAL (0°).
    """
    input_pdf = SAMPLES_DIR / "06_all_four_orientations.pdf"
    original_hash = _file_sha256(input_pdf)

    output_pdf = tmp_path / "corrected_4_orientations.pdf"
    json_report = tmp_path / "report.json"
    csv_report = tmp_path / "report.csv"

    cfg = DetectorConfig(render_dpi=160, orientation_conf_threshold=10.0, auto_fix=True)
    pipeline = PDFOrientationPipeline(config=cfg)
    result = pipeline.scan_document(input_pdf)
    result = pipeline.update_decisions_and_artifacts(
        result=result,
        output_pdf_path=output_pdf,
        json_report_path=json_report,
        csv_report_path=csv_report,
        write_corrected_pdf=True,
    )

    assert _file_sha256(input_pdf) == original_hash
    assert output_pdf.exists()
    with PDFLoader(output_pdf) as out_loader:
        assert out_loader.page_count == 4

    assert [d.corrected for d in result.decisions] == [False, True, True, True]
    assert result.report_data["summary"]["corrected_pages"] == 3

    verify_result = pipeline.scan_document(output_pdf)
    for dec in verify_result.decisions:
        assert dec.status == STATUS_NORMAL
        assert dec.detected_rotation == 0


def test_metadata_rotation_modulo_360_correction(tmp_path: Path) -> None:
    """
    Case 11 (`11_metadata_rotation_nonzero.pdf`):
    Trang 1 có existing_pdf_rotation=180, correction=180 -> sau khi sửa rotation metadata == 0.
    Trang 2 có existing_pdf_rotation=90, correction=270 -> sau khi sửa rotation metadata == 0.
    """
    input_pdf = SAMPLES_DIR / "11_metadata_rotation_nonzero.pdf"
    output_pdf = tmp_path / "corrected_metadata.pdf"

    cfg = DetectorConfig(render_dpi=160, orientation_conf_threshold=10.0, auto_fix=True)
    pipeline = PDFOrientationPipeline(config=cfg)
    result = pipeline.scan_document(input_pdf)
    pipeline.update_decisions_and_artifacts(
        result=result,
        output_pdf_path=output_pdf,
        write_corrected_pdf=True,
    )

    with PDFLoader(output_pdf) as loader:
        assert loader.get_page_rotation(1) == 0
        assert loader.get_page_rotation(2) == 0


def test_review_pages_are_never_auto_rotated(tmp_path: Path) -> None:
    """
    Kiểm tra trên `07_blank_page.pdf` và `08_very_few_chars.pdf`:
    Khi bật auto_fix=True, các trang REVIEW tuyệt đối không bị xoay nhầm.
    """
    for filename in ("07_blank_page.pdf", "08_very_few_chars.pdf"):
        input_pdf = SAMPLES_DIR / filename
        output_pdf = tmp_path / f"corrected_{filename}"

        cfg = DetectorConfig(
            render_dpi=160, orientation_conf_threshold=10.0, auto_fix=True
        )
        pipeline = PDFOrientationPipeline(config=cfg)
        result = pipeline.scan_document(input_pdf)
        pipeline.update_decisions_and_artifacts(
            result=result,
            output_pdf_path=output_pdf,
            write_corrected_pdf=True,
        )

        review_decision = result.decisions[1]
        assert review_decision.status == STATUS_REVIEW
        assert review_decision.corrected is False
        assert review_decision.correction_angle == 0

        with PDFLoader(output_pdf) as loader:
            assert loader.get_page_rotation(2) == 0

