"""
Kiểm thử đơn vị (Unit Tests) và kiểm thử tích hợp (Integration Tests với Tesseract thực tế)
cho `ai_engine.preprocessing.image_preprocessor` và `ai_engine.detectors.tesseract_detector`.
"""

from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest

from ai_engine.core.config import (
    DETECTION_UNKNOWN,
    SAMPLES_DIR,
    STATUS_NORMAL,
    STATUS_REVIEW,
    STATUS_ROTATED,
)
from ai_engine.detectors.tesseract_detector import (
    OSDAttemptResult,
    OrientationDetector,
    TesseractOrientationDetector,
    parse_osd_text,
)
from ai_engine.preprocessing.image_preprocessor import (
    denoise_image,
    enhance_contrast_clahe,
    normalize_dimensions,
    prepare_original_grayscale,
    preprocess_for_ocr,
    to_grayscale,
)
from backend.services.orientation_service import PDFOrientationPipeline
from backend.repositories.file_utils import combine_rotation_angles, normalize_angle


def test_normalize_and_combine_angles() -> None:
    """Kiểm tra chuẩn hóa góc và cộng góc theo modulo 360."""
    assert normalize_angle(0) == 0
    assert normalize_angle(90) == 90
    assert normalize_angle(180) == 180
    assert normalize_angle(270) == 270
    assert normalize_angle(360) == 0
    assert normalize_angle(-90) == 270

    with pytest.raises(ValueError):
        normalize_angle(45)

    assert combine_rotation_angles(90, 180) == 270
    assert combine_rotation_angles(270, 90) == 0
    assert combine_rotation_angles(180, 180) == 0


def test_parse_osd_text() -> None:
    """Kiểm tra hàm phân tích chuỗi kết quả OSD từ Tesseract."""
    raw = (
        "Page number: 0\n"
        "Orientation in degrees: 90\n"
        "Rotate: 270\n"
        "Orientation confidence: 14.32\n"
        "Script: Latin\n"
        "Script confidence: 6.23\n"
    )
    parsed = parse_osd_text(raw)
    assert parsed["orientation"] == 90
    assert parsed["rotate"] == 270
    assert parsed["orientation_conf"] == pytest.approx(14.32)
    assert parsed["script"] == "Latin"
    assert parsed["script_conf"] == pytest.approx(6.23)


def test_image_preprocessing_functions() -> None:
    """Kiểm tra các bước tiền xử lý ảnh bằng OpenCV."""
    rgb = np.random.default_rng(0).integers(
        20, 230, size=(1000, 800, 3), dtype=np.uint8
    )
    gray = to_grayscale(rgb)
    assert gray.ndim == 2
    assert gray.shape == (1000, 800)

    resized = normalize_dimensions(gray, max_dim=500, min_dim=200)
    assert max(resized.shape) == 500

    denoised = denoise_image(resized, ksize=3)
    assert denoised.shape == resized.shape

    enhanced = enhance_contrast_clahe(denoised)
    assert enhanced.shape == resized.shape

    orig_prep = prepare_original_grayscale(rgb, max_dim=600, min_dim=200)
    full_prep = preprocess_for_ocr(rgb, max_dim=600, min_dim=200)
    assert orig_prep.ndim == 2
    assert full_prep.ndim == 2


def test_detector_retry_mechanism_with_mock() -> None:
    """
    Kiểm tra cơ chế retry của TesseractOrientationDetector:
    Khi lần 1 trên ảnh gốc trả về UNKNOWN, tự động thử lại lần 2 trên ảnh tiền xử lý.
    """
    detector = TesseractOrientationDetector(
        conf_threshold=10.0, retry_with_preprocessed=True
    )
    dummy_img = np.random.default_rng(1).integers(
        0, 255, size=(900, 700), dtype=np.uint8
    )

    attempt1 = OSDAttemptResult(
        method="tesseract_osd",
        status=DETECTION_UNKNOWN,
        error="Too few characters",
    )
    attempt2 = OSDAttemptResult(
        method="tesseract_osd_preprocessed",
        status="OK",
        orientation=180,
        rotate=180,
        orientation_conf=14.8,
        script="Latin",
        script_conf=8.5,
    )

    with patch.object(
        detector, "_run_single_osd", side_effect=[attempt1, attempt2]
    ) as mock_osd:
        result = detector.detect_page(dummy_img, page_number=1, existing_pdf_rotation=0)
        assert mock_osd.call_count == 2
        assert result.status == "OK"
        assert result.rotate == 180
        assert result.orientation_conf == pytest.approx(14.8)
        assert result.detection_method == "tesseract_osd_preprocessed"
        assert result.has_conflict is False


def test_detector_conflict_between_attempts_with_mock() -> None:
    """
    Khi lần 1 có confidence đáng kể (>= 50% ngưỡng, ví dụ 6.0, góc 90°) và lần 2 cho góc khác (180°),
    phải đánh dấu has_conflict=True và không cộng/trung bình điểm confidence.
    """
    detector = OrientationDetector(conf_threshold=10.0, retry_with_preprocessed=True)
    dummy_img = np.random.default_rng(2).integers(
        0, 255, size=(900, 700), dtype=np.uint8
    )

    attempt1 = OSDAttemptResult(
        method="tesseract_osd",
        status="OK",
        orientation=270,
        rotate=90,
        orientation_conf=6.0,
    )
    attempt2 = OSDAttemptResult(
        method="tesseract_osd_preprocessed",
        status="OK",
        orientation=180,
        rotate=180,
        orientation_conf=12.5,
    )

    with patch.object(detector, "_run_single_osd", side_effect=[attempt1, attempt2]):
        result = detector.detect_page(dummy_img, page_number=2, existing_pdf_rotation=0)
        assert result.has_conflict is True
        assert result.orientation_conf in (6.0, 12.5)


@pytest.mark.parametrize(
    "sample_filename",
    [
        "01_all_normal.pdf",
        "02_single_upside_down_180.pdf",
        "03_multiple_upside_down.pdf",
        "04_rotated_90.pdf",
        "05_rotated_270.pdf",
        "06_all_four_orientations.pdf",
        "07_blank_page.pdf",
        "08_very_few_chars.pdf",
        "09_noisy_blurry_scan.pdf",
        "11_metadata_rotation_nonzero.pdf",
        "12_mixed_vector_and_scan.pdf",
    ],
)
def test_integration_real_tesseract_against_ground_truth(
    sample_pdfs_fixture: Dict[str, Any],
    sample_filename: str,
) -> None:
    """
    Kiểm thử tích hợp thực tế với Tesseract OSD trên toàn bộ các file PDF mẫu hợp lệ
    và đối chiếu kết quả từng trang với ground truth.
    """
    gt_info = sample_pdfs_fixture[sample_filename]
    pdf_path = SAMPLES_DIR / sample_filename
    assert pdf_path.exists()

    pipeline = PDFOrientationPipeline()
    result = pipeline.scan_document(pdf_path)

    assert result.doc_info.page_count == gt_info["page_count"]
    assert len(result.decisions) == gt_info["page_count"]

    for page_gt, decision in zip(gt_info["pages"], result.decisions):
        expected_status = page_gt["expected_status"]
        expected_angle = page_gt["expected_correction_angle"]

        assert decision.page_number == page_gt["page_number"]
        assert decision.status == expected_status, (
            f"File {sample_filename} trang {decision.page_number}: "
            f"kỳ vọng status={expected_status}, thực tế={decision.status} (conf={decision.orientation_conf})"
        )
        assert decision.correction_angle == expected_angle, (
            f"File {sample_filename} trang {decision.page_number}: "
            f"kỳ vọng correction_angle={expected_angle}, thực tế={decision.correction_angle}"
        )
        if expected_status == STATUS_REVIEW:
            assert decision.should_rotate is False
        elif expected_status == STATUS_NORMAL:
            assert decision.detected_rotation == 0
        elif expected_status == STATUS_ROTATED:
            assert decision.detected_rotation == expected_angle
