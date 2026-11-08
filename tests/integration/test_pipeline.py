"""
Kiểm thử đơn vị và tích hợp cho kiến trúc Multi-Defect PDF Quality Inspection:
- `DetectorRegistry` (`AVAILABLE`, `ENABLED`, `DISABLED`, `NOT_IMPLEMENTED`, chống trùng lặp).
- `PageContext` và `CoordinateTransform` (dữ liệu dùng chung, bảo vệ read-only, giải phóng RAM).
- `UnifiedQualityEngine` (`DecisionEngine`) & Multi-label `DefectResult` schema.
- `DocumentPipeline` (`PDFOrientationPipeline`), `ReportGenerator` và CLI (`backend.api.cli:run_cli`).
"""

import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from ai_engine.core.config import (
    DETECTION_UNKNOWN,
    SAMPLES_DIR,
    STATUS_NORMAL,
    STATUS_REVIEW,
    STATUS_ROTATED,
)
from ai_engine.detectors.base_detector import PageLevelDetector
from ai_engine.detectors.registry import DetectorRegistry, create_default_registry
from backend.api.cli import run_cli
from backend.schemas.context import PageContext
from backend.schemas.schemas import (
    DefectResult,
    DefectStatus,
    DefectType,
    DetectorStatus,
    InspectionStatus,
    OrientationDetectionResult,
)
from ai_engine.pipelines.document_pipeline import DocumentPipeline
from backend.services.orientation_service import DecisionEngine, UnifiedQualityEngine


def test_detector_registry_lifecycle_and_unimplemented_guards() -> None:
    """
    Kiểm tra DetectorRegistry:
    - Đăng ký OrientationDetector, StripeDetector và PageBoundaryDetector ở trạng thái ENABLED (AVAILABLE).
    - Hỗ trợ ghi nhận detector mở rộng tương lai ở trạng thái NOT_IMPLEMENTED.
    - Không cho phép đăng ký trùng detector_id.
    - Không cho phép bật hoặc chạy detector NOT_IMPLEMENTED.
    - Cho phép disable / enable detector đã triển khai.
    """
    registry = create_default_registry()

    assert registry.get_status("orientation") == DetectorStatus.ENABLED.value
    assert registry.is_available("orientation") is True
    assert registry.get_status("stripe") == DetectorStatus.ENABLED.value
    assert registry.is_available("stripe") is True
    assert registry.get_status("boundary") == DetectorStatus.ENABLED.value
    assert registry.is_available("boundary") is True

    # Cả 3 detector đều nằm trong danh sách enabled_detectors được thực thi
    enabled_ids = registry.get_enabled_detector_ids()
    assert enabled_ids == ["orientation", "stripe", "boundary"]
    assert registry.get_unsupported_checks() == []

    # Kiểm tra cơ chế bảo vệ cho detector chưa triển khai (NOT_IMPLEMENTED)
    registry.register_unimplemented(
        "future_detector",
        "Future Detector",
        "PAGE_LEVEL",
        ("FUTURE_DEFECT",),
    )
    assert registry.get_status("future_detector") == DetectorStatus.NOT_IMPLEMENTED.value
    assert registry.is_available("future_detector") is False
    with pytest.raises(RuntimeError, match="NOT_IMPLEMENTED"):
        registry.enable("future_detector")

    # Ngăn đăng ký trùng detector_id
    ori_instance = registry.get_detector("orientation")
    assert ori_instance is not None
    with pytest.raises(ValueError, match="đã được đăng ký"):
        registry.register(ori_instance)

    # Cho phép tắt và bật lại detector hợp lệ
    registry.disable("stripe")
    assert registry.get_status("stripe") == DetectorStatus.DISABLED.value
    assert registry.get_enabled_detector_ids() == ["orientation", "boundary"]
    assert set(registry.get_unsupported_checks()) == {
        DefectType.STRIPE.value,
        "FUTURE_DEFECT",
    }
    registry.enable("stripe")
    assert registry.get_status("stripe") == DetectorStatus.ENABLED.value
    assert len(registry.get_enabled_detectors()) == 3


def test_shared_page_context_immutability_and_coordinate_transform() -> None:
    """
    Kiểm tra PageContext:
    - Lưu đầy đủ document_id, page_number, pdf_page_size, original_rotation, dpi, metadata.
    - Bảo vệ original_image không bị ghi đè trực tiếp (read-only).
    - Tái sử dụng grayscale_image và processed_image (lazy caching).
    - Quản lý ảnh xoay thử (get_rotated_image) và ánh xạ tọa độ pixel <-> PDF point.
    - Giải phóng bộ nhớ khi gọi release().
    """
    raw_img = np.full((800, 600), 200, dtype=np.uint8)
    ctx = PageContext(
        document_id="doc_test.pdf",
        page_number=3,
        pdf_page_size=(600.0, 800.0),
        original_rotation=90,
        original_image=raw_img,
        dpi=144,
        metadata={"author": "QA"},
    )

    assert ctx.document_id == "doc_test.pdf"
    assert ctx.page_number == 3
    assert ctx.pdf_page_size == (600.0, 800.0)
    assert ctx.original_rotation == 90
    assert ctx.existing_rotation == 90
    assert ctx.dpi == 144

    # Không cho phép biến đổi trực tiếp original_image
    with pytest.raises(ValueError):
        ctx.original_image[0, 0] = 0

    # Lazy cache: truy cập nhiều lần trả về cùng một đối tượng ndarray trong bộ nhớ
    g1 = ctx.grayscale_image
    g2 = ctx.grayscale_image
    assert g1 is g2

    p1 = ctx.processed_image
    p2 = ctx.processed_image
    assert p1 is p2

    # Ảnh xoay 90° có kích thước hoán đổi (600, 800) và không làm thay đổi ảnh gốc
    rot90 = ctx.get_rotated_image(90)
    assert rot90.shape == (600, 800)
    assert ctx.original_image.shape == (800, 600)

    # Kiểm tra quy đổi tọa độ
    pt_x, pt_y = ctx.coordinate_transform.pixel_to_pdf_point(300, 400)
    assert pt_x == pytest.approx(300.0)
    assert pt_y == pytest.approx(400.0)

    # Giải phóng RAM
    ctx.release()
    with pytest.raises(RuntimeError, match="đã được giải phóng"):
        _ = ctx.original_image


def test_multi_label_defects_and_failed_detector_handling() -> None:
    """
    Kiểm tra UnifiedQualityEngine và DocumentPipeline:
    - Một trang có thể đồng thời chứa nhiều lỗi khác nhau (Multi-label Defect Detection).
    - Một detector thất bại (DETECTOR_FAILED) sẽ đặt requires_review=True và inspection_status=PARTIAL_FAILURE,
      tuyệt đối không tự coi là không có lỗi (NO_DEFECT).
    - Trang không có lỗi Orientation vẫn ghi nhận STRIPE và BOUNDARY là NOT_CHECKED.
    """
    engine = UnifiedQualityEngine(conf_threshold=10.0, auto_fix=True)
    det_rot = OrientationDetectionResult(
        page_number=1,
        status="OK",
        orientation=180,
        rotate=180,
        orientation_conf=19.5,
        existing_pdf_rotation=0,
        detection_method="tesseract_osd",
    )
    failing_second_defect = DefectResult(
        defect_type="CUSTOM_CHECK",
        detector_id="custom_detector",
        status=DefectStatus.DETECTOR_FAILED.value,
        requires_review=True,
        error="Custom detector timeout",
    )

    decision = engine.evaluate_page(
        det_rot, additional_defects=[failing_second_defect]
    )
    assert len(decision.defects) == 2
    assert {d.defect_type for d in decision.defects} == {"ROTATION", "CUSTOM_CHECK"}
    assert decision.requires_review is True
    assert decision.inspection_status == InspectionStatus.PARTIAL_FAILURE.value
    assert decision.detector_outcomes["ROTATION"] == DefectStatus.CONFIRMED.value
    assert (
        decision.detector_outcomes["CUSTOM_CHECK"] == DefectStatus.DETECTOR_FAILED.value
    )
    assert decision.detector_outcomes["STRIPE"] == DefectStatus.NOT_CHECKED.value
    assert decision.detector_outcomes["BOUNDARY"] == DefectStatus.NOT_CHECKED.value


def test_pipeline_executes_registered_custom_page_detector() -> None:
    """
    Kiểm tra DocumentPipeline tự động gọi mọi Page-level Detector đang ENABLED trên cùng PageContext.
    """
    observed_contexts: List[Tuple[int, bool]] = []

    class DummyAuxDetector(PageLevelDetector):
        @property
        def detector_id(self) -> str:
            return "aux_test"

        @property
        def supported_defect_types(self) -> Tuple[str, ...]:
            return ("AUX_DEFECT",)

        def detect(self, context: PageContext) -> List[DefectResult]:
            # Kiểm tra grayscale_image đã được tạo và có thể tái sử dụng ngay trên PageContext
            has_gray = context.grayscale_image is not None and context.grayscale_image.size > 0
            observed_contexts.append((context.page_number, has_gray))
            return [
                DefectResult(
                    defect_type="AUX_DEFECT",
                    detector_id=self.detector_id,
                    status=DefectStatus.NO_DEFECT.value,
                    requires_review=False,
                )
            ]

    registry = create_default_registry()
    registry.register(DummyAuxDetector(), enabled=True)

    pipeline = DocumentPipeline(registry=registry)
    sample_pdf = SAMPLES_DIR / "01_all_normal.pdf"
    result = pipeline.scan_document(sample_pdf)

    assert len(observed_contexts) == 3
    assert all(has_gray for _, has_gray in observed_contexts)
    assert "aux_test" in result.enabled_detectors


def test_decision_normal_page_above_threshold() -> None:
    """Trang có góc xoay 0° và độ tin cậy >= ngưỡng phải được phân loại là NORMAL."""
    engine = DecisionEngine(conf_threshold=10.0, auto_fix=True)
    det = OrientationDetectionResult(
        page_number=1,
        status="OK",
        orientation=0,
        rotate=0,
        orientation_conf=18.5,
        existing_pdf_rotation=0,
        detection_method="tesseract_osd",
    )
    decision = engine.evaluate_page(det)

    assert decision.status == STATUS_NORMAL
    assert decision.detected_rotation == 0
    assert decision.correction_angle == 0
    assert decision.should_rotate is False
    assert decision.corrected is False
    assert decision.error is None
    assert decision.requires_review is False
    assert decision.inspection_status == InspectionStatus.COMPLETED.value
    assert decision.defects == []
    assert decision.detector_outcomes["ROTATION"] == DefectStatus.NO_DEFECT.value
    assert decision.detector_outcomes["STRIPE"] == DefectStatus.NOT_CHECKED.value
    assert decision.detector_outcomes["BOUNDARY"] == DefectStatus.NOT_CHECKED.value


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_decision_rotated_page_auto_fix_enabled(angle: int) -> None:
    """Trang có góc xoay 90°, 180°, 270° và confidence >= ngưỡng khi bật auto_fix phải được đánh dấu xoay."""
    engine = DecisionEngine(conf_threshold=10.0, auto_fix=True)
    det = OrientationDetectionResult(
        page_number=2,
        status="OK",
        orientation=(360 - angle) % 360,
        rotate=angle,
        orientation_conf=15.2,
        existing_pdf_rotation=0,
        detection_method="tesseract_osd",
    )
    decision = engine.evaluate_page(det)

    assert decision.status == STATUS_ROTATED
    assert decision.detected_rotation == angle
    assert decision.correction_angle == angle
    assert decision.should_rotate is True
    assert len(decision.defects) == 1
    assert decision.defects[0].defect_type == DefectType.ROTATION.value
    assert decision.defects[0].status == DefectStatus.CONFIRMED.value
    assert decision.defects[0].correction_angle == angle


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_decision_rotated_page_auto_fix_disabled(angle: int) -> None:
    """Khi tắt auto_fix, trang ROTATED vẫn giữ thông tin góc cần xoay nhưng should_rotate=False."""
    engine = DecisionEngine(conf_threshold=10.0, auto_fix=False)
    det = OrientationDetectionResult(
        page_number=3,
        status="OK",
        orientation=(360 - angle) % 360,
        rotate=angle,
        orientation_conf=22.0,
        existing_pdf_rotation=0,
        detection_method="tesseract_osd",
    )
    decision = engine.evaluate_page(det)

    assert decision.status == STATUS_ROTATED
    assert decision.correction_angle == angle
    assert decision.should_rotate is False


def test_decision_review_on_low_confidence_never_rotates() -> None:
    """Trang có confidence thấp hơn ngưỡng phải vào REVIEW và tuyệt đối không tự động xoay."""
    engine = DecisionEngine(conf_threshold=10.0, auto_fix=True)
    det = OrientationDetectionResult(
        page_number=4,
        status="OK",
        orientation=180,
        rotate=180,
        orientation_conf=4.5,
        existing_pdf_rotation=0,
        detection_method="tesseract_osd",
    )
    decision = engine.evaluate_page(det)

    assert decision.status == STATUS_REVIEW
    assert decision.requires_review is True
    assert decision.inspection_status == InspectionStatus.COMPLETED_WITH_REVIEW.value
    assert decision.detected_rotation == 180
    assert decision.correction_angle == 0
    assert decision.should_rotate is False
    assert decision.error is not None
    assert "thấp hơn ngưỡng" in decision.error
    assert decision.defects[0].status == DefectStatus.SUSPECTED_LOW_CONFIDENCE.value


def test_decision_review_on_unknown_osd_never_rotates() -> None:
    """Khi OCR trả về UNKNOWN, phải phân loại là REVIEW (DETECTOR_FAILED), không bao giờ tự suy đoán là NORMAL."""
    engine = DecisionEngine(conf_threshold=10.0, auto_fix=True)
    det = OrientationDetectionResult(
        page_number=5,
        status=DETECTION_UNKNOWN,
        orientation=None,
        rotate=None,
        orientation_conf=None,
        existing_pdf_rotation=0,
        detection_method="tesseract_osd",
        error="Tesseract OSD không thể nhận diện: Too few characters",
    )
    decision = engine.evaluate_page(det)

    assert decision.status == STATUS_REVIEW
    assert decision.requires_review is True
    assert decision.inspection_status == InspectionStatus.PARTIAL_FAILURE.value
    assert decision.detected_rotation is None
    assert decision.correction_angle == 0
    assert decision.should_rotate is False
    assert decision.error is not None
    assert decision.defects[0].status == DefectStatus.DETECTOR_FAILED.value


def test_decision_review_on_conflicting_attempts() -> None:
    """Khi các lần thử cho kết quả mâu thuẫn (has_conflict=True), phải đưa vào REVIEW dù confidence cao."""
    engine = DecisionEngine(conf_threshold=10.0, auto_fix=True)
    det = OrientationDetectionResult(
        page_number=6,
        status="OK",
        orientation=180,
        rotate=180,
        orientation_conf=19.0,
        existing_pdf_rotation=0,
        detection_method="tesseract_osd_preprocessed",
        has_conflict=True,
        error="Mâu thuẫn kết quả giữa ảnh gốc (90°) và ảnh tiền xử lý (180°).",
    )
    decision = engine.evaluate_page(det)

    assert decision.status == STATUS_REVIEW
    assert decision.requires_review is True
    assert decision.correction_angle == 0
    assert decision.should_rotate is False


def test_decision_manual_override_on_review_page() -> None:
    """Người dùng có thể chủ động chọn góc xoay thủ công cho trang REVIEW."""
    engine = DecisionEngine(conf_threshold=10.0, auto_fix=False)
    det = OrientationDetectionResult(
        page_number=7,
        status=DETECTION_UNKNOWN,
        orientation=None,
        rotate=None,
        orientation_conf=None,
        existing_pdf_rotation=90,
        detection_method="tesseract_osd",
        error="Too few characters",
    )
    decision = engine.evaluate_page(det, manual_angle=180)

    assert decision.status == STATUS_REVIEW
    assert decision.manual_override is True
    assert decision.correction_angle == 180
    assert decision.should_rotate is True
    assert decision.target_pdf_rotation == (90 + 180) % 360 == 270


def test_cli_and_report_consistency(tmp_path: Path) -> None:
    """
    Kiểm tra chạy công cụ CLI (`backend.api.cli:run_cli`) end-to-end và đối chiếu tính nhất quán
    giữa kết quả xử lý, báo cáo JSON (bao gồm cả schema Unified Quality Result) và báo cáo CSV.
    """
    input_pdf = SAMPLES_DIR / "03_multiple_upside_down.pdf"
    output_pdf = tmp_path / "cli_corrected.pdf"
    json_report = tmp_path / "cli_report.json"
    csv_report = tmp_path / "cli_report.csv"

    exit_code = run_cli(
        [
            str(input_pdf),
            "--dpi",
            "160",
            "--threshold",
            "10",
            "--auto-fix",
            "--output",
            str(output_pdf),
            "--report",
            str(json_report),
            "--csv",
            str(csv_report),
        ]
    )
    assert exit_code == 0
    assert output_pdf.exists()
    assert json_report.exists()
    assert csv_report.exists()

    data = json.loads(json_report.read_text(encoding="utf-8"))
    assert data["input_file"] == "03_multiple_upside_down.pdf"
    assert data["total_pages"] == 4
    assert data["normal_pages"] == 2
    assert data["rotated_pages"] == 2
    assert data["review_pages"] == 0
    assert data["corrected_pages"] == 2
    assert data["enabled_detectors"] == ["orientation", "stripe", "boundary"]
    assert set(data["unsupported_checks"]) == set()
    assert len(data["pages"]) == 4

    required_page_keys = {
        "page_number",
        "status",
        "inspection_status",
        "requires_review",
        "detected_rotation",
        "orientation_conf",
        "existing_pdf_rotation",
        "correction_angle",
        "corrected",
        "detection_method",
        "error",
        "defects",
        "detector_outcomes",
        "enabled_detectors",
        "unsupported_checks",
    }
    for page_entry in data["pages"]:
        assert required_page_keys.issubset(page_entry.keys())

    with csv_report.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert rows[0]["status"] == "ROTATED"
    assert rows[0]["corrected"] == "True"
    assert rows[1]["status"] == "NORMAL"
    assert rows[1]["corrected"] == "False"
