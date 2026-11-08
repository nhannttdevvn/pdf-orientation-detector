"""
DocumentPipeline — Thành phần điều phối trung tâm của kiến trúc Multi-Defect PDF Quality Inspection
(`src/pdf_inspector/pipeline/document_pipeline.py`).

Trách nhiệm:
1. Tiếp nhận đường dẫn PDF đầu vào.
2. Kiểm tra tính hợp lệ của file và đọc metadata thông qua `PDFLoader`.
3. Render tuần tự từng trang và khởi tạo `PageContext` dùng chung.
4. Chuyển `PageContext` đến tất cả các `Page-level Detector` đang ở trạng thái `ENABLED` trong `DetectorRegistry`.
5. Cung cấp điểm mở rộng (`_run_cross_page_detectors`) cho các `Cross-page / Document-level Detector` tương lai.
6. Thu thập kết quả `DefectResult` và tổng hợp qua `UnifiedQualityEngine` (`DecisionEngine`).
7. Giải phóng bộ nhớ ảnh của từng trang ngay sau khi xử lý xong (`page_ctx.release()`).
8. Tạo `Quality Report` (`ReportGenerator`) và chuyển các trang hợp lệ sang `Correction` (`PDFCorrector`).
9. Trả về `PipelineResult` thống nhất cho cả CLI và Streamlit UI.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Union

from ai_engine.core.config import DETECTION_UNKNOWN, DetectorConfig
from ai_engine.core.logging_config import setup_logger
from ai_engine.detectors.base_detector import BaseDetector, BaseOrientationDetector
from ai_engine.detectors.boundary.boundary_detector import PageBoundaryDetector
from ai_engine.detectors.defects.stripe_detector import StripeDetector, StripePageResult
from ai_engine.detectors.registry import DetectorRegistry, create_default_registry
from backend.schemas.boundary_schemas import PageLayoutAnalysisResult
from backend.schemas.context import CoordinateTransform, PageContext
from backend.schemas.schemas import (
    DefectResult,
    DefectStatus,
    DefectType,
    DetectorScope,
    OrientationDetectionResult,
    PageDecision,
    PipelineResult,
    ProgressCallback,
)
from ai_engine.preprocessing.pdf_loader import PDFLoader
from backend.services.correction_service import PDFCorrector
from backend.services.orientation_service import UnifiedQualityEngine
from backend.services.report_service import ReportGenerator

logger = setup_logger(__name__)


class DocumentPipeline:
    """
    Pipeline điều phối kiểm tra chất lượng tài liệu PDF đa lỗi (Multi-Defect PDF Quality Inspection):
    PDF Input -> PDFLoader -> Shared PageContext -> DetectorRegistry (Enabled Detectors)
    -> UnifiedQualityEngine -> Quality Report -> Supported Correction -> PDF Output.
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        detector: Optional[BaseDetector] = None,
        registry: Optional[DetectorRegistry] = None,
    ) -> None:
        self.config: DetectorConfig = config or DetectorConfig()
        if registry is not None:
            self.registry: DetectorRegistry = registry
        else:
            self.registry = create_default_registry(
                config=self.config,
                orientation_detector=detector,
            )

    def _execute_page_detectors(
        self,
        page_ctx: PageContext,
        enabled_page_detectors: List[BaseDetector],
        stripe_page_results: Optional[Dict[int, StripePageResult]] = None,
    ) -> tuple[OrientationDetectionResult, List[DefectResult]]:
        """
        Chạy toàn bộ các Page-level Detector đang bật (`ENABLED`) trên cùng một `PageContext`.
        Trả về `(orientation_detection_result, additional_defects)`.
        Nếu một detector gặp ngoại lệ, lỗi được ghi nhận thành `DefectResult(status=DETECTOR_FAILED)`
        chứ tuyệt đối không coi là không có lỗi (`NO_DEFECT`).
        """
        orientation_result: Optional[OrientationDetectionResult] = None
        additional_defects: List[DefectResult] = []

        for det in enabled_page_detectors:
            try:
                if isinstance(det, BaseOrientationDetector):
                    defects = det.detect(page_ctx)
                    if defects and "orientation_detection" in defects[0].details:
                        orientation_result = defects[0].details["orientation_detection"]
                    else:
                        orientation_result = det.detect_page(
                            image=page_ctx.original_image,
                            page_number=page_ctx.page_number,
                            existing_pdf_rotation=page_ctx.original_rotation,
                            conf_threshold=self.config.orientation_conf_threshold,
                        )
                elif isinstance(det, StripeDetector):
                    try:
                        clahe_img = page_ctx.ensure_clahe()
                    except Exception:
                        clahe_img = None
                    s_res = det.detect_page(
                        image=page_ctx.original_image,
                        page_number=page_ctx.page_number,
                        dpi=page_ctx.dpi,
                        clahe_image=clahe_img,
                    )
                    if stripe_page_results is not None:
                        stripe_page_results[page_ctx.page_number] = s_res
                    defects = det.build_defect_results(
                        page_result=s_res,
                        dpi=page_ctx.dpi,
                        coordinate_transform=page_ctx.coordinate_transform,
                    )
                    additional_defects.extend(defects)
                else:
                    defects = det.detect(page_ctx)
                    additional_defects.extend(defects)
            except Exception as exc:
                err_msg = (
                    f"Detector '{det.detector_id}' gặp lỗi trên trang {page_ctx.page_number}: {exc}"
                )
                logger.error(err_msg)
                if det.detector_id == "orientation":
                    orientation_result = OrientationDetectionResult(
                        page_number=page_ctx.page_number,
                        status=DETECTION_UNKNOWN,
                        existing_pdf_rotation=page_ctx.original_rotation,
                        error=err_msg,
                    )
                else:
                    for dtype in det.supported_defect_types:
                        additional_defects.append(
                            DefectResult(
                                defect_type=dtype,
                                detector_id=det.detector_id,
                                status=DefectStatus.DETECTOR_FAILED.value,
                                requires_review=True,
                                error=err_msg,
                            )
                        )

        if orientation_result is None:
            orientation_result = OrientationDetectionResult(
                page_number=page_ctx.page_number,
                status=DETECTION_UNKNOWN,
                existing_pdf_rotation=page_ctx.original_rotation,
                error="Orientation Detector không được bật trong DetectorRegistry.",
            )

        return orientation_result, additional_defects

    def _run_cross_page_detectors(
        self,
        enabled_cross_page_detectors: List[BaseDetector],
        page_summaries: List[Dict[str, object]],
        additional_defects_by_page: Dict[int, List[DefectResult]],
        boundary_layouts_by_page: Optional[Dict[int, PageLayoutAnalysisResult]] = None,
        page_transforms: Optional[Dict[int, CoordinateTransform]] = None,
        boundary_errors_by_page: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Thực thi các `Cross-page / Document-level Detector` đang ở trạng thái `ENABLED`
        (bao gồm `PageBoundaryDetector` và bất kỳ custom Cross-page Detector nào khác).
        """
        if not enabled_cross_page_detectors:
            return

        for det in enabled_cross_page_detectors:
            try:
                if isinstance(det, PageBoundaryDetector):
                    _, b_defects_map = det.evaluate_document_boundaries(
                        layouts_by_page=boundary_layouts_by_page or {},
                        transforms_by_page=page_transforms or {},
                        page_errors=boundary_errors_by_page or {},
                    )
                    for pno, b_defs in b_defects_map.items():
                        existing = [
                            d
                            for d in additional_defects_by_page.get(pno, [])
                            if d.detector_id not in ("boundary", "page_boundary")
                        ]
                        additional_defects_by_page[pno] = existing + b_defs
                else:
                    cross_defects = det.detect(page_summaries)
                    for defect in cross_defects:
                        page_num = int(defect.details.get("page_number", 1))
                        additional_defects_by_page.setdefault(page_num, []).append(defect)
            except Exception as exc:
                logger.error("Cross-page detector '%s' gặp lỗi: %s", det.detector_id, exc)
                for p_summary in page_summaries:
                    pno = int(p_summary.get("page_number", 1))  # type: ignore[arg-type]
                    additional_defects_by_page.setdefault(pno, []).append(
                        DefectResult(
                            defect_type=DefectType.BOUNDARY.value,
                            detector_id=det.detector_id,
                            status=DefectStatus.DETECTOR_FAILED.value,
                            requires_review=True,
                            error=str(exc),
                        )
                    )

    def scan_document(
        self,
        pdf_path: Union[str, Path],
        progress_callback: Optional[ProgressCallback] = None,
        manual_overrides: Optional[Dict[int, int]] = None,
    ) -> PipelineResult:
        """
        Quét tuần tự toàn bộ các trang của file PDF qua `DetectorRegistry`,
        tái sử dụng `PageContext` giữa các detector và giải phóng ảnh từng trang ngay sau khi xử lý.
        """
        start_time = time.perf_counter()

        enabled_page_detectors = self.registry.get_enabled_detectors(
            scope=DetectorScope.PAGE_LEVEL.value
        )
        enabled_cross_detectors = self.registry.get_enabled_detectors(
            scope=DetectorScope.CROSS_PAGE.value
        )
        enabled_ids = self.registry.get_enabled_detector_ids()
        unsupported_checks = self.registry.get_unsupported_checks()
        registry_info = self.registry.to_summary_list()

        # Cập nhật cấu hình threshold cho OrientationDetector, StripeDetector và PageBoundaryDetector
        active_stripe_detector: Optional[StripeDetector] = None
        active_boundary_detector: Optional[PageBoundaryDetector] = None
        for det in enabled_page_detectors:
            if isinstance(det, BaseOrientationDetector):
                det.conf_threshold = self.config.orientation_conf_threshold
                if hasattr(det, "retry_with_preprocessed"):
                    det.retry_with_preprocessed = self.config.retry_with_preprocessed
            elif isinstance(det, StripeDetector):
                det.confirm_threshold = self.config.stripe_confirm_threshold
                det.review_threshold = self.config.stripe_review_threshold
                det.min_relative_length = self.config.stripe_min_relative_length
                det.max_thickness_px = self.config.stripe_max_thickness_px
                det.min_contrast = self.config.stripe_min_contrast
                active_stripe_detector = det

        for det in enabled_cross_detectors:
            if isinstance(det, PageBoundaryDetector):
                det.confirm_threshold = self.config.boundary_confirm_threshold
                det.review_threshold = self.config.boundary_review_threshold
                det.overlap_threshold = self.config.boundary_overlap_threshold
                det.use_optional_layout_model = (
                    self.config.boundary_use_optional_layout_model
                )
                active_boundary_detector = det

        quality_engine = UnifiedQualityEngine(
            conf_threshold=self.config.orientation_conf_threshold,
            auto_fix=self.config.auto_fix,
            enabled_detectors=enabled_ids,
            unsupported_checks=unsupported_checks,
        )

        detections: List[OrientationDetectionResult] = []
        decisions: List[PageDecision] = []
        additional_defects_by_page: Dict[int, List[DefectResult]] = {}
        stripe_page_results: Dict[int, StripePageResult] = {}
        boundary_layouts_by_page: Dict[int, PageLayoutAnalysisResult] = {}
        boundary_errors_by_page: Dict[int, str] = {}
        page_transforms: Dict[int, CoordinateTransform] = {}
        page_summaries: List[Dict[str, object]] = []
        warnings: List[str] = []
        system_errors: List[str] = []
        overrides = manual_overrides or {}

        with PDFLoader(pdf_path) as loader:
            doc_info = loader.get_document_info()
            if doc_info.has_digital_signature:
                warnings.append(
                    "CẢNH BÁO CHỮ KÝ SỐ: File PDF gốc có chứa chữ ký số (Digital Signature). "
                    "Việc lưu bản PDF đã xoay trang có thể khiến chữ ký số trên file mới không còn hợp lệ. "
                    "File PDF gốc vẫn được giữ nguyên vẹn."
                )

            total_pages = doc_info.page_count
            for idx in range(total_pages):
                page_num = idx + 1
                existing_rot = doc_info.page_rotations.get(page_num, 0)
                page_ctx: Optional[PageContext] = None
                try:
                    page_ctx = loader.build_page_context(
                        page_number=page_num,
                        dpi=self.config.render_dpi,
                        grayscale=True,
                        max_ocr_dimension=self.config.max_ocr_dimension,
                        min_ocr_dimension=self.config.min_ocr_dimension,
                        median_blur_ksize=self.config.median_blur_ksize,
                    )
                    page_summaries.append(
                        {
                            "page_number": page_num,
                            "pdf_page_size": page_ctx.pdf_page_size,
                            "original_rotation": page_ctx.original_rotation,
                        }
                    )
                    page_transforms[page_num] = page_ctx.coordinate_transform
                    det_result, extra_defects = self._execute_page_detectors(
                        page_ctx=page_ctx,
                        enabled_page_detectors=enabled_page_detectors,
                        stripe_page_results=stripe_page_results,
                    )
                    if extra_defects:
                        additional_defects_by_page[page_num] = extra_defects

                    if active_boundary_detector is not None:
                        try:
                            rec_rot = (
                                int(det_result.rotate)
                                if (
                                    det_result.status == "OK"
                                    and det_result.rotate in (90, 180, 270)
                                    and (det_result.orientation_conf or 0.0)
                                    >= self.config.orientation_conf_threshold
                                )
                                else 0
                            )
                            boundary_layouts_by_page[page_num] = (
                                active_boundary_detector.extract_page_layout(
                                    page_ctx=page_ctx,
                                    recommended_rotation_cw=rec_rot,
                                )
                            )
                        except Exception as b_exc:
                            b_err = f"Boundary layout analysis thất bại trên trang {page_num}: {b_exc}"
                            logger.error(b_err)
                            boundary_errors_by_page[page_num] = b_err
                except Exception as exc:
                    err_msg = f"Lỗi khi đọc/kiểm tra trang {page_num}: {exc}"
                    logger.error(err_msg)
                    system_errors.append(err_msg)
                    boundary_errors_by_page[page_num] = err_msg
                    det_result = OrientationDetectionResult(
                        page_number=page_num,
                        status=DETECTION_UNKNOWN,
                        existing_pdf_rotation=existing_rot,
                        error=err_msg,
                    )
                    extra_defects = []
                finally:
                    if page_ctx is not None:
                        page_ctx.release()
                        del page_ctx

                detections.append(det_result)
                decision = quality_engine.evaluate_page(
                    det_result,
                    manual_angle=overrides.get(page_num),
                    additional_defects=additional_defects_by_page.get(page_num),
                )
                decisions.append(decision)

                if progress_callback is not None:
                    progress_callback(page_num, total_pages, decision)

        # Áp dụng bằng chứng lặp lại vị trí sọc liên trang (Repeated-position evidence across pages)
        if active_stripe_detector is not None and len(stripe_page_results) >= 2:
            active_stripe_detector.apply_cross_page_repetition_evidence(
                stripe_page_results
            )
            for pno, s_res in stripe_page_results.items():
                non_stripe_defs = [
                    d
                    for d in additional_defects_by_page.get(pno, [])
                    if d.detector_id != "stripe"
                ]
                updated_stripe_defs = active_stripe_detector.build_defect_results(
                    page_result=s_res,
                    dpi=self.config.render_dpi,
                    coordinate_transform=page_transforms.get(pno),
                )
                additional_defects_by_page[pno] = non_stripe_defs + updated_stripe_defs
            decisions = quality_engine.evaluate_all(
                detections=detections,
                manual_overrides=overrides,
                additional_defects_by_page=additional_defects_by_page,
            )

        # Thực thi các Cross-page Detectors (PageBoundaryDetector + custom Cross-page Detectors)
        if enabled_cross_detectors:
            self._run_cross_page_detectors(
                enabled_cross_page_detectors=enabled_cross_detectors,
                page_summaries=page_summaries,
                additional_defects_by_page=additional_defects_by_page,
                boundary_layouts_by_page=boundary_layouts_by_page,
                page_transforms=page_transforms,
                boundary_errors_by_page=boundary_errors_by_page,
            )
            decisions = quality_engine.evaluate_all(
                detections=detections,
                manual_overrides=overrides,
                additional_defects_by_page=additional_defects_by_page,
            )

        elapsed = time.perf_counter() - start_time
        report_data = ReportGenerator.build_report_data(
            input_file_name=doc_info.file_name,
            decisions=decisions,
            processing_time_seconds=elapsed,
            config_used=self.config,
            input_path=doc_info.file_path,
            output_path=None,
            has_digital_signature=doc_info.has_digital_signature,
            warnings=warnings,
            system_errors=system_errors,
            enabled_detectors=enabled_ids,
            unsupported_checks=unsupported_checks,
            detector_registry_info=registry_info,
        )

        return PipelineResult(
            doc_info=doc_info,
            config=self.config,
            detections=detections,
            decisions=decisions,
            processing_time_seconds=elapsed,
            report_data=report_data,
            warnings=warnings,
            system_errors=system_errors,
            enabled_detectors=enabled_ids,
            unsupported_checks=unsupported_checks,
            detector_registry_info=registry_info,
            additional_defects_by_page=additional_defects_by_page,
        )

    # Bí danh ngữ nghĩa cho kiến trúc Quality Inspection
    inspect_document = scan_document

    @staticmethod
    def update_decisions_and_artifacts(
        result: PipelineResult,
        conf_threshold: Optional[float] = None,
        auto_fix: Optional[bool] = None,
        manual_overrides: Optional[Dict[int, int]] = None,
        output_pdf_path: Optional[Union[str, Path]] = None,
        json_report_path: Optional[Union[str, Path]] = None,
        csv_report_path: Optional[Union[str, Path]] = None,
        write_corrected_pdf: bool = False,
    ) -> PipelineResult:
        """
        Cập nhật lại kết quả phân loại (`UnifiedQualityEngine`), file PDF đã sửa (`PDFCorrector`)
        và báo cáo (`ReportGenerator`) từ kết quả nhận diện đã lưu mà KHÔNG chạy lại Detector.
        """
        if conf_threshold is not None:
            result.config.orientation_conf_threshold = float(conf_threshold)
        if auto_fix is not None:
            result.config.auto_fix = bool(auto_fix)

        engine = UnifiedQualityEngine(
            conf_threshold=result.config.orientation_conf_threshold,
            auto_fix=result.config.auto_fix,
            enabled_detectors=result.enabled_detectors,
            unsupported_checks=result.unsupported_checks,
        )
        result.decisions = engine.evaluate_all(
            result.detections,
            manual_overrides=manual_overrides,
            additional_defects_by_page=result.additional_defects_by_page,
        )

        has_manual = bool(
            manual_overrides and any(v % 360 != 0 for v in manual_overrides.values())
        )
        should_export_pdf = write_corrected_pdf or result.config.auto_fix or has_manual

        if should_export_pdf and output_pdf_path is not None:
            corrector = PDFCorrector()
            summary = corrector.apply_corrections(
                input_pdf_path=result.doc_info.file_path,
                output_pdf_path=output_pdf_path,
                decisions=result.decisions,
                auto_fix=result.config.auto_fix,
            )
            result.output_pdf_path = summary.output_path
            for w in summary.warnings:
                if w not in result.warnings:
                    result.warnings.append(w)

        result.report_data = ReportGenerator.build_report_data(
            input_file_name=result.doc_info.file_name,
            decisions=result.decisions,
            processing_time_seconds=result.processing_time_seconds,
            config_used=result.config,
            input_path=result.doc_info.file_path,
            output_path=result.output_pdf_path,
            has_digital_signature=result.doc_info.has_digital_signature,
            warnings=result.warnings,
            system_errors=result.system_errors,
            enabled_detectors=result.enabled_detectors,
            unsupported_checks=result.unsupported_checks,
            detector_registry_info=result.detector_registry_info,
        )

        if json_report_path is not None:
            saved_json = ReportGenerator.save_json_report(
                result.report_data, json_path=json_report_path
            )
            result.json_report_path = str(saved_json)

        if csv_report_path is not None:
            saved_csv = ReportGenerator.save_csv_report(
                result.decisions, csv_path=csv_report_path
            )
            result.csv_report_path = str(csv_report_path)
            result.csv_report_path = str(saved_csv)

        return result


# Bí danh tương thích ngược
PDFOrientationPipeline = DocumentPipeline

