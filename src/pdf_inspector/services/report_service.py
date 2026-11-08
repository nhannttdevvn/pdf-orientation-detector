"""
Dịch vụ tạo báo cáo kiểm tra chất lượng tài liệu (`src/pdf_inspector/services/report_service.py`).
Xuất báo cáo chi tiết và tổng hợp kết quả Quality Inspection & Correction
dưới hai định dạng: JSON (`orientation_report.json`) và CSV (`orientation_report.csv`).

Đảm bảo:
- Thể hiện rõ `enabled_detectors`, `unsupported_checks` và danh sách `defects` đa nhãn trên từng trang.
- Không bao giờ kết luận tài liệu hoàn toàn sạch mọi lỗi scan khi mới chỉ kiểm tra `orientation`.
- Giữ tương thích ngược 100% với các khóa JSON/CSV hiện có.
"""

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pdf_inspector.core.config import (
    OUTPUTS_DIR,
    STATUS_NORMAL,
    STATUS_REVIEW,
    STATUS_ROTATED,
    STRIPE_STATUS_CLEAN,
    STRIPE_STATUS_DETECTED,
    STRIPE_STATUS_ERROR,
    STRIPE_STATUS_REVIEW,
    DetectorConfig,
)
from pdf_inspector.core.logging_config import setup_logger
from pdf_inspector.models.schemas import DefectStatus, DefectType, PageDecision
from pdf_inspector.services.boundary_inspection_service import compute_boundary_summary

logger = setup_logger(__name__)

CSV_FIELDNAMES: Tuple[str, ...] = (
    "page_number",
    "status",
    "stripe_status",
    "boundary_status",
    "detected_rotation",
    "orientation_conf",
    "existing_pdf_rotation",
    "correction_angle",
    "corrected",
    "detection_method",
    "error",
)


class ReportGenerator:
    """
    Tổng hợp dữ liệu thực tế từ `DocumentPipeline` và xuất báo cáo JSON / CSV.
    """

    @staticmethod
    def build_report_data(
        input_file_name: str,
        decisions: List[PageDecision],
        processing_time_seconds: float,
        config_used: Union[DetectorConfig, Dict[str, Any]],
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        has_digital_signature: bool = False,
        warnings: Optional[List[str]] = None,
        system_errors: Optional[List[str]] = None,
        enabled_detectors: Optional[List[str]] = None,
        unsupported_checks: Optional[List[str]] = None,
        detector_registry_info: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Tạo cấu trúc dữ liệu báo cáo tổng hợp và chi tiết từng trang theo chuẩn Unified Quality Result.
        Đảm bảo không bao giờ sinh dữ liệu giả hoặc báo cáo đã kiểm tra các detector chưa triển khai.
        """
        total_pages = len(decisions)
        normal_count = sum(1 for d in decisions if d.status == STATUS_NORMAL)
        rotated_count = sum(1 for d in decisions if d.status == STATUS_ROTATED)
        review_count = sum(1 for d in decisions if d.status == STATUS_REVIEW)
        corrected_count = sum(1 for d in decisions if d.corrected)

        # Thống kê Stripe Defect Detection
        confirmed_stripe_pages = sum(
            1 for d in decisions if d.stripe_status == STRIPE_STATUS_DETECTED
        )
        stripe_review_pages = sum(
            1 for d in decisions if d.stripe_status == STRIPE_STATUS_REVIEW
        )
        stripe_clean_pages = sum(
            1 for d in decisions if d.stripe_status == STRIPE_STATUS_CLEAN
        )
        stripe_error_pages = sum(
            1 for d in decisions if d.stripe_status == STRIPE_STATUS_ERROR
        )
        pages_with_stripes = confirmed_stripe_pages + stripe_review_pages
        total_confirmed_stripes = sum(
            sum(
                1
                for s in d.stripe_defects
                if s.status == DefectStatus.CONFIRMED.value
            )
            for d in decisions
        )
        total_suspected_stripes = sum(
            sum(
                1
                for s in d.stripe_defects
                if s.status == DefectStatus.SUSPECTED_LOW_CONFIDENCE.value
            )
            for d in decisions
        )
        quality_review_pages = sum(
            1 for d in decisions if d.status == STATUS_REVIEW or d.requires_review
        )
        boundary_summary = compute_boundary_summary(decisions)

        active_detectors = (
            list(enabled_detectors)
            if enabled_detectors is not None
            else (
                list(decisions[0].enabled_detectors)
                if decisions
                else ["orientation", "stripe", "boundary"]
            )
        )
        unchecked_types = (
            list(unsupported_checks)
            if unsupported_checks is not None
            else (
                list(decisions[0].unsupported_checks)
                if decisions
                else []
            )
        )

        page_records: List[Dict[str, Any]] = [d.to_report_dict() for d in decisions]

        collected_errors: List[Dict[str, Any]] = []
        for d in decisions:
            if d.error:
                collected_errors.append(
                    {
                        "page_number": d.page_number,
                        "status": d.status,
                        "error": d.error,
                    }
                )
        for sys_err in system_errors or []:
            collected_errors.append(
                {
                    "page_number": None,
                    "status": "SYSTEM_ERROR",
                    "error": sys_err,
                }
            )

        cfg_dict = (
            config_used.to_dict()
            if isinstance(config_used, DetectorConfig)
            else dict(config_used)
        )

        rounded_time = round(float(processing_time_seconds), 3)
        warn_list = list(warnings or [])

        inspection_scope = {
            "enabled_detectors": active_detectors,
            "unsupported_checks": unchecked_types,
            "supported_corrections": [DefectType.ROTATION.value],
            "interactive_reconstructions": [DefectType.BOUNDARY.value],
            "detector_registry": list(detector_registry_info or []),
            "scope_notice": (
                "Hệ thống thực thi đồng thời kiểm tra hướng trang (Orientation), vệt sọc máy scan (Stripe) "
                "và ranh giới / lẫn trang (Boundary & Mixed-Page). "
                "Chức năng tự động sửa lỗi khi quét chỉ xoay trang (ROTATION); "
                "tách/ghép trang (Boundary Reconstruction) chỉ thực hiện sau khi người dùng xem trước và xác nhận."
            ),
        }

        stripe_summary = {
            "total_pages": total_pages,
            "clean_pages": stripe_clean_pages,
            "pages_with_stripes": pages_with_stripes,
            "confirmed_stripe_pages": confirmed_stripe_pages,
            "stripe_review_pages": stripe_review_pages,
            "stripe_error_pages": stripe_error_pages,
            "total_confirmed_stripes": total_confirmed_stripes,
            "total_suspected_stripes": total_suspected_stripes,
        }

        summary_block = {
            "input_file": input_file_name,
            "input_path": input_path,
            "output_path": output_path,
            "total_pages": total_pages,
            "normal_pages": normal_count,
            "rotated_pages": rotated_count,
            "review_pages": review_count,
            "corrected_pages": corrected_count,
            "quality_review_pages": quality_review_pages,
            "pages_with_stripes": pages_with_stripes,
            "confirmed_stripe_pages": confirmed_stripe_pages,
            "stripe_review_pages": stripe_review_pages,
            "stripe_error_pages": stripe_error_pages,
            "stripe_summary": stripe_summary,
            "pages_with_boundary_defects": boundary_summary["flagged_pages"],
            "mixed_page_count": boundary_summary["mixed_page_count"],
            "split_page_count": boundary_summary["split_page_count"],
            "content_overlap_count": boundary_summary["content_overlap_count"],
            "content_truncated_count": boundary_summary["content_truncated_count"],
            "reconstructable_pages": boundary_summary["reconstructable_pages"],
            "missing_content_pages": boundary_summary["missing_content_pages"],
            "boundary_review_pages": boundary_summary["boundary_review_pages"],
            "boundary_summary": boundary_summary,
            "processing_time_seconds": rounded_time,
            "has_digital_signature": has_digital_signature,
            "enabled_detectors": active_detectors,
            "unsupported_checks": unchecked_types,
            "inspection_scope": inspection_scope,
            "warnings": warn_list,
            "errors": collected_errors,
            "config_used": cfg_dict,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "input_file": input_file_name,
            "input_path": input_path,
            "output_path": output_path,
            "total_pages": total_pages,
            "normal_pages": normal_count,
            "rotated_pages": rotated_count,
            "review_pages": review_count,
            "corrected_pages": corrected_count,
            "quality_review_pages": quality_review_pages,
            "pages_with_stripes": pages_with_stripes,
            "confirmed_stripe_pages": confirmed_stripe_pages,
            "stripe_review_pages": stripe_review_pages,
            "stripe_error_pages": stripe_error_pages,
            "stripe_summary": stripe_summary,
            "pages_with_boundary_defects": boundary_summary["flagged_pages"],
            "mixed_page_count": boundary_summary["mixed_page_count"],
            "split_page_count": boundary_summary["split_page_count"],
            "content_overlap_count": boundary_summary["content_overlap_count"],
            "content_truncated_count": boundary_summary["content_truncated_count"],
            "reconstructable_pages": boundary_summary["reconstructable_pages"],
            "missing_content_pages": boundary_summary["missing_content_pages"],
            "boundary_review_pages": boundary_summary["boundary_review_pages"],
            "boundary_summary": boundary_summary,
            "processing_time_seconds": rounded_time,
            "has_digital_signature": has_digital_signature,
            "enabled_detectors": active_detectors,
            "unsupported_checks": unchecked_types,
            "inspection_scope": inspection_scope,
            "warnings": warn_list,
            "errors": collected_errors,
            "config_used": cfg_dict,
            "summary": summary_block,
            "pages": page_records,
        }

    @staticmethod
    def to_json_string(report_data: Dict[str, Any]) -> str:
        """Chuyển dữ liệu báo cáo thành chuỗi JSON định dạng đẹp (UTF-8)."""
        return json.dumps(report_data, ensure_ascii=False, indent=2)

    @staticmethod
    def to_csv_string(decisions: List[PageDecision]) -> str:
        """Chuyển danh sách quyết định từng trang thành chuỗi CSV."""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        for decision in decisions:
            row = decision.to_report_dict()
            writer.writerow({k: row.get(k) for k in CSV_FIELDNAMES})
        return buffer.getvalue()

    @classmethod
    def save_json_report(
        cls,
        report_data: Dict[str, Any],
        json_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Lưu báo cáo JSON ra file (mặc định: `data/outputs/orientation_report.json`)."""
        target_path = (
            Path(json_path).expanduser().resolve()
            if json_path
            else (OUTPUTS_DIR / "orientation_report.json").resolve()
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        json_content = cls.to_json_string(report_data)
        target_path.write_text(json_content, encoding="utf-8")
        logger.info("Đã xuất báo cáo JSON tại '%s'", target_path)
        return target_path

    @classmethod
    def save_csv_report(
        cls,
        decisions: List[PageDecision],
        csv_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Lưu báo cáo CSV ra file (mặc định: `data/outputs/orientation_report.csv`)."""
        target_path = (
            Path(csv_path).expanduser().resolve()
            if csv_path
            else (OUTPUTS_DIR / "orientation_report.csv").resolve()
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        csv_content = cls.to_csv_string(decisions)
        target_path.write_text(csv_content, encoding="utf-8")
        logger.info("Đã xuất báo cáo CSV tại '%s'", target_path)
        return target_path
