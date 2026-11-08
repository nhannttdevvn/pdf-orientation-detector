"""
Dịch vụ quét và quản lý kết quả nhận diện dấu độ mật độc lập (`src/pdf_inspector/services/secret_scan_service.py`).

Tuân thủ nghiêm ngặt kiến trúc hai workflow độc lập (Phase 4 — Part C3 & C4):
- Chia sẻ chung `PDFLoader` và nền tảng PyMuPDF/OpenCV nhưng hoàn toàn tách biệt với `DocumentPipeline`
  (không gọi Orientation / Stripe / Boundary Detector khi chạy Secret Marking Workflow).
- Render tuần tự từng trang PDF và giải phóng bộ nhớ mảng ảnh ngay sau mỗi trang.
- Không ghi nội dung văn bản tài liệu vào log; không lưu ảnh crop dấu mật xuống ổ đĩa;
  không tự động tách file PDF; chỉ xuất báo cáo JSON / CSV kèm khuyến cáo kỹ thuật.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from pdf_inspector.core.config import (
    PREVIEW_DPI,
    SECRET_CONFIRM_THRESHOLD,
    SECRET_RENDER_DPI,
    SECRET_REVIEW_THRESHOLD,
    SECRET_STATUS_DETECTED,
    SECRET_STATUS_ERROR,
    SECRET_STATUS_NO_MARK,
    SECRET_STATUS_REVIEW,
    SECRET_TEMPLATES_DIR,
)
from pdf_inspector.core.logging_config import setup_logger
from pdf_inspector.detectors.secret.secret_detector import (
    DetectedSecretMark,
    SecretMarkingDetector,
    SecretPageResult,
    crop_secret_mark,
    render_secret_overlay,
)
from pdf_inspector.models.schemas import PDFDocumentInfo
from pdf_inspector.processing.pdf_loader import PDFLoader

logger = setup_logger(__name__)

TECHNICAL_DISCLAIMER: str = (
    "Báo cáo này là kết quả hỗ trợ rà soát kỹ thuật tự động trên hình ảnh trang scan, "
    "không phải kết luận pháp lý về độ mật và không tự động tách file PDF."
)


@dataclass
class SecretDocumentResult:
    """Tổng hợp kết quả kiểm tra dấu độ mật cho toàn bộ tài liệu PDF."""

    doc_info: PDFDocumentInfo
    pages: List[SecretPageResult] = field(default_factory=list)
    total_time_seconds: float = 0.0
    render_dpi: int = SECRET_RENDER_DPI
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_pages(self) -> int:
        return self.doc_info.page_count

    @property
    def detected_pages_count(self) -> int:
        """Số trang phát hiện dấu độ mật đã xác nhận (hoặc đã được người dùng xác nhận)."""
        return sum(
            1 for p in self.pages if p.effective_status == SECRET_STATUS_DETECTED
        )

    @property
    def review_pages_count(self) -> int:
        """Số trang có kết quả nghi vấn cần kiểm tra hoặc lỗi xử lý."""
        return sum(
            1
            for p in self.pages
            if p.effective_status in (SECRET_STATUS_REVIEW, SECRET_STATUS_ERROR)
        )

    @property
    def total_marks_detected(self) -> int:
        """Tổng số dấu độ mật / nghi vấn hiệu lực phát hiện trên toàn tài liệu."""
        return sum(p.active_marks_count for p in self.pages)

    @property
    def flagged_pages(self) -> List[SecretPageResult]:
        """Danh sách các trang có phát hiện dấu hoặc cần kiểm tra (ẩn trang không có dấu)."""
        return [
            p
            for p in self.pages
            if p.effective_status != SECRET_STATUS_NO_MARK
        ]

    def get_page(self, page_number: int) -> Optional[SecretPageResult]:
        for p in self.pages:
            if p.page_number == page_number:
                return p
        return None

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "workflow": "secret_marking_detection",
            "disclaimer": TECHNICAL_DISCLAIMER,
            "generated_at": self.created_at,
            "file_name": self.doc_info.file_name,
            "page_count": self.total_pages,
            "file_size_bytes": self.doc_info.file_size_bytes,
            "total_time_seconds": round(float(self.total_time_seconds), 3),
            "summary": {
                "total_pages": self.total_pages,
                "pages_with_marks": self.detected_pages_count,
                "total_marks_detected": self.total_marks_detected,
                "review_required_pages": self.review_pages_count,
                "clean_no_mark_pages": sum(
                    1 for p in self.pages if p.effective_status == SECRET_STATUS_NO_MARK
                ),
            },
            "flagged_pages": [p.to_dict() for p in self.flagged_pages],
            "all_pages": [p.to_dict() for p in self.pages],
        }


class SecretScanService:
    """
    Service điều phối quét dấu độ mật cho PDF nhiều trang, rà soát thủ công và xuất báo cáo.
    Đối tượng `SecretMarkingDetector` được nạp template một lần duy nhất và tái sử dụng xuyên suốt.
    """

    def __init__(
        self,
        template_dir: Union[str, Path] = SECRET_TEMPLATES_DIR,
        confirm_threshold: float = SECRET_CONFIRM_THRESHOLD,
        review_threshold: float = SECRET_REVIEW_THRESHOLD,
        render_dpi: int = SECRET_RENDER_DPI,
    ) -> None:
        self.render_dpi: int = int(render_dpi)
        self.detector = SecretMarkingDetector(
            template_dir=template_dir,
            confirm_threshold=confirm_threshold,
            review_threshold=review_threshold,
            dpi=self.render_dpi,
        )

    def scan_pdf(
        self,
        pdf_path: Union[str, Path],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SecretDocumentResult:
        """
        Quét tuần tự từng trang của file PDF bằng `PDFLoader` dùng chung.
        Không nạp toàn bộ ảnh PDF vào RAM; không ghi nội dung tài liệu vào log.
        """
        t_start = time.perf_counter()
        resolved_path = Path(pdf_path).expanduser().resolve()

        with PDFLoader(resolved_path) as loader:
            doc_info = loader.get_document_info()
            page_results: List[SecretPageResult] = []

            for page_num in range(1, doc_info.page_count + 1):
                page_img: Optional[np.ndarray] = None
                try:
                    page_img = loader.render_page(
                        page_number=page_num,
                        dpi=self.render_dpi,
                        grayscale=False,
                    )
                    res = self.detector.detect_page(
                        image=page_img,
                        page_number=page_num,
                        color_order="RGB",
                    )
                except Exception as exc:
                    logger.warning(
                        "Không thể render/kiểm tra dấu độ mật trang %d ('%s'): %s",
                        page_num,
                        doc_info.file_name,
                        exc,
                    )
                    res = SecretPageResult(
                        page_number=page_num,
                        status=SECRET_STATUS_ERROR,
                        render_dpi=self.render_dpi,
                        error_message=str(exc),
                    )
                finally:
                    if page_img is not None:
                        del page_img

                page_results.append(res)
                if progress_callback is not None:
                    progress_callback(page_num, doc_info.page_count)

        total_sec = round(time.perf_counter() - t_start, 3)
        doc_result = SecretDocumentResult(
            doc_info=doc_info,
            pages=page_results,
            total_time_seconds=total_sec,
            render_dpi=self.render_dpi,
        )
        logger.info(
            "Hoàn tất SecretMarkingWorkflow cho '%s' (%d trang) trong %.2fs: "
            "detected_pages=%d, review_pages=%d, total_marks=%d",
            doc_info.file_name,
            doc_info.page_count,
            total_sec,
            doc_result.detected_pages_count,
            doc_result.review_pages_count,
            doc_result.total_marks_detected,
        )
        return doc_result

    @staticmethod
    def apply_review_action(
        doc_result: SecretDocumentResult,
        page_number: int,
        action: str,
        override_label: Optional[str] = None,
        note: str = "",
    ) -> Optional[SecretPageResult]:
        """
        Áp dụng thao tác rà soát của người dùng trên trang `page_number`:
        - `CONFIRMED`: Xác nhận nhãn phát hiện.
        - `IGNORED`: Bỏ qua cảnh báo (coi như trang không có dấu độ mật).
        - `EDITED`: Chỉnh sửa sang nhãn độ mật khác (`MẬT` | `TỐI MẬT` | `TUYỆT MẬT` | `GIẢI MẬT` | `NO_MARK_DETECTED`).
        """
        page_res = doc_result.get_page(page_number)
        if page_res is None:
            return None

        norm_action = action.strip().upper()
        if norm_action not in ("CONFIRMED", "IGNORED", "EDITED"):
            raise ValueError(f"Thao tác rà soát không hợp lệ: {action}")

        page_res.review_action = norm_action
        if norm_action == "EDITED" and override_label:
            page_res.override_label = override_label.strip()
        elif norm_action == "CONFIRMED":
            page_res.override_label = page_res.primary_label
        elif norm_action == "IGNORED":
            page_res.override_label = SECRET_STATUS_NO_MARK

        if note:
            page_res.notes = note.strip()
        return page_res

    @staticmethod
    def render_page_and_ephemeral_crops(
        pdf_path: Union[str, Path],
        page_result: SecretPageResult,
        preview_dpi: int = PREVIEW_DPI,
        crop_dpi: int = SECRET_RENDER_DPI,
    ) -> Tuple[np.ndarray, List[Tuple[DetectedSecretMark, np.ndarray]]]:
        """
        Lazy-render ảnh xem trước đã vẽ bounding box và cắt ảnh phóng to dấu độ mật trong bộ nhớ RAM.
        Tuyệt đối không ghi ảnh crop ra ổ đĩa (tuân thủ mục C4).
        """
        with PDFLoader(pdf_path) as loader:
            preview_rgb = loader.render_preview(
                page_number=page_result.page_number,
                dpi=preview_dpi,
                extra_rotation=0,
            )
            annotated_rgb = render_secret_overlay(preview_rgb, page_result)

            crops: List[Tuple[DetectedSecretMark, np.ndarray]] = []
            if page_result.marks:
                hi_rgb = loader.render_page(
                    page_number=page_result.page_number,
                    dpi=crop_dpi,
                    grayscale=False,
                )
                try:
                    src_size = (
                        page_result.image_width or hi_rgb.shape[1],
                        page_result.image_height or hi_rgb.shape[0],
                    )
                    for mark in page_result.marks:
                        crop_arr = crop_secret_mark(hi_rgb, mark, source_size=src_size)
                        if crop_arr is not None:
                            crops.append((mark, crop_arr))
                finally:
                    del hi_rgb

        return annotated_rgb, crops

    @staticmethod
    def export_report_json_bytes(doc_result: SecretDocumentResult) -> bytes:
        """Xuất báo cáo kết quả kiểm tra dấu độ mật ra JSON bytes (không tách PDF)."""
        payload = doc_result.to_report_dict()
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    @staticmethod
    def export_report_csv_bytes(doc_result: SecretDocumentResult) -> bytes:
        """Xuất báo cáo kết quả kiểm tra dấu độ mật ra CSV UTF-8-SIG (hiển thị chuẩn tiếng Việt trên Excel)."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "Tên file",
                "Trang",
                "Nhãn nhận diện gốc",
                "Nhãn sau rà soát",
                "Trạng thái gốc",
                "Trạng thái hiệu lực",
                "Số dấu trên trang",
                "Độ tin cậy cao nhất",
                "Phương pháp phát hiện",
                "Tọa độ BBox (px)",
                "Thao tác rà soát",
                "Ghi chú",
            ]
        )
        for p in doc_result.pages:
            if p.effective_status == SECRET_STATUS_NO_MARK and p.status == SECRET_STATUS_NO_MARK:
                continue
            top_conf = max((m.confidence for m in p.marks), default=0.0)
            methods = "; ".join(sorted({m.detection_method for m in p.marks}))
            bboxes = "; ".join(str(list(m.bbox)) for m in p.marks)
            writer.writerow(
                [
                    doc_result.doc_info.file_name,
                    p.page_number,
                    p.primary_label or "",
                    p.effective_label or "",
                    p.status,
                    p.effective_status,
                    len(p.marks),
                    f"{top_conf:.4f}" if p.marks else "",
                    methods,
                    bboxes,
                    p.review_action or "",
                    p.notes or p.error_message or "",
                ]
            )
        return buf.getvalue().encode("utf-8-sig")

