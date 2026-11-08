"""
Dịch vụ chỉnh sửa lỗi chất lượng PDF (`src/pdf_inspector/services/correction_service.py`).
Tách biệt hoàn toàn giai đoạn Detection và Correction:
- Detection chỉ xác định lỗi và đề xuất hành động.
- Correction chỉ thực thi những thao tác đã được hỗ trợ và cho phép.
- Hiện tại chỉ hỗ trợ `RotationCorrector` (`PDFCorrector`) cho lỗi `ROTATION`:
  + Chỉ sửa các trang `ROTATED` khi bật Auto Fix, hoặc các trang được người dùng chọn góc xoay thủ công (`manual_override`).
  + Tuyệt đối không tự động xoay trang `REVIEW` (`requires_review=True` mà không có `manual_override`).
  + Sử dụng `page.set_rotation(...)` cập nhật metadata `/Rotate` modulo 360, không rasterize lại PDF.
  + Không bao giờ ghi đè lên file PDF gốc.
"""

from pathlib import Path
from typing import List, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore[no-redef]

from pdf_inspector.core.config import OUTPUTS_DIR, STATUS_REVIEW, STATUS_ROTATED
from pdf_inspector.core.exceptions import PDFCorrectionError
from pdf_inspector.core.logging_config import setup_logger
from pdf_inspector.models.schemas import CorrectionSummary, DefectType, PageDecision
from pdf_inspector.processing.pdf_loader import PDFLoader
from pdf_inspector.utils.file_utils import (
    combine_rotation_angles,
    is_same_file_path,
    normalize_angle,
)

logger = setup_logger(__name__)


class PDFCorrector:
    """
    Bộ chỉnh sửa hướng trang PDF (`RotationCorrector`).
    Thực hiện xoay trang PDF an toàn và xuất file PDF mới mà không làm thay đổi file gốc.
    Các loại sửa lỗi khác (Stripe Repair, Page Boundary Reconstruction) chưa được triển khai.
    """

    supported_defect_types = (DefectType.ROTATION.value,)

    def apply_corrections(
        self,
        input_pdf_path: Union[str, Path],
        output_pdf_path: Union[str, Path, None],
        decisions: List[PageDecision],
        auto_fix: bool = True,
    ) -> CorrectionSummary:
        """
        Áp dụng góc xoay từ danh sách `decisions` và lưu sang `output_pdf_path`.
        Cập nhật thuộc tính `decision.corrected` trực tiếp trên từng phần tử của `decisions`.
        """
        in_path = Path(input_pdf_path).expanduser().resolve()
        if not in_path.exists() or not in_path.is_file():
            raise PDFCorrectionError(f"File PDF đầu vào không tồn tại: {in_path}")

        if output_pdf_path is None:
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = (OUTPUTS_DIR / "corrected.pdf").resolve()
        else:
            out_path = Path(output_pdf_path).expanduser().resolve()

        # Nguyên tắc an toàn: Không bao giờ ghi đè file PDF đầu vào
        if is_same_file_path(in_path, out_path):
            raise PDFCorrectionError(
                f"Không được phép ghi đè lên file PDF gốc ('{in_path}'). "
                "Vui lòng chọn đường dẫn xuất file khác."
            )

        warnings: List[str] = []
        with PDFLoader(in_path) as loader:
            doc_info = loader.get_document_info()
            has_signature = doc_info.has_digital_signature

        if has_signature:
            sig_warning = (
                "CẢNH BÁO CHỮ KÝ SỐ: File PDF gốc có chứa chữ ký số (Digital Signature). "
                "Việc lưu bản PDF đã xoay trang có thể khiến chữ ký số trên file mới không còn hợp lệ. "
                "File PDF gốc vẫn được giữ nguyên vẹn."
            )
            logger.warning(sig_warning)
            warnings.append(sig_warning)

        decision_map = {d.page_number: d for d in decisions}
        corrected_pages: List[int] = []

        try:
            doc = fitz.open(str(in_path))
        except Exception as exc:
            raise PDFCorrectionError(
                f"Không thể mở file PDF để chỉnh sửa ('{in_path.name}'): {exc}"
            ) from exc

        try:
            total_pages = doc.page_count
            for idx in range(total_pages):
                page_num = idx + 1
                decision = decision_map.get(page_num)
                if decision is None:
                    continue

                page = doc.load_page(idx)
                try:
                    current_pdf_rot = normalize_angle(page.rotation)
                except ValueError:
                    current_pdf_rot = 0

                decision.existing_pdf_rotation = current_pdf_rot

                allow_rotate = False
                if decision.manual_override:
                    allow_rotate = normalize_angle(decision.correction_angle) != 0
                elif decision.status == STATUS_REVIEW:
                    allow_rotate = False
                    decision.correction_angle = 0
                elif decision.status == STATUS_ROTATED and auto_fix:
                    allow_rotate = normalize_angle(decision.correction_angle) != 0
                else:
                    allow_rotate = False

                if allow_rotate:
                    new_pdf_rot = combine_rotation_angles(
                        current_pdf_rot, decision.correction_angle
                    )
                    page.set_rotation(new_pdf_rot)
                    decision.corrected = True
                    decision.should_rotate = True
                    corrected_pages.append(page_num)
                    logger.info(
                        "Đã xoay trang %d: existing=%d° + correction=%d° -> new_rotation=%d°",
                        page_num,
                        current_pdf_rot,
                        decision.correction_angle,
                        new_pdf_rot,
                    )
                else:
                    decision.corrected = False
                    decision.should_rotate = False

            out_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(out_path))
            logger.info(
                "Đã lưu PDF chỉnh sửa tại '%s' (đã xoay %d/%d trang).",
                out_path,
                len(corrected_pages),
                total_pages,
            )
            return CorrectionSummary(
                input_path=str(in_path),
                output_path=str(out_path),
                total_pages=total_pages,
                corrected_pages=corrected_pages,
                corrected_count=len(corrected_pages),
                has_digital_signature=has_signature,
                warnings=warnings,
            )
        except PDFCorrectionError:
            raise
        except Exception as exc:
            logger.error("Lỗi khi lưu file PDF đã chỉnh sửa '%s': %s", out_path, exc)
            raise PDFCorrectionError(
                f"Không thể lưu file PDF đã chỉnh sửa tại '{out_path}': {exc}"
            ) from exc
        finally:
            doc.close()


# Bí danh ngữ nghĩa cho kiến trúc Correction mở rộng
RotationCorrector = PDFCorrector
