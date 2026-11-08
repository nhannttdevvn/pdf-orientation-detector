"""
DetectorRegistry — Quản lý tập trung vòng đời và trạng thái của các Detector (`src/pdf_inspector/detectors/registry.py`).
Đảm bảo:
- Đăng ký detector hợp lệ (`BaseDetector`).
- Cho phép bật/tắt (`ENABLED` / `DISABLED`) đối với các detector đã triển khai (`AVAILABLE`).
- Ngăn đăng ký trùng `detector_id`.
- Ghi nhận rõ các kiểm tra chưa triển khai (`NOT_IMPLEMENTED`: Stripe, Boundary) và TUYỆT ĐỐI không thực thi
  hoặc coi detector chưa chạy là đã kiểm tra sạch lỗi.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pdf_inspector.core.config import DetectorConfig
from pdf_inspector.detectors.base_detector import BaseDetector
from pdf_inspector.models.schemas import DefectType, DetectorScope, DetectorStatus


@dataclass
class DetectorDescriptor:
    """Thông tin mô tả và trạng thái của một Detector trong Registry."""

    detector_id: str
    name: str
    scope: str
    supported_defect_types: Tuple[str, ...]
    status: str  # ENABLED, DISABLED, NOT_IMPLEMENTED
    version: str = "1.0.0"
    description: str = ""
    instance: Optional[BaseDetector] = None

    @property
    def is_available(self) -> bool:
        """Trả về True nếu detector đã có implementation thực tế (`AVAILABLE`)."""
        return (
            self.instance is not None
            and self.status != DetectorStatus.NOT_IMPLEMENTED.value
        )

    @property
    def is_enabled(self) -> bool:
        """Trả về True nếu detector đã triển khai và đang được bật (`ENABLED`)."""
        return (
            self.instance is not None
            and self.status == DetectorStatus.ENABLED.value
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "name": self.name,
            "version": self.version,
            "scope": self.scope,
            "supported_defect_types": list(self.supported_defect_types),
            "status": self.status,
            "availability": (
                DetectorStatus.AVAILABLE.value
                if self.is_available
                else DetectorStatus.NOT_IMPLEMENTED.value
            ),
            "description": self.description,
        }


class DetectorRegistry:
    """
     Sổ đăng ký trung tâm quản lý tất cả các Detector trong hệ thống PDF Quality Inspection.
    """

    def __init__(self) -> None:
        self._descriptors: Dict[str, DetectorDescriptor] = {}

    def register(
        self,
        detector: BaseDetector,
        name: Optional[str] = None,
        enabled: bool = True,
        description: str = "",
    ) -> DetectorDescriptor:
        """
        Đăng ký một Detector đã triển khai hợp lệ vào Registry.
        Ném `TypeError` nếu không kế thừa `BaseDetector`, hoặc `ValueError` nếu trùng `detector_id`.
        """
        if not isinstance(detector, BaseDetector):
            raise TypeError(
                f"Detector đăng ký phải kế thừa BaseDetector, nhận được: {type(detector)!r}"
            )

        det_id = detector.detector_id.strip()
        if not det_id:
            raise ValueError("detector_id không được rỗng.")

        if det_id in self._descriptors:
            raise ValueError(f"Detector '{det_id}' đã được đăng ký trước đó.")

        status = (
            DetectorStatus.ENABLED.value
            if enabled
            else DetectorStatus.DISABLED.value
        )
        descriptor = DetectorDescriptor(
            detector_id=det_id,
            name=name or det_id.replace("_", " ").title(),
            scope=detector.scope,
            supported_defect_types=tuple(detector.supported_defect_types),
            status=status,
            version=detector.version,
            description=description,
            instance=detector,
        )
        self._descriptors[det_id] = descriptor
        return descriptor

    def register_unimplemented(
        self,
        detector_id: str,
        name: str,
        scope: str,
        supported_defect_types: Tuple[str, ...],
        description: str = "",
    ) -> DetectorDescriptor:
        """
        Khai báo thông tin một loại detector nằm trong kiến trúc thiết kế nhưng CHƯA triển khai
        (`status = NOT_IMPLEMENTED`, `instance = None`).
        """
        det_id = detector_id.strip()
        if not det_id:
            raise ValueError("detector_id không được rỗng.")
        if det_id in self._descriptors:
            raise ValueError(f"Detector '{det_id}' đã được đăng ký trước đó.")

        descriptor = DetectorDescriptor(
            detector_id=det_id,
            name=name,
            scope=scope,
            supported_defect_types=tuple(supported_defect_types),
            status=DetectorStatus.NOT_IMPLEMENTED.value,
            version="N/A",
            description=description,
            instance=None,
        )
        self._descriptors[det_id] = descriptor
        return descriptor

    @staticmethod
    def _resolve_id(detector_id: str) -> str:
        norm = detector_id.strip()
        return "boundary" if norm == "page_boundary" else norm

    def enable(self, detector_id: str) -> None:
        """
        Bật một detector đã triển khai.
        Tuyệt đối không cho phép bật detector đang ở trạng thái `NOT_IMPLEMENTED`.
        """
        det_id = self._resolve_id(detector_id)
        if det_id not in self._descriptors:
            raise KeyError(f"Detector '{detector_id}' không tồn tại trong Registry.")
        desc = self._descriptors[det_id]
        if desc.status == DetectorStatus.NOT_IMPLEMENTED.value or desc.instance is None:
            raise RuntimeError(
                f"Không thể kích hoạt detector '{detector_id}' vì chưa được triển khai (NOT_IMPLEMENTED)."
            )
        desc.status = DetectorStatus.ENABLED.value

    def disable(self, detector_id: str) -> None:
        """Tắt một detector đã triển khai (chuyển sang `DISABLED`)."""
        det_id = self._resolve_id(detector_id)
        if det_id not in self._descriptors:
            raise KeyError(f"Detector '{detector_id}' không tồn tại trong Registry.")
        desc = self._descriptors[det_id]
        if desc.status == DetectorStatus.NOT_IMPLEMENTED.value:
            return
        desc.status = DetectorStatus.DISABLED.value

    def get_status(self, detector_id: str) -> str:
        """Trả về trạng thái hiện tại của `detector_id` (`ENABLED`, `DISABLED`, `NOT_IMPLEMENTED`)."""
        det_id = self._resolve_id(detector_id)
        if det_id not in self._descriptors:
            raise KeyError(f"Detector '{detector_id}' không tồn tại trong Registry.")
        return self._descriptors[det_id].status

    def is_available(self, detector_id: str) -> bool:
        """Kiểm tra xem `detector_id` có thuộc trạng thái `AVAILABLE` (đã triển khai) hay không."""
        det_id = self._resolve_id(detector_id)
        if det_id not in self._descriptors:
            return False
        return self._descriptors[det_id].is_available

    def get_detector(self, detector_id: str) -> Optional[BaseDetector]:
        """Trả về instance của detector nếu đã triển khai, ngược lại trả về None."""
        det_id = self._resolve_id(detector_id)
        desc = self._descriptors.get(det_id)
        return desc.instance if desc else None

    def get_enabled_detectors(self, scope: Optional[str] = None) -> List[BaseDetector]:
        """
        Trả về danh sách các Detector đang ở trạng thái `ENABLED` (và có instance thực thi hợp lệ).
        Không bao giờ trả về detector `DISABLED` hoặc `NOT_IMPLEMENTED`.
        """
        enabled_list: List[BaseDetector] = []
        for desc in self._descriptors.values():
            if desc.is_enabled and desc.instance is not None:
                if scope is None or desc.scope == scope:
                    enabled_list.append(desc.instance)
        return enabled_list

    def get_enabled_detector_ids(self) -> List[str]:
        """Danh sách `detector_id` của các detector đang hoạt động (`ENABLED`)."""
        return [d.detector_id for d in self._descriptors.values() if d.is_enabled]

    def get_unimplemented_detector_ids(self) -> List[str]:
        """Danh sách `detector_id` của các detector chưa được triển khai (`NOT_IMPLEMENTED`)."""
        return [
            d.detector_id
            for d in self._descriptors.values()
            if d.status == DetectorStatus.NOT_IMPLEMENTED.value
        ]

    def get_unsupported_checks(self) -> List[str]:
        """
        Danh sách các loại lỗi (`DefectType`) hiện chưa được kiểm tra
        (do detector tương ứng đang `NOT_IMPLEMENTED` hoặc `DISABLED`).
        """
        checked_types = set()
        for desc in self._descriptors.values():
            if desc.is_enabled:
                checked_types.update(desc.supported_defect_types)

        unsupported: List[str] = []
        for desc in self._descriptors.values():
            if not desc.is_enabled:
                for dt in desc.supported_defect_types:
                    if dt not in checked_types and dt not in unsupported:
                        unsupported.append(dt)
        return unsupported

    def list_detectors(self) -> List[DetectorDescriptor]:
        """Liệt kê toàn bộ các DetectorDescriptor trong Registry."""
        return list(self._descriptors.values())

    def to_summary_list(self) -> List[Dict[str, Any]]:
        """Xuất danh sách thông tin trạng thái của toàn bộ các detector phục vụ báo cáo và UI."""
        return [d.to_dict() for d in self._descriptors.values()]


def create_default_registry(
    config: Optional[DetectorConfig] = None,
    orientation_detector: Optional[BaseDetector] = None,
    stripe_detector: Optional[BaseDetector] = None,
    boundary_detector: Optional[BaseDetector] = None,
) -> DetectorRegistry:
    """
    Khởi tạo `DetectorRegistry` mặc định của hệ thống:
    - Đăng ký `OrientationDetector` (`PAGE_LEVEL`) ở trạng thái `ENABLED` (`AVAILABLE`).
    - Đăng ký `StripeDetector` (`PAGE_LEVEL`) ở trạng thái `ENABLED` (`AVAILABLE`).
    - Đăng ký `PageBoundaryDetector` (`CROSS_PAGE`) ở trạng thái `ENABLED` (`AVAILABLE`) theo Phase 3.
    """
    from pdf_inspector.detectors.boundary.boundary_detector import PageBoundaryDetector
    from pdf_inspector.detectors.defects.stripe_detector import StripeDetector
    from pdf_inspector.detectors.orientation.detector import TesseractOrientationDetector

    cfg = config or DetectorConfig()
    registry = DetectorRegistry()

    ori_det = orientation_detector or TesseractOrientationDetector(
        tesseract_cmd=cfg.tesseract_cmd,
        conf_threshold=cfg.orientation_conf_threshold,
        retry_with_preprocessed=cfg.retry_with_preprocessed,
        max_ocr_dimension=cfg.max_ocr_dimension,
        min_ocr_dimension=cfg.min_ocr_dimension,
        median_blur_ksize=cfg.median_blur_ksize,
    )
    registry.register(
        detector=ori_det,
        name="Orientation Detector (Tesseract OSD + OpenCV)",
        enabled=True,
        description="Phát hiện trang bị xoay 90°, 180°, 270° và đề xuất góc xoay chỉnh sửa.",
    )

    str_det = stripe_detector or StripeDetector(
        confirm_threshold=cfg.stripe_confirm_threshold,
        review_threshold=cfg.stripe_review_threshold,
        min_relative_length=cfg.stripe_min_relative_length,
        max_thickness_px=cfg.stripe_max_thickness_px,
        min_contrast=cfg.stripe_min_contrast,
        horizontal_min_page_ratio=cfg.horizontal_min_page_ratio,
        horizontal_margin_extension=cfg.horizontal_margin_extension,
    )
    registry.register(
        detector=str_det,
        name="Stripe Detector (Morphology + HoughLinesP + Projection Profile)",
        enabled=bool(cfg.enable_stripe_detection),
        description="Phát hiện lỗi sọc đen/xám/dải sọc (dọc, ngang, nghiêng) do bụi kính máy scan.",
    )

    bnd_det = boundary_detector or PageBoundaryDetector(
        confirm_threshold=cfg.boundary_confirm_threshold,
        review_threshold=cfg.boundary_review_threshold,
        overlap_threshold=cfg.boundary_overlap_threshold,
        use_optional_layout_model=cfg.boundary_use_optional_layout_model,
    )
    registry.register(
        detector=bnd_det,
        name="Boundary & Mixed-Page Detector (Layout + Cross-Page Validation)",
        enabled=bool(cfg.enable_boundary_detection),
        description=(
            "Phát hiện lỗi lẫn trang (MIXED_PAGE), chia tách trang (SPLIT_PAGE), "
            "chồng lặp nội dung (CONTENT_OVERLAP) và cắt cụt mép trang (CONTENT_TRUNCATED)."
        ),
    )
    return registry

