"""
Chuẩn hóa Interface cho các Detector trong kiến trúc Multi-Defect PDF Quality Inspection (`src/pdf_inspector/detectors/base_detector.py`).
Mỗi Detector bắt buộc khai báo:
- `detector_id`: Định danh duy nhất (ví dụ: `"orientation"`).
- `version`: Phiên bản thuật toán (ví dụ: `"1.0.0"`).
- `scope`: Phạm vi xử lý (`PAGE_LEVEL`, `DOCUMENT_LEVEL`, hoặc `CROSS_PAGE`).
- `supported_defect_types`: Danh sách các loại lỗi mà detector kiểm tra (ví dụ: `("ROTATION",)`).
- `detect(context)`: Phương thức thực thi nhận diện và trả về danh sách `DefectResult`.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from pdf_inspector.models.context import PageContext
from pdf_inspector.models.schemas import (
    DefectResult,
    DefectType,
    DetectorScope,
    OrientationDetectionResult,
)


class BaseDetector(ABC):
    """
    Interface cơ sở cho mọi Detector kiểm tra chất lượng PDF.
    Không ép buộc Page-level và Cross-page Detector phải dùng chung cùng một kiểu dữ liệu đầu vào.
    """

    @property
    @abstractmethod
    def detector_id(self) -> str:
        """Định danh duy nhất của detector (ví dụ: 'orientation')."""
        raise NotImplementedError

    @property
    def version(self) -> str:
        """Phiên bản hiện tại của detector."""
        return "1.0.0"

    @property
    @abstractmethod
    def scope(self) -> str:
        """Phạm vi phân tích: PAGE_LEVEL, DOCUMENT_LEVEL hoặc CROSS_PAGE."""
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_defect_types(self) -> Tuple[str, ...]:
        """Danh sách mã loại lỗi mà detector này phụ trách (ví dụ: ('ROTATION',))."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, context: Any) -> List[DefectResult]:
        """Thực thi kiểm tra lỗi trên `context` và trả về danh sách `DefectResult`."""
        raise NotImplementedError


class PageLevelDetector(BaseDetector):
    """
    Interface dành cho các Detector hoạt động độc lập trên từng trang PDF (`PageContext`).
    """

    @property
    def scope(self) -> str:
        return DetectorScope.PAGE_LEVEL.value

    @abstractmethod
    def detect(self, context: PageContext) -> List[DefectResult]:
        """Phân tích một trang thông qua `PageContext` dùng chung."""
        raise NotImplementedError


class DocumentLevelDetector(BaseDetector):
    """
    Extension Point (Interface mở rộng) dành cho các Cross-page / Document-level Detector trong tương lai
    (ví dụ: phân tích quan hệ kích thước, lề trang hoặc thứ tự trang trên toàn tài liệu).
    """

    @property
    def scope(self) -> str:
        return DetectorScope.CROSS_PAGE.value

    @abstractmethod
    def detect(self, context: Sequence[Any]) -> List[DefectResult]:
        """Phân tích dữ liệu liên trang của tài liệu."""
        raise NotImplementedError


class BaseOrientationDetector(PageLevelDetector):
    """
    Interface chuẩn cho nhóm Detector phát hiện hướng trang (`detector_id = 'orientation'`).
    Hỗ trợ cả phương thức `detect(context: PageContext) -> List[DefectResult]` theo kiến trúc mới
    và `detect_page(image, ...) -> OrientationDetectionResult` để tương thích ngược.
    """

    @property
    def detector_id(self) -> str:
        return "orientation"

    @property
    def supported_defect_types(self) -> Tuple[str, ...]:
        return (DefectType.ROTATION.value,)

    @abstractmethod
    def detect_page(
        self,
        image: np.ndarray,
        page_number: int = 1,
        existing_pdf_rotation: int = 0,
        conf_threshold: Optional[float] = None,
    ) -> OrientationDetectionResult:
        """Phát hiện hướng của một trang tài liệu từ mảng ảnh NumPy."""
        raise NotImplementedError
