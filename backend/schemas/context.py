"""
Shared PageContext và quy ước hệ tọa độ (`src/pdf_inspector/models/context.py`).
Thiết kế cho phép nhiều detector trên cùng một trang tái sử dụng ảnh đã render và ảnh tiền xử lý
mà không cần tự render lại PDF, đồng thời giải phóng bộ nhớ ngay sau khi xử lý xong từng trang.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ai_engine.core.config import MAX_OCR_DIMENSION, MEDIAN_BLUR_KSIZE, MIN_OCR_DIMENSION, RENDER_DPI
from backend.repositories.file_utils import normalize_angle


class CoordinateSpace(str, Enum):
    """Hệ quy chiếu tọa độ của vùng lỗi."""

    RENDERED_IMAGE_PX = "RENDERED_IMAGE_PX"
    PDF_POINT = "PDF_POINT"
    NORMALIZED = "NORMALIZED"


@dataclass(frozen=True)
class CoordinateTransform:
    """
    Quy ước chuyển đổi tọa độ giữa trang PDF gốc (đơn vị point: 1 inch = 72 pt, gốc tọa độ góc trên-trái)
    và ảnh bitmap đã render (đơn vị pixel, gốc tọa độ góc trên-trái):
    - `pdf_width_pt`, `pdf_height_pt`: Kích thước trang PDF khi hiển thị ở góc `original_rotation`.
    - `image_width_px`, `image_height_px`: Kích thước mảng ảnh đã render ở `dpi`.
    - `original_rotation`: Góc xoay metadata `/Rotate` hiện tại của trang PDF (0, 90, 180, 270 theo chiều kim đồng hồ).
    """

    pdf_width_pt: float
    pdf_height_pt: float
    image_width_px: int
    image_height_px: int
    dpi: int
    original_rotation: int = 0

    @property
    def scale_x(self) -> float:
        """Tỷ lệ quy đổi từ point sang pixel theo trục ngang."""
        return self.image_width_px / max(1.0, float(self.pdf_width_pt))

    @property
    def scale_y(self) -> float:
        """Tỷ lệ quy đổi từ point sang pixel theo trục dọc."""
        return self.image_height_px / max(1.0, float(self.pdf_height_pt))

    def pixel_to_pdf_point(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Ánh xạ tọa độ điểm từ ảnh pixel về tọa độ point trên trang PDF."""
        return (float(x_px) / self.scale_x, float(y_px) / self.scale_y)

    def pdf_point_to_pixel(self, x_pt: float, y_pt: float) -> Tuple[float, float]:
        """Ánh xạ tọa độ điểm từ trang PDF (point) sang tọa độ pixel trên ảnh."""
        return (float(x_pt) * self.scale_x, float(y_pt) * self.scale_y)

    def pixel_to_normalized(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Ánh xạ tọa độ pixel trên ảnh render sang tọa độ chuẩn hóa [0.0 .. 1.0]."""
        nx = float(x_px) / max(1.0, float(self.image_width_px))
        ny = float(y_px) / max(1.0, float(self.image_height_px))
        return (round(min(1.0, max(0.0, nx)), 4), round(min(1.0, max(0.0, ny)), 4))

    def normalized_to_pixel(self, x_norm: float, y_norm: float) -> Tuple[int, int]:
        """Ánh xạ tọa độ chuẩn hóa [0.0 .. 1.0] về tọa độ pixel trên ảnh render."""
        px = int(round(float(x_norm) * self.image_width_px))
        py = int(round(float(y_norm) * self.image_height_px))
        return (
            min(self.image_width_px, max(0, px)),
            min(self.image_height_px, max(0, py)),
        )

    def normalized_to_pdf_point(
        self, x_norm: float, y_norm: float
    ) -> Tuple[float, float]:
        """Ánh xạ tọa độ chuẩn hóa [0.0 .. 1.0] sang tọa độ point trên trang PDF."""
        return (
            round(float(x_norm) * self.pdf_width_pt, 2),
            round(float(y_norm) * self.pdf_height_pt, 2),
        )

    def rotated_pixel_to_rendered_pixel(
        self, x_rot: float, y_rot: float, applied_rotation_cw: int
    ) -> Tuple[float, float]:
        """
        Chuyển tọa độ điểm `(x_rot, y_rot)` trên ảnh đã xoay thêm góc `applied_rotation_cw` (0, 90, 180, 270 CW)
        ngược về tọa độ `(x_px, y_px)` trên ảnh render gốc (`original_image`).
        """
        norm = normalize_angle(applied_rotation_cw)
        w = float(self.image_width_px)
        h = float(self.image_height_px)
        if norm == 0:
            return (float(x_rot), float(y_rot))
        if norm == 90:
            # Khi xoay 90° CW: (x_rot, y_rot) trên ảnh kích thước (H, W) tương ứng với x_orig = y_rot, y_orig = H - 1 - x_rot
            return (float(y_rot), max(0.0, h - 1.0 - float(x_rot)))
        if norm == 180:
            return (max(0.0, w - 1.0 - float(x_rot)), max(0.0, h - 1.0 - float(y_rot)))
        # norm == 270 CW:
        return (max(0.0, w - 1.0 - float(y_rot)), float(x_rot))

    def rendered_pixel_to_rotated_pixel(
        self, x_px: float, y_px: float, applied_rotation_cw: int
    ) -> Tuple[float, float]:
        """
        Chuyển tọa độ điểm `(x_px, y_px)` trên ảnh render gốc sang hệ tọa độ của ảnh đã xoay
        thêm góc `applied_rotation_cw` (0, 90, 180, 270 CW).
        """
        norm = normalize_angle(applied_rotation_cw)
        w = float(self.image_width_px)
        h = float(self.image_height_px)
        if norm == 0:
            return (float(x_px), float(y_px))
        if norm == 90:
            return (max(0.0, h - 1.0 - float(y_px)), float(x_px))
        if norm == 180:
            return (max(0.0, w - 1.0 - float(x_px)), max(0.0, h - 1.0 - float(y_px)))
        # norm == 270 CW:
        return (float(y_px), max(0.0, w - 1.0 - float(x_px)))

    def pixel_bbox_to_pdf_rect(
        self, x0_px: float, y0_px: float, x1_px: float, y1_px: float
    ) -> Tuple[float, float, float, float]:
        """Ánh xạ vùng chữ nhật lỗi (bounding box) từ ảnh pixel về tọa độ PDF `(x0, y0, x1, y1)`."""
        px0, py0 = self.pixel_to_pdf_point(x0_px, y0_px)
        px1, py1 = self.pixel_to_pdf_point(x1_px, y1_px)
        return (round(min(px0, px1), 2), round(min(py0, py1), 2), round(max(px0, px1), 2), round(max(py0, py1), 2))

    def bbox_px_to_pdf(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Ánh xạ tuple `bbox = (x0, y0, x1, y1)` theo pixel sang tọa độ PDF point."""
        return self.pixel_bbox_to_pdf_rect(bbox[0], bbox[1], bbox[2], bbox[3])


class PageContext:
    """
    Ngữ cảnh dữ liệu dùng chung của một trang PDF trong quá trình kiểm tra chất lượng (Quality Inspection).
    Đảm bảo:
    - Tái sử dụng `original_image`, `grayscale_image` và `processed_image` giữa mọi Page-level Detector.
    - Không cho phép các detector ghi đè trực tiếp lên `original_image` (`writeable = False`).
    - Hỗ trợ giải phóng bộ nhớ (`release()`) ngay sau khi trang được kiểm tra xong để tiết kiệm RAM.
    """

    def __init__(
        self,
        page_number: int,
        page_index: Optional[int] = None,
        existing_rotation: int = 0,
        width_pt: float = 0.0,
        height_pt: float = 0.0,
        image: Optional[np.ndarray] = None,
        *,
        document_id: str = "",
        pdf_page_size: Optional[Tuple[float, float]] = None,
        original_rotation: Optional[int] = None,
        original_image: Optional[np.ndarray] = None,
        grayscale_image: Optional[np.ndarray] = None,
        processed_image: Optional[np.ndarray] = None,
        dpi: int = RENDER_DPI,
        metadata: Optional[Dict[str, Any]] = None,
        max_ocr_dimension: int = MAX_OCR_DIMENSION,
        min_ocr_dimension: int = MIN_OCR_DIMENSION,
        median_blur_ksize: int = MEDIAN_BLUR_KSIZE,
    ) -> None:
        self.document_id: str = document_id
        self.page_number: int = int(page_number)
        self.page_index: int = (
            int(page_index) if page_index is not None else max(0, self.page_number - 1)
        )

        resolved_rot = (
            original_rotation if original_rotation is not None else existing_rotation
        )
        self.original_rotation: int = normalize_angle(resolved_rot)

        if pdf_page_size is not None:
            self.pdf_page_size: Tuple[float, float] = (
                float(pdf_page_size[0]),
                float(pdf_page_size[1]),
            )
        else:
            self.pdf_page_size = (float(width_pt), float(height_pt))

        raw_img = original_image if original_image is not None else image
        if raw_img is None:
            raise ValueError("PageContext yêu cầu cung cấp `original_image` hoặc `image`.")

        # Khóa quyền ghi trực tiếp lên ảnh gốc để không detector nào vô tình làm hỏng dữ liệu chung
        self._original_image: Optional[np.ndarray] = raw_img
        if self._original_image.flags.writeable:
            self._original_image.flags.writeable = False

        self._grayscale_image: Optional[np.ndarray] = grayscale_image
        self._processed_image: Optional[np.ndarray] = processed_image
        self._clahe_image: Optional[np.ndarray] = None
        self._rotated_cache: Dict[int, np.ndarray] = {}

        self.dpi: int = int(dpi)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.max_ocr_dimension: int = int(max_ocr_dimension)
        self.min_ocr_dimension: int = int(min_ocr_dimension)
        self.median_blur_ksize: int = int(median_blur_ksize)

        img_h, img_w = self._original_image.shape[:2]
        self.coordinate_transform = CoordinateTransform(
            pdf_width_pt=self.pdf_page_size[0] or float(img_w * 72.0 / max(1, self.dpi)),
            pdf_height_pt=self.pdf_page_size[1] or float(img_h * 72.0 / max(1, self.dpi)),
            image_width_px=img_w,
            image_height_px=img_h,
            dpi=self.dpi,
            original_rotation=self.original_rotation,
        )

    @property
    def existing_rotation(self) -> int:
        """Thuộc tính tương thích ngược cho `original_rotation`."""
        return self.original_rotation

    @property
    def width_pt(self) -> float:
        """Chiều rộng trang PDF (đơn vị point)."""
        return self.pdf_page_size[0]

    @property
    def height_pt(self) -> float:
        """Chiều cao trang PDF (đơn vị point)."""
        return self.pdf_page_size[1]

    @property
    def original_image(self) -> np.ndarray:
        """Ảnh render gốc của trang PDF (chưa xoay thêm, read-only)."""
        if self._original_image is None:
            raise RuntimeError(
                f"Ảnh của trang {self.page_number} đã được giải phóng khỏi bộ nhớ."
            )
        return self._original_image

    @property
    def image(self) -> np.ndarray:
        """Thuộc tính tương thích ngược trả về `original_image`."""
        return self.original_image

    @property
    def grayscale_image(self) -> np.ndarray:
        """
        Ảnh grayscale đã chuẩn hóa kích thước, được khởi tạo lười (lazy) và lưu đệm
        để mọi detector trên cùng trang tái sử dụng.
        """
        if self._grayscale_image is None:
            from ai_engine.preprocessing.image_preprocessor import prepare_original_grayscale

            self._grayscale_image = prepare_original_grayscale(
                self.original_image,
                max_dim=self.max_ocr_dimension,
                min_dim=self.min_ocr_dimension,
            )
            self._grayscale_image.flags.writeable = False
        return self._grayscale_image

    @property
    def processed_image(self) -> np.ndarray:
        """
        Ảnh sau tiền xử lý (Median Blur + CLAHE), được khởi tạo lười (lazy) và lưu đệm
        để mọi detector trên cùng trang tái sử dụng.
        """
        if self._processed_image is None:
            from ai_engine.preprocessing.image_preprocessor import preprocess_for_ocr

            self._processed_image = preprocess_for_ocr(
                self.original_image,
                ksize=self.median_blur_ksize,
                max_dim=self.max_ocr_dimension,
                min_dim=self.min_ocr_dimension,
            )
            self._processed_image.flags.writeable = False
        return self._processed_image

    def get_rotated_image(self, angle_cw: int, use_processed: bool = False) -> np.ndarray:
        """
        Trả về ảnh đã xoay thêm góc `angle_cw` (0, 90, 180, 270 theo chiều kim đồng hồ)
        mà không làm thay đổi `original_image`.
        """
        norm = normalize_angle(angle_cw)
        base = self.processed_image if use_processed else self.original_image
        if norm == 0:
            return base
        cache_key = norm + (1000 if use_processed else 0)
        if cache_key not in self._rotated_cache:
            if norm == 90:
                rot_img = np.rot90(base, k=-1).copy()
            elif norm == 180:
                rot_img = np.rot90(base, k=2).copy()
            else:
                rot_img = np.rot90(base, k=1).copy()
            rot_img.flags.writeable = False
            self._rotated_cache[cache_key] = rot_img
        return self._rotated_cache[cache_key]

    def ensure_clahe(self) -> np.ndarray:
        """
        Trả về ảnh tăng cường tương phản cục bộ CLAHE cùng kích thước với `original_image`
        (không qua Median Blur để bảo toàn các vệt sọc mảnh 1-2px), có lưu đệm trên `PageContext`.
        """
        if self._clahe_image is None:
            from ai_engine.preprocessing.image_preprocessor import enhance_contrast_clahe, to_grayscale

            gray = to_grayscale(self.original_image)
            self._clahe_image = enhance_contrast_clahe(gray)
            self._clahe_image.flags.writeable = False
        return self._clahe_image

    def release(self) -> None:
        """Giải phóng toàn bộ các mảng ảnh của trang khỏi RAM sau khi các detector đã xử lý xong."""
        self._original_image = None
        self._grayscale_image = None
        self._processed_image = None
        self._clahe_image = None
        self._rotated_cache.clear()

