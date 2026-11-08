# Đặc tả Detector Interface, PageContext, DetectorRegistry & DefectResult (Phase 3)

## 1. Chuẩn Giao diện Detector (`src/pdf_inspector/detectors/base_detector.py`)

Mọi Detector trong hệ thống đều kế thừa từ `BaseDetector` (hoặc các lớp chuyên biệt `PageLevelDetector`, `DocumentLevelDetector`):

```python
class BaseDetector(ABC):
    @property
    @abstractmethod
    def detector_id(self) -> str:
        """Định danh duy nhất của detector (ví dụ: 'orientation', 'stripe', 'boundary')."""

    @property
    def version(self) -> str:
        """Phiên bản thuật toán (mặc định: '1.0.0')."""

    @property
    @abstractmethod
    def scope(self) -> str:
        """Phạm vi xử lý: 'PAGE_LEVEL', 'DOCUMENT_LEVEL', hoặc 'CROSS_PAGE'."""

    @property
    @abstractmethod
    def supported_defect_types(self) -> Tuple[str, ...]:
        """Các mã loại lỗi mà detector kiểm tra."""

    @abstractmethod
    def detect(self, context: Any) -> List[DefectResult]:
        """Thực thi kiểm tra lỗi trên context và trả về danh sách DefectResult."""
```

- **`PageLevelDetector` (`scope = "PAGE_LEVEL"`):** Nhận đầu vào `context: PageContext`, xử lý độc lập từng trang (`OrientationDetector`, `StripeDetector`).
- **`DocumentLevelDetector` (`scope = "CROSS_PAGE"`):** Phân tích quan hệ giữa một trang và các trang liền kề (`PageBoundaryDetector` kiểm tra các lỗi `MIXED_PAGE`, `SPLIT_PAGE`, `CONTENT_OVERLAP`, `CONTENT_TRUNCATED`, `PAGE_BOUNDARY_SUSPECTED`).

---

## 2. Đặc tả `PageContext` & Quy ước Tọa độ (`src/pdf_inspector/models/context.py`)

`PageContext` đóng gói toàn bộ dữ liệu của một trang PDF trong quá trình kiểm tra:

| Thuộc tính / Phương thức | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `document_id` | `str` | Tên hoặc định danh tài liệu PDF đang kiểm tra. |
| `page_number` | `int` | Số thứ tự trang (1-based: `1..N`). |
| `pdf_page_size` | `Tuple[float, float]` | Kích thước trang PDF `(width_pt, height_pt)` tính bằng đơn vị point (`1 inch = 72 pt`). |
| `original_rotation` | `int` | Góc xoay hiện tại trong metadata `/Rotate` của trang PDF (`0`, `90`, `180`, `270`). |
| `original_image` | `np.ndarray` | Ảnh trang vừa được render ở `dpi`, bị khóa quyền ghi (`writeable = False`) để tránh biến đổi trực tiếp. |
| `grayscale_image` | `np.ndarray` | Ảnh grayscale đã chuẩn hóa kích thước, được khởi tạo lười (lazy cache) khi detector đầu tiên truy cập. |
| `processed_image` | `np.ndarray` | Ảnh đã qua lọc nhiễu Median Blur + tăng tương phản CLAHE, được khởi tạo lười và chia sẻ giữa các detector. |
| `ensure_clahe()` | `np.ndarray` | Ảnh CLAHE cùng độ phân giải với `original_image` (không qua Median Blur) để bảo toàn các vệt sọc mảnh `1–2 px`. |
| `get_rotated_image(angle_cw)` | `np.ndarray` | Trả về ảnh đã xoay thử theo chiều kim đồng hồ (`90°`, `180°`, `270°`), phục vụ `PageBoundaryDetector` phân tích đúng hướng đứng khi trang bị xoay ngược. |
| `coordinate_transform` | `CoordinateTransform` | Cung cấp chuyển đổi hai chiều giữa `RENDERED_IMAGE_PX`, `NORMALIZED (0.0..1.0)`, `PDF_POINT`, và quy đổi tọa độ giữa ảnh đã xoay (`rotated_pixel_to_rendered_pixel`). |
| `release()` | `None` | Giải phóng toàn bộ các mảng ảnh khỏi bộ nhớ ngay sau khi kiểm tra xong trang. |

---

## 3. Đặc tả `DetectorRegistry` (`src/pdf_inspector/detectors/registry.py`)

Trong Phase 3, `create_default_registry()` khởi tạo đầy đủ 3 detector ở trạng thái `ENABLED` (`AVAILABLE`):
- `orientation`: `ENABLED` (`AVAILABLE`) — `TesseractOrientationDetector` (`scope = "PAGE_LEVEL"`)
- `stripe`: `ENABLED` (`AVAILABLE`) — `StripeDetector` (`scope = "PAGE_LEVEL"`)
- `boundary`: `ENABLED` (`AVAILABLE`) — `PageBoundaryDetector` (`scope = "CROSS_PAGE"`, hỗ trợ alias `"page_boundary"`)

---

## 4. Đặc tả `DefectResult` cho Page Boundary Detection (`src/pdf_inspector/models/boundary_schemas.py` & `src/pdf_inspector/models/schemas.py`)

Khi phát hiện lỗi ranh giới trang / lẫn trang, `PageBoundaryDetector` trả về `DefectResult` chuẩn hóa:

```json
{
  "defect_type": "MIXED_PAGE",
  "detector_id": "boundary",
  "status": "CONFIRMED",
  "requires_review": true,
  "confidence": 0.96,
  "suspicion_score": 0.96,
  "confidence_threshold": 0.68,
  "detection_method": "horizontal_projection+illumination_step+layout_transition",
  "bbox": [0, 868, 1320, 1003],
  "bbox_pdf_pt": [0.0, 390.6, 594.0, 451.35],
  "severity": "HIGH",
  "reason": "Phát hiện ranh giới chuyển trang ngang tại y=0.50 (bước nhảy độ sáng nền (delta=18.0), dấu hiệu số trang/footer giữa trang, lệch lề văn bản (0.08)).",
  "coordinate_space": "RENDERED_IMAGE_PX",
  "dpi": 160,
  "related_pages": [],
  "boundary_candidates": [
    {
      "candidate_id": "p1_h_1",
      "orientation": "HORIZONTAL",
      "position": 0.5003,
      "position_px": 935,
      "position_pdf_pt": 420.98,
      "coordinate_system": "normalized_page",
      "detected_by": ["horizontal_projection", "illumination_step", "layout_transition"],
      "suspicion_score": 0.96,
      "status": "CONFIRMED",
      "safe_cut_line_px": 935,
      "safe_cut_line_norm": 0.50,
      "crosses_text": false
    }
  ],
  "details": {
    "suspected_boundary_norm": 0.5003,
    "suspected_boundary_px": 935,
    "safe_cut_line_px": 935,
    "safe_cut_line_norm": 0.50,
    "crosses_text": false,
    "related_pages": [],
    "reconstruction_proposal": {
      "reconstructable": true,
      "action": "SPLIT_SINGLE_PAGE",
      "requires_user_confirmation": true,
      "safe_cut_line_norm": 0.50,
      "crosses_text": false
    }
  }
}
```
