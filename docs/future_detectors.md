# Thiết kế Mở rộng Tương lai — Post-Phase 3 Roadmap

> **Trạng thái hiện tại sau Phase 3:**
> - **Orientation Detector (`orientation` — Phase 1):** Đã triển khai và đang hoạt động (`ENABLED`).
> - **Stripe Detector (`stripe` — Phase 2):** Đã triển khai và đang hoạt động (`ENABLED`).
> - **Page Boundary & Mixed-Page Detector (`boundary` — Phase 3):** Đã triển khai hoàn chỉnh (`src/pdf_inspector/detectors/boundary/`, `src/pdf_inspector/processing/page_reconstructor.py`) và đang hoạt động (`ENABLED`).

---

## 1. Các Hướng Mở rộng Tiếp theo (Future Phases)

### 1.1. Fine-grained Deskew & Dewarping Detector (`detector_id = "deskew_dewarp"`)
- Phát hiện góc nghiêng nhỏ (`0.5° – 12°` deskew angle) và độ cong gáy sách 3D (page curl / warping) trên tài liệu đóng quyển.
- Tích hợp vào `DetectorRegistry` dưới dạng `PageLevelDetector`.

### 1.2. Blur, Low-Contrast & Bleed-Through Detector (`detector_id = "image_fidelity"`)
- Phát hiện trang bị nhòe nét (motion blur / out-of-focus qua Laplacian variance), trang quá mờ hoặc bị thấm mực mặt sau (ink bleed-through trên giấy mỏng).

### 1.3. Optional Deep Layout / Document Segmentation Plugin
- Mở rộng `OptionalLayoutModelVerifier` (`src/pdf_inspector/detectors/boundary/boundary_candidate_detector.py`) để hỗ trợ thêm các mô hình phân vùng bố cục tài liệu chuyên sâu (như DocLayout-YOLO / PP-Structure) chạy hậu kiểm trên các trang có `boundary_status == "BOUNDARY_SUSPECTED"` khi chạy trên máy chủ có GPU.
