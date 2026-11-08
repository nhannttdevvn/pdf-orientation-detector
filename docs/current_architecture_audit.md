# Báo cáo Kiểm toán Kiến trúc Hiện tại (Phase 1 — Current Architecture Audit)

**Thời điểm kiểm toán:** 2026-10-28  
**Trạng thái Git:** Workspace hiện nằm trên ổ đĩa ngoài `/Volumes/TRANSCEND` và chưa khởi tạo Git repository (không có thay đổi Git chưa commit nào bị rủi ro mất mát; toàn bộ dữ liệu người dùng tại `data/uploads/6b63db7ab409_0041.pdf`, `data/samples/` và `data/outputs/` được bảo vệ nguyên vẹn).  
**Trạng thái Kiểm thử Baseline:** `35 passed in 19.33s` (`pytest -v`).

---

## 1. Cây thư mục và Entry Points hiện tại

### 1.1. Cây thư mục hiện có
```text
pdf-orientation-detector/
├── app/
│   ├── __init__.py
│   ├── main.py                      # Entry point CLI
│   ├── core/
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── logging_config.py
│   ├── detectors/
│   │   ├── base_detector.py
│   │   └── tesseract_detector.py
│   ├── models/
│   │   └── schemas.py
│   ├── processing/
│   │   ├── image_preprocessor.py
│   │   └── page_renderer.py
│   ├── services/
│   │   ├── correction_service.py
│   │   ├── orientation_service.py
│   │   ├── pdf_service.py
│   │   └── report_service.py
│   ├── ui/
│   │   ├── streamlit_app.py         # Entry point Streamlit UI
│   │   └── components/
│   │       ├── preview_panel.py
│   │       ├── results_view.py
│   │       └── sidebar.py
│   └── utils/
│       └── file_utils.py
├── data/
│   ├── samples/                     # 12 file PDF mẫu + ground_truth.json
│   ├── uploads/                     # File PDF người dùng tải lên (6b63db7ab409_0041.pdf)
│   └── outputs/                     # corrected.pdf, orientation_report.json, orientation_report.csv
├── docs/
│   ├── architecture.md
│   └── workflow.md
├── logs/
│   └── app.log
├── scripts/
│   └── generate_samples.py
├── tests/
│   ├── conftest.py
│   ├── test_correction.py
│   ├── test_orientation.py
│   ├── test_pdf_service.py
│   └── test_pipeline.py
└── main.py                          # Wrapper CLI ở thư mục gốc
```

### 1.2. Các Entry Points
1. **Giao diện Web (Streamlit):** `app/ui/streamlit_app.py` (chạy bằng `streamlit run app/ui/streamlit_app.py`).
2. **Giao diện Dòng lệnh (CLI):** `app/main.py` và wrapper tương thích ngược `main.py` (chạy bằng `python -m app.main` hoặc `python main.py`).
3. **Script tạo dữ liệu mẫu:** `scripts/generate_samples.py`.

---

## 2. Phân tích Luồng Xử lý và Các Module Hiện tại

1. **Đọc & Render PDF (`app/services/pdf_service.py` & `app/processing/page_renderer.py`):**
   - Sử dụng **PyMuPDF (`pymupdf as fitz`)** thông qua lớp `PDFLoader` để kiểm tra tính hợp lệ của file PDF, phát hiện mật khẩu, kiểm tra chữ ký số (`Digital Signature`), đọc góc xoay metadata hiện tại (`page.rotation`) và render tuần tự từng trang ở `160 DPI` sang mảng NumPy (`render_page_pixmap`), giải phóng Pixmap và bộ nhớ đệm MuPDF sau mỗi trang.
2. **Tiền xử lý ảnh (`app/processing/image_preprocessor.py`):**
   - Sử dụng **OpenCV (`cv2`)** và **NumPy** để chuẩn hóa kích thước (`normalize_dimensions`), lọc nhiễu hạt (`denoise_image` với Median Blur $3 \times 3$), tăng tương phản cục bộ (`enhance_contrast_clahe`) và trích xuất dải dòng chữ theo hình thái học (`extract_text_line_canvas`).
3. **Nhận diện hướng trang (`app/detectors/tesseract_detector.py`):**
   - Sử dụng **Tesseract OCR (`pytesseract`)**:
     - Lần thử 1 (`tesseract_osd`): Chạy `--psm 0` trên ảnh grayscale gốc.
     - Lần thử 2 (`tesseract_osd_preprocessed`): Khi lần 1 trả về `UNKNOWN` hoặc `orientation_conf < threshold` (`10.0`), chạy lại `--psm 0` trên ảnh đã lọc nhiễu + CLAHE; nếu vẫn yếu, chạy kiểm định 2 chiều trên các dải dòng văn bản (`_run_strip_ocr_verification` với `--psm 6`).
   - Hiện tại **chưa tích hợp PaddleOCR**.
4. **Decision Engine & Logic `REVIEW` (`app/services/orientation_service.py`):**
   - Phân loại từng trang thành `NORMAL` (`rotate == 0°`, `conf >= threshold`, không xung đột), `ROTATED` (`rotate` $\in \{90°, 180°, 270°\}$, `conf >= threshold`, không xung đột) hoặc `REVIEW` (`status == UNKNOWN`, `conf < threshold`, hoặc `has_conflict == True`).
   - Nguyên tắc an toàn: Tuyệt đối không tự động xoay trang `REVIEW`; cho phép người dùng ghi đè góc xoay thủ công (`manual_override`).
5. **PDF Corrector (`app/services/correction_service.py`):**
   - Sử dụng `page.set_rotation((existing_rotation + correction_angle) % 360)` để xoay metadata `/Rotate` của trang PDF mà không rasterize lại nội dung, chống ghi đè lên file PDF gốc.
6. **Report Generator (`app/services/report_service.py`):**
   - Xuất báo cáo ra `orientation_report.json` và `orientation_report.csv`.

---

## 3. Các hạn chế hiện tại so với Kiến trúc Multi-Defect PDF Quality Inspection

1. **Thiết kế đơn lỗi (Single-Defect Coupling):**
   - `PDFOrientationPipeline` trong `app/services/orientation_service.py` hiện đang gắn chặt với bài toán Orientation duy nhất, chưa tách thành `DocumentPipeline` tổng quát trong `app/pipeline/`.
   - Chưa có `DetectorRegistry` để quản lý danh sách các detector theo trạng thái (`AVAILABLE`, `ENABLED`, `DISABLED`, `NOT_IMPLEMENTED`) và phạm vi (`PAGE_LEVEL`, `DOCUMENT_LEVEL` / `CROSS_PAGE`).
2. **Dữ liệu `PageContext` chưa đầy đủ cho chia sẻ đa Detector:**
   - `PageContext` hiện tại chỉ lưu `page_number`, `page_index`, `existing_rotation`, `width_pt`, `height_pt`, `image`.
   - Khi có nhiều detector cùng chạy trên một trang, nếu không chuẩn hóa `PageContext` (gồm `document_id`, `page_number`, `pdf_page_size`, `original_rotation`, `original_image`, `grayscale_image`, `processed_image`, `dpi`, `metadata`, `coordinate_transform`), mỗi detector sẽ phải tự render hoặc tự tiền xử lý lại ảnh, gây tốn CPU/RAM.
3. **Kết quả đầu ra chưa theo chuẩn Multi-Label Defect (`Unified Quality Result`):**
   - Schema báo cáo hiện tại chỉ xoay quanh trạng thái đơn (`NORMAL`, `ROTATED`, `REVIEW`) của Orientation.
   - Chưa có cấu trúc `defects: List[DefectResult]`, `inspection_status`, `requires_review`, `enabled_detectors`, `unsupported_checks` để biểu diễn:
     - Một trang có thể có nhiều lỗi đồng thời (Multi-label Defect Detection).
     - Một trang không có lỗi Orientation (`NO_DEFECT` ở `orientation`) **không** đồng nghĩa trang đó hoàn toàn sạch lỗi (`CLEAN` toàn diện), vì các kiểm tra `stripe` (sọc scan) và `boundary` (lệch/mất biên trang) đang ở trạng thái `NOT_IMPLEMENTED` (`unsupported_checks`).
4. **Cấu trúc thư mục Detector:**
   - `app/detectors/` hiện đặt trực tiếp `tesseract_detector.py` thay vì phân nhóm theo loại lỗi `app/detectors/orientation/`.

---

## 4. Kế hoạch Chuyển đổi (Migration Plan — Phases 2 đến 10)

1. **Phase 2 — Reorganize Project Structure:**
   - Tổ chức lại `app/` với đầy đủ `app/core/`, `app/pipeline/`, `app/detectors/orientation/`, `app/processing/`, `app/services/`, `app/models/`, `app/ui/`, `app/utils/`.
   - Đặt `pdf_loader.py` trong `app/processing/` (và giữ re-export tại `app/services/pdf_service.py` để bảo toàn 100% tương thích ngược).
2. **Phase 3 & 4 — Standardize Detector Interface, Registry & DocumentPipeline:**
   - Xây dựng `BaseDetector` (`detector_id`, `version`, `scope`, `supported_defect_types`, `detect(context)`) và `PageLevelDetector` / `DocumentLevelDetector` extension point.
   - Xây dựng `DetectorRegistry` quản lý các trạng thái `AVAILABLE`, `ENABLED`, `DISABLED`, `NOT_IMPLEMENTED` (ghi nhận rõ `stripe` và `boundary` là `NOT_IMPLEMENTED`, tuyệt đối không thực thi hoặc báo cáo giả là đã kiểm tra).
   - Xây dựng `DocumentPipeline` trong `app/pipeline/document_pipeline.py` điều phối từ PDF Input $\rightarrow$ `PDFLoader` $\rightarrow$ `PageContext` $\rightarrow$ `DetectorRegistry` $\rightarrow$ `UnifiedQualityEngine` $\rightarrow$ `QualityReport` $\rightarrow$ `CorrectionService`.
3. **Phase 5 — Shared `PageContext` & Coordinate Convention:**
   - Nâng cấp `PageContext` hỗ trợ tạo sẵn hoặc lazy-cache `original_image`, `grayscale_image`, `processed_image`, `pdf_page_size`, `original_rotation`, `dpi`, `metadata`, `coordinate_transform` cho từng trang và giải phóng ngay sau khi các page-level detectors hoàn tất trên trang đó.
4. **Phase 6 — Refactor Existing Orientation Detection into `app/detectors/orientation/`:**
   - Di chuyển thuật toán nhận diện hướng vào `app/detectors/orientation/` (`TesseractOrientationDetector` / `OrientationDefectDetector`), giữ nguyên 100% thuật toán Tesseract OSD + OpenCV Strip Verification và ngưỡng confidence `10.0`.
5. **Phase 7 & 8 — Unified Quality Result & Correction Architecture:**
   - Chuẩn hóa `DefectResult`, `PageQualityResult`, `DocumentQualityReport` hỗ trợ đa lỗi (`defects`), `inspection_status`, `requires_review`, `enabled_detectors`, `unsupported_checks`, đồng thời giữ đầy đủ các trường tương thích ngược với schema báo cáo JSON/CSV hiện có.
   - Tách biệt rõ Detection và Correction (`RotationCorrector` / `PDFCorrector` chỉ thực thi cho lỗi `ROTATION` đã xác nhận hoặc có `manual_override`).
6. **Phase 9 & 10 — Update UI/CLI, Regression Testing & Documentation:**
   - Cập nhật Streamlit UI và CLI sang `DocumentPipeline`, hiển thị rõ trạng thái Detector (`Orientation: ACTIVE`, `Stripe: NOT IMPLEMENTED`, `Boundary: NOT IMPLEMENTED`) và cảnh báo phạm vi kiểm tra.
   - Chạy kiểm thử toàn diện và đối chiếu kết quả Regression trước/sau refactor trên toàn bộ tập PDF mẫu.
   - Hoàn thiện bộ tài liệu `docs/architecture.md`, `docs/workflow.md`, `docs/detector_interface.md`, `docs/future_detectors.md` và `README.md`.

