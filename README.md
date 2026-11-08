# PDF Quality Inspection — Multi-Defect Architecture (Orientation, Stripe & Page Boundary Detector — Phase 3)

Hệ thống kiểm tra chất lượng tài liệu PDF scan chạy **offline trên macOS / Linux / Docker**, được thiết kế theo kiến trúc **Multi-Defect PDF Quality Inspection** (`DocumentPipeline`, `DetectorRegistry`, `Shared PageContext`, `UnifiedQualityEngine`).

- **Các Detector đang hoạt động (`ACTIVE / ENABLED`):**
  1. **Orientation Detector (`src/pdf_inspector/detectors/orientation/` — Phase 1):** Phát hiện trang bị xoay `90°`, `180°`, `270°` bằng **Tesseract OSD (`--psm 0`)** kết hợp tiền xử lý **OpenCV (`Median Blur` + `CLAHE` + `Morphological Text-Line Strip Verification`)** và tự động xoay chỉnh sửa metadata `/Rotate` không làm giảm chất lượng PDF.
  2. **Stripe Detector (`src/pdf_inspector/detectors/defects/` — Phase 2):** Tự động phát hiện các vệt sọc máy scan (`VERTICAL`, `HORIZONTAL`, `DIAGONAL`, `BAND`) bằng sự kết hợp của 3 kỹ thuật Computer Vision (**Morphological Operations**, **Probabilistic Hough Line Transform `HoughLinesP`**, **Projection Profile / Intensity Analysis**) kèm bộ lọc False Positive (`StripeValidator`) và hiển thị lớp phủ trực quan (`render_stripe_overlay`).
  3. **Page Boundary & Mixed-Page Detector (`src/pdf_inspector/detectors/boundary/` — Phase 3):** Tự động phát hiện các lỗi ranh giới trang và lẫn trang khi scan:
     - `MIXED_PAGE`: Một trang PDF chứa đồng thời phần cuối của trang A và phần đầu của trang B (hoặc 2 tờ giấy bị kéo chồng trong khay ADF).
     - `SPLIT_PAGE`: Một trang tài liệu vật lý bị chia tách thành 2 trang PDF liên tiếp (`N` và `N+1`).
     - `CONTENT_OVERLAP`: Hai trang liên tiếp bị trùng lặp dải nội dung ở vùng biên.
     - `CONTENT_TRUNCATED`: Nội dung văn bản ở sát mép trang (`TOP`, `BOTTOM`, `LEFT`, `RIGHT`) bị cắt cụt vật lý (`MISSING_CONTENT` / `REQUIRES_RESCAN`).
     - `PAGE_BOUNDARY_SUSPECTED`: Ranh giới trang nghi vấn cần người dùng kiểm tra thủ công.
     - Hỗ trợ **Xem trước Tách/Ghép trang an toàn (`PageReconstructor`)** không cắt ngang chữ (`crosses_text = False`) và chỉ xuất file `reconstructed.pdf` (kèm hồ sơ truy vết `reconstruction_provenance.json`) khi người dùng chủ động xác nhận (`user_confirmed = True`).

---

## 1. Cấu trúc Thư mục Dự án

```text
pdf-orientation-detector/
│
├── src/
│   └── pdf_inspector/                       # Package chính (src layout — import bằng `pdf_inspector.*`)
│       ├── __init__.py
│       ├── __main__.py                      # Cho phép chạy `python -m pdf_inspector`
│       ├── cli.py                           # Entry point giao diện dòng lệnh (CLI)
│       ├── core/                            # Cấu hình (env-overridable), logging, exceptions
│       │   ├── config.py
│       │   ├── logging_config.py
│       │   └── exceptions.py
│       ├── models/                          # Schema dữ liệu, PageContext và kết quả đa lỗi
│       │   ├── context.py                   # PageContext, CoordinateTransform, CoordinateSpace
│       │   ├── schemas.py                   # DefectResult, PageDecision, PipelineResult...
│       │   └── boundary_schemas.py          # BoundaryDefectCode, BoundaryStatus, BoundaryCandidate...
│       ├── processing/                      # Đọc PDF, render trang, tiền xử lý ảnh & tái tạo trang
│       │   ├── pdf_loader.py                # PDFLoader & khởi tạo PageContext
│       │   ├── page_renderer.py             # Render trang PDF sang NumPy array
│       │   ├── image_preprocessor.py        # Tiền xử lý OpenCV & trích xuất dòng chữ
│       │   └── page_reconstructor.py        # Xem trước tách/ghép & xuất PDF tái tạo có xác nhận
│       ├── detectors/                       # Các thuật toán nhận diện lỗi & DetectorRegistry
│       │   ├── base_detector.py             # BaseDetector, PageLevelDetector, DocumentLevelDetector
│       │   ├── registry.py                  # DetectorRegistry (AVAILABLE/ENABLED/DISABLED/NOT_IMPLEMENTED)
│       │   ├── tesseract_detector.py        # Re-export tương thích ngược
│       │   ├── orientation/                 # Orientation Detector [Phase 1]
│       │   ├── defects/                     # Stripe Defect Detector [Phase 2]
│       │   ├── boundary/                    # Page Boundary & Mixed-Page Detector [Phase 3]
│       │   └── secret/                      # Secret Marking Detector [Phase 4]
│       │       └── templates/               # Ảnh mẫu dấu độ mật (đóng gói kèm package-data)
│       ├── pipeline/
│       │   └── document_pipeline.py         # DocumentPipeline (Multi-Defect Orchestrator)
│       ├── services/                        # Nghiệp vụ: quyết định, chỉnh sửa, báo cáo, secret scan
│       │   ├── orientation_service.py       # UnifiedQualityEngine (DecisionEngine)
│       │   ├── correction_service.py        # RotationCorrector (PDFCorrector)
│       │   ├── boundary_inspection_service.py
│       │   ├── secret_scan_service.py
│       │   ├── report_service.py            # ReportGenerator (JSON & CSV)
│       │   └── pdf_service.py               # Re-export PDFLoader tương thích ngược
│       ├── ui/                              # Giao diện Web Streamlit
│       │   ├── streamlit_app.py
│       │   ├── state_manager.py
│       │   ├── styles.py
│       │   └── components/                  # header, sidebar, upload, summary, preview, results, secret tab...
│       └── utils/
│           └── file_utils.py
│
├── tests/                                   # Pytest (pythonpath = src + gốc dự án, cấu hình trong pyproject.toml)
├── scripts/                                 # Sinh dữ liệu mẫu (generate_*) & chạy benchmark (evaluate_*)
├── data/
│   ├── samples/                             # PDF mẫu + ground truth (orientation, stripe, horizontal stripe, boundary)
│   ├── uploads/                             # File người dùng upload qua UI (runtime, không commit)
│   └── outputs/                             # PDF đã sửa, báo cáo JSON/CSV, feedback & kết quả benchmark (runtime)
├── docs/                                    # Kiến trúc, workflow, đặc tả detector & báo cáo benchmark
├── logs/                                    # File log (app.log)
├── .streamlit/config.toml                   # Theme Streamlit (đọc từ thư mục chạy lệnh)
├── .env.example                             # Mẫu biến môi trường (copy thành .env)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                           # Metadata, dependencies, console script, cấu hình pytest
├── requirements.txt                         # Dependencies cho Docker / cài nhanh
└── README.md
```

Cài đặt cho phát triển (khuyến nghị — giúp import `pdf_inspector` ở mọi nơi):
```bash
pip install -e ".[dev]"
```

---

## 2. Hướng dẫn Sử dụng

### A. Chạy giao diện Web (Streamlit)
```bash
streamlit run src/pdf_inspector/ui/streamlit_app.py
```
Truy cập địa chỉ `http://localhost:8501` trên trình duyệt.

### B. Chạy công cụ dòng lệnh (CLI)
```bash
# Quét kiểm tra chất lượng toàn diện (Orientation + Stripe + Boundary) và xuất báo cáo JSON/CSV
python -m pdf_inspector data/samples/boundary_benchmark/15_multi_page_multi_defect.pdf --report data/outputs/report.json

# Quét, tự động sửa các trang ROTATED và lưu PDF mới
python -m pdf_inspector data/samples/06_all_four_orientations.pdf \
  --dpi 160 \
  --threshold 10 \
  --auto-fix \
  --output data/outputs/corrected.pdf \
  --report data/outputs/orientation_report.json
```

### C. Chạy kiểm thử tự động và Benchmark
```bash
# Chạy toàn bộ bộ kiểm thử (Orientation + Stripe + Boundary + Cross-Page + Reconstruction + Pipeline)
pytest --import-mode=importlib -v

# Chạy đánh giá định lượng Stripe Benchmark (Phase 2)
python scripts/evaluate_stripe_benchmark.py

# Chạy đánh giá định lượng Page Boundary Benchmark (Phase 3)
python scripts/evaluate_boundary_benchmark.py
```

### D. Chạy với Docker / Docker Compose
```bash
# Build và khởi chạy dịch vụ qua Docker Compose
docker compose up --build -d

# Hoặc chạy kiểm tra CLI trực tiếp trong container
docker run --rm pdf-orientation-detector python -m pdf_inspector data/samples/boundary_benchmark/02_two_pages_in_one_image.pdf
```
