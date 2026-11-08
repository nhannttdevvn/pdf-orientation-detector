# PDF Scan Quality Inspector — Enterprise 5-Zone Document AI Architecture

Hệ thống kiểm tra chất lượng tài liệu PDF scan và nhận diện dấu độ mật chạy **100% offline trên macOS / Linux / Docker**, được tổ chức theo kiến trúc **5 phân khu chuyên biệt** (`ai_engine/`, `backend/`, `frontend/`, `training_lab/`, `benchmarks/`) lấy cảm hứng từ mô hình tách tầng `AI – Backend – Frontend – Evaluation` và mở rộng chiều sâu cho bài toán xử lý hồ sơ số hóa.

---

## 1. Kiến trúc 5 Phân khu & Cấu trúc Thư mục

```text
pdf-orientation-detector/
│
├── ai_engine/                               # 1. LÕI AI & COMPUTER VISION (Độc lập hoàn toàn với UI)
│   ├── core/                                # Cấu hình trung tâm, hằng số, logging, exceptions
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── logging_config.py
│   ├── preprocessing/                       # Đọc PDF, render ảnh trang & tiền xử lý OpenCV
│   │   ├── pdf_loader.py                    # PDFLoader & khởi tạo PageContext
│   │   ├── page_renderer.py                 # Render trang PDF sang mảng NumPy
│   │   └── image_preprocessor.py            # Median Blur, CLAHE, trích xuất dải dòng chữ
│   ├── detectors/                           # 4 nhóm thuật toán phát hiện chuyên biệt & Registry
│   │   ├── base_detector.py                 # BaseDetector, PageLevelDetector, DocumentLevelDetector
│   │   ├── registry.py                      # DetectorRegistry quản lý vòng đời detector
│   │   ├── orientation/                     # [Task 1] Phát hiện trang xoay 90°/180°/270° (OSD + Heuristic)
│   │   ├── defects/                         # [Task 2] Phát hiện sọc scan dọc/ngang/chéo & lọc đường biểu mẫu
│   │   ├── boundary/                        # [Task 3] Phát hiện lẫn trang, tách trang, cắt cụt mép
│   │   └── secret/                          # [Task 4] Nhận diện dấu MẬT / TỐI MẬT / TUYỆT MẬT / GIẢI MẬT
│   │       └── templates/                   # Bộ mẫu dấu chuẩn 200 DPI
│   ├── reconstruction/                      # Tách/ghép & phục dựng trang bị lẫn có xác nhận
│   │   └── page_reconstructor.py
│   └── pipelines/                           # Điều phối đa detector trên Shared PageContext
│       └── document_pipeline.py
│
├── backend/                                 # 2. TẦNG DỊCH VỤ, SCHEMAS, REPOSITORIES & WORKERS
│   ├── api/                                 # Giao diện dòng lệnh & cổng điều phối
│   │   └── cli.py
│   ├── schemas/                             # Data models, DTOs, PageContext, Boundary & Defect schemas
│   │   ├── context.py
│   │   ├── schemas.py
│   │   └── boundary_schemas.py
│   ├── services/                            # Business Logic Services
│   │   ├── orientation_service.py           # UnifiedQualityEngine (DecisionEngine)
│   │   ├── correction_service.py            # Xoay trang PDF bảo toàn chất lượng gốc
│   │   ├── boundary_inspection_service.py   # Kiểm định & phục dựng trang lẫn
│   │   ├── secret_scan_service.py           # Quét dấu độ mật tuần tự (an toàn RAM)
│   │   ├── report_service.py                # Xuất báo cáo JSON / CSV
│   │   └── pdf_service.py
│   ├── repositories/                        # Tiện ích lưu trữ file, hash & quản lý tài nguyên
│   │   └── file_utils.py
│   └── workers/                             # Xử lý lô (Batch folder scan & xuất ZIP báo cáo)
│       ├── batch_manager.py
│       └── batch_export.py
│
├── frontend/                                # 3. GIAO DIỆN NGƯỜI DÙNG (Streamlit Web Dashboard)
│   ├── app/                                 # Entrypoint ứng dụng Web
│   │   └── streamlit_app.py
│   ├── state/                               # Quản lý session state cách ly 2 chế độ & RAM cache
│   │   └── state_manager.py
│   ├── styles/                              # Hệ thống CSS & SVG icons chuẩn thiết kế 1534x960
│   │   └── styles.py
│   ├── pages/                               # Các trang chế độ làm việc độc lập
│   │   ├── scan_quality_page.py             # Tab 1 — Kiểm tra & sửa lỗi PDF
│   │   └── secret_detection_page.py         # Tab 2 — Kiểm tra dấu độ mật
│   └── components/                          # Các component giao diện tái sử dụng
│       ├── header_bar.py
│       ├── upload_card.py
│       ├── summary_cards.py
│       ├── results_view.py
│       ├── preview_panel.py
│       ├── secret_tab_view.py
│       ├── sidebar.py
│       └── footer_bar.py
│
├── training_lab/                            # 4. KHU VỰC HUẤN LUYỆN MÔ HÌNH (Deep Learning / ResNet-18)
│   ├── dataset_prep/                        # Kiểm toán dữ liệu (audit_dataset) & chia tập train/val/test
│   ├── colab_resnet18/                      # Pipeline huấn luyện ResNet-18 trên Google Colab & unit tests
│   └── checkpoints/                         # Trọng số mô hình & cấu hình ánh xạ nhãn (orientation/)
│
├── benchmarks/                              # 5. ĐÁNH GIÁ THỰC NGHIỆM (Generators, Evaluators & Reports)
│   ├── generators/                          # Sinh bộ dữ liệu PDF kiểm chuẩn tự động
│   ├── evaluators/                          # Đánh giá định lượng Precision / Recall / F1 / Latency
│   └── reports/                             # Báo cáo kết quả thực nghiệm & huấn luyện
│
├── desktop_tool/                            # Công cụ Desktop đóng gói độc lập (.dmg / .exe)
├── tests/                                   # Hệ thống kiểm thử tự động phân tầng (99 tests)
│   ├── unit/                                # Kiểm thử đơn vị từng detector (Orientation, Stripe, Boundary, Secret)
│   ├── integration/                         # Kiểm thử tích hợp Pipeline, Services, Reconstruction, Batch Scan
│   └── ui/                                  # Kiểm thử giao diện 2 chế độ độc lập
├── docs/                                    # Tài liệu kiến trúc & báo cáo kiểm toán kỹ thuật
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## 2. Hướng dẫn Sử dụng Nhanh

### A. Cài đặt môi trường
```bash
pip install -e ".[dev]"
```

### B. Chạy giao diện Web Dashboard (2 Chế độ)
```bash
PYTHONPATH=. streamlit run frontend/app/streamlit_app.py
```
Truy cập `http://localhost:8501` trên trình duyệt.

### C. Chạy công cụ dòng lệnh (CLI)
```bash
PYTHONPATH=. python -m backend.api.cli input.pdf --auto-fix --output data/outputs/corrected.pdf --report data/outputs/report.json
```

### D. Chạy bộ kiểm thử tự động & Benchmark
```bash
# Chạy toàn bộ kiểm thử phân tầng (Unit + Integration + UI)
PYTHONPATH=. pytest --import-mode=importlib tests/ -q

# Chạy các bộ đánh giá định lượng (Benchmarks)
PYTHONPATH=. python benchmarks/evaluators/evaluate_horizontal_stripe_benchmark.py
PYTHONPATH=. python benchmarks/evaluators/evaluate_secret_benchmark.py
PYTHONPATH=. python benchmarks/evaluators/evaluate_boundary_benchmark.py
```
