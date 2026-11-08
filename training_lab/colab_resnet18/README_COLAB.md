# Hướng dẫn huấn luyện ResNet-18 trên Google Colab (CUDA)

Dự án: **pdf-orientation-detector** — nhận diện hướng trang PDF (0° / 90° / 180° / 270°).

Tài liệu này dành cho người **mới dùng Colab**, không cần biết Linux. Bạn chỉ cần
bấm chạy từng ô (cell); gần như không phải sửa code.

---

## ⚠️ CẢNH BÁO BẢO MẬT — ĐỌC TRƯỚC KHI UPLOAD

Dữ liệu huấn luyện là **ảnh scan trang tài liệu thật**. Trong nguồn có file tên
`..._ho_so_kiem_tra_do_mat_...` ("độ mật"), gợi ý **có thể có tài liệu đóng dấu mật**.

- **Google Colab + Google Drive là hạ tầng công cộng.** Chỉ upload khi bạn **được
  phép** xử lý dữ liệu này ở đó.
- Nếu có tài liệu thuộc **bí mật nhà nước / nội bộ nhạy cảm**, **KHÔNG** đưa lên
  Colab công cộng. Hãy loại các trang đó khỏi dataset (đánh dấu "Loại" trong công
  cụ review rồi tạo lại package) hoặc dùng hạ tầng được phép.
- Package **không** chứa PDF gốc, chỉ chứa ảnh trang đã chọn. Việc chuẩn bị package
  trên máy **không có nghĩa** là đã được phép upload.

Claude **không tự upload** bất kỳ dữ liệu nào. Quyết định upload là của bạn.

---

## Nội dung package (`pdf_orientation_colab_package.zip`)

```
pdf_orientation_colab_package/
├── images/rotations/{000,090,180,270}/*.jpg   # ảnh đã chọn (hợp lệ), đã gán nhãn
├── train_val_test_split.csv                   # nhãn + split (rel_path tương đối)
├── split_report.json                          # thống kê & kiểm tra leakage
├── class_mapping.json                         # ánh xạ lớp cố định
├── package_manifest.json
├── configs/training_config.json
├── scripts/{dataset.py,model.py,train.py,evaluate.py}
├── train_resnet18_colab.ipynb                 # notebook (mở bằng Colab)
└── README_COLAB.md
```

Package **không** gồm: PDF gốc, ảnh trùng/bị loại, ảnh upright master, manifest nội bộ.

---

## Các bước thực hiện

### 1. Mở Google Drive
Vào https://drive.google.com → đăng nhập.

### 2. Tạo thư mục lưu dataset
Tạo thư mục, ví dụ: **`My Drive / pdf_orientation`**.

### 3. Upload file ZIP
Kéo `pdf_orientation_colab_package.zip` vào thư mục vừa tạo. Đợi upload xong
(file ~ vài trăm MB, tùy mạng có thể mất ít phút).

### 4. Mở Google Colab
- Cách A: Vào https://colab.research.google.com → **File → Upload notebook** →
  chọn `train_resnet18_colab.ipynb` (có trong ZIP, giải nén ra lấy, hoặc tải riêng).
- Cách B: Trong Drive, chuột phải vào file `.ipynb` → **Open with → Google Colaboratory**.

### 5. Bật GPU (CUDA)
Trên Colab: **Runtime → Change runtime type → Hardware accelerator → GPU → Save**.
(Thường được cấp **NVIDIA T4**.)

### 6. Kiểm tra CUDA
Chạy **Cell 1**. Phải thấy `CUDA available: True` và tên GPU. Nếu `False`, quay lại
bước 5. Notebook sẽ **tự dừng** trước khi train nếu không có GPU.

### 7. Chỉnh đường dẫn ZIP
Trong **Cell 2–3**, sửa biến `PACKAGE_PATH` cho đúng vị trí ZIP của bạn, ví dụ:
```python
PACKAGE_PATH = "/content/drive/MyDrive/pdf_orientation/pdf_orientation_colab_package.zip"
```
(Chỉ cần sửa đúng **một dòng** này.)

### 8. Chạy lần lượt các cell
Bấm chạy từ trên xuống (hoặc **Runtime → Run all**):
1. Kiểm tra môi trường & CUDA.
2. Mount Drive.
3. Giải nén ZIP vào `/content/pdf_orientation/` (ổ cục bộ Colab, **không** đọc ảnh
   trực tiếp từ Drive để nhanh).
4. Kiểm tra dataset & leakage.
5. Cấu hình.
6. DataLoader.
7. Dựng ResNet-18.
8. **Smoke test** 1 batch.
9. **Huấn luyện** (hiện Train/Val Loss & Accuracy theo từng epoch).
10. Đánh giá trên **Test**.
11. Vẽ biểu đồ + confusion matrix.
12. Lưu kết quả về Drive theo `runs/run_XXX/`.

### 9. Xem biểu đồ training
Cell 11 vẽ Loss/Accuracy và confusion matrix ngay trong notebook.

### 10. Tìm checkpoint
Kết quả lưu ở Drive: **`My Drive / PDF_Orientation_Training / runs / run_XXX /`**
gồm `best_resnet18.pth`, `test_metrics.json`, `confusion_matrix.png`, báo cáo...

### 11. Tải kết quả về máy
Vào thư mục `runs/run_XXX/` trên Drive → tải về (hoặc dùng cell tạo ZIP kết quả).

### 12. Resume khi Colab bị ngắt
Nếu phiên Colab rớt giữa chừng: chạy lại Cell 1–7, rồi ở Cell huấn luyện đặt
`RESUME = True`. Notebook nạp `last_checkpoint.pth` trong run gần nhất và train tiếp.

---

## Ý nghĩa nhãn
Nhãn là **góc mà ảnh đang BỊ XOAY theo chiều kim đồng hồ** so với hướng đọc đúng.
Góc **cần xoay để sửa** trang về đúng chiều = `(360 − nhãn) % 360`
(0→0°, 90→270°, 180→180°, 270→90°). Ánh xạ này lưu trong `class_mapping.json` và
trong mỗi checkpoint.

## Thông số mặc định (chỉnh trong Cell 5 nếu cần)
`IMAGE_SIZE=224 · BATCH_SIZE=32 · EPOCHS=20 · LR=1e-4 · SEED=42`.
Nếu báo lỗi hết VRAM (OOM): giảm `BATCH_SIZE` (vd 16) rồi chạy lại cell huấn luyện.

## Sau khi có model
Checkpoint `best_resnet18.pth` dùng để tích hợp vào PDF Scan Quality Inspector ở
giai đoạn sau. **Chưa** bật tự động xoay trong Web Tool ở bước này.
