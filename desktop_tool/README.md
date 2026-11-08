# Kiểm Tra Hồ Sơ Lưu Trữ (thư mục `tool`)

Công cụ kiểm tra hồ sơ lưu trữ dạng PDF scan, chạy offline trên Windows, macOS và Linux. Đây là bản sao của `tool_tailieu`, có thêm:
- phát hiện và sửa **trang bị xoay**;
- **đối chiếu Mục lục** với văn bản gốc;
- kết luận **3 mức**;
- cột **Trang Xoay** trên giao diện ghi rõ **trang nào cần xoay**.

Giao diện giữ nguyên như bản gốc (Tkinter, `gui.py`). Bản gốc `tool_tailieu` cũng giữ nguyên.

---

## 1. Chức năng

| Chức năng | Làm gì | Mức kết luận |
|---|---|---|
| **Trang bị xoay** | Phát hiện trang xoay 90°/180°/270°. Xoay ảnh về đúng chiều trước các bước kiểm tra khác; xuất bản PDF đã sửa (không ghi đè file gốc) | Bị xoay → **Lỗi**. Không chắc → **Cần kiểm tra** (không tự xoay) |
| **Bóc tách văn bản** | Tìm trang đầu/cuối từng văn bản (tiêu ngữ, dấu đỏ, từ khóa Anh/Việt); nhận biết ngôn ngữ theo nội dung chữ | Chưa xác định đủ đầu/cuối → **Cần kiểm tra** |
| **Đối chiếu Mục lục** | Bóc tách bảng Mục lục và so với Số ký hiệu, Ngày tháng, Số tờ trên trang đầu từng văn bản | Xem mục 2 |
| **Dấu mật** | Nhận diện MẬT / TỐI MẬT / TUYỆT MẬT / GIẢI MẬT (template matching, ngưỡng 0.80) | Ghi nhận, không tính lỗi |
| **Sọc scan** | Phát hiện sọc dọc chạy ≥ 95% chiều cao trang (Hough) | **Lỗi** |

Kết luận của mỗi file là mức cao nhất trong các mục: **Lỗi** > **Cần kiểm tra** > **Không lỗi**.

## 2. Đối chiếu Mục lục — cách hoạt động

1. **Nhóm hồ sơ:** các file nối tiếp cùng tiền tố (`0056_1.pdf … 0056_7.pdf` → hồ sơ `0056`) được xử lý như một hồ sơ.
2. **Tìm Mục lục:** trang đầu file đầu tiên có tiêu đề "MỤC LỤC", cùng các trang bảng kẻ ô liền sau.
3. **Bóc tách bảng:**
   - OpenCV tìm đường kẻ ngang/dọc để dựng lưới ô;
   - RapidOCR đọc chữ cả trang rồi gán vào từng ô (ô nào bị sót thì OCR lại riêng ô đó);
   - ánh xạ cột theo tiêu đề: STT, Số ký hiệu, Ngày tháng, Trích yếu, Tác giả, Trang số, Ghi chú.
4. **Định vị văn bản:** đánh số tờ liên tục qua các file của hồ sơ, bỏ trang Mục lục. Nhờ vậy cột "Trang số" chỉ chính xác tới trang của file nào, **không cần OCR số tờ viết tay**.
5. **Đọc trang đầu văn bản:** OCR vùng đầu trang, cắt 3 vùng **Số ký hiệu**, **Ngày tháng**, **Số tờ** (góc trên phải) làm ảnh minh chứng.

**Quy tắc đánh giá:**
- **Khớp:** các trường đọc được trùng Mục lục.
- **Lỗi:** chỉ báo khi đọc **chắc chắn** mà vẫn khác: chữ đánh máy, OCR điểm ≥ 0.90, đủ chữ số, không có mực bút màu. Lỗi của chính bảng Mục lục (trùng STT, bỏ trống Trang số) cũng là Lỗi.
- **Cần kiểm tra:** số/ngày viết tay, OCR đọc thiếu chữ số, điểm OCR thấp, không đọc được, không thấy dấu đỏ. Người kiểm tra xem ảnh minh chứng để xác nhận.

**Mã lỗi:**

| Mã | Ý nghĩa |
|---|---|
| `LECH_SO_KY_HIEU` | Lệch Số/ký hiệu giữa Mục lục và văn bản gốc |
| `KHUYET_SO_KY_HIEU` | Văn bản gốc khuyết hoặc không đọc được phần số |
| `LECH_NGAY_THANG` | Lệch ngày tháng ban hành |
| `KHUYET_NGAY_THANG` | Văn bản gốc khuyết hoặc không đọc được ngày tháng |
| `LECH_SO_TRANG` | Trang Mục lục chỉ tới không phải trang đầu văn bản / không tồn tại |
| `ML_KHUYET_TRANG` | Mục lục bỏ trống cột Trang số |
| `TRUNG_STT_MUC_LUC` | Đánh trùng STT trong Mục lục |
| `VB_CHIA_CAT_QUA_FILE` | Văn bản trải qua nhiều file PDF (cảnh báo) |
| `THIEU_DAU_DO_END_DOC` | Không thấy dấu đỏ trên các trang của văn bản |

**Kết quả thử trên hồ sơ `0056`** (7 file, 32 dòng Mục lục):
- **6 Khớp:** các văn bản có số/ngày đánh máy. Trong lần chạy trọn bộ là 5 khớp, vì STT 13 có con dấu bị phai màu, bộ nhận diện dấu đỏ không thấy nên chuyển sang Cần kiểm tra.
- **26 Cần kiểm tra:** các văn bản có số/ngày viết tay.
- **0 Lỗi nhầm.**
- **Bắt đúng lỗi trùng STT 31** trong Mục lục.

Bản cũ của chức năng này báo 30/32 Lỗi, phần lớn do OCR đọc sai chữ viết tay.

## 3. Cách chạy

**Trên Windows (nhanh nhất):** bấm đúp **`CHAY_TOOL.bat`**. Lần đầu file này tự cài thư viện (cần Internet và Python 3.11), các lần sau mở giao diện ngay.

**Giao diện** (mặc định, giao diện Tkinter ban đầu):
```bash
python chay_tool.py        # hoặc: python gui.py
```
- Chọn thư mục PDF đầu vào và thư mục xuất báo cáo, rồi bấm **BẮT ĐẦU KIỂM TRA & BÓC TÁCH HỒ SƠ**.
- Cột **Trang Xoay** ghi rõ trang cần xoay, ví dụ `Trang 2 (cần xoay 90°), Trang 5 (cần kiểm tra)`. Nhật ký và hộp thông báo cuối cũng liệt kê theo từng file.
- Kết quả đối chiếu Mục lục xem trong Excel, sheet "Đối Chiếu Mục Lục".

**Dòng lệnh:**
```bash
python main.py --input data/pdf --output reports
python main.py --input data/pdf --output reports --khong-sua-xoay          # không xuất PDF đã sửa
python main.py --input data/pdf --output reports --khong-doi-chieu-muc-luc # bỏ đối chiếu Mục lục
```
Mã thoát: `0` Không lỗi, `1` Lỗi, `2` Cần kiểm tra.

**Cài thư viện** (khi chạy từ mã nguồn): `pip install -r requirement.txt`

## 4. Kết quả (`reports/`)

| File / thư mục | Nội dung |
|---|---|
| `bao_cao_kiem_tra.xlsx` | Excel 6 sheet: Theo File · Bóc Tách Văn Bản · Tài Liệu Mật · Sọc Scan · Trang Bị Xoay · **Đối Chiếu Mục Lục** |
| `ket_qua_kiem_tra_van_ban.txt` | Báo cáo chữ kết quả đối chiếu Mục lục |
| `evidence_crops/` | Ảnh trang đầu/cuối văn bản; ảnh Số ký hiệu / Ngày / Số tờ của các dòng Mục lục chưa khớp |
| `pdf_da_xoay/` | PDF đã sửa trang bị xoay |
| `trang_bi_xoay/`, `tai_lieu_mat/`, `anh_co_soc/` | Ảnh minh chứng tương ứng |

## 5. Kiểm tra trang xoay: bộ máy nào được dùng

Tool tự chọn theo thứ tự (log in ra dòng `Kiểm tra trang bị xoay bằng: ...`):
1. **Tesseract** đóng gói kèm tool (thư mục `Tesseract-OCR/`);
2. **Tesseract** cài trên máy (phải có gói `osd`);
3. **RapidOCR** (dự phòng, có sẵn): trùng Tesseract 66/66 trang thử nghiệm.

Đặt `KTHS_NO_TESSERACT=1` để luôn dùng RapidOCR.

## 6. Cấu trúc thư mục

```text
tool/
├── chay_tool.py                  # Điểm khởi chạy: giao diện gui.py (mặc định) hoặc dòng lệnh (khi có tham số)
├── gui.py                        # Giao diện (Tkinter, như bản gốc)
├── giao_dien.py                  # Giao diện PySide6 thử nghiệm — KHÔNG dùng (chay_tool.py mở gui.py)
├── main.py                       # Dòng lệnh (giữ như bản gốc)
├── kiem_tra_ho_so.py             # Bộ điều phối: quét trang, bóc tách, kết luận, xuất Excel
├── doi_chieu_muc_luc.py          # Đối chiếu Mục lục với văn bản gốc
├── xoay_trang.py                 # Phát hiện & sửa trang bị xoay
├── detect_confidential_docs_fast.py, hough_line_detector.py   # Dấu mật, sọc scan (giữ như bản gốc)
├── app_paths.py                  # Đường dẫn & mở file cho Windows/macOS/Linux
├── templates/                    # Ảnh mẫu dấu mật
├── packaging/                    # Đóng gói bản tải về chạy độc lập
└── .github/workflows/build.yml   # Build tự động 4 hệ điều hành
```

## 7. Đóng gói bản Windows tải về (không cần cài gì)

Sản phẩm: `KiemTraHoSo-windows-x64.zip`. Giải nén ra thư mục `KiemTraHoSo` gồm:
- `KiemTraHoSo.exe`: giao diện;
- `KiemTraHoSo_CLI.exe`: dòng lệnh;
- `HUONG_DAN.txt`, `data/pdf/`, `reports/`;
- Tesseract đóng gói sẵn.

**Cách build** (PyInstaller chỉ build được bản Windows trên Windows):
- **Trên một máy Windows:**
  1. Cài Python 3.11 64-bit (tick "Add python.exe to PATH"). Nên cài thêm Tesseract (UB-Mannheim).
  2. Bấm đúp `packaging\build_windows.bat`.
  3. Khoảng 5–10 phút sau, file zip nằm trong `packaging\_build\`.
- **Không có máy Windows:** đưa thư mục này lên GitHub → tab **Actions** → **Build ban chay doc lap** → **Run workflow**. Khoảng 10 phút sau, tải file zip ở mục **Artifacts**.

**Lưu ý:**
- Một số phần mềm diệt virus có thể cảnh báo nhầm với file `.exe` tạo bằng PyInstaller chưa ký số. Khi đó cần cho phép (whitelist) thư mục tool.
- Đường dẫn và tên file có dấu tiếng Việt (vd. `D:\Hồ sơ\`) đọc/ghi bình thường: tool đã vá giới hạn của OpenCV trên Windows.

## 8. Giới hạn đã biết

- **Bóc tách văn bản** dựa trên quy tắc (từ khóa, tiêu ngữ, dấu đỏ), hợp với văn bản Đảng/hành chính. Với hồ sơ loại khác (ví dụ hợp đồng tiếng Anh), kết quả chỉ mang tính gợi ý, nên luôn ở mức "Cần kiểm tra".
- **Số và ngày viết tay** không đọc được chắc chắn bằng OCR, nên luôn ra "Cần kiểm tra" kèm ảnh minh chứng.
- **Dấu mật** dùng bộ nhận diện bản cũ `FastConfidentialDocDetector`. **Sọc scan** chỉ bắt sọc dọc. Hai phần này giữ nguyên như bản gốc.
- **Dấu đỏ bị phai** (scan gần như đen trắng) không nhận ra được.
- **Thời gian xử lý** khoảng 5 giây/trang.
