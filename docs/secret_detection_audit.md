# Báo Cáo Kiểm Toán & Đánh Giá Thuật Toán Nhận Diện Dấu Độ Mật (`Part C1`)

> **Tài liệu kỹ thuật Phase 4 — PDF Scan Quality Inspector**  
> Phạm vi: Kiểm toán mã nguồn gốc (`detect_confidential_docs.py` vs `detect_confidential_docs_fast.py`), kiểm tra bộ template chuẩn `200 DPI`, đối chiếu thực nghiệm trên dữ liệu biến dạng có kiểm chứng trực quan và thiết kế kiến trúc tích hợp `SecretMarkingDetector`.

---

## 1. Thành Phần Nguồn Đã Kiểm Toán

Các thành phần đã được đọc và kiểm tra trực tiếp trên hệ thống:

1. `detect_confidential_docs.py` (`399` dòng) — Phiên bản gốc (`ConfidentialDocDetector`).
2. `detect_confidential_docs_fast.py` (`460` dòng) — Phiên bản tối ưu tốc độ (`FastConfidentialDocDetector`).
3. `templates/` — Bộ `13` ảnh PNG chuẩn ở `200 DPI`:
   - Có khung viền: `box_mat.png` (`138×72`), `mat_box.png` (`154×79`), `box_toi_mat.png` (`226×76`), `box_tuyet_mat.png` (`293×72`), `box_giai_mat.png` (`244×77`).
   - Không khung viền: `text_mat.png` (`118×62`), `mat_text.png` (`107×56`), `mat_text_bold.png`, `mat_text_super_bold.png`, `mat_text_thin.png`, `text_toi_mat.png` (`206×66`), `text_tuyet_mat.png` (`273×62`), `text_giai_mat.png` (`224×67`).
4. `test_distortions/` — `13` ảnh trang văn bản biến dạng (`01_mo_chu.png` .. `13_co_ca_mat_va_giai_mat.png`) đã kiểm chứng trực quan bằng mắt từng ảnh (không suy diễn nhãn từ tên thư mục).
5. Tuân thủ nghiêm ngặt quy tắc bảo mật `Part C4`: Tuyệt đối không ghi log nội dung văn bản từ thư mục `tài liệu mật/`, không đưa tài liệu nhạy cảm vào Git/Docker.

---

## 2. Phân Tích Chi Tiết Hai Phiên Bản Mã Nguồn Gốc

### 2.1. Bản gốc `detect_confidential_docs.py` (`ConfidentialDocDetector`)

- **Cơ chế xử lý:**
  - Chuyển ảnh sang đa mức xám đơn kênh (`cv2.cvtColor(..., COLOR_BGR2GRAY)`), cắt vùng góc trên bên trái cố định `55% × 35%` (`h * 0.35`, `w * 0.55`), chỉ đọc **trang đầu tiên (`page_num=0`)** của PDF.
  - Kết hợp dò contour khung chữ nhật (`cv2.RETR_LIST`, `approxPolyDP` 4 đỉnh) và quét trượt `cv2.matchTemplate(..., TM_CCOEFF_NORMED)` ở `5` tỷ lệ `(0.75, 0.88, 1.0, 1.12, 1.25)`.
  - Nếu điểm tối đa `< 0.78`, chạy nhánh nâng tương phản CLAHE (`clipLimit=3.0`) trên toàn ROI và quét trượt lại toàn bộ template.
- **Các lỗi nghiêm trọng phát hiện khi kiểm toán mã nguồn:**
  1. **Lỗi ánh xạ nhãn chuỗi con (`Bug nghiêm trọng tại dòng 48–70`):** Từ điển `file_map` đặt khóa `"mat": "MẬT"` ở đầu tiên và dùng lệnh `if key in fname.lower(): ... break`. Vì các file `box_toi_mat.png`, `box_tuyet_mat.png`, `box_giai_mat.png` đều chứa chuỗi con `"mat"`, **cả 13/13 template đều bị gom nhầm vào nhãn `"MẬT"`** (`"TỐI MẬT": 0`, `"TUYỆT MẬT": 0`, `"GIẢI MẬT": 0`). Hệ quả là bản gốc hoàn toàn không thể trả về nhãn `TỐI MẬT` hay `TUYỆT MẬT`.
  2. **Không kiểm tra trang `2..N` và bỏ sót góc trên bên phải:** Nhiều văn bản đóng dấu độ mật ở góc trên bên phải (`x > 0.55 * W`) hoặc ở phụ lục giữa file PDF nhiều trang đều bị bỏ lọt 100%.
  3. **Nguy cơ báo nhầm từ `"mật"` trong tiêu đề văn bản thường:** Nhánh quét trượt không kiểm tra khoảng trắng cô lập hai bên dòng cơ sở (`baseline`), dễ khớp nhầm chữ `"MẬT"` in hoa trong tiêu đề `"BÁO CÁO CÔNG TÁC BẢO MẬT"`.
  4. **Độ trễ cao:** Quét trượt toàn vùng ROI khiến các trang nhiễu/không khung tốn trung bình `~1,500 ms/trang`.

### 2.2. Bản tối ưu `detect_confidential_docs_fast.py` (`FastConfidentialDocDetector`)

- **Cải tiến so với bản gốc:**
  - Đã sửa thứ tự `file_map` thành danh sách ưu tiên từ dài đến ngắn (`giai_mat` -> `tuyet_mat` -> `toi_mat` -> `mat`), nạp đúng `7` mẫu `MẬT`, `2` mẫu `TỐI MẬT`, `2` mẫu `TUYỆT MẬT`, `2` mẫu `GIẢI MẬT`.
  - Bổ sung chiến lược **Box-First Verification** (tìm hộp chữ nhật trước rồi chỉ so khớp bên trong hộp ở 3 tỷ lệ `0.9, 1.0, 1.1`), rút thời gian xử lý các mẫu có khung rõ xuống còn `~70–90 ms`.
- **Các hạn chế & lỗi còn tồn tại trong bản `fast`:**
  1. **Lỗi lệch độ phân giải khi chạy đa luồng (`dòng 413`):** Hàm worker khởi tạo `FastConfidentialDocDetector(..., dpi=150)` nhưng bộ `templates/` được cắt ở `200 DPI` (tỷ lệ thực `0.75x`), trong khi Box-First chỉ thử `scales = [0.9, 1.0, 1.1]` nên trượt toàn bộ nhánh nhanh khi quét PDF thực!
  2. **Vẫn chỉ đọc trang 1 (`page_num=0`) và cắt cứng `60% × 35%` góc trái.**
  3. **Mù màu do dùng `BGR2GRAY` đơn kênh:** Khi con dấu tròn màu đỏ đóng đè lên khung dấu đen (`04_combo_mo_va_de_dau.png`), nét dấu đỏ làm đứt contour hộp đen trên kênh xám; đồng thời dấu độ mật đóng bằng mực đỏ nhạt bị giảm tương phản mạnh.
  4. **Chỉ trả về tối đa 1 dấu trên mỗi trang (`13_co_ca_mat_va_giai_mat.png`):** Khi trang có cả hộp `[MẬT]` và hộp `[GIẢI MẬT]`, bản `fast` ghi đè và chỉ trả về `"GIẢI MẬT"`, làm mất thông tin văn bản từng mang dấu `"MẬT"`.
  5. **Không có trạng thái `REVIEW` và bỏ sót dấu bị gạch xóa nặng (`07_gach_xoa_nang.png`):** Cả bản thường (`conf=0.607` khớp nhầm chữ `"KẾT"`) lẫn bản `fast` (`conf=0.598`) đều kết luận `"BÌNH THƯỜNG"` thay vì đưa vào diện `Cần kiểm tra (REVIEW)`.

---

## 3. So Sánh Thực Nghiệm Trên Cùng Bộ Dữ Liệu

### 3.1. Tập 13 ảnh biến dạng `test_distortions/` (Nhãn kiểm chứng trực quan)

| File mẫu | Bản gốc (`std`) | Bản nhanh (`fast`) | `SecretMarkingDetector` (Tích hợp Phase 4) |
|---|---|---|---|
| `01_mo_chu.png` | `MẬT` (`0.887`, `282ms`) | `MẬT` (`0.887`, `475ms`) | **`DETECTED` — `MẬT`** (`0.841`, `CLAHE_ENHANCED`) |
| `02_de_dau_do.png` | `MẬT` (`0.877`, `1511ms`) | `MẬT` (`0.876`, `217ms`) | **`DETECTED` — `MẬT`** (`0.970`, `RED_SUPPRESSED`) |
| `03_but_phe_de_len.png` | `MẬT` (`0.732`, `1520ms`) | `MẬT` (`0.731`, `75ms`) | **`DETECTED` — `MẬT`** (`0.732`, `BOX_TEMPLATE_MATCH`) |
| `04_combo_mo_va_de_dau.png` | `MẬT` (`0.840`, `1486ms`) | `MẬT` (`0.840`, `1179ms`) | **`DETECTED` — `MẬT`** (`0.912`, `RED_SUPPRESSED`) |
| `05_photocopy_net_dut.png` | `MẬT` (`0.801`, `1531ms`) | `MẬT` (`0.801`, `73ms`) | **`DETECTED` — `MẬT`** (`0.801`, `BOX_TEMPLATE_MATCH`) |
| `06_font_arial_khong_chan.png` | `MẬT` (`0.809`, `1536ms`) | `MẬT` (`0.807`, `71ms`) | **`DETECTED` — `MẬT`** (`0.811`, `BOX_TEMPLATE_MATCH`) |
| `07_gach_xoa_nang.png` | **BỎ SÓT** (`BÌNH THƯỜNG`) | **BỎ SÓT** (`BÌNH THƯỜNG`) | **`REVIEW` — `MẬT`** (`0.753`, khử nét gạch xóa nội hộp) |
| `08_chu_dam_net_to.png` | `MẬT` (`0.923`, `267ms`) | `MẬT` (`0.923`, `68ms`) | **`DETECTED` — `MẬT`** (`0.923`, `BOX_TEMPLATE_MATCH`) |
| `09_chu_nhat_net_manh.png` | `MẬT` (`0.901`, `490ms`) | `MẬT` (`0.901`, `596ms`) | **`DETECTED` — `MẬT`** (`0.901`, `BOX_TEMPLATE_MATCH`) |
| `10_sieu_dam_loang_muc.png` | `MẬT` (`0.783`, `1493ms`) | `MẬT` (`0.788`, `70ms`) | **`DETECTED` — `MẬT`** (`0.774`, `BOX_TEMPLATE_MATCH`) |
| `11_sieu_nhat_het_muc.png` | `MẬT` (`1.000`, `140ms`) | `MẬT` (`1.000`, `211ms`) | **`DETECTED` — `MẬT`** (`1.000`, `BOX_TEMPLATE_MATCH`) |
| `12_nhieu_hop_den_hoa_toc.png` | `MẬT` (`1.000`, `152ms`) | `MẬT` (`1.000`, `175ms`) | **`DETECTED` — `MẬT`** (`1.000`, loại trừ `[HỎA TỐC]`) |
| `13_co_ca_mat_va_giai_mat.png` | Chỉ thấy `MẬT` (`1` dấu) | Ghi đè thành `GIẢI MẬT` | **`DETECTED` — `MẬT`** (`marks_count=2`, ưu tiên độ mật cao) |
| **Tổng kết Recall (`13` mẫu)** | `12/13` (`92.31%`, mất nhãn `TỐI/TUYỆT MẬT`) | `11/13` đúng nhãn `MẬT` (`84.62%`) | **`13/13` (`100.0%`)** |

### 3.2. Tập Benchmark Chuẩn 16 Kịch Bản (`Part C5`)

Kết quả chạy `scripts/evaluate_secret_benchmark.py` (lưu tại `data/outputs/secret_detector_benchmark.json`):

- **Tổng số mẫu:** `16` (`12` mẫu dương tính: `MẬT`, `TỐI MẬT`, `TUYỆT MẬT`, dấu đỏ, dấu đen, dấu mờ `18%` alpha, dấu xoay `+8°` / `-9°`, dấu thu nhỏ `0.65x`, phóng lớn `1.35x`; `4` mẫu âm tính: trang sạch, tiêu đề chứa `"BẢO MẬT"`, tiêu đề chứa `"MẬT ĐỘ"`, hộp hành chính `[HỎA TỐC]`).
- **Kết quả định lượng:** **`TP = 12`, `TN = 4`, `FP = 0`, `FN = 0` -> `Precision = 1.0000`, `Recall = 1.0000`, `F1 = 1.0000`**.
- **Độ trễ trung bình:** `~40–150 ms/trang` cho các dấu có khung viền (nhánh `Stage 1 Box-First`); trung bình toàn bộ 16 kịch bản đạt `765.7 ms/trang`.

---

## 4. Kiến Trúc Tích Hợp Đã Lựa Chọn (`SecretMarkingDetector` & `SecretScanService`)

1. **Tái sử dụng có chọn lọc:** Giữ nguyên toàn bộ `13` template chuẩn `200 DPI` từ gói công cụ gốc và tư tưởng `Box-First Verification` của bản `fast`, nhưng thiết kế lại hoàn toàn pipeline xử lý đa kênh tại `app/detectors/secret/secret_detector.py`:
   - **Đa kênh màu thông minh:** `GRAY` + `RED_INK_ENHANCED = min(R, G, B)` (làm sậm tối đa cả dấu mực đỏ, xanh lá `GIẢI MẬT`, xanh dương) + `RED_SUPPRESSED = R` (triệt tiêu con dấu tròn đỏ đè lên dấu đen) + `CLAHE_ENHANCED` (bắt dấu phai mực mờ).
   - **Stage 1 (`Box-First Multi-Scale + Per-Box Deskew + Scribble Inpaint Recovery`):** Quét toàn bộ chiều ngang vùng Header (`40%` đầu trang) và Footer (`32%` cuối trang), sửa lỗi góc `minAreaRect` `90°`, khử nghiêng cục bộ từng hộp dấu và hỗ trợ khôi phục nét gạch xóa bên trong lòng hộp sang trạng thái `REVIEW`.
   - **Stage 2 (`0.5x Coarse-to-Fine Pyramid + Baseline Isolation Guard`):** Dùng bộ template đa tỷ lệ & góc nghiêng nhỏ đã cache sẵn trong RAM một lần khi khởi tạo; kiểm tra khoảng trắng cô lập hai bên dòng cơ sở (`_is_isolated_stamp_region`) để chặn 100% báo nhầm từ `"mật"` trong văn bản thông thường.
2. **Hai Workflow độc lập tuyệt đối (`Part C3` & `C4`):**
   - `SecretScanService` (`app/services/secret_scan_service.py`) dùng chung `PDFLoader` nhưng tách rời hoàn toàn khỏi `DocumentPipeline`.
   - Xử lý offline 100%, giải phóng ảnh trang ngay sau mỗi vòng lặp, giữ ảnh crop phóng to dấu **chỉ trong RAM phiên làm việc**, không ghi log nội dung văn bản và **không tự động tách file PDF**.

