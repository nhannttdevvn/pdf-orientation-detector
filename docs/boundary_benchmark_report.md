# Báo Cáo Đánh Giá Định Lượng & Benchmark — Phase 3: Page Boundary & Mixed-Page Detection

## 1. Tổng Quan Kiến Trúc & Phạm Vi Đánh Giá

Module **Page Boundary & Mixed-Page Detection** (`detector_id = "boundary"`, bí danh `"page_boundary"`, `scope = "CROSS_PAGE"`) được tích hợp trực tiếp vào `DocumentPipeline`, `DetectorRegistry`, `PageContext` và `UnifiedQualityEngine` mà không làm thay đổi thuật toán của `OrientationDetector` và `StripeDetector`.

Hệ thống phát hiện và phân loại tự động 5 nhóm lỗi bố cục/ranh giới trang khi scan PDF:
1. **`MIXED_PAGE`**: Một trang PDF chứa phần cuối của trang tài liệu $A$ và phần đầu của trang tài liệu $B$ (hoặc ghép 2 trang vào chung 1 ảnh).
2. **`SPLIT_PAGE`**: Một trang tài liệu vật lý bị chia thành 2 trang liên tiếp $(N, N+1)$ trong file PDF.
3. **`CONTENT_OVERLAP`**: Nội dung ở cuối trang $N$ bị lặp lại ở đầu trang $N+1$ do nạp giấy trượt hoặc đặt lại vị trí quét.
4. **`CONTENT_TRUNCATED`**: Nội dung văn bản/bảng biểu bị cắt cụt sát mép trang (`TOP`, `BOTTOM`, `LEFT`, `RIGHT`). Nếu phần bị thiếu không xuất hiện ở trang liền kề, hệ thống gán trạng thái **`MISSING_CONTENT`** và đề xuất **`REQUIRES_RESCAN`**.
5. **`PAGE_BOUNDARY_SUSPECTED`**: Phát hiện ranh giới nghi vấn ở mức trung bình (`0.38 <= suspicion_score < 0.56`) cần người dùng kiểm tra (`REVIEW`).

---

## 2. Công Thức Tính Điểm Nghi Vấn Đa Đặc Trưng (`Suspicion Score`)

Mỗi ứng viên ranh giới nội trang (`BoundaryCandidate`) được chấm điểm trong khoảng $[0.0, 1.0]$ dựa trên tổ hợp có trọng số của 6 nhóm đặc trưng bố cục & hình học:

$$\text{SuspicionScore} = \operatorname{clip}\left(0.20 \cdot S_{\text{valley}} + 0.25 \cdot S_{\text{seam}} + 0.16 \cdot S_{\text{illum}} + 0.18 \cdot S_{\text{restart}} + 0.11 \cdot S_{\text{geom}} + 0.10 \cdot S_{\text{balance}} - P_{\text{fp}}, \; 0.0, \; 1.0\right)$$

Trong đó:
- $S_{\text{valley}}$: Độ sâu và độ rộng chuẩn hóa của thung lũng khoảng trắng trên biểu đồ chiếu ngang/dọc (`Projection Valley Score`).
- $S_{\text{seam}}$: Cường độ đường ghép vật lý / mép bóng giấy quét qua `Sobel` + `Canny` + `HoughLinesP` (`Physical Seam Score`).
- $S_{\text{illum}}$: Độ chênh lệch độ sáng nền giấy hai phía ranh giới (`Illumination Step Score`, $\Delta I$).
- $S_{\text{restart}}$: Tín hiệu kết thúc trang phía trên (`Footer` / số trang) và bắt đầu trang mới phía dưới (`Header` / tiêu đề chương in hoa).
- $S_{\text{geom}}$: Độ lệch lề trái/phải, bề rộng cột chữ hoặc mật độ dòng giữa hai khối văn bản.
- $S_{\text{balance}}$: Tỷ lệ cân đối diện tích nội dung hai phía đường cắt ứng viên.
- $P_{\text{fp}}$: Điểm phạt từ 9 bộ lọc chống dương tính giả (`paragraph_spacing`, `section_gap`, `table_grid_border`, `content_divider_rule`, `text_underline`, `signature_block`, `large_margin`, `two_column_layout`, `valid_book_spread`).

### Ngưỡng Quyết Định (`Decision Thresholds`)
- **`CONFIRMED` (`MIXED_PAGE_DETECTED` / `RECONSTRUCTABLE`)**: $\text{SuspicionScore} \ge 0.56$ (hoặc $\text{CombinedCrossPageScore} \ge 0.62$).
- **`SUSPECTED` (`BOUNDARY_SUSPECTED` / `REVIEW`)**: $0.38 \le \text{SuspicionScore} < 0.56$.
- **`NORMAL` (`REJECTED`)**: $\text{SuspicionScore} < 0.38$ hoặc bị loại bởi bộ lọc False Positive.

---

## 3. Bộ Dữ Liệu Kiểm Thử Chuẩn 15 Kịch Bản (`data/samples/boundary_benchmark/`)

Bộ dữ liệu chuẩn gồm **15 file PDF (22 trang)** được sinh tự động qua `scripts/generate_boundary_samples.py` kèm nhãn chuẩn xác thực (`verified_ground_truth: true`) tại `data/samples/boundary_benchmark/boundary_ground_truth.json`:

| STT | File mẫu | Số trang | Kịch bản kiểm thử | Trạng thái kỳ vọng | Kết quả thực tế | Sai số vị trí (`norm` / `px`) | Kết luận |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: | :---: |
| 01 | `01_normal_multi_page.pdf` | 3 | Tài liệu chuẩn nhiều trang | `NORMAL` (x3) | `NORMAL` (x3) | — | **PASS** |
| 02 | `02_two_pages_in_one_image.pdf` | 1 | Ghép 2 trang vào 1 ảnh (`y=0.50`) | `MIXED_PAGE_DETECTED` | `MIXED_PAGE_DETECTED` | `0.0032` (`6.0 px`) | **PASS** |
| 03 | `03_page_a_with_top_of_b.pdf` | 2 | Trang 1 chứa phần đầu của Trang 2 (`y=0.70`) | P1: `MIXED_PAGE_DETECTED`, P2: `RECONSTRUCTABLE` (`rel=[1,2]`) | P1: `MIXED_PAGE_DETECTED`, P2: `RECONSTRUCTABLE` (`rel=[1,2]`) | `0.0123` (`23.0 px`) | **PASS** |
| 04 | `04_page_b_continuation_from_prev.pdf` | 2 | 1 trang vật lý bị chia thành 2 trang (`SPLIT_PAGE`) | P1, P2: `RECONSTRUCTABLE` (`SPLIT_PAGE`, `rel=[1,2]`) | P1, P2: `RECONSTRUCTABLE` (`SPLIT_PAGE`, `rel=[1,2]`) | `0.0191` (`35.7 px`) | **PASS** |
| 05 | `05_missing_top_content.pdf` | 1 | Cắt mất mép trên (`TOP`) không khôi phục được | `MISSING_CONTENT` (`CONTENT_TRUNCATED`) | `MISSING_CONTENT` (`CONTENT_TRUNCATED`) | `0.0000` (`0.0 px`) | **PASS** |
| 06 | `06_missing_bottom_content.pdf` | 1 | Cắt mất mép dưới (`BOTTOM`) không khôi phục được | `MISSING_CONTENT` (`CONTENT_TRUNCATED`) | `MISSING_CONTENT` (`CONTENT_TRUNCATED`) | `0.0000` (`0.0 px`) | **PASS** |
| 07 | `07_content_overlap_pages.pdf` | 2 | Trùng lặp 20% nội dung cuối P1 & đầu P2 | P1, P2: `RECONSTRUCTABLE` (`CONTENT_OVERLAP`, `rel=[1,2]`) | P1, P2: `RECONSTRUCTABLE` (`CONTENT_OVERLAP`, `rel=[1,2]`) | `0.0706` (`132.0 px`) | **PASS** |
| 08 | `08_large_whitespace_normal.pdf` | 1 | Khoảng trắng lớn hợp lệ giữa 2 phần (FP Control) | `NORMAL` | `NORMAL` | — | **PASS** |
| 09 | `09_long_table_normal.pdf` | 1 | Bảng biểu nhiều đường kẻ ngang/dọc (FP Control) | `NORMAL` | `NORMAL` | — | **PASS** |
| 10 | `10_signature_and_rule_normal.pdf` | 1 | Đường kẻ phân cách + Khối chữ ký cuối trang (FP Control) | `NORMAL` | `NORMAL` | — | **PASS** |
| 11 | `11_two_column_normal.pdf` | 1 | Trang 2 cột báo chí/học thuật (FP Control) | `NORMAL` | `NORMAL` | — | **PASS** |
| 12 | `12_valid_book_spread.pdf` | 1 | Trang sách mở đôi 2-up cân xứng (FP Control) | `NORMAL` | `NORMAL` | — | **PASS** |
| 13 | `13_rotated_180_and_mixed_page.pdf` | 1 | Đa lỗi: Xoay `180°` + `MIXED_PAGE` | `Ori=ROTATED(180°)` + `MIXED_PAGE_DETECTED` | `Ori=ROTATED(180°)` + `MIXED_PAGE_DETECTED` | `0.0032` (`6.0 px`) | **PASS** |
| 14 | `14_striped_and_mixed_page.pdf` | 1 | Đa lỗi: Có sọc dọc `STRIPE_DETECTED` + `MIXED_PAGE` | `Stripe=STRIPE_DETECTED` + `MIXED_PAGE_DETECTED` | `Stripe=STRIPE_DETECTED` + `MIXED_PAGE_DETECTED` | `0.0032` (`6.0 px`) | **PASS** |
| 15 | `15_multi_page_multi_defect.pdf` | 4 | Đa trang đa lỗi tổng hợp (Clean, Rotated+Stripe, Mixed, Missing) | P1:`NORMAL`, P2:`ROTATED+STRIPE`, P3:`MIXED_PAGE`, P4:`MISSING_CONTENT` | P1:`NORMAL`, P2:`ROTATED+STRIPE`, P3:`MIXED_PAGE`, P4:`MISSING_CONTENT` | `0.0016` (`3.0 px`) | **PASS** |

---

## 4. Các Chỉ Số Đánh Giá Định Lượng (`Quantitative Metrics`)

Kết quả trích xuất trực tiếp từ `data/outputs/boundary_benchmark_metrics.json` (chạy qua `scripts/evaluate_boundary_benchmark.py`):

### 4.1. Độ Chính Xác Phân Loại & Định Vị Ranh Giới (22 trang trong 15 kịch bản chuẩn)
- **Confusion Matrix (Page-level Boundary Defect vs. Normal)**:
  - `True Positives (TP)`: **12 / 12** trang lỗi Boundary
  - `True Negatives (TN)`: **10 / 10** trang bình thường (bao gồm toàn bộ các ca kiểm soát dương tính giả 08–12)
  - `False Positives (FP)`: **0**
  - `False Negatives (FN)`: **0**
- **Precision**: **1.0000 (100.0%)**
- **Recall**: **1.0000 (100.0%)**
- **F1-Score**: **1.0000 (100.0%)**
- **Accuracy**: **1.0000 (100.0%)**
- **False Positive Rate (FPR)**: **0.0000 (0.0%)**
- **Defect Type Classification Accuracy (`MIXED_PAGE`, `SPLIT_PAGE`, `CONTENT_OVERLAP`, `CONTENT_TRUNCATED`)**: **1.0000 (100.0%)**
- **Cross-Page Matching Accuracy (`related_pages` giữa trang $N$ và $N \pm 1$)**: **1.0000 (100.0%)**
- **Mean Boundary Localization Error**: **`0.0170`** (tương đương **`31.87 px`** trên trang cao `1870 px` ở `160 DPI`, đạt độ chính xác định vị trung bình **98.3%**).

### 4.2. Kiểm Thử Đối Chứng Âm Tính Mở Rộng (`Negative Control`)
1. **Trên tập 11 file PDF mẫu của Orientation Detector (`28 trang`):**
   - Số trang `NORMAL` về Boundary: **28 / 28 trang**
   - Tỷ lệ báo nhầm (`False Positive Rate`): **0.0%**
2. **Trên file PDF scan thực tế 38 trang (`data/input/6b63db7ab409_0041.pdf` — sách mở đôi quét xoay ngang 90° kèm sọc ngang máy quét):**
   - Kết quả Orientation: Phát hiện chính xác **38 / 38 trang xoay 90°**.
   - Kết quả Stripe: Phát hiện **6 trang có vệt sọc thật (`STRIPE_DETECTED`)**.
   - Kết quả Boundary: **38 / 38 trang `NORMAL`** (**0 False Positive** — bộ lọc `is_book_spread` và khử sọc máy quét hoạt động chính xác, không báo nhầm rãnh gáy sách mở đôi hay vệt sọc máy quét thành lỗi ghép trang).

### 4.3. Hiệu Năng Xử Lý & Mức Tiêu Thụ Bộ Nhớ
- **Thời gian xử lý trung bình mỗi trang (gồm cả 3 detector Orientation + Stripe + Boundary ở `160 DPI`)**: **`2087.79 ms / trang`** (~`2.09 giây / trang`).
- **Thời gian chạy toàn bộ 38 trang tài liệu thực tế `6b63db7ab409_0041.pdf`**: **`82.56 giây`** (~`2.17 giây / trang`).
- **Mức tiêu thụ RAM đỉnh (`Peak RAM Usage` qua `tracemalloc`)**: **`36.96 MB`** nhờ cơ chế giải phóng ngay `PageContext.release()` sau mỗi trang và `PageLayoutAnalysisResult.release_strips()` ngay sau khi hoàn tất kiểm tra cửa sổ trượt liên trang.

