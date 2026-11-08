#!/usr/bin/env python3
"""
Trình tạo bộ dữ liệu kiểm thử chuyên sâu cho Horizontal Stripe Detection (Phase 4 — Part A11 & A5)
(`scripts/generate_horizontal_stripe_samples.py`).

Sinh 2 nhóm dữ liệu kiểm thử có nhãn xác thực (`horizontal_stripe_ground_truth.json`) vào
`data/samples/horizontal_stripe_benchmark/` (giữ nguyên bộ `stripe_benchmark` cũ theo mục A10):

Nhóm 1 — 15 kịch bản bắt buộc theo mục A11:
  01. Trang văn bản hành chính có đường kẻ dưới Quốc hiệu, Tiêu ngữ -> CLEAN
  02. Trang có bảng biểu nhiều đường ngang (kể cả bảng chỉ có kẻ ngang, không kẻ dọc) -> CLEAN
  03. Trang có nhiều đường gạch chân tiêu đề và câu văn -> CLEAN
  04. Trang có footer line / đường phân cách chân trang -> CLEAN
  05. Trang có chữ ký và đường kẻ ký tên / điền biểu mẫu -> CLEAN
  06. Trang có sọc scan chạy ngang gần hết chiều rộng trang (~94%) -> STRIPE_DETECTED
  07. Trang có sọc ngang xám nhạt mảnh (2px, contrast vừa/thấp) -> STRIPE_DETECTED / REVIEW
  08. Trang có sọc ngang đen dày (14px roller/toner band) -> STRIPE_DETECTED
  09. Trang có sọc ngang chạy vượt cả hai lề trái và phải -> STRIPE_DETECTED
  10. Trang có sọc ngang hơi nghiêng (~4.8°) -> STRIPE_DETECTED
  11. Trang có sọc ngang scan đè xuyên qua bảng biểu ra ngoài lề -> STRIPE_DETECTED
  12. Trang có đường ngang dài 95% chiều rộng nhưng thuộc khung biểu mẫu hợp lệ -> CLEAN
  13. Trang có sọc ngang ngắn hơn ngưỡng (~62%) nhưng rất bất thường (tràn mép giấy & cắt ngang dòng chữ) -> STRIPE_DETECTED / REVIEW
  14. Trang văn bản sạch hoàn toàn -> CLEAN
  15. Trang vừa có sọc dọc vừa có sọc ngang -> STRIPE_DETECTED (phát hiện đủ cả VERTICAL & HORIZONTAL)

Nhóm 2 — Bộ quét ngưỡng tỷ lệ chiều dài (Threshold Sweep 60%, 70%, 80%, 85%, 90%, 95%, 100%) theo mục A5:
  Gồm các trang chứa đường kẻ tài liệu hợp lệ và các trang chứa vệt sọc scan ở nhiều tỷ lệ chiều rộng khác nhau
  để đo thực nghiệm False Positive / False Negative tại từng ngưỡng `HORIZONTAL_STRIPE_MIN_PAGE_RATIO`.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pdf_inspector.core.config import SAMPLES_DIR

H_STRIPE_BENCHMARK_DIR: Path = SAMPLES_DIR / "horizontal_stripe_benchmark"
H_GROUND_TRUTH_PATH: Path = H_STRIPE_BENCHMARK_DIR / "horizontal_stripe_ground_truth.json"

PAGE_W_PX = 1320
PAGE_H_PX = 1870
RENDER_DPI = 160


def _save_image_as_pdf(img: np.ndarray, pdf_path: Path, page_w: int = PAGE_W_PX, page_h: int = PAGE_H_PX) -> None:
    """Nhúng ảnh grayscale 160 DPI vào trang PDF tương ứng."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    ok, png_buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Không thể mã hóa PNG cho {pdf_path}")

    width_pt = page_w * 72.0 / RENDER_DPI
    height_pt = page_h * 72.0 / RENDER_DPI
    doc = fitz.open()
    page = doc.new_page(width=width_pt, height=height_pt)
    page.insert_image(fitz.Rect(0, 0, width_pt, height_pt), stream=png_buf.tobytes())
    doc.save(str(pdf_path))
    doc.close()


def _create_document_canvas(
    seed: int = 200,
    left_margin: int = 155,
    right_margin: int = 1165,
    bg_value: int = 247,
    draw_body: bool = True,
) -> np.ndarray:
    """
    Tạo nền trang tài liệu scan thực tế với lề trái/phải tùy biến (không cố định một lề A4 duy nhất).
    """
    rng = np.random.default_rng(seed)
    img = np.full((PAGE_H_PX, PAGE_W_PX), bg_value, dtype=np.uint8)
    noise = rng.integers(-2, 3, size=(PAGE_H_PX, PAGE_W_PX), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if not draw_body:
        return img

    font = cv2.FONT_HERSHEY_SIMPLEX
    usable_w = right_margin - left_margin
    max_chars = max(32, int(usable_w / 13.2))

    raw_lines = [
        "Bao cao ket qua kiem tra chat luong tai lieu so hoa dinh ky quy III nam 2026.",
        "He thong tu dong phan tich bo cuc trang, uoc luong vung le noi dung thuc te.",
        "Cac duong ke ngang hop le trong bieu mau, bang bieu, chan trang va chu ky",
        "khong duoc bao nham thanh loi vet soc may scan trong qua trinh kiem duyet.",
        "Quy trinh xu ly dam bao giu nguyen do chinh xac cua phat hien soc doc ADF",
        "cung nhu phat hien trang xoay nguoc 90, 180, 270 do va loi lan trang.",
    ]

    y = 330
    for sec in range(3):
        cv2.putText(
            img,
            f"Muc {sec + 1}. Noi dung danh gia ky thuat he thong",
            (left_margin, y),
            font,
            0.74,
            (32,),
            2,
            cv2.LINE_AA,
        )
        y += 44
        for line in raw_lines:
            clipped = line[:max_chars]
            cv2.putText(
                img,
                clipped,
                (left_margin, y),
                font,
                0.60,
                (42,),
                1,
                cv2.LINE_AA,
            )
            y += 38
        y += 34

    return img


def generate_horizontal_stripe_benchmark() -> Dict[str, Any]:
    """Sinh đầy đủ 15 case mục A11 và bộ sweep mục A5."""
    H_STRIPE_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    cases_gt: List[Dict[str, Any]] = []

    def _add_case(
        case_id: str,
        img: np.ndarray,
        desc: str,
        expected_has_stripe: bool,
        expected_statuses: List[str],
        expected_types: Optional[List[str]] = None,
        gt_bbox: Optional[List[int]] = None,
        category: str = "a11_core",
    ) -> None:
        fname = f"{case_id}.pdf"
        _save_image_as_pdf(img, H_STRIPE_BENCHMARK_DIR / fname)
        cases_gt.append(
            {
                "case_id": case_id,
                "file_name": fname,
                "category": category,
                "description": desc,
                "expected_has_stripe": expected_has_stripe,
                "expected_statuses": expected_statuses,
                "expected_types": expected_types or [],
                "gt_bbox": gt_bbox,
            }
        )

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Case 01: Đường kẻ dưới Quốc hiệu, Tiêu ngữ & Số hiệu văn bản hành chính
    img_01 = _create_document_canvas(seed=201, left_margin=150, right_margin=1170)
    cv2.putText(img_01, "BO KHOA HOC VA CONG NGHE", (150, 125), font, 0.58, (30,), 1, cv2.LINE_AA)
    cv2.line(img_01, (215, 142), (415, 142), (32,), 2)
    cv2.putText(img_01, "CONG HOA XA HOI CHU NGHIA VIET NAM", (615, 125), font, 0.60, (28,), 2, cv2.LINE_AA)
    cv2.putText(img_01, "Doc lap - Tu do - Hanh phuc", (700, 162), font, 0.58, (30,), 1, cv2.LINE_AA)
    # Đường kẻ ngang dưới tiêu ngữ (~44% chiều rộng) và đường kẻ phân cách tiêu đề (~77% trong lề)
    cv2.line(img_01, (640, 180), (1120, 180), (28,), 2)
    cv2.putText(img_01, "QUYET DINH BAN HANH QUY TRINH SO HOA", (310, 255), font, 0.78, (25,), 2, cv2.LINE_AA)
    cv2.line(img_01, (150, 276), (1170, 276), (35,), 3)
    _add_case(
        "01_quoc_hieu_header_rule",
        img_01,
        "Văn bản hành chính có đường kẻ dưới Quốc hiệu, Tiêu ngữ và đường phân cách đầu trang",
        False,
        ["CLEAN"],
    )

    # Case 02: Bảng biểu nhiều đường ngang (gồm cả bảng hiện đại CHỈ có đường kẻ ngang, không có đường kẻ dọc!)
    img_02 = _create_document_canvas(seed=202, left_margin=140, right_margin=1180, draw_body=False)
    cv2.putText(img_02, "BANG TONG HOP SO LIEU KIEM TRA HOSO", (140, 160), font, 0.80, (28,), 2, cv2.LINE_AA)
    cv2.putText(img_02, "Don vi thuc hien: Trung tam Luu tru So - Khu vuc I", (140, 210), font, 0.60, (42,), 1, cv2.LINE_AA)
    # Bảng 1 (y=260..860): Bảng hiện đại chỉ có các đường kẻ ngang song song từ x=140..1180 (~78.8% trang), KHÔNG có kẻ dọc
    h_only_rows = [260, 325, 405, 485, 565, 645, 725, 805, 885]
    for idx_r, ry in enumerate(h_only_rows):
        thick = 3 if idx_r in (0, 1, len(h_only_rows) - 1) else 2
        cv2.line(img_02, (140, ry), (1180, ry), (32,), thick)
        if idx_r < len(h_only_rows) - 1:
            cv2.putText(img_02, f"0{idx_r + 1}", (160, ry + 45), font, 0.56, (38,), 1, cv2.LINE_AA)
            cv2.putText(img_02, f"Ho so nghiep vu ma so HS-2026-0{idx_r + 1}", (280, ry + 45), font, 0.56, (38,), 1, cv2.LINE_AA)
            cv2.putText(img_02, "Dat yeu cau", (960, ry + 45), font, 0.56, (38,), 1, cv2.LINE_AA)
    # Đoạn văn ngắn phía dưới bảng
    cv2.putText(img_02, "Ghi chu: Bang tren su dung dinh dang ke ngang khong co vach ngan cot doc.", (140, 960), font, 0.58, (42,), 1, cv2.LINE_AA)
    # Bảng 2 (y=1040..1540): Bảng đầy đủ khung dọc và ngang
    for ry in [1040, 1120, 1220, 1320, 1420, 1520]:
        cv2.line(img_02, (140, ry), (1180, ry), (35,), 2)
    for cx in [140, 380, 820, 1180]:
        cv2.line(img_02, (cx, 1040), (cx, 1520), (35,), 2)
    for r_i, ry in enumerate([1040, 1120, 1220, 1320, 1420]):
        cv2.putText(img_02, f"Phan loai {r_i + 1}", (165, ry + 52), font, 0.55, (40,), 1, cv2.LINE_AA)
        cv2.putText(img_02, "Da doi chieu toan bo trang", (410, ry + 52), font, 0.55, (40,), 1, cv2.LINE_AA)
        cv2.putText(img_02, "100%", (950, ry + 52), font, 0.55, (40,), 1, cv2.LINE_AA)
    _add_case(
        "02_horizontal_tables_page",
        img_02,
        "Trang chứa cả bảng chỉ có đường ngang (không kẻ dọc) và bảng có kẻ ô đầy đủ",
        False,
        ["CLEAN"],
    )

    # Case 03: Trang có đường gạch chân tiêu đề, gạch chân câu văn và đường phân cách mục
    img_03 = _create_document_canvas(seed=203, left_margin=145, right_margin=1175)
    cv2.putText(img_03, "BAO CAO TONG KET CONG TAC LUU TRU", (145, 180), font, 0.85, (28,), 2, cv2.LINE_AA)
    cv2.line(img_03, (145, 198), (980, 198), (30,), 2)
    # Đường gạch chân có khoảng cách hơi xa chữ (14px dưới chân chữ) và đường phân cách giữa 2 mục
    cv2.line(img_03, (145, 342), (760, 342), (34,), 2)
    cv2.line(img_03, (145, 635), (1165, 635), (45,), 2)  # Section divider 77% width trong lề
    cv2.line(img_03, (145, 975), (1165, 975), (45,), 2)  # Section divider thứ hai
    _add_case(
        "03_underlines_and_section_dividers",
        img_03,
        "Trang có đường gạch chân tiêu đề và các đường kẻ ngang phân cách mục nằm trong lề nội dung",
        False,
        ["CLEAN"],
    )

    # Case 04: Trang có footer line / đường phân cách chân trang rộng ~80% trang
    img_04 = _create_document_canvas(seed=204, left_margin=150, right_margin=1190)
    # Đường kẻ chân trang điển hình ở y=1695 (y_norm = 0.906), rộng từ lề trái đến lề phải (78.8% trang)
    cv2.line(img_04, (150, 1695), (1190, 1695), (32,), 3)
    cv2.putText(img_04, "Tai lieu luu hanh noi bo - Trang 04/12", (150, 1742), font, 0.56, (50,), 1, cv2.LINE_AA)
    cv2.putText(img_04, "Ma bieu mau: BM-QA-04", (920, 1742), font, 0.56, (50,), 1, cv2.LINE_AA)
    _add_case(
        "04_footer_separator_line",
        img_04,
        "Trang văn bản có đường kẻ ngang phân cách chân trang (footer rule) nằm gọn trong lề",
        False,
        ["CLEAN"],
    )

    # Case 05: Trang có chữ ký và đường kẻ ký tên / đường điền thông tin biểu mẫu
    img_05 = _create_document_canvas(seed=205, left_margin=150, right_margin=1170)
    # Các dòng điền thông tin biểu mẫu hợp lệ
    cv2.putText(img_05, "Ho va ten nguoi kiem tra:", (150, 1360), font, 0.60, (35,), 1, cv2.LINE_AA)
    cv2.line(img_05, (445, 1365), (1165, 1365), (45,), 2)
    cv2.putText(img_05, "Nhan xet bo sung:", (150, 1425), font, 0.60, (35,), 1, cv2.LINE_AA)
    cv2.line(img_05, (365, 1430), (1165, 1430), (45,), 2)
    cv2.line(img_05, (150, 1490), (1165, 1490), (45,), 2)
    # Khung chữ ký bên trái và bên phải kèm đường kẻ ngang ký tên
    cv2.putText(img_05, "NGUOI LAP BIEU", (230, 1570), font, 0.62, (30,), 2, cv2.LINE_AA)
    cv2.putText(img_05, "THU TRUONG DON VI", (805, 1570), font, 0.62, (30,), 2, cv2.LINE_AA)
    sig_pts = np.array([[820, 1680], [855, 1605], [880, 1695], [930, 1615], [985, 1675], [1055, 1640]], dtype=np.int32)
    cv2.polylines(img_05, [sig_pts], False, (28,), 3, cv2.LINE_AA)
    cv2.line(img_05, (180, 1720), (520, 1720), (35,), 2)
    cv2.line(img_05, (760, 1720), (1130, 1720), (35,), 2)
    _add_case(
        "05_signature_and_form_lines",
        img_05,
        "Trang có các đường kẻ điền biểu mẫu, chữ ký tay và đường kẻ ngang ký tên",
        False,
        ["CLEAN"],
    )

    # Case 06: Trang có sọc scan chạy ngang gần hết chiều rộng trang (~94.5%, tràn lề)
    img_06 = _create_document_canvas(seed=206, left_margin=160, right_margin=1160)
    hy0, hy1 = 742, 746
    sx0, sx1 = 28, PAGE_W_PX - 44  # 1248 / 1320 = 94.5%, vượt xa cả lề trái (160) và lề phải (1160)
    img_06[hy0:hy1, sx0:sx1] = 32
    _add_case(
        "06_near_full_width_horizontal_stripe",
        img_06,
        "Trang có sọc scan chạy ngang gần hết chiều rộng trang (~94.5%) và vượt ra ngoài hai lề",
        True,
        ["STRIPE_DETECTED"],
        ["HORIZONTAL"],
        [sx0, hy0, sx1, hy1],
    )

    # Case 07: Trang có sọc ngang xám nhạt mảnh (2px, contrast thấp/vừa, xuyên suốt chiều rộng)
    img_07 = _create_document_canvas(seed=207, left_margin=155, right_margin=1165)
    gy0, gy1 = 918, 920
    img_07[gy0:gy1, 12 : PAGE_W_PX - 12] = np.minimum(
        img_07[gy0:gy1, 12 : PAGE_W_PX - 12], 212
    )
    _add_case(
        "07_faint_gray_horizontal_stripe",
        img_07,
        "Trang có sọc ngang xám nhạt mảnh 2px chạy xuyên hai lề",
        True,
        ["STRIPE_DETECTED", "REVIEW"],
        ["HORIZONTAL"],
        [12, gy0, PAGE_W_PX - 12, gy1],
    )

    # Case 08: Trang có sọc ngang đen dày (14px thick band)
    img_08 = _create_document_canvas(seed=208, left_margin=155, right_margin=1165)
    by0, by1 = 1085, 1099  # 14px thick dark horizontal band across full width
    img_08[by0:by1, 4 : PAGE_W_PX - 4] = np.minimum(
        img_08[by0:by1, 4 : PAGE_W_PX - 4], 42
    )
    _add_case(
        "08_thick_dark_horizontal_stripe",
        img_08,
        "Trang có dải sọc ngang đen dày 14px chạy hết chiều rộng trang",
        True,
        ["STRIPE_DETECTED"],
        ["BAND", "HORIZONTAL"],
        [4, by0, PAGE_W_PX - 4, by1],
    )

    # Case 09: Trang có sọc ngang mảnh chạy vượt rõ rệt cả hai lề trái và phải (~92%)
    img_09 = _create_document_canvas(seed=209, left_margin=185, right_margin=1135)
    ey0, ey1 = 530, 533
    ex0, ex1 = 52, 1268  # 1216 / 1320 = 92.1%, vượt lề trái 133px (10.1%) và lề phải 133px (10.1%)
    img_09[ey0:ey1, ex0:ex1] = 36
    _add_case(
        "09_extends_both_margins_stripe",
        img_09,
        "Trang có sọc ngang vượt sâu ra ngoài cả lề trái và lề phải của vùng văn bản",
        True,
        ["STRIPE_DETECTED"],
        ["HORIZONTAL"],
        [ex0, ey0, ex1, ey1],
    )

    # Case 10: Trang có sọc ngang hơi nghiêng (~4.6° so với phương ngang)
    img_10 = _create_document_canvas(seed=210, left_margin=155, right_margin=1165)
    dx1, dy1 = 18, 610
    dx2, dy2 = PAGE_W_PX - 18, 715  # Góc nghiêng ~4.64° chạy ngang qua trang
    cv2.line(img_10, (dx1, dy1), (dx2, dy2), (30,), thickness=3, lineType=cv2.LINE_AA)
    _add_case(
        "10_slightly_tilted_horizontal_stripe",
        img_10,
        "Trang có sọc scan chạy ngang hơi nghiêng (~4.6 độ) xuyên qua lề và văn bản",
        True,
        ["STRIPE_DETECTED"],
        ["DIAGONAL", "HORIZONTAL"],
        [dx1, dy1 - 3, dx2, dy2 + 3],
    )

    # Case 11: Trang có sọc ngang scan đè xuyên qua bảng biểu ra ngoài hai lề
    img_11 = _create_document_canvas(seed=211, left_margin=150, right_margin=1170, draw_body=False)
    cv2.putText(img_11, "BANG DOI CHIEU CHUNG TU KE TOAN", (150, 175), font, 0.80, (28,), 2, cv2.LINE_AA)
    col_xs = [150, 350, 720, 960, 1170]
    row_ys = [250, 340, 430, 520, 610, 700, 790, 880, 970, 1060]
    for cx in col_xs:
        cv2.line(img_11, (cx, 250), (cx, 1060), (35,), 2)
    for ry in row_ys:
        cv2.line(img_11, (150, ry), (1170, ry), (35,), 2)
    for r_i in range(len(row_ys) - 1):
        cv2.putText(img_11, f"CT-0{r_i + 1}", (175, row_ys[r_i] + 55), font, 0.55, (40,), 1, cv2.LINE_AA)
        cv2.putText(img_11, "Noi dung hach toan chi tiet", (375, row_ys[r_i] + 55), font, 0.55, (40,), 1, cv2.LINE_AA)
        cv2.putText(img_11, "12.500.000", (985, row_ys[r_i] + 55), font, 0.55, (40,), 1, cv2.LINE_AA)
    cv2.putText(img_11, "Xac nhan cua ke toan truong don vi kiem toan.", (150, 1160), font, 0.60, (40,), 1, cv2.LINE_AA)
    # Sọc scan thật chạy ngang tại y=655..659 xuyên qua giữa bảng từ sát mép trái (x=8) tới sát mép phải (x=1312)
    tsy0, tsy1 = 655, 659
    img_11[tsy0:tsy1, 8 : PAGE_W_PX - 8] = 26
    _add_case(
        "11_stripe_crossing_through_table",
        img_11,
        "Trang có bảng biểu hợp lệ nhưng bị một sọc scan ngang chạy xuyên qua bảng ra sát hai mép giấy",
        True,
        ["STRIPE_DETECTED"],
        ["HORIZONTAL"],
        [8, tsy0, PAGE_W_PX - 8, tsy1],
    )

    # Case 12: Trang có đường ngang dài 95% chiều rộng nhưng thuộc khung biểu mẫu/bảng rộng hợp lệ (có góc vuông khép kín)
    img_12 = _create_document_canvas(seed=212, left_margin=33, right_margin=1287, draw_body=False)
    fx0, fx1 = 33, 1287  # 1254 / 1320 = 95.0% chiều rộng trang!
    # Khung tiêu đề biểu mẫu rộng 95% có cạnh dọc đóng khung khép kín (L/T corners rõ ràng ở 2 đầu x=33 và x=1287)
    cv2.rectangle(img_12, (fx0, 90), (fx1, 250), (32,), 2)
    cv2.line(img_12, (fx0, 165), (fx1, 165), (32,), 2)
    cv2.line(img_12, (360, 90), (360, 250), (32,), 2)
    cv2.line(img_12, (980, 90), (980, 250), (32,), 2)
    cv2.putText(img_12, "DON VI KIEM DINH", (60, 138), font, 0.58, (32,), 1, cv2.LINE_AA)
    cv2.putText(img_12, "BIEU MAU KIEM SOAT CHAT LUONG BAN QUET", (390, 138), font, 0.65, (28,), 2, cv2.LINE_AA)
    cv2.putText(img_12, "BM-09/2026", (1020, 138), font, 0.58, (32,), 1, cv2.LINE_AA)
    cv2.putText(img_12, "Ma so: QA-95", (60, 215), font, 0.55, (40,), 1, cv2.LINE_AA)
    cv2.putText(img_12, "Ap dung cho tai lieu kho A4 in sat le bieu mau ky thuat", (390, 215), font, 0.55, (40,), 1, cv2.LINE_AA)
    cv2.putText(img_12, "Lan ban hanh: 02", (1020, 215), font, 0.55, (40,), 1, cv2.LINE_AA)
    # Bảng nội dung rộng 95% bên dưới
    cv2.rectangle(img_12, (fx0, 310), (fx1, 1680), (34,), 2)
    for ry in [390, 520, 650, 780, 910, 1040]:
        cv2.line(img_12, (fx0, ry), (fx1, ry), (34,), 2)
    for cx in [180, 760, 1040]:
        cv2.line(img_12, (cx, 310), (cx, 1680), (34,), 2)
    for idx_r, ry in enumerate([310, 390, 520, 650, 780, 910]):
        cv2.putText(img_12, f"HM-0{idx_r + 1}", (55, ry + 52), font, 0.55, (38,), 1, cv2.LINE_AA)
        cv2.putText(img_12, "Hang muc kiem tra thong so ky thuat bieu mau rong", (205, ry + 52), font, 0.55, (38,), 1, cv2.LINE_AA)
        cv2.putText(img_12, "Binh thuong", (790, ry + 52), font, 0.55, (38,), 1, cv2.LINE_AA)
        cv2.putText(img_12, "Dat", (1080, ry + 52), font, 0.55, (38,), 1, cv2.LINE_AA)
    _add_case(
        "12_valid_95pct_wide_form_frame",
        img_12,
        "Trang có các đường ngang dài 95% chiều rộng nhưng thuộc khung biểu mẫu/bảng hợp lệ có cạnh đóng góc",
        False,
        ["CLEAN"],
    )

    # Case 13: Trang có sọc ngang ngắn hơn ngưỡng 90% (~64% chiều rộng) nhưng RẤT BẤT THƯỜNG
    # (dải mực bẩn bắt đầu sát mép giấy x=0 tràn xuyên lề trái và cắt đè trực tiếp qua giữa thân dòng chữ!)
    img_13 = _create_document_canvas(seed=213, left_margin=160, right_margin=1160)
    # Dòng chữ tại y=418 (thân chữ từ ~398..418): đặt vệt mực bẩn dày 9px tại y=406..415 từ x=0..845 (~64% trang)
    ab_y0, ab_y1 = 406, 415
    ab_x0, ab_x1 = 0, 845
    img_13[ab_y0:ab_y1, ab_x0:ab_x1] = np.minimum(img_13[ab_y0:ab_y1, ab_x0:ab_x1], 48)
    _add_case(
        "13_short_abnormal_edge_bleed_stripe",
        img_13,
        "Trang có vệt sọc ngang ngắn (~64% trang) nhưng bắt đầu từ mép giấy x=0, tràn lề và đè cắt ngang thân dòng chữ",
        True,
        ["STRIPE_DETECTED", "REVIEW"],
        ["BAND", "HORIZONTAL"],
        [ab_x0, ab_y0, ab_x1, ab_y1],
    )

    # Case 14: Trang sạch hoàn toàn
    img_14 = _create_document_canvas(seed=214, left_margin=155, right_margin=1165)
    _add_case(
        "14_clean_document_page",
        img_14,
        "Trang văn bản sạch hoàn toàn không có đường kẻ ngang hay vệt sọc",
        False,
        ["CLEAN"],
    )

    # Case 15: Trang vừa có sọc dọc vừa có sọc ngang
    img_15 = _create_document_canvas(seed=215, left_margin=160, right_margin=1160)
    # Sọc dọc tại x=310..314
    img_15[12 : PAGE_H_PX - 12, 310:314] = 30
    # Sọc ngang tại y=835..839 chạy xuyên hai lề
    img_15[835:839, 10 : PAGE_W_PX - 10] = 32
    _add_case(
        "15_both_vertical_and_horizontal_stripes",
        img_15,
        "Trang vừa có sọc dọc (VERTICAL) vừa có sọc ngang (HORIZONTAL) — phải phát hiện đủ cả hai",
        True,
        ["STRIPE_DETECTED"],
        ["VERTICAL", "HORIZONTAL"],
        [10, 835, PAGE_W_PX - 10, 839],
    )

    # =========================================================================
    # NHÓM 2: BỘ THỬ NGHIỆM QUÉT NGƯỠNG TỶ LỆ (A5 — 60%, 70%, 80%, 85%, 90%, 95%, 100%)
    # =========================================================================
    ratios = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    for r in ratios:
        pct = int(round(r * 100))
        line_w = int(round(PAGE_W_PX * r))
        # (A) Negative control cho các tỷ lệ <= 85%: đường kẻ phân cách hợp lệ căn giữa nằm gọn trong lề nội dung rộng
        if r <= 0.85:
            margin_l = max(70, (PAGE_W_PX - max(line_w, 980)) // 2 - 15)
            margin_r = PAGE_W_PX - margin_l
            img_legit = _create_document_canvas(seed=300 + pct, left_margin=margin_l, right_margin=margin_r)
            lx0 = (PAGE_W_PX - line_w) // 2
            lx1 = lx0 + line_w
            cv2.line(img_legit, (lx0, 635), (lx1, 635), (38,), 2)
            _add_case(
                f"sweep_legit_rule_{pct}pct",
                img_legit,
                f"Đường phân cách nội dung hợp lệ dài {pct}% chiều rộng trang nằm trong vùng lề văn bản",
                False,
                ["CLEAN"],
                category="a5_threshold_sweep",
            )
        elif r == 0.90:
            # Ở 90%: đường khung bảng biểu hợp lệ có cạnh dọc đóng khung ở 2 đầu
            img_legit90 = _create_document_canvas(seed=390, left_margin=66, right_margin=1254)
            lx0, lx1 = 66, 66 + line_w  # 90%
            cv2.rectangle(img_legit90, (lx0, 120), (lx1, 250), (35,), 2)
            cv2.line(img_legit90, (lx0, 185), (lx1, 185), (35,), 2)
            cv2.putText(img_legit90, "KHUNG TIEU DE BIEU MAU RONG 90% CO GOC DONG", (95, 165), font, 0.62, (32,), 2, cv2.LINE_AA)
            _add_case(
                "sweep_legit_boxed_90pct",
                img_legit90,
                "Đường kẻ khung tiêu đề dài 90% chiều rộng có cạnh dọc khép kín hợp lệ",
                False,
                ["CLEAN"],
                category="a5_threshold_sweep",
            )

        # (B) Positive scan defect cho các tỷ lệ >= 90% (sọc scan tràn lề không thuộc bảng/footer)
        if r >= 0.90:
            img_defect = _create_document_canvas(seed=400 + pct, left_margin=160, right_margin=1160)
            sx0 = max(0, (PAGE_W_PX - line_w) // 2)
            sx1 = min(PAGE_W_PX, sx0 + line_w)
            img_defect[790:794, sx0:sx1] = 30
            _add_case(
                f"sweep_scan_stripe_{pct}pct",
                img_defect,
                f"Sọc scan ngang dài {pct}% chiều rộng trang vượt qua cả hai lề nội dung",
                True,
                ["STRIPE_DETECTED"],
                ["HORIZONTAL"],
                [sx0, 790, sx1, 794],
                category="a5_threshold_sweep",
            )

    payload: Dict[str, Any] = {
        "dataset_name": "phase4_horizontal_stripe_benchmark",
        "dpi": RENDER_DPI,
        "page_size_px": [PAGE_W_PX, PAGE_H_PX],
        "verified_ground_truth": True,
        "total_cases": len(cases_gt),
        "cases": cases_gt,
    }
    H_GROUND_TRUTH_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    gt = generate_horizontal_stripe_benchmark()
    print(f"Generated {gt['total_cases']} horizontal benchmark PDFs at {H_STRIPE_BENCHMARK_DIR}")

