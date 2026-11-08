#!/usr/bin/env python3
"""
Trình tạo bộ dữ liệu kiểm thử chuẩn cho Stripe Defect Detection (`scripts/generate_stripe_samples.py`).
Sinh 12 kịch bản kiểm thử tổng hợp (synthetic benchmark dataset) vào `data/samples/stripe_benchmark/`
kèm nhãn chuẩn xác thực (`stripe_ground_truth.json`), bao phủ:
1. Clean page (Trang văn bản sạch)
2. Dark vertical stripe (Sọc dọc đậm rõ ràng)
3. Thin gray vertical stripe (Sọc dọc xám mảnh 2px độ tương phản vừa/thấp)
4. Horizontal stripe (Sọc ngang kéo dài xuyên lề)
5. Slightly diagonal stripe (Sọc hơi nghiêng ~85°)
6. Thick band stripe (Dải sọc dày ~14px)
7. Dashed / interrupted stripe (Sọc dọc đứt quãng)
8. Uneven background stripe (Sọc dọc trên nền giấy có bóng sáng tối không đều)
9. Table page (Trang có bảng biểu — kiểm soát False Positive)
10. Underline page (Trang có nhiều đường gạch chân văn bản — kiểm soát False Positive)
11. Signature page (Trang có chữ ký tay — kiểm soát False Positive)
12. Rotated 180° + vertical stripe page (Trang vừa bị xoay 180° vừa có sọc dọc — kiểm thử Multi-Defect)
"""

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
import fitz
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pdf_inspector.core.config import SAMPLES_DIR

STRIPE_BENCHMARK_DIR = SAMPLES_DIR / "stripe_benchmark"
GROUND_TRUTH_PATH = STRIPE_BENCHMARK_DIR / "stripe_ground_truth.json"

PAGE_W_PX = 1320
PAGE_H_PX = 1870
RENDER_DPI = 160


def _create_base_text_page(seed: int = 42, bg_value: int = 246) -> np.ndarray:
    """Tạo trang văn bản scan giả lập (1320x1870 ở 160 DPI) với chữ rõ ràng."""
    rng = np.random.default_rng(seed)
    img = np.full((PAGE_H_PX, PAGE_W_PX), bg_value, dtype=np.uint8)
    noise = rng.integers(-2, 3, size=(PAGE_H_PX, PAGE_W_PX), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        "TECHNICAL QUALITY INSPECTION REPORT - SCAN SAMPLE",
        (140, 150),
        font,
        0.95,
        (28,),
        2,
        cv2.LINE_AA,
    )

    paragraphs = [
        "Document scanning pipelines require reliable automatic quality inspection.",
        "Orientation detection identifies pages rotated by 90, 180, or 270 degrees.",
        "Stripe defect detection locates vertical, horizontal, and diagonal streaks.",
        "Dust on the ADF slit glass typically produces a continuous vertical line.",
        "Valid document structures such as tables, underlines, and signatures must",
        "not be misclassified as scanner stripe defects during automated inspection.",
    ]

    y = 240
    for block in range(4):
        cv2.putText(
            img,
            f"Section {block + 1}. Operational Verification",
            (140, y),
            font,
            0.78,
            (32,),
            2,
            cv2.LINE_AA,
        )
        y += 46
        for line in paragraphs:
            cv2.putText(
                img,
                line,
                (140, y),
                font,
                0.62,
                (42,),
                1,
                cv2.LINE_AA,
            )
            y += 38
        y += 36

    cv2.putText(
        img,
        "Page 1 of 1 - Confidential Internal Quality Benchmark",
        (140, PAGE_H_PX - 110),
        font,
        0.58,
        (60,),
        1,
        cv2.LINE_AA,
    )
    return img


def _save_image_as_pdf(img: np.ndarray, pdf_path: Path) -> None:
    """Nhúng mảng ảnh grayscale 160 DPI vào trang PDF chuẩn A4 (595 x 842 pt)."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    ok, png_buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Không thể mã hóa PNG cho {pdf_path}")

    width_pt = PAGE_W_PX * 72.0 / RENDER_DPI
    height_pt = PAGE_H_PX * 72.0 / RENDER_DPI
    doc = fitz.open()
    page = doc.new_page(width=width_pt, height=height_pt)
    page.insert_image(fitz.Rect(0, 0, width_pt, height_pt), stream=png_buf.tobytes())
    doc.save(str(pdf_path))
    doc.close()


def generate_all_stripe_samples() -> Dict[str, Any]:
    """Sinh đầy đủ 12 case kiểm thử Stripe và lưu ground truth JSON."""
    STRIPE_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    cases_gt: List[Dict[str, Any]] = []

    # 1. Clean page
    img_01 = _create_base_text_page(seed=101)
    path_01 = STRIPE_BENCHMARK_DIR / "01_clean_page.pdf"
    _save_image_as_pdf(img_01, path_01)
    cases_gt.append(
        {
            "case_id": "01_clean_page",
            "file_name": path_01.name,
            "description": "Clean document page without any stripe defects",
            "expected_has_stripe": False,
            "expected_stripe_status": "CLEAN",
            "expected_stripe_type": None,
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": None,
        }
    )

    # 2. Dark vertical stripe
    img_02 = _create_base_text_page(seed=102)
    vx0, vx1 = 380, 384  # 4px wide vertical dark stripe from y=10..1860
    img_02[10 : PAGE_H_PX - 10, vx0:vx1] = 28
    path_02 = STRIPE_BENCHMARK_DIR / "02_dark_vertical_stripe.pdf"
    _save_image_as_pdf(img_02, path_02)
    cases_gt.append(
        {
            "case_id": "02_dark_vertical_stripe",
            "file_name": path_02.name,
            "description": "Continuous dark vertical stripe across page margins and text",
            "expected_has_stripe": True,
            "expected_stripe_status": "STRIPE_DETECTED",
            "expected_stripe_type": "VERTICAL",
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": [vx0, 10, vx1, PAGE_H_PX - 10],
        }
    )

    # 3. Thin gray vertical stripe
    img_03 = _create_base_text_page(seed=103)
    gx0, gx1 = 820, 822  # 2px thin gray vertical stripe (intensity ~215 on 246 bg => contrast ~31)
    img_03[15 : PAGE_H_PX - 15, gx0:gx1] = np.minimum(
        img_03[15 : PAGE_H_PX - 15, gx0:gx1], 214
    )
    path_03 = STRIPE_BENCHMARK_DIR / "03_thin_gray_vertical_stripe.pdf"
    _save_image_as_pdf(img_03, path_03)
    cases_gt.append(
        {
            "case_id": "03_thin_gray_vertical_stripe",
            "file_name": path_03.name,
            "description": "Thin 2px faint gray vertical stripe across the page",
            "expected_has_stripe": True,
            "expected_stripe_status": "STRIPE_DETECTED",
            "expected_stripe_type": "VERTICAL",
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": [gx0, 15, gx1, PAGE_H_PX - 15],
        }
    )

    # 4. Horizontal stripe
    img_04 = _create_base_text_page(seed=104)
    hy0, hy1 = 625, 629  # 4px horizontal stripe across full width (x=10..1310)
    img_04[hy0:hy1, 10 : PAGE_W_PX - 10] = 35
    path_04 = STRIPE_BENCHMARK_DIR / "04_horizontal_stripe.pdf"
    _save_image_as_pdf(img_04, path_04)
    cases_gt.append(
        {
            "case_id": "04_horizontal_stripe",
            "file_name": path_04.name,
            "description": "Long continuous horizontal scan line across page and margins",
            "expected_has_stripe": True,
            "expected_stripe_status": "STRIPE_DETECTED",
            "expected_stripe_type": "HORIZONTAL",
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": [10, hy0, PAGE_W_PX - 10, hy1],
        }
    )

    # 5. Slightly diagonal stripe
    img_05 = _create_base_text_page(seed=105)
    dx1, dy1 = 520, 40
    dx2, dy2 = 665, PAGE_H_PX - 40  # ~85.4 degrees (tilted ~4.6 deg from vertical)
    cv2.line(img_05, (dx1, dy1), (dx2, dy2), (32,), thickness=3, lineType=cv2.LINE_AA)
    path_05 = STRIPE_BENCHMARK_DIR / "05_diagonal_stripe.pdf"
    _save_image_as_pdf(img_05, path_05)
    cases_gt.append(
        {
            "case_id": "05_diagonal_stripe",
            "file_name": path_05.name,
            "description": "Slightly diagonal scan streak (~85.4 deg)",
            "expected_has_stripe": True,
            "expected_stripe_status": "STRIPE_DETECTED",
            "expected_stripe_type": "DIAGONAL",
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": [min(dx1, dx2) - 3, dy1, max(dx1, dx2) + 3, dy2],
        }
    )

    # 6. Thick band stripe
    img_06 = _create_base_text_page(seed=106)
    bx0, bx1 = 960, 976  # 16px thick band stripe
    img_06[12 : PAGE_H_PX - 12, bx0:bx1] = np.minimum(
        img_06[12 : PAGE_H_PX - 12, bx0:bx1], 165
    )
    path_06 = STRIPE_BENCHMARK_DIR / "06_thick_band_stripe.pdf"
    _save_image_as_pdf(img_06, path_06)
    cases_gt.append(
        {
            "case_id": "06_thick_band_stripe",
            "file_name": path_06.name,
            "description": "Thick 16px vertical roller/toner band stripe",
            "expected_has_stripe": True,
            "expected_stripe_status": "STRIPE_DETECTED",
            "expected_stripe_type": "BAND",
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": [bx0, 12, bx1, PAGE_H_PX - 12],
        }
    )

    # 7. Dashed / interrupted stripe
    img_07 = _create_base_text_page(seed=107)
    ix0, ix1 = 445, 448
    for y_start in range(20, PAGE_H_PX - 60, 95):
        y_end = min(PAGE_H_PX - 20, y_start + 72)  # 72px line, 23px gap
        img_07[y_start:y_end, ix0:ix1] = 45
    path_07 = STRIPE_BENCHMARK_DIR / "07_dashed_interrupted_stripe.pdf"
    _save_image_as_pdf(img_07, path_07)
    cases_gt.append(
        {
            "case_id": "07_dashed_interrupted_stripe",
            "file_name": path_07.name,
            "description": "Interrupted / dashed vertical scan streak across the page",
            "expected_has_stripe": True,
            "expected_stripe_status": "STRIPE_DETECTED",
            "expected_stripe_type": "VERTICAL",
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": [ix0, 20, ix1, PAGE_H_PX - 20],
        }
    )

    # 8. Uneven background stripe
    img_08 = _create_base_text_page(seed=108)
    # Tạo gradient bóng mờ nền giấy không đều theo cả chiều ngang và dọc (từ 245 xuống 195)
    grad_x = np.linspace(0.0, -35.0, PAGE_W_PX, dtype=np.float32)[None, :]
    grad_y = np.linspace(0.0, -20.0, PAGE_H_PX, dtype=np.float32)[:, None]
    img_08 = np.clip(img_08.astype(np.float32) + grad_x + grad_y, 0, 255).astype(
        np.uint8
    )
    ux0, ux1 = 710, 713
    img_08[15 : PAGE_H_PX - 15, ux0:ux1] = np.clip(
        img_08[15 : PAGE_H_PX - 15, ux0:ux1].astype(np.int16) - 38, 0, 255
    ).astype(np.uint8)
    path_08 = STRIPE_BENCHMARK_DIR / "08_uneven_background_stripe.pdf"
    _save_image_as_pdf(img_08, path_08)
    cases_gt.append(
        {
            "case_id": "08_uneven_background_stripe",
            "file_name": path_08.name,
            "description": "Vertical stripe on uneven illumination / shaded paper background",
            "expected_has_stripe": True,
            "expected_stripe_status": "STRIPE_DETECTED",
            "expected_stripe_type": "VERTICAL",
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": [ux0, 15, ux1, PAGE_H_PX - 15],
        }
    )

    # 9. Table page (False Positive Control)
    img_09 = _create_base_text_page(seed=109)
    # Xóa vùng giữa trang để vẽ bảng dữ liệu nhiều hàng nhiều cột có khung khép kín
    img_09[600:1360, 130:1190] = 246
    tx0, tx1 = 150, 1170
    ty0, ty1 = 640, 1320
    col_xs = [150, 360, 650, 910, 1170]
    row_ys = [640, 720, 800, 880, 960, 1040, 1120, 1200, 1320]
    for cx in col_xs:
        cv2.line(img_09, (cx, ty0), (cx, ty1), (35,), thickness=2)
    for ry in row_ys:
        cv2.line(img_09, (tx0, ry), (tx1, ry), (35,), thickness=2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r_idx in range(len(row_ys) - 1):
        for c_idx in range(len(col_xs) - 1):
            cell_x = col_xs[c_idx] + 18
            cell_y = row_ys[r_idx] + 48
            cv2.putText(
                img_09,
                f"Item {r_idx + 1}.{c_idx + 1}",
                (cell_x, cell_y),
                font,
                0.55,
                (40,),
                1,
                cv2.LINE_AA,
            )
    path_09 = STRIPE_BENCHMARK_DIR / "09_table_page.pdf"
    _save_image_as_pdf(img_09, path_09)
    cases_gt.append(
        {
            "case_id": "09_table_page",
            "file_name": path_09.name,
            "description": "Document page with structured data table (False Positive control)",
            "expected_has_stripe": False,
            "expected_stripe_status": "CLEAN",
            "expected_stripe_type": None,
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": None,
        }
    )

    # 10. Underline page (False Positive Control)
    img_10 = _create_base_text_page(seed=110)
    # Vẽ các đường gạch chân ngay dưới các tiêu đề và dòng chữ trong vùng lề văn bản
    underline_rows = [158, 248, 330, 406, 558, 638, 868, 948]
    for uy in underline_rows:
        cv2.line(img_10, (140, uy), (890, uy), (35,), thickness=2)
    path_10 = STRIPE_BENCHMARK_DIR / "10_underline_page.pdf"
    _save_image_as_pdf(img_10, path_10)
    cases_gt.append(
        {
            "case_id": "10_underline_page",
            "file_name": path_10.name,
            "description": "Document page with underlined headings and sentences (False Positive control)",
            "expected_has_stripe": False,
            "expected_stripe_status": "CLEAN",
            "expected_stripe_type": None,
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": None,
        }
    )

    # 11. Signature page (False Positive Control)
    img_11 = _create_base_text_page(seed=111)
    # Vẽ nét chữ ký uốn lượn và nét gạch ngắn ở góc dưới trang
    sig_pts = np.array(
        [
            [780, 1620],
            [815, 1545],
            [835, 1640],
            [875, 1555],
            [905, 1625],
            [955, 1570],
            [1010, 1610],
            [1080, 1585],
        ],
        dtype=np.int32,
    )
    cv2.polylines(img_11, [sig_pts], isClosed=False, color=(30,), thickness=3, lineType=cv2.LINE_AA)
    cv2.line(img_11, (760, 1645), (1095, 1632), (35,), thickness=2, lineType=cv2.LINE_AA)
    path_11 = STRIPE_BENCHMARK_DIR / "11_signature_page.pdf"
    _save_image_as_pdf(img_11, path_11)
    cases_gt.append(
        {
            "case_id": "11_signature_page",
            "file_name": path_11.name,
            "description": "Document page with handwritten signature strokes (False Positive control)",
            "expected_has_stripe": False,
            "expected_stripe_status": "CLEAN",
            "expected_stripe_type": None,
            "expected_orientation_status": "NORMAL",
            "expected_rotation": 0,
            "gt_bbox": None,
        }
    )

    # 12. Rotated 180° + vertical stripe page (Multi-defect test case)
    img_12_base = _create_base_text_page(seed=112)
    img_12_rot = np.rot90(img_12_base, k=2).copy()  # Xoay 180°
    rx0, rx1 = 310, 315
    img_12_rot[10 : PAGE_H_PX - 10, rx0:rx1] = 26
    path_12 = STRIPE_BENCHMARK_DIR / "12_rotated_180_with_stripe.pdf"
    _save_image_as_pdf(img_12_rot, path_12)
    cases_gt.append(
        {
            "case_id": "12_rotated_180_with_stripe",
            "file_name": path_12.name,
            "description": "Upside-down (180 deg rotated) page that also has a continuous vertical scan stripe",
            "expected_has_stripe": True,
            "expected_stripe_status": "STRIPE_DETECTED",
            "expected_stripe_type": "VERTICAL",
            "expected_orientation_status": "ROTATED",
            "expected_rotation": 180,
            "gt_bbox": [rx0, 10, rx1, PAGE_H_PX - 10],
        }
    )

    gt_payload = {
        "dataset_name": "PDF Scan Stripe Defect Benchmark v1.0",
        "render_dpi": RENDER_DPI,
        "page_dimensions_px": [PAGE_W_PX, PAGE_H_PX],
        "total_cases": len(cases_gt),
        "positive_stripe_cases": sum(1 for c in cases_gt if c["expected_has_stripe"]),
        "negative_clean_cases": sum(
            1 for c in cases_gt if not c["expected_has_stripe"]
        ),
        "cases": cases_gt,
    }
    GROUND_TRUTH_PATH.write_text(
        json.dumps(gt_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return gt_payload


if __name__ == "__main__":
    payload = generate_all_stripe_samples()
    print(
        f"Đã sinh thành công {payload['total_cases']} file PDF kiểm thử Stripe tại '{STRIPE_BENCHMARK_DIR}'."
    )
