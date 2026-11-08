#!/usr/bin/env python3
"""
Trình tạo bộ dữ liệu kiểm thử chuẩn cho Page Boundary & Mixed-Page Detection (`scripts/generate_boundary_samples.py`).
Sinh đầy đủ 15 kịch bản kiểm thử tổng hợp (synthetic benchmark dataset) theo đặc tả Phase 3.10
vào thư mục `data/samples/boundary_benchmark/` kèm nhãn chuẩn xác thực (`boundary_ground_truth.json`):
1. `01_normal_multi_page.pdf`: PDF nhiều trang bình thường
2. `02_two_pages_in_one_image.pdf`: 2 trang nằm chung trong 1 ảnh trang (`MIXED_PAGE`)
3. `03_page_a_with_top_of_b.pdf`: Trang A bị dính phần đầu của trang B (`MIXED_PAGE` + liên trang)
4. `04_page_b_continuation_from_prev.pdf`: Trang B chứa phần tiếp nối từ trang trước (`SPLIT_PAGE`)
5. `05_missing_top_content.pdf`: Trang bị cắt mất dòng đầu, không có trên trang trước (`CONTENT_TRUNCATED` / `MISSING_CONTENT`)
6. `06_missing_bottom_content.pdf`: Trang bị cắt mất dòng cuối, không có trên trang sau (`CONTENT_TRUNCATED` / `MISSING_CONTENT`)
7. `07_content_overlap_pages.pdf`: Hai trang bị trùng lặp nội dung (`CONTENT_OVERLAP`)
8. `08_large_whitespace_normal.pdf`: Trang có khoảng trắng lớn nhưng không bị lỗi (`NORMAL`)
9. `09_long_table_normal.pdf`: Trang có bảng biểu dài (`NORMAL`)
10. `10_signature_and_rule_normal.pdf`: Trang có chữ ký cuối trang và đường kẻ ngang (`NORMAL`)
11. `11_two_column_normal.pdf`: Trang có 2 cột văn bản (`NORMAL`)
12. `12_valid_book_spread.pdf`: Trang sách mở đôi hợp lệ (`NORMAL`)
13. `13_rotated_180_and_mixed_page.pdf`: Trang vừa xoay 180° vừa bị `MIXED_PAGE`
14. `14_striped_and_mixed_page.pdf`: Trang vừa có `STRIPE` vừa bị `MIXED_PAGE`
15. `15_multi_page_multi_defect.pdf`: PDF nhiều trang có kết hợp nhiều lỗi
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

import cv2
import fitz
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ai_engine.core.config import SAMPLES_DIR

BOUNDARY_BENCHMARK_DIR = SAMPLES_DIR / "boundary_benchmark"
GROUND_TRUTH_PATH = BOUNDARY_BENCHMARK_DIR / "boundary_ground_truth.json"

PAGE_W_PX = 1320
PAGE_H_PX = 1870
RENDER_DPI = 160


def _save_images_as_pdf(images: Sequence[np.ndarray], pdf_path: Path) -> None:
    """Nhúng danh sách ảnh grayscale 160 DPI vào các trang của file PDF."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    try:
        for img in images:
            h, w = img.shape[:2]
            ok, png_buf = cv2.imencode(".png", img)
            if not ok:
                raise RuntimeError(f"Không thể mã hóa PNG cho {pdf_path}")
            width_pt = w * 72.0 / RENDER_DPI
            height_pt = h * 72.0 / RENDER_DPI
            page = doc.new_page(width=width_pt, height=height_pt)
            page.insert_image(
                fitz.Rect(0, 0, width_pt, height_pt),
                stream=png_buf.tobytes(),
            )
        doc.save(str(pdf_path))
    finally:
        doc.close()


def _blank_canvas(
    seed: int = 42,
    bg_value: int = 246,
    width: int = PAGE_W_PX,
    height: int = PAGE_H_PX,
) -> np.ndarray:
    """Tạo nền trang giấy scan có nhiễu hạt nhẹ tự nhiên."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width), bg_value, dtype=np.uint8)
    noise = rng.integers(-2, 3, size=(height, width), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _draw_paragraph_block(
    img: np.ndarray,
    lines: Sequence[str],
    x_start: int,
    y_start: int,
    line_step: int = 36,
    font_scale: float = 0.62,
    thickness: int = 1,
    color: int = 38,
) -> int:
    """Vẽ một khối văn bản nhiều dòng và trả về tọa độ y kế tiếp."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = y_start
    for line in lines:
        cv2.putText(
            img,
            line,
            (x_start, y),
            font,
            font_scale,
            (color,),
            thickness,
            cv2.LINE_AA,
        )
        y += line_step
    return y


def _create_normal_page(page_idx: int = 1, total_pages: int = 3, seed: int = 100) -> np.ndarray:
    """Tạo trang văn bản scan bình thường, cân đối."""
    img = _blank_canvas(seed=seed + page_idx, bg_value=246)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        f"STANDARD TECHNICAL SPECIFICATION - DOCUMENT PAGE {page_idx}",
        (140, 145),
        font,
        0.88,
        (26,),
        2,
        cv2.LINE_AA,
    )
    paragraphs = [
        "Automated document inspection ensures high-fidelity archival of physical scans.",
        "Page boundary analysis verifies that each digital page maps to one physical sheet.",
        "Cross-page correlation compares adjacent strips using projection profiles and ORB.",
        "False positive suppression protects multi-section layouts, tables, and signatures.",
        "Original PDF files are never overwritten during inspection or preview workflows.",
        "User confirmation is strictly required before exporting any reconstructed page.",
    ]
    y = 230
    for sec in range(4):
        cv2.putText(
            img,
            f"Section {page_idx}.{sec + 1} - Operational Verification Protocol",
            (140, y),
            font,
            0.74,
            (32,),
            2,
            cv2.LINE_AA,
        )
        y += 44
        y = _draw_paragraph_block(img, paragraphs, x_start=140, y_start=y, line_step=36)
        y += 32

    cv2.putText(
        img,
        f"Page {page_idx} of {total_pages} - Quality Assurance Archive",
        (140, PAGE_H_PX - 110),
        font,
        0.58,
        (55,),
        1,
        cv2.LINE_AA,
    )
    return img


def _create_mixed_two_pages_in_one(seed: int = 202) -> np.ndarray:
    """
    Case 02: Hai trang vật lý nằm chung trong 1 ảnh trang (`MIXED_PAGE`) với ranh giới tại y=935 (`y_norm=0.500`).
    Nửa trên và nửa dưới có nền sáng tối chênh lệch (`246` vs `228`), bóng mép giấy chuyển tiếp tự nhiên tại y=928..942,
    và độ rộng lề khác nhau để mô phỏng 2 tờ giấy bị kéo chồng trong khay ADF.
    """
    img = _blank_canvas(seed=seed, bg_value=246)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Nửa dưới là tờ giấy thứ 2 có độ sáng nền 228 (illumination step ~ 18 gray levels, chuyển tiếp đơn điệu không tạo vệt sọc tối cục bộ)
    rng = np.random.default_rng(seed + 7)
    bottom_bg = np.clip(
        228 + rng.integers(-2, 3, size=(PAGE_H_PX - 935, PAGE_W_PX)), 0, 255
    ).astype(np.uint8)
    img[935:, :] = bottom_bg

    # Nội dung Trang vật lý 1 ở nửa trên (y=115..855, lề 130..1180)
    cv2.putText(
        img,
        "PHYSICAL SHEET A - EXECUTIVE SUMMARY REPORT",
        (130, 135),
        font,
        0.86,
        (25,),
        2,
        cv2.LINE_AA,
    )
    lines_top = [
        "This first physical document sheet ends above the horizontal scanner splice seam.",
        "Multi-feature boundary analysis combines whitespace ratio, paper illumination",
        "step across the gap, and layout discontinuity between upper and lower sheets.",
        "Each line is rendered clearly at 160 DPI to support both OCR and CV inspection.",
    ]
    y = 195
    for blk in range(3):
        cv2.putText(
            img,
            f"Part A.{blk + 1} - Financial & Compliance Overview",
            (130, y),
            font,
            0.72,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y += 38
        y = _draw_paragraph_block(
            img, lines_top, x_start=130, y_start=y, line_step=34, font_scale=0.62
        )
        y += 20
    cv2.putText(
        img,
        "Page 1 of 2",
        (600, 865),
        font,
        0.58,
        (45,),
        1,
        cv2.LINE_AA,
    )

    # Nội dung Trang vật lý 2 ở nửa dưới (y=1005..1780, lề hẹp hơn 220..1080 để thể hiện layout shift)
    cv2.putText(
        img,
        "CHAPTER 2 - SYSTEM ARCHITECTURE (PHYSICAL SHEET B)",
        (220, 1035),
        font,
        0.82,
        (25,),
        2,
        cv2.LINE_AA,
    )
    lines_bottom = [
        "Second physical sheet pulled into the same scan frame due to ADF double-feed.",
        "Notice the shifted horizontal margins and darker paper background below y=935.",
        "Safe reconstruction can split this composite frame at y_norm=0.500 cleanly.",
        "Automated quality verification preserves all original document text intact.",
    ]
    y = 1095
    for blk in range(3):
        cv2.putText(
            img,
            f"Section B.{blk + 1} - Cross-Page Boundary Verification",
            (220, y),
            font,
            0.70,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y += 38
        y = _draw_paragraph_block(
            img, lines_bottom, x_start=220, y_start=y, line_step=34, font_scale=0.62
        )
        y += 20
    cv2.putText(
        img,
        "Page 2 of 2 - Sheet B Footer",
        (220, 1790),
        font,
        0.58,
        (48,),
        1,
        cv2.LINE_AA,
    )
    return img


def _create_case03_page_a_with_top_of_b() -> List[np.ndarray]:
    """
    Case 03: Trang 1 chứa toàn bộ Trang A (y=115..1230) và bị dính phần đầu của Trang B ở cuối trang
    sau ranh giới tại `y=1310` (`y_norm=0.700`). Trang 2 bắt đầu bằng đúng phần đầu của Trang B.
    """
    p1 = _blank_canvas(seed=301, bg_value=246)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Đổi nền từ y=1310 trở xuống (illumination step ~ 18 gray levels)
    p1[1310:, :] = np.clip(p1[1310:, :].astype(np.int16) - 18, 0, 255).astype(np.uint8)
    for dy, val in enumerate([242, 238, 233, 229, 226, 228]):
        p1[1307 + dy, :] = val

    cv2.putText(
        p1,
        "DOCUMENT PAGE A - PRIMARY TECHNICAL RECORD",
        (140, 140),
        font,
        0.86,
        (26,),
        2,
        cv2.LINE_AA,
    )
    body_a = [
        "Primary page content occupies the top seventy percent of this scanned frame.",
        "When the scanner timing slips, the top header of the next sheet is captured",
        "at the bottom of the current image frame after a visible horizontal paper step.",
    ]
    y = 210
    for s in range(5):
        cv2.putText(
            p1,
            f"Clause A.{s + 1} - Inspection Specification",
            (140, y),
            font,
            0.70,
            (32,),
            2,
            cv2.LINE_AA,
        )
        y += 38
        y = _draw_paragraph_block(p1, body_a, x_start=140, y_start=y, line_step=34)
        y += 22
    cv2.putText(
        p1,
        "Page 1 of 2",
        (600, 1235),
        font,
        0.58,
        (48,),
        1,
        cv2.LINE_AA,
    )

    # Phần đầu của Trang B được vẽ vào một mảng riêng để dùng chung cho cuối Trang 1 và đầu Trang 2
    b_top_strip = _blank_canvas(seed=302, bg_value=244, width=PAGE_W_PX, height=390)
    cv2.putText(
        b_top_strip,
        "SECTION B - CROSS-PAGE CONTINUATION HEADER",
        (220, 65),
        font,
        0.84,
        (24,),
        2,
        cv2.LINE_AA,
    )
    b_intro_lines = [
        "This opening block belongs to Page B but was captured at the bottom of Page A.",
        "Cross-page template matching and projection profile correlation detect that",
        "the bottom strip of Page 1 matches the top strip of Page 2 with high similarity.",
        "Identifier Code: BOUNDARY-SPILLOVER-2026-REF-9941.",
        "Verification Metric: Normalized Cross-Correlation and ORB Inlier Consistency.",
    ]
    _draw_paragraph_block(
        b_top_strip, b_intro_lines, x_start=220, y_start=115, line_step=40, font_scale=0.62
    )

    # Dán phần đầu Trang B vào cuối Trang 1 (y=1380..1770)
    p1[1380:1770, :] = np.clip(
        b_top_strip.astype(np.int16) - 16, 0, 255
    ).astype(np.uint8)

    # Tạo Trang 2 bắt đầu bằng đúng `b_top_strip` ở đầu trang (y=110..500), sau đó là phần còn lại của Trang B
    p2 = _blank_canvas(seed=303, bg_value=244)
    p2[110:500, :] = b_top_strip
    y2 = 545
    for s in range(4):
        cv2.putText(
            p2,
            f"Section B.{s + 2} - Remaining Body of Page B",
            (220, y2),
            font,
            0.72,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y2 += 40
        y2 = _draw_paragraph_block(
            p2,
            [
                "Subsequent paragraphs of Page B continue normally down the second page.",
                "Deduplicating or splitting the spillover restores clean page boundaries.",
                "All original content is preserved without fabricating any missing text.",
            ],
            x_start=220,
            y_start=y2,
            line_step=36,
            font_scale=0.62,
        )
        y2 += 30
    return [p1, p2]


def _create_case04_split_page_across_two() -> List[np.ndarray]:
    """
    Case 04: Một trang tài liệu vật lý bị chia thành 2 trang PDF liên tiếp (`SPLIT_PAGE`):
    - Trang 1 chỉ chứa nửa đầu (y=130..860), phần dưới trống lớn (`bottom_empty_ratio ~ 0.54`).
    - Trang 2 chứa phần tiếp nối nửa sau (y=130..850), cùng độ rộng lề `140..1180`, phần dưới trống lớn.
    """
    p1 = _blank_canvas(seed=401, bg_value=246)
    p2 = _blank_canvas(seed=402, bg_value=246)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        p1,
        "SINGLE PHYSICAL CONTRACT - PART 1 (UPPER HALF)",
        (140, 155),
        font,
        0.86,
        (26,),
        2,
        cv2.LINE_AA,
    )
    lines_1 = [
        "This physical page was prematurely ejected halfway through the scanning pass.",
        "As a result, only the upper 42 percent of the document appears on Page 1,",
        "leaving more than half of the bottom canvas completely blank whitespace.",
        "The remaining paragraphs were captured onto the immediately following Page 2.",
    ]
    y = 225
    for blk in range(3):
        cv2.putText(
            p1,
            f"Article {blk + 1} - Upper Segment Provisions",
            (140, y),
            font,
            0.72,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y += 40
        y = _draw_paragraph_block(p1, lines_1, x_start=140, y_start=y, line_step=36)
        y += 26

    # Trang 2 tiếp nối ngay nội dung với cùng lề 140..1180 và kết thúc ở y~850
    y2 = 155
    for blk in range(3):
        cv2.putText(
            p2,
            f"Article {blk + 4} - Continuation from Previous Scan Frame",
            (140, y2),
            font,
            0.72,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y2 += 40
        y2 = _draw_paragraph_block(
            p2,
            [
                "Continuation of the split physical sheet with identical left and right margins.",
                "Complementary content heights (0.39 + 0.38 = 0.77 <= 1.18) confirm SPLIT_PAGE.",
                "Users can preview and merge Page 1 and Page 2 into a single complete page.",
            ],
            x_start=140,
            y_start=y2,
            line_step=36,
        )
        y2 += 28
    cv2.putText(
        p2,
        "Page 1 of 1 - End of Physical Contract Sheet",
        (140, 845),
        font,
        0.58,
        (52,),
        1,
        cv2.LINE_AA,
    )
    return [p1, p2]


def _create_case05_missing_top_content() -> np.ndarray:
    """
    Case 05: Trang bị cắt mất dòng đầu sát mép trên (`y=0..18`), không có trên trang trước
    -> `CONTENT_TRUNCATED` / `MISSING_CONTENT`.
    """
    img = _blank_canvas(seed=501, bg_value=246)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Vẽ dòng chữ bị cắt ngang thân chữ ngay tại mép trên y=0..16 (touching_edge_px = 0)
    cv2.putText(
        img,
        "TRUNCATED HEADER LINE CLIPPED AT TOP SCANNER EDGE - MISSING UPPER HALF",
        (140, 14),
        font,
        0.82,
        (22,),
        2,
        cv2.LINE_AA,
    )
    # Đảm bảo nét mực chạm sát dòng pixel y=0..4 trên dải rộng để mô phỏng chữ bị xén ngang
    img[0:5, 145:1140 : 4] = 28

    y = 75
    for sec in range(5):
        cv2.putText(
            img,
            f"Section {sec + 1} - Body Content Below Clipped Top Edge",
            (140, y),
            font,
            0.74,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y += 42
        y = _draw_paragraph_block(
            img,
            [
                "The top header of this page was physically clipped by a late scanner trigger.",
                "Because no previous page contains the missing top region, the system marks",
                "this page as MISSING_CONTENT (CONTENT_TRUNCATED) and recommends rescan.",
                "It will never fabricate or hallucinate the missing text lines.",
            ],
            x_start=140,
            y_start=y,
            line_step=36,
        )
        y += 30
    return img


def _create_case06_missing_bottom_content() -> np.ndarray:
    """
    Case 06: Trang bị cắt mất dòng cuối sát mép dưới (`y=1852..1869`), không có trên trang sau
    -> `CONTENT_TRUNCATED` / `MISSING_CONTENT`.
    """
    img = _blank_canvas(seed=601, bg_value=246)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        "TECHNICAL SPECIFICATION - BOTTOM EDGE TRUNCATION CASE",
        (140, 145),
        font,
        0.86,
        (26,),
        2,
        cv2.LINE_AA,
    )
    y = 225
    for sec in range(5):
        cv2.putText(
            img,
            f"Section {sec + 1} - Complete Upper & Middle Paragraphs",
            (140, y),
            font,
            0.74,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y += 42
        y = _draw_paragraph_block(
            img,
            [
                "Upper and middle sections of this document page are completely intact.",
                "However, the scanner frame ended prematurely at the bottom border,",
                "slicing directly through the final paragraph at y = 1869 px.",
                "Without a continuation page, this truncation requires rescan.",
            ],
            x_start=140,
            y_start=y,
            line_step=36,
        )
        y += 34

    # Dòng văn bản sát mép dưới bị cắt cụt ngay tại biên dưới (y=1850..1869)
    cv2.putText(
        img,
        "FINAL LINE SLICED MID-CHARACTER AT BOTTOM IMAGE BORDER - MISSING LOWER HALF",
        (140, PAGE_H_PX - 3),
        font,
        0.78,
        (22,),
        2,
        cv2.LINE_AA,
    )
    img[PAGE_H_PX - 5 : PAGE_H_PX, 145:1150 : 4] = 26
    return img


def _create_case07_content_overlap_pages() -> List[np.ndarray]:
    """
    Case 07: Hai trang liên tiếp có vùng nội dung trùng lặp (`CONTENT_OVERLAP` / `RECONSTRUCTABLE`):
    Cuối Trang 1 (`y=1420..1780`) và đầu Trang 2 (`y=110..470`) chứa cùng một khối văn bản.
    """
    p1 = _blank_canvas(seed=701, bg_value=245)
    p2 = _blank_canvas(seed=702, bg_value=245)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        p1,
        "AUDIT REPORT PAGE 1 - PRECEDING SECTIONS",
        (140, 145),
        font,
        0.86,
        (26,),
        2,
        cv2.LINE_AA,
    )
    y1 = 220
    for sec in range(4):
        cv2.putText(
            p1,
            f"Section 1.{sec + 1} - Primary Audit Findings",
            (140, y1),
            font,
            0.72,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y1 += 40
        y1 = _draw_paragraph_block(
            p1,
            [
                "Standard audit observations recorded on the first page of the report.",
                "Due to manual re-feeding of a long form, the bottom paragraph block",
                "was scanned twice: once at the bottom of Page 1 and again at top of Page 2.",
            ],
            x_start=140,
            y_start=y1,
            line_step=36,
        )
        y1 += 28

    # Tạo dải nội dung trùng lặp 360px
    shared_strip = _blank_canvas(seed=703, bg_value=245, width=PAGE_W_PX, height=360)
    cv2.putText(
        shared_strip,
        "OVERLAP BLOCK ALPHA - DUPLICATED SCAN REGION (REF #7782)",
        (140, 55),
        font,
        0.80,
        (22,),
        2,
        cv2.LINE_AA,
    )
    _draw_paragraph_block(
        shared_strip,
        [
            "This exact six-line text block appears identically at the bottom of Page 1",
            "and at the top of Page 2 because the operator overlapped the scan window.",
            "Normalized Cross-Correlation (NCC), Horizontal Projection Correlation,",
            "and ORB feature matching jointly confirm CONTENT_OVERLAP across pages.",
            "Checksum Token: 9F-44A-OVERLAP-VERIFIED-2026.",
        ],
        x_start=140,
        y_start=105,
        line_step=42,
        font_scale=0.62,
    )

    p1[1410:1770, :] = shared_strip
    p2[110:470, :] = shared_strip

    y2 = 525
    for sec in range(4):
        cv2.putText(
            p2,
            f"Section 2.{sec + 1} - Post-Overlap Continuation on Page 2",
            (140, y2),
            font,
            0.72,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y2 += 40
        y2 = _draw_paragraph_block(
            p2,
            [
                "Unique content of Page 2 continues below the duplicated top strip.",
                "Approved reconstruction removes the duplicate strip while preserving all text.",
                "Provenance metadata records exact source regions from Page 1 and Page 2.",
            ],
            x_start=140,
            y_start=y2,
            line_step=36,
        )
        y2 += 28
    return [p1, p2]


def _create_case08_large_whitespace_normal() -> np.ndarray:
    """
    Case 08: Trang có khoảng trắng lớn giữa 2 mục (`~135px`) nhưng đồng nhất nền, cùng lề,
    không có seam line hay bước nhảy độ sáng -> `NORMAL` (False Positive Control).
    """
    img = _blank_canvas(seed=801, bg_value=246)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        "OFFICIAL POLICY MANUAL - SECTION SPACING SAMPLE",
        (140, 145),
        font,
        0.86,
        (26,),
        2,
        cv2.LINE_AA,
    )
    lines = [
        "This document intentionally includes a generous whitespace gap between Part I",
        "and Part II to separate major chapters on the same physical sheet of paper.",
        "Because paper illumination, margin width, and font geometry remain consistent",
        "with zero physical seam line, the boundary detector must keep it NORMAL.",
    ]
    y = 225
    for blk in range(2):
        cv2.putText(
            img,
            f"Part I.{blk + 1} - General Administrative Provisions",
            (140, y),
            font,
            0.74,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y += 42
        y = _draw_paragraph_block(img, lines, x_start=140, y_start=y, line_step=36)
        y += 28

    # Khoảng trắng lớn ~135px giữa trang (nhưng hoàn toàn cùng nền 246 và cùng lề 140..1180)
    y += 135

    for blk in range(2):
        cv2.putText(
            img,
            f"Part II.{blk + 1} - Technical Implementation Guidelines",
            (140, y),
            font,
            0.74,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y += 42
        y = _draw_paragraph_block(img, lines, x_start=140, y_start=y, line_step=36)
        y += 28

    cv2.putText(
        img,
        "Page 1 of 1 - Normal Large Whitespace Layout",
        (140, PAGE_H_PX - 110),
        font,
        0.58,
        (55,),
        1,
        cv2.LINE_AA,
    )
    return img


def _create_case09_long_table_normal() -> np.ndarray:
    """Case 09: Trang có bảng biểu dài chiếm phần lớn trang -> `NORMAL` (False Positive Control)."""
    img = _blank_canvas(seed=901, bg_value=246)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        "INVENTORY & QUALITY METRICS SCHEDULE (LONG TABLE)",
        (140, 145),
        font,
        0.84,
        (26,),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Table 1. Multi-row structured inventory table with horizontal and vertical grid lines.",
        (140, 195),
        font,
        0.60,
        (40,),
        1,
        cv2.LINE_AA,
    )

    top_y, bottom_y = 235, 1640
    left_x, right_x = 140, 1180
    col_xs = [140, 310, 760, 980, 1180]
    row_step = 78
    row_ys = list(range(top_y, bottom_y + 1, row_step))

    for ry in row_ys:
        cv2.line(img, (left_x, ry), (right_x, ry), (45,), 2)
    for cx in col_xs:
        cv2.line(img, (cx, top_y), (cx, row_ys[-1]), (45,), 2)

    for idx in range(len(row_ys) - 1):
        ty = row_ys[idx] + 48
        if idx == 0:
            cv2.putText(img, "ITEM ID", (160, ty), font, 0.62, (20,), 2, cv2.LINE_AA)
            cv2.putText(img, "MODULE DESCRIPTION", (330, ty), font, 0.62, (20,), 2, cv2.LINE_AA)
            cv2.putText(img, "STATUS", (780, ty), font, 0.62, (20,), 2, cv2.LINE_AA)
            cv2.putText(img, "SCORE", (1005, ty), font, 0.62, (20,), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, f"MOD-{idx:03d}", (160, ty), font, 0.58, (35,), 1, cv2.LINE_AA)
            cv2.putText(
                img,
                f"Document Inspection Component #{idx}",
                (330, ty),
                font,
                0.58,
                (35,),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(img, "VERIFIED", (780, ty), font, 0.58, (35,), 1, cv2.LINE_AA)
            cv2.putText(img, "99.4%", (1005, ty), font, 0.58, (35,), 1, cv2.LINE_AA)

    cv2.putText(
        img,
        "Page 1 of 1 - Authorized Table Schedule",
        (140, PAGE_H_PX - 110),
        font,
        0.58,
        (55,),
        1,
        cv2.LINE_AA,
    )
    return img


def _create_case10_signature_and_rule_normal() -> np.ndarray:
    """
    Case 10: Trang có đường kẻ phân cách nội dung (nằm trong lề văn bản) và chữ ký tay cuối trang
    -> `NORMAL` (False Positive Control).
    """
    img = _blank_canvas(seed=1001, bg_value=246)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        "CERTIFICATE OF COMPLETION & APPROVAL",
        (140, 150),
        font,
        0.88,
        (26,),
        2,
        cv2.LINE_AA,
    )
    # Đường kẻ phân cách tiêu đề ngắn nằm gọn trong lề văn bản (x=140..540), không phải seam giấy toàn trang
    cv2.line(img, (140, 175), (540, 175), (40,), 2)

    y = 240
    for sec in range(3):
        cv2.putText(
            img,
            f"Declaration {sec + 1} - Quality Assurance Attestation",
            (140, y),
            font,
            0.74,
            (30,),
            2,
            cv2.LINE_AA,
        )
        y += 42
        y = _draw_paragraph_block(
            img,
            [
                "We hereby certify that the multi-defect PDF inspection architecture",
                "has been verified against orientation, stripe, and boundary benchmarks.",
                "Content divider rules and handwritten signatures must remain NORMAL.",
            ],
            x_start=140,
            y_start=y,
            line_step=38,
        )
        y += 32

    # Đường kẻ phân cách nội dung thứ hai trước vùng chữ ký (x=680..1080)
    cv2.line(img, (680, y + 10), (1080, y + 10), (50,), 2)
    y += 70

    cv2.putText(
        img,
        "Approved by Chief Quality Architect:",
        (680, y),
        font,
        0.66,
        (32,),
        2,
        cv2.LINE_AA,
    )
    # Chữ ký tay giả lập bằng đường cong đa giác ở góc dưới phải
    sig_pts = np.array(
        [
            [700, y + 95],
            [745, y + 40],
            [775, y + 105],
            [825, y + 48],
            [860, y + 98],
            [920, y + 55],
            [980, y + 88],
            [1060, y + 62],
        ],
        dtype=np.int32,
    )
    cv2.polylines(img, [sig_pts], isClosed=False, color=(28,), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(
        img,
        "Signed & Dated - Internal Quality Board",
        (680, y + 150),
        font,
        0.60,
        (45,),
        1,
        cv2.LINE_AA,
    )
    return img


def _create_case11_two_column_normal() -> np.ndarray:
    """Case 11: Trang bố cục 2 cột văn bản song song hợp lệ -> `NORMAL` (False Positive Control)."""
    img = _blank_canvas(seed=1101, bg_value=246)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        "JOURNAL OF DOCUMENT IMAGE ANALYSIS - TWO-COLUMN LAYOUT",
        (120, 140),
        font,
        0.78,
        (24,),
        2,
        cv2.LINE_AA,
    )
    col_lines = [
        "Two-column technical articles",
        "arrange paragraphs side by side",
        "with a central vertical gutter.",
        "Layout analysis detects left and",
        "right column bounding boxes that",
        "overlap vertically without being",
        "a spliced or mixed page defect.",
    ]
    y_left = 220
    y_right = 220
    for sec in range(4):
        cv2.putText(
            img,
            f"L.{sec + 1} Column Study",
            (120, y_left),
            font,
            0.66,
            (28,),
            2,
            cv2.LINE_AA,
        )
        y_left += 36
        y_left = _draw_paragraph_block(
            img, col_lines, x_start=120, y_start=y_left, line_step=34, font_scale=0.55
        )
        y_left += 28

        cv2.putText(
            img,
            f"R.{sec + 1} Empirical Results",
            (700, y_right),
            font,
            0.66,
            (28,),
            2,
            cv2.LINE_AA,
        )
        y_right += 36
        y_right = _draw_paragraph_block(
            img, col_lines, x_start=700, y_start=y_right, line_step=34, font_scale=0.55
        )
        y_right += 28

    cv2.putText(
        img,
        "Page 1 of 1 - Two-Column Proceeding Format",
        (120, PAGE_H_PX - 105),
        font,
        0.56,
        (55,),
        1,
        cv2.LINE_AA,
    )
    return img


def _create_case12_valid_book_spread() -> np.ndarray:
    """
    Case 12: Trang sách mở đôi hợp lệ (Landscape `1870 x 1320`, `aspect_ratio = 1.417 > 1.18`)
    với trang trái và trang phải đối xứng -> `NORMAL` (False Positive Control).
    """
    spread_w, spread_h = 1870, 1320
    img = _blank_canvas(seed=1201, bg_value=245, width=spread_w, height=spread_h)
    font = cv2.FONT_HERSHEY_SIMPLEX

    spread_lines = [
        "A valid two-up book spread scan captures the left verso page",
        "and the right recto page side-by-side in landscape orientation.",
        "Because the symmetry is horizontal across a vertical book spine,",
        "horizontal page boundary detection recognizes it as a valid spread.",
    ]
    # Trang trái (x=110..840) và Trang phải (x=1030..1760)
    for x_offset, label in [(110, "LEFT VERSO PAGE 14"), (1030, "RIGHT RECTO PAGE 15")]:
        cv2.putText(
            img,
            f"BOOK CHAPTER III - {label}",
            (x_offset, 130),
            font,
            0.72,
            (26,),
            2,
            cv2.LINE_AA,
        )
        y = 195
        for blk in range(4):
            cv2.putText(
                img,
                f"Paragraph {blk + 1} - Archival Book Scan",
                (x_offset, y),
                font,
                0.62,
                (32,),
                2,
                cv2.LINE_AA,
            )
            y += 34
            y = _draw_paragraph_block(
                img,
                spread_lines,
                x_start=x_offset,
                y_start=y,
                line_step=32,
                font_scale=0.52,
            )
            y += 22
        cv2.putText(
            img,
            label,
            (x_offset, spread_h - 85),
            font,
            0.54,
            (55,),
            1,
            cv2.LINE_AA,
        )
    return img


def _create_case13_rotated_180_and_mixed() -> np.ndarray:
    """
    Case 13: Trang vừa bị xoay ngược `180°` vừa bị `MIXED_PAGE` (`y_norm = 0.500`).
    Kiểm thử khả năng phối hợp giữa `OrientationDetector` và `PageBoundaryDetector`.
    """
    mixed_upright = _create_mixed_two_pages_in_one(seed=1301)
    return cv2.rotate(mixed_upright, cv2.ROTATE_180)


def _create_case14_striped_and_mixed() -> np.ndarray:
    """
    Case 14: Trang vừa có vệt sọc dọc đậm (`VERTICAL STRIPE` tại x=98) chạy xuyên qua khoảng trắng
    vừa bị `MIXED_PAGE` (`y_norm = 0.500`). Kiểm thử cơ chế lọc sọc trước khi tính projection profile.
    """
    img = _create_mixed_two_pages_in_one(seed=1401)
    # Vẽ vệt sọc dọc máy scan tại x=98..101 kéo dài toàn trang xuyên qua cả vùng ranh giới y=935
    img[:, 98:101] = 28
    img[:, 97] = 95
    img[:, 101] = 95
    return img


def generate_all_boundary_samples() -> Dict[str, Any]:
    """Sinh toàn bộ 15 file PDF kiểm thử chuẩn và lưu `boundary_ground_truth.json`."""
    BOUNDARY_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    cases_meta: List[Dict[str, Any]] = []

    def _register_case(
        case_id: str,
        file_name: str,
        description: str,
        images: Sequence[np.ndarray],
        pages_gt: List[Dict[str, Any]],
    ) -> None:
        pdf_path = BOUNDARY_BENCHMARK_DIR / file_name
        _save_images_as_pdf(images, pdf_path)
        cases_meta.append(
            {
                "case_id": case_id,
                "file_name": file_name,
                "description": description,
                "page_count": len(images),
                "pages": pages_gt,
            }
        )

    # 1. Normal multi-page PDF (3 trang)
    _register_case(
        case_id="case_01_normal_multi_page",
        file_name="01_normal_multi_page.pdf",
        description="PDF nhiều trang bình thường (3 trang chuẩn)",
        images=[_create_normal_page(i, 3, seed=100) for i in range(1, 4)],
        pages_gt=[
            {
                "page_number": i,
                "expected_has_boundary_issue": False,
                "expected_boundary_status": ["NORMAL"],
                "expected_defect_types": [],
                "ground_truth_boundary_norm": None,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            }
            for i in range(1, 4)
        ],
    )

    # 2. Two pages in one image (MIXED_PAGE)
    _register_case(
        case_id="case_02_two_pages_in_one_image",
        file_name="02_two_pages_in_one_image.pdf",
        description="2 trang vật lý nằm chung trong 1 ảnh trang (MIXED_PAGE tại y_norm=0.500)",
        images=[_create_mixed_two_pages_in_one(seed=202)],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["MIXED_PAGE_DETECTED", "RECONSTRUCTABLE"],
                "expected_defect_types": ["MIXED_PAGE"],
                "ground_truth_boundary_norm": 0.500,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            }
        ],
    )

    # 3. Page A with top of Page B (MIXED_PAGE + related page 2)
    _register_case(
        case_id="case_03_page_a_with_top_of_b",
        file_name="03_page_a_with_top_of_b.pdf",
        description="Trang A bị dính phần đầu của Trang B ở cuối trang (y_norm=0.700, khớp với đầu Trang 2)",
        images=_create_case03_page_a_with_top_of_b(),
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["MIXED_PAGE_DETECTED", "RECONSTRUCTABLE"],
                "expected_defect_types": ["MIXED_PAGE", "CONTENT_OVERLAP"],
                "ground_truth_boundary_norm": 0.700,
                "expected_related_pages": [2],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            },
            {
                "page_number": 2,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": [
                    "RECONSTRUCTABLE",
                    "MIXED_PAGE_DETECTED",
                    "BOUNDARY_SUSPECTED",
                    "NORMAL",
                ],
                "expected_defect_types": ["CONTENT_OVERLAP", "MIXED_PAGE"],
                "ground_truth_boundary_norm": None,
                "expected_related_pages": [1],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
                "optional_companion_page": True,
            },
        ],
    )

    # 4. Page B continuation from previous page (SPLIT_PAGE)
    _register_case(
        case_id="case_04_page_b_continuation_from_prev",
        file_name="04_page_b_continuation_from_prev.pdf",
        description="Một trang vật lý bị tách làm 2 trang liên tiếp (Trang 2 tiếp nối từ Trang 1 - SPLIT_PAGE)",
        images=_create_case04_split_page_across_two(),
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["RECONSTRUCTABLE", "BOUNDARY_SUSPECTED"],
                "expected_defect_types": ["SPLIT_PAGE"],
                "ground_truth_boundary_norm": 0.460,
                "expected_related_pages": [2],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            },
            {
                "page_number": 2,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["RECONSTRUCTABLE", "BOUNDARY_SUSPECTED"],
                "expected_defect_types": ["SPLIT_PAGE"],
                "ground_truth_boundary_norm": 0.460,
                "expected_related_pages": [1],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            },
        ],
    )

    # 5. Missing top content (CONTENT_TRUNCATED / MISSING_CONTENT)
    _register_case(
        case_id="case_05_missing_top_content",
        file_name="05_missing_top_content.pdf",
        description="Trang bị cắt mất dòng đầu sát mép trên, không có trên trang trước (MISSING_CONTENT)",
        images=[_create_case05_missing_top_content()],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["MISSING_CONTENT"],
                "expected_defect_types": ["CONTENT_TRUNCATED"],
                "ground_truth_boundary_norm": 0.015,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            }
        ],
    )

    # 6. Missing bottom content (CONTENT_TRUNCATED / MISSING_CONTENT)
    _register_case(
        case_id="case_06_missing_bottom_content",
        file_name="06_missing_bottom_content.pdf",
        description="Trang bị cắt mất dòng cuối sát mép dưới, không có trên trang sau (MISSING_CONTENT)",
        images=[_create_case06_missing_bottom_content()],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["MISSING_CONTENT"],
                "expected_defect_types": ["CONTENT_TRUNCATED"],
                "ground_truth_boundary_norm": 0.985,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            }
        ],
    )

    # 7. Content overlap pages (CONTENT_OVERLAP / RECONSTRUCTABLE)
    _register_case(
        case_id="case_07_content_overlap_pages",
        file_name="07_content_overlap_pages.pdf",
        description="Hai trang liên tiếp bị trùng lặp dải nội dung giữa cuối Trang 1 và đầu Trang 2 (CONTENT_OVERLAP)",
        images=_create_case07_content_overlap_pages(),
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["RECONSTRUCTABLE"],
                "expected_defect_types": ["CONTENT_OVERLAP"],
                "ground_truth_boundary_norm": 0.800,
                "expected_related_pages": [2],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            },
            {
                "page_number": 2,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["RECONSTRUCTABLE"],
                "expected_defect_types": ["CONTENT_OVERLAP"],
                "ground_truth_boundary_norm": 0.200,
                "expected_related_pages": [1],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            },
        ],
    )

    # 8. Large whitespace normal page (False Positive Control)
    _register_case(
        case_id="case_08_large_whitespace_normal",
        file_name="08_large_whitespace_normal.pdf",
        description="Trang có khoảng trắng lớn giữa 2 phần nhưng không bị lỗi ranh giới (NORMAL)",
        images=[_create_case08_large_whitespace_normal()],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": False,
                "expected_boundary_status": ["NORMAL"],
                "expected_defect_types": [],
                "ground_truth_boundary_norm": None,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            }
        ],
    )

    # 9. Long table normal page (False Positive Control)
    _register_case(
        case_id="case_09_long_table_normal",
        file_name="09_long_table_normal.pdf",
        description="Trang có bảng biểu dài nhiều dòng kẻ ngang/dọc hợp lệ (NORMAL)",
        images=[_create_case09_long_table_normal()],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": False,
                "expected_boundary_status": ["NORMAL"],
                "expected_defect_types": [],
                "ground_truth_boundary_norm": None,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            }
        ],
    )

    # 10. Signature and horizontal divider rule normal page (False Positive Control)
    _register_case(
        case_id="case_10_signature_and_rule_normal",
        file_name="10_signature_and_rule_normal.pdf",
        description="Trang có chữ ký cuối trang và đường kẻ ngang phân cách nội dung hợp lệ (NORMAL)",
        images=[_create_case10_signature_and_rule_normal()],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": False,
                "expected_boundary_status": ["NORMAL"],
                "expected_defect_types": [],
                "ground_truth_boundary_norm": None,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            }
        ],
    )

    # 11. Two-column normal page (False Positive Control)
    _register_case(
        case_id="case_11_two_column_normal",
        file_name="11_two_column_normal.pdf",
        description="Trang có bố cục 2 cột văn bản song song hợp lệ (NORMAL)",
        images=[_create_case11_two_column_normal()],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": False,
                "expected_boundary_status": ["NORMAL"],
                "expected_defect_types": [],
                "ground_truth_boundary_norm": None,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            }
        ],
    )

    # 12. Valid 2-up book spread (False Positive Control)
    _register_case(
        case_id="case_12_valid_book_spread",
        file_name="12_valid_book_spread.pdf",
        description="Trang sách mở đôi khổ ngang hợp lệ (NORMAL)",
        images=[_create_case12_valid_book_spread()],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": False,
                "expected_boundary_status": ["NORMAL"],
                "expected_defect_types": [],
                "ground_truth_boundary_norm": None,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            }
        ],
    )

    # 13. Rotated 180° + MIXED_PAGE (Multi-Defect)
    _register_case(
        case_id="case_13_rotated_180_and_mixed_page",
        file_name="13_rotated_180_and_mixed_page.pdf",
        description="Trang vừa xoay ngược 180 độ vừa bị MIXED_PAGE (Kiểm thử phối hợp Orientation + Boundary)",
        images=[_create_case13_rotated_180_and_mixed()],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["MIXED_PAGE_DETECTED", "RECONSTRUCTABLE"],
                "expected_defect_types": ["MIXED_PAGE"],
                "ground_truth_boundary_norm": 0.500,
                "expected_related_pages": [],
                "expected_orientation_status": "ROTATED",
                "expected_stripe_status": "CLEAN",
            }
        ],
    )

    # 14. Striped + MIXED_PAGE (Multi-Defect)
    _register_case(
        case_id="case_14_striped_and_mixed_page",
        file_name="14_striped_and_mixed_page.pdf",
        description="Trang vừa có vệt sọc dọc STRIPE xuyên qua khoảng trắng vừa bị MIXED_PAGE",
        images=[_create_case14_striped_and_mixed()],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["MIXED_PAGE_DETECTED", "RECONSTRUCTABLE"],
                "expected_defect_types": ["MIXED_PAGE"],
                "ground_truth_boundary_norm": 0.500,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "STRIPE_DETECTED",
            }
        ],
    )

    # 15. Multi-page multi-defect PDF (4 trang kết hợp nhiều lỗi)
    p1_clean = _create_normal_page(1, 4, seed=1501)
    p2_rot_stripe = cv2.rotate(_create_normal_page(2, 4, seed=1502), cv2.ROTATE_180)
    p2_rot_stripe[:, 96:99] = 26
    p3_mixed = _create_mixed_two_pages_in_one(seed=1503)
    p4_trunc = _create_case05_missing_top_content()
    _register_case(
        case_id="case_15_multi_page_multi_defect",
        file_name="15_multi_page_multi_defect.pdf",
        description="PDF 4 trang kết hợp nhiều lỗi: Trang 1 sạch, Trang 2 xoay 180° + sọc dọc, Trang 3 MIXED_PAGE, Trang 4 MISSING_CONTENT",
        images=[p1_clean, p2_rot_stripe, p3_mixed, p4_trunc],
        pages_gt=[
            {
                "page_number": 1,
                "expected_has_boundary_issue": False,
                "expected_boundary_status": ["NORMAL"],
                "expected_defect_types": [],
                "ground_truth_boundary_norm": None,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            },
            {
                "page_number": 2,
                "expected_has_boundary_issue": False,
                "expected_boundary_status": ["NORMAL"],
                "expected_defect_types": [],
                "ground_truth_boundary_norm": None,
                "expected_related_pages": [],
                "expected_orientation_status": "ROTATED",
                "expected_stripe_status": "STRIPE_DETECTED",
            },
            {
                "page_number": 3,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["MIXED_PAGE_DETECTED", "RECONSTRUCTABLE"],
                "expected_defect_types": ["MIXED_PAGE"],
                "ground_truth_boundary_norm": 0.500,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            },
            {
                "page_number": 4,
                "expected_has_boundary_issue": True,
                "expected_boundary_status": ["MISSING_CONTENT"],
                "expected_defect_types": ["CONTENT_TRUNCATED"],
                "ground_truth_boundary_norm": 0.015,
                "expected_related_pages": [],
                "expected_orientation_status": "NORMAL",
                "expected_stripe_status": "CLEAN",
            },
        ],
    )

    gt_payload = {
        "dataset_name": "PDF Page Boundary & Mixed-Page Detection Benchmark (Phase 3.10)",
        "verified_ground_truth": True,
        "render_dpi": RENDER_DPI,
        "page_dimensions_px": [PAGE_W_PX, PAGE_H_PX],
        "total_cases": len(cases_meta),
        "cases": cases_meta,
    }
    GROUND_TRUTH_PATH.write_text(
        json.dumps(gt_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return gt_payload


if __name__ == "__main__":
    payload = generate_all_boundary_samples()
    print(
        f"Đã tạo thành công {payload['total_cases']} bộ mẫu kiểm thử Boundary tại: {BOUNDARY_BENCHMARK_DIR}"
    )
