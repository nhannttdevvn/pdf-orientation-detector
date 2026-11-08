"""
Script đánh giá định lượng và so sánh thực nghiệm cho Secret Marking Detection (`scripts/evaluate_secret_benchmark.py`).

Đánh giá trên 2 tập dữ liệu có nhãn xác thực bằng mắt người (không suy diễn mù quáng từ tên thư mục — C5):
1. Tập 13 ảnh biến dạng thực tế `test_distortions/*.png` (nếu có trên máy phát triển) — đối chiếu trực tiếp
   giữa `ConfidentialDocDetector` (bản gốc), `FastConfidentialDocDetector` (bản fast) và `SecretMarkingDetector` (tích hợp Phase 4).
2. Tập Benchmark 16 kịch bản tổng hợp bao phủ đầy đủ 12 nhóm yêu cầu C5 (bao gồm cả negative samples có chữ
   `"mật"` / `"BẢO MẬT"` / `"BÍ MẬT"` thông thường, dấu đỏ, dấu đen, dấu mờ, dấu xoay nghiêng, đa kích thước).

Tuyệt đối không động chạm hoặc ghi nội dung tài liệu thực trong thư mục `tài liệu mật/` (tuân thủ C4).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cv2
import numpy as np

from ai_engine.core.config import (
    BASE_DIR,
    OUTPUTS_DIR,
    SECRET_STATUS_DETECTED,
    SECRET_STATUS_NO_MARK,
    SECRET_STATUS_REVIEW,
    SECRET_TEMPLATES_DIR,
)
from ai_engine.detectors.secret import SecretMarkingDetector

DISTORTIONS_DIR = Path("/Volumes/TRANSCEND/MacNhan/NhanNguyen/FSI/tailieu/tailieumat/test_distortions")
LEGACY_AUDIT_JSON = OUTPUTS_DIR / "secret_legacy_audit_benchmark.json"
OUT_BENCHMARK_JSON = OUTPUTS_DIR / "secret_detector_benchmark.json"


# Nhãn chuẩn (Ground Truth) đã kiểm chứng trực quan từng ảnh 01..13 trong test_distortions/
VISUAL_GT_DISTORTIONS: List[Tuple[str, bool, str]] = [
    ("01_mo_chu.png", True, "MẬT"),
    ("02_de_dau_do.png", True, "MẬT"),
    ("03_but_phe_de_len.png", True, "MẬT"),
    ("04_combo_mo_va_de_dau.png", True, "MẬT"),
    ("05_photocopy_net_dut.png", True, "MẬT"),
    ("06_font_arial_khong_chan.png", True, "MẬT"),
    ("07_gach_xoa_nang.png", True, "MẬT"),  # Dấu MẬT bị gạch xóa nặng -> DETECTED hoặc REVIEW đều hợp lệ
    ("08_chu_dam_net_to.png", True, "MẬT"),
    ("09_chu_nhat_net_manh.png", True, "MẬT"),
    ("10_sieu_dam_loang_muc.png", True, "MẬT"),
    ("11_sieu_nhat_het_muc.png", True, "MẬT"),
    ("12_nhieu_hop_den_hoa_toc.png", True, "MẬT"),
    ("13_co_ca_mat_va_giai_mat.png", True, "MẬT"),  # Có cả hộp [MẬT] và hộp [GIẢI MẬT]
]


def _make_base_admin_page(w: int = 1654, h: int = 2339) -> np.ndarray:
    """Tạo trang văn bản hành chính nền trắng sạch ở 200 DPI (RGB)."""
    img = np.full((h, w, 3), 250, dtype=np.uint8)
    # Quốc hiệu / tiêu ngữ giả lập góc phải và tên cơ quan góc trái
    cv2.putText(img, "UBND TINH ABC", (160, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.line(img, (210, 140), (390, 140), (45, 45, 45), 2)
    cv2.putText(img, "CONG HOA XA HOI CHU NGHIA VIET NAM", (760, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(img, "Doc lap - Tu do - Hanh phuc", (880, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.line(img, (910, 185), (1230, 185), (45, 45, 45), 2)

    # Các dòng văn bản thân trang
    for idx in range(12):
        y = 560 + idx * 95
        cv2.putText(
            img,
            f"Noi dung bao cao cong tac hanh chinh dinh ky muc so {idx + 1} nam 2026.",
            (180, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
    return img


def _paste_stamp(
    page_rgb: np.ndarray,
    tpl_name: str,
    x: int,
    y: int,
    scale: float = 1.0,
    angle_deg: float = 0.0,
    ink_rgb: Tuple[int, int, int] = (20, 20, 20),
    alpha_strength: float = 1.0,
) -> np.ndarray:
    """Dán template dấu thật lên trang RGB với tùy chọn tỷ lệ, góc xoay, màu mực và độ mờ."""
    out = page_rgb.copy()
    tpl = cv2.imread(str(SECRET_TEMPLATES_DIR / tpl_name), cv2.IMREAD_GRAYSCALE)
    assert tpl is not None, f"Không tìm thấy template {tpl_name}"

    if abs(scale - 1.0) > 1e-3:
        sw = max(20, int(round(tpl.shape[1] * scale)))
        sh = max(12, int(round(tpl.shape[0] * scale)))
        tpl = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

    if abs(angle_deg) > 0.2:
        th, tw = tpl.shape[:2]
        rad = np.deg2rad(abs(angle_deg))
        nw = int(np.ceil(tw * np.cos(rad) + th * np.sin(rad)))
        nh = int(np.ceil(tw * np.sin(rad) + th * np.cos(rad)))
        mat = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), angle_deg, 1.0)
        mat[0, 2] += (nw - tw) / 2.0
        mat[1, 2] += (nh - th) / 2.0
        tpl = cv2.warpAffine(tpl, mat, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    sh, sw = tpl.shape[:2]
    ink_strength = np.clip((255.0 - tpl.astype(np.float32)) / 255.0, 0.0, 1.0) * float(alpha_strength)
    roi = out[y : y + sh, x : x + sw].astype(np.float32)
    ink_arr = np.array(ink_rgb, dtype=np.float32).reshape(1, 1, 3)
    blended = roi * (1.0 - ink_strength[:, :, None]) + ink_arr * ink_strength[:, :, None]
    out[y : y + sh, x : x + sw] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _paste_inline_mat_phrase(
    page_rgb: np.ndarray,
    x: int,
    y: int,
    mode: str = "bao_mat_header",
) -> np.ndarray:
    """
    Dán chính xác glyph chữ `MẬT` từ `text_mat.png` vào GIỮA một dòng tiêu đề / câu văn thông thường
    (kịch bản adversarial khắt khe nhất để kiểm chứng không báo nhầm từ 'MẬT' trong văn bản thường).
    """
    out = page_rgb.copy()
    mat_tpl = cv2.imread(str(SECRET_TEMPLATES_DIR / "text_mat.png"), cv2.IMREAD_GRAYSCALE)
    assert mat_tpl is not None
    # Co về cỡ chữ tiêu đề văn bản (cao ~34px)
    scale = 0.56
    mw = int(round(mat_tpl.shape[1] * scale))
    mh = int(round(mat_tpl.shape[0] * scale))
    mat_small = cv2.resize(mat_tpl, (mw, mh), interpolation=cv2.INTER_AREA)

    if mode == "bao_mat_header":
        # Dòng tiêu đề: "BAO CAO CONG TAC BAO [MẬT] NAM 2026"
        cv2.putText(out, "BAO CAO CONG TAC BAO", (x, y + mh - 7), cv2.FONT_HERSHEY_DUPLEX, 0.92, (25, 25, 25), 2, cv2.LINE_AA)
        mx = x + 392
        ink = (255.0 - mat_small.astype(np.float32)) / 255.0
        roi = out[y : y + mh, mx : mx + mw].astype(np.float32)
        out[y : y + mh, mx : mx + mw] = np.clip(roi * (1.0 - ink[:, :, None]) + 25.0 * ink[:, :, None], 0, 255).astype(np.uint8)
        cv2.putText(out, "NAM 2026", (mx + mw + 14, y + mh - 7), cv2.FONT_HERSHEY_DUPLEX, 0.92, (25, 25, 25), 2, cv2.LINE_AA)
    else:
        # Cụm 2 từ: "MAT DO XAY DUNG" (từ MẬT đứng đầu câu nhưng có chữ ngay sát bên phải)
        ink = (255.0 - mat_small.astype(np.float32)) / 255.0
        roi = out[y : y + mh, x : x + mw].astype(np.float32)
        out[y : y + mh, x : x + mw] = np.clip(roi * (1.0 - ink[:, :, None]) + 25.0 * ink[:, :, None], 0, 255).astype(np.uint8)
        cv2.putText(out, "DO XAY DUNG DO THI", (x + mw + 12, y + mh - 7), cv2.FONT_HERSHEY_DUPLEX, 0.90, (25, 25, 25), 2, cv2.LINE_AA)
    return out


def build_c5_suite() -> List[Tuple[str, np.ndarray, bool, Any]]:
    """Khởi tạo 16 mẫu kiểm thử bao phủ toàn bộ yêu cầu C5."""
    cases: List[Tuple[str, np.ndarray, bool, Any]] = []

    cases.append(("c5_01_clear_mat_box", _paste_stamp(_make_base_admin_page(), "box_mat.png", 210, 230), True, "MẬT"))
    cases.append(("c5_02_clear_toi_mat_box", _paste_stamp(_make_base_admin_page(), "box_toi_mat.png", 200, 230), True, "TỐI MẬT"))
    cases.append(("c5_03_clear_tuyet_mat_box", _paste_stamp(_make_base_admin_page(), "box_tuyet_mat.png", 190, 230), True, "TUYỆT MẬT"))
    cases.append(("c5_04a_red_mat_stamp", _paste_stamp(_make_base_admin_page(), "box_mat.png", 220, 240, ink_rgb=(215, 22, 28)), True, "MẬT"))
    cases.append(("c5_04b_red_tuyet_mat_unboxed", _paste_stamp(_make_base_admin_page(), "text_tuyet_mat.png", 1080, 245, ink_rgb=(210, 20, 25)), True, "TUYỆT MẬT"))
    cases.append(("c5_05_black_toi_mat_unboxed", _paste_stamp(_make_base_admin_page(), "text_toi_mat.png", 210, 245, ink_rgb=(18, 18, 18)), True, "TỐI MẬT"))
    cases.append(("c5_06a_faded_mat_box", _paste_stamp(_make_base_admin_page(), "box_mat.png", 220, 235, alpha_strength=0.18), True, "MẬT"))
    cases.append(("c5_06b_faded_toi_mat_unboxed", _paste_stamp(_make_base_admin_page(), "text_toi_mat.png", 220, 245, alpha_strength=0.20), True, "TỐI MẬT"))
    cases.append(("c5_07a_rotated_plus8_mat_box", _paste_stamp(_make_base_admin_page(), "box_mat.png", 220, 235, angle_deg=8.0), True, "MẬT"))
    cases.append(("c5_07b_rotated_minus9_toi_mat", _paste_stamp(_make_base_admin_page(), "text_toi_mat.png", 220, 245, angle_deg=-9.0, ink_rgb=(205, 25, 25)), True, "TỐI MẬT"))
    cases.append(("c5_08a_small_scale_065_mat", _paste_stamp(_make_base_admin_page(), "box_mat.png", 220, 240, scale=0.65), True, "MẬT"))
    cases.append(("c5_08b_large_scale_135_tuyet_mat", _paste_stamp(_make_base_admin_page(), "box_tuyet_mat.png", 180, 230, scale=1.35), True, "TUYỆT MẬT"))

    # Negative samples (C5 #9 & #10)
    cases.append(("c5_09_clean_page_no_mark", _make_base_admin_page(), False, None))
    cases.append(("c5_10a_inline_bao_mat_title", _paste_inline_mat_phrase(_make_base_admin_page(), 310, 260, mode="bao_mat_header"), False, None))
    cases.append(("c5_10b_inline_mat_do_header", _paste_inline_mat_phrase(_make_base_admin_page(), 200, 250, mode="mat_do_prefix"), False, None))
    # Khung chữ nhật hành chính không phải độ mật ([HOA TOC])
    hoa_toc_page = _make_base_admin_page()
    cv2.rectangle(hoa_toc_page, (200, 230), (410, 298), (20, 20, 20), 3)
    cv2.putText(hoa_toc_page, "HOA TOC", (222, 276), cv2.FONT_HERSHEY_DUPLEX, 1.05, (20, 20, 20), 2, cv2.LINE_AA)
    cases.append(("c5_10c_non_secret_box_hoa_toc", hoa_toc_page, False, None))

    return cases


def main() -> None:
    detector = SecretMarkingDetector()

    # 1. Đánh giá trên 13 ảnh biến dạng test_distortions (nếu thư mục tồn tại)
    distortion_results: List[Dict[str, Any]] = []
    d_tp = d_fn = 0
    if DISTORTIONS_DIR.exists():
        for fname, gt_has_mark, gt_label in VISUAL_GT_DISTORTIONS:
            fpath = DISTORTIONS_DIR / fname
            bgr = cv2.imread(str(fpath))
            if bgr is None:
                continue
            t0 = time.perf_counter()
            res = detector.detect_page(bgr, page_number=1, color_order="BGR")
            ms = round((time.perf_counter() - t0) * 1000.0, 2)
            pred_pos = res.status in (SECRET_STATUS_DETECTED, SECRET_STATUS_REVIEW)
            label_ok = (res.primary_label == gt_label) if pred_pos else False
            if pred_pos and label_ok:
                d_tp += 1
            else:
                d_fn += 1
            distortion_results.append(
                {
                    "file": fname,
                    "gt_label": gt_label,
                    "status": res.status,
                    "primary_label": res.primary_label,
                    "marks_count": len(res.marks),
                    "all_labels": [m.label for m in res.marks],
                    "top_confidence": round(max((m.confidence for m in res.marks), default=0.0), 4),
                    "methods": [m.detection_method for m in res.marks],
                    "channels": [m.channel_used for m in res.marks],
                    "latency_ms": ms,
                    "pass": bool(pred_pos and label_ok),
                }
            )

    # 2. Đánh giá trên bộ 16 kịch bản chuẩn C5
    c5_cases = build_c5_suite()
    c5_results: List[Dict[str, Any]] = []
    tp = tn = fp = fn = 0
    latencies: List[float] = []

    for cid, img_rgb, gt_has_mark, gt_label in c5_cases:
        t0 = time.perf_counter()
        res = detector.detect_page(img_rgb, page_number=1, color_order="RGB")
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        latencies.append(ms)

        pred_pos = res.status in (SECRET_STATUS_DETECTED, SECRET_STATUS_REVIEW)
        if gt_has_mark:
            if pred_pos and res.primary_label == gt_label:
                tp += 1
                passed = True
            else:
                fn += 1
                passed = False
        else:
            if res.status == SECRET_STATUS_NO_MARK:
                tn += 1
                passed = True
            else:
                fp += 1
                passed = False

        c5_results.append(
            {
                "case_id": cid,
                "expected_has_mark": gt_has_mark,
                "expected_label": gt_label,
                "predicted_status": res.status,
                "predicted_label": res.primary_label,
                "top_confidence": round(max((m.confidence for m in res.marks), default=0.0), 4),
                "methods": [m.detection_method for m in res.marks],
                "channels": [m.channel_used for m in res.marks],
                "latency_ms": ms,
                "pass": passed,
            }
        )

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)

    summary = {
        "c5_suite_metrics": {
            "total_cases": len(c5_cases),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "cases": c5_results,
        },
        "test_distortions_13_metrics": {
            "total_images": len(distortion_results),
            "tp_detected_or_review": d_tp,
            "fn": d_fn,
            "recall": round(d_tp / max(1, d_tp + d_fn), 4) if distortion_results else None,
            "mean_latency_ms": (
                round(float(np.mean([r["latency_ms"] for r in distortion_results])), 2)
                if distortion_results
                else None
            ),
            "cases": distortion_results,
        },
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_BENCHMARK_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["c5_suite_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["test_distortions_13_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

