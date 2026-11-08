#!/usr/bin/env python3
"""
Trình đánh giá định lượng (Benchmark Evaluator) cho Phase 2 — Stripe Defect Detection
(`scripts/evaluate_stripe_benchmark.py`).

Đo đạc thực tế trên tập dữ liệu có nhãn chuẩn (`data/samples/stripe_benchmark/stripe_ground_truth.json`):
- Precision, Recall, F1-score, Accuracy
- False Positive Rate (FPR)
- Độ chính xác định vị vệt sọc (Positional / BBox overlap accuracy)
- Thời gian xử lý trung bình mỗi trang (ms/page)
- Kiểm tra hồi quy (Regression) trên 12 PDF mẫu Orientation (`data/samples/`) và file scan thực tế `0041.pdf`.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ai_engine.core.config import OUTPUTS_DIR, SAMPLES_DIR, UPLOADS_DIR
from ai_engine.detectors.defects import StripeDetector
from ai_engine.pipelines.document_pipeline import DocumentPipeline
from ai_engine.preprocessing.pdf_loader import PDFLoader
from benchmarks.generators.generate_stripe_samples import (
    GROUND_TRUTH_PATH,
    STRIPE_BENCHMARK_DIR,
    generate_all_stripe_samples,
)


def _compute_bbox_iou_and_axis_error(
    pred_bbox: Tuple[int, int, int, int],
    gt_bbox: List[int],
    stripe_type: str,
) -> Tuple[float, float]:
    """
    Tính IoU 2D và sai số vị trí tâm trục chính (theo pixel) giữa `pred_bbox` và `gt_bbox`.
    """
    px0, py0, px1, py1 = pred_bbox
    gx0, gy0, gx1, gy1 = gt_bbox

    ix0 = max(px0, gx0)
    iy0 = max(py0, gy0)
    ix1 = min(px1, gx1)
    iy1 = min(py1, gy1)
    inter_w = max(0, ix1 - ix0)
    inter_h = max(0, iy1 - iy0)
    inter_area = float(inter_w * inter_h)

    p_area = float(max(1, (px1 - px0) * (py1 - py0)))
    g_area = float(max(1, (gx1 - gx0) * (gy1 - gy0)))
    union_area = max(1.0, p_area + g_area - inter_area)
    iou = inter_area / union_area

    if stripe_type == "HORIZONTAL":
        p_center = 0.5 * (py0 + py1)
        g_center = 0.5 * (gy0 + gy1)
    else:
        p_center = 0.5 * (px0 + px1)
        g_center = 0.5 * (gx0 + gx1)
    axis_err_px = abs(p_center - g_center)
    return round(iou, 4), round(axis_err_px, 2)


def run_benchmark() -> Dict[str, Any]:
    if not GROUND_TRUTH_PATH.exists():
        generate_all_stripe_samples()

    gt_data = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    detector = StripeDetector()
    pipeline = DocumentPipeline()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    type_correct = 0
    positive_total = 0
    ious: List[float] = []
    axis_errors: List[float] = []
    stripe_times_ms: List[float] = []
    case_records: List[Dict[str, Any]] = []

    for case in gt_data["cases"]:
        pdf_path = STRIPE_BENCHMARK_DIR / case["file_name"]
        exp_has = bool(case["expected_has_stripe"])
        exp_status = str(case["expected_stripe_status"])
        exp_type = case.get("expected_stripe_type")
        gt_bbox = case.get("gt_bbox")

        with PDFLoader(pdf_path) as loader:
            ctx = loader.build_page_context(1, dpi=160, grayscale=True)
            res = detector.detect_page(
                image=ctx.original_image,
                page_number=1,
                dpi=160,
                clahe_image=ctx.ensure_clahe(),
            )
            ctx.release()

        stripe_times_ms.append(res.processing_time_ms)
        pred_has = bool(res.has_stripe)

        if exp_has and pred_has:
            tp += 1
        elif not exp_has and not pred_has:
            tn += 1
        elif not exp_has and pred_has:
            fp += 1
        else:
            fn += 1

        top_stripe = res.stripes[0] if res.stripes else None
        iou_val: Optional[float] = None
        axis_err_val: Optional[float] = None

        if exp_has:
            positive_total += 1
            if top_stripe is not None and top_stripe.stripe_type == exp_type:
                type_correct += 1
            if top_stripe is not None and gt_bbox is not None:
                iou_val, axis_err_val = _compute_bbox_iou_and_axis_error(
                    top_stripe.bbox, gt_bbox, exp_type or "VERTICAL"
                )
                ious.append(iou_val)
                axis_errors.append(axis_err_val)

        case_records.append(
            {
                "case_id": case["case_id"],
                "file_name": case["file_name"],
                "expected_status": exp_status,
                "predicted_status": res.status,
                "expected_type": exp_type,
                "predicted_type": top_stripe.stripe_type if top_stripe else None,
                "suspicion_score": top_stripe.suspicion_score if top_stripe else 0.0,
                "predicted_bbox": list(top_stripe.bbox) if top_stripe else None,
                "gt_bbox": gt_bbox,
                "bbox_iou": iou_val,
                "axis_center_error_px": axis_err_val,
                "processing_time_ms": res.processing_time_ms,
                "match": (exp_has == pred_has) and (res.status == exp_status),
            }
        )

    precision = tp / float(max(1, tp + fp))
    recall = tp / float(max(1, tp + fn))
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / float(max(1, fp + tn))
    accuracy = (tp + tn) / float(max(1, tp + tn + fp + fn))
    mean_iou = sum(ious) / float(max(1, len(ious)))
    mean_axis_err = sum(axis_errors) / float(max(1, len(axis_errors)))
    mean_time_ms = sum(stripe_times_ms) / float(max(1, len(stripe_times_ms)))

    # Đánh giá trên 11 PDF mẫu Orientation (Clean đối với Stripe) để kiểm tra False Positive mở rộng
    ori_sample_files = [
        "01_all_normal.pdf",
        "02_single_upside_down_180.pdf",
        "03_multiple_upside_down.pdf",
        "04_rotated_90.pdf",
        "05_rotated_270.pdf",
        "06_all_four_orientations.pdf",
        "07_blank_page.pdf",
        "08_very_few_chars.pdf",
        "09_noisy_blurry_scan.pdf",
        "11_metadata_rotation_nonzero.pdf",
        "12_mixed_vector_and_scan.pdf",
    ]
    ori_total_pages = 0
    ori_clean_pages = 0
    for fname in ori_sample_files:
        fpath = SAMPLES_DIR / fname
        if not fpath.exists():
            continue
        with PDFLoader(fpath) as loader:
            for pno in range(1, loader.page_count + 1):
                ori_total_pages += 1
                ctx = loader.build_page_context(pno, dpi=160, grayscale=True)
                s_res = detector.detect_page(
                    ctx.original_image,
                    page_number=pno,
                    dpi=160,
                    clahe_image=ctx.ensure_clahe(),
                )
                ctx.release()
                if s_res.status == "CLEAN":
                    ori_clean_pages += 1

    # Đánh giá trên file PDF scan thực tế 38 trang (data/uploads/6b63db7ab409_0041.pdf)
    real_pdf_path = UPLOADS_DIR / "6b63db7ab409_0041.pdf"
    real_pdf_summary: Dict[str, Any] = {}
    if real_pdf_path.exists():
        t0 = time.perf_counter()
        pipe_res = pipeline.scan_document(real_pdf_path)
        total_sec = round(time.perf_counter() - t0, 3)
        s_sum = pipe_res.report_data["summary"]["stripe_summary"]
        real_pdf_summary = {
            "file_name": real_pdf_path.name,
            "total_pages": pipe_res.doc_info.page_count,
            "orientation_normal": pipe_res.report_data["summary"]["normal_pages"],
            "orientation_rotated": pipe_res.report_data["summary"]["rotated_pages"],
            "orientation_review": pipe_res.report_data["summary"]["review_pages"],
            "stripe_summary": s_sum,
            "total_pipeline_time_seconds": total_sec,
            "avg_pipeline_time_ms_per_page": round(
                total_sec * 1000.0 / max(1, pipe_res.doc_info.page_count), 1
            ),
            "detected_stripe_pages": [
                d.page_number
                for d in pipe_res.decisions
                if d.stripe_status == "STRIPE_DETECTED"
            ],
            "review_stripe_pages": [
                d.page_number
                for d in pipe_res.decisions
                if d.stripe_status == "REVIEW"
            ],
        }

    metrics = {
        "synthetic_benchmark": {
            "total_cases": len(case_records),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "false_positive_rate": round(fpr, 4),
            "accuracy": round(accuracy, 4),
            "stripe_type_accuracy": round(
                type_correct / float(max(1, positive_total)), 4
            ),
            "mean_bbox_iou": round(mean_iou, 4),
            "mean_axis_center_error_px": round(mean_axis_err, 2),
            "mean_stripe_processing_time_ms": round(mean_time_ms, 2),
            "cases": case_records,
        },
        "orientation_samples_negative_control": {
            "total_files": len(ori_sample_files),
            "total_pages": ori_total_pages,
            "clean_pages": ori_clean_pages,
            "false_positive_pages": ori_total_pages - ori_clean_pages,
            "false_positive_rate": round(
                (ori_total_pages - ori_clean_pages) / float(max(1, ori_total_pages)), 4
            ),
        },
        "real_scanned_pdf_0041": real_pdf_summary,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUTS_DIR / "stripe_benchmark_metrics.json"
    out_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    m = run_benchmark()
    print(json.dumps(m["synthetic_benchmark"], ensure_ascii=False, indent=2))
    print(json.dumps(m["orientation_samples_negative_control"], ensure_ascii=False, indent=2))
    print(json.dumps(m["real_scanned_pdf_0041"], ensure_ascii=False, indent=2))

