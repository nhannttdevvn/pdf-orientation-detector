#!/usr/bin/env python3
"""
Trình đánh giá định lượng & quét ngưỡng (Threshold Sweep Evaluator) cho Phase 4 — Part A:
Horizontal Stripe Detection Improvement (`scripts/evaluate_horizontal_stripe_benchmark.py`).

Đo đạc thực nghiệm:
1. Kết quả trên bộ 23 kịch bản Horizontal Stripe Benchmark (15 kịch bản cốt lõi A11 + 8 kịch bản A5).
2. Quét ngưỡng `HORIZONTAL_STRIPE_MIN_PAGE_RATIO` tại [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
   để đo FP, FN, Precision, Recall, FPR, FNR tại từng ngưỡng.
3. Kiểm tra không hồi quy (Zero Regression) trên bộ 12 kịch bản Phase 2 (`stripe_benchmark`).
4. So sánh số lượng Horizontal Stripe False Positive trước và sau trên 3 file scan thực tế trong `data/uploads/`.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ai_engine.core.config import OUTPUTS_DIR, UPLOADS_DIR
from ai_engine.detectors.defects import StripeDetector
from ai_engine.preprocessing.pdf_loader import PDFLoader
from benchmarks.generators.generate_horizontal_stripe_samples import (
    H_GROUND_TRUTH_PATH,
    H_STRIPE_BENCHMARK_DIR,
    generate_horizontal_stripe_benchmark,
)
from benchmarks.generators.generate_stripe_samples import (
    GROUND_TRUTH_PATH as PHASE2_GT_PATH,
    STRIPE_BENCHMARK_DIR as PHASE2_BENCH_DIR,
    generate_all_stripe_samples,
)


def _evaluate_dataset_at_ratio(
    cases: List[Dict[str, Any]],
    base_dir: Path,
    min_page_ratio: float,
    margin_ext: float = 0.03,
) -> Dict[str, Any]:
    det = StripeDetector(
        horizontal_min_page_ratio=min_page_ratio,
        horizontal_margin_extension=margin_ext,
    )
    tp = tn = fp = fn = 0
    times_ms: List[float] = []
    case_outputs: List[Dict[str, Any]] = []

    for c in cases:
        pdf_path = base_dir / c["file_name"]
        with PDFLoader(pdf_path) as loader:
            ctx = loader.build_page_context(1, dpi=160, grayscale=True)
            res = det.detect_page(
                ctx.original_image,
                page_number=1,
                dpi=160,
                clahe_image=ctx.ensure_clahe(),
            )
            ctx.release()

        times_ms.append(res.processing_time_ms)
        exp = bool(c["expected_has_stripe"])
        pred = bool(res.has_stripe)
        if exp and pred:
            tp += 1
        elif not exp and not pred:
            tn += 1
        elif not exp and pred:
            fp += 1
        else:
            fn += 1

        exp_types = set(c.get("expected_types", []))
        pred_types = {s.stripe_type for s in res.stripes}
        types_ok = True
        if c["case_id"] == "15_both_vertical_and_horizontal_stripes":
            types_ok = ("VERTICAL" in pred_types) and ("HORIZONTAL" in pred_types)
        elif exp and exp_types:
            types_ok = bool(exp_types & pred_types)

        case_outputs.append(
            {
                "case_id": c["case_id"],
                "category": c.get("category", "core"),
                "expected_has_stripe": exp,
                "predicted_status": res.status,
                "predicted_has_stripe": pred,
                "predicted_types": sorted(pred_types),
                "types_match": types_ok,
                "stripes_count": len(res.stripes),
                "content_margins": (
                    res.content_margins.to_dict() if res.content_margins else None
                ),
                "reason_codes": [
                    s.reason_codes for s in (res.stripes or res.rejected_candidates[:2])
                ],
                "processing_time_ms": res.processing_time_ms,
                "pass": (exp == pred) and types_ok,
            }
        )

    prec = tp / float(max(1, tp + fp))
    rec = tp / float(max(1, tp + fn))
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = fp / float(max(1, fp + tn))
    fnr = fn / float(max(1, fn + tp))
    avg_ms = sum(times_ms) / float(max(1, len(times_ms)))

    return {
        "horizontal_min_page_ratio": min_page_ratio,
        "horizontal_margin_extension": margin_ext,
        "total_cases": len(cases),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "mean_processing_time_ms": round(avg_ms, 2),
        "cases": case_outputs,
    }


def run_horizontal_evaluation() -> Dict[str, Any]:
    if not H_GROUND_TRUTH_PATH.exists():
        generate_horizontal_stripe_benchmark()
    if not PHASE2_GT_PATH.exists():
        generate_all_stripe_samples()

    h_gt = json.loads(H_GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    p2_gt = json.loads(PHASE2_GT_PATH.read_text(encoding="utf-8"))

    # 1. Quét ngưỡng HORIZONTAL_STRIPE_MIN_PAGE_RATIO theo mục A5
    sweep_thresholds = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    sweep_results: List[Dict[str, Any]] = []
    for ratio in sweep_thresholds:
        ev = _evaluate_dataset_at_ratio(
            cases=h_gt["cases"],
            base_dir=H_STRIPE_BENCHMARK_DIR,
            min_page_ratio=ratio,
            margin_ext=0.03,
        )
        sweep_results.append(
            {
                "threshold_ratio": ratio,
                "tp": ev["tp"],
                "tn": ev["tn"],
                "fp": ev["fp"],
                "fn": ev["fn"],
                "precision": ev["precision"],
                "recall": ev["recall"],
                "f1_score": ev["f1_score"],
                "fpr": ev["false_positive_rate"],
                "fnr": ev["false_negative_rate"],
            }
        )

    # 2. Đánh giá chi tiết ở ngưỡng được chọn (0.90)
    selected_eval = _evaluate_dataset_at_ratio(
        cases=h_gt["cases"],
        base_dir=H_STRIPE_BENCHMARK_DIR,
        min_page_ratio=0.90,
        margin_ext=0.03,
    )

    # 3. Kiểm tra không hồi quy trên bộ 12 case Phase 2 (dọc, nghiêng, band, xoay+sọc...)
    phase2_eval = _evaluate_dataset_at_ratio(
        cases=p2_gt["cases"],
        base_dir=PHASE2_BENCH_DIR,
        min_page_ratio=0.90,
        margin_ext=0.03,
    )

    # 4. Đánh giá trên các file PDF scan thực tế trong data/uploads/
    det = StripeDetector(horizontal_min_page_ratio=0.90, horizontal_margin_extension=0.03)
    real_uploads_after: Dict[str, Any] = {}
    for pdf_path in sorted(UPLOADS_DIR.glob("[!.]*.pdf")):
        h_conf = h_rev = h_doc_rejected = v_conf = 0
        detected_h_pages: List[int] = []
        review_h_pages: List[int] = []
        with PDFLoader(pdf_path) as loader:
            for pno in range(1, loader.page_count + 1):
                ctx = loader.build_page_context(pno, dpi=160, grayscale=True)
                res = det.detect_page(
                    ctx.original_image,
                    page_number=pno,
                    dpi=160,
                    clahe_image=ctx.ensure_clahe(),
                )
                ctx.release()
                for s in res.stripes:
                    if s.horizontal_classification is not None:
                        if s.status == "CONFIRMED":
                            h_conf += 1
                            detected_h_pages.append(pno)
                        else:
                            h_rev += 1
                            review_h_pages.append(pno)
                    else:
                        if s.status == "CONFIRMED":
                            v_conf += 1
                for r in res.rejected_candidates:
                    if r.horizontal_classification == "LIKELY_DOCUMENT_LINE":
                        h_doc_rejected += 1
        real_uploads_after[pdf_path.name] = {
            "horizontal_confirmed": h_conf,
            "horizontal_review": h_rev,
            "horizontal_likely_document_line_filtered": h_doc_rejected,
            "vertical_or_diagonal_confirmed": v_conf,
            "detected_horizontal_pages": sorted(set(detected_h_pages)),
            "review_horizontal_pages": sorted(set(review_h_pages)),
        }

    baseline_before_path = OUTPUTS_DIR / "horizontal_stripe_baseline_before.json"
    baseline_before = (
        json.loads(baseline_before_path.read_text(encoding="utf-8"))
        if baseline_before_path.exists()
        else None
    )

    report = {
        "selected_config": {
            "HORIZONTAL_STRIPE_MIN_PAGE_RATIO": 0.90,
            "HORIZONTAL_STRIPE_MARGIN_EXTENSION": 0.03,
        },
        "before_fix_baseline": (
            {
                "tp": baseline_before["tp"],
                "tn": baseline_before["tn"],
                "fp": baseline_before["fp"],
                "fn": baseline_before["fn"],
                "precision": baseline_before["precision"],
                "recall": baseline_before["recall"],
                "fpr": baseline_before["fpr"],
                "fnr": baseline_before["fnr"],
                "real_uploads_before": baseline_before.get(
                    "real_uploads_horizontal_confirmed_before", {}
                ),
            }
            if baseline_before
            else None
        ),
        "after_fix_horizontal_benchmark": selected_eval,
        "threshold_sweep": sweep_results,
        "phase2_regression_check": {
            "total_cases": phase2_eval["total_cases"],
            "tp": phase2_eval["tp"],
            "tn": phase2_eval["tn"],
            "fp": phase2_eval["fp"],
            "fn": phase2_eval["fn"],
            "precision": phase2_eval["precision"],
            "recall": phase2_eval["recall"],
        },
        "real_uploads_after": real_uploads_after,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUTS_DIR / "horizontal_stripe_evaluation_after.json"
    out_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    rep = run_horizontal_evaluation()
    print("=== THRESHOLD SWEEP ===")
    for row in rep["threshold_sweep"]:
        print(row)
    print("\n=== AFTER FIX (23 HORIZONTAL CASES) ===")
    af = rep["after_fix_horizontal_benchmark"]
    print(
        f"TP={af['tp']}, TN={af['tn']}, FP={af['fp']}, FN={af['fn']} | "
        f"Precision={af['precision']}, Recall={af['recall']}, FPR={af['false_positive_rate']}, FNR={af['false_negative_rate']}, "
        f"AvgTime={af['mean_processing_time_ms']} ms"
    )
    for c in af["cases"]:
        if not c["pass"]:
            print("  FAILED CASE:", c)
    print("\n=== PHASE 2 REGRESSION (12 CASES) ===")
    print(rep["phase2_regression_check"])
    print("\n=== REAL UPLOADS AFTER FIX ===")
    print(json.dumps(rep["real_uploads_after"], ensure_ascii=False, indent=2))

