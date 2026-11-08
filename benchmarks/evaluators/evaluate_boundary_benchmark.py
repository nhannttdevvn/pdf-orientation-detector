#!/usr/bin/env python3
"""
Trình đánh giá định lượng (Benchmark Evaluator) cho Phase 3 — Page Boundary & Mixed-Page Detection
(`scripts/evaluate_boundary_benchmark.py`).

Đo đạc thực tế trên tập dữ liệu 15 kịch bản có nhãn chuẩn (`data/samples/boundary_benchmark/boundary_ground_truth.json`):
- Page-level Precision, Recall, F1-score, Accuracy
- False Positive Rate (FPR)
- Boundary Localization Error (sai số chuẩn hóa `|y_pred - y_gt|` và theo pixel)
- Cross-page Matching Accuracy (độ chính xác xác định `related_pages`)
- Processing Time trung bình mỗi trang (ms/page)
- Peak RAM Usage (MB) qua `tracemalloc`
- Kiểm tra hồi quy (Regression) trên 12 PDF mẫu Orientation (`data/samples/`) và file scan thực tế 38 trang `0041.pdf`.
"""

import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ai_engine.core.config import OUTPUTS_DIR, SAMPLES_DIR, UPLOADS_DIR
from ai_engine.pipelines.document_pipeline import DocumentPipeline
from benchmarks.generators.generate_boundary_samples import (
    BOUNDARY_BENCHMARK_DIR,
    GROUND_TRUTH_PATH,
    PAGE_H_PX,
    generate_all_boundary_samples,
)


def run_boundary_benchmark() -> Dict[str, Any]:
    if not GROUND_TRUTH_PATH.exists():
        generate_all_boundary_samples()

    gt_data = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    pipeline = DocumentPipeline()

    tracemalloc.start()
    t_bench_start = time.perf_counter()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    status_match_count = 0
    defect_type_match_count = 0
    positive_evaluated = 0
    cross_page_total = 0
    cross_page_correct = 0
    localization_errors_norm: List[float] = []
    localization_errors_px: List[float] = []
    page_times_ms: List[float] = []
    case_records: List[Dict[str, Any]] = []

    for case in gt_data["cases"]:
        pdf_path = BOUNDARY_BENCHMARK_DIR / case["file_name"]
        t0 = time.perf_counter()
        res = pipeline.scan_document(pdf_path=pdf_path)
        elapsed_case_ms = (time.perf_counter() - t0) * 1000.0
        per_page_ms = elapsed_case_ms / max(1, len(res.decisions))

        page_evals: List[Dict[str, Any]] = []
        case_passed = True

        for page_gt in case["pages"]:
            pno = int(page_gt["page_number"])
            dec = res.decisions[pno - 1]
            page_times_ms.append(per_page_ms)

            exp_has_issue = bool(page_gt["expected_has_boundary_issue"])
            exp_statuses: List[str] = page_gt["expected_boundary_status"]
            exp_types: List[str] = page_gt["expected_defect_types"]
            gt_bnd_norm: Optional[float] = page_gt.get("ground_truth_boundary_norm")
            exp_related: List[int] = page_gt.get("expected_related_pages", [])
            exp_ori_status = page_gt.get("expected_orientation_status")
            exp_stripe_status = page_gt.get("expected_stripe_status")
            is_optional_companion = bool(page_gt.get("optional_companion_page", False))

            pred_has_issue = dec.boundary_status != "NORMAL"
            pred_types = [b.defect_type for b in dec.boundary_defects]
            pred_related: List[int] = []
            pred_bnd_norm: Optional[float] = None

            for b in dec.boundary_defects:
                b_det = b.details or {}
                for rp in list(b.related_pages or []) + list(b_det.get("related_pages", [])):
                    if int(rp) not in pred_related:
                        pred_related.append(int(rp))
                if pred_bnd_norm is None:
                    if b_det.get("suspected_boundary_norm") is not None:
                        pred_bnd_norm = float(b_det["suspected_boundary_norm"])
                    elif b.boundary_candidates:
                        pred_bnd_norm = float(b.boundary_candidates[0].get("position", 0.5))

            if is_optional_companion:
                status_ok = dec.boundary_status in exp_statuses
                if status_ok:
                    status_match_count += 1
            else:
                if exp_has_issue and pred_has_issue:
                    tp += 1
                elif not exp_has_issue and not pred_has_issue:
                    tn += 1
                elif not exp_has_issue and pred_has_issue:
                    fp += 1
                else:
                    fn += 1

                status_ok = dec.boundary_status in exp_statuses
                if status_ok:
                    status_match_count += 1
                else:
                    case_passed = False

            type_ok = True
            if exp_types and not is_optional_companion:
                positive_evaluated += 1
                type_ok = any(t in pred_types for t in exp_types)
                if type_ok:
                    defect_type_match_count += 1
                else:
                    case_passed = False

            rel_ok = True
            if exp_related and not is_optional_companion:
                cross_page_total += 1
                rel_ok = all(r in pred_related for r in exp_related)
                if rel_ok:
                    cross_page_correct += 1
                else:
                    case_passed = False

            loc_err_norm: Optional[float] = None
            loc_err_px: Optional[float] = None
            if gt_bnd_norm is not None and pred_bnd_norm is not None:
                loc_err_norm = round(abs(pred_bnd_norm - gt_bnd_norm), 4)
                loc_err_px = round(loc_err_norm * PAGE_H_PX, 1)
                localization_errors_norm.append(loc_err_norm)
                localization_errors_px.append(loc_err_px)

            ori_ok = (exp_ori_status is None) or (dec.status == exp_ori_status)
            stripe_ok = (exp_stripe_status is None) or (
                dec.stripe_status == exp_stripe_status
            )
            if not (ori_ok and stripe_ok):
                case_passed = False

            page_evals.append(
                {
                    "page_number": pno,
                    "expected_boundary_status": exp_statuses,
                    "predicted_boundary_status": dec.boundary_status,
                    "expected_defect_types": exp_types,
                    "predicted_defect_types": pred_types,
                    "expected_related_pages": exp_related,
                    "predicted_related_pages": pred_related,
                    "ground_truth_boundary_norm": gt_bnd_norm,
                    "predicted_boundary_norm": pred_bnd_norm,
                    "localization_error_norm": loc_err_norm,
                    "localization_error_px": loc_err_px,
                    "orientation_status": dec.status,
                    "stripe_status": dec.stripe_status,
                    "status_ok": status_ok,
                    "type_ok": type_ok,
                    "related_ok": rel_ok,
                    "ori_ok": ori_ok,
                    "stripe_ok": stripe_ok,
                }
            )

        case_records.append(
            {
                "case_id": case["case_id"],
                "file_name": case["file_name"],
                "page_count": case["page_count"],
                "case_passed": case_passed,
                "elapsed_ms": round(elapsed_case_ms, 2),
                "pages": page_evals,
            }
        )

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = round(peak_mem / (1024.0 * 1024.0), 2)
    total_bench_sec = round(time.perf_counter() - t_bench_start, 2)

    total_pages_eval = tp + tn + fp + fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    accuracy = (tp + tn) / max(1, total_pages_eval)
    fpr = fp / max(1, fp + tn)
    cross_page_acc = cross_page_correct / max(1, cross_page_total)

    # Kiểm tra False Positive trên 12 PDF mẫu Orientation (`data/samples/*.pdf`)
    orientation_sample_files = sorted(
        p for p in SAMPLES_DIR.glob("*.pdf") if not p.name.startswith(".")
    )
    ori_clean_pages = 0
    ori_flagged_pages = 0
    ori_total_pages = 0
    valid_ori_files = 0
    for sp in orientation_sample_files:
        try:
            sres = pipeline.scan_document(pdf_path=sp)
        except Exception:
            continue
        valid_ori_files += 1
        for d in sres.decisions:
            ori_total_pages += 1
            if d.boundary_status == "NORMAL":
                ori_clean_pages += 1
            else:
                ori_flagged_pages += 1

    # Kiểm tra trên file scan thực tế 38 trang (`6b63db7ab409_0041.pdf`)
    real_scan_summary: Optional[Dict[str, Any]] = None
    real_candidates = sorted(
        p for p in UPLOADS_DIR.glob("*0041.pdf") if not p.name.startswith(".")
    )
    if real_candidates:
        real_pdf = real_candidates[0]
        t_real = time.perf_counter()
        real_res = pipeline.scan_document(pdf_path=real_pdf)
        real_elapsed = time.perf_counter() - t_real
        real_summary = real_res.report_data["summary"]
        flagged_list = [
            {
                "page": d.page_number,
                "boundary_status": d.boundary_status,
                "defects": [b.defect_type for b in d.boundary_defects],
                "score": (
                    d.boundary_defects[0].suspicion_score
                    if d.boundary_defects
                    else 0.0
                ),
            }
            for d in real_res.decisions
            if d.boundary_status != "NORMAL"
        ]
        real_scan_summary = {
            "file_name": real_pdf.name,
            "total_pages": real_res.doc_info.page_count,
            "elapsed_seconds": round(real_elapsed, 2),
            "normal_boundary_pages": real_summary.get(
                "boundary_normal_pages",
                real_summary.get("normal_boundary_pages", 0),
            ),
            "pages_with_boundary_issues": real_summary.get(
                "pages_with_boundary_issues", 0
            ),
            "flagged_pages_detail": flagged_list,
        }

    report = {
        "benchmark_dataset": str(GROUND_TRUTH_PATH),
        "total_cases": len(case_records),
        "passed_cases": sum(1 for c in case_records if c["case_passed"]),
        "total_evaluated_pages": total_pages_eval,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "false_positive_rate": round(fpr, 4),
            "defect_type_accuracy": round(
                defect_type_match_count / max(1, positive_evaluated), 4
            ),
            "cross_page_matching_accuracy": round(cross_page_acc, 4),
            "mean_localization_error_norm": (
                round(
                    sum(localization_errors_norm) / len(localization_errors_norm), 4
                )
                if localization_errors_norm
                else 0.0
            ),
            "mean_localization_error_px": (
                round(sum(localization_errors_px) / len(localization_errors_px), 2)
                if localization_errors_px
                else 0.0
            ),
            "avg_processing_time_ms_per_page": round(
                sum(page_times_ms) / max(1, len(page_times_ms)), 2
            ),
            "peak_ram_usage_mb": peak_ram_mb,
            "total_benchmark_time_sec": total_bench_sec,
        },
        "orientation_samples_negative_control": {
            "total_files": valid_ori_files,
            "total_pages": ori_total_pages,
            "normal_boundary_pages": ori_clean_pages,
            "flagged_pages": ori_flagged_pages,
            "false_positive_rate_on_orientation_samples": round(
                ori_flagged_pages / max(1, ori_total_pages), 4
            ),
        },
        "real_scan_0041_evaluation": real_scan_summary,
        "cases": case_records,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "boundary_benchmark_metrics.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    rep = run_boundary_benchmark()
    print(
        json.dumps(
            {
                "passed_cases": f"{rep['passed_cases']}/{rep['total_cases']}",
                "confusion_matrix": rep["confusion_matrix"],
                "metrics": rep["metrics"],
                "orientation_samples_negative_control": rep[
                    "orientation_samples_negative_control"
                ],
                "real_scan_0041_evaluation": rep["real_scan_0041_evaluation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
