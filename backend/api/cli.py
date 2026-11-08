#!/usr/bin/env python3
"""
Entry point dòng lệnh (CLI) của hệ thống PDF Quality Inspection (`src/pdf_inspector/cli.py`).
Tái sử dụng chung `DocumentPipeline` (`ai_engine.pipelines.document_pipeline`) với giao diện Streamlit.

Ví dụ sử dụng:
  python -m backend.api.cli data/samples/06_all_four_orientations.pdf --report data/outputs/report.json
  python -m backend.api.cli input.pdf --auto-fix --output data/outputs/corrected.pdf
  python -m backend.api.cli input.pdf --dpi 160 --threshold 10 --auto-fix --output data/outputs/corrected.pdf
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ai_engine.core.config import (
    AUTO_FIX,
    ORIENTATION_CONF_THRESHOLD,
    OUTPUTS_DIR,
    RENDER_DPI,
    DetectorConfig,
)
from ai_engine.core.exceptions import (
    InvalidPDFError,
    PDFCorrectionError,
    PDFLoaderError,
    PasswordProtectedPDFError,
)
from ai_engine.core.logging_config import setup_logger
from backend.schemas.schemas import PageDecision
from ai_engine.pipelines.document_pipeline import DocumentPipeline

logger = setup_logger("cli")


def build_arg_parser() -> argparse.ArgumentParser:
    """Khởi tạo bộ phân tích tham số dòng lệnh (argparse)."""
    parser = argparse.ArgumentParser(
        prog="pdf-orientation-detector",
        description=(
            "PDF Quality Inspection — Công cụ kiểm tra chất lượng tài liệu PDF scan "
            "(Detector đang hoạt động: Orientation 0°/90°/180°/270°, Stripe Defect Detection, "
            "Page Boundary & Mixed-Page Detection)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_pdf",
        type=str,
        help="Đường dẫn tới file PDF đầu vào cần kiểm tra chất lượng.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=RENDER_DPI,
        help="Độ phân giải (DPI) khi render từng trang PDF phục vụ kiểm tra.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=ORIENTATION_CONF_THRESHOLD,
        help="Ngưỡng độ tin cậy (orientation_conf) tối thiểu để phân loại NORMAL/ROTATED.",
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        default=AUTO_FIX,
        help="Tự động xoay các trang được phân loại ROTATED (không xoay trang REVIEW).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn lưu file PDF sau chỉnh sửa (mặc định: data/outputs/corrected.pdf khi bật --auto-fix).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Đường dẫn lưu báo cáo JSON (mặc định: data/outputs/orientation_report.json).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Đường dẫn lưu báo cáo CSV (mặc định: data/outputs/orientation_report.csv).",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        default=False,
        help="Tắt cơ chế thử lại OCR bằng ảnh đã tiền xử lý.",
    )
    return parser


def run_cli(argv: Optional[List[str]] = None) -> int:
    """
    Hàm thực thi chính của CLI. Trả về mã thoát (0 nếu thành công, != 0 nếu lỗi).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.dpi <= 0:
        print("LỖI: Giá trị --dpi phải lớn hơn 0.", file=sys.stderr)
        return 2

    if args.threshold < 0:
        print("LỖI: Giá trị --threshold không được âm.", file=sys.stderr)
        return 2

    config = DetectorConfig(
        render_dpi=args.dpi,
        orientation_conf_threshold=args.threshold,
        auto_fix=args.auto_fix,
        retry_with_preprocessed=not args.no_retry,
    )

    pipeline = DocumentPipeline(config=config)

    def _on_progress(
        current_page: int, total_pages: int, decision: PageDecision
    ) -> None:
        rot_str = (
            f"{decision.detected_rotation}°"
            if decision.detected_rotation is not None
            else "Unknown"
        )
        conf_str = (
            f"{decision.orientation_conf:.2f}"
            if decision.orientation_conf is not None
            else "N/A"
        )
        print(
            f"  [{current_page}/{total_pages}] Trang {current_page}: "
            f"orientation={decision.status:<7} | stripe={decision.stripe_status:<15} | "
            f"inspection={decision.inspection_status:<21} | "
            f"rot={rot_str:<7} | conf={conf_str:<6}"
        )

    try:
        print(
            f"Đang kiểm tra file PDF: {args.input_pdf} "
            f"(DPI={config.render_dpi}, Threshold={config.orientation_conf_threshold})..."
        )
        result = pipeline.scan_document(
            pdf_path=args.input_pdf,
            progress_callback=_on_progress,
        )

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_pdf_path: Optional[Path] = None
        if args.output:
            out_pdf_path = Path(args.output)
        elif args.auto_fix:
            out_pdf_path = OUTPUTS_DIR / "corrected.pdf"

        json_report_path = (
            Path(args.report)
            if args.report
            else (OUTPUTS_DIR / "orientation_report.json")
        )
        if args.csv:
            csv_report_path = Path(args.csv)
        elif args.report:
            csv_report_path = Path(args.report).with_suffix(".csv")
        else:
            csv_report_path = OUTPUTS_DIR / "orientation_report.csv"

        result = pipeline.update_decisions_and_artifacts(
            result=result,
            output_pdf_path=out_pdf_path,
            json_report_path=json_report_path,
            csv_report_path=csv_report_path,
            write_corrected_pdf=bool(out_pdf_path is not None),
        )

        for warning_msg in result.warnings:
            print(f"\n[!] {warning_msg}")

        summary = result.report_data["summary"]
        print("\n" + "=" * 70)
        print("KẾT QUẢ TỔNG HỢP (PDF QUALITY INSPECTION)")
        print("=" * 70)
        print(f"  File đầu vào             : {summary['input_file']}")
        print(f"  Tổng số trang            : {summary['total_pages']}")
        print(
            f"  Detector đang hoạt động  : {', '.join(summary['enabled_detectors'])}"
        )
        print(
            f"  Kiểm tra chưa triển khai : {', '.join(summary['unsupported_checks'])} (NOT_IMPLEMENTED)"
        )
        print(f"  Orientation - NORMAL     : {summary['normal_pages']}")
        print(f"  Orientation - ROTATED    : {summary['rotated_pages']}")
        print(f"  Orientation - REVIEW     : {summary['review_pages']}")
        print(f"  Trang đã xoay sửa        : {summary['corrected_pages']}")
        print(f"  Stripe - Pages w/ Stripes: {summary.get('pages_with_stripes', 0)}")
        print(f"  Stripe - Confirmed Pages : {summary.get('confirmed_stripe_pages', 0)}")
        print(f"  Stripe - Review Pages    : {summary.get('stripe_review_pages', 0)}")
        print(f"  Stripe - Error Pages     : {summary.get('stripe_error_pages', 0)}")
        print(f"  Boundary - Flagged Pages : {summary.get('pages_with_boundary_defects', 0)}")
        print(f"  Boundary - MIXED_PAGE    : {summary.get('mixed_page_count', 0)}")
        print(f"  Boundary - SPLIT_PAGE    : {summary.get('split_page_count', 0)}")
        print(f"  Boundary - OVERLAP       : {summary.get('content_overlap_count', 0)}")
        print(f"  Boundary - TRUNCATED     : {summary.get('content_truncated_count', 0)}")
        print(
            f"  Thời gian xử lý          : {summary['processing_time_seconds']:.3f} giây"
        )
        if result.output_pdf_path:
            print(f"  File PDF đã chỉnh sửa    : {result.output_pdf_path}")
        if result.json_report_path:
            print(f"  Báo cáo JSON             : {result.json_report_path}")
        if result.csv_report_path:
            print(f"  Báo cáo CSV              : {result.csv_report_path}")
        print("=" * 70)
        return 0

    except PasswordProtectedPDFError as exc:
        print(f"LỖI BẢO MẬT PDF: {exc}", file=sys.stderr)
        return 1
    except InvalidPDFError as exc:
        print(f"LỖI FILE PDF KHÔNG HỢP LỆ: {exc}", file=sys.stderr)
        return 1
    except PDFCorrectionError as exc:
        print(f"LỖI CHỈNH SỬA PDF: {exc}", file=sys.stderr)
        return 1
    except PDFLoaderError as exc:
        print(f"LỖI ĐỌC PDF: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Lỗi không xác định khi chạy CLI: %s", exc)
        print(f"LỖI HỆ THỐNG: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())
