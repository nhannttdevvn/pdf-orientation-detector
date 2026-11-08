"""Page điều phối Tab 1 — Kiểm tra & sửa lỗi PDF (`frontend.pages.scan_quality_page`)."""

from frontend.components.preview_panel import render_preview_panel
from frontend.components.results_view import ProblemPageRow, build_problem_page_rows, render_error_table_card
from frontend.components.summary_cards import compute_dashboard_summary, render_summary_cards

__all__ = [
    "ProblemPageRow",
    "build_problem_page_rows",
    "compute_dashboard_summary",
    "render_summary_cards",
    "render_error_table_card",
    "render_preview_panel",
]
