"""
Các component giao diện Dashboard PDF Scan Quality Inspector (`frontend.components`).
"""

from frontend.components.footer_bar import render_footer_bar
from frontend.components.header_bar import render_header_bar
from frontend.components.preview_panel import (
    render_preview_and_export_section,
    render_preview_panel,
    save_boundary_review_feedback,
    save_stripe_review_feedback,
)
from frontend.components.results_view import (
    build_boundary_results_dataframe,
    build_results_dataframe,
    build_stripe_results_dataframe,
    render_error_table_panel,
    render_results_section,
)
from frontend.components.secret_tab_view import (
    build_secret_flagged_rows,
    filter_secret_rows,
    render_secret_preview_panel,
    render_secret_results_table,
    render_secret_summary_cards,
    render_secret_upload_card,
    save_secret_review_feedback,
)
from frontend.components.sidebar import render_sidebar
from frontend.components.summary_cards import render_summary_cards
from frontend.components.upload_card import (
    render_progress_html,
    render_upload_and_progress_card,
)

__all__ = [
    "render_header_bar",
    "render_upload_and_progress_card",
    "render_progress_html",
    "render_summary_cards",
    "render_error_table_panel",
    "render_preview_panel",
    "render_footer_bar",
    "render_sidebar",
    "build_results_dataframe",
    "build_stripe_results_dataframe",
    "build_boundary_results_dataframe",
    "render_results_section",
    "render_preview_and_export_section",
    "save_stripe_review_feedback",
    "save_boundary_review_feedback",
    "build_secret_flagged_rows",
    "filter_secret_rows",
    "render_secret_upload_card",
    "render_secret_summary_cards",
    "render_secret_results_table",
    "render_secret_preview_panel",
    "save_secret_review_feedback",
]
