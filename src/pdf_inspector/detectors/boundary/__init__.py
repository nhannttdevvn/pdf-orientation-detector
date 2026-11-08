"""
Module phát hiện lỗi ranh giới trang và lẫn trang (Phase 3 — `src/pdf_inspector/detectors/boundary/`).
"""

from pdf_inspector.detectors.boundary.boundary_candidate_detector import (
    OptionalLayoutModelVerifier,
    detect_and_validate_boundary_candidates,
    detect_boundary_candidates,
)
from pdf_inspector.detectors.boundary.boundary_detector import PageBoundaryDetector
from pdf_inspector.detectors.boundary.cross_page_validator import (
    detect_overlap_between_adjacent_pages,
    detect_split_or_spillover_between_adjacent_pages,
    run_cross_page_validation,
)
from pdf_inspector.detectors.boundary.layout_analyzer import (
    analyze_page_layout,
    find_safe_cut_line_px,
)

__all__ = [
    "PageBoundaryDetector",
    "OptionalLayoutModelVerifier",
    "analyze_page_layout",
    "find_safe_cut_line_px",
    "detect_and_validate_boundary_candidates",
    "detect_boundary_candidates",
    "detect_overlap_between_adjacent_pages",
    "detect_split_or_spillover_between_adjacent_pages",
    "run_cross_page_validation",
]
