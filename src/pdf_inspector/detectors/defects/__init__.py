"""
Package `pdf_inspector.detectors.defects` — Chứa các Detector phát hiện lỗi hình ảnh trên trang PDF scan
(Stripe Defect Detection: `StripeDetector`, `LineCandidate`, `ValidatedStripe`, `render_stripe_overlay`).
"""

from pdf_inspector.detectors.defects.line_candidate_detector import (
    LineCandidate,
    PreparedStripeMaps,
    detect_candidates_hough,
    detect_candidates_morphological,
    detect_candidates_projection,
    merge_and_deduplicate_candidates,
    prepare_stripe_maps,
)
from pdf_inspector.detectors.defects.stripe_detector import (
    StripeDetector,
    StripePageResult,
    render_stripe_overlay,
)
from pdf_inspector.detectors.defects.stripe_validator import (
    ContentMargins,
    StripeFeatures,
    ValidatedStripe,
    estimate_content_margins,
    validate_stripe_candidates,
)

__all__ = [
    "ContentMargins",
    "LineCandidate",
    "PreparedStripeMaps",
    "StripeDetector",
    "StripeFeatures",
    "StripePageResult",
    "ValidatedStripe",
    "detect_candidates_hough",
    "detect_candidates_morphological",
    "detect_candidates_projection",
    "estimate_content_margins",
    "merge_and_deduplicate_candidates",
    "prepare_stripe_maps",
    "render_stripe_overlay",
    "validate_stripe_candidates",
]

