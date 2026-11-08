"""
Package nhận diện dấu độ mật độc lập (`src/pdf_inspector/detectors/secret`).
"""

from pdf_inspector.detectors.secret.secret_detector import (
    DetectedSecretMark,
    SecretMarkingDetector,
    SecretPageResult,
    crop_secret_mark,
    render_secret_overlay,
)

__all__ = [
    "DetectedSecretMark",
    "SecretMarkingDetector",
    "SecretPageResult",
    "crop_secret_mark",
    "render_secret_overlay",
]

