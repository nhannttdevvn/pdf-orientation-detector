"""
Pytest fixtures dùng chung cho bộ kiểm thử PDF Orientation Detector.
"""

from typing import Any, Dict

import pytest

from pdf_inspector.core.config import SAMPLES_DIR
from scripts.generate_samples import generate_all_sample_pdfs


@pytest.fixture(scope="session", autouse=True)
def sample_pdfs_fixture() -> Dict[str, Any]:
    """Đảm bảo toàn bộ 12 file PDF mẫu luôn sẵn sàng trong `data/samples/` trước khi chạy pytest."""
    return generate_all_sample_pdfs(SAMPLES_DIR, force=False)
