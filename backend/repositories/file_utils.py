"""
Các hàm tiện ích dùng chung về chuẩn hóa góc xoay, đường dẫn file và lưu trữ (`src/pdf_inspector/utils/file_utils.py`).
"""

import hashlib
from pathlib import Path
from typing import Union

from ai_engine.core.config import UPLOADS_DIR, VALID_ANGLES


def normalize_angle(angle: Union[int, float]) -> int:
    """
    Chuẩn hóa góc xoay về một trong bốn giá trị hợp lệ: 0, 90, 180, 270.
    Ném ValueError nếu góc sau modulo 360 không thuộc {0, 90, 180, 270}.
    """
    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        raise ValueError(f"Góc xoay không hợp lệ: {angle!r}")

    normalized = int(round(angle)) % 360
    if normalized not in VALID_ANGLES:
        raise ValueError(
            f"Góc xoay {angle} (chuẩn hóa: {normalized}) không thuộc {VALID_ANGLES}."
        )
    return normalized


def combine_rotation_angles(existing_angle: int, correction_angle: int) -> int:
    """
    Cộng góc sửa với góc xoay hiện tại trong metadata PDF theo modulo 360.
    """
    norm_existing = normalize_angle(existing_angle)
    norm_correction = normalize_angle(correction_angle)
    return (norm_existing + norm_correction) % 360


def is_same_file_path(path_a: Union[str, Path], path_b: Union[str, Path]) -> bool:
    """
    Kiểm tra hai đường dẫn có trỏ tới cùng một file hay không để chống ghi đè file gốc.
    """
    p1 = Path(path_a).expanduser().resolve()
    p2 = Path(path_b).expanduser().resolve()
    return p1 == p2


def format_file_size(num_bytes: int) -> str:
    """Định dạng dung lượng file dễ đọc (B, KB, MB, GB)."""
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.2f} GB"


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Tính mã băm SHA-256 của file để kiểm chứng tính toàn vẹn."""
    path = Path(file_path).expanduser().resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_uploaded_pdf(
    file_bytes: bytes,
    file_name: str,
    file_hash: str,
    uploads_dir: Path = UPLOADS_DIR,
) -> Path:
    """
    Lưu bản sao file PDF người dùng upload vào `data/uploads/` để xử lý an toàn.
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file_name).name
    target_path = uploads_dir / f"{file_hash[:12]}_{safe_name}"
    if not target_path.exists():
        target_path.write_bytes(file_bytes)
    return target_path

