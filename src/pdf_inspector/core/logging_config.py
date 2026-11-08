"""
Cấu hình logging tập trung cho toàn bộ ứng dụng (`src/pdf_inspector/core/logging_config.py`).
"""

import logging
import os
from pathlib import Path
import sys
from typing import Optional, Union

from pdf_inspector.core.config import LOGS_DIR


def setup_logger(
    name: str = "pdf_orientation_detector",
    log_file: Optional[Union[str, Path]] = None,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Khởi tạo và cấu hình logger ghi ra cả console và file log.
    Đảm bảo không gắn trùng handler khi gọi nhiều lần (ví dụ khi Streamlit rerun).
    """
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        target_log_file = Path(log_file) if log_file else (LOGS_DIR / "app.log")
        target_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target_log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Không thể tạo file log: %s", exc)

    logger.propagate = False
    return logger

