#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiện ích dùng chung cho pipeline huấn luyện (seed, thiết bị, hash, I/O)."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Cố định seed cho random / numpy / torch để tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def pick_device(prefer: Optional[str] = None):
    """Chọn thiết bị theo thứ tự ưu tiên: CUDA -> MPS (Apple Silicon) -> CPU.
    `prefer` ép buộc ('cuda'/'mps'/'cpu') khi cần smoke test."""
    import torch
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device) -> dict:
    """Thông tin thiết bị để ghi vào báo cáo/checkpoint."""
    import torch
    info = {"type": device.type, "torch": torch.__version__}
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    elif device.type == "mps":
        info["name"] = "Apple Silicon GPU (MPS)"
    else:
        info["name"] = "CPU"
    return info


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dataset_fingerprint(csv_paths) -> str:
    """Vân tay (hash) của dataset = hash gộp các CSV định nghĩa dữ liệu & split.
    Lưu kèm checkpoint để biết model huấn luyện trên phiên bản dữ liệu nào."""
    h = hashlib.sha256()
    for p in sorted(csv_paths):
        if os.path.exists(p):
            h.update(os.path.basename(p).encode())
            h.update(file_sha256(p).encode())
    return h.hexdigest()[:16]


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
