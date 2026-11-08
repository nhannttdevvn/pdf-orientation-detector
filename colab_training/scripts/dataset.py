#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DATASET & PREPROCESSING — ResNet-18 phát hiện hướng trang PDF (Colab package)
Dự án: pdf-orientation-detector

Self-contained: chạy được cả trên máy (CPU smoke test) lẫn Colab (CUDA).
Nhãn = góc ảnh BỊ XOAY THEO CHIỀU KIM ĐỒNG HỒ so với hướng chuẩn (upright).

Preprocessing cho tài liệu scan:
  - Chuyển RGB.
  - Pad về VUÔNG bằng nền trắng (giữ nguyên tỷ lệ, KHÔNG crop mất chữ).
  - Resize về IMAGE_SIZE.
  - Chuẩn hóa theo mean/std ImageNet (khớp pretrained ResNet-18).
  - Train: augmentation NHẸ KHÔNG đổi hướng (brightness/contrast/blur/nhiễu).
    Tuyệt đối KHÔNG xoay/lật ngẫu nhiên (sẽ sai nhãn).
  - Val/Test: KHÔNG augmentation ngẫu nhiên.
================================================================================
"""

from __future__ import annotations

import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# --- Ánh xạ lớp CỐ ĐỊNH (không suy từ tên thư mục) ---
CLASS_NAMES = ["0", "90", "180", "270"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
ANGLE_TO_IDX = {int(n): i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_ANGLE = {i: int(n) for i, n in enumerate(CLASS_NAMES)}
# Góc cần xoay để SỬA trang về đúng chiều = (360 - góc bị xoay) % 360
CORRECTION_ANGLE = {0: 0, 90: 270, 180: 180, 270: 90}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PadToSquare:
    """Pad ảnh về vuông bằng màu nền (trắng cho trang scan), giữ nguyên tỷ lệ."""

    def __init__(self, fill: int = 255):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (self.fill, self.fill, self.fill))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


def build_transforms(image_size: int, train: bool,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Transform pipeline. Train có augmentation nhẹ KHÔNG đổi hướng."""
    base = [PadToSquare(fill=255), T.Resize((image_size, image_size))]
    if train:
        aug = [
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        ]
        base = base + aug
    base += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    return T.Compose(base)


class OrientationDataset(Dataset):
    """Đọc ảnh theo split từ train_val_test_split.csv.

    CSV cần cột: rel_path (đường dẫn ảnh TƯƠNG ĐỐI so với data_root), label_idx,
    split, page_uid. Ảnh đọc lazy từ ĐĨA theo từng item (không nạp hết vào RAM)."""

    def __init__(self, split_csv: str, data_root: str, split: str,
                 image_size: int = 224, train: bool = False,
                 path_col: str = "rel_path"):
        df = pd.read_csv(split_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.data_root = data_root
        self.path_col = path_col
        self.transform = build_transforms(image_size, train=train)
        if len(self.df) == 0:
            raise ValueError(f"Split '{split}' rỗng trong {split_csv}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = os.path.join(self.data_root, str(row[self.path_col]))
        with Image.open(path) as im:
            img = im.convert("RGB")
        x = self.transform(img)
        y = int(row["label_idx"])
        return x, y

    def labels(self):
        return self.df["label_idx"].tolist()

    def page_uids(self):
        return self.df["page_uid"].tolist()


def make_loaders(split_csv: str, data_root: str, image_size: int = 224,
                 batch_size: int = 32, num_workers: int = 2,
                 path_col: str = "rel_path"):
    """Tạo 3 DataLoader. Chỉ Train shuffle; Val/Test giữ nguyên thứ tự."""
    pin = torch.cuda.is_available()
    loaders = {}
    for split, is_train in (("train", True), ("val", False), ("test", False)):
        ds = OrientationDataset(split_csv, data_root, split, image_size,
                                train=is_train, path_col=path_col)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=is_train,
            num_workers=num_workers, pin_memory=pin,
            drop_last=False, persistent_workers=(num_workers > 0),
        )
    return loaders
