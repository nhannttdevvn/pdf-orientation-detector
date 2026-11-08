#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CẤU HÌNH HUẤN LUYỆN RESNET-18 — Phát hiện hướng trang PDF (Phase 2)
Dự án: pdf-orientation-detector

Module độc lập với Web Tool chính và các detector hiện có.
Nhãn = góc mà ảnh BỊ XOAY THEO CHIỀU KIM ĐỒNG HỒ so với hướng chuẩn (upright).
================================================================================
"""

from __future__ import annotations

import os

# -----------------------------------------------------------------------------
# ĐƯỜNG DẪN
# -----------------------------------------------------------------------------
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TRAIN_DIR)

DATA_REVIEW_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_review")
DATASET_DIR = os.path.join(DATA_REVIEW_DIR, "dataset")
ROTATIONS_DIR = os.path.join(DATASET_DIR, "rotations")
UPRIGHT_DIR = os.path.join(DATASET_DIR, "upright")
LABELS_CSV = os.path.join(DATASET_DIR, "rotation_labels.csv")
INDEX_CSV = os.path.join(DATASET_DIR, "dataset_index.csv")
MANIFEST_CSV = os.path.join(DATA_REVIEW_DIR, "source_orientation_manifest.csv")

SPLIT_CSV = os.path.join(DATA_REVIEW_DIR, "train_val_test_split.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "orientation")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "orientation_training")
AUDIT_JSON = os.path.join(REPORTS_DIR, "dataset_audit.json")
AUDIT_MD = os.path.join(REPORTS_DIR, "dataset_audit.md")

BEST_CKPT = os.path.join(MODELS_DIR, "best_resnet18.pth")
MODEL_CONFIG_JSON = os.path.join(MODELS_DIR, "model_config.json")
CLASS_MAPPING_JSON = os.path.join(MODELS_DIR, "class_mapping.json")

# -----------------------------------------------------------------------------
# NHÃN LỚP — THỨ TỰ CỐ ĐỊNH, KHÔNG suy từ tên thư mục
# -----------------------------------------------------------------------------
CLASS_NAMES = ["0", "90", "180", "270"]           # index 0..3
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_ANGLE = {i: int(name) for i, name in enumerate(CLASS_NAMES)}
ANGLE_TO_IDX = {int(name): i for i, name in enumerate(CLASS_NAMES)}

# Góc CẦN XOAY để sửa trang về đúng chiều = (360 - góc bị xoay) % 360.
CORRECTION_ANGLE = {0: 0, 90: 270, 180: 180, 270: 90}

# -----------------------------------------------------------------------------
# SIÊU THAM SỐ (giá trị khởi tạo — điều chỉnh theo GPU & validation)
# -----------------------------------------------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

EARLY_STOP_PATIENCE = 5       # số epoch không cải thiện val trước khi dừng
SCHEDULER_PATIENCE = 2        # ReduceLROnPlateau
NUM_WORKERS = 2

# Tỷ lệ chia theo NHÓM TÀI LIỆU NGUỒN (không theo ảnh)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Chuẩn hóa theo ImageNet (dùng khi khởi tạo từ pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Nền padding khi đưa ảnh về vuông (trang scan nền trắng)
PAD_FILL = 255
