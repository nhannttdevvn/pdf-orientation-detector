#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MÔ HÌNH ResNet-18 (4 lớp hướng) + LƯU/NẠP CHECKPOINT (Colab package)
Dự án: pdf-orientation-detector
================================================================================
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torchvision

from dataset import (CLASS_NAMES, CLASS_TO_IDX, IDX_TO_ANGLE, CORRECTION_ANGLE,
                     IMAGENET_MEAN, IMAGENET_STD)


def build_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ResNet-18; thay head FC thành num_classes. Dùng ImageNet weights nếu cho phép."""
    weights = None
    if pretrained:
        try:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
    model = torchvision.models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def class_mapping() -> dict:
    """Ánh xạ lớp cố định, lưu kèm checkpoint để nạp model đúng về sau."""
    return {
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_angle": {str(k): v for k, v in IDX_TO_ANGLE.items()},
        "correction_angle_cw": {str(k): v for k, v in CORRECTION_ANGLE.items()},
        "label_semantics": "Nhãn = góc ảnh BỊ XOAY theo chiều kim đồng hồ so với hướng chuẩn. "
                           "Góc cần xoay để sửa = correction_angle_cw[label].",
    }


def save_checkpoint(path: str, model: nn.Module, *, config: dict, epoch: int,
                    best_metric: float, optimizer=None, scheduler=None,
                    history=None, extra=None) -> None:
    """Lưu checkpoint đầy đủ metadata để tái lập & nạp đúng."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "arch": "resnet18",
        "num_classes": len(CLASS_NAMES),
        "class_mapping": class_mapping(),
        "preprocessing": {
            "image_size": config.get("IMAGE_SIZE"),
            "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "pad_to_square_fill": 255,
        },
        "config": config,
        "epoch": epoch,
        "best_metric": best_metric,
        "history": history or [],
        "torch_version": torch.__version__,
        "seed": config.get("SEED"),
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, map_location="cpu", pretrained_backbone: bool = False):
    """Nạp checkpoint -> (model đã load weights, ckpt dict)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(num_classes=ckpt.get("num_classes", 4), pretrained=pretrained_backbone)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def write_sidecar_json(models_dir: str, config: dict) -> None:
    """Ghi class_mapping.json & model_config.json cạnh checkpoint."""
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(class_mapping(), f, ensure_ascii=False, indent=2)
    with open(os.path.join(models_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump({"arch": "resnet18", "num_classes": len(CLASS_NAMES),
                   "preprocessing": {"image_size": config.get("IMAGE_SIZE"),
                                     "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                                     "pad_to_square_fill": 255},
                   "config": config}, f, ensure_ascii=False, indent=2)
