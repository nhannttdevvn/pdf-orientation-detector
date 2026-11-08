#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit test cho training package (Phase 2). Chạy:
    python3 -m pytest colab_training/tests/test_training.py -v
Bao phủ: class mapping, split leakage/cân bằng, preprocessing, dataset loader,
         model forward 4 lớp, checkpoint save/load, góc sửa, tính xác định của eval.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                     # colab_training/
PROJECT_ROOT = os.path.dirname(PKG)
sys.path.insert(0, os.path.join(PKG, "scripts"))

import dataset as D        # noqa: E402
import model as M          # noqa: E402

SPLIT_CSV = os.path.join(PROJECT_ROOT, "data", "dataset_review", "train_val_test_split.csv")


# ----------------------------- class mapping --------------------------------
def test_class_mapping_fixed():
    assert D.CLASS_NAMES == ["0", "90", "180", "270"]
    assert D.ANGLE_TO_IDX == {0: 0, 90: 1, 180: 2, 270: 3}
    assert D.IDX_TO_ANGLE == {0: 0, 1: 90, 2: 180, 3: 270}
    # góc sửa = (360 - góc bị xoay) % 360
    for ang, corr in D.CORRECTION_ANGLE.items():
        assert corr == (360 - ang) % 360


def test_model_class_mapping_json():
    cm = M.class_mapping()
    assert cm["class_names"] == D.CLASS_NAMES
    assert cm["correction_angle_cw"]["90"] == 270


# ----------------------------- split integrity ------------------------------
@pytest.fixture(scope="module")
def split_df():
    assert os.path.exists(SPLIT_CSV), "Chưa có split CSV — chạy training/make_split.py"
    return pd.read_csv(SPLIT_CSV)


def test_split_no_leakage(split_df):
    for key in ("page_uid", "doc_group", "source_file"):
        counts = split_df.groupby(key)["split"].nunique()
        assert (counts > 1).sum() == 0, f"Leakage theo {key}"


def test_split_four_rotations_same_split(split_df):
    # mỗi page_uid có đúng 4 ảnh và cùng 1 split
    g = split_df.groupby("page_uid")
    assert (g["split"].nunique() == 1).all()
    assert (g.size() == 4).all()


def test_split_class_balance(split_df):
    for sp in ("train", "val", "test"):
        d = split_df[split_df.split == sp]
        counts = d["label_angle"].value_counts().to_dict()
        assert set(counts) == {0, 90, 180, 270}
        assert len(set(counts.values())) == 1, f"Lớp không cân bằng trong {sp}"


def test_split_ratio_approx(split_df):
    n = split_df["page_uid"].nunique()
    tr = split_df[split_df.split == "train"]["page_uid"].nunique()
    assert 0.6 <= tr / n <= 0.8, "Train nên quanh 70%"


# ----------------------------- preprocessing --------------------------------
def test_pad_to_square():
    img = Image.new("RGB", (100, 40), (10, 20, 30))
    out = D.PadToSquare(fill=255)(img)
    assert out.size == (100, 100)
    # nền thêm vào phải là trắng
    assert out.getpixel((0, 0)) == (255, 255, 255)
    # nội dung gốc còn nguyên ở giữa
    assert out.getpixel((50, 50)) == (10, 20, 30)


def test_transforms_eval_deterministic():
    img = Image.new("RGB", (120, 80), (123, 50, 200))
    t = D.build_transforms(224, train=False)
    a, b = t(img), t(img)
    assert torch.allclose(a, b), "Transform eval phải xác định (không random)"
    assert a.shape == (3, 224, 224)


# ----------------------------- dataset loader -------------------------------
def test_dataset_getitem():
    ds = D.OrientationDataset(SPLIT_CSV, PROJECT_ROOT, "test", image_size=224,
                              train=False, path_col="image_path")
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (3, 224, 224)
    assert 0 <= y <= 3


# ----------------------------- model ----------------------------------------
def test_model_forward_four_classes():
    m = M.build_model(num_classes=4, pretrained=False)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)


def test_checkpoint_save_load(tmp_path):
    m = M.build_model(num_classes=4, pretrained=False)
    p = str(tmp_path / "ck.pth")
    M.save_checkpoint(p, m, config={"IMAGE_SIZE": 224, "SEED": 42}, epoch=3, best_metric=0.9)
    m2, ck = M.load_checkpoint(p)
    assert ck["num_classes"] == 4 and ck["arch"] == "resnet18"
    assert ck["epoch"] == 3 and abs(ck["best_metric"] - 0.9) < 1e-9
    # trọng số khớp
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    # metadata preprocessing đủ để nạp đúng
    assert ck["preprocessing"]["image_size"] == 224
    assert ck["class_mapping"]["class_names"] == D.CLASS_NAMES


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
