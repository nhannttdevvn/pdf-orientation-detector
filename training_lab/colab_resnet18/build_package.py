#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ĐÓNG GÓI TRAINING PACKAGE CHO GOOGLE COLAB (quy trình chạy lại được)
Dự án: pdf-orientation-detector

Tạo pdf_orientation_colab_package.zip gồm:
  package/
  ├── images/rotations/{000,090,180,270}/*.jpg   (CHỈ ảnh hợp lệ trong split)
  ├── train_val_test_split.csv                   (có cột rel_path tương đối)
  ├── split_report.json
  ├── class_mapping.json
  ├── configs/training_config.json
  ├── scripts/{dataset.py,model.py,train.py,evaluate.py}
  └── README_COLAB.md

KHÔNG kèm: PDF nguồn, ảnh trùng/bị loại, ảnh upright master, manifest nội bộ.
Đường dẫn trong package là TƯƠNG ĐỐI. KHÔNG tự upload lên Drive.
================================================================================
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training_lab", "dataset_prep"))
import config as C  # noqa: E402  (dùng lại đường dẫn & class mapping của dự án)

BUILD_DIR = os.path.join(HERE, "_package_build")
PKG_NAME = "pdf_orientation_colab_package"
ZIP_PATH = os.path.join(HERE, f"{PKG_NAME}.zip")

DATASET_PREFIX = os.path.relpath(C.DATASET_DIR, C.PROJECT_ROOT)  # data/dataset_review/dataset


def _rel_in_pkg(image_path: str) -> str:
    """image_path (vd data/dataset_review/dataset/rotations/000/x.jpg) -> images/rotations/000/x.jpg"""
    rel = os.path.relpath(image_path, DATASET_PREFIX)   # rotations/000/x.jpg
    return os.path.join("images", rel)


def build(log=print) -> dict:
    if not os.path.exists(C.SPLIT_CSV):
        raise FileNotFoundError(f"Thiếu {C.SPLIT_CSV}. Chạy training/make_split.py trước.")
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    split = pd.read_csv(C.SPLIT_CSV)
    log(f"[pkg] Split có {len(split)} ảnh.")

    # --- Copy ảnh + dựng cột rel_path ---
    rel_paths, copied, missing = [], 0, 0
    for _, r in split.iterrows():
        src = os.path.join(C.PROJECT_ROOT, r["image_path"])
        rel = _rel_in_pkg(r["image_path"])
        dst = os.path.join(BUILD_DIR, rel)
        rel_paths.append(rel)
        if not os.path.exists(src):
            missing += 1; continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    if missing:
        raise RuntimeError(f"{missing} ảnh nguồn không tồn tại — dừng đóng gói.")
    log(f"[pkg] Đã copy {copied} ảnh vào images/.")

    # --- Split CSV bản package (đường dẫn tương đối) ---
    pkg_split = split.copy()
    pkg_split["rel_path"] = rel_paths
    keep = [c for c in ["rel_path", "page_uid", "source_file", "doc_group",
                        "label_angle", "label_idx", "split"] if c in pkg_split.columns]
    pkg_split[keep].to_csv(os.path.join(BUILD_DIR, "train_val_test_split.csv"), index=False)

    # --- Metadata & code ---
    for src, dst in [
        (os.path.join(C.DATA_REVIEW_DIR, "split_report.json"), "split_report.json"),
    ]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(BUILD_DIR, dst))

    # class_mapping.json (từ model.py của package)
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    import model as pkg_model  # noqa: E402
    with open(os.path.join(BUILD_DIR, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(pkg_model.class_mapping(), f, ensure_ascii=False, indent=2)

    # scripts/ & configs/
    os.makedirs(os.path.join(BUILD_DIR, "scripts"))
    for fn in ("dataset.py", "model.py", "train.py", "evaluate.py"):
        shutil.copy2(os.path.join(HERE, "scripts", fn), os.path.join(BUILD_DIR, "scripts", fn))
    os.makedirs(os.path.join(BUILD_DIR, "configs"))
    shutil.copy2(os.path.join(HERE, "configs", "training_config.json"),
                 os.path.join(BUILD_DIR, "configs", "training_config.json"))
    readme = os.path.join(HERE, "README_COLAB.md")
    if os.path.exists(readme):
        shutil.copy2(readme, os.path.join(BUILD_DIR, "README_COLAB.md"))
    nb = os.path.join(HERE, "train_resnet18_colab.ipynb")
    if os.path.exists(nb):
        shutil.copy2(nb, os.path.join(BUILD_DIR, "train_resnet18_colab.ipynb"))

    # --- Thống kê split để kiểm nhanh ---
    stats = {"total_images": len(pkg_split)}
    for sp in ("train", "val", "test"):
        d = pkg_split[pkg_split["split"] == sp]
        stats[sp] = {"images": int(len(d)), "pages": int(d["page_uid"].nunique()),
                     "per_class": {str(a): int((d["label_angle"] == a).sum()) for a in (0, 90, 180, 270)}}
    with open(os.path.join(BUILD_DIR, "package_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # --- ZIP (đường dẫn trong zip có tiền tố PKG_NAME/) ---
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    n = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, _d, files in os.walk(BUILD_DIR):
            for fn in files:
                if fn.startswith("._"):
                    continue
                full = os.path.join(root, fn)
                arc = os.path.join(PKG_NAME, os.path.relpath(full, BUILD_DIR))
                z.write(full, arc); n += 1
    size_mb = os.path.getsize(ZIP_PATH) / 1e6
    log(f"[pkg] ZIP: {os.path.relpath(ZIP_PATH, C.PROJECT_ROOT)} — {n} file, {size_mb:.1f} MB")
    # ignore_errors: trên exfat/macOS có file rác ._* sinh/xóa bất đồng bộ
    shutil.rmtree(BUILD_DIR, ignore_errors=True)  # dọn staging (đã nằm trong zip)
    return {"zip": ZIP_PATH, "files": n, "size_mb": round(size_mb, 1), "stats": stats}


if __name__ == "__main__":
    import pprint
    pprint.pprint(build())
