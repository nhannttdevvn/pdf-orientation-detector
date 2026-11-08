#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinh notebook Colab `train_resnet18_colab.ipynb` (nbformat 4.5, valid JSON).
Chạy lại được; không viết tay JSON thủ công.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "train_resnet18_colab.ipynb")

_id = [0]


def _nid():
    _id[0] += 1
    return f"cell{_id[0]:02d}"


def md(text):
    return {"cell_type": "markdown", "id": _nid(), "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": _nid(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = []
CELLS.append(md("""
# 🧭 Huấn luyện ResNet-18 — Phát hiện hướng trang PDF (Google Colab, CUDA)

Dự án **pdf-orientation-detector**. 4 lớp: **0° / 90° / 180° / 270°**
(góc ảnh bị xoay theo chiều kim đồng hồ so với hướng chuẩn).

> ⚠️ **BẢO MẬT:** Dữ liệu là ảnh scan tài liệu thật, có thể gồm trang đóng dấu
> "độ mật". Colab/Drive là hạ tầng công cộng — chỉ chạy nếu bạn **được phép** xử lý
> dữ liệu này ở đây. Xem `README_COLAB.md`.

Chạy các cell lần lượt từ trên xuống. Chỉ cần sửa **một dòng** `PACKAGE_PATH` ở Cell 2.
"""))

CELLS.append(md("## Cell 1 — Kiểm tra môi trường & CUDA"))
CELLS.append(code("""
import sys, subprocess, torch, torchvision
print("Python     :", sys.version.split()[0])
print("PyTorch    :", torch.__version__)
print("torchvision:", torchvision.__version__)
CUDA = torch.cuda.is_available()
print("CUDA avail :", CUDA)
if CUDA:
    print("CUDA ver   :", torch.version.cuda)
    print("GPU        :", torch.cuda.get_device_name(0))
    print("VRAM (GB)  :", round(torch.cuda.get_device_properties(0).total_memory/1e9, 2))
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
else:
    print("\\n❌ CHƯA BẬT GPU. Vào: Runtime → Change runtime type → Hardware accelerator → GPU → Save.")
    print("   Rồi chạy lại Cell 1. KHÔNG train bằng CPU trên Colab (rất chậm).")
"""))

CELLS.append(md("## Cell 2 — Mount Google Drive & chọn file ZIP"))
CELLS.append(code("""
from google.colab import drive
drive.mount('/content/drive')

# >>> SỬA DÒNG NÀY cho đúng vị trí file ZIP của bạn trên Drive <<<
PACKAGE_PATH = "/content/drive/MyDrive/pdf_orientation/pdf_orientation_colab_package.zip"

import os
assert os.path.exists(PACKAGE_PATH), f"Không thấy ZIP tại: {PACKAGE_PATH}. Sửa lại PACKAGE_PATH."
print("ZIP:", PACKAGE_PATH, "| size (MB):", round(os.path.getsize(PACKAGE_PATH)/1e6, 1))

# Nơi lưu kết quả trên Drive (mỗi lần chạy tạo run_XXX riêng, không ghi đè)
DRIVE_RESULTS = "/content/drive/MyDrive/PDF_Orientation_Training/runs"
os.makedirs(DRIVE_RESULTS, exist_ok=True)
"""))

CELLS.append(md("## Cell 3 — Giải nén vào ổ cục bộ `/content` (không đọc ảnh trực tiếp từ Drive)"))
CELLS.append(code("""
import zipfile, os, shutil

EXTRACT_DIR = "/content/pdf_orientation"
free_gb = shutil.disk_usage("/content").free / 1e9
print("Dung lượng trống /content (GB):", round(free_gb, 1))
assert free_gb > 3, "Ổ /content còn quá ít dung lượng."

# Giải nén AN TOÀN (chặn path traversal)
if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR)
with zipfile.ZipFile(PACKAGE_PATH) as z:
    for m in z.namelist():
        dest = os.path.realpath(os.path.join(EXTRACT_DIR, m))
        assert dest.startswith(os.path.realpath(EXTRACT_DIR)), f"Đường dẫn không an toàn: {m}"
    z.extractall(EXTRACT_DIR)

# Tìm thư mục gốc package
roots = [d for d in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, d))]
PKG = os.path.join(EXTRACT_DIR, "pdf_orientation_colab_package")
if not os.path.isdir(PKG) and roots:
    PKG = os.path.join(EXTRACT_DIR, roots[0])
print("Package:", PKG)

SPLIT_CSV = os.path.join(PKG, "train_val_test_split.csv")
SCRIPTS   = os.path.join(PKG, "scripts")
n_imgs = sum(len(f) for _,_,f in os.walk(os.path.join(PKG, "images")))
print("Số ảnh sau giải nén:", n_imgs)
assert os.path.exists(SPLIT_CSV) and os.path.isdir(SCRIPTS), "Package thiếu file."
"""))

CELLS.append(md("## Cell 4 — Kiểm tra dataset, split & leakage"))
CELLS.append(code("""
import pandas as pd, json
df = pd.read_csv(SPLIT_CSV)
print("Tổng ảnh:", len(df), "| trang:", df['page_uid'].nunique(), "| nhóm tài liệu:", df['doc_group'].nunique())
for sp in ["train","val","test"]:
    d = df[df.split==sp]
    pc = {a:int((d.label_angle==a).sum()) for a in [0,90,180,270]}
    print(f"  {sp:5s}: ảnh={len(d):4d} trang={d.page_uid.nunique():3d} nhóm={d.doc_group.nunique():2d} per_class={pc}")

# Kiểm tra leakage: không page_uid / nhóm / PDF nào ở >1 split
leak = {k: int((df.groupby(k)['split'].nunique()>1).sum()) for k in ['page_uid','doc_group','source_file']}
print("Leakage (phải = 0):", leak)
assert sum(leak.values())==0, "CÓ leakage giữa các split!"
rp = os.path.join(PKG, "split_report.json")
if os.path.exists(rp): print("split_report:", json.load(open(rp)).get("leakage"))
print("✅ Dataset & split hợp lệ.")
"""))

CELLS.append(md("## Cell 5 — Cấu hình huấn luyện"))
CELLS.append(code("""
import json, os
cfg = json.load(open(os.path.join(PKG, "configs", "training_config.json")))
IMAGE_SIZE  = cfg["IMAGE_SIZE"]      # 224
BATCH_SIZE  = cfg["BATCH_SIZE"]      # 32 — giảm còn 16 nếu OOM
EPOCHS      = cfg["EPOCHS"]          # 20
LEARNING_RATE = cfg["LEARNING_RATE"]
SEED        = cfg["SEED"]
NUM_WORKERS = cfg["NUM_WORKERS"]
PATIENCE    = cfg["EARLY_STOP_PATIENCE"]
SCHED_PATIENCE = cfg["SCHEDULER_PATIENCE"]
RESUME      = False   # đặt True để train tiếp từ last_checkpoint khi Colab bị ngắt
print(cfg)
"""))

CELLS.append(md("## Cell 6 — DataLoader (ảnh đọc từ ổ cục bộ, pad giữ tỷ lệ, chuẩn hóa ImageNet)"))
CELLS.append(code("""
sys.path.insert(0, SCRIPTS)
from dataset import make_loaders, CLASS_NAMES
loaders = make_loaders(SPLIT_CSV, PKG, image_size=IMAGE_SIZE,
                       batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, path_col="rel_path")
print("Lớp:", CLASS_NAMES)
print("Số batch train/val/test:", len(loaders['train']), len(loaders['val']), len(loaders['test']))
xb, yb = next(iter(loaders['train']))
print("Batch:", tuple(xb.shape), "nhãn ví dụ:", yb[:8].tolist())
"""))

CELLS.append(md("## Cell 7 — Dựng ResNet-18 (pretrained ImageNet, head 4 lớp)"))
CELLS.append(code("""
from model import build_model, class_mapping
model = build_model(num_classes=4, pretrained=True)
print(model.fc)   # Linear(512 -> 4)
print("Ánh xạ lớp:", class_mapping()["idx_to_angle"])
print("Ý nghĩa nhãn:", class_mapping()["label_semantics"])
"""))

CELLS.append(md("## Cell 8 — Smoke test (1 batch: forward + backward + shape)"))
CELLS.append(code("""
import argparse
from train import train as run_train
ns = argparse.Namespace(data_root=PKG, split_csv=SPLIT_CSV, out_dir="/content/_smoke",
    image_size=IMAGE_SIZE, batch_size=min(BATCH_SIZE,8), epochs=1, lr=LEARNING_RATE,
    weight_decay=cfg["WEIGHT_DECAY"], seed=SEED, num_workers=0, patience=PATIENCE,
    sched_patience=SCHED_PATIENCE, device=None, no_amp=False, no_pretrained=False,
    resume=False, smoke=True)
print(run_train(ns))
"""))

CELLS.append(md("## Cell 9 — Huấn luyện (Train/Val mỗi epoch, early stopping, best checkpoint)"))
CELLS.append(code("""
assert torch.cuda.is_available(), "❌ Không có CUDA — hãy bật GPU (Cell 1) trước khi train."
# Tạo run mới, không ghi đè run cũ
import glob
RUN_ID = 1 + max([int(os.path.basename(p).split('_')[-1]) for p in glob.glob('/content/_runs/run_*')] or [0])
OUT_DIR = f"/content/_runs/run_{RUN_ID:03d}"
os.makedirs(OUT_DIR, exist_ok=True)
print("Run dir:", OUT_DIR)

ns = argparse.Namespace(data_root=PKG, split_csv=SPLIT_CSV, out_dir=OUT_DIR,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE,
    weight_decay=cfg["WEIGHT_DECAY"], seed=SEED, num_workers=NUM_WORKERS, patience=PATIENCE,
    sched_patience=SCHED_PATIENCE, device="cuda", no_amp=False, no_pretrained=False,
    resume=RESUME, smoke=False)
result = run_train(ns)
print(result)
"""))

CELLS.append(md("## Cell 10 — Đánh giá trên Test độc lập (Accuracy, F1, Confusion, FAR, inference time)"))
CELLS.append(code("""
from evaluate import evaluate as run_eval
ens = argparse.Namespace(data_root=PKG, split_csv=SPLIT_CSV,
    ckpt=os.path.join(OUT_DIR, "best_resnet18.pth"), out_dir=OUT_DIR,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device="cuda")
metrics = run_eval(ens)
import json; print(json.dumps(metrics, ensure_ascii=False, indent=2))
"""))

CELLS.append(md("## Cell 11 — Biểu đồ: Loss/Accuracy & Confusion Matrix"))
CELLS.append(code("""
from IPython.display import Image as IPImage, display
for png in ["training_curves.png", "confusion_matrix.png"]:
    p = os.path.join(OUT_DIR, png)
    if os.path.exists(p):
        print(png); display(IPImage(filename=p))
# Vài mẫu dự đoán sai (không hiển thị ảnh tài liệu để tránh lộ nội dung)
import pandas as pd
mp = os.path.join(OUT_DIR, "misclassified_samples.csv")
if os.path.exists(mp):
    print("Mẫu dự đoán sai (đầu bảng):")
    display(pd.read_csv(mp).head(10))
"""))

CELLS.append(md("## Cell 12 — Lưu kết quả về Google Drive (run riêng, kèm ZIP)"))
CELLS.append(code("""
import shutil, os
dst = os.path.join(DRIVE_RESULTS, f"run_{RUN_ID:03d}")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(OUT_DIR, dst)
# ZIP kết quả để tải nhanh về máy
zip_base = os.path.join(DRIVE_RESULTS, f"run_{RUN_ID:03d}_results")
shutil.make_archive(zip_base, "zip", OUT_DIR)
print("✅ Đã lưu lên Drive:", dst)
print("✅ ZIP kết quả:", zip_base + ".zip")
print("Tải về: mở Drive → PDF_Orientation_Training/runs/ → tải run_%03d hoặc file .zip" % RUN_ID)
"""))

NB = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(NB, f, ensure_ascii=False, indent=1)
    # xác thực JSON hợp lệ
    json.load(open(OUT, encoding="utf-8"))
    print("Đã tạo:", OUT, "|", len(CELLS), "cells, JSON hợp lệ.")
