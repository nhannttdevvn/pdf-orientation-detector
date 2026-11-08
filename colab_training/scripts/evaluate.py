#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ĐÁNH GIÁ ResNet-18 trên TEST độc lập (Colab package)
Dự án: pdf-orientation-detector

Xuất: test_metrics.json, confusion_matrix.png, training_curves.png,
      misclassified_samples.csv, training_report.md
Chỉ số: Accuracy, Macro P/R/F1, per-class P/R/F1, Confusion Matrix,
        False Auto-Rotation Rate (trang 0° bị dự đoán cần xoay), Inference time.

CHẠY:
  python evaluate.py --data-root <pkg> --split-csv <pkg>/split.csv \
                     --ckpt runs/run_001/best_resnet18.pth --out-dir runs/run_001
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import OrientationDataset, CLASS_NAMES, IDX_TO_ANGLE  # noqa: E402
from model import load_checkpoint  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score)  # noqa: E402


def pick_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, confs, times = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        t0 = time.time()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) / x.size(0))
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(1)
        ys.append(y.numpy()); ps.append(pred.cpu().numpy()); confs.append(conf.cpu().numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(confs),
            float(np.mean(times) * 1000.0))


def evaluate(args, log=print) -> dict:
    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.to(device)
    log(f"[eval] device={device} | ckpt epoch={ckpt.get('epoch')} best_val={ckpt.get('best_metric')}")

    img_size = ckpt.get("preprocessing", {}).get("image_size", args.image_size)
    ds = OrientationDataset(args.split_csv, args.data_root, "test", image_size=img_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    log(f"[eval] Test: {len(ds)} ảnh")

    y, p, conf, ms_per_img = predict(model, loader, device)
    acc = accuracy_score(y, p)
    labels_idx = list(range(len(CLASS_NAMES)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels_idx, zero_division=0)
    macro_pr, macro_rc, macro_f1, _ = precision_recall_fscore_support(
        y, p, labels=labels_idx, average="macro", zero_division=0)
    cm = confusion_matrix(y, p, labels=labels_idx)

    # False Auto-Rotation Rate: trang THỰC SỰ 0° (upright) bị dự đoán != 0° (cần xoay)
    idx0 = 0  # lớp "0"
    mask0 = (y == idx0)
    far = float(((p[mask0] != idx0).sum()) / max(mask0.sum(), 1))

    # Phân tích nhầm lẫn cặp đối diện
    def pair_confuse(a_angle, b_angle):
        ai, bi = CLASS_NAMES.index(str(a_angle)), CLASS_NAMES.index(str(b_angle))
        return int(cm[ai, bi]), int(cm[bi, ai])

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4),
                           "f1": round(float(f1[i]), 4), "support": int(sup[i])}

    metrics = {
        "test_images": int(len(y)),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_pr), 4),
        "macro_recall": round(float(macro_rc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "false_auto_rotation_rate": round(far, 4),
        "confusion_0_vs_180": {"0->180": pair_confuse(0, 180)[0], "180->0": pair_confuse(0, 180)[1]},
        "confusion_90_vs_270": {"90->270": pair_confuse(90, 270)[0], "270->90": pair_confuse(90, 270)[1]},
        "inference_ms_per_image": round(ms_per_img, 3),
        "mean_confidence": round(float(np.mean(conf)), 4),
        "device": device.type,
        "checkpoint": os.path.relpath(args.ckpt, args.out_dir) if args.out_dir else args.ckpt,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # --- Confusion matrix PNG ---
    _plot_cm(cm, CLASS_NAMES, os.path.join(args.out_dir, "confusion_matrix.png"))
    # --- Training curves PNG (nếu có history) ---
    hp = os.path.join(args.out_dir, "training_history.csv")
    if os.path.exists(hp):
        _plot_curves(hp, os.path.join(args.out_dir, "training_curves.png"))

    # --- Misclassified samples CSV ---
    df = ds.df.copy().reset_index(drop=True)
    wrong = p != y
    mis = df[wrong].copy()
    mis["true_angle"] = [IDX_TO_ANGLE[int(t)] for t in y[wrong]]
    mis["pred_angle"] = [IDX_TO_ANGLE[int(pp)] for pp in p[wrong]]
    mis["confidence"] = np.round(conf[wrong], 4)
    cols = [c for c in ["rel_path", "page_uid", "source_file", "true_angle", "pred_angle", "confidence"]
            if c in mis.columns or c in ("true_angle", "pred_angle", "confidence")]
    mis.to_csv(os.path.join(args.out_dir, "misclassified_samples.csv"),
               index=False, columns=[c for c in cols if c in mis.columns])

    _write_report(args.out_dir, metrics, ckpt)
    log(f"[eval] Accuracy={acc:.4f} MacroF1={macro_f1:.4f} FAR={far:.4f} "
        f"inf={ms_per_img:.2f}ms/ảnh | sai={int(wrong.sum())}/{len(y)}")
    return metrics


def _plot_cm(cm, names, path):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels([f"{n}°" for n in names]); ax.set_yticklabels([f"{n}°" for n in names])
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title("Confusion Matrix (Test)")
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_curves(hist_csv, path):
    h = pd.read_csv(hist_csv)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["epoch"], h["train_loss"], "-o", label="train", ms=3)
    ax[0].plot(h["epoch"], h["val_loss"], "-o", label="val", ms=3)
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h["epoch"], h["train_acc"], "-o", label="train", ms=3)
    ax[1].plot(h["epoch"], h["val_acc"], "-o", label="val", ms=3)
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _write_report(out_dir, m, ckpt):
    L = ["# Báo cáo huấn luyện — ResNet-18 hướng trang PDF", "",
         f"- Checkpoint: epoch {ckpt.get('epoch')} · best_val_acc {ckpt.get('best_metric')}",
         f"- Thiết bị đánh giá: {m['device']} · inference {m['inference_ms_per_image']} ms/ảnh", "",
         "## Kết quả Test (độc lập)", "",
         "| Chỉ số | Giá trị |", "|---|---:|",
         f"| Accuracy | {m['accuracy']} |",
         f"| Macro Precision | {m['macro_precision']} |",
         f"| Macro Recall | {m['macro_recall']} |",
         f"| Macro F1 | {m['macro_f1']} |",
         f"| **False Auto-Rotation Rate** | **{m['false_auto_rotation_rate']}** |",
         f"| Mean confidence | {m['mean_confidence']} |", "",
         "## Per-class", "", "| Lớp | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for n, v in m["per_class"].items():
        L.append(f"| {n}° | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |")
    L += ["", "## Nhầm lẫn cặp đối diện",
          f"- 0°↔180°: 0→180 = {m['confusion_0_vs_180']['0->180']}, 180→0 = {m['confusion_0_vs_180']['180->0']}",
          f"- 90°↔270°: 90→270 = {m['confusion_90_vs_270']['90->270']}, 270→90 = {m['confusion_90_vs_270']['270->90']}",
          "", "Xem `confusion_matrix.png`, `training_curves.png`, `misclassified_samples.csv`."]
    with open(os.path.join(out_dir, "training_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def build_argparser():
    p = argparse.ArgumentParser(description="Đánh giá ResNet-18 orientation trên Test.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--split-csv", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    return p


if __name__ == "__main__":
    evaluate(build_argparser().parse_args())
