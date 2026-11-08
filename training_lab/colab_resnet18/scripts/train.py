#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HUẤN LUYỆN ResNet-18 — hướng trang PDF (Colab package)
Dự án: pdf-orientation-detector

- AdamW + CrossEntropyLoss + ReduceLROnPlateau.
- Mixed precision (AMP) khi CUDA; tắt trên MPS/CPU.
- Early stopping theo Val Accuracy; lưu best + last checkpoint; history CSV.
- Resume từ last_checkpoint.pth. Progress bar. Log VRAM khi có CUDA.

CHẠY (ví dụ):
  python train.py --data-root <pkg> --split-csv <pkg>/train_val_test_split.csv \
                  --out-dir runs/run_001 --epochs 20 --batch-size 32
Smoke test 1 batch:
  python train.py ... --smoke
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import make_loaders, CLASS_NAMES  # noqa: E402
from model import build_model, save_checkpoint, write_sidecar_json  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x


def pick_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
               amp=False, desc="", smoke=False):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if amp:
                with torch.cuda.amp.autocast():
                    out = model(x); loss = criterion(out, y)
            else:
                out = model(x); loss = criterion(out, y)
            if train:
                if amp:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
            if smoke:
                break
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(args, log=print) -> dict:
    set_seed(args.seed)
    device = pick_device(args.device)
    amp = (device.type == "cuda") and not args.no_amp
    log(f"[train] device={device} | AMP={amp}")

    loaders = make_loaders(args.split_csv, args.data_root, image_size=args.image_size,
                           batch_size=args.batch_size, num_workers=args.num_workers)
    log(f"[train] batches train/val/test = "
        f"{len(loaders['train'])}/{len(loaders['val'])}/{len(loaders['test'])}")

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.sched_patience)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    cfg = {"IMAGE_SIZE": args.image_size, "BATCH_SIZE": args.batch_size,
           "EPOCHS": args.epochs, "LEARNING_RATE": args.lr, "WEIGHT_DECAY": args.weight_decay,
           "SEED": args.seed, "pretrained": not args.no_pretrained, "device": device.type,
           "amp": amp}

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "best_resnet18.pth")
    last_path = os.path.join(args.out_dir, "last_checkpoint.pth")
    hist_path = os.path.join(args.out_dir, "training_history.csv")
    write_sidecar_json(args.out_dir, cfg)

    start_epoch, best_val_acc, history, no_improve = 1, -1.0, [], 0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck: optimizer.load_state_dict(ck["optimizer_state"])
        if "scheduler_state" in ck: scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck.get("epoch", 0) + 1
        best_val_acc = ck.get("best_metric", -1.0)
        history = ck.get("history", [])
        log(f"[train] RESUME từ epoch {start_epoch} (best_val_acc={best_val_acc:.4f})")

    if args.smoke:
        log("[train] SMOKE TEST: 1 batch forward+backward.")
        tl, ta = _run_epoch(model, loaders["train"], criterion, device, optimizer, scaler, amp,
                            desc="smoke-train", smoke=True)
        vl, va = _run_epoch(model, loaders["val"], criterion, device, desc="smoke-val", smoke=True)
        xb, yb = next(iter(loaders["train"]))
        with torch.no_grad():
            out = model(xb.to(device))
        log(f"[train] output shape={tuple(out.shape)} (kỳ vọng [B,4]) | "
            f"train_loss={tl:.4f} val_loss={vl:.4f}")
        assert out.shape[1] == 4, "Output phải có 4 lớp!"
        return {"smoke": True, "output_shape": list(out.shape), "train_loss": tl, "val_loss": vl}

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], criterion, device, optimizer, scaler, amp,
                                     desc=f"E{epoch} train")
        va_loss, va_acc = _run_epoch(model, loaders["val"], criterion, device, desc=f"E{epoch} val")
        scheduler.step(va_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        vram = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
        row = {"epoch": epoch, "train_loss": round(tr_loss, 5), "train_acc": round(tr_acc, 5),
               "val_loss": round(va_loss, 5), "val_acc": round(va_acc, 5),
               "lr": lr_now, "sec": round(time.time() - t0, 1), "vram_gb": round(vram, 2)}
        history.append(row)
        log(f"[E{epoch:02d}] tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | lr={lr_now:.2e} | "
            f"{row['sec']}s | vram={row['vram_gb']}GB")

        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)

        improved = va_acc > best_val_acc
        if improved:
            best_val_acc = va_acc; no_improve = 0
            save_checkpoint(best_path, model, config=cfg, epoch=epoch, best_metric=best_val_acc,
                            optimizer=optimizer, scheduler=scheduler, history=history)
            log(f"      ↳ lưu BEST (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
        save_checkpoint(last_path, model, config=cfg, epoch=epoch, best_metric=best_val_acc,
                        optimizer=optimizer, scheduler=scheduler, history=history)

        if no_improve >= args.patience:
            log(f"[train] Early stopping: {no_improve} epoch không cải thiện.")
            break

    log(f"[train] XONG. best_val_acc={best_val_acc:.4f} -> {best_path}")
    return {"best_val_acc": best_val_acc, "epochs_run": len(history), "best_ckpt": best_path}


def build_argparser():
    p = argparse.ArgumentParser(description="Train ResNet-18 orientation.")
    p.add_argument("--data-root", required=True, help="Thư mục gốc chứa ảnh (ảnh = data_root/rel_path).")
    p.add_argument("--split-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--sched-patience", type=int, default=2)
    p.add_argument("--device", default=None, help="ép cuda/mps/cpu (mặc định tự chọn).")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Smoke test 1 batch rồi thoát.")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
