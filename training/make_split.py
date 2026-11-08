#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CHIA TRAIN / VALIDATION / TEST THEO TÀI LIỆU NGUỒN (Phase 2 — mục 3)
Dự án: pdf-orientation-detector

Chống rò rỉ (leakage) nhiều tầng:
  - 4 ảnh xoay của cùng 1 trang  -> cùng split (chia theo page_uid, không theo ảnh).
  - Các trang cùng 1 PDF nguồn    -> cùng split (nhóm theo source_file).
  - Các PDF trùng/gần trùng nội dung -> gộp cùng nhóm bằng Union-Find dựa trên
    bằng chứng: chung image_sha256 (trùng hoàn toàn) hoặc chung near_dup_group
    (gần trùng) trong manifest.

Chia theo NHÓM TÀI LIỆU (greedy theo số trang) để đạt ~70/15/15, seed=42.
Nhãn 4 lớp tự cân bằng trong mỗi split vì mỗi trang đóng góp đúng 1 ảnh/ lớp.

Xuất: data/dataset_review/train_val_test_split.csv  (mỗi dòng = 1 ẢNH)
================================================================================
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from utils import set_seed, save_json  # noqa: E402


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _valid_mask(d: pd.DataFrame) -> pd.Series:
    b = lambda col: d[col].astype(str).str.strip().str.lower().isin(("1", "true", "yes", "y", "x"))
    return b("confirmed") & ~b("is_duplicate") & ~b("flag_exclude") & ~b("flag_blank") & ~b("flag_pair") & ~b("flag_hard")


def build_groups(log=print):
    """Trả (uid_to_source, uid_to_group, group_to_uids) cho các trang hợp lệ."""
    man = pd.read_csv(C.MANIFEST_CSV, dtype=str).fillna("")
    valid = man[_valid_mask(man)].copy()
    log(f"[split] Trang hợp lệ trong manifest: {len(valid)}")

    uid_to_source = dict(zip(valid["page_uid"], valid["source_file"]))

    uf = UnionFind()
    for s in valid["source_file"].unique():
        uf.find(s)  # khởi tạo mỗi PDF là 1 nhóm

    # Gộp các PDF chỉ khi có BẰNG CHỨNG MẠNH: chung trang trùng HOÀN TOÀN (image_sha256).
    # KHÔNG dùng near_dup_group (dHash thumbnail) để gộp: bằng chứng yếu, xâu chuỗi
    # nhầm nhiều PDF khác nhau qua trang trắng/header giống nhau -> phá cấu trúc tài liệu.
    # (Sau khử trùng, các trang hợp lệ gần như không còn chung sha256 giữa các PDF,
    #  nên thực tế mỗi PDF là một nhóm độc lập.)
    STRONG_KEYS = ("image_sha256",)
    for key in STRONG_KEYS:
        groups = defaultdict(set)
        for _, r in valid.iterrows():
            if r[key]:
                groups[r[key]].add(r["source_file"])
        for files in groups.values():
            files = sorted(files)
            for f in files[1:]:
                uf.union(files[0], f)

    # Ghi nhận (để báo cáo) các cặp PDF có chồng lấn GẦN trùng — cảnh báo, không gộp.
    near_overlap = defaultdict(set)
    for _, r in valid.iterrows():
        if r["near_dup_group"]:
            near_overlap[r["near_dup_group"]].add(r["source_file"])
    build_groups.near_dup_cross_pdf = sum(1 for v in near_overlap.values() if len(v) > 1)

    uid_to_group = {u: uf.find(s) for u, s in uid_to_source.items()}
    group_to_uids = defaultdict(list)
    for u, g in uid_to_group.items():
        group_to_uids[g].append(u)
    log(f"[split] Số PDF nguồn: {valid['source_file'].nunique()} -> gộp còn {len(group_to_uids)} nhóm tài liệu độc lập.")
    return uid_to_source, uid_to_group, group_to_uids


def assign_splits(group_to_uids, log=print):
    """Greedy: gán nhóm (lớn trước) vào split đang thiếu nhiều nhất so với chỉ tiêu trang."""
    set_seed(C.SEED)
    total_pages = sum(len(v) for v in group_to_uids.values())
    targets = {k: C.SPLIT_RATIOS[k] * total_pages for k in C.SPLIT_RATIOS}
    current = {k: 0 for k in C.SPLIT_RATIOS}
    assign = {}

    # Sắp nhóm theo kích thước giảm dần; xáo trộn ổn định theo seed cho các nhóm bằng nhau
    import random
    groups = sorted(group_to_uids.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    rng = random.Random(C.SEED)
    rng.shuffle(groups)
    groups.sort(key=lambda kv: -len(kv[1]))  # ổn định: lớn trước, trong cùng cỡ đã xáo

    for g, uids in groups:
        # Deficit TUYỆT ĐỐI: nhóm lớn (xử lý trước) rơi vào split còn thiếu nhiều trang
        # nhất (thường là train) -> PDF lớn dồn về train, PDF nhỏ rải cho val/test
        # giúp val/test có nhiều nhóm tài liệu hơn.
        deficit = {k: targets[k] - current[k] for k in current}
        best = max(deficit, key=lambda k: deficit[k])
        assign[g] = best
        current[best] += len(uids)
    log(f"[split] Số trang mỗi split: {current} (chỉ tiêu ~{ {k:round(v) for k,v in targets.items()} })")
    return assign


def write_split(log=print) -> dict:
    uid_to_source, uid_to_group, group_to_uids = build_groups(log)
    assign = assign_splits(group_to_uids, log)
    uid_split = {u: assign[g] for u, g in uid_to_group.items()}

    labels = pd.read_csv(C.LABELS_CSV, dtype=str).fillna("")
    labels["label_angle"] = labels["label_angle"].astype(int)

    rows = []
    skipped = 0
    for _, r in labels.iterrows():
        u = r["page_uid"]
        if u not in uid_split:   # ảnh của trang không hợp lệ -> loại khỏi training package
            skipped += 1
            continue
        rows.append({
            "image_path": r["image_path"],
            "page_uid": u,
            "source_file": uid_to_source[u],
            "doc_group": uid_to_group[u],
            "label_angle": r["label_angle"],
            "label_idx": C.ANGLE_TO_IDX[r["label_angle"]],
            "split": uid_split[u],
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(C.SPLIT_CSV), exist_ok=True)
    df.to_csv(C.SPLIT_CSV, index=False)
    log(f"[split] Ghi {len(df)} ảnh vào {os.path.relpath(C.SPLIT_CSV, C.PROJECT_ROOT)} (bỏ {skipped} ảnh trang không hợp lệ).")

    # --- Kiểm tra leakage: không nhóm/ trang/ PDF nào nằm ở >1 split ---
    leak = {}
    for key in ("page_uid", "doc_group", "source_file"):
        g = df.groupby(key)["split"].nunique()
        bad = g[g > 1]
        leak[key] = bad.index.tolist()

    # --- Thống kê ---
    stats = {"total_images": len(df), "total_pages": df["page_uid"].nunique(),
             "total_doc_groups": df["doc_group"].nunique(), "splits": {}}
    for sp in ("train", "val", "test"):
        d = df[df["split"] == sp]
        stats["splits"][sp] = {
            "images": len(d),
            "pages": d["page_uid"].nunique(),
            "doc_groups": d["doc_group"].nunique(),
            "source_pdfs": d["source_file"].nunique(),
            "per_class": {str(a): int((d["label_angle"] == a).sum()) for a in (0, 90, 180, 270)},
        }
    stats["leakage"] = {
        "page_uid_in_multiple_splits": len(leak["page_uid"]),
        "doc_group_in_multiple_splits": len(leak["doc_group"]),
        "source_pdf_in_multiple_splits": len(leak["source_file"]),
        "clean": all(len(v) == 0 for v in leak.values()),
    }
    # Giới hạn: test có đủ nhóm độc lập không
    stats["test_independent_groups"] = stats["splits"]["test"]["doc_groups"]
    if stats["splits"]["test"]["doc_groups"] < 2:
        stats["warning"] = ("Test chỉ gồm <2 nhóm tài liệu độc lập -> độ tin cậy đánh giá hạn chế; "
                            "cần thêm tài liệu nguồn đa dạng để Test đại diện hơn.")

    save_json(os.path.join(C.DATA_REVIEW_DIR, "split_report.json"), stats)
    log(f"[split] Leakage sạch: {stats['leakage']['clean']} | "
        f"train/val/test ảnh = "
        f"{stats['splits']['train']['images']}/{stats['splits']['val']['images']}/{stats['splits']['test']['images']}")
    return stats


if __name__ == "__main__":
    import json
    s = write_split()
    print(json.dumps(s, ensure_ascii=False, indent=2))
