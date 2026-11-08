#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AUDIT DATASET TRƯỚC KHI TRAIN (Phase 2 — mục 2)
Kiểm tra THỰC TẾ (không tin số liệu báo cáo):
  1. Đủ 1876 ảnh? 2. Mỗi lớp 469? 3. Ảnh đọc được bằng Pillow?
  4. Nhãn góc đúng? 5. Ảnh hỏng/trống/kích thước bất thường?
  6. Mỗi page_uid đủ 4 phiên bản? 7. Ảnh trùng/gần trùng giữa các PDF khác nhau?
  8. page_uid liên kết đúng tài liệu nguồn?

KHÔNG sửa ảnh, KHÔNG gán lại nhãn. Chỉ báo cáo bất thường.
================================================================================
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from utils import save_json  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(C.PROJECT_ROOT, p)


def audit(log=print) -> dict:
    issues = []  # danh sách bất thường (mô tả), không tự sửa

    # --- Nạp CSV ---
    if not os.path.exists(C.LABELS_CSV):
        raise FileNotFoundError(f"Thiếu {C.LABELS_CSV}")
    labels = pd.read_csv(C.LABELS_CSV, dtype=str).fillna("")
    index = pd.read_csv(C.INDEX_CSV, dtype=str).fillna("") if os.path.exists(C.INDEX_CSV) else pd.DataFrame()
    manifest = pd.read_csv(C.MANIFEST_CSV, dtype=str).fillna("") if os.path.exists(C.MANIFEST_CSV) else pd.DataFrame()

    labels["label_angle"] = labels["label_angle"].astype(int)
    n_images = len(labels)
    log(f"[audit] rotation_labels.csv: {n_images} dòng")

    # --- (1) Tổng số ảnh & (2) phân bố lớp ---
    per_class = Counter(labels["label_angle"].tolist())
    log(f"[audit] Phân bố nhãn: {dict(sorted(per_class.items()))}")
    if n_images != 1876:
        issues.append(f"Tổng ảnh = {n_images}, khác 1876 kỳ vọng.")
    for ang in (0, 90, 180, 270):
        if per_class.get(ang, 0) != 469:
            issues.append(f"Lớp {ang}°: {per_class.get(ang,0)} ảnh, khác 469.")

    # --- (6) Mỗi page_uid đủ 4 phiên bản {0,90,180,270} ---
    by_uid = defaultdict(set)
    for _, r in labels.iterrows():
        by_uid[r["page_uid"]].add(int(r["label_angle"]))
    n_pages = len(by_uid)
    bad_uid = {u: sorted(s) for u, s in by_uid.items() if s != {0, 90, 180, 270}}
    log(f"[audit] Số page_uid: {n_pages} (kỳ vọng 469)")
    if bad_uid:
        issues.append(f"{len(bad_uid)} page_uid KHÔNG đủ 4 góc. Ví dụ: {list(bad_uid.items())[:3]}")

    # --- (3,4,5) Mở từng ảnh bằng Pillow: tồn tại, đọc được, kích thước, nhãn-thư mục ---
    missing, unreadable, abnormal = [], [], []
    sizes = []
    label_dir_mismatch = []
    for _, r in labels.iterrows():
        ap = _abs(r["image_path"])
        if not os.path.exists(ap):
            missing.append(r["image_path"]); continue
        try:
            with Image.open(ap) as im:
                im.verify()               # phát hiện file hỏng
            with Image.open(ap) as im2:
                w, h = im2.size
                mode = im2.mode
            sizes.append((w, h))
            if w < 32 or h < 32:
                abnormal.append((r["image_path"], f"{w}x{h} quá nhỏ"))
        except Exception as exc:  # noqa: BLE001
            unreadable.append((r["image_path"], str(exc)[:60])); continue
        # Nhãn phải khớp thư mục cha rotations/<angle padded>
        parent = os.path.basename(os.path.dirname(ap))
        try:
            if int(parent) != int(r["label_angle"]):
                label_dir_mismatch.append((r["image_path"], parent, r["label_angle"]))
        except ValueError:
            pass

    if missing:
        issues.append(f"{len(missing)} ảnh THIẾU trên đĩa. Ví dụ: {missing[:3]}")
    if unreadable:
        issues.append(f"{len(unreadable)} ảnh KHÔNG đọc được. Ví dụ: {unreadable[:3]}")
    if abnormal:
        issues.append(f"{len(abnormal)} ảnh kích thước bất thường. Ví dụ: {abnormal[:3]}")
    if label_dir_mismatch:
        issues.append(f"{len(label_dir_mismatch)} ảnh có nhãn KHÔNG khớp thư mục. Ví dụ: {label_dir_mismatch[:3]}")

    size_counter = Counter(sizes)
    log(f"[audit] Mở ảnh OK: {len(sizes)}/{n_images} | thiếu={len(missing)} hỏng={len(unreadable)}")
    log(f"[audit] Top kích thước: {size_counter.most_common(3)}")

    # --- (8) page_uid liên kết tài liệu nguồn (qua manifest) ---
    src_of_uid = {}
    uid_unlinked = []
    if not manifest.empty:
        msrc = dict(zip(manifest["page_uid"], manifest["source_file"]))
        for u in by_uid:
            s = msrc.get(u)
            if not s:
                uid_unlinked.append(u)
            else:
                src_of_uid[u] = s
    if uid_unlinked:
        issues.append(f"{len(uid_unlinked)} page_uid KHÔNG map được tài liệu nguồn trong manifest. Ví dụ: {uid_unlinked[:3]}")

    # --- (7) Ảnh trùng/gần trùng giữa các PDF khác nhau (dùng hash trong manifest) ---
    cross_pdf_dups = []
    cross_pdf_near = []
    if not manifest.empty:
        valid_uids = set(by_uid.keys())
        mv = manifest[manifest["page_uid"].isin(valid_uids)]
        # Trùng hoàn toàn: cùng image_sha256 nhưng khác source_file
        by_sha = defaultdict(set)
        for _, r in mv.iterrows():
            if r["image_sha256"]:
                by_sha[r["image_sha256"]].add(r["source_file"])
        cross_pdf_dups = [(sha[:10], sorted(files)) for sha, files in by_sha.items() if len(files) > 1]
        # Gần trùng: cùng near_dup_group nhưng khác source_file
        by_near = defaultdict(set)
        for _, r in mv.iterrows():
            if r["near_dup_group"]:
                by_near[r["near_dup_group"]].add(r["source_file"])
        cross_pdf_near = [(g, sorted(files)) for g, files in by_near.items() if len(files) > 1]
    if cross_pdf_dups:
        issues.append(f"{len(cross_pdf_dups)} nhóm ảnh TRÙNG HOÀN TOÀN giữa các PDF khác nhau "
                      f"(trong tập hợp lệ) -> phải nhóm cùng split. Ví dụ: {cross_pdf_dups[:2]}")
    if cross_pdf_near:
        log(f"[audit] {len(cross_pdf_near)} nhóm GẦN trùng trải trên nhiều PDF (cảnh báo nhóm split).")

    # --- Tổng hợp ---
    summary = {
        "n_images": n_images,
        "per_class": {str(k): v for k, v in sorted(per_class.items())},
        "n_pages": n_pages,
        "pages_missing_variants": len(bad_uid),
        "images_missing_on_disk": len(missing),
        "images_unreadable": len(unreadable),
        "images_abnormal_size": len(abnormal),
        "label_dir_mismatch": len(label_dir_mismatch),
        "uid_unlinked_to_source": len(uid_unlinked),
        "distinct_sizes": len(size_counter),
        "top_sizes": [{"size": f"{w}x{h}", "count": c} for (w, h), c in size_counter.most_common(5)],
        "cross_pdf_exact_dup_groups": len(cross_pdf_dups),
        "cross_pdf_near_dup_groups": len(cross_pdf_near),
        "cross_pdf_exact_examples": cross_pdf_dups[:5],
        "issues": issues,
        "passed": len(issues) == 0,
    }
    save_json(C.AUDIT_JSON, summary)
    _write_md(summary)
    log(f"[audit] {'PASS — không bất thường' if summary['passed'] else f'CÓ {len(issues)} bất thường (xem báo cáo)'}")
    return summary


def _write_md(s: dict) -> None:
    L = ["# Audit dataset — Phase 2", "",
         f"- Tổng ảnh: **{s['n_images']}** | Số trang (page_uid): **{s['n_pages']}**",
         f"- Phân bố lớp: {s['per_class']}",
         f"- Ảnh thiếu/hỏng/bất thường: {s['images_missing_on_disk']}/{s['images_unreadable']}/{s['images_abnormal_size']}",
         f"- page_uid thiếu biến thể: {s['pages_missing_variants']} | nhãn lệch thư mục: {s['label_dir_mismatch']}",
         f"- page_uid không map nguồn: {s['uid_unlinked_to_source']}",
         f"- Nhóm trùng hoàn toàn giữa các PDF (tập hợp lệ): {s['cross_pdf_exact_dup_groups']}",
         f"- Nhóm gần trùng trải nhiều PDF: {s['cross_pdf_near_dup_groups']}",
         f"- Kích thước ảnh (top): {s['top_sizes']}",
         ""]
    if s["issues"]:
        L.append("## Bất thường phát hiện")
        for i in s["issues"]:
            L.append(f"- {i}")
    else:
        L.append("## Kết luận: PASS — không phát hiện bất thường.")
    os.makedirs(C.REPORTS_DIR, exist_ok=True)
    with open(C.AUDIT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    audit()
