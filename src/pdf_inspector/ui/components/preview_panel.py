"""
Component Khu vực 4 (Cột phải ~38.5%) — 'Xem trước trang lỗi', Hộp Chi Tiết & 3 Nút Tương Tác
(`src/pdf_inspector/ui/components/preview_panel.py`).

Đảm bảo:
- Header bên phải có 2 tab 'Bản gốc' / 'Bản đề xuất' (không chạy lại OCR khi chuyển tab).
- Vùng xem trước cao 220px dùng ảnh render thật từ PDF kèm overlay tọa độ thực từ backend
  (Boundary / Stripe / Rotation) và nhãn pill nổi.
- Hộp chi tiết gồm đúng 4 hàng: Trang | Lỗi | Trạng thái | Đề xuất (hỗ trợ chọn riêng từng lỗi
  khi một trang có nhiều lỗi).
- Hàng 3 nút: 'Chấp nhận' | 'Bỏ qua' | 'Sửa thủ công' kết nối trực tiếp với workflow
  `DocumentPipeline.update_decisions_and_artifacts`, `BoundaryInspectionService`,
  `save_stripe_review_feedback` và `save_boundary_review_feedback`.
"""

import base64
from collections import OrderedDict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Import boundary_detector trước để tránh vòng lặp import sẵn có trong backend
import pdf_inspector.detectors.boundary.boundary_detector  # noqa: F401
from pdf_inspector.core.config import (
    BOUNDARY_FEEDBACK_PATH,
    PREVIEW_DPI,
    STRIPE_FEEDBACK_PATH,
    VALID_ANGLES,
)
from pdf_inspector.models.schemas import DefectResult, PageDecision, PipelineResult
from pdf_inspector.services.boundary_inspection_service import BoundaryInspectionService
from pdf_inspector.services.report_service import ReportGenerator
from pdf_inspector.ui.styles import clean_html
from pdf_inspector.ui.state_manager import (
    DefectViewItem,
    PageIssueRow,
    build_problem_page_rows,
    encode_rgb_to_png_bytes,
    get_cached_page_preview_rgb,
    get_default_boundary_cut_norm,
    rebuild_outputs_without_ocr,
)


# ==============================================================================
# LƯU PHẢN HỒI NGƯỜI DÙNG (GIỮ NGUYÊN CHỮ KÝ TƯƠNG THÍCH KIỂM THỬ)
# ==============================================================================


def save_stripe_review_feedback(
    file_name: str,
    page_number: int,
    stripe_status: str,
    feedback_action: str,
    stripe_defects: List[DefectResult],
    feedback_path: Path = STRIPE_FEEDBACK_PATH,
) -> Path:
    """
    Lưu phản hồi đánh giá sọc của người dùng (`Confirm defect`, `False positive`, `Uncertain`)
    vào `data/outputs/stripe_review_feedback.json` (`verified_ground_truth: false`).
    """
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "verified_ground_truth": False,
        "source": "user_ui_review_feedback",
        "note": (
            "Unverified user review feedback collected from Streamlit UI. "
            "Do not use directly as ground-truth training data without independent verification."
        ),
        "entries": [],
    }
    if feedback_path.exists():
        try:
            loaded = json.loads(feedback_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("entries"), list):
                payload = loaded
        except Exception:
            pass

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file_name": file_name,
        "page_number": page_number,
        "detector_stripe_status": stripe_status,
        "user_feedback": feedback_action,
        "verified_ground_truth": False,
        "detected_stripes": [d.to_dict() for d in stripe_defects],
    }

    entries: List[Dict[str, Any]] = payload.get("entries", [])
    updated = False
    for idx, item in enumerate(entries):
        if (
            item.get("file_name") == file_name
            and item.get("page_number") == page_number
        ):
            entries[idx] = entry
            updated = True
            break
    if not updated:
        entries.append(entry)

    payload["entries"] = entries
    feedback_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return feedback_path


def save_boundary_review_feedback(
    file_name: str,
    page_number: int,
    boundary_status: str,
    feedback_action: str,
    adjusted_boundary_norm: Optional[float],
    boundary_defects: List[DefectResult],
    feedback_path: Path = BOUNDARY_FEEDBACK_PATH,
) -> Path:
    """
    Lưu phản hồi đánh giá Page Boundary của người dùng (`Confirm`, `False Positive`, `Adjust Boundary`)
    vào `data/outputs/boundary_review_feedback.json` (`verified_ground_truth: false`).
    """
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "verified_ground_truth": False,
        "source": "user_ui_boundary_review_feedback",
        "note": (
            "Unverified user boundary review feedback collected from Streamlit UI. "
            "Do not use directly as ground-truth training data without independent verification."
        ),
        "entries": [],
    }
    if feedback_path.exists():
        try:
            loaded = json.loads(feedback_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("entries"), list):
                payload = loaded
        except Exception:
            pass

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file_name": file_name,
        "page_number": page_number,
        "detector_boundary_status": boundary_status,
        "user_feedback": feedback_action,
        "adjusted_boundary_norm": adjusted_boundary_norm,
        "verified_ground_truth": False,
        "detected_boundary_defects": [d.to_dict() for d in boundary_defects],
    }

    entries: List[Dict[str, Any]] = payload.get("entries", [])
    updated = False
    for idx, item in enumerate(entries):
        if (
            item.get("file_name") == file_name
            and item.get("page_number") == page_number
        ):
            entries[idx] = entry
            updated = True
            break
    if not updated:
        entries.append(entry)

    payload["entries"] = entries
    feedback_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return feedback_path


# ==============================================================================
# CÁC HÀM DỰNG OVERLAY TỌA ĐỘ THỰC TẾ TRÊN ẢNH PDF RENDER THẬT
# ==============================================================================


def _img_to_data_uri(rgb_img: np.ndarray) -> str:
    b64 = base64.b64encode(encode_rgb_to_png_bytes(rgb_img)).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _build_boundary_operation_dict(
    decision: PageDecision,
    total_pages: int,
    cut_norm: float,
    explicit_action: Optional[str] = None,
) -> Dict[str, Any]:
    """Tạo cấu trúc thao tác tái tạo trang gửi vào `BoundaryInspectionService`."""
    proposal = decision.reconstruction_proposal or {}
    raw_action = explicit_action or str(
        proposal.get("action_type") or "SPLIT_SINGLE_PAGE"
    )
    pno = decision.page_number
    op: Dict[str, Any] = {
        "action": raw_action,
        "page_number": pno,
        "cut_norm": float(cut_norm),
        "user_confirmed": True,
    }

    rel_pages = proposal.get("related_pages") or decision.related_pages or []
    if raw_action in (
        "MERGE_SPLIT_PAGES",
        "MERGE_WITH_NEXT",
        "REMOVE_OVERLAP_AND_MERGE",
        "DEDUPLICATE_OVERLAP_WITH_NEXT",
        "SPLIT_AND_MERGE_SPILLOVER",
    ):
        adj = (
            int(rel_pages[0])
            if rel_pages
            else (pno + 1 if pno < total_pages else max(1, pno - 1))
        )
        if adj != pno:
            op["second_page_number"] = adj
        ov_norm = float(proposal.get("overlap_strip_norm") or 0.15)
        for bdef in decision.boundary_defects:
            if bdef.defect_type == "CONTENT_OVERLAP":
                ov_norm = float((bdef.details or {}).get("overlap_ratio", ov_norm))
                break
        op["overlap_norm"] = ov_norm
    return op


def _get_cached_boundary_proposal_preview(
    pdf_path: Path,
    file_hash: str,
    decision: PageDecision,
    total_pages: int,
    cut_norm: float,
) -> Dict[str, Any]:
    """Lazy-render và cache ảnh xem trước 'Bản đề xuất' cho lỗi Boundary."""
    cache: OrderedDict = st.session_state.setdefault(
        "proposal_preview_cache", OrderedDict()
    )
    op = _build_boundary_operation_dict(decision, total_pages, cut_norm)
    act = str(op["action"])
    sec_p = op.get("second_page_number")
    key = (file_hash, decision.page_number, act, round(float(cut_norm), 3), sec_p)

    if key in cache:
        cache.move_to_end(key)
        return cache[key]

    if sec_p is not None and act in (
        "MERGE_SPLIT_PAGES",
        "REMOVE_OVERLAP_AND_MERGE",
        "SPLIT_AND_MERGE_SPILLOVER",
    ):
        mode_str = (
            "CONTENT_OVERLAP"
            if act == "REMOVE_OVERLAP_AND_MERGE"
            else ("MIXED_PAGE" if act == "SPLIT_AND_MERGE_SPILLOVER" else "SPLIT_PAGE")
        )
        data = BoundaryInspectionService.generate_adjacent_merge_preview(
            pdf_path=pdf_path,
            first_page_number=decision.page_number,
            second_page_number=int(sec_p),
            mode=mode_str,
            split_norm_on_first_page=cut_norm,
            overlap_ratio=float(op.get("overlap_norm", 0.15)),
            dpi=PREVIEW_DPI,
        )
        res = {"kind": "MERGE", "data": data}
    else:
        data = BoundaryInspectionService.generate_split_preview(
            pdf_path=pdf_path,
            page_number=decision.page_number,
            boundary_norm=cut_norm,
            dpi=PREVIEW_DPI,
        )
        res = {"kind": "SPLIT", "data": data}

    cache[key] = res
    while len(cache) > 10:
        cache.popitem(last=False)
    return res


def _render_original_preview_html(
    orig_rgb: np.ndarray,
    decision: PageDecision,
    active_item: DefectViewItem,
    scan_dpi: int,
) -> str:
    """
    Sinh HTML vùng xem trước 'Bản gốc' dùng ảnh PDF thật kèm overlay tọa độ thực từ backend:
    - Boundary: hiển thị khung phóng cận vùng ranh giới trang với hộp nét đứt đỏ hồng chuẩn xác theo tọa độ thực.
    - Stripe: vẽ hộp đánh dấu đúng tọa độ vệt sọc (`bbox` / `endpoints_px`) từ `StripeDetector`.
    - Rotation: hiển thị bản gốc bị xoay kèm nhãn hướng phát hiện.
    """
    h_img, w_img = orig_rgb.shape[:2]
    scale = float(PREVIEW_DPI) / float(max(1, scan_dpi))
    pill_cls = "orange" if active_item.is_suspected else ""
    safe_pill = html.escape(active_item.pill_label)

    # 1. LỖI BOUNDARY / LẪN TRANG
    if active_item.key == "boundary":
        saved_cuts: Dict[int, float] = st.session_state.get("boundary_adjustments", {})
        cut_norm = float(
            saved_cuts.get(
                decision.page_number, get_default_boundary_cut_norm(decision)
            )
        )
        b0 = decision.boundary_defects[0] if decision.boundary_defects else None
        b_type = b0.defect_type if b0 else "MIXED_PAGE"

        if b_type == "CONTENT_TRUNCATED" and b0 and b0.bbox:
            bx0, by0, bx1, by1 = b0.bbox
            x0_n = max(0.01, min(0.95, (bx0 * scale) / w_img))
            y0_n = max(0.0, min(0.95, (by0 * scale) / h_img))
            x1_n = max(x0_n + 0.05, min(0.99, (bx1 * scale) / w_img))
            y1_n = max(y0_n + 0.08, min(1.0, (by1 * scale) / h_img))
            edge_side = str((b0.details or {}).get("edge_side", "TOP")).upper()
            win_y0 = 0.0 if edge_side != "BOTTOM" else 0.42
            win_y1 = 0.58 if edge_side != "BOTTOM" else 1.0
        elif b_type == "CONTENT_OVERLAP" and b0 and b0.bbox:
            bx0, by0, bx1, by1 = b0.bbox
            x0_n = max(0.02, min(0.94, (bx0 * scale) / w_img))
            y0_n = max(0.0, min(0.92, (by0 * scale) / h_img))
            x1_n = max(x0_n + 0.10, min(0.98, (bx1 * scale) / w_img))
            y1_n = max(y0_n + 0.12, min(1.0, (by1 * scale) / h_img))
            mid_y = 0.5 * (y0_n + y1_n)
            win_y0 = max(0.0, mid_y - 0.32)
            win_y1 = min(1.0, mid_y + 0.32)
        else:
            # MIXED_PAGE / SPLIT_PAGE / PAGE_BOUNDARY_SUSPECTED: đánh dấu phần nội dung lẫn từ đường ranh giới `cut_norm`
            cut_clamped = max(0.08, min(0.90, cut_norm))
            win_y0 = max(0.0, cut_clamped - 0.34)
            win_y1 = min(1.0, cut_clamped + 0.32)
            x0_n, x1_n = 0.02, 0.98
            y0_n, y1_n = cut_clamped, win_y1 - 0.015

        r0 = max(0, min(h_img - 20, int(round(win_y0 * h_img))))
        r1 = max(r0 + 20, min(h_img, int(round(win_y1 * h_img))))
        cropped = orig_rgb[r0:r1, :]
        span_n = max(0.05, win_y1 - win_y0)

        top_pct = max(0.0, min(94.0, ((y0_n - win_y0) / span_n) * 100.0))
        bot_pct = max(top_pct + 8.0, min(99.0, ((y1_n - win_y0) / span_n) * 100.0))
        left_pct = x0_n * 100.0
        width_pct = max(6.0, (x1_n - x0_n) * 100.0)
        height_pct = max(8.0, bot_pct - top_pct)

        box_cls = "suspected" if active_item.is_suspected else ""
        uri = _img_to_data_uri(cropped)
        return f"""
        <div class="doc-preview-stage">
          <div class="doc-sheet-frame boundary-focus">
            <img src="{uri}" alt="Trang {decision.page_number}" />
            <div class="defect-bbox-overlay {box_cls}"
                 style="left:{left_pct:.1f}%; top:{top_pct:.1f}%; width:{width_pct:.1f}%; height:{height_pct:.1f}%;"></div>
          </div>
          <div class="preview-floating-pill {pill_cls}" style="top:52%;">
            <span class="preview-pill-dot"></span>
            <span>{safe_pill}</span>
          </div>
        </div>
        """

    # 2. LỖI VỆT SỌC (STRIPE) — Vẽ đúng tọa độ thực của từng sọc phát hiện được
    if active_item.key == "stripe":
        overlays_html: List[str] = []
        for s in decision.stripe_defects:
            if not s.bbox:
                continue
            bx0, by0, bx1, by1 = s.bbox
            x0_p = max(0.0, min(98.0, ((bx0 * scale) / w_img) * 100.0))
            y0_p = max(0.0, min(98.0, ((by0 * scale) / h_img) * 100.0))
            x1_p = max(x0_p + 0.5, min(100.0, ((bx1 * scale) / w_img) * 100.0))
            y1_p = max(y0_p + 0.5, min(100.0, ((by1 * scale) / h_img) * 100.0))

            # Mở rộng đệm thị giác tối thiểu 3.2% quanh tâm vệt sọc mảnh để mắt thường thấy rõ vùng lỗi
            w_p = x1_p - x0_p
            h_p = y1_p - y0_p
            if w_p < 3.4:
                cx = 0.5 * (x0_p + x1_p)
                x0_p = max(0.0, cx - 1.7)
                w_p = min(100.0 - x0_p, 3.4)
            if h_p < 3.4:
                cy = 0.5 * (y0_p + y1_p)
                y0_p = max(0.0, cy - 1.7)
                h_p = min(100.0 - y0_p, 3.4)

            s_cls = (
                "suspected"
                if s.status != "CONFIRMED"
                else ""
            )
            overlays_html.append(
                f'<div class="defect-bbox-overlay {s_cls}" '
                f'style="left:{x0_p:.1f}%; top:{y0_p:.1f}%; width:{w_p:.1f}%; height:{h_p:.1f}%;"></div>'
            )

        uri = _img_to_data_uri(orig_rgb)
        return f"""
        <div class="doc-preview-stage">
          <div class="doc-sheet-frame">
            <img src="{uri}" alt="Trang {decision.page_number}" />
            {''.join(overlays_html)}
          </div>
          <div class="preview-floating-pill {pill_cls}">
            <span class="preview-pill-dot"></span>
            <span>{safe_pill}</span>
          </div>
        </div>
        """

    # 3. LỖI XOAY HƯỚNG (ROTATION)
    uri = _img_to_data_uri(orig_rgb)
    box_cls = "suspected" if active_item.is_suspected else ""
    return f"""
    <div class="doc-preview-stage">
      <div class="doc-sheet-frame">
        <img src="{uri}" alt="Trang {decision.page_number}" />
        <div class="defect-bbox-overlay {box_cls}" style="inset:3px;"></div>
      </div>
      <div class="preview-floating-pill {pill_cls}">
        <span class="preview-pill-dot"></span>
        <span>{safe_pill}</span>
      </div>
    </div>
    """


def _render_proposed_preview_html(
    pdf_path: Path,
    file_hash: str,
    decision: PageDecision,
    active_item: DefectViewItem,
    total_pages: int,
) -> str:
    """Sinh HTML cho tab 'Bản đề xuất' từ kết quả xử lý/tái tạo thật của backend."""
    pno = decision.page_number

    # 1. Đề xuất cho Boundary (Tách / Ghép trang thật từ `BoundaryInspectionService`)
    if active_item.key == "boundary":
        if not active_item.can_accept:
            orig_rgb = get_cached_page_preview_rgb(
                pdf_path, file_hash, pno, dpi=PREVIEW_DPI, extra_rotation=0
            )
            uri = _img_to_data_uri(orig_rgb)
            return f"""
            <div class="doc-preview-stage">
              <div class="doc-sheet-frame">
                <img src="{uri}" alt="Trang {pno}" />
              </div>
              <div class="preview-floating-pill orange">
                <span class="preview-pill-dot"></span>
                <span>Cần quét lại trang gốc</span>
              </div>
            </div>
            """

        saved_cuts: Dict[int, float] = st.session_state.get("boundary_adjustments", {})
        cut_norm = float(saved_cuts.get(pno, get_default_boundary_cut_norm(decision)))
        rec_bundle = _get_cached_boundary_proposal_preview(
            pdf_path=pdf_path,
            file_hash=file_hash,
            decision=decision,
            total_pages=total_pages,
            cut_norm=cut_norm,
        )
        pdata = rec_bundle["data"]
        if rec_bundle["kind"] == "SPLIT":
            uri_a = _img_to_data_uri(pdata["top_part_rgb"])
            uri_b = _img_to_data_uri(pdata["bottom_part_rgb"])
            return f"""
            <div class="doc-preview-stage">
              <div class="split-preview-pair">
                <div class="split-preview-card">
                  <img src="{uri_a}" alt="Phần trên" />
                  <span class="split-preview-cap">Trang {pno}A (Trên)</span>
                </div>
                <div class="split-preview-card">
                  <img src="{uri_b}" alt="Phần dưới" />
                  <span class="split-preview-cap">Trang {pno}B (Dưới)</span>
                </div>
              </div>
              <div class="preview-floating-pill green" style="top:18%; right:10px;">
                <span class="preview-pill-dot"></span>
                <span>Tách thành 2 trang</span>
              </div>
            </div>
            """

        uri_m1 = _img_to_data_uri(pdata["reconstructed_page_1_rgb"])
        if pdata.get("reconstructed_page_2_rgb") is not None:
            uri_m2 = _img_to_data_uri(pdata["reconstructed_page_2_rgb"])
            return f"""
            <div class="doc-preview-stage">
              <div class="split-preview-pair">
                <div class="split-preview-card">
                  <img src="{uri_m1}" alt="Trang 1 sau ghép" />
                  <span class="split-preview-cap">Trang {pdata['first_page']} mới</span>
                </div>
                <div class="split-preview-card">
                  <img src="{uri_m2}" alt="Trang 2 sau ghép" />
                  <span class="split-preview-cap">Trang {pdata['second_page']} mới</span>
                </div>
              </div>
              <div class="preview-floating-pill green" style="top:18%; right:10px;">
                <span class="preview-pill-dot"></span>
                <span>Đã tách &amp; ghép lại</span>
              </div>
            </div>
            """
        return f"""
        <div class="doc-preview-stage">
          <div class="doc-sheet-frame">
            <img src="{uri_m1}" alt="Trang ghép hoàn chỉnh" />
          </div>
          <div class="preview-floating-pill green">
            <span class="preview-pill-dot"></span>
            <span>Đã ghép hoàn chỉnh</span>
          </div>
        </div>
        """

    # 2. Đề xuất cho Rotation (Ảnh thật sau khi xoay)
    if active_item.key == "rotation":
        manual_map: Dict[int, int] = st.session_state.get("manual_overrides", {})
        if pno in manual_map:
            prop_rot = int(manual_map[pno])
        elif decision.correction_angle in (90, 180, 270):
            prop_rot = int(decision.correction_angle)
        elif decision.detected_rotation in (90, 180, 270):
            prop_rot = int(decision.detected_rotation)
        else:
            prop_rot = 0

        rot_rgb = get_cached_page_preview_rgb(
            pdf_path, file_hash, pno, dpi=PREVIEW_DPI, extra_rotation=prop_rot
        )
        uri_rot = _img_to_data_uri(rot_rgb)
        badge_txt = (
            f"Đã xoay {prop_rot}°" if prop_rot != 0 else "Giữ nguyên hướng (0°)"
        )
        return f"""
        <div class="doc-preview-stage">
          <div class="doc-sheet-frame">
            <img src="{uri_rot}" alt="Trang {pno} đề xuất" />
          </div>
          <div class="preview-floating-pill green">
            <span class="preview-pill-dot"></span>
            <span>{html.escape(badge_txt)}</span>
          </div>
        </div>
        """

    # 3. Đề xuất cho Stripe (Backend giữ nguyên điểm ảnh, nếu trang có xoay thì áp dụng góc xoay)
    eff_rot = decision.correction_angle if decision.corrected else 0
    clean_rgb = get_cached_page_preview_rgb(
        pdf_path, file_hash, pno, dpi=PREVIEW_DPI, extra_rotation=eff_rot
    )
    uri_clean = _img_to_data_uri(clean_rgb)
    note_lbl = (
        f"Giữ nguyên chữ + Xoay {eff_rot}°"
        if eff_rot != 0
        else "Bảo toàn chữ gốc (không tự xóa sọc)"
    )
    return f"""
    <div class="doc-preview-stage">
      <div class="doc-sheet-frame">
        <img src="{uri_clean}" alt="Trang {pno}" />
      </div>
      <div class="preview-floating-pill orange">
        <span class="preview-pill-dot"></span>
        <span>{html.escape(note_lbl)}</span>
      </div>
    </div>
    """


# ==============================================================================
# HỘP THOẠI SỬA THỦ CÔNG (@st.dialog) — GIỮ NGUYÊN BỐ CỤC CHÍNH 100%
# ==============================================================================


@st.dialog("Sửa thủ công trang lỗi")
def _open_manual_edit_dialog(
    row: PageIssueRow,
    active_defect_key: str,
    file_name: str,
    total_pages: int,
) -> None:
    """
    Hộp thoại chỉnh sửa thủ công kết nối trực tiếp tới backend:
    - Chỉnh góc xoay thủ công (0° / 90° / 180° / 270°) qua `DocumentPipeline.update_decisions_and_artifacts`.
    - Chỉnh vị trí đường cắt an toàn `y_norm` & chế độ tách/ghép qua `BoundaryInspectionService`.
    - Xuất nhanh báo cáo JSON/CSV kiểm định nếu người dùng cần tải riêng.
    """
    pno = row.page_number
    decision = row.decision
    st.markdown(f"**Trang đang chỉnh sửa:** Trang `{pno}` / `{total_pages}`")

    has_boundary_edit = any(
        it.key == "boundary" and it.can_manual_edit for it in row.defect_items
    )
    has_rotation_issue = row.rotation_state != "NONE" or active_defect_key == "rotation"

    # Phần 1: Góc xoay trang (luôn khả dụng để kỹ thuật viên xoay tay bất kỳ trang lỗi nào)
    st.markdown("##### 1. Góc xoay trang PDF")
    cur_manual = st.session_state["manual_overrides"].get(pno)
    rot_opts = [
        "Tự động theo hệ thống",
        "0° (Giữ nguyên)",
        "90° (Xoay phải)",
        "180° (Lật ngược)",
        "270° (Xoay trái)",
    ]
    default_rot_idx = (
        0 if cur_manual is None else (VALID_ANGLES.index(cur_manual) + 1)
    )
    chosen_rot_str = st.selectbox(
        "Chọn góc xoay áp dụng cho trang này:",
        options=rot_opts,
        index=default_rot_idx,
        key=f"dlg_rot_sel_{pno}",
    )

    # Phần 2: Chỉnh ranh giới tách / ghép trang nếu trang có lỗi Boundary hỗ trợ tái tạo
    chosen_cut_norm: Optional[float] = None
    chosen_bnd_action: Optional[str] = None
    if has_boundary_edit:
        st.divider()
        st.markdown("##### 2. Tinh chỉnh ranh giới tách / ghép trang")
        def_cut = get_default_boundary_cut_norm(decision)
        cur_cut = float(
            st.session_state.get("boundary_adjustments", {}).get(pno, def_cut)
        )
        chosen_cut_norm = st.slider(
            "Vị trí đường cắt chuẩn hóa theo chiều dọc (tự động căn vào khe trắng an toàn):",
            min_value=0.08,
            max_value=0.92,
            value=min(0.92, max(0.08, cur_cut)),
            step=0.01,
            key=f"dlg_bnd_slider_{pno}",
        )
        action_labels = [
            ("SPLIT_SINGLE_PAGE", "Tách trang hiện tại thành 2 trang độc lập"),
            ("SPLIT_AND_MERGE_SPILLOVER", "Tách phần tràn và ghép sang trang liền kề"),
            ("MERGE_SPLIT_PAGES", "Ghép nối trang bị tách đôi với trang kế tiếp"),
            ("REMOVE_OVERLAP_AND_MERGE", "Khử vùng trùng lặp với trang kế tiếp"),
        ]
        proposal_act = str(
            (decision.reconstruction_proposal or {}).get(
                "action_type", "SPLIT_SINGLE_PAGE"
            )
        )
        act_codes = [a[0] for a in action_labels]
        def_act_idx = act_codes.index(proposal_act) if proposal_act in act_codes else 0
        sel_act_pair = st.selectbox(
            "Phương thức tái tạo trang:",
            options=action_labels,
            index=def_act_idx,
            format_func=lambda x: x[1],
            key=f"dlg_bnd_act_{pno}",
        )
        chosen_bnd_action = sel_act_pair[0]

    st.divider()
    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Áp dụng thay đổi", type="primary", width="stretch"):
            if chosen_rot_str == "Tự động theo hệ thống":
                st.session_state["manual_overrides"].pop(pno, None)
            else:
                ang_val = int(chosen_rot_str.split("°")[0])
                st.session_state["manual_overrides"][pno] = ang_val
                if has_rotation_issue or ang_val != 0:
                    st.session_state["defect_review_decisions"][
                        f"{pno}:rotation"
                    ] = "MANUAL"

            if has_boundary_edit and chosen_cut_norm is not None:
                st.session_state["boundary_adjustments"][pno] = float(chosen_cut_norm)
                op_dict = _build_boundary_operation_dict(
                    decision=decision,
                    total_pages=total_pages,
                    cut_norm=float(chosen_cut_norm),
                    explicit_action=chosen_bnd_action,
                )
                st.session_state["approved_boundary_ops"][pno] = op_dict
                st.session_state["boundary_feedback"][pno] = "Adjust Boundary"
                st.session_state["defect_review_decisions"][
                    f"{pno}:boundary"
                ] = "MANUAL"
                save_boundary_review_feedback(
                    file_name=file_name,
                    page_number=pno,
                    boundary_status=decision.boundary_status,
                    feedback_action="Adjust Boundary",
                    adjusted_boundary_norm=float(chosen_cut_norm),
                    boundary_defects=decision.boundary_defects,
                )

            st.session_state["has_unexported_changes"] = True
            st.session_state["has_exported_current"] = False
            rebuild_outputs_without_ocr()
            st.session_state["preview_tab"] = "PROPOSED"
            st.session_state["action_feedback_msg"] = (
                f"Đã áp dụng chỉnh sửa thủ công cho Trang {pno}."
            )
            st.rerun()

    with col_cancel:
        if st.button("Đóng", width="stretch"):
            st.rerun()

    # Tùy chọn phụ tải báo cáo JSON/CSV kiểm định không làm ảnh hưởng bố cục chính
    result_obj: Optional[PipelineResult] = st.session_state.get("pipeline_result")
    if result_obj is not None:
        with st.expander("Tải báo cáo kiểm định chi tiết (JSON / CSV)"):
            rc1, rc2 = st.columns(2)
            with rc1:
                st.download_button(
                    "Tải JSON Report",
                    data=ReportGenerator.to_json_string(result_obj.report_data).encode(
                        "utf-8"
                    ),
                    file_name="orientation_report.json",
                    mime="application/json",
                    width="stretch",
                )
            with rc2:
                st.download_button(
                    "Tải CSV Report",
                    data=ReportGenerator.to_csv_string(result_obj.decisions).encode(
                        "utf-8"
                    ),
                    file_name="orientation_report.csv",
                    mime="text/csv",
                    width="stretch",
                )


# ==============================================================================
# XỬ LÝ SỰ KIỆN NÚT CHẤP NHẬN / BỎ QUA
# ==============================================================================


def _apply_accept_action(
    row: PageIssueRow,
    active_item: DefectViewItem,
    file_name: str,
    total_pages: int,
) -> None:
    """Thực thi hành động 'Chấp nhận' thật cho lỗi đang chọn trên trang."""
    pno = row.page_number
    d = row.decision

    if active_item.key == "boundary":
        cut_norm = float(
            st.session_state.get("boundary_adjustments", {}).get(
                pno, get_default_boundary_cut_norm(d)
            )
        )
        op_dict = _build_boundary_operation_dict(
            decision=d, total_pages=total_pages, cut_norm=cut_norm
        )
        st.session_state["approved_boundary_ops"][pno] = op_dict
        st.session_state["boundary_feedback"][pno] = "Confirm"
        save_boundary_review_feedback(
            file_name=file_name,
            page_number=pno,
            boundary_status=d.boundary_status,
            feedback_action="Confirm",
            adjusted_boundary_norm=cut_norm,
            boundary_defects=d.boundary_defects,
        )
        st.session_state["defect_review_decisions"][f"{pno}:boundary"] = "ACCEPTED"
        st.session_state["action_feedback_msg"] = (
            f"Đã chấp nhận tái tạo ranh giới cho Trang {pno}."
        )

    elif active_item.key == "rotation":
        target_angle = (
            d.correction_angle
            if d.correction_angle in (90, 180, 270)
            else (d.detected_rotation if d.detected_rotation in (90, 180, 270) else 0)
        )
        st.session_state["manual_overrides"][pno] = int(target_angle)
        st.session_state["defect_review_decisions"][f"{pno}:rotation"] = "ACCEPTED"
        st.session_state["action_feedback_msg"] = (
            f"Đã áp dụng xoay {target_angle}° cho Trang {pno}."
        )

    elif active_item.key == "stripe":
        st.session_state["stripe_feedback"][pno] = "Confirm defect"
        save_stripe_review_feedback(
            file_name=file_name,
            page_number=pno,
            stripe_status=d.stripe_status,
            feedback_action="Confirm defect",
            stripe_defects=d.stripe_defects,
        )
        st.session_state["defect_review_decisions"][f"{pno}:stripe"] = "ACCEPTED"
        st.session_state["action_feedback_msg"] = (
            f"Đã xác nhận lỗi vệt sọc trên Trang {pno} vào báo cáo kiểm định."
        )

    st.session_state["has_unexported_changes"] = True
    st.session_state["has_exported_current"] = False
    rebuild_outputs_without_ocr()


def _apply_skip_action(
    row: PageIssueRow,
    active_item: DefectViewItem,
    file_name: str,
) -> None:
    """Thực thi hành động 'Bỏ qua' thật cho lỗi đang chọn trên trang."""
    pno = row.page_number
    d = row.decision

    if active_item.key == "boundary":
        st.session_state["approved_boundary_ops"].pop(pno, None)
        st.session_state["boundary_feedback"][pno] = "False Positive"
        save_boundary_review_feedback(
            file_name=file_name,
            page_number=pno,
            boundary_status=d.boundary_status,
            feedback_action="False Positive",
            adjusted_boundary_norm=None,
            boundary_defects=d.boundary_defects,
        )
        st.session_state["defect_review_decisions"][f"{pno}:boundary"] = "SKIPPED"
        st.session_state["action_feedback_msg"] = (
            f"Đã bỏ qua đề xuất tách/ghép trên Trang {pno} (giữ nguyên trang)."
        )

    elif active_item.key == "rotation":
        st.session_state["manual_overrides"][pno] = 0
        st.session_state["defect_review_decisions"][f"{pno}:rotation"] = "SKIPPED"
        st.session_state["action_feedback_msg"] = (
            f"Đã bỏ qua xoay Trang {pno} (giữ nguyên góc 0°)."
        )

    elif active_item.key == "stripe":
        st.session_state["stripe_feedback"][pno] = "False positive"
        save_stripe_review_feedback(
            file_name=file_name,
            page_number=pno,
            stripe_status=d.stripe_status,
            feedback_action="False positive",
            stripe_defects=d.stripe_defects,
        )
        st.session_state["defect_review_decisions"][f"{pno}:stripe"] = "SKIPPED"
        st.session_state["action_feedback_msg"] = (
            f"Đã đánh dấu bỏ qua cảnh báo sọc trên Trang {pno}."
        )

    st.session_state["has_unexported_changes"] = True
    st.session_state["has_exported_current"] = False
    rebuild_outputs_without_ocr()


# ==============================================================================
# COMPONENT CHÍNH: PANEL XEM TRƯỚC TRANG LỖI (CỘT PHẢI DASHBOARD)
# ==============================================================================


def render_preview_panel(
    selected_row: Optional[PageIssueRow],
    active_pdf_path: Optional[Path],
    file_hash: Optional[str],
    total_pages: int = 0,
    file_name: str = "document.pdf",
    scan_dpi: int = 160,
) -> None:
    """Hiển thị Panel bên phải: 'Xem trước trang lỗi' chuẩn xác theo thiết kế."""
    preview_tab: str = st.session_state.get("preview_tab", "ORIGINAL")

    with st.container(key="right_preview_card"):
        # 10.1. Header + 2 Tabs 'Bản gốc' / 'Bản đề xuất'
        with st.container(key="preview_hdr_row"):
            h_left, h_right = st.columns([5.4, 4.6], vertical_alignment="center")
            with h_left:
                st.markdown(
                    '<div class="panel-heading">Xem trước trang lỗi</div>',
                    unsafe_allow_html=True,
                )
            with h_right:
                with st.container(horizontal=True, key="preview_tabs_group"):
                    orig_key = (
                        "ptab_act_orig"
                        if preview_tab == "ORIGINAL"
                        else "ptab_inact_orig"
                    )
                    prop_key = (
                        "ptab_act_prop"
                        if preview_tab == "PROPOSED"
                        else "ptab_inact_prop"
                    )
                    if st.button("Bản gốc", key=orig_key):
                        if preview_tab != "ORIGINAL":
                            st.session_state["preview_tab"] = "ORIGINAL"
                            st.rerun()
                    if st.button("Bản đề xuất", key=prop_key):
                        if preview_tab != "PROPOSED":
                            st.session_state["preview_tab"] = "PROPOSED"
                            st.rerun()

        # Trường hợp chưa chọn trang hoặc tài liệu không có lỗi
        if (
            selected_row is None
            or active_pdf_path is None
            or file_hash is None
            or not selected_row.defect_items
        ):
            st.markdown(
                clean_html(
                    """
                    <div class="doc-preview-stage">
                      <div style="text-align:center;color:#70849F;font-size:13.5px;padding:0 20px;">
                        Chọn một trang lỗi ở bảng bên trái để xem ảnh trang và vùng lỗi chi tiết.
                      </div>
                    </div>
                    <div class="error-detail-card">
                      <div class="error-detail-row"><span class="error-detail-label">Trang:</span><span class="error-detail-val">&mdash;</span></div>
                      <div class="error-detail-row"><span class="error-detail-label">Lỗi:</span><span class="error-detail-val">Chưa chọn trang lỗi</span></div>
                      <div class="error-detail-row"><span class="error-detail-label">Trạng thái:</span><span class="error-detail-val">&mdash;</span></div>
                      <div class="error-detail-row"><span class="error-detail-label">Đề xuất:</span><span class="error-detail-val">&mdash;</span></div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            with st.container(key="preview_actions_bar"):
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.button(
                        "Chấp nhận",
                        icon=":material/check:",
                        disabled=True,
                        width="stretch",
                        key="act_accept",
                    )
                with b2:
                    st.button(
                        "Bỏ qua",
                        icon=":material/skip_next:",
                        disabled=True,
                        width="stretch",
                        key="act_skip",
                    )
                with b3:
                    st.button(
                        "Sửa thủ công",
                        icon=":material/edit:",
                        disabled=True,
                        width="stretch",
                        key="act_manual",
                    )
            return

        pno = selected_row.page_number
        items = selected_row.defect_items
        sel_map: Dict[int, str] = st.session_state.setdefault(
            "selected_defect_key_by_page", {}
        )
        cur_def_key = sel_map.get(pno, items[0].key)
        if all(it.key != cur_def_key for it in items):
            cur_def_key = items[0].key
            sel_map[pno] = cur_def_key

        active_item = next(it for it in items if it.key == cur_def_key)

        # 10.2. Vùng ảnh xem trước thật (220px) + Overlay tọa độ thực tế
        if preview_tab == "PROPOSED":
            stage_html = _render_proposed_preview_html(
                pdf_path=active_pdf_path,
                file_hash=file_hash,
                decision=selected_row.decision,
                active_item=active_item,
                total_pages=total_pages,
            )
        else:
            orig_rgb = get_cached_page_preview_rgb(
                pdf_path=active_pdf_path,
                file_hash=file_hash,
                page_number=pno,
                dpi=PREVIEW_DPI,
                extra_rotation=0,
            )
            stage_html = _render_original_preview_html(
                orig_rgb=orig_rgb,
                decision=selected_row.decision,
                active_item=active_item,
                scan_dpi=scan_dpi,
            )
        st.markdown(clean_html(stage_html), unsafe_allow_html=True)

        # Nếu trang có từ 2 lỗi trở lên, hiển thị chip nhỏ gọn để chọn xác nhận riêng từng lỗi
        if len(items) > 1:
            with st.container(horizontal=True, key="multi_defect_chips"):
                for it in items:
                    is_cur = it.key == active_item.key
                    ckey = (
                        f"mdchip_act_{pno}_{it.key}"
                        if is_cur
                        else f"mdchip_inact_{pno}_{it.key}"
                    )
                    if st.button(it.short_chip, key=ckey):
                        sel_map[pno] = it.key
                        st.session_state["action_feedback_msg"] = None
                        st.rerun()

        # Trạng thái hiển thị cho đúng lỗi đang xem trong Error Detail Box
        item_dec = st.session_state.get("defect_review_decisions", {}).get(
            f"{pno}:{active_item.key}"
        )
        if item_dec == "ACCEPTED":
            det_status_txt, det_status_tone = "Đã chấp nhận", "green"
        elif item_dec == "MANUAL":
            det_status_txt, det_status_tone = "Đã sửa thủ công", "green"
        elif item_dec == "SKIPPED":
            det_status_txt, det_status_tone = "Đã bỏ qua", "gray"
        else:
            det_status_txt, det_status_tone = (
                selected_row.status_label,
                selected_row.status_tone,
            )

        # 10.3. Error Detail Box (Đúng 4 hàng: Trang | Lỗi | Trạng thái | Đề xuất)
        st.markdown(
            clean_html(
                f"""
                <div class="error-detail-card">
                  <div class="error-detail-row">
                    <span class="error-detail-label">Trang:</span>
                    <span class="error-detail-val">{pno}</span>
                  </div>
                  <div class="error-detail-row">
                    <span class="error-detail-label">Lỗi:</span>
                    <span class="error-detail-val">{html.escape(active_item.defect_title)}</span>
                  </div>
                  <div class="error-detail-row">
                    <span class="error-detail-label">Trạng thái:</span>
                    <span class="error-detail-val">
                      <span class="status-pill {det_status_tone}">{html.escape(det_status_txt)}</span>
                    </span>
                  </div>
                  <div class="error-detail-row">
                    <span class="error-detail-label">Đề xuất:</span>
                    <span class="error-detail-val">{html.escape(active_item.proposal_text)}</span>
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        # 10.4. Hàng 3 nút hành động: Chấp nhận | Bỏ qua | Sửa thủ công
        with st.container(key="preview_actions_bar"):
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button(
                    "Chấp nhận",
                    icon=":material/check:",
                    disabled=not active_item.can_accept,
                    help=active_item.accept_disabled_reason
                    or "Áp dụng đề xuất xử lý cho lỗi này",
                    width="stretch",
                    key="act_accept",
                ):
                    _apply_accept_action(
                        row=selected_row,
                        active_item=active_item,
                        file_name=file_name,
                        total_pages=total_pages,
                    )
                    st.rerun()

            with b2:
                if st.button(
                    "Bỏ qua",
                    icon=":material/skip_next:",
                    help="Bỏ qua cảnh báo lỗi này và giữ nguyên trang trong file xuất",
                    width="stretch",
                    key="act_skip",
                ):
                    _apply_skip_action(
                        row=selected_row,
                        active_item=active_item,
                        file_name=file_name,
                    )
                    st.rerun()

            with b3:
                if st.button(
                    "Sửa thủ công",
                    icon=":material/edit:",
                    disabled=not active_item.can_manual_edit,
                    help=active_item.manual_disabled_reason
                    or "Tùy chỉnh góc xoay hoặc vị trí đường cắt thủ công",
                    width="stretch",
                    key="act_manual",
                ):
                    _open_manual_edit_dialog(
                        row=selected_row,
                        active_defect_key=active_item.key,
                        file_name=file_name,
                        total_pages=total_pages,
                    )

        # Thông báo ngắn gọn khi phương án tự động bị khóa hoặc vừa áp dụng thao tác
        fb_msg = st.session_state.get("action_feedback_msg")
        if not active_item.can_accept and active_item.accept_disabled_reason:
            st.caption(f"ℹ️ {active_item.accept_disabled_reason}")
        elif not active_item.can_manual_edit and active_item.manual_disabled_reason:
            st.caption(f"ℹ️ {active_item.manual_disabled_reason}")
        elif fb_msg:
            st.caption(f"✓ {fb_msg}")


def render_preview_and_export_section(
    result: PipelineResult,
    active_pdf_path: Path,
    problem_pages: List[PageDecision],
    conf_threshold: float,
    auto_fix_enabled: bool,
    export_json: bool,
    export_csv: bool,
    refresh_callback: Callable[[float, bool, bool, bool], None],
) -> None:
    """Wrapper tương thích ngược với giao diện component cũ."""
    rows = build_problem_page_rows(result)
    sel_row = rows[0] if rows else None
    render_preview_panel(
        selected_row=sel_row,
        active_pdf_path=active_pdf_path,
        file_hash=st.session_state.get("file_hash") or "compat",
        total_pages=result.doc_info.page_count,
        file_name=result.doc_info.file_name,
        scan_dpi=result.config.render_dpi,
    )
