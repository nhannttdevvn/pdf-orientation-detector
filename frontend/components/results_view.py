"""
Component Khu vực 9 (Panel trái ~51%) — 'Kết quả kiểm tra' (`frontend/components/results_view.py`).

Tuân thủ Mục 9:
- Tiêu đề: `Kết quả kiểm tra`, bên phải có icon filter và icon chọn/chế độ hiển thị kèm tooltip.
- 5 Filter dạng pill: `Tất cả (n)`, `Sọc (n)`, `Lẫn trang (n)`, `Xoay (n)`, `Cần rà soát (n)`.
- Bảng dữ liệu 7 cột: `| Chọn | Trang | Sọc | Lẫn | Xoay | Trạng thái | Thao tác |`.
- Chỉ hiển thị trang có lỗi hoặc cần kiểm tra; không liệt kê trang bình thường.
- Giữ nguyên các hàm `build_results_dataframe`, `build_stripe_results_dataframe`, `build_boundary_results_dataframe`
  để tương thích ngược với bộ kiểm thử.
"""

import html
import math
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ai_engine.core.config import (
    BOUNDARY_STATUS_MISSING,
    BOUNDARY_STATUS_MIXED,
    BOUNDARY_STATUS_RECONSTRUCTABLE,
    STRIPE_STATUS_DETECTED,
    STRIPE_STATUS_ERROR,
)
from backend.schemas.schemas import PageDecision, PipelineResult
from frontend.state.state_manager import (
    PageIssueRow,
    build_problem_page_rows,
    rebuild_outputs_without_ocr,
)
from frontend.styles.styles import SVG_TABLE_CHECK_GREEN, SVG_TABLE_SUSPECT_ORANGE, clean_html

ROWS_PER_PAGE: int = 7


# ==============================================================================
# CÁC HÀM DATAFRAME TƯƠNG THÍCH NGƯỢC
# ==============================================================================


def build_results_dataframe(decisions: List[PageDecision]) -> pd.DataFrame:
    """Tạo DataFrame tổng hợp kết quả từng trang (giữ tương thích ngược)."""
    rows = []
    for d in decisions:
        rot_display = (
            f"{d.detected_rotation}°" if d.detected_rotation is not None else "Unknown"
        )
        conf_display = (
            f"{d.orientation_conf:.1f}" if d.orientation_conf is not None else "N/A"
        )
        defect_labels = (
            ", ".join(f"{df.defect_type}:{df.status}" for df in d.defects)
            if d.defects
            else "NO_DEFECT"
        )
        rows.append(
            {
                "Page": d.page_number,
                "Status": d.status,
                "Stripe Status": d.stripe_status,
                "Boundary Status": d.boundary_status,
                "Inspection Status": d.inspection_status,
                "Rotation": rot_display,
                "Confidence": conf_display,
                "Action": d.action_label,
                "Correction Angle": f"{d.correction_angle}°",
                "Corrected": "Yes" if d.corrected else "No",
                "Defects": defect_labels,
                "Existing PDF Rotation": f"{d.existing_pdf_rotation}°",
                "Method": d.detection_method,
                "Note / Error": d.error or "",
            }
        )
    return pd.DataFrame(rows)


def build_stripe_results_dataframe(
    decisions: List[PageDecision],
    feedback_by_page: Dict[int, str],
) -> pd.DataFrame:
    """Tạo DataFrame chi tiết Stripe (giữ tương thích ngược)."""
    rows = []
    for d in decisions:
        user_fb = feedback_by_page.get(d.page_number)
        if d.stripe_defects:
            for s in d.stripe_defects:
                score_val = (
                    s.suspicion_score
                    if s.suspicion_score is not None
                    else (s.confidence or 0.0)
                )
                default_action = (
                    "Inspect / Re-scan page"
                    if d.stripe_status == STRIPE_STATUS_DETECTED
                    else "Manual visual review"
                )
                rows.append(
                    {
                        "Page number": d.page_number,
                        "Defect type": s.defect_type,
                        "Stripe type": s.stripe_type or "N/A",
                        "Angle": (
                            f"{s.angle_deg:.1f}°" if s.angle_deg is not None else "N/A"
                        ),
                        "Relative length": (
                            f"{s.relative_length:.2f}"
                            if s.relative_length is not None
                            else "N/A"
                        ),
                        "Width (px)": (
                            f"{s.width_px:.1f}" if s.width_px is not None else "N/A"
                        ),
                        "Confidence / suspicion score": f"{score_val:.2f}",
                        "Status": d.stripe_status,
                        "Review action": (
                            f"Feedback: {user_fb}" if user_fb else default_action
                        ),
                        "Reason": s.reason or "",
                    }
                )
        else:
            rows.append(
                {
                    "Page number": d.page_number,
                    "Defect type": "NONE",
                    "Stripe type": "NONE",
                    "Angle": "-",
                    "Relative length": "0.00",
                    "Width (px)": "-",
                    "Confidence / suspicion score": "0.00",
                    "Status": d.stripe_status,
                    "Review action": (
                        f"Feedback: {user_fb}"
                        if user_fb
                        else (
                            "Check error"
                            if d.stripe_status == STRIPE_STATUS_ERROR
                            else "None"
                        )
                    ),
                    "Reason": "Không phát hiện vệt sọc bất thường.",
                }
            )
    return pd.DataFrame(rows)


def build_boundary_results_dataframe(
    decisions: List[PageDecision],
    boundary_feedback_by_page: Dict[int, str],
) -> pd.DataFrame:
    """Tạo DataFrame chi tiết Page Boundary (giữ tương thích ngược)."""
    rows = []
    for d in decisions:
        user_fb = boundary_feedback_by_page.get(d.page_number)
        if d.boundary_defects:
            for b in d.boundary_defects:
                details = b.details or {}
                rel_pages = details.get("related_pages", [])
                rel_str = (
                    ", ".join(str(p) for p in rel_pages) if rel_pages else "None"
                )
                bnd_norm = details.get("suspected_boundary_norm")
                bnd_str = f"y={bnd_norm:.3f}" if bnd_norm is not None else "-"
                score_val = (
                    b.suspicion_score
                    if b.suspicion_score is not None
                    else (b.confidence or 0.0)
                )
                default_action = (
                    "REQUIRES_RESCAN"
                    if d.boundary_status == BOUNDARY_STATUS_MISSING
                    else (
                        "PREVIEW_RECONSTRUCTION"
                        if d.boundary_status
                        in (BOUNDARY_STATUS_MIXED, BOUNDARY_STATUS_RECONSTRUCTABLE)
                        else "MANUAL_REVIEW"
                    )
                )
                rows.append(
                    {
                        "Page": d.page_number,
                        "Defect Type": b.defect_type,
                        "Related Pages": rel_str,
                        "Suspected Boundary": bnd_str,
                        "Score": f"{score_val:.2f}",
                        "Status": d.boundary_status,
                        "Action": f"User: {user_fb}" if user_fb else default_action,
                        "Reason": b.reason or "",
                    }
                )
        else:
            rows.append(
                {
                    "Page": d.page_number,
                    "Defect Type": "NONE",
                    "Related Pages": "-",
                    "Suspected Boundary": "-",
                    "Score": "0.00",
                    "Status": d.boundary_status,
                    "Action": f"User: {user_fb}" if user_fb else "KEEP_AS_IS",
                    "Reason": "Bố cục trang bình thường.",
                }
            )
    return pd.DataFrame(rows)


def render_results_section(result: PipelineResult) -> List[PageDecision]:
    """Wrapper tương thích ngược trả về danh sách `PageDecision` của các trang lỗi."""
    issue_rows = build_problem_page_rows(result)
    render_error_table_panel(issue_rows, has_scanned=True)
    return [r.decision for r in issue_rows]


# ==============================================================================
# COMPONENT CHÍNH: BẢNG KẾT QUẢ KIỂM TRA (PANEL TRÁI MỤC 9)
# ==============================================================================


SVG_CHK_RED = """
<svg width="18" height="18" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4.5 10.5L8.2 14.2L15.8 6.2" stroke="#E93D4D" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_CHK_ORANGE = """
<svg width="18" height="18" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4.5 10.5L8.2 14.2L15.8 6.2" stroke="#E98914" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_CHK_BLUE = """
<svg width="18" height="18" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4.5 10.5L8.2 14.2L15.8 6.2" stroke="#0866E8" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_PILL_DONE = """
<svg width="13" height="13" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <circle cx="8" cy="8" r="7" fill="#15965D"/>
  <path d="M5 8.2L7 10.2L11.2 5.8" stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_PILL_REVIEW = """
<svg width="13" height="13" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <circle cx="7" cy="7" r="4.8" stroke="#D97706" stroke-width="1.7"/>
  <line x1="10.6" y1="10.6" x2="14" y2="14" stroke="#D97706" stroke-width="1.8" stroke-linecap="round"/>
</svg>
"""

SVG_PILL_ERR = """
<svg width="13" height="13" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <circle cx="8" cy="8" r="7" fill="#E93D4D"/>
  <line x1="8" y1="4.5" x2="8" y2="8.8" stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round"/>
  <circle cx="8" cy="11.4" r="1" fill="#FFFFFF"/>
</svg>
"""


def _render_defect_indicator_html(state: str, tooltip: str, kind: str = "stripe") -> str:
    """Sinh HTML hiển thị ô Sọc / Lẫn / Xoay theo đúng mẫu: dấu check màu đặc trưng hoặc gạch ngang."""
    if state == "NONE":
        return '<span class="tbl-dash">&mdash;</span>'
    safe_tip = html.escape(tooltip or "")
    icon_svg = SVG_CHK_RED if kind == "stripe" else (SVG_CHK_ORANGE if kind == "boundary" else SVG_CHK_BLUE)
    return clean_html(f'<span class="tbl-icon-tip" title="{safe_tip}">{icon_svg}</span>')


def _render_status_pill_html(tone: str, label: str) -> str:
    """Sinh HTML hiển thị pill Trạng thái có icon kèm nhãn khớp ảnh tham chiếu."""
    safe_lbl = html.escape("Lỗi" if label == "Lỗi kiểm tra" else label)
    if tone == "green":
        return clean_html(f'<span class="status-pill green">{SVG_PILL_DONE}<span>{safe_lbl}</span></span>')
    if tone == "orange":
        return clean_html(f'<span class="status-pill orange">{SVG_PILL_REVIEW}<span>{safe_lbl}</span></span>')
    if tone == "red":
        return clean_html(f'<span class="status-pill red">{SVG_PILL_ERR}<span>{safe_lbl}</span></span>')
    return clean_html(f'<span class="status-pill gray"><span>{safe_lbl}</span></span>')


def _filter_rows(rows: List[PageIssueRow], filter_key: str) -> List[PageIssueRow]:
    if filter_key == "STRIPE":
        return [r for r in rows if r.stripe_state != "NONE"]
    if filter_key == "BOUNDARY":
        return [r for r in rows if r.boundary_state != "NONE"]
    if filter_key == "ROTATION":
        return [r for r in rows if r.rotation_state != "NONE"]
    if filter_key == "REVIEW":
        return [r for r in rows if r.needs_review_now]
    return rows


def _filter_tagged_rows(
    tagged: List[Tuple[Optional[str], Optional[str], PageIssueRow]],
    filter_key: str,
) -> List[Tuple[Optional[str], Optional[str], PageIssueRow]]:
    keep = {id(r) for r in _filter_rows([t[2] for t in tagged], filter_key)}
    return [t for t in tagged if id(t[2]) in keep]


def _default_page_select(_file_key: Optional[str], page_number: int) -> None:
    st.session_state["selected_page_num"] = page_number
    st.session_state["preview_view_rotation_deg"] = 0


def render_error_table_panel(
    problem_rows: List[PageIssueRow],
    has_scanned: bool,
    total_doc_pages: int = 0,
    row_files: Optional[List[Tuple[str, str]]] = None,
    selected_file_key: Optional[str] = None,
    on_select: Optional[Callable[[Optional[str], int], None]] = None,
    render_file_filter: Optional[Callable[[], None]] = None,
    empty_message: Optional[Tuple[str, str]] = None,
) -> None:
    """
    Hiển thị Panel bên trái (Mục 9): 'Kết quả kiểm tra' gồm:
    - Tiêu đề + Icon filter & Icon chọn hàng loạt bên phải
    - 5 Filter Pills: Tất cả (n) | Sọc (n) | Lẫn trang (n) | Xoay (n) | Cần rà soát (n)
    - Bảng 7 cột chuẩn xác: | Chọn | Trang | Sọc | Lẫn | Xoay | Trạng thái | Thao tác |
    """
    multi_file = row_files is not None
    tagged: List[Tuple[Optional[str], Optional[str], PageIssueRow]] = (
        [(fk, lbl, r) for (fk, lbl), r in zip(row_files, problem_rows)]
        if multi_file
        else [(None, None, r) for r in problem_rows]
    )
    select_fn = on_select or _default_page_select

    n_all = len(tagged)
    n_stripe = sum(1 for t in tagged if t[2].stripe_state != "NONE")
    n_boundary = sum(1 for t in tagged if t[2].boundary_state != "NONE")
    n_rotation = sum(1 for t in tagged if t[2].rotation_state != "NONE")
    n_review = sum(1 for t in tagged if t[2].needs_review_now)

    active_filter: str = st.session_state.get("active_filter", "ALL")
    filter_defs = [
        ("ALL", f"Tất cả ({n_all})"),
        ("STRIPE", f"Sọc ({n_stripe})"),
        ("BOUNDARY", f"Lẫn trang ({n_boundary})"),
        ("ROTATION", f"Xoay ({n_rotation})"),
        ("REVIEW", f"Cần rà soát ({n_review})"),
    ]
    filtered_rows = _filter_tagged_rows(tagged, active_filter)
    selected_keys_list: List[str] = list(st.session_state.get("selected_row_keys", []))

    with st.container(key="left_error_card"):
        # Tiêu đề 'Kết quả kiểm tra' + 2 icon bên phải
        hdr_l, hdr_r = st.columns([7.8, 2.2], vertical_alignment="center")
        with hdr_l:
            st.markdown(
                '<div class="panel-heading">Kết quả kiểm tra</div>',
                unsafe_allow_html=True,
            )
        with hdr_r:
            with st.container(horizontal=True, vertical_alignment="center", key="left_panel_hdr_icons"):
                if st.button(
                    "",
                    icon=":material/filter_alt:",
                    key="btn_tbl_hdr_reset_flt",
                    help="Đặt lại bộ lọc về Tất cả",
                ):
                    if active_filter != "ALL":
                        st.session_state["active_filter"] = "ALL"
                        st.session_state["table_page_idx"] = 0
                        st.rerun()
                if st.button(
                    "",
                    icon=":material/grid_view:",
                    key="btn_tbl_hdr_toggle_all",
                    disabled=(not filtered_rows),
                    help="Chọn / Bỏ chọn tất cả các trang trong danh sách hiện tại",
                ):
                    all_vis_keys = [
                        f"{fk}_{r.page_number}" if fk else f"{r.page_number}"
                        for fk, _lbl, r in filtered_rows
                    ]
                    if all_vis_keys and all(k in selected_keys_list for k in all_vis_keys):
                        st.session_state["selected_row_keys"] = []
                    else:
                        st.session_state["selected_row_keys"] = all_vis_keys
                    st.rerun()

        if render_file_filter is not None:
            render_file_filter()

        # Hàng 5 bộ lọc dạng pill
        with st.container(horizontal=True, key="filter_pills_bar"):
            for f_id, f_label in filter_defs:
                is_act = active_filter == f_id
                btn_key = (
                    f"flt_act_{f_id.lower()}"
                    if is_act
                    else f"flt_inact_{f_id.lower()}"
                )
                if st.button(f_label, key=btn_key):
                    if active_filter != f_id:
                        st.session_state["active_filter"] = f_id
                        st.session_state["table_page_idx"] = 0
                        st.session_state["action_feedback_msg"] = None
                        new_filtered = _filter_tagged_rows(tagged, f_id)
                        cur_sel = st.session_state.get("selected_page_num")
                        if new_filtered and all(
                            not ((fk is None or fk == selected_file_key) and r.page_number == cur_sel)
                            for fk, _lbl, r in new_filtered
                        ):
                            select_fn(new_filtered[0][0], new_filtered[0][2].page_number)
                        st.rerun()

        # Thanh thao tác hàng loạt khi người dùng tích chọn checkbox
        if selected_keys_list and filtered_rows:
            with st.container(horizontal=True, vertical_alignment="center", key="batch_selection_bar"):
                st.markdown(
                    f'<span style="font-size:12.5px;font-weight:600;color:#0866E8;">Đã chọn {len(selected_keys_list)} trang:</span>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Bỏ qua đã chọn",
                    icon=":material/skip_next:",
                    key="btn_batch_skip_selected",
                    help="Đánh dấu bỏ qua cảnh báo cho các trang đang tích chọn",
                ):
                    for fk, _lbl, r in filtered_rows:
                        sfx = f"{fk}_{r.page_number}" if fk else f"{r.page_number}"
                        if sfx in selected_keys_list:
                            for it in r.defect_items:
                                st.session_state["defect_review_decisions"][f"{r.page_number}:{it.key}"] = "SKIPPED"
                            if r.rotation_state != "NONE":
                                st.session_state["manual_overrides"][r.page_number] = 0
                    st.session_state["selected_row_keys"] = []
                    st.session_state["has_unexported_changes"] = True
                    rebuild_outputs_without_ocr()
                    st.rerun()

        col_weights = (
            [0.52, 1.55, 0.78, 0.75, 0.75, 0.80, 1.48, 1.15]
            if multi_file
            else [0.52, 0.85, 0.82, 0.82, 0.85, 1.55, 1.18]
        )

        with st.container(key="error_table_wrapper"):
            # Header bảng 7 cột: | ☐ | Trang ↕ | Sọc | Lẫn | Xoay | Trạng thái | Thao tác |
            with st.container(key="tbl_hdr_row"):
                h_cols = st.columns(col_weights, gap=None, vertical_alignment="center")
                with h_cols[0]:
                    with st.container(key="tbl_chk_cell_hdr"):
                        all_vis_keys = [
                            f"{fk}_{r.page_number}" if fk else f"{r.page_number}"
                            for fk, _lbl, r in filtered_rows
                        ]
                        hdr_checked = bool(all_vis_keys and all(k in selected_keys_list for k in all_vis_keys))
                        new_hdr_chk = st.checkbox(
                            "Chọn tất cả",
                            value=hdr_checked,
                            key="chk_hdr_all_rows",
                            label_visibility="collapsed",
                            disabled=(not filtered_rows),
                        )
                        if new_hdr_chk != hdr_checked:
                            st.session_state["selected_row_keys"] = all_vis_keys if new_hdr_chk else []
                            st.rerun()

                headers = ["Trang &#8597;", "Sọc", "Lẫn", "Xoay", "Trạng thái", "Thao tác"]
                if multi_file:
                    headers = ["Tệp PDF", "Trang &#8597;", "Sọc", "Lẫn", "Xoay", "Trạng thái", "Thao tác"]
                for col_obj, h_text in zip(h_cols[1:], headers):
                    col_obj.markdown(
                        f'<div class="tbl-hdr-cell">{h_text}</div>',
                        unsafe_allow_html=True,
                    )

            # Empty States hoặc Data Rows
            if empty_message is not None and not tagged:
                st.markdown(
                    clean_html(
                        f"""
                        <div class="tbl-empty-state">
                          <div class="tbl-empty-title">{html.escape(empty_message[0])}</div>
                          <div class="tbl-empty-sub">{html.escape(empty_message[1])}</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
            elif not has_scanned:
                st.markdown(
                    clean_html(
                        """
                        <div class="tbl-empty-state">
                          <div class="tbl-empty-title">Chưa có dữ liệu kiểm tra</div>
                          <div class="tbl-empty-sub">Tải lên một file PDF hoặc thư mục ở khung phía trên để hệ thống tự động quét và liệt kê các trang cần xử lý.</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
            elif not tagged:
                st.markdown(
                    clean_html(
                        f"""
                        <div class="tbl-empty-state">
                          <div class="tbl-empty-title" style="color:#15965D;">Không phát hiện trang lỗi nào</div>
                          <div class="tbl-empty-sub">Toàn bộ {total_doc_pages} trang của tài liệu đều bình thường (không bị sọc, không lẫn trang, đúng hướng xoay).</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
            elif not filtered_rows:
                st.markdown(
                    clean_html(
                        """
                        <div class="tbl-empty-state">
                          <div class="tbl-empty-title">Không có trang nào thuộc bộ lọc này</div>
                          <div class="tbl-empty-sub">Hãy chọn bộ lọc "Tất cả" hoặc nhóm lỗi khác để xem các trang cần xử lý.</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
            else:
                total_filtered = len(filtered_rows)
                total_tbl_pages = max(1, math.ceil(total_filtered / ROWS_PER_PAGE))
                cur_pg_idx = int(st.session_state.get("table_page_idx", 0))
                cur_pg_idx = max(0, min(total_tbl_pages - 1, cur_pg_idx))
                st.session_state["table_page_idx"] = cur_pg_idx

                start_i = cur_pg_idx * ROWS_PER_PAGE
                end_i = min(total_filtered, start_i + ROWS_PER_PAGE)
                visible_rows = filtered_rows[start_i:end_i]

                selected_page_num = st.session_state.get("selected_page_num")

                for file_key, file_label, row in visible_rows:
                    pno = row.page_number
                    is_selected = pno == selected_page_num and (
                        file_key is None or file_key == selected_file_key
                    )
                    sfx = f"{file_key}_{pno}" if file_key else f"{pno}"
                    row_key = (
                        f"tbl_row_sel_{sfx}" if is_selected else f"tbl_row_norm_{sfx}"
                    )

                    stripe_html = _render_defect_indicator_html(
                        row.stripe_state, row.stripe_tooltip, kind="stripe"
                    )
                    boundary_html = _render_defect_indicator_html(
                        row.boundary_state, row.boundary_tooltip, kind="boundary"
                    )
                    rot_html = _render_defect_indicator_html(
                        row.rotation_state,
                        f"{row.rotation_display} · {row.rotation_tooltip}".strip(" ·"),
                        kind="rotation",
                    )
                    status_html = _render_status_pill_html(row.status_tone, row.status_label)

                    with st.container(key=row_key):
                        r_cols = st.columns(
                            col_weights, gap=None, vertical_alignment="center"
                        )
                        # Cột 1: Chọn (Checkbox)
                        with r_cols[0]:
                            with st.container(key=f"tbl_chk_cell_{sfx}"):
                                is_chk = sfx in selected_keys_list
                                new_chk = st.checkbox(
                                    f"Chọn trang {pno}",
                                    value=is_chk,
                                    key=f"chk_row_item_{sfx}",
                                    label_visibility="collapsed",
                                )
                                if new_chk and not is_chk:
                                    selected_keys_list.append(sfx)
                                    st.session_state["selected_row_keys"] = selected_keys_list
                                    st.rerun()
                                elif not new_chk and is_chk:
                                    selected_keys_list.remove(sfx)
                                    st.session_state["selected_row_keys"] = selected_keys_list
                                    st.rerun()

                        rem_cols = r_cols[1:]
                        if multi_file:
                            safe_lbl = html.escape(file_label or "")
                            rem_cols[0].markdown(
                                f'<div class="tbl-data-cell tbl-file-cell" title="{safe_lbl}">{safe_lbl}</div>',
                                unsafe_allow_html=True,
                            )
                            rem_cols = rem_cols[1:]

                        rem_cols[0].markdown(
                            f'<div class="tbl-data-cell"><span class="tbl-page-num">{pno}</span></div>',
                            unsafe_allow_html=True,
                        )
                        rem_cols[1].markdown(
                            f'<div class="tbl-data-cell">{stripe_html}</div>',
                            unsafe_allow_html=True,
                        )
                        rem_cols[2].markdown(
                            f'<div class="tbl-data-cell">{boundary_html}</div>',
                            unsafe_allow_html=True,
                        )
                        rem_cols[3].markdown(
                            f'<div class="tbl-data-cell">{rot_html}</div>',
                            unsafe_allow_html=True,
                        )
                        rem_cols[4].markdown(
                            f'<div class="tbl-data-cell">{status_html}</div>',
                            unsafe_allow_html=True,
                        )
                        with rem_cols[5]:
                            with st.container(horizontal=True, key=f"tbl_action_cell_{sfx}"):
                                if st.button(
                                    "",
                                    icon=":material/visibility:",
                                    key=f"btn_view_p_{sfx}",
                                    help=f"Xem trang {pno}",
                                ):
                                    select_fn(file_key, pno)
                                    st.session_state["action_feedback_msg"] = None
                                    if active_filter == "BOUNDARY" and any(
                                        it.key == "boundary" for it in row.defect_items
                                    ):
                                        st.session_state["selected_defect_key_by_page"][pno] = "boundary"
                                    elif active_filter == "STRIPE" and any(
                                        it.key == "stripe" for it in row.defect_items
                                    ):
                                        st.session_state["selected_defect_key_by_page"][pno] = "stripe"
                                    elif active_filter == "ROTATION" and any(
                                        it.key == "rotation" for it in row.defect_items
                                    ):
                                        st.session_state["selected_defect_key_by_page"][pno] = "rotation"
                                    st.rerun()
                                if st.button(
                                    "",
                                    icon=":material/edit:",
                                    key=f"btn_edit_p_{sfx}",
                                    help=f"Sửa thủ công trang {pno}",
                                ):
                                    select_fn(file_key, pno)
                                    st.session_state["open_manual_edit_page"] = pno
                                    st.rerun()

        # Thanh phân trang gọn cuối bảng khi nhiều hàng
        if has_scanned and len(filtered_rows) > ROWS_PER_PAGE:
            total_filtered = len(filtered_rows)
            total_tbl_pages = max(1, math.ceil(total_filtered / ROWS_PER_PAGE))
            cur_pg_idx = int(st.session_state.get("table_page_idx", 0))
            with st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
                key="tbl_pagination_bar",
            ):
                if st.button(
                    "‹ Trước",
                    disabled=(cur_pg_idx <= 0),
                    key="pg_nav_prev",
                ):
                    st.session_state["table_page_idx"] = max(0, cur_pg_idx - 1)
                    st.rerun()

                st.markdown(
                    f'<span style="font-size:12.5px;color:#667892;font-weight:500;">'
                    f"Trang bảng {cur_pg_idx + 1}/{total_tbl_pages} &middot; "
                    f"Tổng {total_filtered} trang cần xử lý</span>",
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Tiếp ›",
                    disabled=(cur_pg_idx >= total_tbl_pages - 1),
                    key="pg_nav_next",
                ):
                    st.session_state["table_page_idx"] = min(
                        total_tbl_pages - 1, cur_pg_idx + 1
                    )
                    st.rerun()
