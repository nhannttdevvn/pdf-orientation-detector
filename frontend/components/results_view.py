"""
Component Khu vực 4 (Cột trái ~61.5%) — Bảng 'Danh sách trang có lỗi' (`src/pdf_inspector/ui/components/results_view.py`).

Đảm bảo:
- Đúng 6 cột: Trang | Lỗi sọc | Lẫn trang | Xoay | Trạng thái | Xem.
- Chỉ hiển thị trang có lỗi hoặc cần kiểm tra; ẩn toàn bộ trang bình thường.
- Hàng bộ lọc pill: Tất cả | Sọc | Lẫn trang | Xoay (chuyển bộ lọc không chạy lại OCR).
- Phân biệt rõ lỗi đã xác nhận (dấu check xanh lá) và nghi vấn độ tin cậy thấp (dấu hỏi cam kèm tooltip).
- Hỗ trợ phân trang gọn gàng khi có trên 5 trang lỗi và Empty State khi không có lỗi.
- Giữ các hàm `build_results_dataframe`, `build_stripe_results_dataframe`, `build_boundary_results_dataframe`
  để tương thích ngược với bộ kiểm thử hiện có.
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
from frontend.state.state_manager import PageIssueRow, build_problem_page_rows
from frontend.styles.styles import SVG_TABLE_CHECK_GREEN, SVG_TABLE_SUSPECT_ORANGE, clean_html

ROWS_PER_PAGE: int = 5


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
# COMPONENT CHÍNH: BẢNG DANH SÁCH TRANG CÓ LỖI (CỘT TRÁI DASHBOARD)
# ==============================================================================


def _render_defect_indicator_html(state: str, tooltip: str) -> str:
    """Sinh HTML hiển thị ô Lỗi sọc / Lẫn trang: check xanh (xác nhận), ? cam (nghi vấn), hoặc gạch ngang."""
    safe_tip = html.escape(tooltip or "")
    if state == "CONFIRMED":
        return clean_html(f'<span class="tbl-icon-tip" title="{safe_tip}">{SVG_TABLE_CHECK_GREEN}</span>')
    if state == "SUSPECTED":
        return clean_html(f'<span class="tbl-icon-tip" title="{safe_tip}">{SVG_TABLE_SUSPECT_ORANGE}</span>')
    return '<span class="tbl-dash">&mdash;</span>'


def _filter_rows(rows: List[PageIssueRow], filter_key: str) -> List[PageIssueRow]:
    if filter_key == "STRIPE":
        return [r for r in rows if r.stripe_state != "NONE"]
    if filter_key == "BOUNDARY":
        return [r for r in rows if r.boundary_state != "NONE"]
    if filter_key == "ROTATION":
        return [r for r in rows if r.rotation_state != "NONE"]
    return rows


def _filter_tagged_rows(
    tagged: List[Tuple[Optional[str], Optional[str], PageIssueRow]],
    filter_key: str,
) -> List[Tuple[Optional[str], Optional[str], PageIssueRow]]:
    """Áp dụng đúng bộ lọc hiện có (`_filter_rows`) lên danh sách dòng có gắn tệp PDF."""
    keep = {id(r) for r in _filter_rows([t[2] for t in tagged], filter_key)}
    return [t for t in tagged if id(t[2]) in keep]


def _default_page_select(_file_key: Optional[str], page_number: int) -> None:
    st.session_state["selected_page_num"] = page_number


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
    Hiển thị Panel bên trái: 'Danh sách trang có lỗi' gồm:
    - Hàng Filter Pills: Tất cả | Sọc | Lẫn trang | Xoay
    - Bảng 6 cột chuẩn xác: Trang | Lỗi sọc | Lẫn trang | Xoay | Trạng thái | Xem
    - Highlight xanh nhạt toàn bộ hàng đang được chọn
    - Phân trang khi số hàng > 5
    Khi quét nhiều PDF (`row_files` = (mã tệp, đường dẫn hiển thị) song song với `problem_rows`):
    thêm bộ lọc 'Tệp PDF' (`render_file_filter`) và cột 'Tệp PDF'; chọn dòng gọi `on_select(mã tệp, trang)`.
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

    active_filter: str = st.session_state.get("active_filter", "ALL")
    filter_defs = [
        ("ALL", f"Tất cả ({n_all})"),
        ("STRIPE", f"Sọc ({n_stripe})"),
        ("BOUNDARY", f"Lẫn trang ({n_boundary})"),
        ("ROTATION", f"Xoay ({n_rotation})"),
    ]

    with st.container(key="left_error_card"):
        st.markdown(
            '<div class="panel-heading">Danh sách trang có lỗi</div>',
            unsafe_allow_html=True,
        )

        if render_file_filter is not None:
            render_file_filter()

        # Hàng bộ lọc dạng pill
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

        filtered_rows = _filter_tagged_rows(tagged, active_filter)
        col_weights = (
            [2.0, 0.80, 1.0, 1.05, 0.92, 1.62, 1.18]
            if multi_file
            else [0.90, 1.12, 1.18, 1.02, 1.82, 1.32]
        )

        with st.container(key="error_table_wrapper"):
            # Header bảng 6 cột
            with st.container(key="tbl_hdr_row"):
                h_cols = st.columns(col_weights, gap=None, vertical_alignment="center")
                headers = ["Trang", "Lỗi sọc", "Lẫn trang", "Xoay", "Trạng thái", "Xem"]
                if multi_file:
                    headers = ["Tệp PDF"] + headers
                for col_obj, h_text in zip(h_cols, headers):
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
                          <div class="tbl-empty-sub">Tải lên một file PDF ở khung phía trên để hệ thống tự động quét và liệt kê các trang có lỗi tại đây.</div>
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
                          <div class="tbl-empty-title" style="color:#079B60;">Không phát hiện trang lỗi nào</div>
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
                          <div class="tbl-empty-sub">Hãy chọn bộ lọc "Tất cả" hoặc nhóm lỗi khác để xem các trang cần kiểm tra.</div>
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
                        row.stripe_state, row.stripe_tooltip
                    )
                    boundary_html = _render_defect_indicator_html(
                        row.boundary_state, row.boundary_tooltip
                    )
                    if row.rotation_state == "NONE":
                        rot_html = '<span class="tbl-dash">&mdash;</span>'
                    else:
                        safe_rtip = html.escape(row.rotation_tooltip)
                        rot_html = (
                            f'<span class="tbl-rot-txt" title="{safe_rtip}">'
                            f"{html.escape(row.rotation_display)}</span>"
                        )

                    status_html = (
                        f'<span class="status-pill {row.status_tone}">'
                        f"{html.escape(row.status_label)}</span>"
                    )

                    with st.container(key=row_key):
                        r_cols = st.columns(
                            col_weights, gap=None, vertical_alignment="center"
                        )
                        if multi_file:
                            safe_lbl = html.escape(file_label or "")
                            r_cols[0].markdown(
                                f'<div class="tbl-data-cell tbl-file-cell" title="{safe_lbl}">{safe_lbl}</div>',
                                unsafe_allow_html=True,
                            )
                            r_cols = r_cols[1:]
                        r_cols[0].markdown(
                            f'<div class="tbl-data-cell"><span class="tbl-page-num">{pno}</span></div>',
                            unsafe_allow_html=True,
                        )
                        r_cols[1].markdown(
                            f'<div class="tbl-data-cell">{stripe_html}</div>',
                            unsafe_allow_html=True,
                        )
                        r_cols[2].markdown(
                            f'<div class="tbl-data-cell">{boundary_html}</div>',
                            unsafe_allow_html=True,
                        )
                        r_cols[3].markdown(
                            f'<div class="tbl-data-cell">{rot_html}</div>',
                            unsafe_allow_html=True,
                        )
                        r_cols[4].markdown(
                            f'<div class="tbl-data-cell">{status_html}</div>',
                            unsafe_allow_html=True,
                        )
                        with r_cols[5]:
                            with st.container(key=f"tbl_action_cell_{sfx}"):
                                if st.button(
                                    "Xem",
                                    icon=":material/visibility:",
                                    key=f"btn_view_p_{sfx}",
                                ):
                                    select_fn(file_key, pno)
                                    st.session_state["action_feedback_msg"] = None
                                    # Nếu bộ lọc đang chọn 1 loại lỗi cụ thể, ưu tiên mở đúng lỗi đó
                                    if active_filter == "BOUNDARY" and any(
                                        it.key == "boundary" for it in row.defect_items
                                    ):
                                        st.session_state[
                                            "selected_defect_key_by_page"
                                        ][pno] = "boundary"
                                    elif active_filter == "STRIPE" and any(
                                        it.key == "stripe" for it in row.defect_items
                                    ):
                                        st.session_state[
                                            "selected_defect_key_by_page"
                                        ][pno] = "stripe"
                                    elif active_filter == "ROTATION" and any(
                                        it.key == "rotation" for it in row.defect_items
                                    ):
                                        st.session_state[
                                            "selected_defect_key_by_page"
                                        ][pno] = "rotation"
                                    st.rerun()

        # Thanh phân trang nhỏ gọn khi có nhiều hơn 5 trang lỗi
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
                    "‹ Trang trước",
                    disabled=(cur_pg_idx <= 0),
                    key="pg_nav_prev",
                ):
                    st.session_state["table_page_idx"] = max(0, cur_pg_idx - 1)
                    st.rerun()

                st.markdown(
                    f'<span style="font-size:13px;color:#70849F;font-weight:500;">'
                    f"Trang bảng {cur_pg_idx + 1}/{total_tbl_pages} &middot; "
                    f"Tổng {total_filtered} trang lỗi</span>",
                    unsafe_allow_html=True,
                )

                if st.button(
                    "Trang sau ›",
                    disabled=(cur_pg_idx >= total_tbl_pages - 1),
                    key="pg_nav_next",
                ):
                    st.session_state["table_page_idx"] = min(
                        total_tbl_pages - 1, cur_pg_idx + 1
                    )
                    st.rerun()
