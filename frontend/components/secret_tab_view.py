"""
Component Giao diện Tab 2 — Kiểm tra dấu độ mật (`src/pdf_inspector/ui/components/secret_tab_view.py`).

Tuân thủ tuyệt đối Design System đã phê duyệt ở Tab 1 và yêu cầu bảo mật Part B3 & C4:
- Upload Card (33% Upload Zone | 67% Thông tin file & Thanh tiến trình trực tiếp).
- 4 Summary Cards: Tổng số trang | Trang phát hiện dấu | Tổng số dấu phát hiện | Kết quả cần kiểm tra.
- Bảng kết quả 4 cột: Trang | Dấu nhận diện | Trạng thái | Xem, kèm 5 bộ lọc pill
  (`Tất cả` | `Mật` | `Tối mật` | `Tuyệt mật` | `Cần kiểm tra`). Chỉ hiển thị trang có dấu hoặc nghi vấn.
- Preview bên phải: Ảnh PDF gốc kèm bounding box thực tế, nhãn dự đoán, ảnh crop phóng to dấu (chỉ trong RAM),
  độ tin cậy, phương pháp phát hiện, 3 nút Review (`Xác nhận` | `Bỏ qua` | `Chỉnh sửa kết quả`)
  và nút xuất báo cáo JSON / CSV (tuyệt đối không tự động tách file PDF).
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from ai_engine.core.config import (
    PREVIEW_DPI,
    SECRET_FEEDBACK_PATH,
    SECRET_RENDER_DPI,
    SECRET_STATUS_DETECTED,
    SECRET_STATUS_ERROR,
    SECRET_STATUS_NO_MARK,
    SECRET_STATUS_REVIEW,
)
from ai_engine.detectors.secret.secret_detector import DetectedSecretMark, SecretPageResult
from backend.schemas.schemas import PDFDocumentInfo
from backend.services.secret_scan_service import SecretDocumentResult, SecretScanService
from backend.workers.batch_manager import (
    TAB_SECRET,
    BatchState,
    bump_dir_uploader,
    clear_batch,
    get_batch,
)
from frontend.components.upload_card import (
    render_batch_info,
    render_batch_progress,
    render_progress_html,
    render_upload_zone,
)
from frontend.state.state_manager import encode_rgb_to_png_bytes, release_secret_resources
from frontend.styles.styles import (
    SVG_PDF_BADGE,
    SVG_PDF_BADGE_MUTED,
    SVG_STAT_PAGES,
    SVG_STAT_SEARCH,
    SVG_STAT_SHIELD_RED,
    SVG_STAT_STAMP_BLUE,
    clean_html,
)
from backend.repositories.file_utils import format_file_size

SECRET_ROWS_PER_PAGE: int = 5
MAX_SECRET_PREVIEW_CACHE: int = 12


def _rgb_to_data_uri(rgb_img: np.ndarray) -> str:
    b64 = base64.b64encode(encode_rgb_to_png_bytes(rgb_img)).decode("ascii")
    return f"data:image/png;base64,{b64}"


def save_secret_review_feedback(
    file_name: str,
    page_result: SecretPageResult,
    action: str,
    override_label: Optional[str] = None,
    note: str = "",
    feedback_path: Path = SECRET_FEEDBACK_PATH,
) -> Path:
    """
    Ghi nhận thao tác rà soát của người dùng (chỉ lưu metadata kỹ thuật, không lưu nội dung văn bản
    hay ảnh crop tài liệu mật — tuân thủ C4).
    """
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "verified_ground_truth": False,
        "source": "user_ui_secret_marking_review",
        "note": (
            "Metadata-only user review log from Secret Marking Workflow. "
            "Not a legal classification conclusion."
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
        "file_name": Path(file_name).name,
        "page_number": page_result.page_number,
        "detector_status": page_result.status,
        "detector_primary_label": page_result.primary_label,
        "review_action": action,
        "effective_label": override_label or page_result.effective_label,
        "marks_count": len(page_result.marks),
        "note": note,
        "verified_ground_truth": False,
    }
    entries: List[Dict[str, Any]] = payload.get("entries", [])
    replaced = False
    for idx, item in enumerate(entries):
        if (
            item.get("file_name") == entry["file_name"]
            and item.get("page_number") == page_result.page_number
        ):
            entries[idx] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)

    payload["entries"] = entries
    feedback_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return feedback_path


def get_cached_secret_preview_bundle(
    pdf_path: Path,
    file_hash: str,
    page_result: SecretPageResult,
) -> Tuple[np.ndarray, List[Tuple[DetectedSecretMark, np.ndarray]]]:
    """
    Lazy-render ảnh xem trước trang kèm bounding box và cắt ảnh phóng to dấu độ mật trong RAM
    (giới hạn LRU tối đa `MAX_SECRET_PREVIEW_CACHE` trang, tuyệt đối không lưu ảnh crop ra ổ đĩa).
    """
    cache: OrderedDict = st.session_state.setdefault(
        "secret_preview_lru_cache", OrderedDict()
    )
    cache_key = (
        file_hash,
        int(page_result.page_number),
        str(page_result.review_action or ""),
        str(page_result.override_label or ""),
    )
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    bundle = SecretScanService.render_page_and_ephemeral_crops(
        pdf_path=pdf_path,
        page_result=page_result,
        preview_dpi=PREVIEW_DPI,
        crop_dpi=SECRET_RENDER_DPI,
    )
    cache[cache_key] = bundle
    while len(cache) > MAX_SECRET_PREVIEW_CACHE:
        cache.popitem(last=False)
    return bundle


def build_secret_flagged_rows(
    doc_result: Optional[SecretDocumentResult],
) -> List[SecretPageResult]:
    """
    Lấy danh sách các trang phát hiện dấu độ mật hoặc nghi vấn cần kiểm tra
    (bao gồm cả trang người dùng đã rà soát trong phiên để có thể xem/chỉnh sửa lại nếu cần;
    ẩn toàn bộ các trang sạch ban đầu `NO_MARK_DETECTED`).
    """
    if doc_result is None:
        return []
    return [
        p
        for p in doc_result.pages
        if p.status != SECRET_STATUS_NO_MARK or p.review_action is not None
    ]


def filter_secret_rows(
    rows: List[SecretPageResult],
    filter_key: str,
) -> List[SecretPageResult]:
    """Lọc danh sách trang theo 5 bộ lọc chuẩn B3: Tất cả | Mật | Tối mật | Tuyệt mật | Cần kiểm tra."""
    if filter_key == "MAT":
        return [
            p
            for p in rows
            if (p.effective_label or p.primary_label) == "MẬT"
            and p.effective_status != SECRET_STATUS_NO_MARK
        ]
    if filter_key == "TOI_MAT":
        return [
            p
            for p in rows
            if (p.effective_label or p.primary_label) == "TỐI MẬT"
            and p.effective_status != SECRET_STATUS_NO_MARK
        ]
    if filter_key == "TUYET_MAT":
        return [
            p
            for p in rows
            if (p.effective_label or p.primary_label) == "TUYỆT MẬT"
            and p.effective_status != SECRET_STATUS_NO_MARK
        ]
    if filter_key == "REVIEW":
        return [
            p
            for p in rows
            if p.effective_status in (SECRET_STATUS_REVIEW, SECRET_STATUS_ERROR)
        ]
    return rows


def _format_label_badge_html(page_res: SecretPageResult) -> str:
    lbl = page_res.effective_label or page_res.primary_label
    if not lbl:
        if page_res.status == SECRET_STATUS_ERROR:
            return '<span class="status-pill orange">Lỗi đọc trang</span>'
        return '<span class="tbl-dash">&mdash;</span>'

    tone_map = {
        "TUYỆT MẬT": "tuyet-mat",
        "TỐI MẬT": "toi-mat",
        "MẬT": "mat",
        "GIẢI MẬT": "giai-mat",
    }
    cls_name = tone_map.get(lbl, "mat")
    extra_cnt = (
        f' <span style="font-size:11.5px;color:#64748B;font-weight:600;">(+{len(page_res.marks) - 1})</span>'
        if len(page_res.marks) > 1
        else ""
    )
    return f'<span class="secret-badge {cls_name}">{html.escape(lbl)}</span>{extra_cnt}'


def _format_secret_status_pill(page_res: SecretPageResult) -> Tuple[str, str]:
    """Trả về `(status_text, tone_class)` cho một dòng trang dấu mật."""
    if page_res.review_action == "IGNORED":
        return ("Đã bỏ qua", "gray")
    if page_res.review_action == "CONFIRMED":
        return ("Đã xác nhận", "green")
    if page_res.review_action == "EDITED":
        if page_res.override_label == SECRET_STATUS_NO_MARK:
            return ("Đã bỏ qua", "gray")
        return ("Đã chỉnh sửa", "green")
    if page_res.status in (SECRET_STATUS_REVIEW, SECRET_STATUS_ERROR):
        return ("Cần kiểm tra", "orange")
    return ("Phát hiện dấu", "green")


def _format_method_vn(method_code: str) -> str:
    if method_code == "BOX_TEMPLATE_MATCH":
        return "So khớp khung dấu chuẩn (Box-First)"
    if method_code == "ISOLATED_TEMPLATE_MATCH":
        return "So khớp mẫu đa tỷ lệ cô lập nền"
    return method_code or "Chưa xác định"


@st.dialog("Chỉnh sửa kết quả nhận diện dấu độ mật")
def _edit_secret_mark_dialog(
    doc_result: SecretDocumentResult,
    page_result: SecretPageResult,
    file_name: str,
) -> None:
    """Hộp thoại chỉnh sửa nhãn độ mật thủ công trên trang đang chọn (không kết luận pháp lý)."""
    st.markdown(
        f"**Trang {page_result.page_number}** — Chọn nhãn độ mật phù hợp sau khi kiểm tra trực quan:"
    )
    options = [
        "MẬT",
        "TỐI MẬT",
        "TUYỆT MẬT",
        "GIẢI MẬT",
        "Không có dấu (Bỏ qua cảnh báo)",
    ]
    cur_lbl = page_result.effective_label or page_result.primary_label or "MẬT"
    default_idx = options.index(cur_lbl) if cur_lbl in options else 0

    chosen = st.selectbox(
        "Nhãn độ mật sau rà soát",
        options=options,
        index=default_idx,
        key=f"sec_dlg_sel_{page_result.page_number}",
    )
    note_val = st.text_input(
        "Ghi chú rà soát (tùy chọn)",
        value=page_result.notes or "",
        placeholder="Ví dụ: Dấu mờ góc trái / Dấu đã gạch xóa...",
        key=f"sec_dlg_note_{page_result.page_number}",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Lưu kết quả", type="primary", width="stretch"):
            override_code = (
                SECRET_STATUS_NO_MARK
                if chosen.startswith("Không có dấu")
                else chosen
            )
            SecretScanService.apply_review_action(
                doc_result=doc_result,
                page_number=page_result.page_number,
                action="EDITED",
                override_label=override_code,
                note=note_val,
            )
            save_secret_review_feedback(
                file_name=file_name,
                page_result=page_result,
                action="EDITED",
                override_label=override_code,
                note=note_val,
            )
            st.session_state["secret_action_feedback_msg"] = (
                f"Đã cập nhật kết quả rà soát cho trang {page_result.page_number}."
            )
            st.rerun()
    with c2:
        if st.button("Hủy", width="stretch"):
            st.rerun()


def render_secret_upload_card() -> Tuple[Optional[Any], List[Any], Any]:
    """
    Hiển thị Upload & Live Progress Card riêng cho Tab 2 (cùng Design System Tab 1).
    Trả về `(uploaded_file, dir_files, progress_placeholder)`.
    """
    uploader_ver = int(st.session_state.get("secret_uploader_key_ver", 0))
    dir_uploader_ver = int(st.session_state.get("secret_dir_uploader_key_ver", 0))
    batch: Optional[BatchState] = get_batch(TAB_SECRET)
    doc_info: Optional[PDFDocumentInfo] = st.session_state.get("secret_doc_info")
    display_name: Optional[str] = st.session_state.get("secret_file_name")
    file_size_bytes: int = int(st.session_state.get("secret_file_size_bytes", 0))
    doc_result: Optional[SecretDocumentResult] = st.session_state.get(
        "secret_doc_result"
    )
    doc_error: Optional[str] = st.session_state.get("secret_doc_error")

    with st.container(key="upload_card"):
        col_left, col_right = st.columns([33, 67], vertical_alignment="center")

        with col_left:
            with st.container(key="upload_zone_box"):
                uploaded_file, dir_files = render_upload_zone(
                    file_key=f"secret_pdf_uploader_{uploader_ver}",
                    dir_key=f"secret_pdf_dir_uploader_{dir_uploader_ver}",
                    file_label="Chọn file PDF kiểm tra dấu độ mật",
                )

        with col_right:
            with st.container(key="file_proc_col"):
                top_info_col, top_close_col = st.columns(
                    [9.3, 0.7], vertical_alignment="top"
                )
                with top_info_col:
                    if batch is not None:
                        render_batch_info(batch)
                    elif doc_info is not None:
                        safe_fname = html.escape(display_name or doc_info.file_name)
                        size_label = format_file_size(
                            file_size_bytes or doc_info.file_size_bytes
                        )
                        st.markdown(
                            clean_html(
                                f"""
                                <div class="file-meta-row">
                                  {SVG_PDF_BADGE}
                                  <div>
                                    <p class="file-name-txt">{safe_fname}</p>
                                    <p class="file-sub-txt">{doc_info.page_count} trang &middot; {size_label}</p>
                                  </div>
                                </div>
                                """
                            ),
                            unsafe_allow_html=True,
                        )
                    elif doc_error:
                        safe_fname = html.escape(
                            display_name or "Tài liệu không hợp lệ"
                        )
                        st.markdown(
                            clean_html(
                                f"""
                                <div class="file-meta-row">
                                  {SVG_PDF_BADGE}
                                  <div>
                                    <p class="file-name-txt">{safe_fname}</p>
                                    <p class="file-sub-txt" style="color:#EB364C;">{html.escape(doc_error)}</p>
                                  </div>
                                </div>
                                """
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            clean_html(
                                f"""
                                <div class="file-meta-row">
                                  {SVG_PDF_BADGE_MUTED}
                                  <div>
                                    <p class="file-name-txt" style="color:#64748B;">Chưa chọn tài liệu PDF</p>
                                    <p class="file-sub-txt">Nhận diện dấu MẬT, TỐI MẬT, TUYỆT MẬT trên PDF nhiều trang &middot; Xử lý offline an toàn</p>
                                  </div>
                                </div>
                                """
                            ),
                            unsafe_allow_html=True,
                        )

                with top_close_col:
                    if batch is not None:
                        if st.button(
                            "✕",
                            key="btn_clear_file",
                            help="Bỏ chọn thư mục và xóa bộ đệm ảnh tạm trong RAM",
                        ):
                            clear_batch(TAB_SECRET)
                            release_secret_resources(increment_uploader=True)
                            bump_dir_uploader(TAB_SECRET)
                            st.rerun()
                    elif doc_info is not None or doc_error is not None:
                        if st.button(
                            "✕",
                            key="btn_clear_file",
                            help="Đóng tài liệu hiện tại và xóa bộ đệm ảnh tạm trong RAM",
                        ):
                            release_secret_resources(increment_uploader=True)
                            st.rerun()

                progress_slot = st.empty()
                if batch is not None:
                    render_batch_progress(progress_slot, batch)
                elif doc_result is not None and doc_info is not None:
                    n_total = doc_info.page_count
                    progress_slot.markdown(
                        render_progress_html(
                            percent=100,
                            status_text=f"Đã kiểm tra xong {n_total}/{n_total} trang",
                            state="done",
                        ),
                        unsafe_allow_html=True,
                    )
                elif doc_error:
                    progress_slot.markdown(
                        render_progress_html(
                            percent=0,
                            status_text="Không thể kiểm tra tài liệu PDF này.",
                            state="idle",
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    progress_slot.markdown(
                        render_progress_html(
                            percent=0,
                            status_text="Tải file PDF lên để bắt đầu nhận diện dấu độ mật",
                            state="idle",
                        ),
                        unsafe_allow_html=True,
                    )

    return uploaded_file, dir_files, progress_slot


def compute_secret_summary(doc_result: Optional[SecretDocumentResult]) -> Dict[str, int]:
    """4 chỉ số thẻ thống kê của một tài liệu (dùng để cộng dồn khi quét nhiều PDF)."""
    if doc_result is None:
        return {"total_pages": 0, "pages_with_marks": 0, "total_marks": 0, "review_pages": 0}
    return {
        "total_pages": doc_result.total_pages,
        "pages_with_marks": sum(
            1
            for p in doc_result.pages
            if p.effective_status in (SECRET_STATUS_DETECTED, SECRET_STATUS_REVIEW)
        ),
        "total_marks": doc_result.total_marks_detected,
        "review_pages": doc_result.review_pages_count,
    }


def render_secret_summary_cards(
    doc_result: Optional[SecretDocumentResult],
    stats: Optional[Dict[str, int]] = None,
) -> None:
    """
    Hiển thị 4 Summary Cards chuẩn B3:
    1. Tổng số trang
    2. Trang phát hiện dấu
    3. Tổng số dấu phát hiện
    4. Kết quả cần kiểm tra
    Khi quét nhiều PDF, `stats` là số liệu đã cộng dồn trên phạm vi các file đang lọc.
    """
    stats = stats if stats is not None else compute_secret_summary(doc_result)
    total_pages = stats["total_pages"]
    pages_with_marks = stats["pages_with_marks"]
    total_marks = stats["total_marks"]
    review_pages = stats["review_pages"]

    st.markdown(
        clean_html(
            f"""
            <div class="summary-cards-grid">
              <div class="summary-stat-card">
                <div class="summary-icon-box blue">{SVG_STAT_PAGES}</div>
                <div>
                  <div class="summary-stat-label">Tổng số trang</div>
                  <div class="summary-stat-val navy">{total_pages}</div>
                </div>
              </div>
              <div class="summary-stat-card">
                <div class="summary-icon-box red">{SVG_STAT_SHIELD_RED}</div>
                <div>
                  <div class="summary-stat-label">Trang phát hiện dấu</div>
                  <div class="summary-stat-val red">{pages_with_marks}</div>
                </div>
              </div>
              <div class="summary-stat-card">
                <div class="summary-icon-box green">{SVG_STAT_STAMP_BLUE}</div>
                <div>
                  <div class="summary-stat-label">Tổng số dấu phát hiện</div>
                  <div class="summary-stat-val green">{total_marks}</div>
                </div>
              </div>
              <div class="summary-stat-card">
                <div class="summary-icon-box orange">{SVG_STAT_SEARCH}</div>
                <div>
                  <div class="summary-stat-label">Kết quả cần kiểm tra</div>
                  <div class="summary-stat-val orange">{review_pages}</div>
                </div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _filter_tagged_secret_rows(
    tagged: List[Tuple[Optional[str], Optional[str], SecretPageResult]],
    filter_key: str,
) -> List[Tuple[Optional[str], Optional[str], SecretPageResult]]:
    """Áp dụng đúng bộ lọc hiện có (`filter_secret_rows`) lên danh sách dòng có gắn tệp PDF."""
    keep = {id(p) for p in filter_secret_rows([t[2] for t in tagged], filter_key)}
    return [t for t in tagged if id(t[2]) in keep]


def _default_secret_select(_file_key: Optional[str], page_number: int) -> None:
    st.session_state["secret_selected_page_num"] = page_number


def render_secret_results_table(
    flagged_rows: List[SecretPageResult],
    has_scanned: bool,
    total_doc_pages: int = 0,
    row_files: Optional[List[Tuple[str, str]]] = None,
    selected_file_key: Optional[str] = None,
    on_select: Optional[Callable[[Optional[str], int], None]] = None,
    render_file_filter: Optional[Callable[[], None]] = None,
    empty_message: Optional[Tuple[str, str]] = None,
) -> None:
    """
    Hiển thị Bảng kết quả 4 cột chuẩn B3:
    `Trang` | `Dấu nhận diện` | `Trạng thái` | `Xem`
    kèm 5 bộ lọc: `Tất cả` | `Mật` | `Tối mật` | `Tuyệt mật` | `Cần kiểm tra`.
    Khi quét nhiều PDF (`row_files` = (mã tệp, đường dẫn hiển thị) song song với `flagged_rows`):
    thêm bộ lọc 'Tệp PDF' (`render_file_filter`) phía trên và cột 'Tệp PDF'; chọn dòng gọi `on_select(mã tệp, trang)`.
    """
    multi_file = row_files is not None
    tagged: List[Tuple[Optional[str], Optional[str], SecretPageResult]] = (
        [(fk, lbl, r) for (fk, lbl), r in zip(row_files, flagged_rows)]
        if multi_file
        else [(None, None, r) for r in flagged_rows]
    )
    select_fn = on_select or _default_secret_select

    n_all = len(tagged)
    n_mat = len(_filter_tagged_secret_rows(tagged, "MAT"))
    n_toi_mat = len(_filter_tagged_secret_rows(tagged, "TOI_MAT"))
    n_tuyet_mat = len(_filter_tagged_secret_rows(tagged, "TUYET_MAT"))
    n_review = len(_filter_tagged_secret_rows(tagged, "REVIEW"))

    active_filter: str = st.session_state.get("secret_active_filter", "ALL")
    filter_defs = [
        ("ALL", f"Tất cả ({n_all})"),
        ("MAT", f"Mật ({n_mat})"),
        ("TOI_MAT", f"Tối mật ({n_toi_mat})"),
        ("TUYET_MAT", f"Tuyệt mật ({n_tuyet_mat})"),
        ("REVIEW", f"Cần kiểm tra ({n_review})"),
    ]

    with st.container(key="left_error_card"):
        st.markdown(
            '<div class="panel-heading">Danh sách trang phát hiện dấu độ mật</div>',
            unsafe_allow_html=True,
        )

        if render_file_filter is not None:
            render_file_filter()

        with st.container(horizontal=True, key="filter_pills_bar"):
            for f_id, f_label in filter_defs:
                is_act = active_filter == f_id
                btn_key = (
                    f"flt_act_sec_{f_id.lower()}"
                    if is_act
                    else f"flt_inact_sec_{f_id.lower()}"
                )
                if st.button(f_label, key=btn_key):
                    if active_filter != f_id:
                        st.session_state["secret_active_filter"] = f_id
                        st.session_state["secret_table_page_idx"] = 0
                        st.session_state["secret_action_feedback_msg"] = None
                        new_filtered = _filter_tagged_secret_rows(tagged, f_id)
                        cur_sel = st.session_state.get("secret_selected_page_num")
                        if new_filtered and all(
                            not ((fk is None or fk == selected_file_key) and r.page_number == cur_sel)
                            for fk, _lbl, r in new_filtered
                        ):
                            select_fn(new_filtered[0][0], new_filtered[0][2].page_number)
                        st.rerun()

        filtered_rows = _filter_tagged_secret_rows(tagged, active_filter)
        col_weights = [2.3, 0.9, 2.1, 1.7, 1.2] if multi_file else [1.0, 2.3, 1.8, 1.3]
        col_titles = (
            ["Tệp PDF", "Trang", "Dấu nhận diện", "Trạng thái", "Xem"]
            if multi_file
            else ["Trang", "Dấu nhận diện", "Trạng thái", "Xem"]
        )

        with st.container(key="secret_table_wrapper"):
            with st.container(key="tbl_hdr_row"):
                h_cols = st.columns(col_weights, gap=None, vertical_alignment="center")
                for col, title in zip(h_cols, col_titles):
                    with col:
                        st.markdown(
                            f'<div class="tbl-hdr-cell">{title}</div>',
                            unsafe_allow_html=True,
                        )

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
                          <div class="tbl-empty-title">Chưa có dữ liệu kiểm tra dấu độ mật</div>
                          <div class="tbl-empty-sub">Hãy tải lên một file PDF ở khung phía trên để hệ thống tự động nhận diện dấu MẬT, TỐI MẬT và TUYỆT MẬT.</div>
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
                          <div class="tbl-empty-title" style="color:#079B60;">Không phát hiện dấu độ mật trên toàn bộ {total_doc_pages} trang</div>
                          <div class="tbl-empty-sub">Các trang không có dấu độ mật (`NO_MARK_DETECTED`) được ẩn khỏi bảng theo đúng quy chuẩn.</div>
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
                          <div class="tbl-empty-title">Không có trang nào khớp bộ lọc đang chọn</div>
                          <div class="tbl-empty-sub">Nhấn sang bộ lọc 'Tất cả' để xem đầy đủ các trang phát hiện dấu hoặc nghi vấn.</div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )
            else:
                total_pages_tbl = max(
                    1, math.ceil(len(filtered_rows) / SECRET_ROWS_PER_PAGE)
                )
                cur_pg = max(
                    0,
                    min(
                        int(st.session_state.get("secret_table_page_idx", 0)),
                        total_pages_tbl - 1,
                    ),
                )
                st.session_state["secret_table_page_idx"] = cur_pg
                start_i = cur_pg * SECRET_ROWS_PER_PAGE
                visible_rows = filtered_rows[start_i : start_i + SECRET_ROWS_PER_PAGE]
                selected_pno = st.session_state.get("secret_selected_page_num")

                for file_key, file_label, row in visible_rows:
                    pno = row.page_number
                    is_selected = pno == selected_pno and (
                        file_key is None or file_key == selected_file_key
                    )
                    sfx = f"{file_key}_{pno}" if file_key else f"{pno}"
                    row_key = (
                        f"tbl_row_sel_sec_{sfx}"
                        if is_selected
                        else f"tbl_row_norm_sec_{sfx}"
                    )
                    status_txt, status_tone = _format_secret_status_pill(row)

                    with st.container(key=row_key):
                        r_cols = st.columns(
                            col_weights, gap=None, vertical_alignment="center"
                        )
                        offset = 0
                        if multi_file:
                            safe_lbl = html.escape(file_label or "")
                            with r_cols[0]:
                                st.markdown(
                                    f'<div class="tbl-data-cell tbl-file-cell" title="{safe_lbl}">{safe_lbl}</div>',
                                    unsafe_allow_html=True,
                                )
                            offset = 1
                        with r_cols[offset]:
                            st.markdown(
                                f'<div class="tbl-data-cell"><span class="tbl-page-num">{pno}</span></div>',
                                unsafe_allow_html=True,
                            )
                        with r_cols[offset + 1]:
                            st.markdown(
                                f'<div class="tbl-data-cell">{_format_label_badge_html(row)}</div>',
                                unsafe_allow_html=True,
                            )
                        with r_cols[offset + 2]:
                            st.markdown(
                                f'<div class="tbl-data-cell"><span class="status-pill {status_tone}">{html.escape(status_txt)}</span></div>',
                                unsafe_allow_html=True,
                            )
                        with r_cols[offset + 3]:
                            with st.container(key=f"tbl_action_cell_sec_{sfx}"):
                                if st.button(
                                    "Xem",
                                    icon=":material/visibility:",
                                    key=f"btn_view_p_sec_{sfx}",
                                ):
                                    select_fn(file_key, pno)
                                    st.session_state[
                                        "secret_action_feedback_msg"
                                    ] = None
                                    st.rerun()

                if total_pages_tbl > 1:
                    with st.container(horizontal=True, key="tbl_pagination_bar"):
                        p_col1, p_col2, p_col3 = st.columns(
                            [1.2, 2.6, 1.2], vertical_alignment="center"
                        )
                        with p_col1:
                            if st.button(
                                "← Trước",
                                disabled=(cur_pg <= 0),
                                key="pg_nav_sec_prev",
                            ):
                                st.session_state["secret_table_page_idx"] = cur_pg - 1
                                st.rerun()
                        with p_col2:
                            st.markdown(
                                f'<div style="text-align:center;font-size:12.5px;color:#70849F;">'
                                f"Trang bảng {cur_pg + 1}/{total_pages_tbl} ({len(filtered_rows)} trang có dấu/nghi vấn)</div>",
                                unsafe_allow_html=True,
                            )
                        with p_col3:
                            if st.button(
                                "Sau →",
                                disabled=(cur_pg >= total_pages_tbl - 1),
                                key="pg_nav_sec_next",
                            ):
                                st.session_state["secret_table_page_idx"] = cur_pg + 1
                                st.rerun()


def render_secret_preview_panel(
    doc_result: Optional[SecretDocumentResult],
    selected_page: Optional[SecretPageResult],
    active_pdf_path: Optional[Path],
    file_hash: Optional[str],
    file_name: str,
) -> None:
    """
    Hiển thị Panel bên phải Tab 2 chuẩn B3:
    - Ảnh PDF gốc kèm bounding box dấu mật & nhãn dự đoán.
    - Ảnh crop phóng to dấu (chỉ tồn tại trong RAM).
    - Độ tin cậy (Confidence) & Phương pháp phát hiện.
    - 3 nút Review Actions: `Xác nhận` | `Bỏ qua` | `Chỉnh sửa kết quả`.
    - Xuất báo cáo kết quả JSON / CSV (tuyệt đối không tự động tách PDF).
    """
    with st.container(key="right_preview_card"):
        st.markdown(
            '<div class="panel-heading">Xem trước dấu nhận diện</div>',
            unsafe_allow_html=True,
        )

        if (
            doc_result is None
            or selected_page is None
            or active_pdf_path is None
            or not active_pdf_path.exists()
        ):
            st.markdown(
                clean_html(
                    """
                    <div class="doc-preview-stage">
                      <div style="text-align:center;color:#70849F;font-size:13.5px;padding:0 16px;">
                        Chọn một trang trong danh sách bên trái để xem vị trí dấu độ mật, ảnh phóng to dấu và thao tác rà soát.
                      </div>
                    </div>
                    <div class="error-detail-card" style="margin-top:8px;">
                      <div class="error-detail-row"><span class="error-detail-label">Trang:</span><span class="error-detail-val">&mdash;</span></div>
                      <div class="error-detail-row"><span class="error-detail-label">Nhãn dự đoán:</span><span class="error-detail-val">&mdash;</span></div>
                      <div class="error-detail-row"><span class="error-detail-label">Độ tin cậy:</span><span class="error-detail-val">&mdash;</span></div>
                      <div class="error-detail-row"><span class="error-detail-label">Phương pháp:</span><span class="error-detail-val">&mdash;</span></div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            if doc_result is not None:
                _render_secret_export_buttons(doc_result, file_name)
            return

        annotated_rgb, crops = get_cached_secret_preview_bundle(
            pdf_path=active_pdf_path,
            file_hash=file_hash or "sec",
            page_result=selected_page,
        )
        page_uri = _rgb_to_data_uri(annotated_rgb)

        eff_lbl = (
            selected_page.effective_label
            or selected_page.primary_label
            or "Cần kiểm tra"
        )
        pill_tone = (
            "orange"
            if selected_page.effective_status in (SECRET_STATUS_REVIEW, SECRET_STATUS_ERROR)
            else ("green" if eff_lbl == "GIẢI MẬT" else "")
        )

        st.markdown(
            clean_html(
                f"""
                <div class="doc-preview-stage" style="height:192px;">
                  <div class="doc-sheet-frame" style="max-height:176px;">
                    <img src="{page_uri}" alt="Trang {selected_page.page_number}" style="max-height:176px;" />
                  </div>
                  <div class="preview-floating-pill {pill_tone}">
                    <span class="preview-pill-dot"></span>
                    <span>{html.escape(eff_lbl)}</span>
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        # Khung ảnh crop phóng to dấu (chỉ trong RAM — C4)
        if crops:
            top_mark, top_crop = crops[0]
            crop_uri = _rgb_to_data_uri(top_crop)
            bbox_txt = f"({top_mark.bbox[0]}, {top_mark.bbox[1]}) – ({top_mark.bbox[2]}, {top_mark.bbox[3]}) px"
            st.markdown(
                clean_html(
                    f"""
                    <div class="secret-crop-strip">
                      <div class="secret-crop-thumb">
                        <img src="{crop_uri}" alt="Crop dấu độ mật" />
                      </div>
                      <div class="secret-crop-meta">
                        <strong>Ảnh phóng to vùng dấu (chỉ hiển thị trong RAM)</strong>
                        <span>Vị trí BBox: {html.escape(bbox_txt)} &middot; Kênh: {html.escape(top_mark.channel_used)}</span>
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        top_conf_str = (
            f"{max(m.confidence for m in selected_page.marks) * 100:.1f}%"
            if selected_page.marks
            else "N/A"
        )
        method_str = (
            _format_method_vn(selected_page.marks[0].detection_method)
            if selected_page.marks
            else (selected_page.error_message or "Không xác định")
        )
        status_txt, status_tone = _format_secret_status_pill(selected_page)

        st.markdown(
            clean_html(
                f"""
                <div class="error-detail-card">
                  <div class="error-detail-row">
                    <span class="error-detail-label">Trang:</span>
                    <span class="error-detail-val">{selected_page.page_number} <span class="status-pill {status_tone}" style="margin-left:8px;">{html.escape(status_txt)}</span></span>
                  </div>
                  <div class="error-detail-row">
                    <span class="error-detail-label">Nhãn dự đoán:</span>
                    <span class="error-detail-val">{_format_label_badge_html(selected_page)}</span>
                  </div>
                  <div class="error-detail-row">
                    <span class="error-detail-label">Độ tin cậy:</span>
                    <span class="error-detail-val">{html.escape(top_conf_str)}</span>
                  </div>
                  <div class="error-detail-row">
                    <span class="error-detail-label">Phương pháp:</span>
                    <span class="error-detail-val">{html.escape(method_str)}</span>
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        # Hàng 3 nút Review Actions: Xác nhận | Bỏ qua | Chỉnh sửa kết quả
        pno = selected_page.page_number
        with st.container(key="preview_actions_bar"):
            b1, b2, b3 = st.columns([1.05, 0.92, 1.25], vertical_alignment="center")
            with b1:
                with st.container(key="act_accept"):
                    if st.button(
                        "Xác nhận",
                        icon=":material/check:",
                        width="stretch",
                        disabled=(selected_page.status == SECRET_STATUS_ERROR),
                        key=f"sec_btn_confirm_{pno}",
                    ):
                        SecretScanService.apply_review_action(
                            doc_result, pno, "CONFIRMED"
                        )
                        save_secret_review_feedback(
                            file_name=file_name,
                            page_result=selected_page,
                            action="CONFIRMED",
                        )
                        st.session_state["secret_action_feedback_msg"] = (
                            f"Đã xác nhận dấu '{selected_page.effective_label}' tại trang {pno}."
                        )
                        st.rerun()
            with b2:
                with st.container(key="act_skip"):
                    if st.button(
                        "Bỏ qua",
                        icon=":material/skip_next:",
                        width="stretch",
                        key=f"sec_btn_ignore_{pno}",
                    ):
                        SecretScanService.apply_review_action(
                            doc_result, pno, "IGNORED"
                        )
                        save_secret_review_feedback(
                            file_name=file_name,
                            page_result=selected_page,
                            action="IGNORED",
                        )
                        st.session_state["secret_action_feedback_msg"] = (
                            f"Đã bỏ qua cảnh báo tại trang {pno}."
                        )
                        st.rerun()
            with b3:
                with st.container(key="act_manual"):
                    if st.button(
                        "Chỉnh sửa kết quả",
                        icon=":material/edit:",
                        width="stretch",
                        key=f"sec_btn_edit_{pno}",
                    ):
                        _edit_secret_mark_dialog(
                            doc_result=doc_result,
                            page_result=selected_page,
                            file_name=file_name,
                        )

        fb_msg = st.session_state.get("secret_action_feedback_msg")
        if fb_msg:
            st.markdown(
                f'<div style="font-size:12.5px;color:#079B60;font-weight:600;">{html.escape(fb_msg)}</div>',
                unsafe_allow_html=True,
            )

        _render_secret_export_buttons(doc_result, file_name)


def _render_secret_export_buttons(
    doc_result: SecretDocumentResult,
    file_name: str,
) -> None:
    """Hiển thị 2 nút xuất báo cáo JSON / CSV (không tách PDF theo đúng B3 & C4)."""
    stem = Path(file_name or "document.pdf").stem
    json_bytes = SecretScanService.export_report_json_bytes(doc_result)
    csv_bytes = SecretScanService.export_report_csv_bytes(doc_result)

    with st.container(key="sec_export_bar"):
        ec1, ec2 = st.columns(2, vertical_alignment="center")
        with ec1:
            with st.container(key="sec_exp_json"):
                st.download_button(
                    label="Báo cáo JSON",
                    data=json_bytes,
                    file_name=f"{stem}_secret_report.json",
                    mime="application/json",
                    icon=":material/description:",
                    width="stretch",
                    key="sec_dl_json_panel",
                )
        with ec2:
            with st.container(key="sec_exp_csv"):
                st.download_button(
                    label="Báo cáo CSV",
                    data=csv_bytes,
                    file_name=f"{stem}_secret_report.csv",
                    mime="text/csv",
                    icon=":material/table_view:",
                    width="stretch",
                    key="sec_dl_csv_panel",
                )

