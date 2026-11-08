"""
Component Khu vực 2 — PDF Upload & Processing Progress Card (`src/pdf_inspector/ui/components/upload_card.py`).
Chia 2 phần ngang:
- Trái (~33%): Khung nét đứt màu xanh nhạt kèm 2 `st.file_uploader` hoạt động thật, chia đôi khung:
  nửa trái "Chọn file PDF" (1 file, như cũ) | nửa phải "Chọn thư mục" (quét toàn bộ cây thư mục).
- Phải (~67%): Thông tin file PDF thực tế, nút X xóa/reset có xác nhận khi cần,
  thanh tiến trình cập nhật trực tiếp theo pipeline và dòng trạng thái hoàn tất.
"""

import html
from typing import Any, List, Optional, Tuple

import streamlit as st

from backend.schemas.schemas import PDFDocumentInfo
from backend.workers.batch_manager import (
    TAB_SCAN,
    BatchState,
    batch_done_text,
    batch_summary_line,
    bump_dir_uploader,
    clear_batch,
    get_batch,
)
from frontend.state.state_manager import release_document_resources
from frontend.styles.styles import (
    SVG_CHECK_CIRCLE_GREEN,
    SVG_PDF_BADGE,
    SVG_PDF_BADGE_MUTED,
    SVG_UPLOAD_CLOUD,
    clean_html,
)
from backend.repositories.file_utils import format_file_size


@st.dialog("Xác nhận xóa file đang kiểm tra")
def _confirm_clear_file_dialog() -> None:
    """Hộp thoại xác nhận trước khi xóa file nếu có chỉnh sửa review chưa tải xuống."""
    st.markdown(
        "Bạn có các thao tác kiểm duyệt / chỉnh sửa trên tài liệu hiện tại **chưa tải PDF đã sửa xuống**. "
        "Bạn có chắc chắn muốn đóng file và đặt lại trạng thái không?"
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Xóa file & đặt lại", type="primary", width="stretch"):
            release_document_resources(increment_uploader=True)
            st.rerun()
    with c2:
        if st.button("Giữ lại", width="stretch"):
            st.rerun()


def render_progress_html(
    percent: int,
    status_text: str,
    state: str = "done",
) -> str:
    """
    Tạo HTML cho thanh Progress Bar 9px + tỷ lệ % bên phải + dòng trạng thái bên dưới.
    `state` nhận `'done'`, `'scanning'`, hoặc `'idle'`.
    """
    pct_clamped = max(0, min(100, int(percent)))
    safe_status = html.escape(status_text)
    if state == "done":
        icon_html = SVG_CHECK_CIRCLE_GREEN
        cls_state = ""
    elif state == "scanning":
        icon_html = (
            '<span style="display:inline-block;width:14px;height:14px;'
            'border:2px solid #93C5FD;border-top-color:#1677F2;border-radius:50%;'
            'flex-shrink:0;"></span>'
        )
        cls_state = "scanning"
    else:
        icon_html = (
            '<span style="display:inline-block;width:8px;height:8px;'
            'background:#CBD5E1;border-radius:50%;flex-shrink:0;"></span>'
        )
        cls_state = "idle"

    return clean_html(
        f"""
        <div class="proc-progress-wrap">
          <div class="proc-bar-row">
            <div class="proc-bar-track">
              <div class="proc-bar-fill" style="width: {pct_clamped}%;"></div>
            </div>
            <span class="proc-pct-txt">{pct_clamped}%</span>
          </div>
          <div class="proc-status-row {cls_state}">
            {icon_html}
            <span>{safe_status}</span>
          </div>
        </div>
        """
    )


def render_upload_zone(
    file_key: str,
    dir_key: str,
    file_label: str,
) -> Tuple[Optional[Any], List[Any]]:
    """
    Vùng tải lên dùng chung cho cả 2 tab, giữ nguyên khung nét đứt 104px hiện có:
    - Nửa trái "Chọn file PDF": tải 1 file PDF (chức năng cũ).
    - Nửa phải "Chọn thư mục": trình duyệt cho chọn thư mục gốc và gửi mọi file trong cây thư mục kèm
      đường dẫn tương đối (`accept_multiple_files="directory"`). Không giới hạn đuôi ở trình duyệt để backend
      đếm được file bị bỏ qua; backend tự kiểm tra định dạng PDF thực tế.
    Trả về `(file_đơn, danh_sách_file_từ_thư_mục)`.
    """
    st.markdown(
        clean_html(
            f"""
            <div class="upload-visual-overlay upload-visual-dual">
              <div class="upload-dual-top">
                {SVG_UPLOAD_CLOUD}
                <p class="upload-visual-title">Kéo thả PDF hoặc chọn thư mục hồ sơ</p>
              </div>
              <div class="upload-dual-opts">
                <span class="upload-opt upload-opt-file">Chọn file PDF</span>
                <span class="upload-opt upload-opt-dir">Chọn thư mục</span>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    with st.container(key="upl_half_file"):
        single_file = st.file_uploader(
            file_label,
            type=["pdf"],
            label_visibility="collapsed",
            key=file_key,
        )
    with st.container(key="upl_half_dir"):
        dir_files = st.file_uploader(
            f"{file_label} — chọn thư mục",
            accept_multiple_files="directory",
            label_visibility="collapsed",
            key=dir_key,
        )
    return single_file, list(dir_files or [])


def render_batch_info(batch: BatchState) -> None:
    """Khối thông tin khi chọn thư mục: 'Đã chọn: HO_SO' + '3 thư mục con · 5 file PDF · 1 file bỏ qua'."""
    skipped_tip = html.escape(
        "\n".join(f"{p} — {r}" for p, r in batch.skipped[:30])
        + ("\n..." if len(batch.skipped) > 30 else "")
    )
    st.markdown(
        clean_html(
            f"""
            <div class="file-meta-row">
              {SVG_PDF_BADGE}
              <div>
                <p class="file-name-txt">Đã chọn: {html.escape(batch.root_name)}</p>
                <p class="file-sub-txt" title="{skipped_tip}">{html.escape(batch_summary_line(batch))}</p>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_batch_progress(progress_slot: Any, batch: BatchState) -> None:
    """Thanh tiến độ hiện có cho đợt quét thư mục (chưa xong thì không báo hoàn thành)."""
    total = len(batch.entries)
    if batch.has_pending:
        pct = int(round(batch.finished_count / max(1, total) * 100))
        progress_slot.markdown(
            render_progress_html(
                percent=pct,
                status_text=f"Đang kiểm tra {batch.finished_count}/{total} file PDF — {pct}%",
                state="scanning",
            ),
            unsafe_allow_html=True,
        )
    else:
        progress_slot.markdown(
            render_progress_html(
                percent=100 if total else 0,
                status_text=batch_done_text(batch),
                state="done" if total and batch.error_count == 0 else "idle",
            ),
            unsafe_allow_html=True,
        )


def render_upload_and_progress_card() -> Tuple[Optional[Any], List[Any], Any]:
    """
    Hiển thị Card Upload & Progress (33% trái | 67% phải).
    Trả về `(uploaded_file, dir_files, progress_placeholder)` để `streamlit_app.py` cập nhật tiến trình trực tiếp khi quét.
    """
    uploader_ver = int(st.session_state.get("uploader_key_ver", 0))
    dir_uploader_ver = int(st.session_state.get("dir_uploader_key_ver", 0))
    batch: Optional[BatchState] = get_batch(TAB_SCAN)
    doc_info: Optional[PDFDocumentInfo] = st.session_state.get("doc_info")
    display_name: Optional[str] = st.session_state.get("display_file_name")
    file_size_bytes: int = int(st.session_state.get("file_size_bytes", 0))
    pipeline_result = st.session_state.get("pipeline_result")
    doc_error: Optional[str] = st.session_state.get("doc_error")

    with st.container(key="upload_card"):
        col_left, col_right = st.columns([33, 67], vertical_alignment="center")

        with col_left:
            with st.container(key="upload_zone_box"):
                uploaded_file, dir_files = render_upload_zone(
                    file_key=f"pdf_uploader_{uploader_ver}",
                    dir_key=f"pdf_dir_uploader_{dir_uploader_ver}",
                    file_label="Chọn file PDF scan",
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
                        sig_badge = (
                            ' <span title="Tài liệu có chữ ký số — File gốc luôn được giữ nguyên" '
                            'style="font-size:12px;color:#D97706;font-weight:600;">&middot; Có chữ ký số</span>'
                            if doc_info.has_digital_signature
                            else ""
                        )
                        st.markdown(
                            clean_html(
                                f"""
                                <div class="file-meta-row">
                                  {SVG_PDF_BADGE}
                                  <div>
                                    <p class="file-name-txt">{safe_fname}</p>
                                    <p class="file-sub-txt">{doc_info.page_count} trang &middot; {size_label}{sig_badge}</p>
                                  </div>
                                </div>
                                """
                            ),
                            unsafe_allow_html=True,
                        )
                    elif doc_error:
                        safe_fname = html.escape(display_name or "Tài liệu không hợp lệ")
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
                                    <p class="file-sub-txt">Hỗ trợ PDF nhiều trang &middot; Tự động phát hiện sọc, lẫn trang &amp; xoay hướng</p>
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
                            help="Bỏ chọn thư mục và đặt lại trạng thái",
                        ):
                            clear_batch(TAB_SCAN)
                            release_document_resources(increment_uploader=True)
                            bump_dir_uploader(TAB_SCAN)
                            st.rerun()
                    elif doc_info is not None or doc_error is not None:
                        if st.button(
                            "✕",
                            key="btn_clear_file",
                            help="Xóa file đang chọn và đặt lại trạng thái",
                        ):
                            has_unsaved = bool(
                                st.session_state.get("has_unexported_changes")
                                and not st.session_state.get("has_exported_current")
                            )
                            if has_unsaved:
                                _confirm_clear_file_dialog()
                            else:
                                release_document_resources(increment_uploader=True)
                                st.rerun()

                progress_slot = st.empty()
                if batch is not None:
                    render_batch_progress(progress_slot, batch)
                elif pipeline_result is not None and doc_info is not None:
                    total_p = doc_info.page_count
                    progress_slot.markdown(
                        render_progress_html(
                            percent=100,
                            status_text=f"Đã quét xong {total_p}/{total_p} trang",
                            state="done",
                        ),
                        unsafe_allow_html=True,
                    )
                elif doc_error:
                    progress_slot.markdown(
                        render_progress_html(
                            percent=0,
                            status_text="Không thể xử lý file PDF này",
                            state="idle",
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    progress_slot.markdown(
                        render_progress_html(
                            percent=0,
                            status_text="Tải file PDF lên để bắt đầu quét tự động",
                            state="idle",
                        ),
                        unsafe_allow_html=True,
                    )

    return uploaded_file, dir_files, progress_slot
