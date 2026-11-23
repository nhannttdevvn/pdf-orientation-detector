"""
Component Khu vực Upload lớn (Mục 5) và Thanh thông tin file 1 hàng ngang (Mục 6)
theo Thiết kế Cuối cùng Đã Duyệt (`frontend/components/upload_card.py`).
"""

import html
from typing import Any, List, Optional, Tuple

import streamlit as st

from backend.schemas.schemas import PDFDocumentInfo, PipelineResult
from backend.workers.batch_manager import (
    DIR_UPLOADER_VER_KEY,
    TAB_SCAN,
    BatchState,
    batch_done_text,
    batch_summary_line,
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

SVG_LEFT_PDF_ILLUST = """
<svg width="110" height="96" viewBox="0 0 110 96" fill="none" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="56" cy="52" rx="44" ry="34" fill="#DBEAFE" fill-opacity="0.7"/>
  <g transform="rotate(-10 48 48)">
    <rect x="24" y="14" width="46" height="60" rx="5" fill="#BFDBFE" stroke="#93C5FD" stroke-width="1.4"/>
  </g>
  <g transform="rotate(-3 56 48)">
    <rect x="30" y="12" width="48" height="62" rx="5" fill="#FFFFFF" stroke="#93C5FD" stroke-width="1.6"/>
    <path d="M64 12L78 26H68C65.7909 26 64 24.2091 64 22V12Z" fill="#DBEAFE"/>
    <line x1="38" y1="26" x2="56" y2="26" stroke="#93C5FD" stroke-width="2.2" stroke-linecap="round"/>
    <line x1="38" y1="33" x2="68" y2="33" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
    <line x1="38" y1="39" x2="64" y2="39" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
    <rect x="35" y="47" width="38" height="16" rx="3.5" fill="#EF4444"/>
    <text x="54" y="58.5" text-anchor="middle" fill="#FFFFFF" font-family="Inter, sans-serif" font-size="10.5" font-weight="800" letter-spacing="0.5">PDF</text>
  </g>
</svg>
"""

SVG_RIGHT_FOLDER_ILLUST = """
<svg width="114" height="96" viewBox="0 0 114 96" fill="none" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="58" cy="54" rx="44" ry="34" fill="#DBEAFE" fill-opacity="0.65"/>
  <g transform="rotate(6 58 48)">
    <path d="M24 30C24 26.6863 26.6863 24 30 24H47L53 31H84C87.3137 31 90 33.6863 90 37V70C90 73.3137 87.3137 76 84 76H30C26.6863 76 24 73.3137 24 70V30Z" fill="#F59E0B"/>
    <rect x="33" y="18" width="44" height="44" rx="3.5" transform="rotate(-6 33 18)" fill="#E0F2FE" stroke="#93C5FD" stroke-width="1.2"/>
    <rect x="38" y="21" width="44" height="44" rx="3.5" transform="rotate(2 38 21)" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.3"/>
    <line x1="45" y1="30" x2="68" y2="31" stroke="#CBD5E1" stroke-width="2" stroke-linecap="round"/>
    <line x1="45" y1="36" x2="73" y2="37" stroke="#E2E8F0" stroke-width="2" stroke-linecap="round"/>
    <path d="M22 39C22 36.2386 24.2386 34 27 34H87C89.7614 34 92 36.2386 92 39L89 71C89 73.7614 86.7614 76 84 76H30C27.2386 76 25 73.7614 25 71L22 39Z" fill="#FBBF24"/>
  </g>
</svg>
"""

SVG_BTN_FILE_ICON = """
<svg width="15" height="15" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <path d="M4 1.5H9.5L13 5V13.5C13 14.0523 12.5523 14.5 12 14.5H4C3.44772 14.5 3 14.0523 3 13.5V2.5C3 1.94772 3.44772 1.5 4 1.5Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
  <path d="M9 1.5V5.5H13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_BTN_FOLDER_ICON = """
<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <path d="M1.5 4C1.5 3.17157 2.17157 2.5 3 2.5H6.2L7.7 4.2H13C13.8284 4.2 14.5 4.87157 14.5 5.7V12C14.5 12.8284 13.8284 13.5 13 13.5H3C2.17157 13.5 1.5 12.8284 1.5 12V4Z" stroke="#0866E8" stroke-width="1.5" stroke-linejoin="round"/>
</svg>
"""

SVG_META_PAGE_ICON = """
<svg width="13" height="13" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <rect x="3" y="1.8" width="10" height="12.4" rx="1.6" stroke="#64748B" stroke-width="1.4"/>
  <line x1="5.5" y1="5.5" x2="10.5" y2="5.5" stroke="#64748B" stroke-width="1.3" stroke-linecap="round"/>
  <line x1="5.5" y1="8.2" x2="10.5" y2="8.2" stroke="#64748B" stroke-width="1.3" stroke-linecap="round"/>
  <line x1="5.5" y1="10.9" x2="8.5" y2="10.9" stroke="#64748B" stroke-width="1.3" stroke-linecap="round"/>
</svg>
"""

SVG_META_DB_ICON = """
<svg width="13" height="13" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <ellipse cx="8" cy="4" rx="5" ry="2" stroke="#64748B" stroke-width="1.4"/>
  <path d="M3 4V8C3 9.10457 5.23858 10 8 10C10.7614 10 13 9.10457 13 8V4" stroke="#64748B" stroke-width="1.4"/>
  <path d="M3 8V12C3 13.1046 5.23858 14 8 14C10.7614 14 13 13.1046 13 12V8" stroke="#64748B" stroke-width="1.4"/>
</svg>
"""


def _format_file_size(size_bytes: int) -> str:
    """Định dạng kích thước file sang KB / MB chuẩn xác."""
    if size_bytes <= 0:
        return "0 B"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def render_progress_html(
    percent: int,
    status_text: str,
    state: str = "done",
) -> str:
    """
    Sinh mã HTML cho thanh tiến trình 1 hàng ngang duy nhất (Mục 6 & Ảnh tham chiếu):
    [ ✔ Hoàn thành ]   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   100%
    `state` thuộc `{"done", "scanning", "idle"}`.
    """
    pct = max(0, min(100, int(percent)))
    safe_tip = html.escape(status_text, quote=True)

    if state == "done":
        badge_html = (
            f'<div class="proc-badge-pill done" title="{safe_tip}">'
            f"{SVG_CHECK_CIRCLE_GREEN}<span>Hoàn thành</span></div>"
        )
        bar_cls = "proc-bar-fill done"
    elif state == "scanning":
        badge_html = (
            f'<div class="proc-badge-pill scanning" title="{safe_tip}">'
            f'<span class="proc-dot-spin">●</span><span>{html.escape(status_text)}</span></div>'
        )
        bar_cls = "proc-bar-fill scanning"
    else:
        badge_html = (
            f'<div class="proc-badge-pill idle" title="{safe_tip}">'
            f"<span>● {html.escape(status_text)}</span></div>"
        )
        bar_cls = "proc-bar-fill idle"

    return clean_html(
        f"""
        <div class="proc-single-line-row" title="{safe_tip}">
          {badge_html}
          <div class="proc-bar-track">
            <div class="{bar_cls}" style="width: {pct}%;"></div>
          </div>
          <div class="proc-pct-txt">{pct}%</div>
        </div>
        """
    )


def render_upload_zone_html() -> str:
    """HTML lớp phủ trực quan của khu vực Upload lớn (Mục 5 & Ảnh tham chiếu)."""
    return clean_html(
        f"""
        <div class="upload-visual-overlay">
          <div class="upload-illust-col left">{SVG_LEFT_PDF_ILLUST}</div>
          <div class="upload-center-col">
            <div class="upload-cloud-wrap">{SVG_UPLOAD_CLOUD}</div>
            <div class="upload-visual-title">Kéo thả file PDF hoặc thư mục vào đây</div>
            <div class="upload-dual-opts">
              <span class="upload-opt upload-opt-file">{SVG_BTN_FILE_ICON}<span>Chọn file PDF</span></span>
              <span class="upload-opt upload-opt-dir">{SVG_BTN_FOLDER_ICON}<span>Chọn thư mục</span></span>
            </div>
          </div>
          <div class="upload-illust-col right">{SVG_RIGHT_FOLDER_ILLUST}</div>
        </div>
        """
    )


def render_upload_zone(
    file_key: str,
    dir_key: str,
    file_label: str = "Tải lên file PDF",
) -> Tuple[Optional[Any], List[Any]]:
    """
    Vùng tải lên dùng chung cho cả 2 tab theo mẫu duyệt (Mục 5):
    - Nửa trái 'Chọn file PDF': tải 1 file PDF.
    - Nửa phải 'Chọn thư mục': chọn thư mục chứa cây hồ sơ PDF.
    """
    st.markdown(render_upload_zone_html(), unsafe_allow_html=True)
    with st.container(key="upl_half_file"):
        single_file = st.file_uploader(
            label=file_label,
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=file_key,
        )
    with st.container(key="upl_half_dir"):
        dir_files = st.file_uploader(
            label=f"{file_label} — chọn thư mục",
            type=None,
            accept_multiple_files="directory",
            label_visibility="collapsed",
            key=dir_key,
        )
    if single_file is not None and str(getattr(single_file, "name", "")).startswith("._"):
        single_file = None
    return single_file, list(dir_files or [])


def render_batch_info(batch: BatchState) -> None:
    """Khối thông tin khi chọn thư mục dùng chung cho cả 2 tab."""
    skipped_tip = html.escape(
        "\n".join(f"{p} — {r}" for p, r in batch.skipped[:30])
        + ("\n..." if len(batch.skipped) > 30 else ""),
        quote=True,
    )
    st.markdown(
        clean_html(
            f"""
            <div class="file-meta-row">
              {SVG_PDF_BADGE}
              <div style="min-width:0; flex:1;">
                <p class="file-name-txt" title="{html.escape(batch.root_name, quote=True)}">Đã chọn: {html.escape(batch.root_name)}</p>
                <p class="file-sub-txt" title="{skipped_tip}">{html.escape(batch_summary_line(batch))}</p>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_batch_progress(progress_slot: Any, batch: BatchState) -> None:
    """Thanh tiến độ dùng chung cho đợt quét thư mục."""
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


def _batch_file_meta(batch: BatchState) -> Tuple[str, str, int, str, str]:
    """(tên hiển thị, dòng phụ, phần trăm, dòng trạng thái, state) cho chế độ thư mục."""
    total_pages = sum(e.page_count for e in batch.entries)
    total_bytes = sum(e.size_bytes for e in batch.entries)
    n_ok = batch.done_count
    sub_txt = batch_summary_line(batch)
    if total_pages:
        sub_txt = f"{sub_txt} · {total_pages} trang · {_format_file_size(total_bytes)}"
    return (
        f"Đã chọn: {batch.root_name}",
        sub_txt,
        100 if batch.entries else 0,
        batch_done_text(batch),
        "done" if n_ok > 0 and batch.error_count == 0 else "idle",
    )


@st.dialog("Thông tin chi tiết tài liệu PDF")
def _open_file_info_dialog(
    file_name: str,
    page_count: int,
    file_size_str: str,
    status_text: str,
    input_path: Optional[str] = None,
) -> None:
    st.markdown(
        f"""
        - **Tên tài liệu:** `{file_name}`
        - **Số trang:** `{page_count} trang`
        - **Dung lượng:** `{file_size_str}`
        - **Trạng thái phiên:** {status_text}
        - **Đường dẫn bản sao tạm:** `{input_path or "Trong bộ nhớ phiên"}`
        - **Bảo toàn dữ liệu:** Mọi thao tác xử lý chỉ thực hiện trên bản sao tạm thời; file PDF gốc của người dùng được giữ nguyên 100%.
        """
    )


def render_upload_and_progress_card() -> Tuple[Optional[Any], Optional[List[Any]], Any]:
    """
    Hiển thị:
    - Mục 5: Khu vực Upload lớn ngay dưới Header (170–200px).
    - Mục 6: Thanh thông tin file 1 hàng ngang duy nhất (65–75px) ngay dưới vùng Upload.
    Trả về `(uploaded_file, uploaded_dir_files, progress_placeholder)`.
    """
    uploader_ver: int = int(st.session_state.get("uploader_key_ver", 0))
    dir_ver: int = int(st.session_state.get(DIR_UPLOADER_VER_KEY[TAB_SCAN], 0))
    doc_info: Optional[PDFDocumentInfo] = st.session_state.get("doc_info")
    result: Optional[PipelineResult] = st.session_state.get("pipeline_result")
    doc_error: Optional[str] = st.session_state.get("doc_error")
    batch: Optional[BatchState] = get_batch(TAB_SCAN)

    # =========================================================================
    # MỤC 5: UPLOAD AREA — GIỮ ĐÚNG MẪU (Ngay dưới Header)
    # =========================================================================
    with st.container(key="upload_zone_box"):
        st.markdown(render_upload_zone_html(), unsafe_allow_html=True)
        with st.container(key="upl_half_file"):
            uploaded_file = st.file_uploader(
                label="Tải lên file PDF",
                type=["pdf"],
                accept_multiple_files=False,
                label_visibility="collapsed",
                key=f"pdf_uploader_widget_{uploader_ver}",
            )
        with st.container(key="upl_half_dir"):
            dir_files = st.file_uploader(
                label="Tải lên thư mục chứa PDF",
                type=None,
                accept_multiple_files="directory",
                label_visibility="collapsed",
                key=f"pdf_dir_uploader_{dir_ver}",
            )

    # Bỏ qua file ẩn macOS AppleDouble (._*) nếu trình duyệt gửi kèm
    if uploaded_file is not None and str(getattr(uploaded_file, "name", "")).startswith("._"):
        uploaded_file = None

    # =========================================================================
    # MỤC 6: THANH THÔNG TIN FILE (1 thanh ngang duy nhất 65–75px)
    # =========================================================================
    with st.container(key="upload_card"):
        col_meta, col_prog, col_act = st.columns([2.9, 5.5, 1.6], vertical_alignment="center")

        has_active_doc = False
        file_name = "Chưa chọn tài liệu PDF"
        page_count = 0
        size_str = "0 B"
        status_text = "Chờ tải file PDF"

        if batch is not None and not batch.has_pending:
            has_active_doc = True
            file_name, sub_txt, pct, status_text, prog_state = _batch_file_meta(batch)
            badge_svg = SVG_PDF_BADGE if batch.done_count > 0 else SVG_PDF_BADGE_MUTED
            meta_sub_html = f'<div class="file-sub-txt">{html.escape(sub_txt)}</div>'
        elif doc_info is not None and result is not None:
            has_active_doc = True
            file_name = st.session_state.get("display_file_name") or doc_info.file_name
            page_count = doc_info.page_count
            file_size = int(st.session_state.get("file_size_bytes") or doc_info.file_size_bytes)
            size_str = _format_file_size(file_size)
            pct = 100
            status_text = f"Đã quét xong {page_count}/{page_count} trang"
            prog_state = "done"
            badge_svg = SVG_PDF_BADGE
            meta_sub_html = (
                f'<div class="file-sub-txt">'
                f'<span class="file-meta-item">{SVG_META_PAGE_ICON}<span>{page_count} trang</span></span>'
                f'<span class="file-meta-item">{SVG_META_DB_ICON}<span>{size_str}</span></span>'
                f"</div>"
            )
        elif doc_error:
            has_active_doc = True
            file_name = st.session_state.get("display_file_name") or "Tài liệu lỗi"
            pct = 0
            status_text = "Không thể mở PDF"
            prog_state = "idle"
            badge_svg = SVG_PDF_BADGE
            short_err = doc_error.split("(")[0].strip() or "File PDF không hợp lệ hoặc bị hỏng"
            meta_sub_html = f'<div class="file-sub-txt" style="color:#E93D4D;" title="{html.escape(doc_error, quote=True)}">{html.escape(short_err)}</div>'
        else:
            pct = 0
            prog_state = "idle"
            badge_svg = SVG_PDF_BADGE_MUTED
            meta_sub_html = (
                f'<div class="file-sub-txt">'
                f'<span class="file-meta-item">{SVG_META_PAGE_ICON}<span>0 trang</span></span>'
                f'<span class="file-meta-item">{SVG_META_DB_ICON}<span>0 B</span></span>'
                f"</div>"
            )

        with col_meta:
            safe_fname = html.escape(str(file_name))
            st.markdown(
                clean_html(
                    f"""
                    <div class="file-meta-row">
                      {badge_svg}
                      <div style="min-width:0; flex:1;">
                        <p class="file-name-txt" title="{safe_fname}">{safe_fname}</p>
                        {meta_sub_html}
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        with col_prog:
            progress_slot = st.empty()
            progress_slot.markdown(
                render_progress_html(
                    percent=pct,
                    status_text=status_text,
                    state=prog_state,
                ),
                unsafe_allow_html=True,
            )

        with col_act:
            with st.container(horizontal=True, vertical_alignment="center", key="file_bar_actions_row"):
                if st.button(
                    "",
                    icon=":material/folder_open:",
                    key="btn_view_file_info",
                    disabled=not has_active_doc,
                    help="Xem thông tin tệp / thư mục đang xử lý",
                ):
                    _open_file_info_dialog(
                        file_name=str(file_name),
                        page_count=page_count,
                        file_size_str=size_str,
                        status_text=status_text,
                        input_path=st.session_state.get("input_pdf_path"),
                    )

                if st.button(
                    "",
                    icon=":material/info:",
                    key="btn_doc_tech_info",
                    disabled=not has_active_doc,
                    help="Thông tin chi tiết tiến trình kiểm tra",
                ):
                    _open_file_info_dialog(
                        file_name=str(file_name),
                        page_count=page_count,
                        file_size_str=size_str,
                        status_text=status_text,
                        input_path=st.session_state.get("input_pdf_path"),
                    )

                if st.button(
                    "",
                    icon=":material/delete_outline:",
                    key="btn_clear_file",
                    disabled=not has_active_doc,
                    help="Bỏ file khỏi phiên xử lý (không xóa PDF gốc trên máy)",
                ):
                    clear_batch(TAB_SCAN)
                    release_document_resources(increment_uploader=True)
                    st.rerun()

    return uploaded_file, dir_files, progress_slot
