"""
Giao diện Dashboard thống nhất cho hệ thống PDF Scan Quality Inspector (`src/pdf_inspector/ui/streamlit_app.py`).

Hỗ trợ hai chế độ làm việc độc lập hoàn toàn trên cùng Design System đã phê duyệt (không Sidebar):
- **Tab 1 — Kiểm tra & sửa lỗi PDF (`SCAN_QUALITY`):**
  Header -> Upload Card -> 4 Summary Cards -> Bảng 6 cột Danh sách trang có lỗi + Preview & Sửa lỗi -> Footer.
  Kết nối trực tiếp `DocumentPipeline`, `PDFCorrector`, `BoundaryInspectionService`, `ReportGenerator`.
- **Tab 2 — Kiểm tra dấu độ mật (`SECRET_DETECTION`):**
  Header -> Upload Card -> 4 Summary Cards -> Bảng 4 cột Dấu nhận diện + Preview & Crop trong RAM + Review & Xuất JSON/CSV -> Footer.
  Kết nối trực tiếp `SecretScanService` (chỉ chạy workflow được chọn, không tự động tách PDF, không trộn trạng thái).

Cả 2 tab hỗ trợ chọn 1 file PDF (như cũ) hoặc chọn thư mục hồ sơ: quét toàn bộ PDF trong cây thư mục
(`ui/batch_manager.py`), xử lý lần lượt từng file bằng đúng pipeline 1 file, hiển thị trên giao diện sẵn có
kèm bộ lọc 'Tệp PDF' và cột 'Tệp PDF' khi có nhiều file.
"""

import hashlib
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st

from ai_engine.core.config import SECRET_STATUS_ERROR, SECRET_STATUS_REVIEW, UPLOADS_DIR
from ai_engine.core.exceptions import (
    InvalidPDFError,
    PDFLoaderError,
    PasswordProtectedPDFError,
)
from ai_engine.core.logging_config import setup_logger
from ai_engine.detectors.secret.secret_detector import SecretPageResult
from backend.schemas.schemas import PageDecision, PipelineResult
from ai_engine.preprocessing.pdf_loader import PDFLoader
from backend.services.secret_scan_service import SecretDocumentResult
from backend.workers.batch_manager import (
    FILTER_ALL,
    STATUS_DONE,
    STATUS_ERROR,
    TAB_SCAN,
    TAB_SECRET,
    BatchEntry,
    BatchState,
    activate_entry,
    bump_dir_uploader,
    clear_batch,
    entry_state,
    file_filter_options,
    first_entry_to_show,
    get_batch,
    prepare_batch,
    run_batch,
    scope_entries,
    set_batch,
    sync_active_snapshot,
)
from frontend.components.secret_tab_view import compute_secret_summary
from frontend.components import (
    build_secret_flagged_rows,
    render_error_table_panel,
    render_footer_bar,
    render_header_bar,
    render_preview_panel,
    render_progress_html,
    render_secret_preview_panel,
    render_secret_results_table,
    render_secret_summary_cards,
    render_secret_upload_card,
    render_summary_cards,
    render_upload_and_progress_card,
)
from frontend.state.state_manager import (
    PageIssueRow,
    build_problem_page_rows,
    compute_dashboard_summary,
    get_cached_pipeline,
    get_cached_secret_service,
    init_dashboard_state,
    rebuild_outputs_without_ocr,
    release_document_resources,
    release_secret_resources,
)
from frontend.styles.styles import inject_dashboard_styles

logger = setup_logger("streamlit_app")


def _save_uploaded_pdf(file_bytes: bytes, file_name: str, file_hash: str) -> Path:
    """Lưu bản sao tạm của file PDF tải lên vào `data/uploads/` (bảo toàn 100% file gốc)."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file_name).name
    target_path = UPLOADS_DIR / f"{file_hash[:12]}_{safe_name}"
    if not target_path.exists():
        target_path.write_bytes(file_bytes)
    return target_path


def _batch_progress_html(
    batch_progress: Tuple[int, int],
    file_label: str,
    curr_page: int,
    total_pages: int,
) -> str:
    """Tiến độ khi quét thư mục: 'Đang kiểm tra 3/5 file PDF — 60%' (kèm trang đang xử lý của file hiện tại)."""
    idx, n_files = batch_progress
    frac = ((idx - 1) + (curr_page / max(1, total_pages))) / max(1, n_files)
    pct = max(1, min(99, int(frac * 100)))
    return render_progress_html(
        percent=pct,
        status_text=(
            f"Đang kiểm tra {idx}/{n_files} file PDF — {pct}% · "
            f"{file_label}: trang {curr_page}/{total_pages}"
        ),
        state="scanning",
    )


def _execute_automatic_scan(
    pdf_path: Path,
    display_name: str,
    file_size_bytes: int,
    file_hash: str,
    progress_slot,
    batch_progress: Optional[Tuple[int, int]] = None,
    progress_label: Optional[str] = None,
) -> None:
    """
    Thực thi tự động luồng kiểm tra chất lượng scan (Tab 1 — `Scan Quality Workflow`):
    Upload -> PDFLoader -> DocumentPipeline -> Xuất PDF tạm & Báo cáo.
    Tuyệt đối không gọi `SecretMarkingDetector`.
    """
    release_document_resources(increment_uploader=False)
    st.session_state["file_hash"] = file_hash
    st.session_state["input_pdf_path"] = str(pdf_path)
    st.session_state["display_file_name"] = display_name
    st.session_state["file_size_bytes"] = file_size_bytes

    try:
        with PDFLoader(pdf_path) as loader:
            info = loader.get_document_info()
            info.file_name = display_name
            st.session_state["doc_info"] = info
            st.session_state["doc_error"] = None
    except (PasswordProtectedPDFError, InvalidPDFError, PDFLoaderError) as exc:
        logger.warning("Không thể mở file PDF '%s': %s", display_name, exc)
        st.session_state["doc_info"] = None
        st.session_state["doc_error"] = str(exc)
        return

    total_pages = info.page_count
    progress_slot.markdown(
        render_progress_html(
            percent=2,
            status_text=f"Đang khởi tạo quét 0/{total_pages} trang...",
            state="scanning",
        ),
        unsafe_allow_html=True,
    )

    pipeline = get_cached_pipeline()

    def _on_page_progress(curr: int, total: int, _dec: PageDecision) -> None:
        if batch_progress is not None:
            progress_slot.markdown(
                _batch_progress_html(batch_progress, progress_label or display_name, curr, total),
                unsafe_allow_html=True,
            )
            return
        pct = max(3, min(98, int(round((curr / max(1, total)) * 96))))
        progress_slot.markdown(
            render_progress_html(
                percent=pct,
                status_text=f"Đang quét trang {curr}/{total}...",
                state="scanning",
            ),
            unsafe_allow_html=True,
        )

    result = pipeline.scan_document(
        pdf_path=pdf_path,
        progress_callback=_on_page_progress,
        manual_overrides={},
    )
    result.doc_info.file_name = display_name
    st.session_state["pipeline_result"] = result

    rebuild_outputs_without_ocr()

    issue_rows = build_problem_page_rows(st.session_state["pipeline_result"])
    if issue_rows:
        first_review = next(
            (r.page_number for r in issue_rows if r.needs_review_now),
            issue_rows[0].page_number,
        )
        st.session_state["selected_page_num"] = first_review

    if batch_progress is not None:
        return  # Quét thư mục: chỉ báo hoàn thành khi mọi file đã xử lý xong
    progress_slot.markdown(
        render_progress_html(
            percent=100,
            status_text=f"Đã quét xong {total_pages}/{total_pages} trang",
            state="done",
        ),
        unsafe_allow_html=True,
    )


def _execute_automatic_secret_scan(
    pdf_path: Path,
    display_name: str,
    file_size_bytes: int,
    file_hash: str,
    progress_slot,
    batch_progress: Optional[Tuple[int, int]] = None,
    progress_label: Optional[str] = None,
) -> None:
    """
    Thực thi tự động luồng nhận diện dấu độ mật (Tab 2 — `Secret Marking Workflow`):
    Upload -> PDFLoader -> SecretScanService -> Lưu kết quả trong RAM phiên.
    Tuyệt đối không chạy Orientation / Stripe / Boundary Detector.
    """
    release_secret_resources(increment_uploader=False)
    st.session_state["secret_file_hash"] = file_hash
    st.session_state["secret_pdf_path"] = str(pdf_path)
    st.session_state["secret_file_name"] = display_name
    st.session_state["secret_file_size_bytes"] = file_size_bytes

    try:
        with PDFLoader(pdf_path) as loader:
            info = loader.get_document_info()
            info.file_name = display_name
            st.session_state["secret_doc_info"] = info
            st.session_state["secret_doc_error"] = None
    except (PasswordProtectedPDFError, InvalidPDFError, PDFLoaderError) as exc:
        logger.warning("Không thể mở file PDF '%s': %s", display_name, exc)
        st.session_state["secret_doc_info"] = None
        st.session_state["secret_doc_error"] = str(exc)
        return

    total_pages = info.page_count
    progress_slot.markdown(
        render_progress_html(
            percent=2,
            status_text=f"Đang khởi tạo kiểm tra dấu độ mật 0/{total_pages} trang...",
            state="scanning",
        ),
        unsafe_allow_html=True,
    )

    secret_service = get_cached_secret_service()

    def _on_secret_progress(curr: int, total: int) -> None:
        if batch_progress is not None:
            progress_slot.markdown(
                _batch_progress_html(batch_progress, progress_label or display_name, curr, total),
                unsafe_allow_html=True,
            )
            return
        pct = max(4, min(98, int(round((curr / max(1, total)) * 96))))
        progress_slot.markdown(
            render_progress_html(
                percent=pct,
                status_text=f"Đang kiểm tra dấu độ mật trang {curr}/{total}...",
                state="scanning",
            ),
            unsafe_allow_html=True,
        )

    doc_result = secret_service.scan_pdf(
        pdf_path=pdf_path,
        progress_callback=_on_secret_progress,
    )
    doc_result.doc_info.file_name = display_name
    st.session_state["secret_doc_result"] = doc_result

    flagged = build_secret_flagged_rows(doc_result)
    if flagged:
        first_target = next(
            (
                p.page_number
                for p in flagged
                if p.status in (SECRET_STATUS_REVIEW, SECRET_STATUS_ERROR)
            ),
            flagged[0].page_number,
        )
        st.session_state["secret_selected_page_num"] = first_target

    if batch_progress is not None:
        return  # Quét thư mục: chỉ báo hoàn thành khi mọi file đã xử lý xong
    progress_slot.markdown(
        render_progress_html(
            percent=100,
            status_text=f"Đã kiểm tra xong {total_pages}/{total_pages} trang",
            state="done",
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# QUÉT THƯ MỤC HỒ SƠ (dùng chung cho 2 tab; mỗi tab giữ pipeline riêng)
# =============================================================================
def _start_folder_batch(tab: str, dir_files: List) -> None:
    """Người dùng vừa chọn thư mục: lưu bản sao các PDF phía server rồi giải phóng file khỏi RAM phiên."""
    clear_batch(tab)
    if tab == TAB_SCAN:
        release_document_resources(increment_uploader=True)
    else:
        release_secret_resources(increment_uploader=True)
    set_batch(tab, prepare_batch(dir_files))
    bump_dir_uploader(tab)


def _run_folder_batch(tab: str, batch: BatchState, progress_slot) -> None:
    """Xử lý lần lượt từng PDF bằng đúng pipeline 1 file của tab; file lỗi không làm dừng cả đợt."""
    execute = _execute_automatic_scan if tab == TAB_SCAN else _execute_automatic_secret_scan

    def _process_one(entry: BatchEntry, batch_progress: Tuple[int, int]) -> None:
        execute(
            pdf_path=Path(entry.saved_path),
            display_name=entry.file_name,
            file_size_bytes=entry.size_bytes,
            file_hash=entry.file_hash,
            progress_slot=progress_slot,
            batch_progress=batch_progress,
            progress_label=entry.display_path,
        )

    def _on_file_start(idx: int, total: int, entry: BatchEntry) -> None:
        pct = int(round((idx - 1) / max(1, total) * 100))
        progress_slot.markdown(
            render_progress_html(
                percent=pct,
                status_text=f"Đang kiểm tra {idx}/{total} file PDF — {pct}% · {entry.display_path}",
                state="scanning",
            ),
            unsafe_allow_html=True,
        )

    run_batch(tab, batch, _process_one, on_file_start=_on_file_start)
    first = first_entry_to_show(batch)
    if first is not None:
        activate_entry(tab, batch, first.key)
    logger.info(
        "Hoàn tất quét thư mục '%s' (%s): %d thành công, %d lỗi, %d file bỏ qua",
        batch.root_name, tab, batch.done_count, batch.error_count, len(batch.skipped),
    )


def _render_file_filter(tab: str, batch: BatchState) -> None:
    """Bộ lọc nhỏ 'Tệp PDF: [Tất cả tệp ▼]' phía trên bảng kết quả (chỉ hiện khi có nhiều PDF)."""
    opts = file_filter_options(batch)
    values = [v for v, _ in opts]
    labels = dict(opts)
    cur_idx = values.index(batch.file_filter) if batch.file_filter in values else 0
    with st.container(key="file_filter_bar"):
        c_lbl, c_sel = st.columns([1.0, 5.6], vertical_alignment="center")
        with c_lbl:
            st.markdown('<div class="file-filter-label">Tệp PDF:</div>', unsafe_allow_html=True)
        with c_sel:
            choice = st.selectbox(
                "Tệp PDF",
                options=values,
                index=cur_idx,
                format_func=lambda v: labels.get(v, v),
                label_visibility="collapsed",
                key=f"file_filter_sel_{tab}_{batch.batch_id}",
            )
    if choice != batch.file_filter:
        batch.file_filter = choice
        st.session_state["table_page_idx" if tab == TAB_SCAN else "secret_table_page_idx"] = 0
        if choice != FILTER_ALL:
            activate_entry(tab, batch, choice)
        st.rerun()


def _batch_empty_message(batch: BatchState, scope: List[BatchEntry]) -> Optional[Tuple[str, str]]:
    """Thông báo trong bảng khi phạm vi đang chọn không có file nào xử lý thành công."""
    if not batch.entries:
        return (
            "Thư mục không có file PDF hợp lệ",
            f"Đã bỏ qua {len(batch.skipped)} file không phải PDF. Hãy chọn thư mục khác.",
        )
    if any(e.status == STATUS_DONE for e in scope):
        return None
    errs = [e for e in scope if e.status == STATUS_ERROR]
    if len(scope) == 1 and errs:
        return ("Không thể xử lý tệp PDF này", errs[0].error or "Lỗi không xác định")
    return ("Không có tệp PDF nào xử lý thành công", f"{len(errs)} tệp gặp lỗi khi kiểm tra.")


def _ensure_active_entry(tab: str, batch: BatchState) -> None:
    """File đang hiển thị ở khung xem trước phải thuộc phạm vi bộ lọc 'Tệp PDF'."""
    if batch.file_filter != FILTER_ALL:
        if batch.active_key != batch.file_filter and batch.get(batch.file_filter) is not None:
            activate_entry(tab, batch, batch.file_filter)
    elif batch.get(batch.active_key) is None:
        first = first_entry_to_show(batch)
        if first is not None:
            activate_entry(tab, batch, first.key)


def _render_scan_batch_body(batch: BatchState) -> None:
    """Tab 1 khi quét thư mục: thẻ thống kê, bảng (bộ lọc + cột Tệp PDF) và xem trước file đang chọn."""
    _ensure_active_entry(TAB_SCAN, batch)
    sync_active_snapshot(TAB_SCAN, batch)
    scope = scope_entries(batch)
    multi_file = len(batch.entries) > 1

    rows: List[PageIssueRow] = []
    row_files: List[Tuple[str, str]] = []
    stats: Dict[str, int] = {"total_pages": 0, "error_pages": 0, "auto_rotated_pages": 0, "review_pages": 0}
    scope_pages = 0
    for e in scope:
        if e.status != STATUS_DONE:
            continue
        with entry_state(TAB_SCAN, batch, e):
            e_result = st.session_state.get("pipeline_result")
            e_rows = build_problem_page_rows(e_result)
            e_stats = compute_dashboard_summary(e_result, e_rows)
        rows.extend(e_rows)
        row_files.extend([(e.key, e.display_path)] * len(e_rows))
        for k in stats:
            stats[k] += e_stats[k]
        scope_pages += e.page_count

    # Dòng đang chọn phải nằm trong bảng; nếu không, chọn dòng đầu tiên (và mở đúng file của dòng đó)
    sel_pno = st.session_state.get("selected_page_num")
    if rows and not any(fk == batch.active_key and r.page_number == sel_pno for (fk, _l), r in zip(row_files, rows)):
        first_fk = row_files[0][0]
        if first_fk != batch.active_key:
            activate_entry(TAB_SCAN, batch, first_fk, reset_table_page=False)
        st.session_state["selected_page_num"] = rows[0].page_number

    result: Optional[PipelineResult] = st.session_state.get("pipeline_result")
    active_rows = build_problem_page_rows(result)
    sel_pno = st.session_state.get("selected_page_num")
    selected_row = next((r for r in active_rows if r.page_number == sel_pno), None)

    render_summary_cards(stats)

    def _on_select(file_key: Optional[str], pno: int) -> None:
        if file_key and file_key != batch.active_key:
            activate_entry(TAB_SCAN, batch, file_key, reset_table_page=False)
        st.session_state["selected_page_num"] = pno

    active_pdf_str: Optional[str] = st.session_state.get("input_pdf_path")
    with st.container(key="main_two_col_row"):
        col_table, col_preview = st.columns([61.5, 38.5], vertical_alignment="top")
        with col_table:
            render_error_table_panel(
                problem_rows=rows,
                has_scanned=True,
                total_doc_pages=scope_pages,
                row_files=row_files if multi_file else None,
                selected_file_key=batch.active_key,
                on_select=_on_select,
                render_file_filter=(lambda: _render_file_filter(TAB_SCAN, batch)) if multi_file else None,
                empty_message=_batch_empty_message(batch, scope),
            )
        with col_preview:
            render_preview_panel(
                selected_row=selected_row,
                active_pdf_path=Path(active_pdf_str) if active_pdf_str else None,
                file_hash=st.session_state.get("file_hash"),
                total_pages=result.doc_info.page_count if result is not None else 0,
                file_name=st.session_state.get("display_file_name") or "document.pdf",
                scan_dpi=result.config.render_dpi if result is not None else 160,
            )


def _render_secret_batch_body(batch: BatchState) -> None:
    """Tab 2 khi quét thư mục: thẻ thống kê, bảng (bộ lọc + cột Tệp PDF) và xem trước file đang chọn."""
    _ensure_active_entry(TAB_SECRET, batch)
    sync_active_snapshot(TAB_SECRET, batch)
    scope = scope_entries(batch)
    multi_file = len(batch.entries) > 1

    rows: List[SecretPageResult] = []
    row_files: List[Tuple[str, str]] = []
    stats: Dict[str, int] = {"total_pages": 0, "pages_with_marks": 0, "total_marks": 0, "review_pages": 0}
    for e in scope:
        doc_res = e.snapshot.get("secret_doc_result")
        if e.status != STATUS_DONE or doc_res is None:
            continue
        e_rows = build_secret_flagged_rows(doc_res)
        rows.extend(e_rows)
        row_files.extend([(e.key, e.display_path)] * len(e_rows))
        e_stats = compute_secret_summary(doc_res)
        for k in stats:
            stats[k] += e_stats[k]

    sel_pno = st.session_state.get("secret_selected_page_num")
    if rows and not any(fk == batch.active_key and r.page_number == sel_pno for (fk, _l), r in zip(row_files, rows)):
        first_fk = row_files[0][0]
        if first_fk != batch.active_key:
            activate_entry(TAB_SECRET, batch, first_fk, reset_table_page=False)
        st.session_state["secret_selected_page_num"] = rows[0].page_number

    doc_result: Optional[SecretDocumentResult] = st.session_state.get("secret_doc_result")
    sel_pno = st.session_state.get("secret_selected_page_num")
    selected_page = (
        next((p for p in build_secret_flagged_rows(doc_result) if p.page_number == sel_pno), None)
        if doc_result is not None
        else None
    )

    render_secret_summary_cards(doc_result, stats=stats)

    def _on_select(file_key: Optional[str], pno: int) -> None:
        if file_key and file_key != batch.active_key:
            activate_entry(TAB_SECRET, batch, file_key, reset_table_page=False)
        st.session_state["secret_selected_page_num"] = pno

    sec_pdf_str: Optional[str] = st.session_state.get("secret_pdf_path")
    with st.container(key="main_two_col_row"):
        col_table, col_preview = st.columns([61.5, 38.5], vertical_alignment="top")
        with col_table:
            render_secret_results_table(
                flagged_rows=rows,
                has_scanned=True,
                total_doc_pages=stats["total_pages"],
                row_files=row_files if multi_file else None,
                selected_file_key=batch.active_key,
                on_select=_on_select,
                render_file_filter=(lambda: _render_file_filter(TAB_SECRET, batch)) if multi_file else None,
                empty_message=_batch_empty_message(batch, scope),
            )
        with col_preview:
            render_secret_preview_panel(
                doc_result=doc_result,
                selected_page=selected_page,
                active_pdf_path=Path(sec_pdf_str) if sec_pdf_str else None,
                file_hash=st.session_state.get("secret_file_hash"),
                file_name=st.session_state.get("secret_file_name") or "document.pdf",
            )


def _render_scan_quality_tab() -> None:
    """Hiển thị toàn bộ nội dung Tab 1 — Kiểm tra & sửa lỗi PDF."""
    uploaded_file, dir_files, progress_slot = render_upload_and_progress_card()
    batch = get_batch(TAB_SCAN)

    if dir_files:  # Vừa chọn thư mục hồ sơ
        _start_folder_batch(TAB_SCAN, dir_files)
        st.rerun()
    if batch is not None and batch.has_pending:
        _run_folder_batch(TAB_SCAN, batch, progress_slot)
        st.rerun()
    if uploaded_file is not None and batch is not None:  # Chọn lại 1 file PDF: thoát chế độ thư mục
        clear_batch(TAB_SCAN)
        release_document_resources(increment_uploader=False)
        batch = None
    if batch is not None:
        _render_scan_batch_body(batch)
        return

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        incoming_hash = hashlib.sha256(raw_bytes).hexdigest()
        if incoming_hash != st.session_state.get("file_hash"):
            saved_path = _save_uploaded_pdf(
                file_bytes=raw_bytes,
                file_name=uploaded_file.name,
                file_hash=incoming_hash,
            )
            _execute_automatic_scan(
                pdf_path=saved_path,
                display_name=uploaded_file.name,
                file_size_bytes=len(raw_bytes),
                file_hash=incoming_hash,
                progress_slot=progress_slot,
            )
            st.rerun()

    result: Optional[PipelineResult] = st.session_state.get("pipeline_result")
    has_scanned = result is not None
    problem_rows = build_problem_page_rows(result)
    summary_stats = compute_dashboard_summary(result, problem_rows)

    selected_row: Optional[PageIssueRow] = None
    if problem_rows:
        sel_pno = st.session_state.get("selected_page_num")
        selected_row = next(
            (r for r in problem_rows if r.page_number == sel_pno), None
        )
        if selected_row is None:
            selected_row = problem_rows[0]
            st.session_state["selected_page_num"] = selected_row.page_number

    render_summary_cards(summary_stats)

    active_pdf_str: Optional[str] = st.session_state.get("input_pdf_path")
    active_pdf_path = Path(active_pdf_str) if active_pdf_str else None
    total_doc_pages = result.doc_info.page_count if result is not None else 0
    cur_file_name = (
        st.session_state.get("display_file_name")
        or (result.doc_info.file_name if result is not None else "document.pdf")
    )
    scan_dpi = result.config.render_dpi if result is not None else 160

    with st.container(key="main_two_col_row"):
        col_table, col_preview = st.columns([61.5, 38.5], vertical_alignment="top")

        with col_table:
            render_error_table_panel(
                problem_rows=problem_rows,
                has_scanned=has_scanned,
                total_doc_pages=total_doc_pages,
            )

        with col_preview:
            render_preview_panel(
                selected_row=selected_row,
                active_pdf_path=active_pdf_path,
                file_hash=st.session_state.get("file_hash"),
                total_pages=total_doc_pages,
                file_name=cur_file_name,
                scan_dpi=scan_dpi,
            )


def _render_secret_detection_tab() -> None:
    """Hiển thị toàn bộ nội dung Tab 2 — Kiểm tra dấu độ mật."""
    uploaded_file, dir_files, progress_slot = render_secret_upload_card()
    batch = get_batch(TAB_SECRET)

    if dir_files:  # Vừa chọn thư mục hồ sơ
        _start_folder_batch(TAB_SECRET, dir_files)
        st.rerun()
    if batch is not None and batch.has_pending:
        _run_folder_batch(TAB_SECRET, batch, progress_slot)
        st.rerun()
    if uploaded_file is not None and batch is not None:  # Chọn lại 1 file PDF: thoát chế độ thư mục
        clear_batch(TAB_SECRET)
        release_secret_resources(increment_uploader=False)
        batch = None
    if batch is not None:
        _render_secret_batch_body(batch)
        return

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        incoming_hash = hashlib.sha256(raw_bytes).hexdigest()
        if incoming_hash != st.session_state.get("secret_file_hash"):
            saved_path = _save_uploaded_pdf(
                file_bytes=raw_bytes,
                file_name=uploaded_file.name,
                file_hash=incoming_hash,
            )
            _execute_automatic_secret_scan(
                pdf_path=saved_path,
                display_name=uploaded_file.name,
                file_size_bytes=len(raw_bytes),
                file_hash=incoming_hash,
                progress_slot=progress_slot,
            )
            st.rerun()

    doc_result: Optional[SecretDocumentResult] = st.session_state.get(
        "secret_doc_result"
    )
    has_scanned = doc_result is not None
    flagged_rows = build_secret_flagged_rows(doc_result)

    selected_page: Optional[SecretPageResult] = None
    if flagged_rows:
        sel_pno = st.session_state.get("secret_selected_page_num")
        selected_page = next(
            (p for p in flagged_rows if p.page_number == sel_pno), None
        )
        if selected_page is None:
            selected_page = flagged_rows[0]
            st.session_state["secret_selected_page_num"] = selected_page.page_number

    render_secret_summary_cards(doc_result)

    sec_pdf_str: Optional[str] = st.session_state.get("secret_pdf_path")
    sec_pdf_path = Path(sec_pdf_str) if sec_pdf_str else None
    total_doc_pages = doc_result.total_pages if doc_result is not None else 0
    cur_file_name = (
        st.session_state.get("secret_file_name")
        or (doc_result.doc_info.file_name if doc_result is not None else "document.pdf")
    )

    with st.container(key="main_two_col_row"):
        col_table, col_preview = st.columns([61.5, 38.5], vertical_alignment="top")

        with col_table:
            render_secret_results_table(
                flagged_rows=flagged_rows,
                has_scanned=has_scanned,
                total_doc_pages=total_doc_pages,
            )

        with col_preview:
            render_secret_preview_panel(
                doc_result=doc_result,
                selected_page=selected_page,
                active_pdf_path=sec_pdf_path,
                file_hash=st.session_state.get("secret_file_hash"),
                file_name=cur_file_name,
            )


def main() -> None:
    st.set_page_config(
        page_title="PDF Scan Quality Inspector",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_dashboard_state()
    inject_dashboard_styles()

    # 1. HEADER BAR + 2-MODE SWITCHER
    render_header_bar()

    # 2–4. BODY THEO CHẾ ĐỘ ĐANG CHỌN (không chạy thừa workflow còn lại)
    if st.session_state.get("active_mode", "SCAN_QUALITY") == "SECRET_DETECTION":
        _render_secret_detection_tab()
    else:
        _render_scan_quality_tab()

    # 5. FOOTER INFORMATION BAR
    render_footer_bar()


if __name__ == "__main__":
    main()
