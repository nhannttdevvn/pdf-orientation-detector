"""
Quét cây thư mục hồ sơ PDF (nhiều file) dùng chung cho cả 2 tab (`src/pdf_inspector/ui/batch_manager.py`).

Nguyên tắc:
- Trình duyệt gửi từng file kèm đường dẫn tương đối (`webkitRelativePath`, vd. `HO_SO/Ho_so_A/Phu_luc/03.pdf`).
  Backend KHÔNG tin dữ liệu frontend: chuẩn hóa & kiểm tra đường dẫn (chống path traversal), kiểm tra chữ ký
  `%PDF-` thực tế, và KHÔNG dùng đường dẫn người dùng để ghi file (bản sao phía server đặt tên theo mã băm).
- Mỗi PDF là một tác vụ độc lập, xử lý tuần tự; một file lỗi không làm dừng cả đợt quét.
- Giữ nguyên toàn bộ component/pipeline của chế độ 1 file: trạng thái của từng file được lưu riêng (`snapshot`)
  và nạp lại vào đúng các khóa `session_state` mà component hiện có đang dùng khi người dùng chọn file đó.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import re
import shutil
from pathlib import Path
import unicodedata
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import streamlit as st

from ai_engine.core.config import UPLOADS_DIR

TAB_SCAN = "scan"
TAB_SECRET = "secret"

FILTER_ALL = "__ALL__"

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_ERROR = "ERROR"
STATUS_LABELS = {
    STATUS_PENDING: "Chờ xử lý",
    STATUS_RUNNING: "Đang xử lý",
    STATUS_DONE: "Hoàn thành",
    STATUS_ERROR: "Lỗi",
}

MAX_BATCH_PDF_FILES = 2000
PDF_SIGNATURE = b"%PDF-"

# Các khóa session_state chứa trạng thái của MỘT tài liệu ở từng tab (đúng các khóa component hiện có đang dùng)
DOC_STATE_KEYS: Dict[str, Dict[str, Any]] = {
    TAB_SCAN: {
        "file_hash": None,
        "input_pdf_path": None,
        "display_file_name": None,
        "file_size_bytes": 0,
        "doc_info": None,
        "doc_error": None,
        "pipeline_result": None,
        "selected_page_num": None,
        "selected_defect_key_by_page": {},
        "preview_tab": "ORIGINAL",
        "manual_overrides": {},
        "stripe_feedback": {},
        "boundary_feedback": {},
        "boundary_adjustments": {},
        "approved_boundary_ops": {},
        "defect_review_decisions": {},
        "has_unexported_changes": False,
        "has_exported_current": False,
        "export_pdf_bytes": None,
        "export_file_name": "corrected.pdf",
        "action_feedback_msg": None,
    },
    TAB_SECRET: {
        "secret_file_hash": None,
        "secret_pdf_path": None,
        "secret_file_name": None,
        "secret_file_size_bytes": 0,
        "secret_doc_info": None,
        "secret_doc_error": None,
        "secret_doc_result": None,
        "secret_selected_page_num": None,
        "secret_action_feedback_msg": None,
    },
}

# Khóa phiên bản widget tải lên theo thư mục (tách riêng với widget tải 1 file để reset độc lập)
DIR_UPLOADER_VER_KEY = {TAB_SCAN: "dir_uploader_key_ver", TAB_SECRET: "secret_dir_uploader_key_ver"}
TABLE_PAGE_IDX_KEY = {TAB_SCAN: "table_page_idx", TAB_SECRET: "secret_table_page_idx"}


@dataclass
class BatchEntry:
    """Một file PDF trong đợt quét thư mục."""

    key: str
    rel_path: str  # đường dẫn tương đối đã chuẩn hóa, gồm cả thư mục gốc (vd. HO_SO/Ho_so_A/01.pdf)
    file_name: str
    size_bytes: int
    file_hash: str
    saved_path: str  # bản sao phía server (không dùng đường dẫn người dùng gửi lên)
    status: str = STATUS_PENDING
    error: Optional[str] = None
    page_count: int = 0
    snapshot: Dict[str, Any] = field(default_factory=dict)

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

    @property
    def display_path(self) -> str:
        """Đường dẫn hiển thị trong bảng: bỏ tên thư mục gốc (vd. Ho_so_A/Phu_luc/03.pdf)."""
        parts = self.rel_path.split("/")
        return "/".join(parts[1:]) if len(parts) > 1 else self.rel_path


@dataclass
class BatchState:
    """Một đợt quét thư mục hồ sơ."""

    batch_id: str
    root_name: str
    entries: List[BatchEntry]
    subfolder_count: int
    skipped: List[Tuple[str, str]]  # (đường dẫn tương đối, lý do bỏ qua)
    storage_dir: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    active_key: Optional[str] = None
    file_filter: str = FILTER_ALL

    @property
    def has_pending(self) -> bool:
        return any(e.status in (STATUS_PENDING, STATUS_RUNNING) for e in self.entries)

    @property
    def done_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_DONE)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_ERROR)

    @property
    def finished_count(self) -> int:
        return self.done_count + self.error_count

    def get(self, key: Optional[str]) -> Optional[BatchEntry]:
        return next((e for e in self.entries if e.key == key), None)


# -----------------------------------------------------------------------------
# Kiểm tra & chuẩn bị đầu vào (không tin dữ liệu frontend)
# -----------------------------------------------------------------------------
def normalize_relative_path(raw_name: str) -> Optional[str]:
    """
    Chuẩn hóa đường dẫn tương đối do trình duyệt gửi lên; trả về None nếu không hợp lệ / có dấu hiệu path traversal.
    - Chuẩn hóa Unicode NFC (macOS gửi tên tiếng Việt dạng NFD).
    - Đổi '\\' thành '/', bỏ phần rỗng và '.', từ chối '..', đường dẫn tuyệt đối, ổ đĩa Windows, ký tự điều khiển.
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None
    name = unicodedata.normalize("NFC", raw_name).replace("\\", "/")
    if "\x00" in name or any(ord(ch) < 32 for ch in name):
        return None
    if name.startswith("/") or (len(name) >= 2 and name[1] == ":"):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


def is_real_pdf(data: bytes) -> bool:
    """Kiểm tra định dạng PDF thực tế: chữ ký `%PDF-` nằm trong 1024 byte đầu (theo đặc tả PDF)."""
    return bool(data) and PDF_SIGNATURE in data[:1024]


def prepare_batch(uploaded_files: List[Any], uploads_dir: Path = UPLOADS_DIR) -> BatchState:
    """
    Thu thập danh sách PDF từ các file trình duyệt gửi lên (mọi cấp thư mục con), bỏ qua file không phải PDF,
    và lưu bản sao phía server (đọc & ghi lần lượt từng file, không giữ toàn bộ nội dung trong RAM).
    """
    batch_id = uuid.uuid4().hex[:12]
    storage_dir = uploads_dir / f"batch_{batch_id}"
    storage_dir.mkdir(parents=True, exist_ok=True)

    entries: List[BatchEntry] = []
    skipped: List[Tuple[str, str]] = []
    root_names: List[str] = []
    subfolders = set()

    def _natural_key(up: Any) -> List[Any]:
        rel_name = normalize_relative_path(str(getattr(up, "name", "") or "")) or ""
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", rel_name)]

    # Trình duyệt gửi file theo thứ tự bất kỳ: sắp xếp tự nhiên theo đường dẫn để danh sách bám đúng cây thư mục
    for up in sorted(uploaded_files, key=_natural_key):
        raw_name = str(getattr(up, "name", "") or "")
        rel = normalize_relative_path(raw_name)
        if rel is None:
            skipped.append((raw_name or "(không tên)", "Đường dẫn không hợp lệ"))
            continue
        parts = rel.split("/")
        if len(parts) > 1:
            root_names.append(parts[0])
            for depth in range(2, len(parts)):
                subfolders.add("/".join(parts[:depth]))

        if not parts[-1].lower().endswith(".pdf"):
            skipped.append((rel, "Không phải file PDF"))
            continue
        data = up.getvalue()
        if not is_real_pdf(data):
            skipped.append((rel, "Có đuôi .pdf nhưng nội dung không phải PDF"))
            continue
        if len(entries) >= MAX_BATCH_PDF_FILES:
            skipped.append((rel, f"Vượt giới hạn {MAX_BATCH_PDF_FILES} file PDF mỗi đợt"))
            continue

        file_hash = hashlib.sha256(data).hexdigest()
        idx = len(entries) + 1
        saved = storage_dir / f"{idx:04d}_{file_hash[:12]}.pdf"
        saved.write_bytes(data)
        del data
        entries.append(
            BatchEntry(
                key=f"f{idx:04d}",
                rel_path=rel,
                file_name=parts[-1],
                size_bytes=int(getattr(up, "size", 0) or saved.stat().st_size),
                file_hash=file_hash,
                saved_path=str(saved),
            )
        )

    root_name = max(set(root_names), key=root_names.count) if root_names else "Thư mục đã chọn"
    return BatchState(
        batch_id=batch_id,
        root_name=root_name,
        entries=entries,
        subfolder_count=len(subfolders),
        skipped=skipped,
        storage_dir=str(storage_dir),
    )


# -----------------------------------------------------------------------------
# Lưu / nạp trạng thái theo file (giữ nguyên component 1 file hiện có)
# -----------------------------------------------------------------------------
def _batch_session_key(tab: str) -> str:
    return f"batch_{tab}"


def get_batch(tab: str) -> Optional[BatchState]:
    return st.session_state.get(_batch_session_key(tab))


def set_batch(tab: str, batch: Optional[BatchState]) -> None:
    st.session_state[_batch_session_key(tab)] = batch


def clear_batch(tab: str) -> None:
    """Hủy đợt quét thư mục: xóa bản sao phía server (không đụng tới file gốc của người dùng)."""
    batch = get_batch(tab)
    if batch is not None:
        shutil.rmtree(batch.storage_dir, ignore_errors=True)
    set_batch(tab, None)


def bump_dir_uploader(tab: str) -> None:
    """Đặt lại widget chọn thư mục (giải phóng các file trình duyệt đã gửi khỏi bộ nhớ phiên)."""
    key = DIR_UPLOADER_VER_KEY[tab]
    st.session_state[key] = int(st.session_state.get(key, 0)) + 1


def _capture_doc_state(tab: str) -> Dict[str, Any]:
    snap = {k: st.session_state.get(k, default) for k, default in DOC_STATE_KEYS[tab].items()}
    if tab == TAB_SCAN:
        # Không giữ bytes PDF đã sửa của file không hoạt động trong RAM; dựng lại khi mở lại file
        snap["export_pdf_bytes"] = None
    return snap


def _load_doc_state(tab: str, snapshot: Dict[str, Any]) -> None:
    for k, default in DOC_STATE_KEYS[tab].items():
        val = snapshot.get(k, default)
        if isinstance(default, dict) and val is default:
            val = {}
        st.session_state[k] = val


def sync_active_snapshot(tab: str, batch: BatchState) -> None:
    """Ghi trạng thái hiện tại (đang nằm trong session_state) vào snapshot của file đang hoạt động."""
    entry = batch.get(batch.active_key)
    if entry is not None and entry.status in (STATUS_DONE, STATUS_ERROR):
        entry.snapshot = _capture_doc_state(tab)


def activate_entry(tab: str, batch: BatchState, key: str, reset_table_page: bool = True) -> None:
    """
    Chuyển file đang hoạt động: lưu trạng thái file cũ, nạp trạng thái file mới vào session_state.
    `reset_table_page=False` khi chọn một dòng trong chế độ 'Tất cả tệp' (bảng giữ nguyên trang đang xem).
    """
    if batch.active_key == key:
        return
    sync_active_snapshot(tab, batch)
    entry = batch.get(key)
    if entry is None:
        return
    _load_doc_state(tab, entry.snapshot)
    batch.active_key = key
    if reset_table_page:
        st.session_state[TABLE_PAGE_IDX_KEY[tab]] = 0
    if tab == TAB_SCAN and st.session_state.get("pipeline_result") is not None:
        from frontend.state.state_manager import rebuild_outputs_without_ocr

        rebuild_outputs_without_ocr()


@contextmanager
def entry_state(tab: str, batch: BatchState, entry: BatchEntry) -> Iterator[None]:
    """Tạm nạp trạng thái của `entry` vào session_state (để tái sử dụng hàm dựng bảng hiện có), rồi khôi phục."""
    if entry.key == batch.active_key:
        yield
        return
    saved = {k: st.session_state.get(k, default) for k, default in DOC_STATE_KEYS[tab].items()}
    try:
        _load_doc_state(tab, entry.snapshot)
        yield
    finally:
        for k, v in saved.items():
            st.session_state[k] = v


# -----------------------------------------------------------------------------
# Xử lý hàng loạt
# -----------------------------------------------------------------------------
def _public_error(entry: BatchEntry, message: str) -> str:
    """Thông báo lỗi hiển thị cho người dùng: thay đường dẫn bản sao nội bộ của server bằng đường dẫn tương đối."""
    saved = Path(entry.saved_path)
    return message.replace(str(saved), entry.rel_path).replace(saved.name, entry.file_name)


def run_batch(
    tab: str,
    batch: BatchState,
    process_one: Callable[[BatchEntry, Tuple[int, int]], None],
    on_file_start: Optional[Callable[[int, int, BatchEntry], None]] = None,
) -> None:
    """
    Xử lý lần lượt từng file đang chờ. `process_one(entry, (thứ tự, tổng))` chạy pipeline 1 file hiện có
    (ghi kết quả vào các khóa session_state của tab); sau mỗi file lưu snapshot và cập nhật trạng thái.
    Mọi ngoại lệ của một file được ghi nhận là Lỗi và không làm dừng các file còn lại.
    """
    total = len(batch.entries)
    sync_active_snapshot(tab, batch)
    for idx, entry in enumerate(batch.entries, start=1):
        if entry.status not in (STATUS_PENDING, STATUS_RUNNING):
            continue
        entry.status = STATUS_RUNNING
        if on_file_start is not None:
            on_file_start(idx, total, entry)
        error_key = "doc_error" if tab == TAB_SCAN else "secret_doc_error"
        info_key = "doc_info" if tab == TAB_SCAN else "secret_doc_info"
        try:
            process_one(entry, (idx, total))
            err = st.session_state.get(error_key)
            if err:
                entry.status, entry.error = STATUS_ERROR, _public_error(entry, str(err))
            else:
                entry.status = STATUS_DONE
            info = st.session_state.get(info_key)
            entry.page_count = int(getattr(info, "page_count", 0) or 0)
        except Exception as exc:  # một file lỗi không làm dừng cả đợt
            entry.status, entry.error = STATUS_ERROR, _public_error(entry, f"{type(exc).__name__}: {exc}")
        if entry.status == STATUS_ERROR:
            st.session_state[error_key] = entry.error
        entry.snapshot = _capture_doc_state(tab)
    batch.active_key = None


def first_entry_to_show(batch: BatchState) -> Optional[BatchEntry]:
    done = [e for e in batch.entries if e.status == STATUS_DONE]
    return done[0] if done else (batch.entries[0] if batch.entries else None)


def scope_entries(batch: BatchState) -> List[BatchEntry]:
    """Các file thuộc phạm vi đang lọc (Tất cả tệp / một tệp cụ thể)."""
    if batch.file_filter == FILTER_ALL:
        return list(batch.entries)
    entry = batch.get(batch.file_filter)
    return [entry] if entry is not None else list(batch.entries)


def batch_summary_line(batch: BatchState) -> str:
    """Dòng tóm tắt dưới 'Đã chọn: ...' (vd. '3 thư mục con · 5 file PDF · 1 file bỏ qua')."""
    return (
        f"{batch.subfolder_count} thư mục con · {len(batch.entries)} file PDF · "
        f"{len(batch.skipped)} file bỏ qua"
    )


def batch_done_text(batch: BatchState) -> str:
    total = len(batch.entries)
    if total == 0:
        return "Thư mục không có file PDF hợp lệ để kiểm tra"
    return (
        f"Đã kiểm tra {batch.finished_count}/{total} file PDF · "
        f"{batch.done_count} thành công · {batch.error_count} lỗi"
    )


def file_filter_options(batch: BatchState) -> List[Tuple[str, str]]:
    """Tùy chọn bộ lọc 'Tệp PDF': (giá trị, nhãn) — nhãn kèm trạng thái xử lý của từng file."""
    opts = [(FILTER_ALL, f"Tất cả tệp ({len(batch.entries)})")]
    for e in batch.entries:
        suffix = "" if e.status == STATUS_DONE else f" — {e.status_label}"
        opts.append((e.key, f"{e.display_path}{suffix}"))
    return opts


# -----------------------------------------------------------------------------
# Báo cáo JSON nhiều file (Tab 2)
# -----------------------------------------------------------------------------
def build_secret_batch_report(batch: BatchState, disclaimer: str) -> Dict[str, Any]:
    """Báo cáo JSON truy vết từng file trong đợt quét thư mục (mỗi file giữ nguyên cấu trúc báo cáo 1 file cũ)."""
    files: List[Dict[str, Any]] = []
    totals = {"total_pages": 0, "pages_with_marks": 0, "total_marks_detected": 0, "review_required_pages": 0}
    for e in batch.entries:
        doc_result = e.snapshot.get("secret_doc_result")
        report = doc_result.to_report_dict() if (e.status == STATUS_DONE and doc_result is not None) else None
        if report is not None:
            for k in totals:
                totals[k] += int(report["summary"].get(k, 0))
        files.append(
            {
                "root_folder": batch.root_name,
                "relative_path": e.rel_path,
                "file_name": e.file_name,
                "page_count": e.page_count,
                "file_size_bytes": e.size_bytes,
                "processing_status": e.status_label,
                "processing_status_code": e.status,
                "error": e.error,
                "report": report,
            }
        )
    return {
        "workflow": "secret_marking_detection",
        "mode": "folder_batch",
        "disclaimer": disclaimer,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root_folder": batch.root_name,
        "summary": {
            "pdf_files": len(batch.entries),
            "succeeded_files": batch.done_count,
            "failed_files": batch.error_count,
            "skipped_files": len(batch.skipped),
            "subfolder_count": batch.subfolder_count,
            **totals,
        },
        "skipped_files": [{"relative_path": p, "reason": r} for p, r in batch.skipped],
        "files": files,
    }
