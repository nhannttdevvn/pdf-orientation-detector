"""
Xuất kết quả ZIP khi quét thư mục ở Tab 1 — Kiểm tra & sửa lỗi PDF (`src/pdf_inspector/ui/batch_export.py`).

Quy định đầu ra (ghi kèm trong ZIP, file `DOC_TRUOC.txt`):
- Giữ nguyên cấu trúc thư mục tương đối của đầu vào (vd. `HO_SO/Ho_so_A/Phu_luc/03.pdf`).
- File có trang đã được sửa (xoay chắc chắn / góc thủ công / tách-ghép ĐÃ phê duyệt) -> PDF đã sửa, cùng đường dẫn.
- File có lỗi chưa được sửa tự động (cần người kiểm tra, lỗi sọc...) và file không phát hiện lỗi -> giữ nguyên PDF gốc,
  cùng đường dẫn (không làm mất file nào trong bộ hồ sơ).
- File xử lý thất bại và file không phải PDF -> không có trong ZIP, chỉ được liệt kê trong báo cáo kèm lý do.
- Không bao giờ gộp các PDF thành một file, không sửa/ghi đè PDF gốc của người dùng.

`collect_scan_export_items` chạy trên luồng giao diện (đọc session_state); `build_scan_zip_bytes` là hàm thuần,
chạy khi người dùng bấm tải (luồng riêng của `st.download_button`), ghi lần lượt từng file vào ZIP tạm trên đĩa.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import zipfile

import streamlit as st

from backend.workers.batch_manager import STATUS_DONE, TAB_SCAN, BatchState, entry_state

FILE_FIXED = "Đã sửa"
FILE_FIXED_PARTIAL = "Đã sửa một phần — còn trang cần kiểm tra"
FILE_UNFIXED = "Có lỗi chưa sửa tự động — giữ nguyên bản gốc"
FILE_CLEAN = "Không phát hiện lỗi — giữ nguyên bản gốc"
FILE_FAILED = "Xử lý thất bại — không có trong ZIP"
FILE_SKIPPED = "Bỏ qua — không phải PDF hợp lệ"

REPORT_BASENAME = "BAO_CAO_KET_QUA"

README_TEXT = f"""KẾT QUẢ KIỂM TRA & SỬA LỖI PDF THEO THƯ MỤC
=============================================

Cấu trúc thư mục trong ZIP giữ nguyên như thư mục đầu vào. Mỗi file PDF hợp lệ đều có mặt đúng đường dẫn cũ:

- {FILE_FIXED}: PDF đã được sửa (xoay đúng chiều các trang phát hiện chắc chắn, góc xoay chỉnh thủ công,
  hoặc thao tác tách/ghép ranh giới trang ĐÃ được người dùng phê duyệt trên giao diện).
- {FILE_FIXED_PARTIAL}: như trên, nhưng còn trang cần người dùng kiểm tra (chưa tự sửa).
- {FILE_UNFIXED}: có trang lỗi nhưng không đủ chắc chắn để tự sửa, hoặc là lỗi không tự sửa được
  (ví dụ lỗi sọc) -> bản trong ZIP là PDF gốc, chưa thay đổi.
- {FILE_CLEAN}: bản trong ZIP là PDF gốc, chưa thay đổi.
- {FILE_FAILED}: không mở/kiểm tra được (ví dụ PDF hỏng) -> chỉ liệt kê trong báo cáo kèm lý do.
- {FILE_SKIPPED}: file không phải PDF trong thư mục -> chỉ liệt kê trong báo cáo.

Chi tiết từng file (trang lỗi, trang đã sửa, trang cần kiểm tra): {REPORT_BASENAME}.csv (mở bằng Excel) và {REPORT_BASENAME}.json.
Công cụ không gộp các PDF thành một file và không sửa đổi PDF gốc.
"""


@dataclass
class ScanExportItem:
    """Dữ liệu của một file cần cho việc xuất ZIP (thu thập trên luồng giao diện)."""

    rel_path: str
    file_name: str
    status: str
    error: Optional[str]
    saved_path: str
    file_tag: str
    page_count: int
    result: Any = None
    manual_overrides: Dict[int, int] = field(default_factory=dict)
    approved_ops: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    problem_pages: List[int] = field(default_factory=list)
    review_pages: List[int] = field(default_factory=list)
    stripe_pages: List[int] = field(default_factory=list)
    boundary_pages: List[int] = field(default_factory=list)
    rotation_pages: List[int] = field(default_factory=list)


def collect_scan_export_items(batch: BatchState) -> List[ScanExportItem]:
    """Thu thập trạng thái từng file (dùng lại `build_problem_page_rows` hiện có với trạng thái riêng của file)."""
    from frontend.state.state_manager import build_problem_page_rows

    items: List[ScanExportItem] = []
    for e in batch.entries:
        item = ScanExportItem(
            rel_path=e.rel_path,
            file_name=e.file_name,
            status=e.status,
            error=e.error,
            saved_path=e.saved_path,
            file_tag=f"{e.key}_{e.file_hash[:10]}",
            page_count=e.page_count,
        )
        if e.status == STATUS_DONE:
            with entry_state(TAB_SCAN, batch, e):
                result = st.session_state.get("pipeline_result")
                rows = build_problem_page_rows(result)
                item.result = result
                item.manual_overrides = dict(st.session_state.get("manual_overrides") or {})
                item.approved_ops = dict(st.session_state.get("approved_boundary_ops") or {})
            item.problem_pages = [r.page_number for r in rows]
            item.review_pages = [r.page_number for r in rows if r.needs_review_now]
            item.stripe_pages = [r.page_number for r in rows if r.stripe_state != "NONE"]
            item.boundary_pages = [r.page_number for r in rows if r.boundary_state != "NONE"]
            item.rotation_pages = [r.page_number for r in rows if r.rotation_state != "NONE"]
        items.append(item)
    return items


def _pages_str(pages: List[int]) -> str:
    return ", ".join(str(p) for p in sorted(set(pages)))


def build_scan_zip_bytes(
    root_name: str,
    items: List[ScanExportItem],
    skipped: List[Tuple[str, str]],
) -> bytes:
    """
    Tạo ZIP kết quả (hàm thuần, không dùng session_state): dựng PDF đã sửa cho từng file bằng đúng logic xuất
    của chế độ 1 file (`build_corrected_pdf`), ghi lần lượt vào ZIP tạm trên đĩa, kèm báo cáo CSV/JSON và hướng dẫn.
    """
    from frontend.state.state_manager import build_corrected_pdf

    records: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="kq_zip_") as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / "ket_qua.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for item in items:
                rec: Dict[str, Any] = {
                    "relative_path": item.rel_path,
                    "file_name": item.file_name,
                    "page_count": item.page_count,
                    "error_pages": sorted(set(item.problem_pages)),
                    "fixed_pages": [],
                    "review_pages": sorted(set(item.review_pages)),
                    "stripe_pages": sorted(set(item.stripe_pages)),
                    "boundary_pages": sorted(set(item.boundary_pages)),
                    "rotation_pages": sorted(set(item.rotation_pages)),
                    "output_in_zip": None,
                    "error": item.error,
                }
                if item.status != STATUS_DONE or item.result is None:
                    rec["status"] = FILE_FAILED
                    records.append(rec)
                    continue
                try:
                    updated, final_pdf = build_corrected_pdf(
                        result=copy.deepcopy(item.result),
                        manual_overrides=dict(item.manual_overrides),
                        approved_ops_map=copy.deepcopy(item.approved_ops),
                        output_dir=tmp_dir / "pdf",
                        file_tag=item.file_tag,
                    )
                    fixed = sorted(
                        {d.page_number for d in updated.decisions if d.corrected}
                        | {int(k) for k in item.approved_ops}
                    )
                    rec["fixed_pages"] = fixed
                    if fixed and final_pdf.exists():
                        zf.write(final_pdf, item.rel_path)
                        rec["status"] = FILE_FIXED_PARTIAL if item.review_pages else FILE_FIXED
                    else:
                        # Không có trang nào được sửa: giữ nguyên PDF gốc (bản sao phía server, nội dung trùng file gốc)
                        zf.write(item.saved_path, item.rel_path)
                        rec["status"] = FILE_UNFIXED if item.problem_pages else FILE_CLEAN
                    rec["output_in_zip"] = item.rel_path
                    final_pdf.unlink(missing_ok=True)
                except Exception as exc:  # một file lỗi khi xuất không làm hỏng cả ZIP
                    rec["status"] = FILE_FAILED
                    rec["error"] = f"Lỗi khi tạo PDF đã sửa: {type(exc).__name__}: {exc}"
                records.append(rec)

            for rel, reason in skipped:
                records.append(
                    {
                        "relative_path": rel,
                        "file_name": rel.split("/")[-1],
                        "status": FILE_SKIPPED,
                        "page_count": 0,
                        "error_pages": [],
                        "fixed_pages": [],
                        "review_pages": [],
                        "stripe_pages": [],
                        "boundary_pages": [],
                        "rotation_pages": [],
                        "output_in_zip": None,
                        "error": reason,
                    }
                )

            summary = {
                "pdf_files": len(items),
                "fixed_files": sum(1 for r in records if r["status"] in (FILE_FIXED, FILE_FIXED_PARTIAL)),
                "unfixed_error_files": sum(1 for r in records if r["status"] == FILE_UNFIXED),
                "clean_files": sum(1 for r in records if r["status"] == FILE_CLEAN),
                "failed_files": sum(1 for r in records if r["status"] == FILE_FAILED),
                "skipped_files": len(skipped),
                "total_pages": sum(int(r["page_count"]) for r in records),
                "error_pages": sum(len(r["error_pages"]) for r in records),
                "fixed_pages": sum(len(r["fixed_pages"]) for r in records),
                "review_pages": sum(len(r["review_pages"]) for r in records),
            }
            report = {
                "workflow": "scan_quality_inspection",
                "mode": "folder_batch",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "root_folder": root_name,
                "summary": summary,
                "files": records,
            }
            zf.writestr(f"{REPORT_BASENAME}.json", json.dumps(report, ensure_ascii=False, indent=2))

            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                [
                    "STT", "Đường dẫn tương đối", "Tên file", "Kết quả", "Số trang", "Trang lỗi",
                    "Trang đã sửa", "Trang cần kiểm tra", "Trang lỗi sọc", "Trang lẫn trang", "Trang xoay",
                    "File trong ZIP", "Lỗi / lý do",
                ]
            )
            for i, r in enumerate(records, start=1):
                writer.writerow(
                    [
                        i, r["relative_path"], r["file_name"], r["status"], r["page_count"],
                        _pages_str(r["error_pages"]), _pages_str(r["fixed_pages"]), _pages_str(r["review_pages"]),
                        _pages_str(r["stripe_pages"]), _pages_str(r["boundary_pages"]), _pages_str(r["rotation_pages"]),
                        r["output_in_zip"] or "", r["error"] or "",
                    ]
                )
            zf.writestr(f"{REPORT_BASENAME}.csv", buf.getvalue().encode("utf-8-sig"))
            zf.writestr("DOC_TRUOC.txt", README_TEXT)
        return zip_path.read_bytes()
