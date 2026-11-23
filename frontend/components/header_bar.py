"""
Component Header Bar theo Thiết kế Cuối cùng Đã Duyệt (`frontend/components/header_bar.py`).

Tuân thủ Mục 4 & Ảnh tham chiếu:
- Chiều cao 60–64px, nền trắng.
- Bên trái: Icon menu `☰` + Tên `PDF Scan Quality Inspector` (tuyệt đối không có subtitle bên dưới).
- Bên phải (nhóm sát bên phải):
  - Nút `Xuất báo cáo` (nền trắng, viền xanh `#0866E8`).
  - Nút `Xuất PDF đã sửa` (màu xanh dương `#0866E8`).
  - Icon trợ giúp `?` và icon người dùng.
- Chỉ bật nút xuất khi có kết quả và dữ liệu đầu ra tương ứng.
"""

import json
from pathlib import Path
from typing import Optional

import streamlit as st

from backend.schemas.schemas import PipelineResult
from backend.services.report_service import ReportGenerator
from backend.services.secret_scan_service import TECHNICAL_DISCLAIMER, SecretScanService
from backend.workers.batch_export import build_scan_zip_bytes, collect_scan_export_items
from backend.workers.batch_manager import (
    TAB_SCAN,
    TAB_SECRET,
    build_secret_batch_report,
    get_batch,
    sync_active_snapshot,
)

SVG_USER_AVATAR = """
<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" title="Phiên làm việc cục bộ">
  <circle cx="16" cy="16" r="15" fill="#E2E8F0" stroke="#CBD5E1" stroke-width="1"/>
  <circle cx="16" cy="12.5" r="4.5" fill="#64748B"/>
  <path d="M8.5 25.5C9.8 21.8 12.6 20 16 20C19.4 20 22.2 21.8 23.5 25.5" fill="#64748B"/>
</svg>
"""


def _on_download_clicked() -> None:
    """Đánh dấu người dùng đã xuất phiên bản PDF hiện tại."""
    st.session_state["has_exported_current"] = True
    st.session_state["has_unexported_changes"] = False


@st.dialog("Hướng dẫn sử dụng PDF Scan Quality Inspector")
def _open_help_dialog() -> None:
    st.markdown(
        """
        **1. Chế độ Kiểm tra lỗi PDF**
        - Chọn **1 file PDF** hoặc **Chọn thư mục** chứa nhiều file PDF ở vùng Upload phía trên.
        - Hệ thống tự động phát hiện 3 nhóm lỗi: **Sọc (Stripe)**, **Lẫn trang (Boundary)** và **Xoay (Orientation)**.
        - Chọn từng trang ở bảng **Kết quả kiểm tra** bên trái để đối chiếu trực quan ở khung **Xem trang** bên phải và thực hiện **Chấp nhận**, **Bỏ qua** hoặc **Sửa thủ công**.

        **2. Chế độ Dấu độ mật**
        - Chuyển sang mục **Dấu độ mật** ở thanh Sidebar bên trái.
        - Hệ thống thực hiện nhận diện dấu `MẬT`, `TỐI MẬT`, `TUYỆT MẬT` độc lập (không chạy bộ phát hiện xoay/sọc/lẫn trang và không tự động tách PDF).
        """
    )


def _get_active_report_payload(is_scan_mode: bool) -> tuple[Optional[bytes], str]:
    """Lấy dữ liệu báo cáo JSON thực tế tương ứng với chế độ đang mở."""
    if is_scan_mode:
        scan_batch = get_batch(TAB_SCAN)
        if scan_batch is not None and scan_batch.entries and not scan_batch.has_pending:
            sync_active_snapshot(TAB_SCAN, scan_batch)
            items = collect_scan_export_items(scan_batch)
            batch_summary = {
                "root_folder": scan_batch.root_name,
                "total_files": len(scan_batch.entries),
                "done_files": scan_batch.done_count,
                "error_files": scan_batch.error_count,
                "files": [
                    {
                        "relative_path": it.rel_path,
                        "report": it.result.report_data if it.result else None,
                    }
                    for it in items
                ],
            }
            return (
                json.dumps(batch_summary, ensure_ascii=False, indent=2).encode("utf-8"),
                f"{scan_batch.root_name}_bao_cao.json",
            )

        result: Optional[PipelineResult] = st.session_state.get("pipeline_result")
        if result is not None and result.report_data:
            stem = Path(st.session_state.get("display_file_name") or "document.pdf").stem
            return (
                ReportGenerator.to_json_string(result.report_data).encode("utf-8"),
                f"{stem}_bao_cao.json",
            )
        return None, "bao_cao.json"

    sec_batch = get_batch(TAB_SECRET)
    if sec_batch is not None and sec_batch.entries and not sec_batch.has_pending:
        sync_active_snapshot(TAB_SECRET, sec_batch)
        payload = build_secret_batch_report(sec_batch, TECHNICAL_DISCLAIMER)
        return (
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            f"{sec_batch.root_name}_secret_report.json",
        )

    sec_res = st.session_state.get("secret_doc_result")
    if sec_res is not None:
        stem = Path(st.session_state.get("secret_file_name") or "document.pdf").stem
        return (
            SecretScanService.export_report_json_bytes(sec_res),
            f"{stem}_secret_report.json",
        )
    return None, "secret_report.json"


def render_header_bar() -> None:
    """Hiển thị Header 60–64px không có subtitle, khớp Mục 4 và ảnh tham chiếu."""
    active_mode: str = st.session_state.get("active_mode", "SCAN_QUALITY")
    is_scan_mode = active_mode == "SCAN_QUALITY"

    report_bytes, report_name = _get_active_report_payload(is_scan_mode)

    with st.container(key="top_header_bar"):
        col_left, col_right = st.columns([4.8, 5.2], vertical_alignment="center")

        with col_left:
            with st.container(horizontal=True, vertical_alignment="center", key="hdr_left_group"):
                if st.button(
                    "",
                    icon=":material/menu:",
                    key="hdr_menu_toggle_btn",
                    help="Thu gọn / Mở rộng thanh điều hướng bên trái",
                ):
                    st.session_state["sidebar_collapsed"] = not bool(
                        st.session_state.get("sidebar_collapsed", False)
                    )
                    st.rerun()
                st.markdown(
                    '<div class="hdr-title">PDF Scan Quality Inspector</div>',
                    unsafe_allow_html=True,
                )

        with col_right:
            with st.container(horizontal=True, vertical_alignment="center", key="hdr_right_group"):
                # 1. Nút Xuất báo cáo (Nền trắng, viền xanh)
                with st.container(key="hdr_report_btn_wrap"):
                    if report_bytes is not None:
                        st.download_button(
                            label="Xuất báo cáo",
                            data=report_bytes,
                            file_name=report_name,
                            mime="application/json",
                            icon=":material/description:",
                            key="hdr_report_btn_active",
                            help="Tải xuống báo cáo kết quả kiểm tra (JSON)",
                        )
                    else:
                        st.button(
                            label="Xuất báo cáo",
                            icon=":material/description:",
                            disabled=True,
                            key="hdr_report_btn_disabled",
                            help="Chỉ bật khi đã có kết quả kiểm tra tài liệu PDF",
                        )

                # 2. Nút Xuất PDF đã sửa (Màu xanh dương)
                with st.container(key="hdr_download_wrap"):
                    export_bytes = st.session_state.get("export_pdf_bytes")
                    export_name = st.session_state.get("export_file_name") or "corrected.pdf"
                    scan_batch = get_batch(TAB_SCAN)

                    if is_scan_mode and scan_batch is not None and scan_batch.entries and not scan_batch.has_pending:
                        sync_active_snapshot(TAB_SCAN, scan_batch)
                        zip_items = collect_scan_export_items(scan_batch)
                        zip_root = scan_batch.root_name
                        zip_skipped = list(scan_batch.skipped)
                        st.download_button(
                            label="Xuất PDF đã sửa",
                            data=lambda: build_scan_zip_bytes(zip_root, zip_items, zip_skipped),
                            file_name=f"{zip_root}_ket_qua.zip",
                            mime="application/zip",
                            icon=":material/download:",
                            key="hdr_download_btn",
                            on_click="ignore",
                            help="Tải xuống gói ZIP các file PDF đã sửa trong thư mục",
                        )
                    elif is_scan_mode and export_bytes:
                        st.download_button(
                            label="Xuất PDF đã sửa",
                            data=export_bytes,
                            file_name=export_name,
                            mime="application/pdf",
                            icon=":material/download:",
                            key="hdr_download_btn",
                            on_click=_on_download_clicked,
                            help="Tải xuống bản PDF đã sửa (không ghi đè file gốc)",
                        )
                    else:
                        st.button(
                            label="Xuất PDF đã sửa",
                            icon=":material/download:",
                            disabled=True,
                            key="hdr_download_disabled",
                            help=(
                                "Chỉ bật khi có kết quả PDF đã sửa ở chế độ Kiểm tra lỗi PDF"
                                if is_scan_mode
                                else "Chế độ Dấu độ mật không chỉnh sửa hoặc tách file PDF"
                            ),
                        )

                # 3. Icon trợ giúp
                with st.container(key="hdr_help_btn_wrap"):
                    if st.button(
                        "",
                        icon=":material/help_outline:",
                        key="hdr_help_btn",
                        help="Hướng dẫn sử dụng ứng dụng",
                    ):
                        _open_help_dialog()

                # 4. Avatar tài khoản phiên làm việc
                with st.container(key="hdr_user_btn_wrap"):
                    st.markdown(
                        f'<div class="hdr-user-avatar" title="Phiên làm việc cục bộ">{SVG_USER_AVATAR}</div>',
                        unsafe_allow_html=True,
                    )
