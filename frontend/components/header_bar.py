"""
Component Header Bar chuẩn thiết kế (`src/pdf_inspector/ui/components/header_bar.py`).
Chiều cao ~75px, nền trắng tràn ngang:
- Trái: Icon + Tiêu đề + Phụ đề động theo chế độ.
- Giữa: Bộ chuyển hai chế độ độc lập ('Kiểm tra & sửa lỗi PDF' | 'Kiểm tra dấu độ mật').
- Phải: Nút hành động xuất kết quả tương ứng ('Tải PDF đã sửa' ở Tab 1 | 'Xuất báo cáo JSON' ở Tab 2).
  Khi quét thư mục ở Tab 1, cùng nút đó đổi nhãn thành 'Tải kết quả ZIP' (giữ cấu trúc thư mục + báo cáo tổng hợp).
"""

import json
from pathlib import Path
import streamlit as st

from backend.services.secret_scan_service import TECHNICAL_DISCLAIMER, SecretScanService
from backend.workers.batch_export import build_scan_zip_bytes, collect_scan_export_items
from backend.workers.batch_manager import (
    TAB_SCAN,
    TAB_SECRET,
    build_secret_batch_report,
    get_batch,
    sync_active_snapshot,
)
from frontend.styles.styles import SVG_HEADER_DOC_SEARCH, clean_html


def _on_download_clicked() -> None:
    """Đánh dấu người dùng đã xuất phiên bản PDF hiện tại."""
    st.session_state["has_exported_current"] = True
    st.session_state["has_unexported_changes"] = False


def render_header_bar() -> None:
    """Hiển thị khu vực 1 — Header chính của Dashboard với 2 chế độ làm việc độc lập."""
    active_mode: str = st.session_state.get("active_mode", "SCAN_QUALITY")
    is_scan_mode = active_mode == "SCAN_QUALITY"

    subtitle_text = (
        "Tải PDF lên, hệ thống tự phát hiện lỗi và hỗ trợ sửa file."
        if is_scan_mode
        else "Nhận diện dấu MẬT, TỐI MẬT, TUYỆT MẬT độc lập — không tự động tách PDF."
    )

    with st.container(key="top_header_bar"):
        col_brand, col_tabs, col_action = st.columns(
            [4.5, 3.5, 2.0], vertical_alignment="center"
        )

        with col_brand:
            st.markdown(
                clean_html(
                    f"""
                    <div class="hdr-brand-wrap">
                      {SVG_HEADER_DOC_SEARCH}
                      <div class="hdr-brand-text">
                        <div class="hdr-title">PDF Scan Quality Inspector</div>
                        <div class="hdr-subtitle">{subtitle_text}</div>
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        with col_tabs:
            with st.container(horizontal=True, key="hdr_mode_switcher"):
                tab1_key = (
                    "mode_tab_act_scan" if is_scan_mode else "mode_tab_inact_scan"
                )
                tab2_key = (
                    "mode_tab_inact_secret"
                    if is_scan_mode
                    else "mode_tab_act_secret"
                )
                if st.button("Kiểm tra & sửa lỗi PDF", key=tab1_key):
                    if not is_scan_mode:
                        st.session_state["active_mode"] = "SCAN_QUALITY"
                        st.rerun()
                if st.button("Kiểm tra dấu độ mật", key=tab2_key):
                    if is_scan_mode:
                        st.session_state["active_mode"] = "SECRET_DETECTION"
                        st.rerun()

        with col_action:
            with st.container(key="hdr_download_wrap"):
                if is_scan_mode:
                    export_bytes = st.session_state.get("export_pdf_bytes")
                    export_name = (
                        st.session_state.get("export_file_name") or "corrected.pdf"
                    )
                    scan_batch = get_batch(TAB_SCAN)
                    if scan_batch is not None and scan_batch.entries and not scan_batch.has_pending:
                        # Quét thư mục: cùng nút, đổi nhãn; ZIP chỉ được tạo khi người dùng bấm tải
                        sync_active_snapshot(TAB_SCAN, scan_batch)
                        zip_items = collect_scan_export_items(scan_batch)
                        zip_root = scan_batch.root_name
                        zip_skipped = list(scan_batch.skipped)
                        st.download_button(
                            label="Tải kết quả ZIP",
                            data=lambda: build_scan_zip_bytes(zip_root, zip_items, zip_skipped),
                            file_name=f"{zip_root}_ket_qua.zip",
                            mime="application/zip",
                            icon=":material/download:",
                            key="hdr_download_btn",
                            on_click="ignore",
                        )
                    elif scan_batch is not None:
                        st.button(
                            label="Tải kết quả ZIP",
                            icon=":material/download:",
                            disabled=True,
                            key="hdr_download_disabled",
                            help="Chờ kiểm tra xong toàn bộ file PDF trong thư mục để tải kết quả.",
                        )
                    elif export_bytes:
                        st.download_button(
                            label="Tải PDF đã sửa",
                            data=export_bytes,
                            file_name=export_name,
                            mime="application/pdf",
                            icon=":material/download:",
                            key="hdr_download_btn",
                            on_click=_on_download_clicked,
                        )
                    else:
                        st.button(
                            label="Tải PDF đã sửa",
                            icon=":material/download:",
                            disabled=True,
                            key="hdr_download_disabled",
                            help="Tải file PDF lên để hệ thống quét và tạo bản PDF đã sửa.",
                        )
                else:
                    sec_res = st.session_state.get("secret_doc_result")
                    sec_batch = get_batch(TAB_SECRET)
                    if sec_batch is not None and sec_batch.entries and not sec_batch.has_pending:
                        # Quét thư mục: báo cáo truy vết từng file (mỗi file giữ cấu trúc báo cáo 1 file cũ)
                        sync_active_snapshot(TAB_SECRET, sec_batch)
                        payload = build_secret_batch_report(sec_batch, TECHNICAL_DISCLAIMER)
                        st.download_button(
                            label="Xuất báo cáo JSON",
                            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                            file_name=f"{sec_batch.root_name}_secret_report.json",
                            mime="application/json",
                            icon=":material/download:",
                            key="hdr_download_btn",
                        )
                    elif sec_batch is not None:
                        st.button(
                            label="Xuất báo cáo JSON",
                            icon=":material/download:",
                            disabled=True,
                            key="hdr_download_disabled",
                            help="Chờ kiểm tra xong toàn bộ file PDF trong thư mục để xuất báo cáo.",
                        )
                    elif sec_res is not None:
                        stem = Path(
                            st.session_state.get("secret_file_name") or "document.pdf"
                        ).stem
                        json_bytes = SecretScanService.export_report_json_bytes(sec_res)
                        st.download_button(
                            label="Xuất báo cáo JSON",
                            data=json_bytes,
                            file_name=f"{stem}_secret_report.json",
                            mime="application/json",
                            icon=":material/download:",
                            key="hdr_download_btn",
                        )
                    else:
                        st.button(
                            label="Xuất báo cáo JSON",
                            icon=":material/download:",
                            disabled=True,
                            key="hdr_download_disabled",
                            help="Tải file PDF lên ở chế độ Kiểm tra dấu độ mật để xuất báo cáo kết quả.",
                        )
