"""
Component Khu vực 5 — Footer Information Bar (`src/pdf_inspector/ui/components/footer_bar.py`).
"""

import streamlit as st

from pdf_inspector.ui.styles import SVG_FOOTER_INFO, SVG_FOOTER_LOCK, clean_html


def render_footer_bar() -> None:
    """Hiển thị thanh thông tin cuối giao diện tương ứng với chế độ làm việc đang chọn."""
    active_mode = st.session_state.get("active_mode", "SCAN_QUALITY")
    if active_mode == "SECRET_DETECTION":
        left_note = (
            "Chỉ hiển thị trang phát hiện dấu hoặc nghi vấn. Kết quả mang tính hỗ trợ kỹ thuật, không phải kết luận pháp lý."
        )
        right_note = "Xử lý offline 100% · Chưa tự động tách PDF theo độ mật."
    else:
        left_note = (
            "Chỉ hiển thị các trang bị lỗi. Những trang bình thường không xuất hiện trong bảng."
        )
        right_note = "PDF gốc luôn được giữ nguyên."

    st.markdown(
        clean_html(
            f"""
            <div class="footer-info-bar">
              <div class="footer-info-item">
                {SVG_FOOTER_INFO}
                <span>{left_note}</span>
              </div>
              <div class="footer-info-item">
                {SVG_FOOTER_LOCK}
                <span>{right_note}</span>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
