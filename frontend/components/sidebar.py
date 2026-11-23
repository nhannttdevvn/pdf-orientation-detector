"""
Component Sidebar cố định bên trái theo Thiết kế Cuối cùng Đã Duyệt (`frontend/components/sidebar.py`).

Tuân thủ Mục 3 & Ảnh tham chiếu:
- Nền xanh navy đậm (`#142746`), rộng ~200–220px trên desktop, cố định bên trái, cao toàn bộ viewport.
- Đầu sidebar: Biểu tượng tài liệu kết hợp kính lúp trên nền xanh `#0D66E8` + Tên `PDF Scan Quality Inspector`.
- 5 mục chức năng:
  1. Kiểm tra lỗi PDF — icon file-search
  2. Dấu độ mật — icon shield
  3. Báo cáo — icon chart-column (disabled, thông báo rõ)
  4. Lịch sử — icon history (disabled, thông báo rõ)
  5. Cài đặt — icon settings (disabled, thông báo rõ)
- Cuối sidebar hiển thị icon `ⓘ` + phiên bản ứng dụng (`v1.0.0`) cố định ở góc dưới trái.
"""

from typing import Tuple

import streamlit as st

from ai_engine.core.config import ORIENTATION_CONF_THRESHOLD, RENDER_DPI
from frontend.styles.styles import clean_html

APP_VERSION: str = "v1.0.0"

SVG_SIDEBAR_DOC_LENS = """
<svg width="24" height="24" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <rect x="4" y="3" width="15" height="20" rx="2.5" stroke="#FFFFFF" stroke-width="1.9"/>
  <path d="M14.5 3V7H19" stroke="#FFFFFF" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
  <line x1="7.5" y1="10" x2="13.5" y2="10" stroke="#FFFFFF" stroke-width="1.7" stroke-linecap="round"/>
  <line x1="7.5" y1="13.5" x2="14.5" y2="13.5" stroke="#FFFFFF" stroke-width="1.7" stroke-linecap="round"/>
  <circle cx="18.5" cy="18.5" r="4.8" fill="#0D66E8" stroke="#FFFFFF" stroke-width="1.9"/>
  <line x1="22" y1="22" x2="25.2" y2="25.2" stroke="#FFFFFF" stroke-width="2.1" stroke-linecap="round"/>
</svg>
"""

SVG_SIDEBAR_INFO = """
<svg width="16" height="16" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <circle cx="10" cy="10" r="8" stroke="#94A3B8" stroke-width="1.6"/>
  <line x1="10" y1="9" x2="10" y2="13.5" stroke="#94A3B8" stroke-width="1.6" stroke-linecap="round"/>
  <circle cx="10" cy="6.5" r="1" fill="#94A3B8"/>
</svg>
"""


def render_sidebar() -> Tuple[int, float, bool, bool, bool]:
    """Hiển thị Sidebar cố định bên trái (#142746) và trả về tuple cấu hình mặc định."""
    active_mode: str = st.session_state.get("active_mode", "SCAN_QUALITY")
    is_scan_mode = active_mode == "SCAN_QUALITY"

    with st.sidebar:
        st.markdown(
            clean_html(
                f"""
                <div class="sb-brand-header">
                  <div class="sb-brand-icon">{SVG_SIDEBAR_DOC_LENS}</div>
                  <div class="sb-brand-title">PDF Scan<br/>Quality Inspector</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        with st.container(key="sb_nav_menu"):
            scan_key = "sb_menu_act_scan" if is_scan_mode else "sb_menu_inact_scan"
            secret_key = "sb_menu_inact_secret" if is_scan_mode else "sb_menu_act_secret"

            if st.button(
                "Kiểm tra lỗi PDF",
                icon=":material/find_in_page:",
                key=scan_key,
                width="stretch",
                help="Kiểm tra lỗi sọc scan, lẫn trang và sai hướng xoay PDF",
            ):
                if not is_scan_mode:
                    st.session_state["active_mode"] = "SCAN_QUALITY"
                    st.rerun()

            if st.button(
                "Dấu độ mật",
                icon=":material/verified_user:",
                key=secret_key,
                width="stretch",
                help="Phát hiện dấu độ mật (MẬT, TỐI MẬT, TUYỆT MẬT) độc lập",
            ):
                if is_scan_mode:
                    st.session_state["active_mode"] = "SECRET_DETECTION"
                    st.rerun()

            st.button(
                "Báo cáo",
                icon=":material/bar_chart:",
                key="sb_menu_dis_report",
                disabled=True,
                width="stretch",
                help="Trang tổng hợp báo cáo chưa có backend riêng. Vui lòng dùng nút 'Xuất báo cáo' trên thanh Header.",
            )

            st.button(
                "Lịch sử",
                icon=":material/history:",
                key="sb_menu_dis_history",
                disabled=True,
                width="stretch",
                help="Chức năng lưu lịch sử phiên làm việc chưa được triển khai trong backend hiện tại.",
            )

            st.button(
                "Cài đặt",
                icon=":material/settings:",
                key="sb_menu_dis_settings",
                disabled=True,
                width="stretch",
                help="Trang cài đặt chưa được triển khai trong phiên bản hiện tại.",
            )

        st.markdown(
            clean_html(
                f"""
                <div class="sb-version-footer">
                  {SVG_SIDEBAR_INFO}
                  <span>{APP_VERSION}</span>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    return RENDER_DPI, float(ORIENTATION_CONF_THRESHOLD), True, True, True
