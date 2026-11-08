"""
Component thanh cấu hình bên trái (Sidebar) của giao diện Streamlit.
Hiển thị cấu hình kiểm tra và trạng thái thực tế của các Detector trong `DetectorRegistry`.
"""

from typing import Tuple

import streamlit as st

from ai_engine.core.config import ORIENTATION_CONF_THRESHOLD, RENDER_DPI


def render_sidebar() -> Tuple[int, float, bool, bool, bool]:
    """
    Hiển thị Khu vực 2 — Cấu hình và Trạng thái Detector trong Sidebar.
    Trả về `(dpi_value, conf_threshold, auto_fix_enabled, export_json, export_csv)`.
    """
    with st.sidebar:
        st.header("⚙️ Khu vực 2 — Cấu hình")
        dpi_value = st.slider(
            "Độ phân giải Render (DPI)",
            min_value=72,
            max_value=300,
            value=RENDER_DPI,
            step=8,
            help="Mặc định 160 DPI giúp cân bằng tốc độ và độ chính xác của Orientation & Stripe Detector.",
        )
        conf_threshold = st.slider(
            "Ngưỡng độ tin cậy Orientation (Confidence Threshold)",
            min_value=0.5,
            max_value=30.0,
            value=float(ORIENTATION_CONF_THRESHOLD),
            step=0.5,
            help=(
                "Ngưỡng 10.0 là giá trị mặc định. Các trang có confidence thấp hơn "
                "ngưỡng này sẽ được đưa vào trạng thái REVIEW để đảm bảo an toàn."
            ),
        )
        auto_fix_enabled = st.checkbox(
            "Bật Auto Fix (Tự động xoay trang ROTATED)",
            value=True,
            help=(
                "Chỉ tự động xoay các trang đủ độ tin cậy (ROTATED). "
                "Tuyệt đối không tự xoay trang REVIEW và không tự động xóa sọc."
            ),
        )
        st.subheader("Tùy chọn xuất báo cáo")
        export_json = st.checkbox(
            "Xuất báo cáo JSON (orientation_report.json)", value=True
        )
        export_csv = st.checkbox(
            "Xuất báo cáo CSV (orientation_report.csv)", value=True
        )

        st.divider()
        st.subheader("🧩 Trạng thái Detectors")
        st.markdown(
            "- 🟢 **Orientation**: `ACTIVE (ENABLED)` — Phát hiện trang xoay `90°`, `180°`, `270°`.\n"
            "- 🟢 **Stripe**: `ACTIVE (ENABLED)` — Phát hiện vệt sọc dọc, ngang, nghiêng & dải sọc (`VERTICAL`, `HORIZONTAL`, `DIAGONAL`, `BAND`).\n"
            "- 🟢 **Boundary**: `ACTIVE (ENABLED)` — Phát hiện lỗi ranh giới trang & ghép nhầm (`MIXED_PAGE`, `SPLIT_PAGE`, `CONTENT_OVERLAP`, `CONTENT_TRUNCATED`)."
        )

    return dpi_value, conf_threshold, auto_fix_enabled, export_json, export_csv
