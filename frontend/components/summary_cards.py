"""
Component Khu vực 3 — Summary Statistics (4 Cards nằm ngang chia đều chiều rộng)
(`src/pdf_inspector/ui/components/summary_cards.py`).
"""

from typing import Dict

import streamlit as st

from frontend.styles.styles import (
    SVG_STAT_PAGES,
    SVG_STAT_ROTATE,
    SVG_STAT_SEARCH,
    SVG_STAT_WARNING,
    clean_html,
)


def render_summary_cards(stats: Dict[str, int]) -> None:
    """
    Hiển thị 4 Summary Cards lấy trực tiếp từ dữ liệu thực của backend:
    1. Tổng số trang (Navy)
    2. Trang lỗi (Đỏ — mỗi trang đa lỗi chỉ tính 1 lần)
    3. Đã tự xoay (Xanh lá — chỉ tính các trang thực sự đã xoay trong PDF xuất)
    4. Cần kiểm tra (Cam — số trang đang cần người dùng kiểm duyệt)
    """
    total_pages = int(stats.get("total_pages", 0))
    error_pages = int(stats.get("error_pages", 0))
    auto_rotated = int(stats.get("auto_rotated_pages", 0))
    review_pages = int(stats.get("review_pages", 0))

    st.markdown(
        clean_html(
            f"""
            <div class="summary-cards-grid">
              <div class="summary-stat-card">
                <div class="summary-icon-box blue">{SVG_STAT_PAGES}</div>
                <div>
                  <div class="summary-stat-label">Tổng số trang</div>
                  <div class="summary-stat-val navy">{total_pages}</div>
                </div>
              </div>
              <div class="summary-stat-card">
                <div class="summary-icon-box red">{SVG_STAT_WARNING}</div>
                <div>
                  <div class="summary-stat-label">Trang lỗi</div>
                  <div class="summary-stat-val red">{error_pages}</div>
                </div>
              </div>
              <div class="summary-stat-card">
                <div class="summary-icon-box green">{SVG_STAT_ROTATE}</div>
                <div>
                  <div class="summary-stat-label">Đã tự xoay</div>
                  <div class="summary-stat-val green">{auto_rotated}</div>
                </div>
              </div>
              <div class="summary-stat-card">
                <div class="summary-icon-box orange">{SVG_STAT_SEARCH}</div>
                <div>
                  <div class="summary-stat-label">Cần kiểm tra</div>
                  <div class="summary-stat-val orange">{review_pages}</div>
                </div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
