"""
Hệ thống CSS tùy chỉnh có phạm vi rõ ràng (Scoped CSS) và bộ SVG Icon chuẩn thiết kế
cho Dashboard PDF Scan Quality Inspector (`src/pdf_inspector/ui/styles.py`).

Thiết kế tối ưu Pixel-Perfect ở độ phân giải tham chiếu 1534 × 960 px (16:10),
đồng thời tương thích tốt với 1920 × 1080, 1366 × 768, Tablet và Mobile.
"""

import streamlit as st

# ==============================================================================
# SVG ICONS (Vector sắc nét, dùng offline 100%, khớp chuẩn thiết kế)
# ==============================================================================

SVG_HEADER_DOC_SEARCH = """
<svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <rect x="6" y="4" width="28" height="36" rx="4" fill="#EDF5FF" stroke="#1677F2" stroke-width="2"/>
  <path d="M26 4V11C26 12.1046 26.8954 13 28 13H34" stroke="#1677F2" stroke-width="2" stroke-linecap="round"/>
  <line x1="12" y1="15" x2="22" y2="15" stroke="#5B9BF8" stroke-width="2" stroke-linecap="round"/>
  <line x1="12" y1="21" x2="26" y2="21" stroke="#5B9BF8" stroke-width="2" stroke-linecap="round"/>
  <line x1="12" y1="27" x2="21" y2="27" stroke="#5B9BF8" stroke-width="2" stroke-linecap="round"/>
  <circle cx="32" cy="32" r="8.5" fill="#FFFFFF" stroke="#1677F2" stroke-width="2.3"/>
  <circle cx="32" cy="32" r="5.5" fill="#DCEBFF"/>
  <line x1="38.2" y1="38.2" x2="43.5" y2="43.5" stroke="#1677F2" stroke-width="2.6" stroke-linecap="round"/>
</svg>
"""

SVG_UPLOAD_CLOUD = """
<svg width="42" height="42" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M14.5 33H11.8C7.76832 33 4.5 29.7317 4.5 25.7C4.5 22.0546 7.17538 19.0342 10.6688 18.486C11.7972 13.3446 16.3804 9.5 21.86 9.5C28.1892 9.5 33.32 14.6308 33.32 20.96C33.32 21.2244 33.311 21.4867 33.2934 21.7465C36.7915 22.3458 39.5 25.3886 39.5 29.06C39.5 31.236 37.736 33 35.56 33H29.5" stroke="#1677F2" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M22 36V22.5" stroke="#1677F2" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M17.2 27L22 22.2L26.8 27" stroke="#1677F2" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_PDF_BADGE = """
<svg width="36" height="42" viewBox="0 0 36 42" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M0 6C0 2.68629 2.68629 0 6 0H24L36 12V36C36 39.3137 33.3137 42 30 42H6C2.68629 42 0 39.3137 0 36V6Z" fill="#EB364C"/>
  <path d="M24 0L36 12H28C25.7909 12 24 10.2091 24 8V0Z" fill="#F68B97"/>
  <text x="18" y="29" text-anchor="middle" fill="#FFFFFF" font-family="-apple-system, BlinkMacSystemFont, Inter, sans-serif" font-size="10.5" font-weight="800" letter-spacing="0.4">PDF</text>
</svg>
"""

SVG_PDF_BADGE_MUTED = """
<svg width="36" height="42" viewBox="0 0 36 42" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M0 6C0 2.68629 2.68629 0 6 0H24L36 12V36C36 39.3137 33.3137 42 30 42H6C2.68629 42 0 39.3137 0 36V6Z" fill="#CBD5E1"/>
  <path d="M24 0L36 12H28C25.7909 12 24 10.2091 24 8V0Z" fill="#E2E8F0"/>
  <text x="18" y="29" text-anchor="middle" fill="#FFFFFF" font-family="-apple-system, BlinkMacSystemFont, Inter, sans-serif" font-size="10.5" font-weight="800" letter-spacing="0.4">PDF</text>
</svg>
"""

SVG_CHECK_CIRCLE_GREEN = """
<svg width="18" height="18" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <circle cx="10" cy="10" r="9.5" fill="#10A365"/>
  <path d="M6.2 10.3L8.8 12.8L14.0 7.5" stroke="#FFFFFF" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_TABLE_CHECK_GREEN = """
<svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="11" cy="11" r="10.5" fill="#12A667"/>
  <path d="M6.8 11.3L9.7 14.1L15.4 8.3" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_TABLE_SUSPECT_ORANGE = """
<svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="11" cy="11" r="10.5" fill="#F59E0B"/>
  <text x="11" y="15.2" text-anchor="middle" fill="#FFFFFF" font-family="-apple-system, BlinkMacSystemFont, Inter, sans-serif" font-size="13" font-weight="800">?</text>
</svg>
"""

SVG_STAT_PAGES = """
<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="8" y="3.5" width="14.5" height="18.5" rx="2.5" stroke="#1677F2" stroke-width="2.1" fill="#FFFFFF"/>
  <path d="M18 3.5V7.5C18 8.05228 18.4477 8.5 19 8.5H22.5" stroke="#1677F2" stroke-width="2" stroke-linecap="round"/>
  <path d="M5.5 8.5V22.5C5.5 23.8807 6.61929 25 8 25H18.5" stroke="#1677F2" stroke-width="2.1" stroke-linecap="round"/>
</svg>
"""

SVG_STAT_WARNING = """
<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12.26 5.2C13.03 3.86 14.97 3.86 15.74 5.2L24.53 20.45C25.3 21.78 24.34 23.45 22.79 23.45H5.21C3.66 23.45 2.7 21.78 3.47 20.45L12.26 5.2Z" stroke="#EB364C" stroke-width="2.2" stroke-linejoin="round"/>
  <line x1="14" y1="10.8" x2="14" y2="16.2" stroke="#EB364C" stroke-width="2.3" stroke-linecap="round"/>
  <circle cx="14" cy="19.6" r="1.3" fill="#EB364C"/>
</svg>
"""

SVG_STAT_ROTATE = """
<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M22.5 11.5C21.4 7.7 17.9 5 13.8 5C9.4 5 5.7 8.1 5 12.3" stroke="#079B60" stroke-width="2.3" stroke-linecap="round"/>
  <path d="M19.2 11.8H23V8" stroke="#079B60" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M5.5 16.5C6.6 20.3 10.1 23 14.2 23C18.6 23 22.3 19.9 23 15.7" stroke="#079B60" stroke-width="2.3" stroke-linecap="round"/>
  <path d="M8.8 16.2H5V20" stroke="#079B60" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_STAT_SEARCH = """
<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12.8" cy="12.8" r="7.5" stroke="#E97800" stroke-width="2.4"/>
  <line x1="18.4" y1="18.4" x2="23.5" y2="23.5" stroke="#E97800" stroke-width="2.6" stroke-linecap="round"/>
</svg>
"""

SVG_STAT_SHIELD_RED = """
<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M14 3.5L22.5 7V13.6C22.5 18.8 18.9 23.1 14 24.6C9.1 23.1 5.5 18.8 5.5 13.6V7L14 3.5Z" stroke="#EB364C" stroke-width="2.2" stroke-linejoin="round"/>
  <path d="M10.8 14.1L13.1 16.4L17.6 11.8" stroke="#EB364C" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

SVG_STAT_STAMP_BLUE = """
<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="4.5" y="18.5" width="19" height="4.5" rx="1.8" stroke="#079B60" stroke-width="2.1"/>
  <path d="M10.5 18.5V14.5C10.5 13.4 11.4 12.5 12.5 12.5H15.5C16.6 12.5 17.5 13.4 17.5 14.5V18.5" stroke="#079B60" stroke-width="2.1"/>
  <circle cx="14" cy="8.2" r="3.2" stroke="#079B60" stroke-width="2.1"/>
</svg>
"""

SVG_FOOTER_INFO = """
<svg width="16" height="16" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <circle cx="9" cy="9" r="7.8" stroke="#1677F2" stroke-width="1.6"/>
  <line x1="9" y1="8.2" x2="9" y2="12.8" stroke="#1677F2" stroke-width="1.7" stroke-linecap="round"/>
  <circle cx="9" cy="5.6" r="1" fill="#1677F2"/>
</svg>
"""

SVG_FOOTER_LOCK = """
<svg width="15" height="15" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
  <rect x="3.5" y="8" width="11" height="7.5" rx="2" stroke="#5B708B" stroke-width="1.6"/>
  <path d="M6 8V5.8C6 4.14315 7.34315 2.8 9 2.8C10.6569 2.8 12 4.14315 12 5.8V8" stroke="#5B708B" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""


# ==============================================================================
# SCOPED CSS SHEET
# ==============================================================================

DASHBOARD_CSS = """
<style id="pdf-inspector-theme">
:root {
  --sidebar-navy: #142746;
  --sidebar-navy-hover: #1E3660;
  --bg-page: #F6F8FC;
  --bg-card: #FFFFFF;
  --border-card: #DFE6F0;
  --shadow-card: 0 2px 8px rgba(23, 35, 58, 0.04);
  --primary: #0866E8;
  --primary-hover: #0652BA;
  --text-main: #17233A;
  --text-sec: #667892;
  --success: #15965D;
  --warning: #E98914;
  --danger: #E93D4D;
  --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Ẩn hoàn toàn phần tử chứa thẻ <style> để không tạo khoảng trống 16px đầu trang */
div[data-testid="stElementContainer"]:has(#pdf-inspector-theme) {
  display: none !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* Ẩn Streamlit Chrome mặc định (Header, Toolbar, Footer) nhưng GIỮ LẠI stSidebar */
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
div[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapseButton"],
div[data-testid="stSidebarHeader"],
footer {
  display: none !important;
}

html, body, .stApp {
  background-color: var(--bg-page) !important;
  font-family: var(--font-sans) !important;
  color: var(--text-main) !important;
}

/* Container chính: padding gọn gàng chuẩn desktop */
.stMainBlockContainer,
div[data-testid="stAppViewBlockContainer"],
.block-container {
  max-width: 100% !important;
  padding: 0 18px 12px 18px !important;
  margin: 0 !important;
}

/* Kiểm soát khoảng cách dọc mặc định của khối gốc */
div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {
  gap: 10px !important;
}

/* ============================================================================
   0. LEFT SIDEBAR (Cố định bên trái, nền xanh navy đậm #142746, rộng ~212px)
   ============================================================================ */
section[data-testid="stSidebar"] {
  background: var(--sidebar-navy) !important;
  width: 212px !important;
  min-width: 212px !important;
  max-width: 212px !important;
  border-right: 1px solid #0E1D36 !important;
  box-shadow: 2px 0 10px rgba(10, 20, 40, 0.12) !important;
  transition: margin-left 0.2s ease, width 0.2s ease !important;
}

section[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
  background: var(--sidebar-navy) !important;
  padding: 14px 10px 12px 10px !important;
}

.sb-brand-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 2px 6px 14px 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.10);
  margin-bottom: 10px;
}

.sb-brand-icon {
  width: 34px;
  height: 34px;
  min-width: 34px;
  border-radius: 8px;
  background: #0D66E8;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 6px rgba(13, 102, 232, 0.35);
}

.sb-brand-title {
  font-size: 13.8px;
  font-weight: 700;
  color: #FFFFFF;
  line-height: 1.25;
  letter-spacing: -0.01em;
}

.st-key-sb_nav_menu > div[data-testid="stVerticalBlock"] {
  gap: 5px !important;
}

div[class*="st-key-sb_menu_act_"] button {
  background: var(--primary) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--primary) !important;
  border-radius: 8px !important;
  height: 40px !important;
  min-height: 40px !important;
  padding: 0 12px !important;
  justify-content: flex-start !important;
  box-shadow: 0 2px 6px rgba(8, 102, 232, 0.32) !important;
}

div[class*="st-key-sb_menu_act_"] button p,
div[class*="st-key-sb_menu_act_"] button span[data-testid="stIconMaterial"] {
  color: #FFFFFF !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
}

div[class*="st-key-sb_menu_inact_"] button {
  background: transparent !important;
  color: #DCE6F5 !important;
  border: 1px solid transparent !important;
  border-radius: 8px !important;
  height: 40px !important;
  min-height: 40px !important;
  padding: 0 12px !important;
  justify-content: flex-start !important;
  box-shadow: none !important;
}

div[class*="st-key-sb_menu_inact_"] button:hover {
  background: var(--sidebar-navy-hover) !important;
  color: #FFFFFF !important;
}

div[class*="st-key-sb_menu_inact_"] button p,
div[class*="st-key-sb_menu_inact_"] button span[data-testid="stIconMaterial"] {
  color: #DCE6F5 !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
}

div[class*="st-key-sb_menu_dis_"] button,
div[class*="st-key-sb_menu_dis_"] button:disabled {
  background: transparent !important;
  color: #CBD5E1 !important;
  border: 1px solid transparent !important;
  border-radius: 8px !important;
  height: 40px !important;
  min-height: 40px !important;
  padding: 0 12px !important;
  justify-content: flex-start !important;
  opacity: 1 !important;
  box-shadow: none !important;
  cursor: default !important;
}

div[class*="st-key-sb_menu_dis_"] button p,
div[class*="st-key-sb_menu_dis_"] button span[data-testid="stIconMaterial"] {
  color: #CBD5E1 !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
}

.sb-version-footer {
  position: fixed;
  bottom: 0;
  left: 0;
  width: 212px;
  box-sizing: border-box;
  padding: 14px 18px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  background: var(--sidebar-navy);
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #CBD5E1;
  z-index: 20;
}

.sb-ver-txt {
  color: #CBD5E1;
  font-weight: 500;
}

/* ============================================================================
   1. HEADER BAR (Height 60–64px, nền trắng, không có subtitle)
   ============================================================================ */
.st-key-top_header_bar {
  max-width: none !important;
  width: calc(100% + 36px) !important;
  margin-left: -18px !important;
  margin-right: -18px !important;
  height: 62px !important;
  min-height: 62px !important;
  background: #FFFFFF !important;
  border-bottom: 1px solid var(--border-card) !important;
  box-shadow: 0 1px 4px rgba(23, 35, 58, 0.03) !important;
  padding: 0 20px !important;
  box-sizing: border-box !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

.st-key-top_header_bar > div[data-testid="stVerticalBlock"] {
  width: 100% !important;
  justify-content: center !important;
  gap: 0 !important;
}

.st-key-top_header_bar div[data-testid="stHorizontalBlock"] {
  align-items: center !important;
  width: 100% !important;
}

.st-key-hdr_left_group {
  display: flex !important;
  align-items: center !important;
  justify-content: flex-start !important;
  gap: 10px !important;
  width: auto !important;
}

.st-key-hdr_left_group > div[data-testid="stElementContainer"] {
  width: auto !important;
  flex: 0 0 auto !important;
  margin: 0 !important;
}

.st-key-hdr_menu_toggle_btn button {
  width: 34px !important;
  height: 34px !important;
  min-width: 34px !important;
  min-height: 34px !important;
  padding: 0 !important;
  border-radius: 8px !important;
  background: transparent !important;
  border: none !important;
  color: #334155 !important;
  box-shadow: none !important;
}

.st-key-hdr_menu_toggle_btn button:hover {
  background: #EAF2FD !important;
  color: var(--primary) !important;
}

.hdr-title {
  font-size: 19px !important;
  font-weight: 700 !important;
  color: var(--text-main) !important;
  line-height: 1.2 !important;
  letter-spacing: -0.01em !important;
  margin: 0 !important;
  padding: 0 !important;
  white-space: nowrap !important;
}

/* Nhóm nút bên phải Header: Xuất báo cáo | Xuất PDF đã sửa | Trợ giúp | User Avatar */
.st-key-hdr_right_group {
  display: flex !important;
  justify-content: flex-end !important;
  align-items: center !important;
  gap: 10px !important;
  width: 100% !important;
}

.st-key-hdr_right_group > div[data-testid="stElementContainer"] {
  width: auto !important;
  flex: 0 0 auto !important;
  margin: 0 !important;
}

.st-key-hdr_report_btn_wrap,
.st-key-hdr_download_wrap,
.st-key-hdr_help_btn_wrap {
  width: auto !important;
  display: inline-flex !important;
  align-items: center !important;
}

.hdr-user-avatar-wrap {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  flex-shrink: 0;
}

/* Nút Xuất báo cáo: nền trắng, viền xanh */
.st-key-hdr_report_btn_wrap button {
  background-color: #FFFFFF !important;
  color: var(--primary) !important;
  border: 1.5px solid #3B82F6 !important;
  border-radius: 8px !important;
  height: 36px !important;
  min-height: 36px !important;
  padding: 0 14px !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
  white-space: nowrap !important;
  box-shadow: none !important;
}

.st-key-hdr_report_btn_wrap button:hover:not(:disabled) {
  background-color: #EFF6FF !important;
}

.st-key-hdr_report_btn_wrap button p,
.st-key-hdr_report_btn_wrap button span[data-testid="stIconMaterial"] {
  color: var(--primary) !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
}

.st-key-hdr_report_btn_wrap button:disabled {
  background-color: #FFFFFF !important;
  border-color: #93C5FD !important;
  color: var(--primary) !important;
  opacity: 0.78 !important;
}

.st-key-hdr_report_btn_wrap button:disabled p,
.st-key-hdr_report_btn_wrap button:disabled span[data-testid="stIconMaterial"] {
  color: var(--primary) !important;
}

/* Nút Xuất PDF đã sửa: Primary Blue */
.st-key-hdr_download_btn button,
.st-key-hdr_download_disabled button {
  background-color: var(--primary) !important;
  color: #FFFFFF !important;
  border: 1.5px solid var(--primary) !important;
  border-radius: 8px !important;
  height: 36px !important;
  min-height: 36px !important;
  padding: 0 15px !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 6px rgba(8, 102, 232, 0.20) !important;
  white-space: nowrap !important;
}

.st-key-hdr_download_btn button:hover {
  background-color: var(--primary-hover) !important;
  border-color: var(--primary-hover) !important;
  color: #FFFFFF !important;
}

.st-key-hdr_download_disabled button {
  opacity: 0.82 !important;
  box-shadow: none !important;
  cursor: not-allowed !important;
}

.st-key-hdr_download_btn button p,
.st-key-hdr_download_disabled button p {
  font-size: 13.5px !important;
  font-weight: 600 !important;
  color: #FFFFFF !important;
}

.st-key-hdr_download_btn button span[data-testid="stIconMaterial"],
.st-key-hdr_download_disabled button span[data-testid="stIconMaterial"] {
  color: #FFFFFF !important;
  font-size: 18px !important;
}

.st-key-hdr_help_btn_wrap button {
  width: 34px !important;
  height: 34px !important;
  min-width: 34px !important;
  min-height: 34px !important;
  padding: 0 !important;
  border-radius: 50% !important;
  background: transparent !important;
  border: none !important;
  color: #475569 !important;
  box-shadow: none !important;
}

.st-key-hdr_help_btn_wrap button:hover {
  background: #EAF2FD !important;
  color: var(--primary) !important;
}

/* ============================================================================
   TAB 2 — SECRET MARKING DETECTION SCOPED STYLES
   ============================================================================ */
.st-key-secret_table_wrapper {
  border: 1px solid #E6EEF6 !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  gap: 0 !important;
}

.st-key-secret_table_wrapper > div[data-testid="stVerticalBlock"] {
  gap: 0 !important;
}

.st-key-secret_table_wrapper div[data-testid="stHorizontalBlock"] {
  gap: 0 !important;
  align-items: center !important;
}

.st-key-secret_table_wrapper div[data-testid="stColumn"]:not(:last-child) {
  border-right: 1px solid #EEF3F9 !important;
}

.secret-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 3px 11px;
  border-radius: 6px;
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: 0.02em;
  border: 1px solid transparent;
  white-space: nowrap;
}

.secret-badge.tuyet-mat {
  background: #FEECEE;
  color: #C81E35;
  border-color: #FAC5CD;
}

.secret-badge.toi-mat {
  background: #FFF1E6;
  color: #C2410C;
  border-color: #FED7AA;
}

.secret-badge.mat {
  background: #FEF6E7;
  color: #B45309;
  border-color: #FDE68A;
}

.secret-badge.giai-mat {
  background: #E7F8EF;
  color: #079B60;
  border-color: #BCEAD2;
}

.secret-crop-strip {
  display: flex;
  align-items: center;
  gap: 12px;
  background: #F8FAFD;
  border: 1px solid #E5EDF6;
  border-radius: 8px;
  padding: 8px 12px;
}

.secret-crop-thumb {
  background: #FFFFFF;
  border: 1px dashed #EB364C;
  border-radius: 6px;
  padding: 4px 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  max-height: 58px;
  min-width: 78px;
  flex-shrink: 0;
}

.secret-crop-thumb img {
  max-height: 48px;
  max-width: 150px;
  width: auto;
  object-fit: contain;
  display: block;
}

.secret-crop-meta {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 12.5px;
  color: var(--text-sec);
}

.secret-crop-meta strong {
  color: var(--text-main);
  font-size: 13px;
}

.st-key-sec_export_bar > div[data-testid="stHorizontalBlock"] {
  gap: 8px !important;
}

div[class*="st-key-sec_exp_"] button {
  background: #F4F8FE !important;
  color: var(--primary) !important;
  border: 1px solid #CCE0FA !important;
  border-radius: 8px !important;
  height: 36px !important;
  min-height: 36px !important;
  font-weight: 600 !important;
}

div[class*="st-key-sec_exp_"] button p {
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--primary) !important;
}

/* ============================================================================
   2. UPLOAD AREA — GIỮ ĐÚNG MẪU (Height 168px, nền xanh nhạt, viền nét đứt)
   ============================================================================ */
.st-key-upload_zone_box {
  position: relative !important;
  height: 168px !important;
  min-height: 168px !important;
  max-height: 168px !important;
  width: 100% !important;
  border-radius: 11px !important;
  overflow: hidden !important;
  gap: 0 !important;
}

.st-key-upload_zone_box > div[data-testid="stVerticalBlock"] {
  gap: 0 !important;
}

.upload-visual-overlay {
  height: 168px;
  width: 100%;
  border: 1.5px dashed #7BAEF7;
  border-radius: 11px;
  background: #EEF5FF;
  display: flex;
  align-items: center;
  justify-content: space-around;
  gap: 20px;
  padding: 0 48px;
  box-sizing: border-box;
  pointer-events: none;
  transition: background-color 0.15s ease, border-color 0.15s ease;
}

.st-key-upload_zone_box:hover .upload-visual-overlay {
  background: #E6F1FE;
  border-color: var(--primary);
}

.upload-illust-col {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.upload-center-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 7px;
  flex: 1;
}

.upload-cloud-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: -2px;
}

.upload-cloud-wrap svg {
  width: 38px;
  height: 38px;
}

.upload-visual-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-main);
  line-height: 1.3;
  margin: 0;
}

.upload-dual-opts {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-top: 4px;
}

.upload-opt {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  font-size: 13.5px;
  font-weight: 600;
  border-radius: 8px;
  padding: 7px 16px;
  height: 36px;
  box-sizing: border-box;
  transition: background-color 0.15s ease, color 0.15s ease, transform 0.1s ease;
  white-space: nowrap;
}

.upload-opt-file {
  background: var(--primary);
  color: #FFFFFF;
  border: 1.5px solid var(--primary);
  box-shadow: 0 2px 6px rgba(8, 102, 232, 0.22);
}

.upload-opt-dir {
  background: #FFFFFF;
  color: var(--primary);
  border: 1.5px solid #3B82F6;
}

.st-key-upload_zone_box:has(.st-key-upl_half_file:hover) .upload-opt-file {
  background: var(--primary-hover);
  border-color: var(--primary-hover);
}

.st-key-upload_zone_box:has(.st-key-upl_half_dir:hover) .upload-opt-dir {
  background: #EAF2FF;
}

/* Đặt st.file_uploader thật phủ trong suốt khớp 100% lên vùng nét đứt */
.st-key-upload_zone_box div[data-testid="stElementContainer"]:has(div[data-testid="stFileUploader"]) {
  position: absolute !important;
  inset: 0 !important;
  height: 168px !important;
  width: 100% !important;
  z-index: 10 !important;
  margin: 0 !important;
  padding: 0 !important;
}

.st-key-upload_zone_box div[data-testid="stFileUploader"] {
  position: absolute !important;
  inset: 0 !important;
  height: 168px !important;
  width: 100% !important;
  z-index: 10 !important;
  opacity: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

.st-key-upload_zone_box section[data-testid="stFileUploaderDropzone"] {
  height: 168px !important;
  min-height: 168px !important;
  width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  cursor: pointer !important;
}

.st-key-upload_zone_box section[data-testid="stFileUploaderDropzone"] button {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
  cursor: pointer !important;
}

.st-key-upload_zone_box div[data-testid="stFileUploaderFile"],
.st-key-upload_zone_box small {
  display: none !important;
}

.st-key-upload_zone_box .st-key-upl_half_file,
.st-key-upload_zone_box .st-key-upl_half_dir {
  position: absolute !important;
  top: 0 !important;
  height: 168px !important;
  width: 50% !important;
  z-index: 10 !important;
  gap: 0 !important;
  pointer-events: auto !important;
}

.st-key-upload_zone_box .st-key-upl_half_file {
  left: 0 !important;
}

.st-key-upload_zone_box .st-key-upl_half_dir {
  left: 50% !important;
}

.st-key-upload_zone_box div:has(> .st-key-upl_half_file),
.st-key-upload_zone_box div:has(> .st-key-upl_half_dir) {
  pointer-events: none !important;
}

/* ============================================================================
   3. THANH THÔNG TIN FILE (1 hàng ngang duy nhất, cao 65–75px)
   ============================================================================ */
.st-key-upload_card {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-card) !important;
  border-radius: 10px !important;
  box-shadow: var(--shadow-card) !important;
  padding: 8px 16px !important;
  min-height: 66px !important;
}

.st-key-upload_card > div[data-testid="stHorizontalBlock"] {
  align-items: center !important;
  gap: 16px !important;
}

/* Bộ lọc 'Tệp PDF' phía trên bảng kết quả (chỉ hiện khi quét nhiều PDF) */
.st-key-file_filter_bar {
  margin: 2px 0 6px 0 !important;
}

.file-filter-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-sec);
  white-space: nowrap;
}

.st-key-file_filter_bar .react-aria-ComboBox div[role="group"],
.st-key-file_filter_bar div[data-baseweb="select"] > div {
  min-height: 34px !important;
  border: 1px solid var(--border-card) !important;
  border-radius: 8px !important;
  background: #FFFFFF !important;
  padding-left: 10px !important;
}

.st-key-file_filter_bar .react-aria-ComboBox input {
  font-size: 13px !important;
  color: var(--text-main) !important;
}

.tbl-file-cell {
  font-size: 12.5px;
  color: var(--text-main);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-meta-row {
  display: flex;
  align-items: center;
  gap: 11px;
  min-width: 0;
  border-right: 1px solid #E2E8F0;
  padding-right: 16px;
}

.file-name-txt {
  font-size: 14.5px;
  font-weight: 700;
  color: var(--text-main);
  line-height: 1.25;
  margin: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-sub-txt {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12.5px;
  font-weight: 400;
  color: #526580;
  line-height: 1.3;
  margin: 3px 0 0 0;
  white-space: nowrap;
}

.file-sub-sep {
  display: inline-block;
  width: 12px;
}

/* Thanh tiến trình 1 hàng ngang ở giữa thanh thông tin file */
.proc-single-line-row {
  display: flex;
  align-items: center;
  gap: 14px;
  width: 100%;
  padding: 0 6px;
  box-sizing: border-box;
}

.proc-badge-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 12.8px;
  font-weight: 600;
  white-space: nowrap;
  flex-shrink: 0;
}

.proc-badge-pill.done {
  background: #E6F7EE;
  color: #15965D;
}

.proc-badge-pill.scanning {
  background: #EBF3FE;
  color: var(--primary);
}

.proc-badge-pill.idle {
  background: #F1F5F9;
  color: var(--text-sec);
}

.proc-progress-wrap {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 5px;
}

.proc-status-top-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.proc-bar-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.proc-bar-track {
  flex: 1;
  height: 8px;
  background: #E2E8F0;
  border-radius: 999px;
  overflow: hidden;
}

.proc-bar-fill {
  height: 100%;
  background: var(--primary);
  border-radius: 999px;
  transition: width 0.2s ease;
}

.proc-bar-fill.done {
  background: #16A34A;
}

.proc-pct-txt {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-main);
  min-width: 36px;
  text-align: right;
  flex-shrink: 0;
}

.proc-status-row {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  font-size: 12.8px;
  font-weight: 600;
  color: var(--success);
}

.proc-status-row.scanning {
  color: var(--primary);
}

.proc-status-row.idle {
  color: var(--text-sec);
}

/* 3 icon hành động bên phải Thanh thông tin file: Folder | Info | Trash */
.st-key-file_bar_actions_row {
  display: flex !important;
  justify-content: flex-end !important;
  align-items: center !important;
  gap: 6px !important;
  width: 100% !important;
}

.st-key-file_bar_actions_row > div[data-testid="stElementContainer"] {
  width: auto !important;
  flex: 0 0 auto !important;
  margin: 0 !important;
}

.st-key-btn_view_file_info button,
.st-key-btn_doc_tech_info button,
.st-key-btn_clear_file button {
  width: 34px !important;
  height: 34px !important;
  min-width: 34px !important;
  min-height: 34px !important;
  border-radius: 8px !important;
  padding: 0 !important;
  background: #F4F7FB !important;
  border: 1px solid var(--border-card) !important;
  color: #526580 !important;
  box-shadow: none !important;
}

.st-key-btn_view_file_info button:hover:not(:disabled),
.st-key-btn_doc_tech_info button:hover:not(:disabled) {
  background: #EAF2FD !important;
  border-color: var(--primary) !important;
  color: var(--primary) !important;
}

.st-key-btn_clear_file button:hover:not(:disabled) {
  background: #FEE2E2 !important;
  border-color: #FCA5A5 !important;
  color: var(--danger) !important;
}

/* ============================================================================
   4. MAIN WORKSPACE — CHIA HAI PANEL (51% Bảng kết quả | 49% PDF Preview)
   ============================================================================ */
.st-key-main_two_col_row > div[data-testid="stHorizontalBlock"] {
  gap: 12px !important;
  align-items: stretch !important;
}

.st-key-left_error_card,
.st-key-right_preview_card {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-card) !important;
  border-radius: 11px !important;
  box-shadow: var(--shadow-card) !important;
  padding: 14px 16px !important;
  min-height: 515px !important;
  height: 100% !important;
}

.st-key-left_error_card > div[data-testid="stVerticalBlock"],
.st-key-right_preview_card > div[data-testid="stVerticalBlock"] {
  gap: 8px !important;
}

.panel-heading {
  font-size: 17px !important;
  font-weight: 700 !important;
  color: var(--text-main) !important;
  line-height: 1.25 !important;
  margin: 0 !important;
  padding: 0 !important;
  letter-spacing: -0.01em !important;
  white-space: nowrap !important;
}

.st-key-left_hdr_row > div[data-testid="stHorizontalBlock"] {
  align-items: center !important;
}

.st-key-left_panel_hdr_icons,
.st-key-left_hdr_icons {
  display: flex !important;
  justify-content: flex-end !important;
  align-items: center !important;
  gap: 4px !important;
  width: 100% !important;
}

.st-key-left_panel_hdr_icons > div[data-testid="stElementContainer"],
.st-key-left_hdr_icons > div[data-testid="stElementContainer"] {
  width: auto !important;
  flex: 0 0 auto !important;
  margin: 0 !important;
}

.st-key-left_panel_hdr_icons button,
.st-key-left_hdr_icons button {
  width: 30px !important;
  height: 30px !important;
  min-width: 30px !important;
  min-height: 30px !important;
  padding: 0 !important;
  border-radius: 7px !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  color: #475569 !important;
  box-shadow: none !important;
}

.st-key-left_panel_hdr_icons button:hover:not(:disabled),
.st-key-left_hdr_icons button:hover:not(:disabled) {
  border-color: var(--border-card) !important;
  color: var(--primary) !important;
  background: #EAF2FD !important;
}

/* Hàng Filter Pills (Tất cả / Sọc / Lẫn trang / Xoay / Cần rà soát) */
.st-key-filter_pills_bar {
  display: flex !important;
  align-items: center !important;
  gap: 6px !important;
  flex-wrap: wrap !important;
}

.st-key-filter_pills_bar > div[data-testid="stElementContainer"] {
  width: auto !important;
  flex: 0 0 auto !important;
  margin: 0 !important;
}

div[class*="st-key-flt_act_"] button {
  background: var(--primary) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--primary) !important;
  border-radius: 999px !important;
  height: 31px !important;
  min-height: 31px !important;
  padding: 0 12px !important;
  font-size: 12.5px !important;
  font-weight: 600 !important;
  box-shadow: 0 1px 4px rgba(8, 102, 232, 0.18) !important;
}

div[class*="st-key-flt_act_"] button p {
  color: #FFFFFF !important;
  font-size: 12.5px !important;
  font-weight: 600 !important;
}

div[class*="st-key-flt_inact_"] button {
  background: #F4F7FB !important;
  color: #5A6E87 !important;
  border: 1px solid #E5ECF5 !important;
  border-radius: 999px !important;
  height: 31px !important;
  min-height: 31px !important;
  padding: 0 12px !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  box-shadow: none !important;
}

div[class*="st-key-flt_inact_"] button:hover {
  background: #EAF2FD !important;
  border-color: #BDD6F8 !important;
  color: var(--primary) !important;
}

div[class*="st-key-flt_inact_"] button p {
  color: inherit !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
}

/* Bảng kết quả 7 cột: | Chọn | Trang | Sọc | Lẫn | Xoay | Trạng thái | Thao tác | */
.st-key-error_table_wrapper {
  border: 1px solid #E6EEF6 !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  gap: 0 !important;
}

.st-key-error_table_wrapper > div[data-testid="stVerticalBlock"] {
  gap: 0 !important;
}

.st-key-tbl_hdr_row {
  background: #F8FAFC !important;
  border-bottom: 1px solid #E4ECF5 !important;
  min-height: 38px !important;
  padding: 0 !important;
}

div[class*="st-key-tbl_row_norm_"] {
  background: #FFFFFF !important;
  border-bottom: 1px solid #EDF2F8 !important;
  min-height: 46px !important;
  padding: 0 !important;
  transition: background-color 0.12s ease;
}

div[class*="st-key-tbl_row_norm_"]:hover {
  background: #F7FAFE !important;
}

div[class*="st-key-tbl_row_sel_"] {
  background: #E7F1FD !important;
  border-bottom: 1px solid #D8E7FA !important;
  min-height: 46px !important;
  padding: 0 !important;
}

div[class*="st-key-tbl_row_norm_"]:last-child,
div[class*="st-key-tbl_row_sel_"]:last-child {
  border-bottom: none !important;
}

.st-key-error_table_wrapper div[data-testid="stHorizontalBlock"] {
  gap: 0 !important;
  align-items: center !important;
}

.st-key-error_table_wrapper div[data-testid="stColumn"]:not(:last-child) {
  border-right: none !important;
}

.tbl-hdr-cell {
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12.5px;
  font-weight: 600;
  color: #475569;
  text-align: center;
  padding: 0 4px;
}

.tbl-data-cell {
  height: 46px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  color: var(--text-main);
  text-align: center;
  padding: 0 4px;
}

.st-key-tbl_hdr_chk_cell {
  height: 38px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

div[class*="st-key-tbl_chk_cell_"] {
  height: 46px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

.st-key-tbl_hdr_chk_cell div[data-testid="stCheckbox"],
div[class*="st-key-tbl_chk_cell_"] div[data-testid="stCheckbox"] {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin: 0 !important;
}

.tbl-page-num {
  font-size: 13.5px;
  font-weight: 700;
  color: var(--text-main);
}

.tbl-rot-txt {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-main);
}

.tbl-dash {
  font-size: 14px;
  color: #9AAABB;
  font-weight: 400;
}

.tbl-icon-tip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: help;
}

/* Status Pills dùng chung trong Bảng & Hộp Chi Tiết */
.status-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  line-height: 1.3;
  white-space: nowrap;
}

.status-pill.green {
  background: #E2F6EC;
  color: var(--success);
}

.status-pill.orange {
  background: #FFF0D9;
  color: #D97706;
}

.status-pill.red {
  background: #FDECEE;
  color: var(--danger);
}

.status-pill.gray {
  background: #EEF2F7;
  color: #60738A;
}

/* Cột Thao tác: icon Xem (mắt) + icon Chỉnh sửa (bút chì) */
div[class*="st-key-tbl_action_cell_"] {
  height: 46px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 4px !important;
}

div[class*="st-key-tbl_action_cell_"] > div[data-testid="stElementContainer"] {
  width: auto !important;
  flex: 0 0 auto !important;
  margin: 0 !important;
}

div[class*="st-key-btn_view_p_"] button,
div[class*="st-key-btn_edit_p_"] button {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 6px !important;
  width: 28px !important;
  height: 28px !important;
  min-width: 28px !important;
  min-height: 28px !important;
  padding: 0 !important;
  color: var(--primary) !important;
  box-shadow: none !important;
}

div[class*="st-key-btn_view_p_"] button:hover,
div[class*="st-key-btn_edit_p_"] button:hover {
  border-color: #BFDBFE !important;
  color: var(--primary) !important;
  background: #EFF6FF !important;
}

div[class*="st-key-btn_view_p_"] button span[data-testid="stIconMaterial"],
div[class*="st-key-btn_edit_p_"] button span[data-testid="stIconMaterial"] {
  color: var(--primary) !important;
  font-size: 17px !important;
}

/* Pagination gọn cuối bảng */
.st-key-tbl_pagination_bar {
  padding-top: 4px !important;
  align-items: center !important;
}

div[class*="st-key-pg_nav_"] button {
  height: 28px !important;
  min-height: 28px !important;
  padding: 0 10px !important;
  border-radius: 6px !important;
  font-size: 12px !important;
  background: #F5F8FC !important;
  border: 1px solid #E2EAF3 !important;
  color: var(--text-main) !important;
}

.tbl-empty-state {
  min-height: 320px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 24px 20px;
  text-align: center;
  color: var(--text-sec);
}

.tbl-empty-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-main);
  margin: 10px 0 4px 0;
}

.tbl-empty-sub {
  font-size: 13px;
  color: var(--text-sec);
  max-width: 380px;
}

/* ============================================================================
   5. RIGHT PANEL — XEM TRANG [SỐ TRANG], TOOLBAR, CANVAS & THÔNG TIN TRANG
   ============================================================================ */
.st-key-preview_hdr_row > div[data-testid="stHorizontalBlock"] {
  align-items: center !important;
}

div[data-testid="stElementContainer"]:has(.st-key-preview_tabs_group),
.st-key-preview_tabs_group {
  display: flex !important;
  gap: 6px !important;
  justify-content: flex-start !important;
  width: 100% !important;
}

.st-key-preview_tabs_group > div[data-testid="stElementContainer"] {
  width: auto !important;
  flex: 1 1 0 !important;
  margin: 0 !important;
}

div[class*="st-key-ptab_act_"] button {
  background: #EDF5FF !important;
  color: var(--primary) !important;
  border: 1.5px solid var(--primary) !important;
  border-radius: 7px !important;
  height: 30px !important;
  min-height: 30px !important;
  padding: 0 10px !important;
  box-shadow: none !important;
}

div[class*="st-key-ptab_act_"] button p {
  font-size: 12px !important;
  font-weight: 600 !important;
  color: var(--primary) !important;
}

div[class*="st-key-ptab_inact_"] button {
  background: #F4F7FB !important;
  color: var(--text-sec) !important;
  border: 1px solid #E6ECF5 !important;
  border-radius: 7px !important;
  height: 30px !important;
  min-height: 30px !important;
  padding: 0 10px !important;
  box-shadow: none !important;
}

div[class*="st-key-ptab_inact_"] button p {
  font-size: 12px !important;
  font-weight: 500 !important;
  color: var(--text-sec) !important;
}

/* Toolbar điều khiển Preview cùng hàng với tiêu đề Xem trang */
.st-key-preview_toolbar_bar {
  display: flex !important;
  align-items: center !important;
  justify-content: flex-end !important;
  gap: 3px !important;
  width: 100% !important;
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

.st-key-preview_toolbar_bar > div[data-testid="stElementContainer"] {
  width: auto !important;
  flex: 0 0 auto !important;
  margin: 0 !important;
}

.st-key-preview_toolbar_bar button {
  width: 28px !important;
  height: 28px !important;
  min-width: 28px !important;
  min-height: 28px !important;
  padding: 0 !important;
  border-radius: 6px !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  color: #334155 !important;
  font-size: 12px !important;
  box-shadow: none !important;
}

.st-key-preview_toolbar_bar button:hover:not(:disabled) {
  border-color: var(--border-card) !important;
  color: var(--primary) !important;
  background: #EFF6FF !important;
}

.ptb-page-counter {
  height: 28px;
  padding: 0 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  font-size: 13px;
  font-weight: 700;
  color: var(--text-main);
  white-space: nowrap;
}

/* Vùng xem trước ảnh PDF bên trái của Panel phải */
.doc-preview-stage {
  width: 100%;
  height: 420px;
  background: #D8DCE3;
  border: 1px solid #CFD6DF;
  border-radius: 8px;
  position: relative;
  overflow: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 12px;
  box-sizing: border-box;
}

.doc-sheet-frame {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #FFFFFF;
  border-radius: 2px;
  box-shadow: 0 3px 12px rgba(15, 23, 42, 0.16);
  overflow: hidden;
  max-height: 394px;
  max-width: 96%;
}

.doc-sheet-frame.boundary-focus {
  max-width: 92%;
  max-height: 394px;
}

.doc-sheet-frame img {
  display: block;
  max-height: 394px;
  width: auto;
  max-width: 100%;
  object-fit: contain;
}

.doc-sheet-frame.boundary-focus img {
  max-height: 394px;
  width: 100%;
  height: auto;
}

/* Overlay vùng lỗi theo tọa độ thực tế (viền vàng/hổ phách như ảnh mẫu) */
.defect-bbox-overlay {
  position: absolute;
  background: rgba(254, 240, 138, 0.28);
  border: 2px solid #F59E0B;
  border-radius: 3px;
  box-sizing: border-box;
  pointer-events: none;
}

.defect-bbox-overlay.suspected {
  background: rgba(254, 240, 138, 0.30);
  border: 2px solid #F59E0B;
}

.defect-bbox-overlay.proposed-ok {
  background: rgba(21, 150, 93, 0.08);
  border: 1.8px dashed var(--success);
}

/* Nhãn pill nổi trên vùng lỗi */
.preview-floating-pill {
  position: absolute;
  right: 8px;
  top: 12px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 999px;
  background: #FFF1F3;
  border: 1px solid #FBC4CC;
  color: #D92D43;
  font-size: 11.5px;
  font-weight: 600;
  box-shadow: 0 2px 6px rgba(235, 54, 76, 0.12);
  z-index: 4;
  white-space: nowrap;
}

.preview-floating-pill.orange {
  background: #FFF6E8;
  border-color: #FAD6A5;
  color: #C86400;
}

.preview-floating-pill.green {
  background: #EBFAF2;
  border-color: #B4E8CD;
  color: var(--success);
}

.preview-pill-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
  flex-shrink: 0;
}

.split-preview-pair {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 380px;
  width: 100%;
}

.split-preview-card {
  background: #FFFFFF;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.10);
  padding: 3px;
  max-height: 368px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.split-preview-card img {
  max-height: 336px;
  width: auto;
  object-fit: contain;
}

.split-preview-cap {
  font-size: 10.5px;
  font-weight: 600;
  color: var(--text-sec);
  margin-top: 2px;
}

/* Cột Thông tin trang bên phải của Panel Preview */
.error-detail-card {
  border: none;
  border-radius: 0;
  background: transparent;
  overflow: visible;
}

.error-detail-heading {
  font-size: 14.5px;
  font-weight: 700;
  color: var(--text-main);
  padding: 2px 0 10px 0;
  background: transparent;
  border-bottom: 1px solid #EEF2F7;
}

.error-detail-row {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 0;
  border-bottom: 1px solid #EEF2F7;
  font-size: 12.5px;
}

.error-detail-row:last-child {
  border-bottom: none;
}

.error-detail-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #526580;
  font-weight: 500;
  font-size: 12.5px;
  white-space: nowrap;
}

.error-detail-val {
  color: var(--text-main);
  font-weight: 700;
  font-size: 13px;
  line-height: 1.3;
  text-align: right;
}

/* Bộ chọn lỗi phụ trên trang đa lỗi */
.st-key-multi_defect_chips {
  gap: 5px !important;
  margin-top: 2px !important;
  margin-bottom: 2px !important;
}

div[class*="st-key-mdchip_act_"] button {
  height: 24px !important;
  min-height: 24px !important;
  padding: 0 8px !important;
  border-radius: 999px !important;
  background: #E8F2FF !important;
  border: 1px solid var(--primary) !important;
  color: var(--primary) !important;
}

div[class*="st-key-mdchip_act_"] button p {
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--primary) !important;
}

div[class*="st-key-mdchip_inact_"] button {
  height: 24px !important;
  min-height: 24px !important;
  padding: 0 8px !important;
  border-radius: 999px !important;
  background: #F5F7FA !important;
  border: 1px solid #E2E8F0 !important;
  color: var(--text-sec) !important;
}

div[class*="st-key-mdchip_inact_"] button p {
  font-size: 11px !important;
  font-weight: 500 !important;
  color: var(--text-sec) !important;
}

/* 3 nút hành động xếp dọc dưới Thông tin trang: Chấp nhận | Bỏ qua | Sửa thủ công */
.st-key-preview_actions_bar > div[data-testid="stVerticalBlock"] {
  gap: 6px !important;
  margin-top: 4px !important;
}

.st-key-act_accept button {
  background: var(--primary) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--primary) !important;
  border-radius: 8px !important;
  height: 36px !important;
  min-height: 36px !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 5px rgba(8, 102, 232, 0.18) !important;
}

.st-key-act_accept button:hover:not(:disabled) {
  background: var(--primary-hover) !important;
  border-color: var(--primary-hover) !important;
}

.st-key-act_accept button p,
.st-key-act_accept button span[data-testid="stIconMaterial"] {
  color: #FFFFFF !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}

.st-key-act_accept button:disabled {
  background: #A8CCF7 !important;
  border-color: #A8CCF7 !important;
  opacity: 0.75 !important;
}

.st-key-act_skip button,
.st-key-act_manual button {
  background: #FFFFFF !important;
  color: var(--text-main) !important;
  border: 1px solid #DCE5F0 !important;
  border-radius: 8px !important;
  height: 36px !important;
  min-height: 36px !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}

.st-key-act_skip button:hover:not(:disabled),
.st-key-act_manual button:hover:not(:disabled) {
  background: #F6F9FD !important;
  border-color: var(--primary) !important;
  color: var(--primary) !important;
}

.st-key-act_skip button p,
.st-key-act_manual button p {
  color: var(--text-main) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}

.st-key-act_manual button span[data-testid="stIconMaterial"] {
  color: var(--primary) !important;
}

.st-key-act_skip button:disabled,
.st-key-act_manual button:disabled {
  opacity: 0.5 !important;
}

/* ============================================================================
   6. MODAL DIALOGS & RESPONSIVE ADAPTATIONS (1440px, 1280px, Tablet, Mobile)
   ============================================================================ */
div[role="dialog"] {
  background: #FFFFFF !important;
  color: var(--text-main) !important;
  border-radius: 12px !important;
  border: 1px solid var(--border-card) !important;
}

@media (max-width: 1280px) {
  section[data-testid="stSidebar"] {
    width: 196px !important;
    min-width: 196px !important;
    max-width: 196px !important;
  }
  .upload-visual-overlay {
    padding: 0 20px;
  }
}

@media (max-width: 1024px) {
  .upload-illust-col {
    display: none !important;
  }
  .st-key-main_two_col_row > div[data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
  }
  .st-key-main_two_col_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    width: 100% !important;
    flex: 1 1 100% !important;
  }
}

@media (max-width: 768px) {
  .stMainBlockContainer,
  div[data-testid="stAppViewBlockContainer"],
  .block-container {
    padding: 0 10px 12px 10px !important;
  }
  .st-key-top_header_bar {
    margin-left: -10px !important;
    margin-right: -10px !important;
    width: calc(100% + 20px) !important;
    padding: 8px 12px !important;
    height: auto !important;
    min-height: 56px !important;
  }
  .hdr-title { font-size: 16px !important; }
  .st-key-error_table_wrapper,
  .st-key-secret_table_wrapper {
    overflow-x: auto !important;
  }
  .st-key-error_table_wrapper div[data-testid="stHorizontalBlock"] {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    min-width: 560px !important;
  }
  .st-key-secret_table_wrapper div[data-testid="stHorizontalBlock"] {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    min-width: 440px !important;
  }
}
</style>
"""


def clean_html(raw: str) -> str:
    """Loại bỏ thụt đầu dòng và dòng trống trong chuỗi HTML để CommonMark không hiểu nhầm là code block."""
    return "".join(line.strip() for line in raw.splitlines() if line.strip())


def inject_dashboard_styles() -> None:
    """Tiêm CSS có phạm vi rõ ràng cho Dashboard PDF Scan Quality Inspector."""
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    if st.session_state.get("sidebar_collapsed", False):
        st.markdown(
            """
            <style id="pdf-inspector-sidebar-collapsed">
            section[data-testid="stSidebar"] {
              margin-left: -212px !important;
              width: 0px !important;
              min-width: 0px !important;
              max-width: 0px !important;
              overflow: hidden !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
