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
  --bg-page: #F5F8FC;
  --bg-card: #FFFFFF;
  --border-card: #E7EDF4;
  --shadow-card: 0 2px 10px rgba(25, 45, 80, 0.04);
  --primary: #1677F2;
  --primary-hover: #1264D1;
  --text-main: #14213D;
  --text-sec: #70849F;
  --success: #079B60;
  --warning: #E97800;
  --danger: #EB364C;
  --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Ẩn hoàn toàn phần tử chứa thẻ <style> để không tạo khoảng trống 16px đầu trang */
div[data-testid="stElementContainer"]:has(#pdf-inspector-theme) {
  display: none !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* Ẩn Streamlit Chrome mặc định (Header, Toolbar, Sidebar, Footer) */
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
section[data-testid="stSidebar"],
div[data-testid="collapsedControl"],
footer {
  display: none !important;
}

html, body, .stApp {
  background-color: var(--bg-page) !important;
  font-family: var(--font-sans) !important;
  color: var(--text-main) !important;
}

/* Container chính: padding ngang 30px chuẩn desktop 1534x960 */
.stMainBlockContainer,
div[data-testid="stAppViewBlockContainer"],
.block-container {
  max-width: 100% !important;
  padding: 0 30px 12px 30px !important;
  margin: 0 !important;
}

/* Kiểm soát khoảng cách dọc mặc định của khối gốc */
div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {
  gap: 14px !important;
}

/* ============================================================================
   1. HEADER BAR (Height ~76px, nền trắng tràn ngang 100% viewport)
   ============================================================================ */
.st-key-top_header_bar {
  max-width: none !important;
  width: calc(100% + 60px) !important;
  margin-left: -30px !important;
  margin-right: -30px !important;
  height: 75px !important;
  min-height: 75px !important;
  background: #FFFFFF !important;
  border-bottom: 1px solid var(--border-card) !important;
  box-shadow: 0 1px 6px rgba(20, 33, 61, 0.03) !important;
  padding: 0 28px !important;
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

.hdr-brand-wrap {
  display: flex;
  align-items: center;
  gap: 14px;
}

.hdr-brand-text {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.hdr-title {
  font-size: 23px !important;
  font-weight: 700 !important;
  color: var(--text-main) !important;
  line-height: 1.2 !important;
  letter-spacing: -0.015em !important;
  margin: 0 !important;
  padding: 0 !important;
}

.hdr-subtitle {
  font-size: 15.5px !important;
  font-weight: 400 !important;
  color: var(--text-sec) !important;
  line-height: 1.35 !important;
  margin: 2px 0 0 0 !important;
  padding: 0 !important;
}

/* Nút Tải PDF đã sửa ở góc phải Header (sát mép phải ~28px) */
.st-key-hdr_download_wrap,
.st-key-hdr_download_wrap > div[data-testid="stVerticalBlock"],
.st-key-hdr_download_wrap div[data-testid="stElementContainer"],
.st-key-hdr_download_btn,
.st-key-hdr_download_disabled {
  display: flex !important;
  justify-content: flex-end !important;
  align-items: center !important;
  width: 100% !important;
}

.st-key-hdr_download_btn > div,
.st-key-hdr_download_disabled > div {
  display: flex !important;
  justify-content: flex-end !important;
  width: auto !important;
}

.st-key-hdr_download_btn button,
.st-key-hdr_download_disabled button {
  background-color: var(--primary) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--primary) !important;
  border-radius: 8px !important;
  height: 48px !important;
  min-height: 48px !important;
  padding: 0 24px !important;
  font-size: 15.5px !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 8px rgba(22, 119, 242, 0.22) !important;
  transition: background-color 0.15s ease, transform 0.1s ease !important;
  white-space: nowrap !important;
}

.st-key-hdr_download_btn button:hover {
  background-color: var(--primary-hover) !important;
  border-color: var(--primary-hover) !important;
  color: #FFFFFF !important;
}

.st-key-hdr_download_disabled button {
  opacity: 0.58 !important;
  box-shadow: none !important;
  cursor: not-allowed !important;
}

.st-key-hdr_download_btn button p,
.st-key-hdr_download_disabled button p {
  font-size: 15px !important;
  font-weight: 600 !important;
  color: #FFFFFF !important;
}

.st-key-hdr_download_btn button span[data-testid="stIconMaterial"],
.st-key-hdr_download_disabled button span[data-testid="stIconMaterial"] {
  color: #FFFFFF !important;
  font-size: 20px !important;
}

/* Bộ chuyển 2 chế độ làm việc độc lập trên Header (Tab 1 & Tab 2) */
.st-key-hdr_mode_switcher {
  background: #F1F5FA !important;
  border: 1px solid #E1E9F4 !important;
  border-radius: 10px !important;
  padding: 4px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 4px !important;
  width: auto !important;
  margin: 0 auto !important;
}

div[data-testid="stElementContainer"]:has(.st-key-hdr_mode_switcher) {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
}

div[class*="st-key-mode_tab_act_"] button {
  background: #FFFFFF !important;
  color: var(--primary) !important;
  border: 1px solid #D5E4FA !important;
  border-radius: 7px !important;
  height: 36px !important;
  min-height: 36px !important;
  padding: 0 14px !important;
  box-shadow: 0 1px 4px rgba(22, 119, 242, 0.12) !important;
  white-space: nowrap !important;
}

div[class*="st-key-mode_tab_act_"] button p {
  font-size: 13.5px !important;
  font-weight: 700 !important;
  color: var(--primary) !important;
}

div[class*="st-key-mode_tab_inact_"] button {
  background: transparent !important;
  color: #5B708B !important;
  border: 1px solid transparent !important;
  border-radius: 7px !important;
  height: 36px !important;
  min-height: 36px !important;
  padding: 0 14px !important;
  box-shadow: none !important;
  white-space: nowrap !important;
}

div[class*="st-key-mode_tab_inact_"] button:hover {
  background: #E8F0FB !important;
  color: var(--text-main) !important;
}

div[class*="st-key-mode_tab_inact_"] button p {
  font-size: 13.5px !important;
  font-weight: 600 !important;
  color: inherit !important;
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
   2. PDF UPLOAD & PROGRESS CARD (Height ~134px, chia 33% / 67%)
   ============================================================================ */
.st-key-upload_card {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-card) !important;
  border-radius: 12px !important;
  box-shadow: var(--shadow-card) !important;
  padding: 12px 18px !important;
}

.st-key-upload_card > div[data-testid="stHorizontalBlock"] {
  align-items: center !important;
  gap: 22px !important;
}

/* Khung Upload nét đứt bên trái (33%) */
.st-key-upload_zone_box {
  position: relative !important;
  height: 104px !important;
  min-height: 104px !important;
  max-height: 104px !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  gap: 0 !important;
}

.st-key-upload_zone_box > div[data-testid="stVerticalBlock"] {
  gap: 0 !important;
}

.upload-visual-overlay {
  height: 104px;
  width: 100%;
  border: 1.5px dashed #9AC2F4;
  border-radius: 8px;
  background: #EFF6FE;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  padding: 0 18px;
  box-sizing: border-box;
  pointer-events: none;
  transition: background-color 0.15s ease, border-color 0.15s ease;
}

.st-key-upload_zone_box:hover .upload-visual-overlay {
  background: #E6F1FE;
  border-color: var(--primary);
}

.upload-visual-title {
  font-size: 15.5px;
  font-weight: 700;
  color: var(--text-main);
  line-height: 1.3;
  margin: 0;
}

.upload-visual-sub {
  font-size: 13.5px;
  font-weight: 400;
  color: var(--text-sec);
  line-height: 1.35;
  margin: 2px 0 0 0;
}

/* Đặt st.file_uploader thật phủ trong suốt khớp 100% lên vùng nét đứt */
.st-key-upload_zone_box div[data-testid="stElementContainer"]:has(div[data-testid="stFileUploader"]) {
  position: absolute !important;
  inset: 0 !important;
  height: 104px !important;
  width: 100% !important;
  z-index: 10 !important;
  margin: 0 !important;
  padding: 0 !important;
}

.st-key-upload_zone_box div[data-testid="stFileUploader"] {
  position: absolute !important;
  inset: 0 !important;
  height: 104px !important;
  width: 100% !important;
  z-index: 10 !important;
  opacity: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

.st-key-upload_zone_box section[data-testid="stFileUploaderDropzone"] {
  height: 104px !important;
  min-height: 104px !important;
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

/* Hai lựa chọn trong cùng khung 104px: nửa trái "Chọn file PDF" | nửa phải "Chọn thư mục" */
.upload-visual-dual {
  flex-direction: column;
  gap: 9px;
  padding: 8px 12px;
}

.upload-dual-top {
  display: flex;
  align-items: center;
  gap: 10px;
}

.upload-visual-dual svg {
  width: 32px;
  height: 32px;
  flex-shrink: 0;
}

.upload-visual-dual .upload-visual-title {
  font-size: 14.5px;
}

.upload-dual-opts {
  display: flex;
  width: 100%;
  gap: 8px;
}

.upload-opt {
  flex: 1;
  text-align: center;
  font-size: 12.5px;
  font-weight: 600;
  color: var(--primary);
  background: #FFFFFF;
  border: 1px solid #C9DDF7;
  border-radius: 6px;
  padding: 4px 0;
  transition: background-color 0.15s ease, color 0.15s ease;
}

.st-key-upload_zone_box:has(.st-key-upl_half_file:hover) .upload-opt-file,
.st-key-upload_zone_box:has(.st-key-upl_half_dir:hover) .upload-opt-dir {
  background: var(--primary);
  border-color: var(--primary);
  color: #FFFFFF;
}

.st-key-upload_zone_box .st-key-upl_half_file,
.st-key-upload_zone_box .st-key-upl_half_dir {
  position: absolute !important;
  top: 0 !important;
  height: 104px !important;
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

/* Khung bao ngoài do Streamlit sinh ra không được chặn click của nửa còn lại */
.st-key-upload_zone_box div:has(> .st-key-upl_half_file),
.st-key-upload_zone_box div:has(> .st-key-upl_half_dir) {
  pointer-events: none !important;
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

/* Cột 'Tệp PDF': đường dẫn dài được rút gọn, rê chuột để xem đầy đủ */
.tbl-file-cell {
  font-size: 12.8px;
  color: var(--text-main);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Khu vực File Processing bên phải (67%) */
.st-key-file_proc_col {
  gap: 8px !important;
}

.file-meta-row {
  display: flex;
  align-items: center;
  gap: 13px;
}

.file-name-txt {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-main);
  line-height: 1.28;
  margin: 0;
  word-break: break-all;
}

.file-sub-txt {
  font-size: 13.8px;
  font-weight: 400;
  color: var(--text-sec);
  line-height: 1.35;
  margin: 2px 0 0 0;
}

/* Nút X tròn xóa file căn sát mép phải */
div[data-testid="stElementContainer"]:has(.st-key-btn_clear_file),
.st-key-btn_clear_file {
  display: flex !important;
  justify-content: flex-end !important;
  width: 100% !important;
}

.st-key-btn_clear_file button {
  width: 30px !important;
  height: 30px !important;
  min-width: 30px !important;
  min-height: 30px !important;
  border-radius: 50% !important;
  padding: 0 !important;
  background: #F1F5FA !important;
  border: 1px solid #E4ECF5 !important;
  color: #70849F !important;
  box-shadow: none !important;
}

.st-key-btn_clear_file button:hover {
  background: #FEE2E2 !important;
  border-color: #FCA5A5 !important;
  color: var(--danger) !important;
}

.st-key-btn_clear_file button p,
.st-key-btn_clear_file button span {
  font-size: 15px !important;
  line-height: 1 !important;
}

/* Thanh tiến trình (Progress Bar 9px + % bên phải + Trạng thái bên dưới) */
.proc-progress-wrap {
  margin-top: 2px;
}

.proc-bar-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.proc-bar-track {
  flex: 1;
  height: 9px;
  background: #E5EEF9;
  border-radius: 999px;
  overflow: hidden;
}

.proc-bar-fill {
  height: 100%;
  background: var(--primary);
  border-radius: 999px;
  transition: width 0.2s ease;
}

.proc-pct-txt {
  font-size: 13.5px;
  font-weight: 500;
  color: var(--text-sec);
  min-width: 38px;
  text-align: right;
}

.proc-status-row {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-top: 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--success);
}

.proc-status-row.scanning {
  color: var(--primary);
}

.proc-status-row.idle {
  color: var(--text-sec);
}

/* ============================================================================
   3. SUMMARY STATISTICS — 4 CARDS (Height ~100px, gap 16px)
   ============================================================================ */
.summary-cards-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  width: 100%;
  margin-bottom: 15px !important;
}

.summary-stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 11px;
  box-shadow: var(--shadow-card);
  height: 97px;
  padding: 0 22px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-sizing: border-box;
}

.summary-icon-box {
  width: 50px;
  height: 50px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.summary-icon-box.blue { background: #EAF3FF; }
.summary-icon-box.red { background: #FDECEE; }
.summary-icon-box.green { background: #E7F8EF; }
.summary-icon-box.orange { background: #FFF3E3; }

.summary-stat-label {
  font-size: 14.5px;
  font-weight: 500;
  color: #5E738F;
  line-height: 1.25;
  margin-bottom: 4px;
}

.summary-stat-val {
  font-size: 29px;
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -0.02em;
}

.summary-stat-val.navy { color: var(--text-main); }
.summary-stat-val.red { color: var(--danger); }
.summary-stat-val.green { color: var(--success); }
.summary-stat-val.orange { color: var(--warning); }

/* ============================================================================
   4. MAIN TWO-COLUMN AREA (61.5% Error Table | 38.5% Document Preview)
   ============================================================================ */
.st-key-main_two_col_row > div[data-testid="stHorizontalBlock"] {
  gap: 14px !important;
  align-items: stretch !important;
}

.st-key-left_error_card,
.st-key-right_preview_card {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-card) !important;
  border-radius: 12px !important;
  box-shadow: var(--shadow-card) !important;
  padding: 16px 20px !important;
  min-height: 492px !important;
  height: 100% !important;
}

.st-key-left_error_card > div[data-testid="stVerticalBlock"],
.st-key-right_preview_card > div[data-testid="stVerticalBlock"] {
  gap: 10px !important;
}

.panel-heading {
  font-size: 20px !important;
  font-weight: 700 !important;
  color: var(--text-main) !important;
  line-height: 1.25 !important;
  margin: 0 !important;
  padding: 0 !important;
  letter-spacing: -0.01em !important;
  white-space: nowrap !important;
}

/* Hàng Filter Pills (Tất cả / Sọc / Lẫn trang / Xoay) */
.st-key-filter_pills_bar {
  gap: 9px !important;
  flex-wrap: wrap !important;
}

div[class*="st-key-flt_act_"] button {
  background: var(--primary) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--primary) !important;
  border-radius: 13px !important;
  height: 36px !important;
  min-height: 36px !important;
  padding: 0 16px !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 6px rgba(22, 119, 242, 0.18) !important;
}

div[class*="st-key-flt_act_"] button p {
  color: #FFFFFF !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
}

div[class*="st-key-flt_inact_"] button {
  background: #F4F7FB !important;
  color: #5A6E87 !important;
  border: 1px solid #E5ECF5 !important;
  border-radius: 13px !important;
  height: 36px !important;
  min-height: 36px !important;
  padding: 0 16px !important;
  font-size: 13.5px !important;
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
  font-size: 13.5px !important;
  font-weight: 500 !important;
}

/* Bảng kết quả 6 cột */
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
  background: #EFF4FA !important;
  border-bottom: 1px solid #E4ECF5 !important;
  min-height: 42px !important;
  padding: 0 !important;
}

div[class*="st-key-tbl_row_norm_"] {
  background: #FFFFFF !important;
  border-bottom: 1px solid #EDF2F8 !important;
  min-height: 55px !important;
  padding: 0 !important;
  transition: background-color 0.12s ease;
}

div[class*="st-key-tbl_row_norm_"]:hover {
  background: #F7FAFE !important;
}

div[class*="st-key-tbl_row_sel_"] {
  background: #E7F1FD !important;
  border-bottom: 1px solid #D8E7FA !important;
  min-height: 55px !important;
  padding: 0 !important;
}

div[class*="st-key-tbl_row_norm_"]:last-child,
div[class*="st-key-tbl_row_sel_"]:last-child {
  border-bottom: none !important;
}

/* Đường kẻ dọc mảnh phân cách giữa các cột trong bảng */
.st-key-error_table_wrapper div[data-testid="stHorizontalBlock"] {
  gap: 0 !important;
  align-items: center !important;
}

.st-key-error_table_wrapper div[data-testid="stColumn"]:not(:last-child) {
  border-right: 1px solid #EEF3F9 !important;
}

.tbl-hdr-cell {
  height: 42px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13.5px;
  font-weight: 600;
  color: #536780;
  text-align: center;
  padding: 0 6px;
}

.tbl-data-cell {
  height: 55px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  color: var(--text-main);
  text-align: center;
  padding: 0 6px;
}

.tbl-page-num {
  font-size: 14.5px;
  font-weight: 700;
  color: var(--text-main);
}

.tbl-rot-txt {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-main);
}

.tbl-dash {
  font-size: 15px;
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
  padding: 4px 13px;
  border-radius: 999px;
  font-size: 12.8px;
  font-weight: 600;
  line-height: 1.3;
  white-space: nowrap;
}

.status-pill.green {
  background: #E2F6EC;
  color: #079B60;
}

.status-pill.orange {
  background: #FFF0D9;
  color: #D97200;
}

.status-pill.gray {
  background: #EEF2F7;
  color: #60738A;
}

/* Nút Xem trong cột cuối của bảng (căn giữa dọc tuyệt đối) */
div[class*="st-key-tbl_action_cell_"],
div[class*="st-key-tbl_action_cell_"] > div[data-testid="stVerticalBlock"] {
  height: 55px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 0 !important;
}

div[class*="st-key-tbl_action_cell_"] div[data-testid="stElementContainer"] {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: auto !important;
  margin: 0 !important;
}

div[class*="st-key-btn_view_p_"] {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  margin: 0 !important;
}

div[class*="st-key-btn_view_p_"] button {
  background: #FFFFFF !important;
  border: 1px solid #D7E2EE !important;
  border-radius: 7px !important;
  height: 32px !important;
  min-height: 32px !important;
  padding: 0 11px !important;
  color: var(--text-main) !important;
  box-shadow: 0 1px 2px rgba(20, 33, 61, 0.03) !important;
}

div[class*="st-key-btn_view_p_"] button:hover {
  border-color: var(--primary) !important;
  color: var(--primary) !important;
  background: #F4F9FF !important;
}

div[class*="st-key-btn_view_p_"] button p {
  font-size: 13px !important;
  font-weight: 600 !important;
  color: inherit !important;
}

div[class*="st-key-btn_view_p_"] button span[data-testid="stIconMaterial"] {
  color: var(--primary) !important;
  font-size: 17px !important;
}

/* Pagination gọn cuối bảng khi > 5 dòng */
.st-key-tbl_pagination_bar {
  padding-top: 4px !important;
  align-items: center !important;
}

div[class*="st-key-pg_nav_"] button {
  height: 28px !important;
  min-height: 28px !important;
  padding: 0 10px !important;
  border-radius: 6px !important;
  font-size: 12.5px !important;
  background: #F5F8FC !important;
  border: 1px solid #E2EAF3 !important;
  color: var(--text-main) !important;
}

.tbl-empty-state {
  min-height: 280px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 24px 20px;
  text-align: center;
  color: var(--text-sec);
}

.tbl-empty-title {
  font-size: 15.5px;
  font-weight: 600;
  color: var(--text-main);
  margin: 10px 0 4px 0;
}

.tbl-empty-sub {
  font-size: 13.5px;
  color: var(--text-sec);
  max-width: 380px;
}

/* ============================================================================
   5. RIGHT PANEL — PDF PREVIEW, DETAIL BOX & ACTION BUTTONS
   ============================================================================ */
.st-key-preview_hdr_row > div[data-testid="stHorizontalBlock"] {
  align-items: center !important;
}

div[data-testid="stElementContainer"]:has(.st-key-preview_tabs_group),
.st-key-preview_tabs_group {
  display: flex !important;
  gap: 6px !important;
  justify-content: flex-end !important;
  width: 100% !important;
}

div[class*="st-key-ptab_act_"] button {
  background: #EDF5FF !important;
  color: var(--primary) !important;
  border: 1.5px solid var(--primary) !important;
  border-radius: 8px !important;
  height: 33px !important;
  min-height: 33px !important;
  padding: 0 13px !important;
  box-shadow: none !important;
}

div[class*="st-key-ptab_act_"] button p {
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--primary) !important;
}

div[class*="st-key-ptab_inact_"] button {
  background: #F4F7FB !important;
  color: var(--text-sec) !important;
  border: 1px solid #E6ECF5 !important;
  border-radius: 8px !important;
  height: 33px !important;
  min-height: 33px !important;
  padding: 0 13px !important;
  box-shadow: none !important;
}

div[class*="st-key-ptab_inact_"] button p {
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text-sec) !important;
}

/* Vùng xem trước ảnh PDF (Cao chuẩn ~214-220px như ảnh mẫu) */
.doc-preview-stage {
  width: 100%;
  height: 214px;
  background: #ECEFF5;
  border: 1px solid #E3E9F2;
  border-radius: 9px;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px 14px;
  box-sizing: border-box;
}

.doc-sheet-frame {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #FFFFFF;
  border-radius: 3px;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.10);
  overflow: hidden;
  max-height: 194px;
  max-width: 82%;
}

.doc-sheet-frame.boundary-focus {
  max-width: 60%;
  max-height: 198px;
}

.doc-sheet-frame img {
  display: block;
  max-height: 194px;
  width: auto;
  max-width: 100%;
  object-fit: contain;
}

.doc-sheet-frame.boundary-focus img {
  max-height: 198px;
  width: 100%;
  height: auto;
}

/* Overlay vùng lỗi theo tọa độ thực tế */
.defect-bbox-overlay {
  position: absolute;
  background: rgba(235, 54, 76, 0.13);
  border: 2px dashed #EB364C;
  border-radius: 5px;
  box-sizing: border-box;
  pointer-events: none;
}

.defect-bbox-overlay.suspected {
  background: rgba(233, 120, 0, 0.13);
  border: 2px dashed #E97800;
}

.defect-bbox-overlay.proposed-ok {
  background: rgba(7, 155, 96, 0.08);
  border: 1.8px dashed #079B60;
}

/* Nhãn pill nổi bên phải vùng lỗi (VD: ● Lẫn đầu/cuối trang) */
.preview-floating-pill {
  position: absolute;
  right: 12px;
  top: 44%;
  transform: translateY(-50%);
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 999px;
  background: #FFF1F3;
  border: 1px solid #FBC4CC;
  color: #D92D43;
  font-size: 12.5px;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(235, 54, 76, 0.12);
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
  color: #079B60;
}

.preview-pill-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
  flex-shrink: 0;
}

.split-preview-pair {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  height: 192px;
  width: 100%;
}

.split-preview-card {
  background: #FFFFFF;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.10);
  padding: 3px;
  max-height: 188px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.split-preview-card img {
  max-height: 164px;
  width: auto;
  object-fit: contain;
}

.split-preview-cap {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-sec);
  margin-top: 2px;
}

/* Error Detail Box (4 hàng chuẩn xác) */
.error-detail-card {
  border: 1px solid var(--border-card);
  border-radius: 9px;
  background: #FFFFFF;
  overflow: hidden;
}

.error-detail-row {
  display: flex;
  align-items: center;
  min-height: 35px;
  padding: 4px 14px;
  border-bottom: 1px solid #EEF2F7;
  font-size: 13.8px;
}

.error-detail-row:last-child {
  border-bottom: none;
}

.error-detail-label {
  width: 130px;
  flex-shrink: 0;
  color: var(--text-sec);
  font-weight: 500;
}

.error-detail-val {
  flex: 1;
  color: var(--text-main);
  font-weight: 600;
  line-height: 1.3;
}

/* Bộ chọn lỗi phụ trên trang đa lỗi (nhỏ gọn, không phá bố cục) */
.st-key-multi_defect_chips {
  gap: 6px !important;
  margin-top: -4px !important;
  margin-bottom: -4px !important;
}

div[class*="st-key-mdchip_act_"] button {
  height: 24px !important;
  min-height: 24px !important;
  padding: 0 9px !important;
  border-radius: 999px !important;
  background: #E8F2FF !important;
  border: 1px solid var(--primary) !important;
  color: var(--primary) !important;
}

div[class*="st-key-mdchip_act_"] button p {
  font-size: 11.5px !important;
  font-weight: 600 !important;
  color: var(--primary) !important;
}

div[class*="st-key-mdchip_inact_"] button {
  height: 24px !important;
  min-height: 24px !important;
  padding: 0 9px !important;
  border-radius: 999px !important;
  background: #F5F7FA !important;
  border: 1px solid #E2E8F0 !important;
  color: var(--text-sec) !important;
}

div[class*="st-key-mdchip_inact_"] button p {
  font-size: 11.5px !important;
  font-weight: 500 !important;
  color: var(--text-sec) !important;
}

/* Hàng 3 nút hành động: Chấp nhận | Bỏ qua | Sửa thủ công */
.st-key-preview_actions_bar > div[data-testid="stHorizontalBlock"] {
  gap: 9px !important;
}

.st-key-act_accept button {
  background: var(--primary) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--primary) !important;
  border-radius: 8px !important;
  height: 40px !important;
  min-height: 40px !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 6px rgba(22, 119, 242, 0.20) !important;
}

.st-key-act_accept button:hover:not(:disabled) {
  background: var(--primary-hover) !important;
  border-color: var(--primary-hover) !important;
}

.st-key-act_accept button p,
.st-key-act_accept button span[data-testid="stIconMaterial"] {
  color: #FFFFFF !important;
  font-size: 13.8px !important;
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
  height: 40px !important;
  min-height: 40px !important;
  font-weight: 600 !important;
  box-shadow: 0 1px 2px rgba(20, 33, 61, 0.03) !important;
}

.st-key-act_skip button:hover:not(:disabled),
.st-key-act_manual button:hover:not(:disabled) {
  background: #F6F9FD !important;
  border-color: #B9CDE6 !important;
}

.st-key-act_skip button p,
.st-key-act_manual button p {
  color: var(--text-main) !important;
  font-size: 13.8px !important;
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
   6. FOOTER INFORMATION BAR (Height ~42px, nền xanh rất nhạt)
   ============================================================================ */
.footer-info-bar {
  width: 100%;
  min-height: 41px;
  background: #EAF1FC;
  border: 1px solid #DCE7F7;
  border-radius: 8px;
  padding: 8px 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  box-sizing: border-box;
  color: #4E6582;
  font-size: 13.5px;
  font-weight: 400;
}

.footer-info-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

/* ============================================================================
   7. MODAL DIALOGS & RESPONSIVE ADAPTATIONS (1366x768, Tablet, Mobile)
   ============================================================================ */
div[role="dialog"] {
  background: #FFFFFF !important;
  color: var(--text-main) !important;
  border-radius: 12px !important;
  border: 1px solid var(--border-card) !important;
}

@media (max-width: 1400px) {
  .stMainBlockContainer,
  div[data-testid="stAppViewBlockContainer"],
  .block-container {
    padding: 0 22px 12px 22px !important;
  }
  .st-key-top_header_bar {
    margin-left: -22px !important;
    margin-right: -22px !important;
    width: calc(100% + 44px) !important;
    padding: 0 22px !important;
  }
}

@media (max-width: 1024px) {
  .summary-cards-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
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
    padding: 0 14px 14px 14px !important;
  }
  .st-key-top_header_bar {
    margin-left: -14px !important;
    margin-right: -14px !important;
    width: calc(100% + 28px) !important;
    padding: 12px 14px !important;
    height: auto !important;
    min-height: auto !important;
  }
  .hdr-title { font-size: 18px !important; }
  .hdr-subtitle { font-size: 13px !important; }
  .st-key-file_proc_col div[data-testid="stHorizontalBlock"] {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
  }
  .st-key-file_proc_col div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
    flex: 1 1 auto !important;
    width: auto !important;
  }
  .st-key-file_proc_col div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {
    flex: 0 0 36px !important;
    width: 36px !important;
  }
  .summary-cards-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }
  .summary-stat-card {
    height: 86px;
    padding: 0 14px;
    gap: 12px;
  }
  .summary-stat-val { font-size: 24px; }
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
  .st-key-hdr_mode_switcher {
    flex-wrap: wrap !important;
  }
  .st-key-error_table_wrapper div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    min-width: 72px !important;
  }
  .footer-info-bar {
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
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
