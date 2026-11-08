"""
Module nhận diện dấu độ mật (`MẬT`, `TỐI MẬT`, `TUYỆT MẬT`, kèm ghi nhận `GIẢI MẬT`)
trên từng trang tài liệu PDF scan (`src/pdf_inspector/detectors/secret/secret_detector.py`).

Thiết kế cho Phase 4 — Part C:
- Tái sử dụng bộ template chuẩn 200 DPI từ gói công cụ hiện có nhưng khắc phục triệt để:
  1. Lỗi ánh xạ chuỗi con `"mat"` làm mất nhãn `TỐI MẬT`, `TUYỆT MẬT`, `GIẢI MẬT`.
  2. Phân tích đa kênh màu (`GRAY`, `RED_SUPPRESSED`, `RED_INK_ENHANCED`, `CLAHE_ENHANCED`)
     giúp bắt tốt cả dấu đỏ, dấu đen, dấu bị con dấu tròn đỏ đè lên và dấu mờ nhạt.
  3. Hỗ trợ đa tỷ lệ (`0.55x .. 1.45x`), khử nghiêng cục bộ từng khung dấu (`±15°`) và quét góc nghiêng nhỏ.
  4. Bộ lọc cô lập vùng trống hai bên dòng cơ sở (`_is_isolated_stamp_region`) chống báo nhầm
     từ `"mật"` / `"BẢO MẬT"` / `"BÍ MẬT"` / `"MẬT ĐỘ"` trong văn bản thông thường.
  5. Phân tầng 2 ngưỡng (`DETECTED` vs `REVIEW`) và trả về `NO_MARK_DETECTED` hoặc `PROCESSING_ERROR` rõ ràng.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pdf_inspector.core.config import (
    SECRET_CONFIRM_THRESHOLD,
    SECRET_RENDER_DPI,
    SECRET_REVIEW_THRESHOLD,
    SECRET_STATUS_DETECTED,
    SECRET_STATUS_ERROR,
    SECRET_STATUS_NO_MARK,
    SECRET_STATUS_REVIEW,
    SECRET_TEMPLATES_DIR,
)
from pdf_inspector.core.logging_config import setup_logger

logger = setup_logger(__name__)

# Thứ tự ưu tiên độ mật khi tổng hợp nhãn chính trên một trang
LABEL_PRIORITY: Dict[str, int] = {
    "TUYỆT MẬT": 4,
    "TỐI MẬT": 3,
    "MẬT": 2,
    "GIẢI MẬT": 1,
}


@dataclass
class DetectedSecretMark:
    """Thông tin chi tiết một dấu độ mật (hoặc nghi vấn) phát hiện trên trang."""

    mark_id: str
    label: str
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) theo hệ tọa độ ảnh trang ở render_dpi
    confidence: float
    status: str  # DETECTED | REVIEW
    detection_method: str  # BOX_TEMPLATE_MATCH | ISOLATED_TEMPLATE_MATCH
    has_border_box: bool = False
    channel_used: str = "GRAY"
    template_name: str = ""
    angle_deg: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["confidence"] = round(float(self.confidence), 4)
        data["angle_deg"] = round(float(self.angle_deg), 2)
        return data


@dataclass
class SecretPageResult:
    """Kết quả kiểm tra dấu độ mật trên một trang PDF."""

    page_number: int
    status: str  # DETECTED | REVIEW | NO_MARK_DETECTED | PROCESSING_ERROR
    primary_label: Optional[str] = None
    marks: List[DetectedSecretMark] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    render_dpi: int = SECRET_RENDER_DPI
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    # Trạng thái rà soát thủ công của người dùng trên giao diện (không tự động kết luận pháp lý)
    review_action: Optional[str] = None  # CONFIRMED | IGNORED | EDITED
    override_label: Optional[str] = None
    notes: str = ""

    @property
    def effective_status(self) -> str:
        """Trạng thái hiển thị sau khi áp dụng thao tác rà soát của người dùng."""
        if self.review_action == "IGNORED":
            return SECRET_STATUS_NO_MARK
        if self.review_action in ("CONFIRMED", "EDITED"):
            if self.override_label == SECRET_STATUS_NO_MARK:
                return SECRET_STATUS_NO_MARK
            return SECRET_STATUS_DETECTED
        return self.status

    @property
    def effective_label(self) -> Optional[str]:
        """Nhãn độ mật hiệu lực sau khi người dùng xác nhận/chỉnh sửa."""
        if self.review_action == "IGNORED":
            return None
        if self.review_action == "EDITED" and self.override_label:
            if self.override_label == SECRET_STATUS_NO_MARK:
                return None
            return self.override_label
        return self.primary_label

    @property
    def active_marks_count(self) -> int:
        """Số lượng dấu hiệu lực trên trang phục vụ thống kê."""
        eff = self.effective_status
        if eff in (SECRET_STATUS_NO_MARK, SECRET_STATUS_ERROR):
            return 0
        return max(1, len(self.marks))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "status": self.status,
            "effective_status": self.effective_status,
            "primary_label": self.primary_label,
            "effective_label": self.effective_label,
            "marks_count": len(self.marks),
            "active_marks_count": self.active_marks_count,
            "marks": [m.to_dict() for m in self.marks],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "render_dpi": self.render_dpi,
            "processing_time_ms": round(float(self.processing_time_ms), 2),
            "error_message": self.error_message,
            "review_action": self.review_action,
            "override_label": self.override_label,
            "notes": self.notes,
        }


class SecretMarkingDetector:
    """
    Detector nhận diện dấu `MẬT`, `TỐI MẬT`, `TUYỆT MẬT` (và `GIẢI MẬT`) bằng Computer Vision thuần
    (OpenCV Multi-Channel Box-First + Isolated Multi-Scale Template Matching).
    """

    def __init__(
        self,
        template_dir: Union[str, Path] = SECRET_TEMPLATES_DIR,
        confirm_threshold: float = SECRET_CONFIRM_THRESHOLD,
        review_threshold: float = SECRET_REVIEW_THRESHOLD,
        dpi: int = SECRET_RENDER_DPI,
    ) -> None:
        self.template_dir: Path = Path(template_dir)
        self.confirm_threshold: float = float(confirm_threshold)
        self.review_threshold: float = float(review_threshold)
        self.dpi: int = int(dpi)
        self.templates: Dict[str, List[Dict[str, Any]]] = self._load_templates()
        self._stage2_cache: List[Dict[str, Any]] = self._build_stage2_cache()

    def _load_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Nạp một lần toàn bộ template chuẩn 200 DPI vào RAM (bỏ qua file ẩn `._*` trên macOS)
        và bổ sung biến thể nét đậm/mảnh cho các nhãn 2 từ (`TỐI MẬT`, `TUYỆT MẬT`, `GIẢI MẬT`).
        """
        templates: Dict[str, List[Dict[str, Any]]] = {
            "MẬT": [],
            "TỐI MẬT": [],
            "TUYỆT MẬT": [],
            "GIẢI MẬT": [],
        }
        if not self.template_dir.exists():
            logger.warning("Thư mục template dấu mật không tồn tại: %s", self.template_dir)
            return templates

        # Thứ tự ưu tiên bắt buộc: kiểm tra chuỗi cụ thể dài hơn trước 'mat'
        file_map: Sequence[Tuple[str, str]] = (
            ("giai_mat", "GIẢI MẬT"),
            ("tuyet_mat", "TUYỆT MẬT"),
            ("toi_mat", "TỐI MẬT"),
            ("mat", "MẬT"),
        )

        for fname in sorted(os.listdir(self.template_dir)):
            if fname.startswith(".") or not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            fpath = self.template_dir / fname
            img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                continue

            lower_name = fname.lower()
            for key, label in file_map:
                if key in lower_name:
                    is_box = "box" in lower_name
                    use_in_stage2 = fname in (
                        "text_giai_mat.png",
                        "text_tuyet_mat.png",
                        "text_toi_mat.png",
                        "text_mat.png",
                        "mat_text_bold.png",
                        "mat_text_thin.png",
                        "mat_box.png",
                    )
                    templates[label].append(
                        {
                            "name": fname,
                            "image": img,
                            "is_box": is_box,
                            "stage2": use_in_stage2,
                            "aspect": float(img.shape[1]) / max(1.0, float(img.shape[0])),
                        }
                    )
                    # Tạo biến thể nét đậm / mảnh cho Stage 1 (Box-First) với nhãn 2 từ
                    if not is_box and label in ("TỐI MẬT", "TUYỆT MẬT", "GIẢI MẬT"):
                        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        bold_img = cv2.erode(img, k2, iterations=1)
                        thin_img = cv2.dilate(img, k2, iterations=1)
                        templates[label].append(
                            {
                                "name": f"{fname}__bold",
                                "image": bold_img,
                                "is_box": False,
                                "stage2": False,
                                "aspect": float(img.shape[1]) / max(1.0, float(img.shape[0])),
                            }
                        )
                        templates[label].append(
                            {
                                "name": f"{fname}__thin",
                                "image": thin_img,
                                "is_box": False,
                                "stage2": False,
                                "aspect": float(img.shape[1]) / max(1.0, float(img.shape[0])),
                            }
                        )
                    break

        return templates

    def _build_stage2_cache(self) -> List[Dict[str, Any]]:
        """
        Tiền xử lý và lưu sẵn bộ template đa tỷ lệ & góc nghiêng nhỏ (`0.5x` thô + `1.0x` tinh)
        trong RAM một lần duy nhất khi khởi tạo detector để giảm tối đa độ trễ mỗi trang.
        """
        cache: List[Dict[str, Any]] = []
        scales_zero = (0.60, 0.72, 0.85, 0.94, 1.0, 1.08, 1.22, 1.36)
        scales_rot = (0.85, 1.0, 1.18)
        pyr_factor = 0.5

        for angle in (0.0, -8.5, 8.5):
            scale_subset = scales_zero if angle == 0.0 else scales_rot
            for label in ("GIẢI MẬT", "TUYỆT MẬT", "TỐI MẬT", "MẬT"):
                for tpl_info in self.templates.get(label, []):
                    if not tpl_info.get("stage2", False):
                        continue
                    if angle != 0.0 and (
                        tpl_info["is_box"]
                        or tpl_info["name"] in ("mat_text_bold.png", "mat_text_thin.png")
                    ):
                        continue

                    base_tpl = tpl_info["image"]
                    rot_tpl = (
                        base_tpl
                        if angle == 0.0
                        else self._rotate_template_white_bg(base_tpl, angle)
                    )
                    tpl_h, tpl_w = rot_tpl.shape[:2]
                    scale_entries: List[Tuple[float, int, int, int, int, np.ndarray, np.ndarray]] = []
                    for scale in scale_subset:
                        sw = int(round(tpl_w * scale))
                        sh = int(round(tpl_h * scale))
                        if sh < 12 or sw < 24:
                            continue
                        c_sw = max(12, int(round(sw * pyr_factor)))
                        c_sh = max(6, int(round(sh * pyr_factor)))
                        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                        fine_tpl = cv2.resize(rot_tpl, (sw, sh), interpolation=interp)
                        coarse_tpl = cv2.resize(fine_tpl, (c_sw, c_sh), interpolation=cv2.INTER_AREA)
                        scale_entries.append((scale, sw, sh, c_sw, c_sh, coarse_tpl, fine_tpl))

                    if scale_entries:
                        cache.append(
                            {
                                "angle": angle,
                                "label": label,
                                "name": str(tpl_info["name"]),
                                "is_box": bool(tpl_info["is_box"]),
                                "scales": scale_entries,
                            }
                        )
        return cache

    @staticmethod
    def _rotate_template_white_bg(tpl: np.ndarray, angle_deg: float) -> np.ndarray:
        """Xoay template một góc nhỏ với nền trắng (255) phục vụ bắt dấu đóng nghiêng."""
        if abs(angle_deg) < 0.5:
            return tpl
        th, tw = tpl.shape[:2]
        rad = np.deg2rad(abs(angle_deg))
        new_w = int(np.ceil(tw * np.cos(rad) + th * np.sin(rad)))
        new_h = int(np.ceil(tw * np.sin(rad) + th * np.cos(rad)))
        mat = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), angle_deg, 1.0)
        mat[0, 2] += (new_w - tw) / 2.0
        mat[1, 2] += (new_h - th) / 2.0
        return cv2.warpAffine(
            tpl,
            mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    @staticmethod
    def _extract_channels(
        roi_img: np.ndarray,
        color_order: str = "RGB",
    ) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
        """
        Trích xuất các kênh ảnh xám chuyên biệt:
        - `GRAY`: kênh độ chói tiêu chuẩn cho dấu đen/xám.
        - `RED_SUPPRESSED`: kênh Đỏ (`R`), triệt tiêu con dấu tròn đỏ đè lên dấu mật màu đen.
        - `RED_INK_ENHANCED`: `min(G, B)`, làm nổi bật dấu độ mật đóng bằng mực đỏ tươi.
        Trả về `(channels_list, clean_text_channel)` trong đó `clean_text_channel` ưu tiên
        `RED_SUPPRESSED` để kiểm tra chữ văn bản xung quanh mà không bị nhiễu bởi con dấu đỏ.
        """
        if roi_img.ndim == 2:
            return [("GRAY", roi_img)], roi_img

        if color_order.upper() == "BGR":
            b_ch, g_ch, r_ch = roi_img[:, :, 0], roi_img[:, :, 1], roi_img[:, :, 2]
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        else:
            r_ch, g_ch, b_ch = roi_img[:, :, 0], roi_img[:, :, 1], roi_img[:, :, 2]
            gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)

        channels: List[Tuple[str, np.ndarray]] = [("GRAY", gray)]

        # Phát hiện mực màu (đỏ tươi, xanh lá `GIẢI MẬT`, xanh dương...) và con dấu tròn màu đỏ
        max_rgb = np.maximum(np.maximum(r_ch, g_ch), b_ch)
        min_rgb = np.minimum(np.minimum(r_ch, g_ch), b_ch)
        chroma = max_rgb.astype(np.int16) - min_rgb.astype(np.int16)
        colored_pixel_count = int(np.count_nonzero(chroma > 32))

        max_gb = np.maximum(g_ch, b_ch)
        redness = r_ch.astype(np.int16) - max_gb.astype(np.int16)
        red_pixel_count = int(np.count_nonzero(redness > 32))

        if colored_pixel_count >= 60:
            # `min(R, G, B)` làm sậm tối đa mọi loại mực màu (đỏ, lục, lam) lẫn mực đen trên nền giấy trắng
            channels.append(("RED_INK_ENHANCED", min_rgb))

        if red_pixel_count >= 60:
            channels.append(("RED_SUPPRESSED", r_ch))
            clean_text_ch = r_ch
        else:
            clean_text_ch = gray

        return channels, clean_text_ch

    @staticmethod
    def _build_faded_channel(base_gray: np.ndarray) -> np.ndarray:
        """Tăng tương phản cục bộ cho trường hợp dấu bị phai mực rất mờ."""
        p5 = float(np.percentile(base_gray, 2.0))
        p95 = float(np.percentile(base_gray, 98.0))
        if p95 - p5 > 8.0:
            stretched = np.clip((base_gray.astype(np.float32) - p5) * (255.0 / (p95 - p5)), 0, 255).astype(np.uint8)
        else:
            stretched = base_gray
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(stretched)

    @staticmethod
    def _find_candidate_boxes(
        gray_roi: np.ndarray,
    ) -> List[Tuple[int, int, int, int, float, float]]:
        """
        Tìm các khung viền hình chữ nhật ứng viên `(x, y, bw, bh, aspect, tilt_deg)`
        bằng cả ngưỡng Otsu lẫn ngưỡng nhạy mực mờ, kết hợp kiểm tra độ kín 4 cạnh viền
        để không nhận nhầm dòng chữ thường thành khung dấu.
        """
        h_roi, w_roi = gray_roi.shape[:2]
        if h_roi < 20 or w_roi < 40:
            return []

        bg_med = float(np.median(gray_roi))
        _, otsu_bin = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        faint_thr = max(40, min(242, int(bg_med - 15)))
        faint_bin = np.where(gray_roi < faint_thr, 255, 0).astype(np.uint8)

        h_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        v_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))

        raw_candidates: List[Tuple[int, int, int, int, float, float]] = []

        for binary in (otsu_bin, faint_bin):
            bridged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_bridge)
            bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, v_bridge)
            contours, _ = cv2.findContours(bridged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < 1800:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 68 or bh < 34 or bw > int(w_roi * 0.78) or bh > int(h_roi * 0.55):
                    continue

                rect = cv2.minAreaRect(cnt)
                (_, _), (rw, rh), raw_angle = rect
                if rw < 30 or rh < 30:
                    continue

                # Chuẩn hóa chiều rộng dài hơn là cạnh ngang của hộp dấu
                if rw >= rh:
                    true_w, true_h = rw, rh
                    tilt_deg = float(raw_angle)
                else:
                    true_w, true_h = rh, rw
                    tilt_deg = float(raw_angle + 90.0)

                while tilt_deg > 45.0:
                    tilt_deg -= 90.0
                while tilt_deg < -45.0:
                    tilt_deg += 90.0

                if abs(tilt_deg) > 20.0:
                    continue

                aspect = float(true_w) / max(1.0, float(true_h))
                if not (1.32 <= aspect <= 6.2):
                    continue

                # Kiểm tra tính chất khung viền chữ nhật kín 4 cạnh (loại bỏ hoàn toàn từ văn bản thường như "NAM")
                if abs(tilt_deg) <= 3.0:
                    band_y = min(6, max(3, int(round(bh * 0.10))))
                    band_x = min(6, max(3, int(round(bw * 0.06))))
                    sub = bridged[y : y + bh, x : x + bw]
                    top_cov = float(np.mean(np.any(sub[:band_y, :] > 0, axis=0)))
                    bot_cov = float(np.mean(np.any(sub[bh - band_y :, :] > 0, axis=0)))
                    left_cov = float(np.mean(np.any(sub[:, :band_x] > 0, axis=1)))
                    right_cov = float(np.mean(np.any(sub[:, bw - band_x :] > 0, axis=1)))
                    if (
                        top_cov < 0.72
                        or bot_cov < 0.72
                        or left_cov < 0.68
                        or right_cov < 0.68
                    ):
                        continue
                else:
                    box_pts = cv2.boxPoints(rect)
                    mask_rot = np.zeros((bh, bw), dtype=np.uint8)
                    shifted_pts = np.int32(box_pts - np.array([[x, y]], dtype=np.float32))
                    cv2.polylines(mask_rot, [shifted_pts], isClosed=True, color=255, thickness=5)
                    sub = bridged[y : y + bh, x : x + bw]
                    perim_pixels = max(1, int(np.count_nonzero(mask_rot)))
                    overlap = int(np.count_nonzero((mask_rot > 0) & (sub > 0)))
                    if float(overlap) / float(perim_pixels) < 0.62:
                        continue

                raw_candidates.append((x, y, bw, bh, aspect, tilt_deg))

        # Khử trùng lặp các hộp ứng viên chồng lấp cao (cùng viền trong & viền ngoài của 1 khung)
        deduped: List[Tuple[int, int, int, int, float, float]] = []
        raw_candidates.sort(key=lambda item: item[2] * item[3], reverse=True)
        for cand in raw_candidates:
            cx, cy, cbw, cbh, _, _ = cand
            c_box = (cx, cy, cx + cbw, cy + cbh)
            if any(_bbox_iou(c_box, (dx, dy, dx + dbw, dy + dbh)) > 0.65 for dx, dy, dbw, dbh, _, _ in deduped):
                continue
            deduped.append(cand)
        return deduped

    @staticmethod
    def _deskew_box_patch(
        gray_roi: np.ndarray,
        bx: int,
        by: int,
        bw: int,
        bh: int,
        tilt_deg: float,
    ) -> np.ndarray:
        """Cắt và khử nghiêng cục bộ một khung dấu bị đóng lệch góc."""
        pad = max(6, int(round(max(bw, bh) * 0.12)))
        h_roi, w_roi = gray_roi.shape[:2]
        x0 = max(0, bx - pad)
        y0 = max(0, by - pad)
        x1 = min(w_roi, bx + bw + pad)
        y1 = min(h_roi, by + bh + pad)
        patch = gray_roi[y0:y1, x0:x1]
        if abs(tilt_deg) < 1.5 or patch.size == 0:
            return gray_roi[by : by + bh, bx : bx + bw]

        ph, pw = patch.shape[:2]
        mat = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), tilt_deg, 1.0)
        rotated = cv2.warpAffine(
            patch,
            mat,
            (pw, ph),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Cắt lại phần trung tâm tương ứng kích thước thực của hộp
        rad = np.deg2rad(abs(tilt_deg))
        cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
        denom = max(0.25, cos_a * cos_a - sin_a * sin_a)
        true_w = int(round(max(20.0, (bw * cos_a - bh * sin_a) / denom)))
        true_h = int(round(max(12.0, (bh * cos_a - bw * sin_a) / denom)))
        cx, cy = pw // 2, ph // 2
        rx0 = max(0, cx - true_w // 2 - 2)
        ry0 = max(0, cy - true_h // 2 - 2)
        rx1 = min(pw, cx + true_w // 2 + 2)
        ry1 = min(ph, cy + true_h // 2 + 2)
        crop = rotated[ry0:ry1, rx0:rx1]
        return crop if crop.size > 0 else gray_roi[by : by + bh, bx : bx + bw]

    @staticmethod
    def _reject_single_word_mat_in_wide_box(
        box_crop: np.ndarray,
        match_x: int,
        match_w: int,
        box_aspect: float,
    ) -> bool:
        """
        Ngăn template 1 từ `MẬT` khớp nhầm vào nửa bên phải của hộp 2 từ (`[TỐI MẬT]`,
        `[TUYỆT MẬT]`, `[GIẢI MẬT]`, `[HỎA TỐC]`).
        """
        bh, bw = box_crop.shape[:2]
        if box_aspect < 2.45 or bw < 40 or bh < 14:
            return False
        if float(match_w) / float(bw) >= 0.64:
            return False

        # Bỏ viền khung để đo mật độ nét chữ bên trong lòng hộp
        trim_y = max(3, int(round(bh * 0.20)))
        trim_x = max(4, int(round(bw * 0.08)))
        if bh - 2 * trim_y < 6 or bw - 2 * trim_x < 16:
            return False

        inner = box_crop[trim_y : bh - trim_y, trim_x : bw - trim_x]
        bg_val = float(np.percentile(inner, 85.0))
        fg_mask = inner < max(35.0, bg_val - 22.0)

        # Kiểm tra phần lòng hộp nằm ngoài vùng khớp của chữ `MẬT`
        rel_x0 = max(0, match_x - trim_x)
        rel_x1 = min(inner.shape[1], match_x + match_w - trim_x)
        left_part = fg_mask[:, : max(0, rel_x0 - 3)]
        right_part = fg_mask[:, min(inner.shape[1], rel_x1 + 3) :]

        left_density = float(np.mean(left_part)) if left_part.size >= 24 else 0.0
        right_density = float(np.mean(right_part)) if right_part.size >= 24 else 0.0
        return max(left_density, right_density) > 0.065

    @staticmethod
    def _is_isolated_stamp_region(
        clean_gray: np.ndarray,
        match_gray: np.ndarray,
        bx: int,
        by: int,
        bw: int,
        bh: int,
    ) -> bool:
        """
        Kiểm tra vùng khớp không khung viền có thực sự là con dấu độc lập hay chỉ là từ `"mật"` /
        `"MẬT"` nằm giữa câu văn / tiêu đề thông thường (`"BẢO MẬT"`, `"BÍ MẬT"`, `"MẬT ĐỘ"`...).

        Nguyên tắc vật lý:
        - Từ `"mật"` trong văn bản thông thường luôn có các từ lân cận nằm sát bên trái hoặc bên phải
          trên cùng dòng cơ sở (`baseline`), với chiều cao thành phần liên thông tương đương chiều cao chữ.
        - Ngược lại, dấu độ mật không khung viền luôn đứng độc lập với khoảng trắng hai bên dòng chữ
          (khi xét trên kênh `clean_gray` đã triệt tiêu con dấu tròn đỏ nếu có).
        """
        h_roi, w_roi = clean_gray.shape[:2]
        if bw < 22 or bh < 12 or bx < 0 or by < 0 or bx + bw > w_roi or by + bh > h_roi:
            return False

        patch = match_gray[by : by + bh, bx : bx + bw]
        if patch.size == 0:
            return False

        p2 = float(np.percentile(patch, 2.0))
        p85 = float(np.percentile(patch, 85.0))
        # Vùng trắng đồng nhất hoặc nhiễu cực yếu không thể là dấu
        if p85 - p2 < 11.0:
            return False

        # Binarize vùng lân cận trên kênh sạch (đã khử con dấu đỏ)
        pad_x = max(24, int(round(bw * 0.85)))
        pad_y = max(8, int(round(bh * 0.25)))
        nx0 = max(0, bx - pad_x)
        nx1 = min(w_roi, bx + bw + pad_x)
        ny0 = max(0, by - pad_y)
        ny1 = min(h_roi, by + bh + pad_y)
        neigh = clean_gray[ny0:ny1, nx0:nx1]

        bg_med = float(np.percentile(neigh, 82.0))
        thr_val = max(35.0, min(238.0, bg_med - max(14.0, (p85 - p2) * 0.35)))
        fg = (neigh < thr_val).astype(np.uint8)

        # Tọa độ của vùng dấu bên trong cửa sổ `neigh`
        rel_bx0 = bx - nx0
        rel_bx1 = rel_bx0 + bw
        rel_by0 = by - ny0
        rel_by1 = rel_by0 + bh

        # Xét dải dòng chữ trung tâm (loại trừ đường kẻ gạch chân tiêu đề phía trên/dưới nếu có)
        band_y0 = rel_by0 + int(round(bh * 0.16))
        band_y1 = rel_by1 - int(round(bh * 0.16))
        if band_y1 <= band_y0 + 4:
            band_y0, band_y1 = rel_by0, rel_by1

        gap = max(4, int(round(bh * 0.14)))
        flank_span = max(18, int(round(bw * 0.65)))

        # Lọc connected components trên `fg` để chỉ tính các nét chữ của từ lân cận độc lập
        # (loại trừ nét bút phê/gạch xóa kéo xuyên từ bên trong dấu ra ngoài biên)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        inline_text_mask = np.zeros_like(fg)
        for lbl in range(1, num_labels):
            cx = int(stats[lbl, cv2.CC_STAT_LEFT])
            cy = int(stats[lbl, cv2.CC_STAT_TOP])
            cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < 6:
                continue
            inside_overlap_x = min(rel_bx1, cx + cw) - max(rel_bx0, cx)
            if inside_overlap_x >= int(round(bw * 0.35)):
                continue
            # Component chữ trên cùng dòng: chiều cao nằm trong [0.16*bh .. 1.35*bh], không quá dài như đường kẻ
            vert_overlap = min(rel_by1, cy + ch) - max(rel_by0, cy)
            if (
                vert_overlap >= int(round(bh * 0.18))
                and int(round(bh * 0.16)) <= ch <= int(round(bh * 1.35))
                and cw <= int(round(bw * 1.6))
            ):
                inline_text_mask[labels == lbl] = 1

        left_x0 = max(0, rel_bx0 - gap - flank_span)
        left_x1 = max(0, rel_bx0 - gap)
        right_x0 = min(neigh.shape[1], rel_bx1 + gap)
        right_x1 = min(neigh.shape[1], rel_bx1 + gap + flank_span)

        left_flank = inline_text_mask[band_y0:band_y1, left_x0:left_x1]
        right_flank = inline_text_mask[band_y0:band_y1, right_x0:right_x1]

        left_density = float(np.mean(left_flank)) if left_flank.size >= 30 else 0.0
        right_density = float(np.mean(right_flank)) if right_flank.size >= 30 else 0.0

        # Nếu ngay sát bên trái hoặc bên phải trên cùng dòng có từ khác (ví dụ "BẢO MẬT", "MẬT ĐỘ", "bí mật")
        if left_density > 0.042 or right_density > 0.042:
            return False

        return True

    @staticmethod
    def _clean_box_scribbles(
        crop_img: np.ndarray,
        inset_x: int,
        inset_y: int,
        bw: int,
        bh: int,
    ) -> Optional[np.ndarray]:
        """
        Khử tạm thời các nét gạch xóa thẳng dài (`X` / gạch ngang bút phê) nằm bên trong lòng khung dấu
        (giữ nguyên 4 cạnh viền khung bên ngoài) để nhận diện được dấu bị gạch xóa và chuyển sang `REVIEW`.
        Nếu không có nét gạch thẳng dài xuyên qua lòng hộp thì trả về `None` (không can thiệp vào chữ thường).
        """
        ch_h, ch_w = crop_img.shape[:2]
        iy0 = max(0, inset_y + 7)
        iy1 = min(ch_h, inset_y + bh - 7)
        ix0 = max(0, inset_x + 7)
        ix1 = min(ch_w, inset_x + bw - 7)
        if iy1 - iy0 < 14 or ix1 - ix0 < 36:
            return None

        inner = crop_img[iy0:iy1, ix0:ix1]
        bg_val = float(np.percentile(inner, 85.0))
        fg = ((inner < max(45.0, bg_val - 45.0)).astype(np.uint8)) * 255
        min_len = max(28, int(round((ix1 - ix0) * 0.54)))
        lines = cv2.HoughLinesP(
            fg,
            1.0,
            np.pi / 180.0,
            threshold=32,
            minLineLength=min_len,
            maxLineGap=8,
        )
        if lines is None or len(lines) == 0:
            return None

        mask = np.zeros_like(crop_img, dtype=np.uint8)
        for line in lines[:, 0]:
            x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            cv2.line(mask, (ix0 + x1, iy0 + y1), (ix0 + x2, iy0 + y2), 255, 5)
        return cv2.inpaint(crop_img, mask, 3.0, cv2.INPAINT_TELEA)

    def _match_stage1_boxes(
        self,
        channels: Sequence[Tuple[str, np.ndarray]],
        y_offset: int = 0,
    ) -> List[DetectedSecretMark]:
        """
        Nhánh 1 (Box-First Multi-Scale + Per-Box Deskew + Scribble Recovery):
        Tìm khung viền chữ nhật trước rồi so khớp đa tỷ lệ trong lòng khung.
        """
        candidates: List[DetectedSecretMark] = []
        base_scales = (1.0, 0.94, 1.06, 0.85, 1.15, 0.75, 1.28, 0.64)

        for ch_name, ch_img in channels:
            h_ch, w_ch = ch_img.shape[:2]
            boxes = self._find_candidate_boxes(ch_img)
            if not boxes:
                continue

            for bx, by, bw, bh, box_aspect, tilt_deg in boxes:
                pad_b = max(10, int(round(min(bw, bh) * 0.22)))
                px0 = max(0, bx - pad_b)
                py0 = max(0, by - pad_b)
                px1 = min(w_ch, bx + bw + pad_b)
                py1 = min(h_ch, by + bh + pad_b)
                raw_crop = ch_img[py0:py1, px0:px1]
                crops_to_check: List[Tuple[np.ndarray, float, bool]] = [
                    (raw_crop, 0.0, False)
                ]
                if abs(tilt_deg) >= 1.8:
                    deskewed = self._deskew_box_patch(ch_img, bx, by, bw, bh, tilt_deg)
                    deskewed_padded = cv2.copyMakeBorder(
                        deskewed, 10, 10, 12, 12, cv2.BORDER_CONSTANT, value=int(np.percentile(deskewed, 85))
                    )
                    crops_to_check.insert(0, (deskewed_padded, tilt_deg, False))

                scribble_crop = self._clean_box_scribbles(raw_crop, bx - px0, by - py0, bw, bh)
                if scribble_crop is not None:
                    crops_to_check.append((scribble_crop, 0.0, True))

                best_for_box: Optional[DetectedSecretMark] = None

                for crop_img, used_angle, is_scribble_cleaned in crops_to_check:
                    if (
                        is_scribble_cleaned
                        and best_for_box is not None
                        and best_for_box.status == SECRET_STATUS_DETECTED
                    ):
                        break

                    ch_h, ch_w = crop_img.shape[:2]
                    if ch_h < 12 or ch_w < 28:
                        continue

                    for label in ("GIẢI MẬT", "TUYỆT MẬT", "TỐI MẬT", "MẬT"):
                        tpl_list = self.templates.get(label, [])
                        for tpl_info in tpl_list:
                            tpl = tpl_info["image"]
                            tpl_h, tpl_w = tpl.shape[:2]
                            tpl_aspect = float(tpl_info["aspect"])

                            if tpl_info["is_box"] and abs(tpl_aspect - box_aspect) > 1.45:
                                continue

                            fit_ratio_w = (
                                (float(bw + 14) / float(tpl_w))
                                if tpl_info["is_box"]
                                else (float(bw) * 0.88) / float(tpl_w)
                            )
                            fit_ratio_h = (
                                (float(bh + 14) / float(tpl_h))
                                if tpl_info["is_box"]
                                else (float(bh) * 0.84) / float(tpl_h)
                            )
                            dynamic_scales: List[float] = []
                            for s_cand in (fit_ratio_w, fit_ratio_h, (fit_ratio_w + fit_ratio_h) * 0.5):
                                if 0.42 <= s_cand <= 1.95:
                                    dynamic_scales.append(round(s_cand, 3))
                            for s_std in base_scales:
                                if all(abs(s_std - ds) > 0.035 for ds in dynamic_scales):
                                    dynamic_scales.append(s_std)

                            for scale in dynamic_scales:
                                sw = int(round(tpl_w * scale))
                                sh = int(round(tpl_h * scale))
                                if sh > ch_h or sw > ch_w or sh < 10 or sw < 20:
                                    continue

                                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                                scaled_tpl = cv2.resize(tpl, (sw, sh), interpolation=interp)
                                res = cv2.matchTemplate(crop_img, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                                score = float(max_val)

                                min_req = (
                                    max(self.review_threshold, 0.62)
                                    if is_scribble_cleaned
                                    else self.review_threshold
                                )
                                if score < min_req:
                                    continue

                                if label == "MẬT" and self._reject_single_word_mat_in_wide_box(
                                    crop_img, max_loc[0], sw, box_aspect
                                ):
                                    continue

                                if is_scribble_cleaned:
                                    status = SECRET_STATUS_REVIEW
                                    note_txt = "Dấu có nét gạch xóa / bút phê đè lên — cần kiểm tra thủ công."
                                else:
                                    status = (
                                        SECRET_STATUS_DETECTED
                                        if score >= self.confirm_threshold
                                        else SECRET_STATUS_REVIEW
                                    )
                                    note_txt = ""

                                if best_for_box is None or score > best_for_box.confidence:
                                    best_for_box = DetectedSecretMark(
                                        mark_id="",
                                        label=label,
                                        bbox=(bx, by + y_offset, bx + bw, by + bh + y_offset),
                                        confidence=score,
                                        status=status,
                                        detection_method="BOX_TEMPLATE_MATCH",
                                        has_border_box=True,
                                        channel_used=ch_name,
                                        template_name=str(tpl_info["name"]),
                                        angle_deg=used_angle,
                                        notes=note_txt,
                                    )

                                if score >= 0.96 and not is_scribble_cleaned:
                                    break

                if best_for_box is not None:
                    candidates.append(best_for_box)

        return candidates

    def _match_stage2_isolated(
        self,
        channels: Sequence[Tuple[str, np.ndarray]],
        clean_text_ch: np.ndarray,
        y_offset: int = 0,
        only_zero_angle: bool = False,
    ) -> List[DetectedSecretMark]:
        """
        Nhánh 2 (Coarse-to-Fine 0.5x Pyramid + Pre-Cached Multi-Scale & Small-Angle Templates):
        Dành cho dấu không có khung viền hoặc khung viền bị đứt gãy nặng / con dấu đỏ đè lên.
        """
        candidates: List[DetectedSecretMark] = []
        pyr_factor = 0.5
        coarse_gate = max(0.42, self.review_threshold - 0.11)

        for ch_name, ch_img in channels:
            h_roi, w_roi = ch_img.shape[:2]
            if h_roi < 30 or w_roi < 60:
                continue

            flank_ref_ch = ch_img if ch_name == "RED_INK_ENHANCED" else clean_text_ch
            cw_roi = max(30, int(round(w_roi * pyr_factor)))
            ch_roi = max(16, int(round(h_roi * pyr_factor)))
            coarse_roi = cv2.resize(ch_img, (cw_roi, ch_roi), interpolation=cv2.INTER_AREA)

            best_on_channel: Optional[DetectedSecretMark] = None

            for entry in self._stage2_cache:
                angle = float(entry["angle"])
                if only_zero_angle and angle != 0.0:
                    continue
                if angle != 0.0 and best_on_channel is not None and best_on_channel.confidence >= self.confirm_threshold:
                    continue

                label = str(entry["label"])
                if (
                    label == "MẬT"
                    and best_on_channel is not None
                    and best_on_channel.label in ("TUYỆT MẬT", "TỐI MẬT", "GIẢI MẬT")
                    and best_on_channel.confidence >= self.review_threshold + 0.04
                ):
                    continue

                is_box = bool(entry["is_box"])
                tpl_name = str(entry["name"])

                for _scale, sw, sh, c_sw, c_sh, coarse_tpl, fine_tpl in entry["scales"]:
                    if sh >= h_roi or sw >= w_roi or c_sh >= ch_roi or c_sw >= cw_roi:
                        continue

                    work_res = cv2.matchTemplate(coarse_roi, coarse_tpl, cv2.TM_CCOEFF_NORMED)

                    for _ in range(3):
                        _, c_max, _, c_loc = cv2.minMaxLoc(work_res)
                        if float(c_max) < coarse_gate:
                            break

                        cmx, cmy = int(c_loc[0]), int(c_loc[1])
                        sup_x0 = max(0, cmx - max(5, c_sw // 2))
                        sup_y0 = max(0, cmy - max(4, c_sh // 2))
                        sup_x1 = min(work_res.shape[1], cmx + max(5, c_sw // 2))
                        sup_y1 = min(work_res.shape[0], cmy + max(4, c_sh // 2))
                        work_res[sup_y0:sup_y1, sup_x0:sup_x1] = -1.0

                        approx_x = int(round(cmx / pyr_factor))
                        approx_y = int(round(cmy / pyr_factor))
                        margin = 14
                        rx0 = max(0, approx_x - margin)
                        ry0 = max(0, approx_y - margin)
                        rx1 = min(w_roi, approx_x + sw + margin)
                        ry1 = min(h_roi, approx_y + sh + margin)
                        if rx1 - rx0 < sw or ry1 - ry0 < sh:
                            continue

                        fine_patch = ch_img[ry0:ry1, rx0:rx1]
                        fine_res = cv2.matchTemplate(fine_patch, fine_tpl, cv2.TM_CCOEFF_NORMED)
                        _, f_max, _, f_loc = cv2.minMaxLoc(fine_res)
                        score = float(f_max)

                        if score < self.review_threshold:
                            continue

                        mx = rx0 + int(f_loc[0])
                        my = ry0 + int(f_loc[1])

                        if not is_box and not self._is_isolated_stamp_region(
                            flank_ref_ch, ch_img, mx, my, sw, sh
                        ):
                            continue

                        status = (
                            SECRET_STATUS_DETECTED
                            if score >= self.confirm_threshold
                            else SECRET_STATUS_REVIEW
                        )
                        mark = DetectedSecretMark(
                            mark_id="",
                            label=label,
                            bbox=(mx, my + y_offset, mx + sw, my + sh + y_offset),
                            confidence=score,
                            status=status,
                            detection_method="ISOLATED_TEMPLATE_MATCH",
                            has_border_box=is_box,
                            channel_used=ch_name,
                            template_name=tpl_name,
                            angle_deg=angle,
                        )
                        candidates.append(mark)
                        if best_on_channel is None or score > best_on_channel.confidence:
                            best_on_channel = mark
                        break

                    if best_on_channel is not None and best_on_channel.confidence >= 0.94:
                        break

        return candidates

    @staticmethod
    def _nms_marks(
        marks: Sequence[DetectedSecretMark],
        page_number: int,
        iou_threshold: float = 0.32,
    ) -> List[DetectedSecretMark]:
        """
        Khử trùng lặp Non-Maximum Suppression giữa các kênh/tỷ lệ:
        - Ưu tiên kết quả có khung viền (`has_border_box=True`) và nhãn 2 từ đầy đủ hơn khi chồng lấp.
        - Giữ nguyên nhiều dấu độc lập trên cùng trang (ví dụ vừa có `[MẬT]` vừa có `[GIẢI MẬT]`).
        """
        if not marks:
            return []

        def _rank_key(m: DetectedSecretMark) -> Tuple[int, float, int]:
            box_bonus = 0.05 if m.has_border_box else 0.0
            two_word_bonus = 0.14 if m.label in ("TỐI MẬT", "TUYỆT MẬT", "GIẢI MẬT") else 0.0
            is_detected = 1 if (m.status == SECRET_STATUS_DETECTED or two_word_bonus > 0) else 0
            return (is_detected, m.confidence + box_bonus + two_word_bonus, LABEL_PRIORITY.get(m.label, 0))

        ordered = sorted(marks, key=_rank_key, reverse=True)
        kept: List[DetectedSecretMark] = []

        for cand in ordered:
            overlapped = False
            for existing in kept:
                iou = _bbox_iou(cand.bbox, existing.bbox)
                inter_ratio = _intersection_over_smaller(cand.bbox, existing.bbox)
                if iou >= iou_threshold or inter_ratio >= 0.60:
                    overlapped = True
                    break
            if not overlapped:
                kept.append(cand)

        # Sắp xếp lại theo thứ tự xuất hiện từ trên xuống dưới, trái sang phải và gán mark_id
        kept.sort(key=lambda m: (m.bbox[1], m.bbox[0]))
        for idx, m in enumerate(kept, start=1):
            m.mark_id = f"p{page_number}_m{idx}"
        return kept

    def detect_page(
        self,
        image: Optional[np.ndarray],
        page_number: int = 1,
        color_order: str = "RGB",
    ) -> SecretPageResult:
        """
        Phân tích một trang ảnh (`RGB`, `BGR`, hoặc `GRAY`) và trả về `SecretPageResult`.
        Không bao giờ ném ngoại lệ ra ngoài luồng chính: nếu ảnh hỏng hoặc lỗi xử lý,
        trả về `status = PROCESSING_ERROR`.
        """
        t0 = time.perf_counter()
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return SecretPageResult(
                page_number=page_number,
                status=SECRET_STATUS_ERROR,
                render_dpi=self.dpi,
                processing_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                error_message="Ảnh trang rỗng hoặc không hợp lệ khi kiểm tra dấu độ mật.",
            )

        if image.ndim not in (2, 3):
            return SecretPageResult(
                page_number=page_number,
                status=SECRET_STATUS_ERROR,
                render_dpi=self.dpi,
                processing_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                error_message=f"Số chiều mảng ảnh không hợp lệ: ndim={image.ndim}.",
            )

        try:
            h, w = int(image.shape[0]), int(image.shape[1])
            if h < 40 or w < 40:
                return SecretPageResult(
                    page_number=page_number,
                    status=SECRET_STATUS_ERROR,
                    image_width=w,
                    image_height=h,
                    render_dpi=self.dpi,
                    processing_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    error_message=f"Kích thước ảnh trang quá nhỏ ({w}x{h}px).",
                )

            if not any(self.templates.values()):
                return SecretPageResult(
                    page_number=page_number,
                    status=SECRET_STATUS_ERROR,
                    image_width=w,
                    image_height=h,
                    render_dpi=self.dpi,
                    processing_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    error_message="Không nạp được bộ template dấu độ mật.",
                )

            # 1. Vùng Header đầy đủ chiều ngang (40% chiều cao trên cùng)
            header_h = max(40, int(round(h * 0.40)))
            header_roi = image[:header_h, :]
            channels, clean_text_ch = self._extract_channels(header_roi, color_order=color_order)

            all_candidates: List[DetectedSecretMark] = []

            # Stage 1 trên Header ROI (Box-First)
            stage1_marks = self._match_stage1_boxes(channels, y_offset=0)
            all_candidates.extend(stage1_marks)

            # Kiểm tra nhẹ Box-First ở dải lề dưới (32% cuối trang) phòng trường hợp đóng dấu cuối trang / trang ngược
            footer_y0 = int(round(h * 0.68))
            if h - footer_y0 >= 40:
                footer_roi = image[footer_y0:, :]
                footer_channels, _ = self._extract_channels(footer_roi, color_order=color_order)
                all_candidates.extend(self._match_stage1_boxes(footer_channels, y_offset=footer_y0))

            has_strong_box_hit = any(
                m.status == SECRET_STATUS_DETECTED and m.confidence >= 0.84
                for m in all_candidates
            )

            # Stage 2 (Isolated Multi-Scale & Small-Angle) khi chưa có khung dấu rõ hoặc cần bắt dấu không viền
            if not has_strong_box_hit:
                base_for_faded = (
                    channels[1][1]
                    if len(channels) > 1 and channels[1][0] == "RED_INK_ENHANCED"
                    else channels[0][1]
                )
                faded_ch = self._build_faded_channel(base_for_faded)

                # Thử lại Box-First trên kênh CLAHE trước khi quét trượt
                all_candidates.extend(
                    self._match_stage1_boxes([("CLAHE_ENHANCED", faded_ch)], y_offset=0)
                )
                has_box_confirmed = any(
                    m.status == SECRET_STATUS_DETECTED for m in all_candidates
                )
                if not has_box_confirmed:
                    stage2_primary = self._match_stage2_isolated(
                        channels, clean_text_ch=clean_text_ch, y_offset=0
                    )
                    all_candidates.extend(stage2_primary)
                    if not any(m.status == SECRET_STATUS_DETECTED for m in stage2_primary):
                        all_candidates.extend(
                            self._match_stage2_isolated(
                                [("CLAHE_ENHANCED", faded_ch)],
                                clean_text_ch=clean_text_ch,
                                y_offset=0,
                                only_zero_angle=True,
                            )
                        )

            deduped = self._nms_marks(all_candidates, page_number=page_number)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            if not deduped:
                return SecretPageResult(
                    page_number=page_number,
                    status=SECRET_STATUS_NO_MARK,
                    primary_label=None,
                    marks=[],
                    image_width=w,
                    image_height=h,
                    render_dpi=self.dpi,
                    processing_time_ms=elapsed_ms,
                )

            # Xác định trạng thái trang và nhãn ưu tiên (hỗ trợ trường hợp có đồng thời MẬT + GIẢI MẬT)
            has_detected = any(m.status == SECRET_STATUS_DETECTED for m in deduped)
            page_status = SECRET_STATUS_DETECTED if has_detected else SECRET_STATUS_REVIEW

            labels_present = [m.label for m in deduped]
            secret_labels = [lbl for lbl in labels_present if lbl in ("TUYỆT MẬT", "TỐI MẬT", "MẬT")]
            has_giai_mat = "GIẢI MẬT" in labels_present

            notes = ""
            if secret_labels:
                primary_label = max(secret_labels, key=lambda lbl: LABEL_PRIORITY.get(lbl, 0))
                if has_giai_mat:
                    notes = f"Phát hiện đồng thời dấu {primary_label} và dấu GIẢI MẬT trên cùng trang."
            else:
                primary_label = "GIẢI MẬT"
                notes = "Phát hiện dấu GIẢI MẬT."

            return SecretPageResult(
                page_number=page_number,
                status=page_status,
                primary_label=primary_label,
                marks=deduped,
                image_width=w,
                image_height=h,
                render_dpi=self.dpi,
                processing_time_ms=elapsed_ms,
                notes=notes,
            )

        except Exception as exc:
            logger.error("Lỗi khi phân tích dấu độ mật tại trang %d: %s", page_number, exc)
            return SecretPageResult(
                page_number=page_number,
                status=SECRET_STATUS_ERROR,
                render_dpi=self.dpi,
                processing_time_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                error_message=str(exc),
            )


def _bbox_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    ix0 = max(box_a[0], box_b[0])
    iy0 = max(box_a[1], box_b[1])
    ix1 = min(box_a[2], box_b[2])
    iy1 = min(box_a[3], box_b[3])
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
    area_b = float(max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])))
    return inter / max(1.0, area_a + area_b - inter)


def _intersection_over_smaller(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    ix0 = max(box_a[0], box_b[0])
    iy0 = max(box_a[1], box_b[1])
    ix1 = min(box_a[2], box_b[2])
    iy1 = min(box_a[3], box_b[3])
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
    area_b = float(max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])))
    return inter / max(1.0, min(area_a, area_b))


def _load_vietnamese_font(font_size: int = 18) -> ImageFont.ImageFont:
    font_paths = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    )
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_secret_overlay(
    rgb_image: np.ndarray,
    page_result: SecretPageResult,
) -> np.ndarray:
    """
    Vẽ bounding box dấu độ mật thực tế và nhãn tiếng Việt sắc nét lên ảnh xem trước `RGB`
    (tự động quy đổi tỷ lệ tọa độ nếu `rgb_image` được render ở `PREVIEW_DPI` khác `render_dpi`).
    """
    if rgb_image is None or rgb_image.size == 0 or not page_result.marks:
        return rgb_image

    if rgb_image.ndim == 2:
        canvas_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
    else:
        canvas_rgb = rgb_image.copy()

    h_vis, w_vis = canvas_rgb.shape[:2]
    scale_x = float(w_vis) / max(1.0, float(page_result.image_width or w_vis))
    scale_y = float(h_vis) / max(1.0, float(page_result.image_height or h_vis))

    pil_img = Image.fromarray(canvas_rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")
    font = _load_vietnamese_font(font_size=max(13, int(round(16 * min(scale_x, scale_y) + 4))))

    for mark in page_result.marks:
        x0 = max(0, int(round(mark.bbox[0] * scale_x)))
        y0 = max(0, int(round(mark.bbox[1] * scale_y)))
        x1 = min(w_vis - 1, int(round(mark.bbox[2] * scale_x)))
        y1 = min(h_vis - 1, int(round(mark.bbox[3] * scale_y)))

        if mark.status == SECRET_STATUS_REVIEW:
            stroke_rgb = (233, 120, 0, 255)
            fill_rgba = (233, 120, 0, 34)
        elif mark.label == "GIẢI MẬT":
            stroke_rgb = (7, 155, 96, 255)
            fill_rgba = (7, 155, 96, 30)
        else:
            stroke_rgb = (235, 54, 76, 255)
            fill_rgba = (235, 54, 76, 34)

        pad = 4
        bx0 = max(0, x0 - pad)
        by0 = max(0, y0 - pad)
        bx1 = min(w_vis - 1, x1 + pad)
        by1 = min(h_vis - 1, y1 + pad)
        draw.rectangle([bx0, by0, bx1, by1], fill=fill_rgba, outline=stroke_rgb, width=3)

        display_lbl = page_result.effective_label or mark.label
        badge_text = f"{display_lbl} ({mark.confidence * 100:.0f}%)"
        tb = draw.textbbox((0, 0), badge_text, font=font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        tag_x0 = bx0
        tag_y0 = by0 - th - 10 if by0 >= th + 12 else min(h_vis - th - 10, by1 + 4)
        draw.rounded_rectangle(
            [tag_x0, tag_y0, tag_x0 + tw + 14, tag_y0 + th + 8],
            radius=5,
            fill=stroke_rgb,
        )
        draw.text((tag_x0 + 7, tag_y0 + 3), badge_text, fill=(255, 255, 255, 255), font=font)

    return np.array(pil_img.convert("RGB"))


def crop_secret_mark(
    rgb_image: np.ndarray,
    mark: DetectedSecretMark,
    source_size: Tuple[int, int],
    padding_ratio: float = 0.22,
) -> Optional[np.ndarray]:
    """
    Trích xuất ảnh crop phóng to của dấu độ mật hoàn toàn trong RAM (không ghi xuống đĩa — C4).
    """
    if rgb_image is None or rgb_image.size == 0:
        return None
    h_vis, w_vis = rgb_image.shape[:2]
    src_w, src_h = source_size
    scale_x = float(w_vis) / max(1.0, float(src_w or w_vis))
    scale_y = float(h_vis) / max(1.0, float(src_h or h_vis))

    x0 = int(round(mark.bbox[0] * scale_x))
    y0 = int(round(mark.bbox[1] * scale_y))
    x1 = int(round(mark.bbox[2] * scale_x))
    y1 = int(round(mark.bbox[3] * scale_y))

    bw = max(8, x1 - x0)
    bh = max(8, y1 - y0)
    pad_x = max(8, int(round(bw * padding_ratio)))
    pad_y = max(6, int(round(bh * padding_ratio)))

    cx0 = max(0, x0 - pad_x)
    cy0 = max(0, y0 - pad_y)
    cx1 = min(w_vis, x1 + pad_x)
    cy1 = min(h_vis, y1 + pad_y)

    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return rgb_image[cy0:cy1, cx0:cx1].copy()
