#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Công cụ quét và nhận diện tài liệu mật SIÊU TỐC ĐỘ (High-Performance Fast Detector)
Áp dụng:
1. Clipped Grayscale Render (chỉ render 35% vùng Header, xuất thẳng ảnh xám 150 DPI)
2. Fast Thumbnail Deskew (chỉnh nghiêng trên ảnh thu nhỏ)
3. Box-First Verification (tìm khung trước, chỉ so khớp template trong lòng hộp nhỏ 150x80px)
4. Multiprocessing Pool (song song hóa đa nhân CPU)
"""

import os
import sys
import glob
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2
import pymupdf
from PIL import Image, ImageDraw, ImageFont


class FastConfidentialDocDetector:
    def __init__(self, template_dir="templates", threshold=0.80, dpi=200):
        """
        :param template_dir: Thư mục chứa templates mẫu
        :param threshold: Ngưỡng chấp nhận kết quả (mặc định 0.80)
        :param dpi: Độ phân giải chuẩn 200 DPI giữ trọn vẹn độ sắc nét của chữ
        """
        self.template_dir = template_dir
        self.threshold = threshold
        self.dpi = dpi
        self.templates = self._load_templates()

    def _load_templates(self):
        """Tải toàn bộ template chuẩn nguyên bản 200 DPI"""
        templates = {
            "MẬT": [],
            "TỐI MẬT": [],
            "TUYỆT MẬT": [],
            "GIẢI MẬT": []
        }

        if not os.path.exists(self.template_dir):
            return templates

        # Thứ tự kiểm tra ưu tiên: chuỗi dài/chi tiết hơn phải kiểm tra trước 'mat'
        file_map = [
            ("giai_mat", "GIẢI MẬT"),
            ("tuyet_mat", "TUYỆT MẬT"),
            ("toi_mat", "TỐI MẬT"),
            ("mat", "MẬT")
        ]

        for fname in sorted(os.listdir(self.template_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            fpath = os.path.join(self.template_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            for key, label in file_map:
                if key in fname.lower():
                    templates[label].append({
                        "name": fname,
                        "image": img,
                        "is_box": "box" in fname.lower()
                    })
                    break

        return templates

    def deskew_image(self, gray_roi):
        """
        Tự động khử nghiêng (Auto-Deskew) độ chính xác cao trên ảnh gốc chuẩn độ phân giải
        Sử dụng phép nội suy INTER_CUBIC để đảm bảo không làm suy giảm chất lượng và độ sắc nét của chữ.
        """
        thresh = cv2.bitwise_not(gray_roi)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return gray_roi, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle

        # Chỉ xoay nếu góc nghiêng nhỏ hơn 15 độ (tránh xoay sai tài liệu đứng)
        if abs(angle) > 0.5 and abs(angle) < 15.0:
            (h, w) = gray_roi.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray_roi, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated, angle

        return gray_roi, 0.0

    def render_header_roi(self, pdf_path, page_num=0):
        """
        CẤP 1: Render trực tiếp vùng Header bằng PyMuPDF Clip ở chuẩn 200 DPI RGB
        Chuyển đổi xám bằng OpenCV chuẩn ITU-R BT.601 để giữ trọn độ tương phản và chi tiết nét chữ.
        """
        try:
            doc = pymupdf.open(pdf_path)
            if page_num >= len(doc):
                return None, 0, (0, 0)
            page = doc[page_num]
            rect = page.rect
            total_pages = len(doc)

            # Chỉ render góc trên bên trái: 50% chiều ngang, 35% chiều dọc
            clip_header = pymupdf.Rect(0, 0, rect.width * 0.60, rect.height * 0.35)
            pix = page.get_pixmap(dpi=self.dpi, clip=clip_header)
            
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return roi_gray, total_pages, (rect.width, rect.height)
        except Exception as e:
            return None, 0, (0, 0)

    def find_rectangular_boxes(self, gray_roi):
        """Tìm các khung chữ nhật ứng viên bằng Cầu nối hình thái học có hướng"""
        h_roi, w_roi = gray_roi.shape[:2]
        _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Nối các khe nứt vi mô trên viền hộp
        h_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        v_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        bridged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, h_bridge)
        bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, v_bridge)

        contours, _ = cv2.findContours(bridged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / float(bh)
            # Tỷ lệ chuẩn hộp dấu mật
            if 1.5 < aspect < 6.5 and 35 < bw < int(w_roi * 0.8) and 14 < bh < int(h_roi * 0.4):
                candidate_boxes.append((x, y, bw, bh, aspect))

        return candidate_boxes

    def detect_stamp(self, roi_gray):
        """
        Chiến lược nhận diện siêu tốc:
        - Bước 1: Fast Deskew
        - Bước 2 (Box-First): Nếu tìm thấy khung chữ nhật, chỉ so khớp TRONG LÒNG HỘP (1.5ms)
        - Bước 3 (Fallback): Chỉ khi không có hộp mới quét trượt toàn vùng
        """
        roi_clean, _ = self.deskew_image(roi_gray)
        h_roi, w_roi = roi_clean.shape[:2]

        detected_boxes = self.find_rectangular_boxes(roi_clean)

        best_match = {
            "status": "BÌNH THƯỜNG",
            "confidence": 0.0,
            "box": None,
            "template_name": None,
            "has_border_box": False
        }

        # -------------------------------------------------------------
        # NHÁNH 1: BOX-FIRST VERIFICATION (Siêu nhanh: ~1.5 - 3 ms)
        # -------------------------------------------------------------
        found_matches = []
        if detected_boxes:
            for (bx, by, bw, bh, _) in detected_boxes:
                # Cắt đúng lòng hộp chữ nhật
                box_crop = roi_clean[by:by+bh, bx:bx+bw]
                if box_crop.shape[0] < 10 or box_crop.shape[1] < 20:
                    continue

                for label in ["GIẢI MẬT", "TUYỆT MẬT", "TỐI MẬT", "MẬT"]:
                    for tpl_info in self.templates.get(label, []):
                        tpl = tpl_info["image"]
                        best_tpl_score = 0.0
                        # Ưu tiên scale 1.0 chuẩn trước, sau đó mới quét các scale co dãn
                        for scale in [1.0, 0.95, 1.05, 0.85, 1.10, 0.75, 0.70]:
                            sw = int(tpl.shape[1] * scale)
                            sh = int(tpl.shape[0] * scale)
                            if sh >= bh or sw >= bw or sh < 10 or sw < 20:
                                continue

                            scaled_tpl = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
                            res = cv2.matchTemplate(box_crop, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(res)

                            if max_val > best_tpl_score:
                                best_tpl_score = max_val

                            # Chỉ early exit khi điểm cực kỳ xuất sắc (>= 0.95) để không bỏ lỡ đỉnh cao nhất
                            if max_val >= 0.95:
                                break

                        if best_tpl_score >= self.threshold:
                            found_matches.append({
                                "status": label,
                                "confidence": float(best_tpl_score),
                                "box": (bx, by, bw, bh),
                                "template_name": tpl_info["name"],
                                "has_border_box": True
                            })

        # Sắp xếp toàn bộ match theo độ tin cậy từ cao xuống thấp
        found_matches.sort(key=lambda x: x["confidence"], reverse=True)

        # Xử lý thứ tự ưu tiên pháp lý:
        # 1. Nếu văn bản có đóng dấu "GIẢI MẬT" -> Trạng thái chính thức là GIẢI MẬT (dù có dấu MẬT gốc)
        giai_mat_match = next((m for m in found_matches if m["status"] == "GIẢI MẬT"), None)
        if giai_mat_match:
            secret_marks = [m["status"] for m in found_matches if m["status"] != "GIẢI MẬT"]
            if secret_marks:
                giai_mat_match["notes"] = f"Gốc: {secret_marks[0]} -> ĐÃ GIẢI MẬT"
            return giai_mat_match
        elif found_matches:
            # 2. Nếu không có Giải Mật, chọn độ mật cao nhất có độ tin cậy cao nhất
            for pref in ["TUYỆT MẬT", "TỐI MẬT", "MẬT"]:
                m = next((item for item in found_matches if item["status"] == pref), None)
                if m:
                    return m
            return found_matches[0]

        # -------------------------------------------------------------
        # NHÁNH 2: SLIDING WINDOW TOÀN VÙNG (Fallback khi không có hộp)
        # -------------------------------------------------------------
        if best_match["confidence"] < self.threshold:
            scales = np.linspace(0.8, 1.2, 5) # 5 tỉ lệ thô
            for label in ["MẬT", "TỐI MẬT", "TUYỆT MẬT", "GIẢI MẬT"]:
                for tpl_info in self.templates.get(label, []):
                    tpl = tpl_info["image"]
                    for scale in scales:
                        sw = int(tpl.shape[1] * scale)
                        sh = int(tpl.shape[0] * scale)
                        if sh >= h_roi or sw >= w_roi or sh < 12 or sw < 25:
                            continue

                        scaled_tpl = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA)
                        res = cv2.matchTemplate(roi_clean, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)

                        if max_val > best_match["confidence"]:
                            best_match["confidence"] = float(max_val)
                            best_match["status"] = label
                            best_match["box"] = (max_loc[0], max_loc[1], sw, sh)
                            best_match["template_name"] = tpl_info["name"]

                        if best_match["confidence"] >= 0.85:
                            break
                    if best_match["confidence"] >= 0.85:
                        break
                if best_match["confidence"] >= 0.85:
                    break

        if best_match["confidence"] < self.threshold:
            best_match["status"] = "BÌNH THƯỜNG"

        return best_match

    def _draw_unicode_text(self, cv_img, text, pos, font_size=20, text_color=(255, 255, 255), bg_color=(0, 0, 255)):
        """Vẽ chữ tiếng Việt có dấu chuẩn xác (không bị lỗi font M??T)"""
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        font_paths = [
            '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
            '/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf',
            '/Library/Fonts/Arial Bold.ttf',
            'C:\\Windows\\Fonts\\arialbd.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        x, y = pos
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 4
        badge_box = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]

        bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
        fg_rgb = (text_color[2], text_color[1], text_color[0])

        draw.rectangle(badge_box, fill=bg_rgb)
        draw.text((x, y), text, font=font, fill=fg_rgb)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def scan_file(self, file_path, visualize_dir=None):
        """Quét 1 tệp và đo lường latency chính xác"""
        t0 = time.perf_counter()
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            roi_gray, page_count, (orig_w, orig_h) = self.render_header_roi(file_path, 0)
            if roi_gray is None:
                return {"file": file_path, "status": "ERROR", "error": "Không thể render PDF", "latency_ms": 0.0}

            res = self.detect_stamp(roi_gray)
            res["page"] = 1
            res["file"] = file_path
            res["total_pages"] = page_count

            # Lưu ảnh highlight nếu có yêu cầu
            if visualize_dir and ("MẬT" in res["status"] or "GIẢI MẬT" in res["status"]):
                os.makedirs(visualize_dir, exist_ok=True)
                vis_img = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
                bx, by, bw, bh = res["box"]
                color = (0, 0, 255) if res["status"] != "GIẢI MẬT" else (0, 200, 0)
                cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), color, 3)
                label_text = f" {res['status']} ({res['confidence']:.2f}) "
                vis_img = self._draw_unicode_text(vis_img, label_text, (bx, max(10, by - 28)), font_size=18, bg_color=color)
                out_name = os.path.splitext(os.path.basename(file_path))[0] + "_highlight.png"
                cv2.imwrite(os.path.join(visualize_dir, out_name), vis_img)

            res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return res

        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            full_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if full_img is None:
                return {"file": file_path, "status": "ERROR", "error": "Không thể mở ảnh", "latency_ms": 0.0}
            
            h, w = full_img.shape[:2]
            roi_gray = full_img[0:int(h * 0.35), 0:int(w * 0.60)]
            res = self.detect_stamp(roi_gray)
            res["page"] = 1
            res["file"] = file_path
            res["total_pages"] = 1

            if visualize_dir and ("MẬT" in res["status"] or "GIẢI MẬT" in res["status"]):
                os.makedirs(visualize_dir, exist_ok=True)
                vis_img = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
                bx, by, bw, bh = res["box"]
                color = (0, 0, 255) if res["status"] != "GIẢI MẬT" else (0, 200, 0)
                cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), color, 3)
                label_text = f" {res['status']} ({res['confidence']:.2f}) "
                vis_img = self._draw_unicode_text(vis_img, label_text, (bx, max(10, by - 28)), font_size=18, bg_color=color)
                out_name = os.path.splitext(os.path.basename(file_path))[0] + "_highlight.png"
                cv2.imwrite(os.path.join(visualize_dir, out_name), vis_img)

            res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return res

        return {"file": file_path, "status": "SKIPPED", "latency_ms": 0.0}


# Hàm worker toàn cục để ProcessPoolExecutor có thể serialize
_global_detector = None

def _init_worker(template_dir, threshold, dpi):
    global _global_detector
    _global_detector = FastConfidentialDocDetector(template_dir=template_dir, threshold=threshold, dpi=dpi)

def _process_one_file(fpath):
    global _global_detector
    return _global_detector.scan_file(fpath)


def main():
    parser = argparse.ArgumentParser(description="Tool phát hiện tài liệu mật SIÊU TỐC ĐỘ (Fast Detector)")
    parser.add_argument("-i", "--input", required=True, help="Đường dẫn file PDF/Ảnh hoặc thư mục cần quét")
    parser.add_argument("-t", "--threshold", type=float, default=0.80, help="Ngưỡng tin cậy (mặc định: 0.80)")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Số tiến trình song song (mặc định: 1, đề xuất: 4 hoặc 8)")
    parser.add_argument("--templates", default="templates", help="Thư mục chứa templates mẫu")
    parser.add_argument("-v", "--visualize", default=None, help="Thư mục lưu ảnh xem trước")
    parser.add_argument("-j", "--json", default=None, help="Đường dẫn file xuất báo cáo JSON")
    args = parser.parse_args()

    # Thu thập danh sách files
    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "**", "*.pdf"), recursive=True) + \
                glob.glob(os.path.join(args.input, "**", "*.png"), recursive=True) + \
                glob.glob(os.path.join(args.input, "**", "*.jpg"), recursive=True)
    else:
        print(f"[!] Đường dẫn không tồn tại: {args.input}")
        sys.exit(1)

    print("\n" + "="*95)
    print(f" BẮT ĐẦU QUÉT TÀI LIỆU SIÊU TỐC (Tổng số: {len(files)} files | Workers: {args.workers})")
    print("="*95)
    print(f"{'TÊN FILE':<32} | {'KẾT QUẢ':<15} | {'ĐỘ TIN CẬY':<10} | {'KHUNG VIỀN':<11} | {'LATENCY (ms)':<12}")
    print("-"*95)

    all_results = []
    secret_count = 0
    start_wall_time = time.perf_counter()

    if args.workers > 1 and len(files) > 1:
        # Chạy đa tiến trình song song
        with ProcessPoolExecutor(max_workers=args.workers,
                                initializer=_init_worker,
                                initargs=(args.templates, args.threshold, 150)) as executor:
            all_results = list(executor.map(_process_one_file, sorted(files)))
    else:
        # Chạy đơn luồng
        detector = FastConfidentialDocDetector(template_dir=args.templates, threshold=args.threshold)
        for fpath in sorted(files):
            res = detector.scan_file(fpath, visualize_dir=args.visualize)
            all_results.append(res)

    total_latency_ms = 0.0
    for res in all_results:
        fname = os.path.basename(res.get("file", ""))
        status = res.get("status", "ERROR")
        conf = res.get("confidence", 0.0)
        has_box = "Có" if res.get("has_border_box") else "Không"
        latency = res.get("latency_ms", 0.0)
        total_latency_ms += latency

        if status in ["MẬT", "TỐI MẬT", "TUYỆT MẬT"]:
            secret_count += 1
            status_str = f"\033[91m{status}\033[0m" # In màu đỏ
        elif status == "GIẢI MẬT":
            status_str = f"\033[92m{status}\033[0m" # In màu xanh lá
        else:
            status_str = status

        print(f"{fname[:32]:<32} | {status_str:<24} | {conf:<10.3f} | {has_box:<11} | {latency:<10.1f} ms")

    total_wall_time = time.perf_counter() - start_wall_time
    throughput = (len(files) / total_wall_time) if total_wall_time > 0 else 0.0
    avg_latency = (total_latency_ms / len(files)) if files else 0.0

    print("="*95)
    print(f" HOÀN TẤT: Phát hiện {secret_count}/{len(files)} tài liệu có dấu độ mật.")
    print(f" HIỆU NĂNG TỔNG THỂ (WALL CLOCK TIME):")
    print(f"   - Tổng thời gian thực tế: {total_wall_time:.3f} giây")
    print(f"   - Thông lượng quét:       {throughput:.1f} trang/giây (~{avg_latency:.1f} ms/trang)")
    if args.visualize:
        print(f"   - Thư mục highlight:      {args.visualize}")
    print("="*95 + "\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as jf:
            json.dump(all_results, jf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
