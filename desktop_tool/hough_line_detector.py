"""
Module: xu_ly_soc/hough_line_detector.py
Mục đích: Phát hiện các đường sọc (vertical streaks) trong ảnh tài liệu bằng Hough Transform
          kết hợp thuật toán Slope-Aware Segment Stitching (ghép nối phân đoạn thích ứng độ xiên),
          xử lý triệt để các trường hợp sọc bị đứt đoạn do chữ/hình dấu đè, và sọc bị XIÊN / TRÔI NGANG
          lên tới 150-200 pixel do cơ chế cuốn giấy của máy scan.
"""

import sys
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi bảng mã cp1252
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đảm bảo đường dẫn import luôn thông suốt
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent if FILE_DIR.name in ["xu_ly_soc", "convert_pdf"] else FILE_DIR
for p_dir in [str(PROJECT_ROOT), str(FILE_DIR)]:
    if p_dir not in sys.path:
        sys.path.insert(0, p_dir)


def clean_path_string(raw_path: str) -> str:
    """Làm sạch chuỗi đường dẫn (xóa dấu ngoặc kép, nháy đơn do kéo thả trong Windows)."""
    if not raw_path:
        return ""
    return raw_path.strip().strip('"').strip("'")


def resolve_smart_path(input_path_str: Optional[str]) -> Optional[Path]:
    """Chuẩn hóa và tìm kiếm đường dẫn tuyệt đối chính xác cho file/thư mục ảnh.
    Tự động xử lý mọi trường hợp: tuyệt đối, tương đối từ CWD, tương đối từ project root,
    lỗi lặp folder (tool_tailieu/tool_tailieu), kéo thả chuột Windows (dấu ngoặc kép)."""
    if not input_path_str:
        return None

    cleaned = clean_path_string(input_path_str)
    file_dir = Path(__file__).resolve().parent
    project_root = file_dir.parent if file_dir.name in ["xu_ly_soc", "convert_pdf"] else file_dir
    workspace_root = project_root.parent
    cwd = Path.cwd()

    p = Path(cleaned)

    # 1. Đường dẫn tồn tại trực tiếp (tuyệt đối hoặc tương đối so với CWD)
    if p.exists():
        return p.resolve()

    # 2. Tương đối so với project root (tool_tailieu)
    if (project_root / cleaned).exists():
        return (project_root / cleaned).resolve()

    # 3. Tương đối so với workspace root (PythonProject)
    if (workspace_root / cleaned).exists():
        return (workspace_root / cleaned).resolve()

    # 4. Tương đối so với file_dir (xu_ly_soc)
    if (file_dir / cleaned).exists():
        return (file_dir / cleaned).resolve()

    # 5. Xử lý lỗi lặp "tool_tailieu/..." khi CWD đã ở trong tool_tailieu
    parts = list(p.parts)
    while parts and parts[0] == "tool_tailieu":
        parts.pop(0)
        stripped = Path(*parts) if parts else Path(".")
        if stripped.exists():
            return stripped.resolve()
        if (project_root / stripped).exists():
            return (project_root / stripped).resolve()
        if (cwd / stripped).exists():
            return (cwd / stripped).resolve()

    return None


def detect_vertical_streaks(
    image: np.ndarray,
    min_height_ratio: float = 0.95,     # Độ dài tối thiểu so với height (mặc định 95%)
    angle_tolerance: float = 5.0,        # Cho phép độ xiên tối đa lên tới 5.0 độ
    canny_thresh1: int = 30,             # Ngưỡng dưới Canny
    canny_thresh2: int = 100,            # Ngưỡng trên Canny
    base_segment_len: int = 80,          # Độ dài đoạn tối thiểu để Hough bắt từng khúc
    max_segment_gap: int = 150,          # Khoảng cách đứt tối đa của Hough nội bộ
    stitch_y_gap: int = 500,             # Khoảng cách tối đa theo trục y để nhảy qua vùng chữ lớn
    stitch_x_drift: int = 35,            # Độ lệch tối đa so với đường dự phóng theo hệ số góc
    merge_x_dist: int = 30               # Khoảng cách gộp 2 mép song song của cùng 1 vết sọc
) -> Tuple[np.ndarray, List[Dict]]:
    """Phát hiện các đường sọc (kể cả sọc thẳng đứng hoặc sọc xiên/nghiêng) chạy dọc tài liệu."""
    h, w = image.shape[:2]
    min_target_length = h * min_height_ratio

    # 1. Chuyển sang ảnh xám
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canvas = image.copy()
    else:
        gray = image.copy()
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. Trích xuất biên Canny (không dùng Gaussian Blur để giữ nguyên nét sọc 1px)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)

    # 3. Dùng HoughLinesP để tìm các phân đoạn sọc
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=base_segment_len,
        maxLineGap=max_segment_gap
    )

    if raw_lines is None:
        return canvas, []

    # 4. Lọc các đoạn thẳng có phương gần đứng (|góc - 90 độ| <= angle_tolerance)
    lines_array = np.array(raw_lines).reshape(-1, 4)
    segments = []

    for x1, y1, x2, y2 in lines_array:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        dev_from_vertical = 90.0 - angle

        if dev_from_vertical <= angle_tolerance:
            min_y = min(y1, y2)
            max_y = max(y1, y2)
            top_x = float(x1 if y1 < y2 else x2)
            bot_x = float(x2 if y1 < y2 else x1)
            segments.append({
                "y1": int(min_y),
                "y2": int(max_y),
                "top_x": top_x,
                "bot_x": bot_x,
                "avg_x": (top_x + bot_x) / 2.0,
                "length": float(math.hypot(dx, dy)),
                "dev": float(dev_from_vertical)
            })

    if not segments:
        return canvas, []

    # 5. THUẬT TOÁN GHÉP NỐI THÍCH ỨNG ĐỘ XIÊN (Slope-Aware Segment Stitching)
    segments.sort(key=lambda s: s["y1"])

    tracks = []
    for seg in segments:
        matched_track = None
        min_dist = float("inf")

        for trk in tracks:
            # Kiểm tra phân đoạn có kết nối theo trục y (gối đầu hoặc cách nhau <= stitch_y_gap)
            is_connected_y = (seg["y1"] <= trk["y2"] + stitch_y_gap) and (seg["y2"] >= trk["y1"] - stitch_y_gap)
            if is_connected_y:
                trk_dy = trk["y2"] - trk["y1"]
                # Nếu track đã có độ dài đáng kể (> 200px), tính độ dốc xiên (slope) để dự phóng vị trí x
                if trk_dy > 200:
                    slope = (trk["bot_x"] - trk["top_x"]) / trk_dy
                    pred_x = trk["bot_x"] + slope * (seg["y1"] - trk["y2"])
                    dx = abs(seg["top_x"] - pred_x)
                else:
                    dx = abs(seg["top_x"] - trk["bot_x"])

                if dx <= stitch_x_drift:
                    if dx < min_dist:
                        min_dist = dx
                        matched_track = trk

        if matched_track is not None:
            if seg["y1"] < matched_track["y1"]:
                matched_track["top_x"] = seg["top_x"]
                matched_track["y1"] = seg["y1"]
            if seg["y2"] > matched_track["y2"]:
                matched_track["bot_x"] = seg["bot_x"]
                matched_track["y2"] = seg["y2"]
            matched_track["count"] += 1
            matched_track["segments"].append(seg)
        else:
            tracks.append({
                "y1": seg["y1"],
                "y2": seg["y2"],
                "top_x": seg["top_x"],
                "bot_x": seg["bot_x"],
                "count": 1,
                "segments": [seg]
            })

    # 6. Gộp các track song song cùng một vết sọc trước khi lọc độ dài
    tracks.sort(key=lambda t: t["top_x"])
    merged_tracks = []
    for trk in tracks:
        if not merged_tracks:
            merged_tracks.append(trk)
        else:
            prev = merged_tracks[-1]
            dx_top = abs(trk["top_x"] - prev["top_x"])
            dx_bot = abs(trk["bot_x"] - prev["bot_x"])
            is_overlap_y = (trk["y1"] <= prev["y2"] + stitch_y_gap) and (trk["y2"] >= prev["y1"] - stitch_y_gap)

            if (dx_top <= merge_x_dist or dx_bot <= merge_x_dist) and is_overlap_y:
                if trk["y1"] < prev["y1"]:
                    prev["top_x"] = trk["top_x"]
                    prev["y1"] = trk["y1"]
                if trk["y2"] > prev["y2"]:
                    prev["bot_x"] = trk["bot_x"]
                    prev["y2"] = trk["y2"]
                prev["count"] += trk["count"]
                prev["segments"].extend(trk["segments"])
            else:
                merged_tracks.append(trk)

    # 7. Lọc theo ngưỡng độ dài tối thiểu
    final_streaks = []
    for idx, trk in enumerate(merged_tracks, start=1):
        total_len = float(trk["y2"] - trk["y1"])
        if total_len < min_target_length:
            continue

        ratio = total_len / h
        pt1 = (int(trk["top_x"]), int(trk["y1"]))
        pt2 = (int(trk["bot_x"]), int(trk["y2"]))
        avg_x = (trk["top_x"] + trk["bot_x"]) / 2.0
        delta_x = abs(trk["bot_x"] - trk["top_x"])
        tilt_angle = math.degrees(math.atan2(delta_x, total_len))

        final_streaks.append({
            "id": idx,
            "pt1": pt1,
            "pt2": pt2,
            "top_x": trk["top_x"],
            "bot_x": trk["bot_x"],
            "avg_x": avg_x,
            "delta_x": delta_x,
            "tilt_deg": tilt_angle,
            "length": total_len,
            "ratio": ratio,
            "num_segments": len(trk["segments"])
        })

    for idx, s in enumerate(final_streaks, start=1):
        s["id"] = idx

    # 8. Vẽ trực quan vệt sọc lên ảnh
    canvas = draw_streaks_visualization(canvas, final_streaks)
    return canvas, final_streaks


def draw_streaks_visualization(
    image: np.ndarray,
    streaks: List[Dict],
    streak_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 3
) -> np.ndarray:
    """Vẽ vệt sọc (thẳng hoặc xiên) và nhãn thông số chi tiết lên ảnh."""
    out = image.copy()
    h, w = out.shape[:2]

    for s in streaks:
        pt1 = s["pt1"]
        pt2 = s["pt2"]
        avg_x = int(s["avg_x"])
        length = s["length"]
        pct = s["ratio"] * 100
        streak_id = s["id"]
        tilt = s.get("tilt_deg", 0.0)
        dx = s.get("delta_x", 0.0)
        num_segs = s.get("num_segments", 1)

        # Vẽ đường sọc màu đỏ kéo dài từ pt1 đến pt2 (chuẩn xác theo độ xiên)
        cv2.line(out, pt1, pt2, streak_color, thickness, cv2.LINE_AA)

        # Vẽ 2 điểm mút màu xanh lá
        cv2.circle(out, pt1, thickness + 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out, pt2, thickness + 4, (0, 255, 0), -1, cv2.LINE_AA)

        # Nhãn thông tin
        if dx > 15:
            label = f"Soc #{streak_id}: x={s['top_x']:.0f}->{s['bot_x']:.0f} (xien {tilt:.1f}*, troi {dx:.0f}px) | L={length:.0f}px ({pct:.1f}% H)"
        else:
            label = f"Soc #{streak_id}: x={avg_x} | L={length:.0f}px ({pct:.1f}% H) | {num_segs} khuc"

        font_scale = max(0.5, min(0.9, h / 3200.0))
        text_thick = 2 if font_scale > 0.7 else 1
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)

        tx = max(10, min(avg_x + 10, w - tw - 15))
        ty = max(th + 10, min(int(pt1[1] + (pt2[1] - pt1[1]) * 0.35), h - 15))

        # Khung nền đen viền vàng
        cv2.rectangle(out, (tx - 4, ty - th - 6), (tx + tw + 4, ty + baseline + 6), (0, 0, 0), -1)
        cv2.rectangle(out, (tx - 4, ty - th - 6), (tx + tw + 4, ty + baseline + 6), (0, 255, 255), 1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA)

    return out


def process_image(
    image_path: str,
    output_path: Optional[str] = None,
    min_height_ratio: float = 0.90,
    angle_tolerance: float = 5.0
) -> List[Dict]:
    """Xử lý phát hiện sọc cho một file ảnh đơn lẻ."""
    resolved_img_path = resolve_smart_path(image_path)
    if resolved_img_path is None or not resolved_img_path.exists():
        print(f"[LỖI] Không tìm thấy file ảnh: {image_path}")
        return []

    img = cv2.imread(str(resolved_img_path))
    if img is None:
        print(f"[LỖI] Không thể đọc ảnh: {resolved_img_path}")
        return []

    h, w = img.shape[:2]
    threshold_px = h * min_height_ratio
    result_img, streaks = detect_vertical_streaks(
        img,
        min_height_ratio=min_height_ratio,
        angle_tolerance=angle_tolerance
    )

    print(f"\n--- {resolved_img_path.name} ({w}x{h} px, Ngưỡng sọc >= {threshold_px:.0f}px - {min_height_ratio*100:.0f}% H) ---")
    if streaks:
        print(f"[CẢNH BÁO] Phát hiện {len(streaks)} vết sọc:")
        for s in streaks:
            dx_str = f", trôi ngang={s['delta_x']:.0f}px (xiên {s['tilt_deg']:.1f}°)" if s['delta_x'] > 15 else ""
            print(f"  - Sọc #{s['id']}: Cột x={s['top_x']:.0f}->{s['bot_x']:.0f}{dx_str} | Dài={s['length']:.0f}px ({s['ratio']*100:.1f}% H) | y: {s['pt1'][1]} -> {s['pt2'][1]} | Ghép từ {s['num_segments']} khúc")
    else:
        print("[SẠCH] Không phát hiện vết sọc nào đạt ngưỡng.")

    # Xác định đường dẫn file kết quả
    if output_path is None:
        file_dir = Path(__file__).resolve().parent
        project_root = file_dir.parent if file_dir.name in ["xu_ly_soc", "convert_pdf"] else file_dir
        reports_dir = project_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_file = reports_dir / f"{resolved_img_path.stem}_detected.png"
    else:
        out_file = Path(clean_path_string(output_path)).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_file), result_img)
    print(f"-> Đã lưu ảnh kết quả: {out_file.resolve()}")

    return streaks


def process_folder(
    folder_path: str,
    output_dir: Optional[str] = None,
    min_height_ratio: float = 0.95,
    angle_tolerance: float = 5.0
):
    """Xử lý phát hiện sọc cho tất cả các file PNG/JPG trong một thư mục."""
    resolved_folder = resolve_smart_path(folder_path)
    if resolved_folder is None or not resolved_folder.exists():
        print(f"[LỖI] Không tìm thấy thư mục: {folder_path}")
        return

    png_files = sorted(list(resolved_folder.glob("*.png")) + list(resolved_folder.glob("*.jpg")) + list(resolved_folder.glob("*.jpeg")))
    if not png_files:
        print(f"[THÔNG BÁO] Không có ảnh PNG/JPG nào trong: {resolved_folder}")
        return

    if output_dir:
        out_folder = Path(clean_path_string(output_dir)).resolve()
    else:
        file_dir = Path(__file__).resolve().parent
        project_root = file_dir.parent if file_dir.name in ["xu_ly_soc", "convert_pdf"] else file_dir
        out_folder = project_root / "reports" / f"{resolved_folder.name}_detected"

    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"[BẮT ĐẦU] Quét {len(png_files)} file ảnh trong '{resolved_folder}' (Ngưỡng >= {min_height_ratio*100:.0f}% H)...")
    streaked_pages = 0

    for idx, f in enumerate(png_files, 1):
        out_file = out_folder / f"{f.stem}_detected.png"
        streaks = process_image(
            str(f),
            output_path=str(out_file),
            min_height_ratio=min_height_ratio,
            angle_tolerance=angle_tolerance
        )
        if streaks:
            streaked_pages += 1

    print(f"\n=======================================================")
    print(f"TỔNG KẾT: {streaked_pages}/{len(png_files)} trang bị phát hiện có sọc.")
    print(f"Thư mục lưu kết quả: {out_folder.resolve()}")
    print(f"=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phát hiện sọc (kể cả sọc xiên/nghiêng) tài liệu có độ dài >= 90-95% chiều cao ảnh."
    )
    parser.add_argument("--image", "-i", type=str, default=None, help="Đường dẫn tới file ảnh cần kiểm tra")
    parser.add_argument("--dir", "-d", type=str, default=None, help="Đường dẫn thư mục chứa các file ảnh cần quét hàng loạt")
    parser.add_argument("--output", "-o", type=str, default=None, help="Đường dẫn lưu file kết quả hoặc thư mục kết quả")
    parser.add_argument("--min_ratio", "-r", type=float, default=0.95, help="Tỉ lệ độ dài tối thiểu so với height (mặc định 0.95 = 95%%)")
    parser.add_argument("--angle_tol", "-a", type=float, default=5.0, help="Dung sai góc lệch thẳng đứng (độ, mặc định 5.0 để bắt được cả sọc xiên)")

    args = parser.parse_args()

    if args.image:
        process_image(
            image_path=args.image,
            output_path=args.output,
            min_height_ratio=args.min_ratio,
            angle_tolerance=args.angle_tol
        )
    elif args.dir:
        process_folder(
            folder_path=args.dir,
            output_dir=args.output,
            min_height_ratio=args.min_ratio,
            angle_tolerance=args.angle_tol
        )
    else:
        print("[HƯỚNG DẪN SỬ DỤNG HOUGH LINE DETECTOR]")
        print("1. Chạy cho 1 file ảnh:")
        print("   python xu_ly_soc/hough_line_detector.py --image path/to/image.png --min_ratio 0.90")
        print("\n2. Quét hàng loạt ảnh trong 1 thư mục:")
        print("   python xu_ly_soc/hough_line_detector.py --dir path/to/folder --min_ratio 0.90")

