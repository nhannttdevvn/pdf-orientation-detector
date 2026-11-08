"""
Phase 3.8 — Page Reconstruction Preview & User-Approved Export (`src/pdf_inspector/processing/page_reconstructor.py`).

Tách biệt hoàn toàn giữa Detection và Reconstruction/Correction:
1. Hiển thị lớp phủ đường cắt / vùng ranh giới nghi vấn (`render_boundary_overlay`).
2. Hỗ trợ người dùng điều chỉnh tọa độ đường cắt và tự động neo vào rãnh trắng an toàn gần nhất
   để KHÔNG cắt xuyên qua ký tự chữ (`find_safe_cut_line_px`).
3. Xem trước hai phần ảnh sau khi tách (`preview_split_single_page`).
4. Xem trước trang phục hồi sau khi ghép với trang liền kề hoặc khử vùng chồng lặp (`preview_reconstruct_with_adjacent_page`).
5. Chỉ thực hiện xuất file PDF tái cấu trúc (`export_user_approved_reconstruction_pdf`) sau khi người dùng xác nhận rõ ràng (`user_confirmed=True`),
   bảo toàn 100% file PDF gốc và ghi lại truy vết nguồn gốc (`source_regions` provenance).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore[no-redef]

from ai_engine.core.config import RECONSTRUCTION_PROVENANCE_PATH
from ai_engine.detectors.boundary.layout_analyzer import find_safe_cut_line_px
from ai_engine.preprocessing.pdf_loader import PDFLoader


def render_boundary_overlay(
    image: np.ndarray,
    boundary_defects: List[Any],
    custom_cut_norm: Optional[float] = None,
) -> np.ndarray:
    """
    Vẽ lớp phủ trực quan các ứng viên ranh giới trang (`MIXED_PAGE`, `PAGE_BOUNDARY_SUSPECTED`),
    vùng chồng lặp (`CONTENT_OVERLAP`), vùng mép bị cắt (`CONTENT_TRUNCATED`)
    và đường cắt đề xuất / chỉnh tay (`custom_cut_norm`) lên bản sao RGB của `image`.
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào cho render_boundary_overlay bị rỗng.")

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        canvas = image.copy()

    h, w = canvas.shape[:2]
    overlay = canvas.copy()

    for item in boundary_defects:
        d = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        dtype = str(d.get("defect_type", "BOUNDARY"))
        status = str(d.get("status", "SUSPECTED"))
        score = d.get("suspicion_score") or d.get("confidence") or 0.0

        # Chọn màu: Đỏ cho CONFIRMED, Cam cho SUSPECTED, Tím cho CONTENT_OVERLAP
        if dtype == "CONTENT_OVERLAP":
            color = (155, 45, 220)
        elif status == "CONFIRMED":
            color = (225, 35, 35)
        else:
            color = (240, 140, 20)

        bbox = d.get("bbox")
        if bbox and len(bbox) == 4:
            x0 = max(0, min(w - 1, int(bbox[0])))
            y0 = max(0, min(h - 1, int(bbox[1])))
            x1 = max(x0 + 1, min(w, int(bbox[2])))
            y1 = max(y0 + 1, min(h, int(bbox[3])))
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)

        for cand in d.get("boundary_candidates", []):
            orient = cand.get("orientation", "HORIZONTAL")
            pos_norm = float(cand.get("safe_cut_line_norm") or cand.get("position", 0.5))
            if orient == "HORIZONTAL":
                cy = max(0, min(h - 1, int(round(pos_norm * h))))
                cv2.line(canvas, (0, cy), (w - 1, cy), color, 2, lineType=cv2.LINE_AA)
                label = f"{dtype} y={pos_norm:.2f} ({float(score):.2f})"
                ly = max(22, cy - 8)
                cv2.putText(
                    canvas,
                    label,
                    (12, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            else:
                cx = max(0, min(w - 1, int(round(pos_norm * w))))
                cv2.line(canvas, (cx, 0), (cx, h - 1), color, 2, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0)

    # Nếu người dùng kéo thanh trượt chỉnh tọa độ đường cắt (`custom_cut_norm`), vẽ đường cắt xanh lá nổi bật
    if custom_cut_norm is not None:
        cut_y = max(0, min(h - 1, int(round(float(custom_cut_norm) * h))))
        cv2.line(
            blended,
            (0, cut_y),
            (w - 1, cut_y),
            (20, 185, 65),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            blended,
            f"Proposed Cut Line y={float(custom_cut_norm):.3f} ({cut_y}px)",
            (14, min(h - 10, cut_y + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (15, 150, 50),
            2,
            cv2.LINE_AA,
        )

    return blended


def preview_split_single_page(
    page_image: np.ndarray,
    cut_position_norm: float,
    orientation: str = "HORIZONTAL",
    enforce_safe_cut: bool = True,
    source_page_number: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Tách thử một trang ảnh (`page_image`) tại vị trí `cut_position_norm` (`[0.05 .. 0.95]`).
    Nếu `enforce_safe_cut=True`, tự động dịch nhẹ đường cắt về hàng trắng gần nhất để KHÔNG cắt xuyên qua ký tự chữ.
    Trả về `(part_a_image, part_b_image, split_info)`.
    """
    if page_image is None or page_image.size == 0:
        raise ValueError("Ảnh trang đưa vào preview_split_single_page bị rỗng.")

    h, w = page_image.shape[:2]
    gray = (
        cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        if page_image.ndim == 3
        else page_image
    )
    _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    norm_clamped = min(0.95, max(0.05, float(cut_position_norm)))
    if orientation == "HORIZONTAL":
        raw_px = int(round(norm_clamped * h))
        if enforce_safe_cut:
            safe_px, crosses_text = find_safe_cut_line_px(
                otsu_bin, raw_px, orientation="HORIZONTAL", search_radius_px=max(18, int(h * 0.03))
            )
        else:
            safe_px = raw_px
            crosses_text = bool(np.sum(otsu_bin[safe_px, :] > 0) > max(2, int(w * 0.003)))

        safe_px = max(10, min(h - 10, safe_px))
        part_a = page_image[:safe_px, :].copy()
        part_b = page_image[safe_px:, :].copy()
        applied_norm = round(safe_px / float(h), 4)
        source_regions = [
            {
                "part": "part_A_top",
                "source_page": source_page_number,
                "bbox_px": [0, 0, w, safe_px],
                "bbox_norm": [0.0, 0.0, 1.0, applied_norm],
            },
            {
                "part": "part_B_bottom",
                "source_page": source_page_number,
                "bbox_px": [0, safe_px, w, h],
                "bbox_norm": [0.0, applied_norm, 1.0, 1.0],
            },
        ]
    else:
        raw_px = int(round(norm_clamped * w))
        if enforce_safe_cut:
            safe_px, crosses_text = find_safe_cut_line_px(
                otsu_bin, raw_px, orientation="VERTICAL", search_radius_px=max(18, int(w * 0.03))
            )
        else:
            safe_px = raw_px
            crosses_text = bool(np.sum(otsu_bin[:, safe_px] > 0) > max(2, int(h * 0.003)))

        safe_px = max(10, min(w - 10, safe_px))
        part_a = page_image[:, :safe_px].copy()
        part_b = page_image[:, safe_px:].copy()
        applied_norm = round(safe_px / float(w), 4)
        source_regions = [
            {
                "part": "part_A_left",
                "source_page": source_page_number,
                "bbox_px": [0, 0, safe_px, h],
                "bbox_norm": [0.0, 0.0, applied_norm, 1.0],
            },
            {
                "part": "part_B_right",
                "source_page": source_page_number,
                "bbox_px": [safe_px, 0, w, h],
                "bbox_norm": [applied_norm, 0.0, 1.0, 1.0],
            },
        ]

    info = {
        "source_page": source_page_number,
        "orientation": orientation,
        "requested_cut_norm": round(norm_clamped, 4),
        "applied_safe_cut_norm": applied_norm,
        "applied_safe_cut_px": safe_px,
        "crosses_text": crosses_text,
        "safe_to_cut": not crosses_text,
        "source_regions": source_regions,
    }
    return part_a, part_b, info


def _trim_excess_vertical_blank(
    img: np.ndarray,
    pad_px: int = 24,
) -> Tuple[np.ndarray, int, int]:
    """Cắt bỏ khoảng trắng thừa quá lớn ở đầu/cuối mảnh ghép để ghép liền mạch tự nhiên."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    _, bin_m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    margin_x = max(6, int(w * 0.03))
    row_act = np.mean(bin_m[:, margin_x : w - margin_x] > 0, axis=1)
    nz = np.flatnonzero(row_act >= 0.002)
    if nz.size == 0:
        return img.copy(), 0, h
    y0 = max(0, int(nz[0]) - pad_px)
    y1 = min(h, int(nz[-1]) + pad_px)
    return img[y0:y1, :].copy(), y0, y1


def preview_reconstruct_with_adjacent_page(
    primary_image: np.ndarray,
    adjacent_image: np.ndarray,
    action_type: str = "SPLIT_AND_MERGE_SPILLOVER",
    cut_position_norm: float = 0.70,
    overlap_strip_norm: float = 0.0,
    primary_page_num: int = 1,
    adjacent_page_num: int = 2,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Xem trước kết quả khôi phục giữa `primary_image` (trang N) và `adjacent_image` (trang liên quan):
    - `SPLIT_AND_MERGE_SPILLOVER`: Tách phần tràn trên trang N tại `cut_position_norm` và ghép trả về trang liên quan.
    - `MERGE_SPLIT_PAGES`: Ghép hai nửa trang bị chia đôi (`SPLIT_PAGE`) thành một trang hoàn chỉnh.
    - `REMOVE_OVERLAP_AND_MERGE`: Đối chiếu vùng chồng lặp (`CONTENT_OVERLAP`) và khử phần lặp trước khi ghép/xuất.
    Trả về `(restored_page_1, restored_page_2_or_None, provenance_metadata)`.
    """
    if action_type == "REQUIRES_RESCAN":
        raise ValueError(
            "Trường hợp MISSING_CONTENT không đủ dữ liệu để khôi phục tự động. Vui lòng scan lại trang gốc."
        )

    # Chuẩn hóa kênh màu và chiều rộng giữa hai ảnh
    if primary_image.ndim == 2 and adjacent_image.ndim == 3:
        primary_image = cv2.cvtColor(primary_image, cv2.COLOR_GRAY2RGB)
    elif primary_image.ndim == 3 and adjacent_image.ndim == 2:
        adjacent_image = cv2.cvtColor(adjacent_image, cv2.COLOR_GRAY2RGB)

    h1, w1 = primary_image.shape[:2]
    h2, w2 = adjacent_image.shape[:2]
    if w2 != w1:
        new_h2 = max(16, int(round(h2 * (w1 / float(w2)))))
        adjacent_image = cv2.resize(adjacent_image, (w1, new_h2), interpolation=cv2.INTER_AREA)
        h2, w2 = adjacent_image.shape[:2]

    bg_val = int(np.percentile(primary_image, 85))

    if action_type == "SPLIT_AND_MERGE_SPILLOVER":
        part_a, part_b, split_info = preview_split_single_page(
            primary_image,
            cut_position_norm=cut_position_norm,
            orientation="HORIZONTAL",
            enforce_safe_cut=True,
            source_page_number=primary_page_num,
        )
        # Bù khoảng trắng phía dưới cho part_a để giữ nguyên kích thước trang chuẩn (w1, h1)
        restored_p1 = np.full_like(primary_image, bg_val)
        restored_p1[: part_a.shape[0], :] = part_a

        trimmed_b, by0, by1 = _trim_excess_vertical_blank(part_b, pad_px=16)
        trimmed_adj, ay0, ay1 = _trim_excess_vertical_blank(adjacent_image, pad_px=20)

        if adjacent_page_num > primary_page_num:
            # Phần dưới của trang N là phần đầu bị tràn của trang N+1 -> ghép lên đầu trang N+1
            merged_p2 = np.concatenate([trimmed_b, trimmed_adj], axis=0)
        else:
            # Phần trên của trang N là phần cuối của trang N-1 -> ghép vào cuối trang N-1
            trimmed_a, _, _ = _trim_excess_vertical_blank(part_a, pad_px=16)
            merged_p2 = np.concatenate([trimmed_adj, trimmed_a], axis=0)
            restored_p1 = np.full_like(primary_image, bg_val)
            restored_p1[: part_b.shape[0], :] = part_b

        provenance = {
            "action_type": action_type,
            "primary_page": primary_page_num,
            "adjacent_page": adjacent_page_num,
            "applied_safe_cut_norm": split_info["applied_safe_cut_norm"],
            "safe_to_cut": split_info["safe_to_cut"],
            "source_regions": split_info["source_regions"]
            + [
                {
                    "part": "adjacent_page_content",
                    "source_page": adjacent_page_num,
                    "bbox_px": [0, ay0, w2, ay1],
                }
            ],
        }
        return restored_p1, merged_p2, provenance

    if action_type == "REMOVE_OVERLAP_AND_MERGE":
        # Khử dải chồng lặp ở đầu trang thứ hai
        ov_norm = min(0.45, max(0.05, float(overlap_strip_norm or 0.15)))
        ov_px = int(round(ov_norm * h2))
        dedup_second = adjacent_image[ov_px:, :].copy()
        restored_second = np.full_like(adjacent_image, bg_val)
        restored_second[: dedup_second.shape[0], :] = dedup_second

        provenance = {
            "action_type": action_type,
            "primary_page": primary_page_num,
            "adjacent_page": adjacent_page_num,
            "removed_overlap_norm": round(ov_norm, 4),
            "removed_overlap_px": ov_px,
            "safe_to_cut": True,
            "source_regions": [
                {
                    "part": "primary_full",
                    "source_page": primary_page_num,
                    "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                },
                {
                    "part": "adjacent_deduplicated",
                    "source_page": adjacent_page_num,
                    "bbox_norm": [0.0, round(ov_norm, 4), 1.0, 1.0],
                },
            ],
        }
        return primary_image.copy(), restored_second, provenance

    # Mặc định: MERGE_SPLIT_PAGES (ghép 2 phần trang bị chia đôi thành 1 trang hoàn chỉnh)
    first_img = primary_image if primary_page_num <= adjacent_page_num else adjacent_image
    second_img = adjacent_image if primary_page_num <= adjacent_page_num else primary_image
    first_pno = min(primary_page_num, adjacent_page_num)
    second_pno = max(primary_page_num, adjacent_page_num)

    trim_1, y1_0, y1_1 = _trim_excess_vertical_blank(first_img, pad_px=24)
    trim_2, y2_0, y2_1 = _trim_excess_vertical_blank(second_img, pad_px=24)
    merged_single = np.concatenate([trim_1, trim_2], axis=0)

    # Đặt vào khung chuẩn tối thiểu cao bằng h1
    target_h = max(h1, merged_single.shape[0] + 40)
    if merged_single.ndim == 3:
        canvas_out = np.full((target_h, w1, merged_single.shape[2]), bg_val, dtype=np.uint8)
    else:
        canvas_out = np.full((target_h, w1), bg_val, dtype=np.uint8)
    y_offset = max(0, min(20, (target_h - merged_single.shape[0]) // 2))
    canvas_out[y_offset : y_offset + merged_single.shape[0], :] = merged_single

    provenance = {
        "action_type": "MERGE_SPLIT_PAGES",
        "primary_page": primary_page_num,
        "adjacent_page": adjacent_page_num,
        "safe_to_cut": True,
        "source_regions": [
            {
                "part": "upper_half",
                "source_page": first_pno,
                "bbox_px": [0, y1_0, w1, y1_1],
            },
            {
                "part": "lower_half",
                "source_page": second_pno,
                "bbox_px": [0, y2_0, w1, y2_1],
            },
        ],
    }
    return canvas_out, None, provenance


def _append_numpy_image_page(
    doc: fitz.Document,
    img: np.ndarray,
    dpi: int = 160,
) -> None:
    """Nhúng một mảng ảnh `img` thành một trang mới trong tài liệu `fitz.Document`."""
    h, w = img.shape[:2]
    width_pt = w * 72.0 / float(max(72, dpi))
    height_pt = h * 72.0 / float(max(72, dpi))
    to_encode = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    ok, png_buf = cv2.imencode(".png", to_encode)
    if not ok:
        raise RuntimeError("Không thể mã hóa PNG khi xuất trang PDF tái cấu trúc.")
    page = doc.new_page(width=width_pt, height=height_pt)
    page.insert_image(fitz.Rect(0, 0, width_pt, height_pt), stream=png_buf.tobytes())


def export_user_approved_reconstruction_pdf(
    input_pdf_path: Union[str, Path],
    output_pdf_path: Union[str, Path],
    approved_actions: Optional[List[Dict[str, Any]]] = None,
    provenance_output_path: Optional[Union[str, Path]] = None,
    dpi: int = 160,
    *,
    approved_operations: Optional[List[Dict[str, Any]]] = None,
    user_confirmed: Optional[bool] = None,
    provenance_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Chỉ thực hiện xuất bản PDF đã tách/ghép (`output_pdf_path`) sau khi người dùng xác nhận rõ ràng
    (`user_confirmed=True` ở tham số hàm hoặc trên từng hành động trong `approved_actions`/`approved_operations`).
    - Tuyệt đối KHÔNG ghi đè lên `input_pdf_path`.
    - Từ chối thực hiện nếu `user_confirmed` không phải `True` hoặc nếu hành động thuộc loại `REQUIRES_RESCAN`.
    - Lưu đầy đủ thông tin truy vết nguồn (`source_regions`) ra file JSON.
    """
    in_path = Path(input_pdf_path).expanduser().resolve()
    out_path = Path(output_pdf_path).expanduser().resolve()
    if in_path == out_path:
        raise ValueError("Không được phép ghi đè trực tiếp lên file PDF gốc.")

    actions = approved_actions if approved_actions is not None else approved_operations
    if not actions:
        raise ValueError("Danh sách hành động tái cấu trúc được phê duyệt đang rỗng.")

    if user_confirmed is False:
        raise PermissionError(
            "Bắt buộc có xác nhận của người dùng (user_confirmed=True) trước khi xuất PDF tái cấu trúc."
        )

    for act in actions:
        effective_confirmed = bool(act.get("user_confirmed", bool(user_confirmed)))
        if not effective_confirmed:
            raise PermissionError(
                f"Hành động trên trang {act.get('page_number')} chưa được người dùng xác nhận (user_confirmed=True)."
            )
        raw_act = str(act.get("action_type") or act.get("action") or "SPLIT_SINGLE_PAGE")
        if raw_act in ("REQUIRES_RESCAN", "MISSING_CONTENT"):
            raise ValueError(
                f"Trang {act.get('page_number')} bị thiếu dữ liệu (MISSING_CONTENT), cần scan lại chứ không thể tự tạo nội dung."
            )

    action_by_page: Dict[int, Dict[str, Any]] = {
        int(a["page_number"]): a for a in actions
    }
    consumed_pages = set()
    provenance_records: List[Dict[str, Any]] = []
    reconstructed_pages: List[Dict[str, Any]] = []
    total_pages = 0

    out_doc = fitz.open()
    try:
        with PDFLoader(in_path) as loader:
            total_pages = loader.page_count
            for pno in range(1, total_pages + 1):
                if pno in consumed_pages:
                    continue

                if pno not in action_by_page:
                    # Giữ nguyên trang gốc không can thiệp
                    out_doc.insert_pdf(loader.document, from_page=pno - 1, to_page=pno - 1)
                    src_reg = [{"source_page": pno, "bbox_norm": [0.0, 0.0, 1.0, 1.0]}]
                    provenance_records.append(
                        {
                            "output_page_source": pno,
                            "action_type": "KEEP_ORIGINAL",
                            "source_regions": src_reg,
                        }
                    )
                    reconstructed_pages.append(
                        {
                            "output_page_number": len(reconstructed_pages) + 1,
                            "operation": "KEEP_ORIGINAL",
                            "source_regions": src_reg,
                        }
                    )
                    continue

                act = action_by_page[pno]
                raw_type = str(act.get("action_type") or act.get("action") or "SPLIT_SINGLE_PAGE")
                if raw_type in ("MERGE_WITH_NEXT", "MERGE_SPLIT_PAGES", "SPLIT_PAGE"):
                    act_type = "MERGE_SPLIT_PAGES"
                elif raw_type in (
                    "DEDUPLICATE_OVERLAP_WITH_NEXT",
                    "REMOVE_OVERLAP_AND_MERGE",
                    "CONTENT_OVERLAP",
                ):
                    act_type = "REMOVE_OVERLAP_AND_MERGE"
                elif raw_type == "SPLIT_AND_MERGE_SPILLOVER":
                    act_type = "SPLIT_AND_MERGE_SPILLOVER"
                else:
                    act_type = "SPLIT_SINGLE_PAGE"

                cut_norm = float(
                    act.get("cut_position_norm")
                    if act.get("cut_position_norm") is not None
                    else act.get("cut_norm", 0.5)
                )
                orient = str(act.get("orientation", "HORIZONTAL"))
                rel_page = act.get("related_page") or act.get("second_page_number")
                ov_norm = float(
                    act.get("overlap_strip_norm")
                    if act.get("overlap_strip_norm") is not None
                    else act.get("overlap_norm", 0.0)
                )

                ctx_primary = loader.build_page_context(pno, dpi=dpi, grayscale=False)
                img_primary = ctx_primary.original_image.copy()
                ctx_primary.release()

                if act_type == "SPLIT_SINGLE_PAGE" or rel_page is None:
                    part_a, part_b, split_info = preview_split_single_page(
                        img_primary,
                        cut_position_norm=cut_norm,
                        orientation=orient,
                        enforce_safe_cut=True,
                        source_page_number=pno,
                    )
                    _append_numpy_image_page(out_doc, part_a, dpi=dpi)
                    _append_numpy_image_page(out_doc, part_b, dpi=dpi)
                    consumed_pages.add(pno)
                    provenance_records.append(split_info)
                    s_regs = split_info.get("source_regions", [])
                    reconstructed_pages.append(
                        {
                            "output_page_number": len(reconstructed_pages) + 1,
                            "operation": "SPLIT_SINGLE_PAGE_TOP",
                            "source_regions": [s_regs[0]] if len(s_regs) >= 1 else [],
                        }
                    )
                    reconstructed_pages.append(
                        {
                            "output_page_number": len(reconstructed_pages) + 1,
                            "operation": "SPLIT_SINGLE_PAGE_BOTTOM",
                            "source_regions": [s_regs[1]] if len(s_regs) >= 2 else [],
                        }
                    )
                else:
                    adj_pno = int(rel_page)
                    ctx_adj = loader.build_page_context(adj_pno, dpi=dpi, grayscale=False)
                    img_adj = ctx_adj.original_image.copy()
                    ctx_adj.release()

                    rest_1, rest_2, prov = preview_reconstruct_with_adjacent_page(
                        primary_image=img_primary,
                        adjacent_image=img_adj,
                        action_type=act_type,
                        cut_position_norm=cut_norm,
                        overlap_strip_norm=ov_norm,
                        primary_page_num=pno,
                        adjacent_page_num=adj_pno,
                    )
                    _append_numpy_image_page(out_doc, rest_1, dpi=dpi)
                    reconstructed_pages.append(
                        {
                            "output_page_number": len(reconstructed_pages) + 1,
                            "operation": act_type,
                            "source_regions": prov.get("source_regions", []),
                        }
                    )
                    if rest_2 is not None:
                        _append_numpy_image_page(out_doc, rest_2, dpi=dpi)
                        reconstructed_pages.append(
                            {
                                "output_page_number": len(reconstructed_pages) + 1,
                                "operation": f"{act_type}_COMPANION",
                                "source_regions": prov.get("source_regions", []),
                            }
                        )
                    consumed_pages.add(pno)
                    consumed_pages.add(adj_pno)
                    provenance_records.append(prov)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_doc.save(str(out_path))
        new_page_count = len(out_doc)
    finally:
        out_doc.close()

    eff_prov_path = provenance_output_path or provenance_json_path
    prov_path = (
        Path(eff_prov_path).expanduser().resolve()
        if eff_prov_path
        else RECONSTRUCTION_PROVENANCE_PATH.resolve()
    )
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_confirmed": True,
        "input_pdf_path": str(in_path),
        "output_pdf_path": str(out_path),
        "original_page_count": total_pages,
        "output_page_count": new_page_count,
        "reconstructed_page_count": new_page_count,
        "user_approved_actions_count": len(actions),
        "records": provenance_records,
        "reconstructed_pages": reconstructed_pages,
    }
    prov_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_payload
