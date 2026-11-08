"""
Dịch vụ hỗ trợ kiểm tra ranh giới trang, lưu phản hồi người dùng và điều phối khôi phục trang
(`src/pdf_inspector/services/boundary_inspection_service.py`).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pdf_inspector.core.config import BOUNDARY_FEEDBACK_PATH
from pdf_inspector.models.schemas import DefectResult, PageDecision
from pdf_inspector.processing.page_reconstructor import (
    export_user_approved_reconstruction_pdf,
    preview_reconstruct_with_adjacent_page,
    preview_split_single_page,
    render_boundary_overlay,
)
from pdf_inspector.processing.pdf_loader import PDFLoader


def save_boundary_review_feedback(
    file_name: str,
    page_number: int,
    boundary_status: str,
    feedback_action: str,
    boundary_defects: List[DefectResult],
    adjusted_cut_norm: Optional[float] = None,
    related_pages: Optional[List[int]] = None,
    feedback_path: Path = BOUNDARY_FEEDBACK_PATH,
) -> Path:
    """
    Lưu phản hồi đánh giá Page Boundary của người dùng (`Confirm`, `False Positive`, `Adjusted Boundary`)
    vào file `data/outputs/boundary_review_feedback.json`.
    Đánh dấu rõ `verified_ground_truth: false`.
    """
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "verified_ground_truth": False,
        "source": "user_ui_boundary_review_feedback",
        "note": (
            "Unverified user review feedback collected from Streamlit UI. "
            "Do not treat as verified benchmark ground truth without manual QA audit."
        ),
        "entries": [],
    }
    if feedback_path.exists():
        try:
            loaded = json.loads(feedback_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("entries"), list):
                payload = loaded
        except Exception:
            pass

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file_name": file_name,
        "page_number": page_number,
        "detector_boundary_status": boundary_status,
        "user_feedback": feedback_action,
        "adjusted_cut_norm": (
            round(float(adjusted_cut_norm), 4)
            if adjusted_cut_norm is not None
            else None
        ),
        "related_pages": list(related_pages or []),
        "verified_ground_truth": False,
        "detected_boundary_defects": [d.to_dict() for d in boundary_defects],
    }

    entries: List[Dict[str, Any]] = payload.get("entries", [])
    updated = False
    for idx, item in enumerate(entries):
        if (
            item.get("file_name") == file_name
            and item.get("page_number") == page_number
        ):
            entries[idx] = entry
            updated = True
            break
    if not updated:
        entries.append(entry)

    payload["entries"] = entries
    feedback_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return feedback_path


def compute_boundary_summary(decisions: List[PageDecision]) -> Dict[str, Any]:
    """
    Tính toán bảng thống kê tổng hợp cho khu vực **Page Boundary Inspection**:
    - Tổng số trang nghi lỗi (`suspected_or_detected_pages`)
    - Số trang `MIXED_PAGE` (`mixed_page_count`)
    - Số trang `SPLIT_PAGE` (`split_page_count`)
    - Số trường hợp `CONTENT_OVERLAP` (`content_overlap_count`)
    - Số trang `CONTENT_TRUNCATED` (`content_truncated_count`)
    - Số trang `RECONSTRUCTABLE` (`reconstructable_pages`)
    - Số trang `MISSING_CONTENT` (`missing_content_pages`)
    - Số trang cần `REVIEW` (`boundary_review_pages`)
    """
    total_pages = len(decisions)
    normal_pages = sum(1 for d in decisions if d.boundary_status == "NORMAL")
    mixed_pages = sum(
        1
        for d in decisions
        if any(df.defect_type == "MIXED_PAGE" for df in d.boundary_defects)
    )
    split_pages = sum(
        1
        for d in decisions
        if any(df.defect_type == "SPLIT_PAGE" for df in d.boundary_defects)
    )
    overlap_pages = sum(
        1
        for d in decisions
        if any(df.defect_type == "CONTENT_OVERLAP" for df in d.boundary_defects)
    )
    truncated_pages = sum(
        1
        for d in decisions
        if any(df.defect_type == "CONTENT_TRUNCATED" for df in d.boundary_defects)
    )
    suspected_boundary_pages = sum(
        1
        for d in decisions
        if any(df.defect_type == "PAGE_BOUNDARY_SUSPECTED" for df in d.boundary_defects)
    )
    reconstructable_pages = sum(
        1
        for d in decisions
        if d.boundary_status == "RECONSTRUCTABLE"
        or (
            d.reconstruction_proposal is not None
            and bool(d.reconstruction_proposal.get("reconstructable", False))
        )
    )
    missing_content_pages = sum(
        1 for d in decisions if d.boundary_status == "MISSING_CONTENT"
    )
    error_pages = sum(
        1 for d in decisions if d.boundary_status == "PROCESSING_ERROR"
    )
    flagged_pages = sum(
        1 for d in decisions if d.boundary_status not in ("NORMAL", "NOT_CHECKED")
    )
    review_pages = sum(
        1
        for d in decisions
        if d.boundary_status
        in (
            "BOUNDARY_SUSPECTED",
            "MIXED_PAGE_DETECTED",
            "RECONSTRUCTABLE",
            "MISSING_CONTENT",
            "REVIEW",
            "PROCESSING_ERROR",
        )
    )

    return {
        "total_pages": total_pages,
        "normal_pages": normal_pages,
        "flagged_pages": flagged_pages,
        "mixed_page_count": mixed_pages,
        "split_page_count": split_pages,
        "content_overlap_count": overlap_pages,
        "content_truncated_count": truncated_pages,
        "page_boundary_suspected_count": suspected_boundary_pages,
        "reconstructable_pages": reconstructable_pages,
        "missing_content_pages": missing_content_pages,
        "boundary_review_pages": review_pages,
        "boundary_error_pages": error_pages,
    }


class BoundaryInspectionService:
    """
    Lớp dịch vụ tập trung điều phối các thao tác kiểm tra, xem trước tách/ghép trang
    và xuất file PDF đã tái tạo có xác nhận của người dùng (`Phase 3.8 & 3.9`).
    """

    @staticmethod
    def generate_split_preview(
        pdf_path: Path,
        page_number: int,
        boundary_norm: float,
        dpi: int = 120,
    ) -> Dict[str, Any]:
        with PDFLoader(pdf_path) as loader:
            page_rgb = loader.render_preview(
                page_number=page_number,
                dpi=dpi,
                extra_rotation=0,
            )
        top_part, bottom_part, info = preview_split_single_page(
            page_image=page_rgb,
            cut_position_norm=boundary_norm,
            orientation="HORIZONTAL",
            enforce_safe_cut=True,
            source_page_number=page_number,
        )
        annotated_rgb = render_boundary_overlay(
            image=page_rgb,
            boundary_defects=[],
            custom_cut_norm=info["applied_safe_cut_norm"],
        )
        return {
            "page_number": page_number,
            "requested_cut_norm": info["requested_cut_norm"],
            "safe_cut_norm": info["applied_safe_cut_norm"],
            "safe_cut_y_px": info["applied_safe_cut_px"],
            "crosses_text": info["crosses_text"],
            "safe_to_cut": info["safe_to_cut"],
            "annotated_source_rgb": annotated_rgb,
            "top_part_rgb": top_part,
            "bottom_part_rgb": bottom_part,
            "source_regions": info["source_regions"],
        }

    @staticmethod
    def generate_adjacent_merge_preview(
        pdf_path: Path,
        first_page_number: int,
        second_page_number: int,
        mode: str = "SPLIT_PAGE",
        split_norm_on_first_page: Optional[float] = None,
        overlap_ratio: float = 0.15,
        dpi: int = 120,
    ) -> Dict[str, Any]:
        with PDFLoader(pdf_path) as loader:
            img_1 = loader.render_preview(
                page_number=first_page_number,
                dpi=dpi,
                extra_rotation=0,
            )
            img_2 = loader.render_preview(
                page_number=second_page_number,
                dpi=dpi,
                extra_rotation=0,
            )

        if mode in ("CONTENT_OVERLAP", "DEDUPLICATE_OVERLAP_WITH_NEXT", "REMOVE_OVERLAP_AND_MERGE"):
            action_type = "REMOVE_OVERLAP_AND_MERGE"
        elif mode in ("MIXED_PAGE", "SPLIT_AND_MERGE_SPILLOVER"):
            action_type = "SPLIT_AND_MERGE_SPILLOVER"
        elif mode == "REQUIRES_RESCAN":
            action_type = "REQUIRES_RESCAN"
        else:
            action_type = "MERGE_SPLIT_PAGES"

        cut_norm = float(split_norm_on_first_page if split_norm_on_first_page is not None else 0.70)
        rest_1, rest_2, prov = preview_reconstruct_with_adjacent_page(
            primary_image=img_1,
            adjacent_image=img_2,
            action_type=action_type,
            cut_position_norm=cut_norm,
            overlap_strip_norm=overlap_ratio,
            primary_page_num=first_page_number,
            adjacent_page_num=second_page_number,
        )
        return {
            "first_page": first_page_number,
            "second_page": second_page_number,
            "mode": mode,
            "action_type": action_type,
            "reconstructed_page_1_rgb": rest_1,
            "reconstructed_page_2_rgb": rest_2,
            "source_regions": prov["source_regions"],
            "provenance": prov,
        }

    @staticmethod
    def export_approved_reconstruction_pdf(
        input_pdf_path: Path,
        output_pdf_path: Path,
        approved_operations: List[Dict[str, Any]],
        user_confirmed: bool = False,
        provenance_json_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        return export_user_approved_reconstruction_pdf(
            input_pdf_path=input_pdf_path,
            output_pdf_path=output_pdf_path,
            approved_operations=approved_operations,
            user_confirmed=user_confirmed,
            provenance_json_path=provenance_json_path,
        )


__all__ = [
    "BoundaryInspectionService",
    "save_boundary_review_feedback",
    "compute_boundary_summary",
    "render_boundary_overlay",
    "preview_split_single_page",
    "preview_reconstruct_with_adjacent_page",
    "export_user_approved_reconstruction_pdf",
]
