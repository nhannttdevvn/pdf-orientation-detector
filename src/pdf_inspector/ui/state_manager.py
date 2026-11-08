"""
Quản lý trạng thái phiên làm việc (Session State), bộ nhớ đệm Lazy Preview (LRU),
mô hình hiển thị lỗi từng trang (View Models) và điều phối cập nhật kết quả/xuất PDF
mà không chạy lại OCR (`src/pdf_inspector/ui/state_manager.py`).
"""

from collections import OrderedDict
from dataclasses import dataclass
import gc
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore[no-redef]

# Import boundary_detector trước để tránh vòng lặp import sẵn có trong backend
import pdf_inspector.detectors.boundary.boundary_detector  # noqa: F401
from pdf_inspector.core.config import (
    BOUNDARY_STATUS_ERROR,
    BOUNDARY_STATUS_MISSING,
    BOUNDARY_STATUS_MIXED,
    BOUNDARY_STATUS_NORMAL,
    BOUNDARY_STATUS_RECONSTRUCTABLE,
    BOUNDARY_STATUS_REVIEW,
    BOUNDARY_STATUS_SUSPECTED,
    ORIENTATION_CONF_THRESHOLD,
    OUTPUTS_DIR,
    PREVIEW_DPI,
    RECONSTRUCTED_PDF_PATH,
    RECONSTRUCTION_PROVENANCE_PATH,
    RENDER_DPI,
    STATUS_REVIEW,
    STATUS_ROTATED,
    STRIPE_STATUS_DETECTED,
    STRIPE_STATUS_ERROR,
    STRIPE_STATUS_REVIEW,
    DetectorConfig,
)
from pdf_inspector.models.schemas import DefectResult, DefectStatus, PageDecision, PipelineResult
from pdf_inspector.pipeline.document_pipeline import DocumentPipeline
from pdf_inspector.processing.pdf_loader import PDFLoader
from pdf_inspector.services.boundary_inspection_service import BoundaryInspectionService
from pdf_inspector.services.secret_scan_service import SecretDocumentResult, SecretScanService

MAX_PREVIEW_CACHE_ITEMS: int = 16


@dataclass
class DefectViewItem:
    """Thông tin chuẩn hóa của từng lỗi cụ thể trên một trang để hiển thị & xác nhận riêng."""

    key: str  # "boundary", "stripe", "rotation"
    short_chip: str  # Nhãn chip ngắn khi trang có nhiều lỗi
    defect_title: str  # Hiển thị ở hàng "Lỗi:"
    pill_label: str  # Nhãn pill nổi trên vùng preview
    proposal_text: str  # Hiển thị ở hàng "Đề xuất:"
    is_suspected: bool  # True nếu nghi vấn / độ tin cậy thấp
    can_accept: bool
    accept_disabled_reason: Optional[str]
    can_manual_edit: bool
    manual_disabled_reason: Optional[str]
    tooltip: str


@dataclass
class PageIssueRow:
    """Mô hình dữ liệu một dòng trong bảng 'Danh sách trang có lỗi'."""

    page_number: int
    decision: PageDecision
    stripe_state: str  # "NONE" | "CONFIRMED" | "SUSPECTED"
    stripe_tooltip: str
    boundary_state: str  # "NONE" | "CONFIRMED" | "SUSPECTED"
    boundary_tooltip: str
    rotation_state: str  # "NONE" | "CONFIRMED" | "SUSPECTED"
    rotation_display: str  # "180°", "90°", "?", "—"
    rotation_tooltip: str
    status_label: str  # "Đã tự xoay" | "Cần kiểm tra" | "Đã xử lý" | "Đã bỏ qua"
    status_tone: str  # "green" | "orange" | "gray"
    needs_review_now: bool
    defect_items: List[DefectViewItem]


@st.cache_resource(show_spinner=False)
def get_cached_pipeline(
    render_dpi: int = RENDER_DPI,
    conf_threshold: float = ORIENTATION_CONF_THRESHOLD,
) -> DocumentPipeline:
    """
    Khởi tạo và giữ bộ nhớ đệm các Detector trong `DocumentPipeline` xuyên suốt vòng đời ứng dụng,
    đảm bảo không nạp lại detector/model khi UI rerun.
    """
    cfg = DetectorConfig(
        render_dpi=render_dpi,
        orientation_conf_threshold=conf_threshold,
        auto_fix=True,
    )
    return DocumentPipeline(config=cfg)


@st.cache_resource(show_spinner=False)
def get_cached_secret_service() -> SecretScanService:
    """
    Khởi tạo một lần duy nhất `SecretScanService` (nạp bộ template 200 DPI + pyramid cache vào RAM),
    tách biệt hoàn toàn với `DocumentPipeline` để không chạy thừa detector khi đổi chế độ.
    """
    return SecretScanService()


def init_dashboard_state() -> None:
    """Khởi tạo đầy đủ biến trạng thái cần thiết trong `st.session_state` cho cả 2 chế độ độc lập."""
    defaults: Dict[str, Any] = {
        "active_mode": "SCAN_QUALITY",  # "SCAN_QUALITY" | "SECRET_DETECTION"
        # --- Trạng thái Tab 1: Scan Quality Workflow ---
        "uploader_key_ver": 0,
        "file_hash": None,
        "input_pdf_path": None,
        "display_file_name": None,
        "file_size_bytes": 0,
        "doc_info": None,
        "doc_error": None,
        "pipeline_result": None,
        "active_filter": "ALL",  # "ALL" | "STRIPE" | "BOUNDARY" | "ROTATION"
        "table_page_idx": 0,
        "selected_page_num": None,
        "selected_defect_key_by_page": {},  # page_num -> "boundary" | "stripe" | "rotation"
        "preview_tab": "ORIGINAL",  # "ORIGINAL" | "PROPOSED"
        "manual_overrides": {},  # page_num -> manual rotation angle (0, 90, 180, 270)
        "stripe_feedback": {},  # page_num -> "Confirm defect" | "False positive" | "Uncertain"
        "boundary_feedback": {},  # page_num -> "Confirm" | "False Positive" | "Adjust Boundary"
        "boundary_adjustments": {},  # page_num -> float cut_norm
        "approved_boundary_ops": {},  # page_num -> Dict operation for export
        "defect_review_decisions": {},  # f"{page_num}:{defect_key}" -> "ACCEPTED" | "SKIPPED" | "MANUAL"
        "has_unexported_changes": False,
        "has_exported_current": False,
        "export_pdf_bytes": None,
        "export_file_name": "corrected.pdf",
        "preview_lru_cache": OrderedDict(),
        "proposal_preview_cache": OrderedDict(),
        "action_feedback_msg": None,
        # --- Trạng thái Tab 2: Secret Marking Workflow (cô lập 100%) ---
        "secret_uploader_key_ver": 0,
        "secret_file_hash": None,
        "secret_pdf_path": None,
        "secret_file_name": None,
        "secret_file_size_bytes": 0,
        "secret_doc_info": None,
        "secret_doc_error": None,
        "secret_doc_result": None,
        "secret_active_filter": "ALL",  # "ALL" | "MAT" | "TOI_MAT" | "TUYET_MAT" | "REVIEW"
        "secret_table_page_idx": 0,
        "secret_selected_page_num": None,
        "secret_preview_lru_cache": OrderedDict(),
        "secret_action_feedback_msg": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def release_secret_resources(increment_uploader: bool = False) -> None:
    """
    Giải phóng toàn bộ bộ đệm ảnh xem trước tạm thời, ảnh crop dấu mật trong RAM
    và kết quả của riêng Tab 2 mà không ảnh hưởng tới trạng thái Tab 1.
    """
    if "secret_preview_lru_cache" in st.session_state:
        st.session_state["secret_preview_lru_cache"].clear()

    st.session_state["secret_file_hash"] = None
    st.session_state["secret_pdf_path"] = None
    st.session_state["secret_file_name"] = None
    st.session_state["secret_file_size_bytes"] = 0
    st.session_state["secret_doc_info"] = None
    st.session_state["secret_doc_error"] = None
    st.session_state["secret_doc_result"] = None
    st.session_state["secret_active_filter"] = "ALL"
    st.session_state["secret_table_page_idx"] = 0
    st.session_state["secret_selected_page_num"] = None
    st.session_state["secret_action_feedback_msg"] = None

    if increment_uploader:
        st.session_state["secret_uploader_key_ver"] = (
            int(st.session_state.get("secret_uploader_key_ver", 0)) + 1
        )
    gc.collect()


def release_document_resources(increment_uploader: bool = False) -> None:
    """
    Giải phóng toàn bộ bộ nhớ đệm ảnh xem trước, kết quả pipeline và trạng thái review
    khi người dùng xóa file hiện tại hoặc chuyển sang file PDF mới.
    """
    if "preview_lru_cache" in st.session_state:
        st.session_state["preview_lru_cache"].clear()
    if "proposal_preview_cache" in st.session_state:
        st.session_state["proposal_preview_cache"].clear()

    st.session_state["file_hash"] = None
    st.session_state["input_pdf_path"] = None
    st.session_state["display_file_name"] = None
    st.session_state["file_size_bytes"] = 0
    st.session_state["doc_info"] = None
    st.session_state["doc_error"] = None
    st.session_state["pipeline_result"] = None
    st.session_state["active_filter"] = "ALL"
    st.session_state["table_page_idx"] = 0
    st.session_state["selected_page_num"] = None
    st.session_state["selected_defect_key_by_page"] = {}
    st.session_state["preview_tab"] = "ORIGINAL"
    st.session_state["manual_overrides"] = {}
    st.session_state["stripe_feedback"] = {}
    st.session_state["boundary_feedback"] = {}
    st.session_state["boundary_adjustments"] = {}
    st.session_state["approved_boundary_ops"] = {}
    st.session_state["defect_review_decisions"] = {}
    st.session_state["has_unexported_changes"] = False
    st.session_state["has_exported_current"] = False
    st.session_state["export_pdf_bytes"] = None
    st.session_state["export_file_name"] = "corrected.pdf"
    st.session_state["action_feedback_msg"] = None

    if increment_uploader:
        st.session_state["uploader_key_ver"] = (
            int(st.session_state.get("uploader_key_ver", 0)) + 1
        )
    gc.collect()


def get_cached_page_preview_rgb(
    pdf_path: Path,
    file_hash: str,
    page_number: int,
    dpi: int = PREVIEW_DPI,
    extra_rotation: int = 0,
) -> np.ndarray:
    """
    Lazy-render ảnh xem trước RGB của đúng trang đang chọn ở độ phân giải vừa phải (`PREVIEW_DPI`),
    lưu trong bộ đệm LRU giới hạn tối đa `MAX_PREVIEW_CACHE_ITEMS` trang để tiết kiệm RAM.
    """
    cache: OrderedDict = st.session_state.setdefault("preview_lru_cache", OrderedDict())
    cache_key = (file_hash, int(page_number), int(dpi), int(extra_rotation) % 360)
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    with PDFLoader(pdf_path) as loader:
        rgb = loader.render_preview(
            page_number=page_number,
            dpi=dpi,
            extra_rotation=extra_rotation,
        )

    cache[cache_key] = rgb
    while len(cache) > MAX_PREVIEW_CACHE_ITEMS:
        cache.popitem(last=False)
    return rgb


def encode_rgb_to_png_bytes(rgb_img: np.ndarray) -> bytes:
    """Mã hóa nhanh mảng ảnh RGB NumPy sang PNG bytes để hiển thị sắc nét trong khung Preview."""
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) if rgb_img.ndim == 3 else rgb_img
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Không thể mã hóa ảnh xem trước sang định dạng PNG.")
    return buf.tobytes()


def _safe_export_reconstructed_pdf(
    input_pdf_path: Path,
    output_pdf_path: Path,
    approved_ops: List[Dict[str, Any]],
) -> Path:
    """
    Gọi `BoundaryInspectionService.export_approved_reconstruction_pdf` với `user_confirmed=True`.
    Nếu tài liệu nhiều trang gặp lỗi `PDFLoader.document` sẵn có trong backend ở các trang giữ nguyên,
    thực hiện tách trang mục tiêu qua đúng `BoundaryInspectionService.export_approved_reconstruction_pdf`
    rồi ghép lại an toàn với các trang không can thiệp mà không sửa code backend hay chạm vào PDF gốc.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        BoundaryInspectionService.export_approved_reconstruction_pdf(
            input_pdf_path=input_pdf_path,
            output_pdf_path=output_pdf_path,
            approved_operations=approved_ops,
            user_confirmed=True,
            provenance_json_path=RECONSTRUCTION_PROVENANCE_PATH,
        )
        return output_pdf_path
    except AttributeError as exc:
        if "document" not in str(exc):
            raise

    # Xử lý an toàn từng thao tác qua đúng service backend trên file trích tạm 1-2 trang
    ops_by_page = {int(op["page_number"]): dict(op) for op in approved_ops}
    consumed_pages = set()
    src_doc = fitz.open(str(input_pdf_path))
    out_doc = fitz.open()
    try:
        total_pages = src_doc.page_count
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            for pno in range(1, total_pages + 1):
                if pno in consumed_pages:
                    continue
                if pno not in ops_by_page:
                    out_doc.insert_pdf(src_doc, from_page=pno - 1, to_page=pno - 1)
                    continue

                op = ops_by_page[pno]
                sec_page = op.get("second_page_number") or op.get("related_page")
                sub_in = tmp_root / f"slice_in_{pno}.pdf"
                sub_out = tmp_root / f"slice_out_{pno}.pdf"

                slice_doc = fitz.open()
                slice_doc.insert_pdf(src_doc, from_page=pno - 1, to_page=pno - 1)
                local_op = dict(op)
                local_op["page_number"] = 1
                if sec_page is not None and int(sec_page) != pno and 1 <= int(sec_page) <= total_pages:
                    adj_pno = int(sec_page)
                    slice_doc.insert_pdf(src_doc, from_page=adj_pno - 1, to_page=adj_pno - 1)
                    local_op["second_page_number"] = 2
                    consumed_pages.add(adj_pno)
                slice_doc.save(str(sub_in))
                slice_doc.close()

                BoundaryInspectionService.export_approved_reconstruction_pdf(
                    input_pdf_path=sub_in,
                    output_pdf_path=sub_out,
                    approved_operations=[local_op],
                    user_confirmed=True,
                    provenance_json_path=RECONSTRUCTION_PROVENANCE_PATH,
                )
                rec_slice = fitz.open(str(sub_out))
                out_doc.insert_pdf(rec_slice)
                rec_slice.close()
                consumed_pages.add(pno)

        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        out_doc.save(str(output_pdf_path))
    finally:
        out_doc.close()
        src_doc.close()
    return output_pdf_path


def build_corrected_pdf(
    result: PipelineResult,
    manual_overrides: Dict[int, int],
    approved_ops_map: Dict[int, Dict[str, Any]],
    output_dir: Path,
    file_tag: str,
    json_report_path: Optional[Path] = None,
    csv_report_path: Optional[Path] = None,
    reconstructed_path: Optional[Path] = None,
) -> Tuple[PipelineResult, Path]:
    """
    Tạo PDF đã sửa từ kết quả quét (không chạy lại OCR/Detector, không đọc/ghi session_state):
    - Xoay các trang ROTATED (auto fix) và các trang người dùng chỉnh góc thủ công; trang REVIEW không tự xoay.
    - Áp dụng các thao tác tách/ghép ranh giới trang ĐÃ được người dùng phê duyệt.
    PDF đầu vào là bản sao phía server (`result.doc_info.file_path`), không chạm vào PDF gốc.
    Trả về `(kết quả đã cập nhật quyết định, đường dẫn PDF cuối cùng)`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_pdf_path = output_dir / f"corrected_{file_tag}.pdf"
    updated = DocumentPipeline.update_decisions_and_artifacts(
        result=result,
        conf_threshold=result.config.orientation_conf_threshold,
        auto_fix=True,
        manual_overrides=manual_overrides,
        output_pdf_path=corrected_pdf_path,
        json_report_path=json_report_path,
        csv_report_path=csv_report_path,
        write_corrected_pdf=True,
    )
    final_pdf_path = corrected_pdf_path
    if approved_ops_map:
        ops_list = [approved_ops_map[k] for k in sorted(approved_ops_map.keys())]
        final_pdf_path = _safe_export_reconstructed_pdf(
            input_pdf_path=corrected_pdf_path,
            output_pdf_path=reconstructed_path or (output_dir / f"reconstructed_{file_tag}.pdf"),
            approved_ops=ops_list,
        )
    return updated, final_pdf_path


def rebuild_outputs_without_ocr() -> None:
    """
    Cập nhật lại toàn bộ quyết định (`UnifiedQualityEngine`), file PDF đã sửa góc xoay (`PDFCorrector`),
    các trang tách/ghép đã được người dùng phê duyệt (`BoundaryInspectionService`),
    và báo cáo JSON/CSV (`ReportGenerator`) mà KHÔNG chạy lại OCR hay Detector.
    """
    result: Optional[PipelineResult] = st.session_state.get("pipeline_result")
    if result is None:
        return

    overrides: Dict[int, int] = st.session_state.get("manual_overrides", {})
    approved_ops_map: Dict[int, Dict[str, Any]] = st.session_state.get(
        "approved_boundary_ops", {}
    )

    file_hash = (st.session_state.get("file_hash") or "doc")[:10]
    updated, final_pdf_path = build_corrected_pdf(
        result=result,
        manual_overrides=overrides,
        approved_ops_map=approved_ops_map,
        output_dir=OUTPUTS_DIR,
        file_tag=file_hash,
        json_report_path=OUTPUTS_DIR / "orientation_report.json",
        csv_report_path=OUTPUTS_DIR / "orientation_report.csv",
        reconstructed_path=RECONSTRUCTED_PDF_PATH,
    )
    st.session_state["pipeline_result"] = updated

    orig_stem = Path(st.session_state.get("display_file_name") or "document.pdf").stem
    export_name = f"{orig_stem}_da_sua.pdf"

    if final_pdf_path.exists():
        st.session_state["export_pdf_bytes"] = final_pdf_path.read_bytes()
        st.session_state["export_file_name"] = export_name


def get_default_boundary_cut_norm(decision: PageDecision) -> float:
    """Lấy tọa độ đường cắt chuẩn hóa mặc định từ kết quả Boundary Detector."""
    if decision.reconstruction_proposal:
        c_norm = decision.reconstruction_proposal.get("cut_position_norm")
        if c_norm and float(c_norm) > 0.02:
            return float(c_norm)
    for bdef in decision.boundary_defects:
        details = bdef.details or {}
        for k in ("safe_cut_line_norm", "suspected_boundary_norm"):
            if details.get(k) is not None and float(details[k]) > 0.02:
                return float(details[k])
        for cand in bdef.boundary_candidates:
            pos = cand.get("safe_cut_line_norm") or cand.get("position")
            if pos is not None and float(pos) > 0.02:
                return float(pos)
    return 0.50


def _describe_boundary_defect(decision: PageDecision) -> Tuple[str, str, str, bool, str]:
    """
    Trả về `(defect_title, pill_label, proposal_text, can_reconstruct, disabled_reason)`
    dựa trên kết quả thực tế từ `PageBoundaryDetector`.
    """
    primary_type = "MIXED_PAGE"
    reason_tip = "Phát hiện bất thường ranh giới trang."
    if decision.boundary_defects:
        b0 = decision.boundary_defects[0]
        primary_type = b0.defect_type
        reason_tip = b0.reason or reason_tip

    proposal = decision.reconstruction_proposal or {}
    action_type = str(proposal.get("action_type") or "")
    reconstructable = bool(proposal.get("reconstructable", False))

    if (
        decision.boundary_status == BOUNDARY_STATUS_MISSING
        or primary_type == "CONTENT_TRUNCATED"
        or action_type == "REQUIRES_RESCAN"
    ):
        return (
            "Mất nội dung sát mép trang",
            "Cắt cụt mép trang",
            "Quét lại trang gốc (thiếu dữ liệu)",
            False,
            "Trang bị cắt mất nội dung ở sát mép (MISSING_CONTENT) và không có trên trang kề; hệ thống từ chối tự cắt/ghép để bảo toàn dữ liệu.",
        )

    if primary_type == "SPLIT_PAGE" or action_type == "MERGE_SPLIT_PAGES":
        return (
            "Trang bị tách đôi liên tiếp",
            "Tách đôi trang",
            "Ghép nối lại với trang liền kề",
            True,
            "",
        )

    if primary_type == "CONTENT_OVERLAP" or action_type == "REMOVE_OVERLAP_AND_MERGE":
        return (
            "Trùng lặp nội dung liên trang",
            "Trùng lặp đầu/cuối trang",
            "Khử vùng trùng lặp và ghép lại",
            True,
            "",
        )

    if primary_type == "PAGE_BOUNDARY_SUSPECTED" and not reconstructable:
        return (
            "Nghi vấn ranh giới chuyển trang",
            "Nghi lẫn trang",
            "Kiểm tra và tách tại ranh giới an toàn",
            True,
            "",
        )

    # Mặc định: MIXED_PAGE (Lẫn đầu/cuối trang — đúng thuật ngữ trên ảnh tham chiếu)
    return (
        "Lẫn đầu/cuối trang",
        "Lẫn đầu/cuối trang",
        "Tách và ghép lại nội dung",
        True,
        "",
    )


def _describe_stripe_defect(decision: PageDecision) -> Tuple[str, str, str]:
    """Trả về `(defect_title, pill_label, tooltip)` từ `decision.stripe_defects`."""
    if not decision.stripe_defects:
        return (
            "Lỗi vệt sọc máy scan",
            "Vệt sọc máy scan",
            "Nghi vấn lỗi sọc trên trang scan.",
        )
    s0 = decision.stripe_defects[0]
    stype = (s0.stripe_type or s0.defect_type or "VERTICAL").upper()
    type_map = {
        "VERTICAL": ("Lỗi vệt sọc dọc máy scan", "Vệt sọc dọc"),
        "HORIZONTAL": ("Lỗi vệt sọc ngang máy scan", "Vệt sọc ngang"),
        "DIAGONAL": ("Lỗi vệt sọc chéo máy scan", "Vệt sọc chéo"),
        "BAND": ("Lỗi dải sọc đậm máy scan", "Dải sọc máy scan"),
    }
    title, pill = type_map.get(stype, ("Lỗi vệt sọc máy scan", "Vệt sọc máy scan"))
    tip = s0.reason or f"Phát hiện {len(decision.stripe_defects)} vệt sọc ({stype})."
    return title, pill, tip


def build_problem_page_rows(result: Optional[PipelineResult]) -> List[PageIssueRow]:
    """
    Tổng hợp danh sách các trang có lỗi hoặc cần kiểm tra từ `PipelineResult` thật.
    Các trang bình thường (`NORMAL` + `CLEAN` + `NORMAL`) hoàn toàn không đưa vào bảng.
    """
    if result is None:
        return []

    review_decisions: Dict[str, str] = st.session_state.get(
        "defect_review_decisions", {}
    )
    manual_overrides: Dict[int, int] = st.session_state.get("manual_overrides", {})
    rows: List[PageIssueRow] = []

    for d in result.decisions:
        pno = d.page_number

        # 1. Phân tích trạng thái Stripe
        if d.stripe_status == STRIPE_STATUS_DETECTED or any(
            s.status == DefectStatus.CONFIRMED.value for s in d.stripe_defects
        ):
            stripe_state = "CONFIRMED"
        elif (
            d.stripe_status in (STRIPE_STATUS_REVIEW, STRIPE_STATUS_ERROR)
            or bool(d.stripe_defects)
        ):
            stripe_state = "SUSPECTED"
        else:
            stripe_state = "NONE"

        # 2. Phân tích trạng thái Boundary / Lẫn trang
        if d.boundary_status in (
            BOUNDARY_STATUS_MIXED,
            BOUNDARY_STATUS_RECONSTRUCTABLE,
            BOUNDARY_STATUS_MISSING,
        ) or any(
            b.status == DefectStatus.CONFIRMED.value for b in d.boundary_defects
        ):
            boundary_state = "CONFIRMED"
        elif (
            d.boundary_status
            in (
                BOUNDARY_STATUS_SUSPECTED,
                BOUNDARY_STATUS_REVIEW,
                BOUNDARY_STATUS_ERROR,
            )
            or bool(d.boundary_defects)
        ):
            boundary_state = "SUSPECTED"
        else:
            boundary_state = "NONE"

        # 3. Phân tích trạng thái Xoay (Orientation)
        has_manual_rot = pno in manual_overrides
        if d.status == STATUS_ROTATED or (has_manual_rot and d.correction_angle != 0):
            rotation_state = "CONFIRMED"
        elif d.status == STATUS_REVIEW:
            rotation_state = "SUSPECTED"
        else:
            rotation_state = "NONE"

        # Bỏ qua trang hoàn toàn bình thường
        if (
            stripe_state == "NONE"
            and boundary_state == "NONE"
            and rotation_state == "NONE"
            and not d.requires_review
        ):
            continue

        # Xây dựng danh sách DefectViewItem thực tế trên trang (ưu tiên Boundary -> Stripe -> Rotation khi cần review)
        defect_items: List[DefectViewItem] = []

        if boundary_state != "NONE":
            b_title, b_pill, b_prop, can_rec, rec_err = _describe_boundary_defect(d)
            b_tip = (
                d.boundary_defects[0].reason
                if (d.boundary_defects and d.boundary_defects[0].reason)
                else b_title
            )
            if boundary_state == "SUSPECTED":
                b_tip = f"[Nghi vấn cần kiểm tra] {b_tip}"
            defect_items.append(
                DefectViewItem(
                    key="boundary",
                    short_chip="Lẫn trang",
                    defect_title=b_title,
                    pill_label=b_pill,
                    proposal_text=b_prop,
                    is_suspected=(boundary_state == "SUSPECTED"),
                    can_accept=can_rec,
                    accept_disabled_reason=(rec_err if not can_rec else None),
                    can_manual_edit=can_rec,
                    manual_disabled_reason=(rec_err if not can_rec else None),
                    tooltip=b_tip,
                )
            )

        if stripe_state != "NONE":
            s_title, s_pill, s_tip = _describe_stripe_defect(d)
            if stripe_state == "SUSPECTED":
                s_tip = f"[Nghi vấn độ tin cậy thấp] {s_tip}"
            defect_items.append(
                DefectViewItem(
                    key="stripe",
                    short_chip="Lỗi sọc",
                    defect_title=s_title,
                    pill_label=s_pill,
                    proposal_text="Ghi nhận lỗi & quét lại bản gốc (không tự xóa sọc)",
                    is_suspected=(stripe_state == "SUSPECTED"),
                    can_accept=True,
                    accept_disabled_reason=None,
                    can_manual_edit=(rotation_state != "NONE" or boundary_state != "NONE"),
                    manual_disabled_reason=(
                        "Backend chưa hỗ trợ xóa vệt sọc trên ảnh PDF để tránh làm mất chữ gốc."
                        if (rotation_state == "NONE" and boundary_state == "NONE")
                        else None
                    ),
                    tooltip=s_tip,
                )
            )

        rot_display = "—"
        rot_tip = ""
        if rotation_state != "NONE":
            effective_angle = (
                d.correction_angle
                if d.manual_override
                else (d.detected_rotation if d.detected_rotation is not None else None)
            )
            if effective_angle in (90, 180, 270):
                rot_display = f"{effective_angle}°" + (
                    " ?" if (rotation_state == "SUSPECTED" and not d.manual_override) else ""
                )
            elif d.manual_override and d.correction_angle == 0:
                rot_display = "0°"
            else:
                rot_display = "?"

            if d.corrected:
                r_title = f"Trang bị xoay {d.correction_angle}°"
                r_prop = f"Xoay tự động {d.correction_angle}° về đúng chiều"
                can_acc_rot = True
                acc_rot_reason = None
            elif effective_angle in (90, 180, 270):
                r_title = f"Nghi trang bị xoay {effective_angle}°"
                r_prop = f"Xoay {effective_angle}° về đúng chiều"
                can_acc_rot = True
                acc_rot_reason = None
            else:
                r_title = "Không xác định rõ hướng trang"
                r_prop = "Chọn góc xoay thủ công (0° / 90° / 180° / 270°)"
                can_acc_rot = False
                acc_rot_reason = (
                    "OCR không đủ độ tin cậy để đề xuất góc xoay tự động. "
                    "Vui lòng nhấn 'Sửa thủ công' để chọn góc xoay."
                )

            rot_tip = d.error or r_title
            defect_items.append(
                DefectViewItem(
                    key="rotation",
                    short_chip=f"Xoay {effective_angle}°" if effective_angle else "Xoay",
                    defect_title=r_title,
                    pill_label=f"Xoay {effective_angle}°" if effective_angle else "Nghi xoay hướng",
                    proposal_text=r_prop,
                    is_suspected=(rotation_state == "SUSPECTED" and not d.manual_override),
                    can_accept=can_acc_rot,
                    accept_disabled_reason=acc_rot_reason,
                    can_manual_edit=True,
                    manual_disabled_reason=None,
                    tooltip=rot_tip,
                )
            )

        # Tính toán trạng thái tổng hợp của dòng ("Đã tự xoay" | "Cần kiểm tra" | "Đã xử lý" | "Đã bỏ qua")
        unresolved_keys: List[str] = []
        any_accepted_or_manual = False
        all_skipped = True

        for item in defect_items:
            dec_flag = review_decisions.get(f"{pno}:{item.key}")
            if dec_flag in ("ACCEPTED", "MANUAL"):
                any_accepted_or_manual = True
                all_skipped = False
            elif dec_flag == "SKIPPED":
                pass
            else:
                all_skipped = False
                # Nếu lỗi xoay đã được tự động xoay thành công và trang không có lỗi cần review khác
                if item.key == "rotation" and d.corrected and not d.manual_override:
                    continue
                # Nếu trang đã được xoay tự động và chỉ kèm sọc đã xác nhận (requires_review=False)
                if item.key == "stripe" and d.corrected and not d.requires_review:
                    continue
                unresolved_keys.append(item.key)

        if unresolved_keys:
            status_label = "Cần kiểm tra"
            status_tone = "orange"
            needs_review_now = True
        elif all_skipped and defect_items:
            status_label = "Đã bỏ qua"
            status_tone = "gray"
            needs_review_now = False
        elif d.corrected and not d.manual_override and not any_accepted_or_manual:
            status_label = "Đã tự xoay"
            status_tone = "green"
            needs_review_now = False
        elif d.corrected and d.manual_override and not (pno in st.session_state.get("approved_boundary_ops", {})):
            status_label = "Đã xoay"
            status_tone = "green"
            needs_review_now = False
        else:
            status_label = "Đã xử lý"
            status_tone = "green"
            needs_review_now = False

        rows.append(
            PageIssueRow(
                page_number=pno,
                decision=d,
                stripe_state=stripe_state,
                stripe_tooltip=_describe_stripe_defect(d)[2] if stripe_state != "NONE" else "",
                boundary_state=boundary_state,
                boundary_tooltip=(
                    d.boundary_defects[0].reason
                    if (d.boundary_defects and d.boundary_defects[0].reason)
                    else "Phát hiện lỗi ranh giới / lẫn trang"
                )
                if boundary_state != "NONE"
                else "",
                rotation_state=rotation_state,
                rotation_display=rot_display,
                rotation_tooltip=rot_tip,
                status_label=status_label,
                status_tone=status_tone,
                needs_review_now=needs_review_now,
                defect_items=defect_items,
            )
        )

    return rows


def compute_dashboard_summary(
    result: Optional[PipelineResult],
    problem_rows: List[PageIssueRow],
) -> Dict[str, int]:
    """
    Tính toán 4 chỉ số Summary Cards từ dữ liệu thực của backend:
    - `total_pages`: Tổng số trang PDF.
    - `error_pages`: Số trang có ít nhất 1 lỗi (mỗi trang chỉ tính 1 lần).
    - `auto_rotated_pages`: Số trang đã thực sự được áp dụng xoay trong PDF xuất (`d.corrected == True`),
      tuyệt đối không tính các đề xuất xoay chưa áp dụng.
    - `review_pages`: Số trang hiện đang cần người dùng kiểm tra (`needs_review_now == True`).
    """
    if result is None:
        return {
            "total_pages": 0,
            "error_pages": 0,
            "auto_rotated_pages": 0,
            "review_pages": 0,
        }

    total_pages = len(result.decisions)
    error_pages = len(problem_rows)
    auto_rotated_pages = sum(1 for d in result.decisions if d.corrected)
    review_pages = sum(1 for r in problem_rows if r.needs_review_now)

    return {
        "total_pages": total_pages,
        "error_pages": error_pages,
        "auto_rotated_pages": auto_rotated_pages,
        "review_pages": review_pages,
    }

