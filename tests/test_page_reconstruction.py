"""
Kiểm thử đơn vị và tích hợp cho Page Reconstruction Preview & User-Approved Export (`tests/test_page_reconstruction.py`):
- Phase 3.8: Tìm đường cắt an toàn không cắt ngang chữ (`crosses_text == False`).
- Xem trước tách đôi trang (`preview_split_single_page`) và ghép/khử trùng lặp liên trang (`preview_reconstruct_with_adjacent_page`).
- Cơ chế bảo vệ nghiêm ngặt:
  + Bắt buộc `user_confirmed=True` (ném `PermissionError` nếu chưa xác nhận).
  + Từ chối tự động cắt/ghép đối với `REQUIRES_RESCAN` / `MISSING_CONTENT`.
  + Không bao giờ ghi đè lên file PDF gốc.
  + Lưu đầy đủ truy vết nguồn gốc `source_regions` vào `reconstruction_provenance.json`.
"""

import hashlib
import json
from pathlib import Path
import fitz
import pytest

from pdf_inspector.processing.page_reconstructor import (
    export_user_approved_reconstruction_pdf,
    preview_reconstruct_with_adjacent_page,
    preview_split_single_page,
)
from pdf_inspector.services.boundary_inspection_service import BoundaryInspectionService
from scripts.generate_boundary_samples import (
    BOUNDARY_BENCHMARK_DIR,
    GROUND_TRUTH_PATH,
    generate_all_boundary_samples,
)


@pytest.fixture(scope="module", autouse=True)
def ensure_boundary_samples() -> None:
    if not GROUND_TRUTH_PATH.exists():
        generate_all_boundary_samples()


def test_preview_split_single_page_snaps_to_safe_cut_line() -> None:
    """
    Kiểm tra `preview_split_single_page` trên `02_two_pages_in_one_image.pdf`:
    - Tự động căn chỉnh đường cắt vào thung lũng khoảng trắng an toàn (`crosses_text == False`).
    - Sinh đầy đủ ảnh `annotated_source_rgb`, `top_part_rgb`, `bottom_part_rgb` và `source_regions`.
    """
    mixed_pdf = BOUNDARY_BENCHMARK_DIR / "02_two_pages_in_one_image.pdf"
    prev = BoundaryInspectionService.generate_split_preview(
        pdf_path=mixed_pdf,
        page_number=1,
        boundary_norm=0.51,
        dpi=120,
    )
    assert prev["crosses_text"] is False
    assert prev["safe_cut_norm"] == pytest.approx(0.50, abs=0.05)
    assert prev["top_part_rgb"].shape[0] > 100
    assert prev["bottom_part_rgb"].shape[0] > 100
    assert len(prev["source_regions"]) == 2
    assert prev["source_regions"][0]["source_page"] == 1
    assert prev["source_regions"][1]["source_page"] == 1


def test_preview_reconstruct_with_adjacent_page_split_and_overlap() -> None:
    """
    Kiểm tra `preview_reconstruct_with_adjacent_page` cho 2 kịch bản:
    - `SPLIT_PAGE` (`04_page_b_continuation_from_prev.pdf`): Ghép phần nội dung của Trang 1 và Trang 2.
    - `CONTENT_OVERLAP` (`07_content_overlap_pages.pdf`): Khử phần trùng lặp ở đầu Trang 2.
    """
    split_pdf = BOUNDARY_BENCHMARK_DIR / "04_page_b_continuation_from_prev.pdf"
    split_prev = BoundaryInspectionService.generate_adjacent_merge_preview(
        pdf_path=split_pdf,
        first_page_number=1,
        second_page_number=2,
        mode="SPLIT_PAGE",
        dpi=120,
    )
    assert split_prev["mode"] == "SPLIT_PAGE"
    assert split_prev["reconstructed_page_1_rgb"].shape[0] > 200
    assert split_prev["reconstructed_page_2_rgb"] is None
    assert len(split_prev["source_regions"]) == 2

    overlap_pdf = BOUNDARY_BENCHMARK_DIR / "07_content_overlap_pages.pdf"
    ov_prev = BoundaryInspectionService.generate_adjacent_merge_preview(
        pdf_path=overlap_pdf,
        first_page_number=1,
        second_page_number=2,
        mode="CONTENT_OVERLAP",
        overlap_ratio=0.19,
        dpi=120,
    )
    assert ov_prev["mode"] == "CONTENT_OVERLAP"
    assert ov_prev["reconstructed_page_1_rgb"].shape[0] > 200
    assert ov_prev["reconstructed_page_2_rgb"] is not None
    assert len(ov_prev["source_regions"]) == 2


def test_export_reconstruction_requires_confirmation_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    """
    Kiểm tra các rào chắn an toàn của `export_user_approved_reconstruction_pdf`:
    1. Ném `PermissionError` khi `user_confirmed=False`.
    2. Ném `ValueError` khi cố tình yêu cầu tái tạo trang `REQUIRES_RESCAN` / `MISSING_CONTENT`.
    3. Ném `ValueError` nếu `output_pdf_path` trùng `input_pdf_path`.
    4. Khi `user_confirmed=True`, tạo file PDF mới (tách 1 trang `MIXED_PAGE` thành 2 trang PDF),
       không làm thay đổi byte nào của PDF gốc, và ghi đầy đủ `source_regions` vào file provenance JSON.
    """
    mixed_pdf = BOUNDARY_BENCHMARK_DIR / "02_two_pages_in_one_image.pdf"
    orig_hash = hashlib.sha256(mixed_pdf.read_bytes()).hexdigest()

    out_pdf = tmp_path / "reconstructed_split.pdf"
    prov_json = tmp_path / "provenance.json"

    # 1. Từ chối khi user_confirmed=False
    with pytest.raises(PermissionError, match="user_confirmed=True"):
        export_user_approved_reconstruction_pdf(
            input_pdf_path=mixed_pdf,
            output_pdf_path=out_pdf,
            approved_operations=[
                {"action": "SPLIT_SINGLE_PAGE", "page_number": 1, "cut_norm": 0.50}
            ],
            user_confirmed=False,
        )

    # 2. Từ chối khi action là REQUIRES_RESCAN hoặc MISSING_CONTENT
    with pytest.raises(ValueError, match="MISSING_CONTENT"):
        export_user_approved_reconstruction_pdf(
            input_pdf_path=mixed_pdf,
            output_pdf_path=out_pdf,
            approved_operations=[
                {"action": "REQUIRES_RESCAN", "page_number": 1}
            ],
            user_confirmed=True,
        )

    # 3. Từ chối ghi đè lên chính file PDF gốc
    with pytest.raises(ValueError, match="ghi đè"):
        export_user_approved_reconstruction_pdf(
            input_pdf_path=mixed_pdf,
            output_pdf_path=mixed_pdf,
            approved_operations=[
                {"action": "SPLIT_SINGLE_PAGE", "page_number": 1, "cut_norm": 0.50}
            ],
            user_confirmed=True,
        )

    # 4. Xuất file hợp lệ khi user_confirmed=True
    export_res = BoundaryInspectionService.export_approved_reconstruction_pdf(
        input_pdf_path=mixed_pdf,
        output_pdf_path=out_pdf,
        approved_operations=[
            {"action": "SPLIT_SINGLE_PAGE", "page_number": 1, "cut_norm": 0.50}
        ],
        user_confirmed=True,
        provenance_json_path=prov_json,
    )

    assert out_pdf.exists()
    assert prov_json.exists()
    assert export_res["original_page_count"] == 1
    assert export_res["reconstructed_page_count"] == 2
    assert hashlib.sha256(mixed_pdf.read_bytes()).hexdigest() == orig_hash

    with fitz.open(str(out_pdf)) as doc:
        assert len(doc) == 2

    prov_data = json.loads(prov_json.read_text(encoding="utf-8"))
    assert prov_data["user_confirmed"] is True
    assert len(prov_data["reconstructed_pages"]) == 2
    assert prov_data["reconstructed_pages"][0]["operation"] == "SPLIT_SINGLE_PAGE_TOP"
    assert (
        prov_data["reconstructed_pages"][1]["operation"] == "SPLIT_SINGLE_PAGE_BOTTOM"
    )
    assert len(prov_data["reconstructed_pages"][0]["source_regions"]) == 1

