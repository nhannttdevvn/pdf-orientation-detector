"""
Kiểm thử chức năng chọn thư mục hồ sơ & quét toàn bộ cây thư mục (`backend.workers.batch_manager`).
"""

from pathlib import Path
import unicodedata

import pytest

from backend.services.secret_scan_service import TECHNICAL_DISCLAIMER
from backend.workers.batch_manager import (
    STATUS_DONE,
    STATUS_ERROR,
    build_secret_batch_report,
    file_filter_options,
    is_real_pdf,
    normalize_relative_path,
    prepare_batch,
)

PDF_BYTES = b"%PDF-1.7\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


class FakeUpload:
    """Giả lập UploadedFile của Streamlit: tên = đường dẫn tương đối do trình duyệt gửi (webkitRelativePath)."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data
        self.size = len(data)

    def getvalue(self) -> bytes:
        return self._data


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("HO_SO/Ho_so_A/01.pdf", "HO_SO/Ho_so_A/01.pdf"),
        ("HO_SO\\Ho_so_A\\Phu_luc\\03.pdf", "HO_SO/Ho_so_A/Phu_luc/03.pdf"),
        ("HO_SO/./Ho_so_B//04.pdf", "HO_SO/Ho_so_B/04.pdf"),
        ("../etc/passwd.pdf", None),
        ("HO_SO/../../evil.pdf", None),
        ("/abs/path.pdf", None),
        ("C:\\Windows\\x.pdf", None),
        ("HO_SO/a\x00b.pdf", None),
        ("", None),
    ],
)
def test_normalize_relative_path_blocks_traversal(raw, expected):
    assert normalize_relative_path(raw) == expected


def test_normalize_relative_path_unicode_nfc():
    nfd = unicodedata.normalize("NFD", "HỒ_SƠ/Tài liệu/Báo cáo.pdf")  # macOS gửi tên dạng NFD
    out = normalize_relative_path(nfd)
    assert out == unicodedata.normalize("NFC", "HỒ_SƠ/Tài liệu/Báo cáo.pdf")


def test_is_real_pdf_checks_signature_not_extension():
    assert is_real_pdf(PDF_BYTES)
    assert is_real_pdf(b"\xef\xbb\xbf  " + PDF_BYTES)  # chữ ký nằm trong 1024 byte đầu
    assert not is_real_pdf(b"Day la file text doi duoi .pdf")
    assert not is_real_pdf(b"")


def test_prepare_batch_collects_nested_pdfs_and_skips_others(tmp_path: Path):
    uploads = [
        FakeUpload("HO_SO/Ho_so_A/01.pdf", PDF_BYTES),
        FakeUpload("HO_SO/Ho_so_A/02.pdf", PDF_BYTES + b"a"),
        FakeUpload("HO_SO/Ho_so_A/Phu_luc/03.pdf", PDF_BYTES + b"b"),
        FakeUpload("HO_SO/Ho_so_B/04.pdf", PDF_BYTES + b"c"),
        FakeUpload("HO_SO/Ho_so_B/Tài liệu/05.pdf", PDF_BYTES + b"d"),
        FakeUpload("HO_SO/Ho_so_B/01.pdf", PDF_BYTES + b"e"),  # trùng tên ở thư mục khác
        FakeUpload("HO_SO/README.txt", b"khong phai pdf"),
        FakeUpload("HO_SO/Ho_so_B/gia_mao.pdf", b"text gia mao"),
        FakeUpload("HO_SO/../../evil.pdf", PDF_BYTES),
    ]
    batch = prepare_batch(uploads, uploads_dir=tmp_path)

    rel_paths = [e.rel_path for e in batch.entries]
    assert rel_paths == [
        "HO_SO/Ho_so_A/01.pdf",
        "HO_SO/Ho_so_A/02.pdf",
        "HO_SO/Ho_so_A/Phu_luc/03.pdf",
        "HO_SO/Ho_so_B/01.pdf",
        "HO_SO/Ho_so_B/04.pdf",
        "HO_SO/Ho_so_B/Tài liệu/05.pdf",
    ]
    assert batch.root_name == "HO_SO"
    assert batch.subfolder_count == 4  # Ho_so_A, Ho_so_A/Phu_luc, Ho_so_B, Ho_so_B/Tài liệu
    skipped = dict(batch.skipped)
    assert set(skipped) == {"HO_SO/README.txt", "HO_SO/Ho_so_B/gia_mao.pdf", "HO_SO/../../evil.pdf"}

    # Hai file trùng tên '01.pdf' vẫn là 2 mục riêng, phân biệt bằng đường dẫn tương đối
    dup = [e for e in batch.entries if e.file_name == "01.pdf"]
    assert len(dup) == 2 and dup[0].display_path != dup[1].display_path
    assert len({e.key for e in batch.entries}) == len(batch.entries)

    # Bản sao phía server nằm trong thư mục riêng của đợt quét, đặt tên theo mã băm (không dùng đường dẫn người dùng)
    storage = Path(batch.storage_dir).resolve()
    for e in batch.entries:
        saved = Path(e.saved_path).resolve()
        assert storage in saved.parents
        assert saved.read_bytes()[:5] == b"%PDF-"
    assert not (tmp_path.parent / "evil.pdf").exists()


def test_prepare_batch_natural_sort_regardless_of_browser_order(tmp_path: Path):
    uploads = [
        FakeUpload("HO_SO/Ho_so_B/01.pdf", PDF_BYTES),
        FakeUpload("HO_SO/Ho_so_A/10.pdf", PDF_BYTES + b"1"),
        FakeUpload("HO_SO/Ho_so_A/2.pdf", PDF_BYTES + b"2"),
    ]
    batch = prepare_batch(uploads, uploads_dir=tmp_path)
    assert [e.rel_path for e in batch.entries] == [
        "HO_SO/Ho_so_A/2.pdf",
        "HO_SO/Ho_so_A/10.pdf",
        "HO_SO/Ho_so_B/01.pdf",
    ]


def test_prepare_batch_empty_and_no_pdf(tmp_path: Path):
    assert prepare_batch([], uploads_dir=tmp_path).entries == []
    batch = prepare_batch([FakeUpload("HO_SO/ghi_chu.txt", b"x")], uploads_dir=tmp_path)
    assert batch.entries == [] and len(batch.skipped) == 1


def test_file_filter_options_show_processing_status(tmp_path: Path):
    batch = prepare_batch(
        [FakeUpload("HO_SO/a.pdf", PDF_BYTES), FakeUpload("HO_SO/sub/b.pdf", PDF_BYTES + b"1")],
        uploads_dir=tmp_path,
    )
    batch.entries[0].status = STATUS_DONE
    batch.entries[1].status = STATUS_ERROR
    opts = dict(file_filter_options(batch))
    assert opts["__ALL__"] == "Tất cả tệp (2)"
    assert opts[batch.entries[0].key] == "a.pdf"
    assert opts[batch.entries[1].key] == "sub/b.pdf — Lỗi"


def test_secret_batch_report_traceability(tmp_path: Path):
    batch = prepare_batch(
        [FakeUpload("HO_SO/A/01.pdf", PDF_BYTES), FakeUpload("HO_SO/B/02.pdf", PDF_BYTES + b"x"),
         FakeUpload("HO_SO/README.txt", b"x")],
        uploads_dir=tmp_path,
    )
    batch.entries[0].status, batch.entries[0].page_count = STATUS_DONE, 3
    batch.entries[1].status, batch.entries[1].error = STATUS_ERROR, "InvalidPDFError: hỏng"
    report = build_secret_batch_report(batch, TECHNICAL_DISCLAIMER)

    assert report["mode"] == "folder_batch" and report["root_folder"] == "HO_SO"
    assert report["summary"]["pdf_files"] == 2
    assert report["summary"]["succeeded_files"] == 1 and report["summary"]["failed_files"] == 1
    assert report["summary"]["skipped_files"] == 1
    f0, f1 = report["files"]
    assert f0["relative_path"] == "HO_SO/A/01.pdf" and f0["file_name"] == "01.pdf" and f0["page_count"] == 3
    assert f1["processing_status"] == "Lỗi" and f1["error"] == "InvalidPDFError: hỏng" and f1["report"] is None
    assert report["skipped_files"] == [{"relative_path": "HO_SO/README.txt", "reason": "Không phải file PDF"}]


def test_scan_zip_keeps_structure_and_classifies_files(tmp_path: Path):
    """ZIP kết quả Tab 1: giữ cấu trúc thư mục, PDF đã sửa đúng đường dẫn, file không lỗi giữ nguyên, file lỗi chỉ có trong báo cáo."""
    import io
    import json
    import shutil
    import zipfile

    import fitz

    from ai_engine.core.config import SAMPLES_DIR, DetectorConfig
    from ai_engine.pipelines.document_pipeline import DocumentPipeline
    from backend.workers.batch_export import (
        FILE_CLEAN,
        FILE_FAILED,
        FILE_FIXED,
        FILE_SKIPPED,
        ScanExportItem,
        build_scan_zip_bytes,
    )

    rotated_src = tmp_path / "rot.pdf"
    clean_src = tmp_path / "clean.pdf"
    shutil.copy(SAMPLES_DIR / "06_all_four_orientations.pdf", rotated_src)
    shutil.copy(SAMPLES_DIR / "01_all_normal.pdf", clean_src)
    original_bytes = clean_src.read_bytes()

    pipeline = DocumentPipeline(
        config=DetectorConfig(enable_stripe_detection=False, enable_boundary_detection=False)
    )

    def _item(rel: str, src: Path) -> ScanExportItem:
        result = pipeline.scan_document(src)
        problem = [d.page_number for d in result.decisions if d.status != "NORMAL"]
        return ScanExportItem(
            rel_path=rel, file_name=rel.split("/")[-1], status="DONE", error=None,
            saved_path=str(src), file_tag=rel.replace("/", "_"), page_count=result.doc_info.page_count,
            result=result, problem_pages=problem, rotation_pages=problem,
        )

    items = [
        _item("HO_SO/A/Phu_luc/03.pdf", rotated_src),
        _item("HO_SO/B/Tài liệu/01.pdf", clean_src),
        ScanExportItem(
            rel_path="HO_SO/B/04_hong.pdf", file_name="04_hong.pdf", status="ERROR",
            error="InvalidPDFError: hỏng", saved_path="", file_tag="x", page_count=0,
        ),
    ]
    data = build_scan_zip_bytes("HO_SO", items, [("HO_SO/README.txt", "Không phải file PDF")])

    zf = zipfile.ZipFile(io.BytesIO(data))
    names = set(zf.namelist())
    assert {"HO_SO/A/Phu_luc/03.pdf", "HO_SO/B/Tài liệu/01.pdf", "BAO_CAO_KET_QUA.json",
            "BAO_CAO_KET_QUA.csv", "DOC_TRUOC.txt"} <= names
    assert "HO_SO/B/04_hong.pdf" not in names and "HO_SO/README.txt" not in names

    rotations = [p.rotation for p in fitz.open(stream=zf.read("HO_SO/A/Phu_luc/03.pdf"), filetype="pdf")]
    assert rotations == [0, 90, 180, 270]
    assert zf.read("HO_SO/B/Tài liệu/01.pdf") == original_bytes  # file không lỗi giữ nguyên
    assert clean_src.read_bytes() == original_bytes  # không sửa file nguồn

    report = json.loads(zf.read("BAO_CAO_KET_QUA.json"))
    status = {f["relative_path"]: f["status"] for f in report["files"]}
    assert status == {
        "HO_SO/A/Phu_luc/03.pdf": FILE_FIXED,
        "HO_SO/B/Tài liệu/01.pdf": FILE_CLEAN,
        "HO_SO/B/04_hong.pdf": FILE_FAILED,
        "HO_SO/README.txt": FILE_SKIPPED,
    }
    fixed = next(f for f in report["files"] if f["relative_path"] == "HO_SO/A/Phu_luc/03.pdf")
    assert fixed["fixed_pages"] == [2, 3, 4]
    assert report["summary"]["fixed_files"] == 1 and report["summary"]["failed_files"] == 1
