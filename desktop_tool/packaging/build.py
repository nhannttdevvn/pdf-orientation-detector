#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build bản chạy độc lập của tool (người dùng cuối KHÔNG cần cài Python / thư viện / Tesseract).
Chạy trên đúng hệ điều hành muốn build (PyInstaller không build chéo được):

    python -m pip install -r packaging/requirements-build.txt
    python packaging/build.py

Kết quả: packaging/_build/KiemTraHoSo-<hệ điều hành>-<kiến trúc>.zip

Tesseract:
- Windows: tự đóng gói Tesseract từ biến môi trường TESSERACT_DIR hoặc C:\\Program Files\\Tesseract-OCR
  (chỉ lấy file cần thiết: exe, dll, tessdata/osd + eng + configs).
- macOS / Linux: không đóng gói Tesseract; tool dùng Tesseract của máy nếu có, nếu không dùng RapidOCR (có sẵn trong gói).
"""

import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Có thể đặt KTHS_BUILD_DIR để build trên ổ khác (nên dùng ổ APFS/NTFS/ext4: ổ exFAT không hỗ trợ symlink
# và macOS sinh thêm file rác ._* trên exFAT).
BUILD = os.path.abspath(os.environ.get("KTHS_BUILD_DIR") or os.path.join(HERE, "_build"))
APP = "KiemTraHoSo"
JUNK_NAMES = (".DS_Store", "Thumbs.db")


def make_zip(zip_path, base_dir, folder):
    """Nén thư mục, giữ quyền thực thi & symlink, bỏ file rác của hệ điều hành (._*, .DS_Store)."""
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(os.path.join(base_dir, folder)):
            dirnames[:] = [d for d in dirnames if not d.startswith("._")]
            rel_dir = os.path.relpath(dirpath, base_dir)
            zf.write(dirpath, rel_dir)
            for name in filenames:
                if name.startswith("._") or name in JUNK_NAMES:
                    continue
                full = os.path.join(dirpath, name)
                arc = os.path.join(rel_dir, name)
                if os.path.islink(full):
                    info = zipfile.ZipInfo(arc)
                    info.create_system = 3
                    info.external_attr = 0o120777 << 16
                    zf.writestr(info, os.readlink(full))
                else:
                    zf.write(full, arc)


def os_tag():
    system = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}.get(platform.system(), platform.system().lower())
    machine = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{arch}"


def stage_tesseract_windows():
    """Chép phần tối thiểu của Tesseract vào _build/Tesseract-OCR. Trả về đường dẫn hoặc '' nếu không có."""
    src = os.environ.get("TESSERACT_DIR") or r"C:\Program Files\Tesseract-OCR"
    if not os.path.isfile(os.path.join(src, "tesseract.exe")):
        print(f"[!] Không thấy Tesseract tại {src} -> bản build sẽ dùng RapidOCR để kiểm tra trang xoay.")
        return ""
    dst = os.path.join(BUILD, "Tesseract-OCR")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.join(dst, "tessdata"))
    for name in os.listdir(src):
        if name.lower().endswith((".exe", ".dll")):
            shutil.copy2(os.path.join(src, name), dst)
    for lang in ("osd", "eng"):
        f = os.path.join(src, "tessdata", f"{lang}.traineddata")
        if not os.path.isfile(f):
            raise SystemExit(f"[LỖI] Tesseract thiếu {f} (bắt buộc để kiểm tra trang xoay).")
        shutil.copy2(f, os.path.join(dst, "tessdata"))
    configs = os.path.join(src, "tessdata", "configs")
    if os.path.isdir(configs):
        shutil.copytree(configs, os.path.join(dst, "tessdata", "configs"))
    for doc in ("LICENSE", "COPYING"):
        if os.path.isfile(os.path.join(src, doc)):
            shutil.copy2(os.path.join(src, doc), os.path.join(dst, f"{doc}.txt"))
    print(f"[*] Đã đóng gói Tesseract từ {src}")
    return dst


def main():
    os.makedirs(BUILD, exist_ok=True)
    env = dict(os.environ)
    env["KTHS_TESSERACT_STAGE"] = stage_tesseract_windows() if sys.platform.startswith("win") else ""

    subprocess.check_call([
        sys.executable, "-m", "PyInstaller", os.path.join(HERE, f"{APP}.spec"),
        "--noconfirm", "--clean",
        "--distpath", os.path.join(BUILD, "dist"),
        "--workpath", os.path.join(BUILD, "work"),
    ], env=env)

    dist = os.path.join(BUILD, "dist")
    zip_path = os.path.join(BUILD, f"{APP}-{os_tag()}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    if sys.platform == "darwin":
        # macOS: phát hành ứng dụng KiemTraHoSo.app (zip + dmg kéo thả vào Applications)
        stage = os.path.join(BUILD, "mac_stage")
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        subprocess.check_call(["ditto", os.path.join(dist, f"{APP}.app"), os.path.join(stage, f"{APP}.app")])
        shutil.copy2(os.path.join(HERE, "HUONG_DAN.txt"), stage)
        subprocess.check_call(["ditto", "-c", "-k", "--keepParent", "--norsrc", stage, zip_path])
        os.symlink("/Applications", os.path.join(stage, "Applications"))
        dmg_path = os.path.join(BUILD, f"{APP}-{os_tag()}.dmg")
        if os.path.exists(dmg_path):
            os.remove(dmg_path)
        subprocess.check_call(["hdiutil", "create", "-volname", "Kiem Tra Ho So", "-srcfolder", stage,
                               "-ov", "-format", "UDZO", dmg_path], stdout=subprocess.DEVNULL)
        print(f"\n[✓] Xong: {zip_path}\n[✓] Xong: {dmg_path}")
        return

    app_dir = os.path.join(dist, APP)
    os.makedirs(os.path.join(app_dir, "data", "pdf"), exist_ok=True)
    os.makedirs(os.path.join(app_dir, "reports"), exist_ok=True)
    shutil.copy2(os.path.join(HERE, "HUONG_DAN.txt"), app_dir)
    if not sys.platform.startswith("win"):
        launcher = os.path.join(app_dir, "mo_tool.sh")
        shutil.copy2(os.path.join(HERE, "launchers", "mo_tool.sh"), launcher)
        os.chmod(launcher, 0o755)
    make_zip(zip_path, dist, APP)
    print(f"\n[✓] Xong: {zip_path}")


if __name__ == "__main__":
    main()
