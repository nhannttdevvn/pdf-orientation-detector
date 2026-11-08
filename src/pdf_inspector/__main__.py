"""
Cho phép chạy CLI bằng `python -m pdf_inspector` (chuyển tiếp sang `pdf_inspector.cli:run_cli`).
"""

import sys

from pdf_inspector.cli import run_cli

if __name__ == "__main__":
    sys.exit(run_cli())
