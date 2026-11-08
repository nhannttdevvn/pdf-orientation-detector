#!/bin/bash
# Linux: chạy ./mo_tool.sh để mở giao diện (hoặc ./KiemTraHoSo_CLI --input data/pdf --output reports)
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/KiemTraHoSo" "$@"
