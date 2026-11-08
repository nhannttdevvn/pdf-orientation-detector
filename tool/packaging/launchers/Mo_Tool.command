#!/bin/bash
# macOS: bấm đúp để mở giao diện. Lần đầu: chuột phải -> Open (vì ứng dụng chưa ký số của Apple).
DIR="$(cd "$(dirname "$0")" && pwd)"
# Gỡ cờ "tải từ Internet" để macOS không chặn các thư viện đi kèm
xattr -dr com.apple.quarantine "$DIR" 2>/dev/null
"$DIR/KiemTraHoSo" "$@" &
