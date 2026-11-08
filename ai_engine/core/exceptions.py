"""
Hệ thống ngoại lệ (Exceptions) tập trung của ứng dụng (`src/pdf_inspector/core/exceptions.py`).
"""


class PDFOrientationAppError(Exception):
    """Ngoại lệ cơ sở cho toàn bộ ứng dụng PDF Orientation Detector."""


class PDFLoaderError(PDFOrientationAppError):
    """Ngoại lệ cơ sở cho các lỗi khi đọc hoặc tải file PDF."""


class InvalidPDFError(PDFLoaderError):
    """Ngoại lệ khi file không tồn tại, rỗng, bị hỏng hoặc không phải định dạng PDF hợp lệ."""


class PasswordProtectedPDFError(PDFLoaderError):
    """Ngoại lệ khi file PDF bị khóa bằng mật khẩu."""


class PDFCorrectionError(PDFOrientationAppError):
    """Ngoại lệ khi quá trình xoay hoặc lưu file PDF chỉnh sửa gặp lỗi."""


class OrientationDetectionError(PDFOrientationAppError):
    """Ngoại lệ khi bộ nhận diện hướng gặp lỗi cấu hình hoặc hệ thống."""

