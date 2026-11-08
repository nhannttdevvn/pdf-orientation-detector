"""
Module Orientation Detector (`src/pdf_inspector/detectors/orientation/detector.py`).
Sử dụng Tesseract OSD (`--psm 0`) kết hợp tiền xử lý OpenCV và trích xuất dải dòng chữ
(`_run_strip_ocr_verification`) để phát hiện hướng văn bản trên ảnh trang PDF.

Tích hợp đầy đủ vào kiến trúc Multi-Defect PDF Quality Inspection:
- Kế thừa `BaseOrientationDetector` (`PageLevelDetector`).
- Tái sử dụng `grayscale_image` và `processed_image` từ `PageContext` dùng chung.
- Bảo toàn 100% thuật toán nhận diện hướng, ngưỡng confidence và quy tắc an toàn REVIEW.
"""

import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytesseract
from pytesseract import Output, TesseractError, TesseractNotFoundError

from pdf_inspector.core.config import (
    DETECTION_UNKNOWN,
    MAX_OCR_DIMENSION,
    MEDIAN_BLUR_KSIZE,
    MIN_OCR_DIMENSION,
    ORIENTATION_CONF_THRESHOLD,
    TESSERACT_OSD_CONFIG,
    find_tesseract_binary,
)
from pdf_inspector.core.logging_config import setup_logger
from pdf_inspector.detectors.base_detector import BaseOrientationDetector
from pdf_inspector.models.context import PageContext
from pdf_inspector.models.schemas import (
    DefectResult,
    DefectStatus,
    DefectType,
    OSDAttemptResult,
    OrientationDetectionResult,
)
from pdf_inspector.processing.image_preprocessor import (
    extract_text_line_canvas,
    prepare_original_grayscale,
    preprocess_for_ocr,
)
from pdf_inspector.utils.file_utils import normalize_angle

logger = setup_logger(__name__)


def parse_osd_text(raw_osd: str) -> Dict[str, Any]:
    """
    Phân tích chuỗi kết quả OSD thô từ Tesseract (`--psm 0`).
    Hỗ trợ fallback an toàn nếu `image_to_osd(..., output_type=Output.DICT)` gặp định dạng lạ.
    """
    result: Dict[str, Any] = {}
    if not raw_osd or not raw_osd.strip():
        raise ValueError("Kết quả OSD rỗng.")

    patterns = {
        "orientation": r"Orientation in degrees:\s*(\d+)",
        "rotate": r"Rotate:\s*(\d+)",
        "orientation_conf": r"Orientation confidence:\s*([+-]?\d+(?:\.\d+)?)",
        "script": r"Script:\s*([^\r\n]+)",
        "script_conf": r"Script confidence:\s*([+-]?\d+(?:\.\d+)?)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw_osd, flags=re.IGNORECASE)
        if match:
            val_str = match.group(1).strip()
            if key in ("orientation", "rotate"):
                result[key] = int(val_str)
            elif key in ("orientation_conf", "script_conf"):
                result[key] = float(val_str)
            else:
                result[key] = val_str

    if "orientation" not in result or "rotate" not in result:
        raise ValueError(
            f"Không tìm thấy trường orientation/rotate trong OSD: {raw_osd!r}"
        )

    return result


class TesseractOrientationDetector(BaseOrientationDetector):
    """
    Bộ phát hiện hướng trang tài liệu sử dụng Tesseract OSD (`--psm 0`)
    kết hợp phân tích dòng văn bản OpenCV khi trang có bố cục phức tạp (A3 2-up, gạch chân, ít chữ).
    """

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        conf_threshold: float = ORIENTATION_CONF_THRESHOLD,
        retry_with_preprocessed: bool = True,
        max_ocr_dimension: int = MAX_OCR_DIMENSION,
        min_ocr_dimension: int = MIN_OCR_DIMENSION,
        median_blur_ksize: int = MEDIAN_BLUR_KSIZE,
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.retry_with_preprocessed = bool(retry_with_preprocessed)
        self.max_ocr_dimension = int(max_ocr_dimension)
        self.min_ocr_dimension = int(min_ocr_dimension)
        self.median_blur_ksize = int(median_blur_ksize)

        resolved_cmd = tesseract_cmd or find_tesseract_binary()
        if resolved_cmd:
            pytesseract.pytesseract.tesseract_cmd = resolved_cmd
        self.tesseract_cmd = resolved_cmd

    def _run_single_osd(self, image: np.ndarray, method_name: str) -> OSDAttemptResult:
        """
        Thực thi Tesseract OSD (`--psm 0`) trên một ảnh đơn lẻ và chuẩn hóa kết quả.
        Không bao giờ tự suy đoán trang là bình thường (0°) nếu OCR lỗi.
        """
        if image is None or image.size == 0:
            return OSDAttemptResult(
                method=method_name,
                status=DETECTION_UNKNOWN,
                error="Ảnh đầu vào rỗng.",
            )

        std_val = float(np.std(image))
        if std_val < 1.5:
            return OSDAttemptResult(
                method=method_name,
                status=DETECTION_UNKNOWN,
                error=f"Trang trắng hoặc độ tương phản quá thấp (std={std_val:.2f}).",
            )

        try:
            osd_data = pytesseract.image_to_osd(
                image,
                config=TESSERACT_OSD_CONFIG,
                output_type=Output.DICT,
            )
            if not isinstance(osd_data, dict) or "rotate" not in osd_data:
                raw_text = pytesseract.image_to_osd(
                    image,
                    config=TESSERACT_OSD_CONFIG,
                    output_type=Output.STRING,
                )
                osd_data = parse_osd_text(raw_text)

            raw_orientation = osd_data.get("orientation")
            raw_rotate = osd_data.get("rotate")
            if raw_orientation is None or raw_rotate is None:
                return OSDAttemptResult(
                    method=method_name,
                    status=DETECTION_UNKNOWN,
                    error="Tesseract OSD không trả về giá trị orientation hoặc rotate.",
                )

            norm_orientation = normalize_angle(int(raw_orientation))
            norm_rotate = normalize_angle(int(raw_rotate))

            raw_orient_conf = osd_data.get("orientation_conf", 0.0)
            orient_conf = (
                round(float(raw_orient_conf), 2)
                if raw_orient_conf is not None
                else 0.0
            )

            script_val = osd_data.get("script")
            script_str = str(script_val).strip() if script_val is not None else None

            raw_script_conf = osd_data.get("script_conf")
            script_conf = (
                round(float(raw_script_conf), 2)
                if raw_script_conf is not None
                else None
            )

            return OSDAttemptResult(
                method=method_name,
                status="OK",
                orientation=norm_orientation,
                rotate=norm_rotate,
                orientation_conf=orient_conf,
                script=script_str,
                script_conf=script_conf,
                error=None,
            )

        except TesseractNotFoundError as exc:
            err_msg = (
                "Không tìm thấy bộ thực thi Tesseract OCR trên hệ thống. "
                f"Chi tiết: {exc}"
            )
            logger.error(err_msg)
            return OSDAttemptResult(
                method=method_name,
                status=DETECTION_UNKNOWN,
                error=err_msg,
            )
        except TesseractError as exc:
            clean_err = str(exc).strip().replace("\n", " ")
            logger.warning("Tesseract OSD thất bại (%s): %s", method_name, clean_err)
            return OSDAttemptResult(
                method=method_name,
                status=DETECTION_UNKNOWN,
                error=f"Tesseract OSD không thể nhận diện: {clean_err}",
            )
        except Exception as exc:
            clean_err = str(exc).strip().replace("\n", " ")
            logger.warning("Lỗi khi chạy OSD (%s): %s", method_name, clean_err)
            return OSDAttemptResult(
                method=method_name,
                status=DETECTION_UNKNOWN,
                error=f"Lỗi xử lý OSD: {clean_err}",
            )

    def _run_strip_ocr_verification(
        self,
        image: np.ndarray,
        method_name: str = "tesseract_osd_preprocessed",
    ) -> OSDAttemptResult:
        """
        Phương pháp tiền xử lý chuyên sâu (trích xuất dải dòng chữ bằng OpenCV + kiểm định 2 chiều của trục chữ)
        dành cho các trang scan A3 2-up chữ nhỏ, trang có nhiều đường gạch chân/strikethrough
        hoặc trang tiêu đề ít chữ khiến `--psm 0` trên toàn trang bị nhiễu.
        Tuyệt đối giữ nguyên tắc an toàn: nếu số từ nhận diện được < 3 hoặc chênh lệch giữa 2 chiều
        không rõ ràng thì trả về UNKNOWN để đưa vào REVIEW.
        """
        try:
            canvas, candidates = extract_text_line_canvas(image, max_strips=10)
            if canvas is None or canvas.size == 0:
                return OSDAttemptResult(
                    method=method_name,
                    status=DETECTION_UNKNOWN,
                    error="Trang không có đủ dòng văn bản để xác định hướng.",
                )

            scores: Dict[int, Dict[str, float]] = {}
            for rot in candidates:
                if rot == 0:
                    test_img = canvas
                elif rot == 90:
                    test_img = np.rot90(canvas, k=-1)
                elif rot == 180:
                    test_img = np.rot90(canvas, k=2)
                elif rot == 270:
                    test_img = np.rot90(canvas, k=1)
                else:
                    continue

                upscaled = cv2.resize(
                    test_img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC
                )
                data = pytesseract.image_to_data(
                    upscaled, config="--psm 6", output_type=Output.DICT
                )
                valid_confs = [
                    float(c)
                    for c, txt in zip(data.get("conf", []), data.get("text", []))
                    if float(c) > 0
                    and len(str(txt).strip()) >= 2
                    and any(ch.isalpha() for ch in str(txt))
                ]
                word_count = len(valid_confs)
                avg_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
                total_conf = float(np.sum(valid_confs)) if valid_confs else 0.0
                scores[rot] = {
                    "count": float(word_count),
                    "avg_conf": avg_conf,
                    "total_conf": total_conf,
                }

            rot_a, rot_b = candidates[0], candidates[1]
            info_a, info_b = scores[rot_a], scores[rot_b]

            if info_a["total_conf"] >= info_b["total_conf"]:
                best_rot, best_info, other_info = rot_a, info_a, info_b
            else:
                best_rot, best_info, other_info = rot_b, info_b, info_a

            if best_info["count"] < 3 or best_info["avg_conf"] < 60.0:
                return OSDAttemptResult(
                    method=method_name,
                    status=DETECTION_UNKNOWN,
                    error=(
                        f"Số lượng từ nhận diện được quá ít ({int(best_info['count'])} từ, "
                        f"avg_conf={best_info['avg_conf']:.1f}%) để kết luận hướng an toàn."
                    ),
                )

            ratio = best_info["total_conf"] / max(1.0, other_info["total_conf"])
            conf_diff = max(0.0, best_info["avg_conf"] - other_info["avg_conf"])
            if ratio < 1.35 or conf_diff < 15.0:
                return OSDAttemptResult(
                    method=method_name,
                    status=DETECTION_UNKNOWN,
                    error="Chênh lệch độ tin cậy giữa 2 chiều đối diện không đủ rõ ràng.",
                )

            equivalent_osd_conf = round(min(35.0, conf_diff * 0.4), 2)
            detected_orientation = (360 - best_rot) % 360

            return OSDAttemptResult(
                method=method_name,
                status="OK",
                orientation=detected_orientation,
                rotate=best_rot,
                orientation_conf=equivalent_osd_conf,
                script="Latin",
                script_conf=round(best_info["avg_conf"], 2),
                error=None,
            )
        except Exception as exc:
            return OSDAttemptResult(
                method=method_name,
                status=DETECTION_UNKNOWN,
                error=f"Lỗi kiểm định dải dòng văn bản: {exc}",
            )

    def detect_page(
        self,
        image: np.ndarray,
        page_number: int = 1,
        existing_pdf_rotation: int = 0,
        conf_threshold: Optional[float] = None,
        grayscale_image: Optional[np.ndarray] = None,
        processed_image: Optional[np.ndarray] = None,
    ) -> OrientationDetectionResult:
        """
        Phát hiện hướng cho một trang PDF:
        1. Chạy OSD trên ảnh grayscale gốc (`tesseract_osd`), ưu tiên tái sử dụng `grayscale_image` từ `PageContext`.
        2. Nếu thất bại (`UNKNOWN`) hoặc có độ tin cậy thấp hơn ngưỡng (`orientation_conf < threshold`),
           tự động thử lại bằng ảnh đã qua tiền xử lý (`tesseract_osd_preprocessed`), ưu tiên tái sử dụng `processed_image`.
        3. So sánh kết quả hai lần thử mà KHÔNG cộng hay lấy trung bình điểm tin cậy.
        """
        threshold = (
            float(conf_threshold) if conf_threshold is not None else self.conf_threshold
        )
        norm_existing_rot = normalize_angle(existing_pdf_rotation)
        attempts: List[OSDAttemptResult] = []

        # Lần thử 1: Ảnh grayscale gốc
        try:
            orig_gray = (
                grayscale_image
                if grayscale_image is not None
                else prepare_original_grayscale(
                    image,
                    max_dim=self.max_ocr_dimension,
                    min_dim=self.min_ocr_dimension,
                )
            )
            attempt1 = self._run_single_osd(orig_gray, method_name="tesseract_osd")
        except Exception as exc:
            attempt1 = OSDAttemptResult(
                method="tesseract_osd",
                status=DETECTION_UNKNOWN,
                error=f"Lỗi chuẩn bị ảnh gốc: {exc}",
            )
        attempts.append(attempt1)

        need_retry = self.retry_with_preprocessed and (
            attempt1.status == DETECTION_UNKNOWN
            or attempt1.orientation_conf is None
            or attempt1.orientation_conf < threshold
        )

        if not need_retry:
            return OrientationDetectionResult(
                page_number=page_number,
                status=attempt1.status,
                orientation=attempt1.orientation,
                rotate=attempt1.rotate,
                orientation_conf=attempt1.orientation_conf,
                script=attempt1.script,
                script_conf=attempt1.script_conf,
                existing_pdf_rotation=norm_existing_rot,
                detection_method=attempt1.method,
                has_conflict=False,
                error=attempt1.error,
                attempts=attempts,
            )

        # Lần thử 2: Ảnh sau tiền xử lý
        logger.info(
            "Trang %d: OSD lần 1 (status=%s, conf=%s) cần thử lại với ảnh tiền xử lý.",
            page_number,
            attempt1.status,
            attempt1.orientation_conf,
        )
        try:
            prep_gray = (
                processed_image
                if processed_image is not None
                else preprocess_for_ocr(
                    image,
                    ksize=self.median_blur_ksize,
                    max_dim=self.max_ocr_dimension,
                    min_dim=self.min_ocr_dimension,
                )
            )
            attempt2 = self._run_single_osd(
                prep_gray, method_name="tesseract_osd_preprocessed"
            )
            if (
                attempt2.status == DETECTION_UNKNOWN
                or attempt2.orientation_conf is None
                or attempt2.orientation_conf < threshold
            ):
                strip_attempt = self._run_strip_ocr_verification(
                    image, method_name="tesseract_osd_preprocessed"
                )
                if strip_attempt.status == "OK":
                    attempt2 = strip_attempt
                elif attempt2.status == DETECTION_UNKNOWN:
                    attempt2 = strip_attempt
        except Exception as exc:
            attempt2 = OSDAttemptResult(
                method="tesseract_osd_preprocessed",
                status=DETECTION_UNKNOWN,
                error=f"Lỗi tiền xử lý ảnh: {exc}",
            )
        attempts.append(attempt2)

        if attempt1.status == DETECTION_UNKNOWN and attempt2.status == DETECTION_UNKNOWN:
            combined_error = (
                attempt1.error or attempt2.error or "OCR không nhận diện được hướng."
            )
            return OrientationDetectionResult(
                page_number=page_number,
                status=DETECTION_UNKNOWN,
                orientation=None,
                rotate=None,
                orientation_conf=None,
                script=None,
                script_conf=None,
                existing_pdf_rotation=norm_existing_rot,
                detection_method=attempt1.method,
                has_conflict=False,
                error=combined_error,
                attempts=attempts,
            )

        if attempt1.status == DETECTION_UNKNOWN and attempt2.status == "OK":
            return OrientationDetectionResult(
                page_number=page_number,
                status="OK",
                orientation=attempt2.orientation,
                rotate=attempt2.rotate,
                orientation_conf=attempt2.orientation_conf,
                script=attempt2.script,
                script_conf=attempt2.script_conf,
                existing_pdf_rotation=norm_existing_rot,
                detection_method=attempt2.method,
                has_conflict=False,
                error=None,
                attempts=attempts,
            )

        if attempt1.status == "OK" and attempt2.status == DETECTION_UNKNOWN:
            return OrientationDetectionResult(
                page_number=page_number,
                status="OK",
                orientation=attempt1.orientation,
                rotate=attempt1.rotate,
                orientation_conf=attempt1.orientation_conf,
                script=attempt1.script,
                script_conf=attempt1.script_conf,
                existing_pdf_rotation=norm_existing_rot,
                detection_method=attempt1.method,
                has_conflict=False,
                error=attempt1.error,
                attempts=attempts,
            )

        # Cả hai lần thử đều trả về "OK"
        if attempt1.rotate != attempt2.rotate:
            conf1 = attempt1.orientation_conf or 0.0
            conf2 = attempt2.orientation_conf or 0.0
            min_meaningful_conf = max(3.0, threshold * 0.5)

            if conf1 < min_meaningful_conf and conf2 >= threshold:
                logger.info(
                    "Trang %d: Bỏ qua kết quả nhiễu lần 1 (rotate=%s°, conf=%.2f) và dùng kết quả tiền xử lý (rotate=%s°, conf=%.2f).",
                    page_number,
                    attempt1.rotate,
                    conf1,
                    attempt2.rotate,
                    conf2,
                )
                return OrientationDetectionResult(
                    page_number=page_number,
                    status="OK",
                    orientation=attempt2.orientation,
                    rotate=attempt2.rotate,
                    orientation_conf=attempt2.orientation_conf,
                    script=attempt2.script,
                    script_conf=attempt2.script_conf,
                    existing_pdf_rotation=norm_existing_rot,
                    detection_method=attempt2.method,
                    has_conflict=False,
                    error=None,
                    attempts=attempts,
                )

            if conf2 < min_meaningful_conf and conf1 >= threshold:
                return OrientationDetectionResult(
                    page_number=page_number,
                    status="OK",
                    orientation=attempt1.orientation,
                    rotate=attempt1.rotate,
                    orientation_conf=attempt1.orientation_conf,
                    script=attempt1.script,
                    script_conf=attempt1.script_conf,
                    existing_pdf_rotation=norm_existing_rot,
                    detection_method=attempt1.method,
                    has_conflict=False,
                    error=None,
                    attempts=attempts,
                )

            conflict_msg = (
                f"Mâu thuẫn kết quả giữa ảnh gốc (rotate={attempt1.rotate}°, "
                f"conf={attempt1.orientation_conf}) và ảnh tiền xử lý "
                f"(rotate={attempt2.rotate}°, conf={attempt2.orientation_conf})."
            )
            logger.warning("Trang %d: %s", page_number, conflict_msg)
            chosen = attempt2 if conf2 > conf1 else attempt1
            return OrientationDetectionResult(
                page_number=page_number,
                status="OK",
                orientation=chosen.orientation,
                rotate=chosen.rotate,
                orientation_conf=chosen.orientation_conf,
                script=chosen.script,
                script_conf=chosen.script_conf,
                existing_pdf_rotation=norm_existing_rot,
                detection_method=chosen.method,
                has_conflict=True,
                error=conflict_msg,
                attempts=attempts,
            )

        chosen = (
            attempt2
            if (attempt2.orientation_conf or 0.0) > (attempt1.orientation_conf or 0.0)
            else attempt1
        )
        return OrientationDetectionResult(
            page_number=page_number,
            status="OK",
            orientation=chosen.orientation,
            rotate=chosen.rotate,
            orientation_conf=chosen.orientation_conf,
            script=chosen.script,
            script_conf=chosen.script_conf,
            existing_pdf_rotation=norm_existing_rot,
            detection_method=chosen.method,
            has_conflict=False,
            error=None,
            attempts=attempts,
        )

    def detect_from_context(self, context: PageContext) -> OrientationDetectionResult:
        """
        Thực thi nhận diện hướng trực tiếp trên `PageContext` dùng chung,
        tận dụng `context.grayscale_image` và `context.processed_image` được lưu đệm.
        """
        # Gọi detect_page; truyền sẵn grayscale_image từ context
        return self.detect_page(
            image=context.original_image,
            page_number=context.page_number,
            existing_pdf_rotation=context.original_rotation,
            conf_threshold=self.conf_threshold,
            grayscale_image=context.grayscale_image,
            processed_image=None,
        )

    def build_defect_result(
        self,
        detection: OrientationDetectionResult,
        conf_threshold: Optional[float] = None,
    ) -> DefectResult:
        """
        Chuyển đổi `OrientationDetectionResult` sang `DefectResult` chuẩn của kiến trúc Multi-Defect.
        Phân biệt rõ 4 trạng thái thực thi:
        - `DETECTOR_FAILED`: OCR trả về UNKNOWN hoặc thiếu thông tin góc/độ tin cậy -> `requires_review = True`.
        - `SUSPECTED_LOW_CONFIDENCE`: Độ tin cậy < ngưỡng hoặc có mâu thuẫn (`has_conflict`) -> `requires_review = True`.
        - `CONFIRMED`: Góc cần xoay thuộc {90°, 180°, 270°} với độ tin cậy >= ngưỡng -> `requires_review = False`.
        - `NO_DEFECT`: Góc cần xoay == 0° với độ tin cậy >= ngưỡng -> `requires_review = False`.
        """
        threshold = (
            float(conf_threshold) if conf_threshold is not None else self.conf_threshold
        )

        is_unknown = (
            detection.status == DETECTION_UNKNOWN
            or detection.rotate is None
            or detection.orientation_conf is None
        )
        if is_unknown:
            return DefectResult(
                defect_type=DefectType.ROTATION.value,
                detector_id=self.detector_id,
                status=DefectStatus.DETECTOR_FAILED.value,
                requires_review=True,
                correction_angle=0,
                detected_rotation=detection.rotate,
                detected_orientation=detection.orientation,
                raw_confidence=detection.orientation_conf,
                confidence_threshold=threshold,
                detection_method=detection.detection_method,
                error=detection.error or "OCR không nhận diện được hướng văn bản.",
                details={"orientation_detection": detection},
            )

        has_conflict = bool(detection.has_conflict)
        is_low_conf = float(detection.orientation_conf or 0.0) < threshold
        if has_conflict or is_low_conf:
            reason = detection.error
            if not reason:
                if has_conflict:
                    reason = "Kết quả giữa các lần thử OCR mâu thuẫn nhau."
                else:
                    reason = (
                        f"Độ tin cậy OCR ({detection.orientation_conf:.2f}) "
                        f"thấp hơn ngưỡng ({threshold:.2f})."
                    )
            return DefectResult(
                defect_type=DefectType.ROTATION.value,
                detector_id=self.detector_id,
                status=DefectStatus.SUSPECTED_LOW_CONFIDENCE.value,
                requires_review=True,
                correction_angle=0,
                detected_rotation=detection.rotate,
                detected_orientation=detection.orientation,
                raw_confidence=detection.orientation_conf,
                confidence_threshold=threshold,
                detection_method=detection.detection_method,
                error=reason,
                details={"orientation_detection": detection},
            )

        rot_angle = normalize_angle(int(detection.rotate))  # type: ignore[arg-type]
        if rot_angle == 0:
            return DefectResult(
                defect_type=DefectType.ROTATION.value,
                detector_id=self.detector_id,
                status=DefectStatus.NO_DEFECT.value,
                requires_review=False,
                correction_angle=0,
                detected_rotation=0,
                detected_orientation=detection.orientation,
                raw_confidence=detection.orientation_conf,
                confidence_threshold=threshold,
                detection_method=detection.detection_method,
                message="Trang đúng chiều (0°).",
                details={"orientation_detection": detection},
            )

        return DefectResult(
            defect_type=DefectType.ROTATION.value,
            detector_id=self.detector_id,
            status=DefectStatus.CONFIRMED.value,
            requires_review=False,
            correction_angle=rot_angle,
            detected_rotation=rot_angle,
            detected_orientation=detection.orientation,
            raw_confidence=detection.orientation_conf,
            confidence_threshold=threshold,
            detection_method=detection.detection_method,
            message=f"Phát hiện trang bị xoay, cần xoay thêm {rot_angle}° theo chiều kim đồng hồ.",
            details={"orientation_detection": detection},
        )

    def detect(self, context: PageContext) -> List[DefectResult]:
        """
        Thực thi giao diện `PageLevelDetector.detect(context: PageContext) -> List[DefectResult]`.
        """
        det_result = self.detect_from_context(context)
        defect = self.build_defect_result(det_result, conf_threshold=self.conf_threshold)
        return [defect]


# Bí danh tương thích ngược
OrientationDetector = TesseractOrientationDetector

