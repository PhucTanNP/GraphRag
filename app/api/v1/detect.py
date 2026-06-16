"""Detect API — endpoint nhận dạng thông số lốp xe từ URL ảnh.

Backend (Node.js) gọi:
  POST /api/v1/detect
  { image_url: "https://..." }

Trả về brand/size/pattern + steps chi tiết từng bước.
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.tire_detector import TireDetector

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Detect"])

# ── Singleton ────────────────────────────────────────────────────────────────
_detector: TireDetector | None = None


def get_detector() -> TireDetector:
    global _detector
    if _detector is None:
        _detector = TireDetector()
    return _detector


# ── Schemas ──────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    image_url: str = Field(..., description="URL ảnh lốp xe (từ Cloudinary)")


class StepDetail(BaseModel):
    step: int
    name: str
    status: str  # "ok" | "error"
    detail: str = ""
    image: str | None = None  # base64 data URI của ảnh step
    crops: list[dict] | None = None  # [{class, image}] cho step 4
    detect_input_image: str | None = None  # base64 ảnh đầu vào YOLO detect (resized 1280px)


class OcrDetail(BaseModel):
    raw_text: str = ""
    normalized_text: str = ""
    ocr_confidence: float = 0.0
    yolo_confidence: float = 0.0
    crop_image: str | None = None  # base64 ảnh crop vùng detect
    ocr_input_image: str | None = None  # base64 ảnh đầu vào PaddleOCR (48×320 đã pad)


class DetectData(BaseModel):
    brand: str | None = None
    size: str | None = None
    pattern: str | None = None
    brand_ocr: OcrDetail | None = None
    size_ocr: OcrDetail | None = None
    pattern_ocr: OcrDetail | None = None
    detections_count: int = 0


class DetectResponse(BaseModel):
    success: bool = False
    data: DetectData | None = None
    steps: list[StepDetail] = []
    error: str | None = None
    processing_time_ms: float = 0.0


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/detect", response_model=DetectResponse)
def detect_tire(req: DetectRequest):
    """Nhận dạng thông số lốp xe từ URL ảnh.

    - Tải ảnh từ URL
    - Chạy pipeline: segment → unwrap → CLAHE → YOLO detect + OCR
    - Trả brand, size, pattern + steps chi tiết
    """
    import time
    start = time.time()

    try:
        if not req.image_url.strip():
            return DetectResponse(success=False, error="image_url is required")

        detector = get_detector()
        result = detector.detect_from_url(req.image_url.strip())

        def ocr(val):
            if not val:
                return None
            return OcrDetail(**val) if isinstance(val, dict) else None

        data = None
        if result.get("success"):
            data = DetectData(
                brand=result.get("brand"),
                size=result.get("size"),
                pattern=result.get("pattern"),
                brand_ocr=ocr(result.get("brand_ocr")),
                size_ocr=ocr(result.get("size_ocr")),
                pattern_ocr=ocr(result.get("pattern_ocr")),
                detections_count=result.get("detections_count", 0),
            )

        steps = [StepDetail(**s) for s in result.get("steps", [])]

        return DetectResponse(
            success=result.get("success", False),
            data=data,
            steps=steps,
            error=result.get("error"),
            processing_time_ms=(time.time() - start) * 1000,
        )

    except Exception as e:
        logger.exception("Detect error")
        return DetectResponse(success=False, error=str(e), processing_time_ms=(time.time() - start) * 1000)


@router.get("/detect/health")
def health():
    det = get_detector()
    return {"status": "ok", "services": det.health()}

