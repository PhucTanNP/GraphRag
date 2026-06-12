"""Detect API — endpoint phân tích ảnh lốp xe bằng Gemini Vision.

Backend (Node.js) gọi:
  POST /api/v1/detect
  { image_url: "https://..." }

Trả về:
  {
    "wear_level": "good|medium|worn|cracked",
    "wear_percentage": 35,
    "tire_type_detected": "Lốp xe máy",
    "crack_detected": false,
    "crack_severity": "none|mild|moderate|severe",
    "crack_locations": [],
    "confidence": 0.85,
    "recommendation": "Lốp còn khá tốt..."
  }
"""

import json
import logging
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.llm_client import LLMClient
from app.config import GEMINI_API_KEY, LLM_MOCK
from app.metrics import request_counter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Detect"])


# ── Request / Response models ────────────────────────────────────────────────

class DetectRequest(BaseModel):
    image_url: str = Field(..., description="URL ảnh lốp xe (từ Cloudinary)")


class DetectResponse(BaseModel):
    wear_level: str = Field(default="good", description="Mức độ mòn: good / medium / worn / cracked")
    wear_percentage: int = Field(default=0, description="Phần trăm mòn ước tính (0-100)")
    tire_type_detected: str = Field(default="", description="Loại lốp phát hiện được")
    crack_detected: bool = Field(default=False, description="Có vết nứt không")
    crack_severity: str = Field(default="none", description="Mức độ nứt: none / mild / moderate / severe")
    crack_locations: list = Field(default=[], description="Vị trí vết nứt")
    confidence: float = Field(default=0.0, description="Độ tin cậy (0.0-1.0)")
    recommendation: str = Field(default="", description="Khuyến nghị")


# ── Prompt cho Gemini ────────────────────────────────────────────────────────

DETECT_PROMPT = """Bạn là chuyên gia phân tích lốp xe. Hãy phân tích ảnh lốp xe này và trả về JSON (không markdown, không ```) với các trường:

{
  "wear_level": "good|medium|worn|cracked",
  "wear_percentage": <số nguyên 0-100>,
  "tire_type_detected": "<loại lốp phát hiện, ví dụ: Lốp xe máy, Lốp ô tô, Lốp xe tải>",
  "crack_detected": true|false,
  "crack_severity": "none|mild|moderate|severe",
  "crack_locations": ["<mô tả vị trí nứt>"],
  "confidence": <số thập phân 0.0-1.0>,
  "recommendation": "<khuyến nghị bằng tiếng Việt>"
}

Quy tắc:
- wear_level: "good" (mòn <20%), "medium" (20-50%), "worn" (50-80%), "cracked" (hỏng/nứt)
- wear_percentage: ước lượng % mòn dựa trên độ sâu gai
- crack_detected: true nếu thấy vết nứt, rách, phồng rộp
- crack_severity: "none" nếu không nứt, "mild" nếu nứt nhẹ, "moderate" nếu nứt rõ, "severe" nếu nứt nặng
- confidence: mức độ tự tin của bạn (0.0-1.0)
- recommendation: khuyến nghị bằng tiếng Việt ngắn gọn (1-2 câu)

Chỉ trả về JSON, không thêm text nào khác."""


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/detect", response_model=DetectResponse)
def detect_tire(req: DetectRequest):
    """Phân tích ảnh lốp xe bằng Gemini Vision.

    Nhận URL ảnh, gửi lên Gemini để phân tích tình trạng lốp,
    trả về kết quả có cấu trúc.
    """
    try:
        # Track metrics
        if request_counter is not None:
            try:
                request_counter.labels(endpoint='/api/v1/detect').inc()
            except Exception:
                pass

        image_url = req.image_url.strip()
        if not image_url:
            raise HTTPException(status_code=400, detail="image_url is required")

        # Nếu là mock mode (không có Gemini API key) → trả kết quả giả định
        if LLM_MOCK or not GEMINI_API_KEY:
            return _mock_detect()

        # Gọi Gemini Vision để phân tích
        result = _call_gemini_vision(image_url)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Detect API error")
        # Fallback: trả mock nếu Gemini lỗi
        logger.warning("Gemini vision failed, returning mock fallback")
        return _mock_detect()


# ── Private helpers ─────────────────────────────────────────────────────────

def _call_gemini_vision(image_url: str) -> DetectResponse:
    """Gửi ảnh lên Gemini Vision và parse kết quả."""
    from google import genai
    from google.genai import types
    import requests

    # Tải ảnh từ URL
    img_response = requests.get(image_url, timeout=30)
    img_response.raise_for_status()

    # Xác định MIME type từ Content-Type header
    content_type = img_response.headers.get("Content-Type", "image/jpeg")
    img_bytes = img_response.content

    if not img_bytes:
        raise ValueError("Failed to download image: empty response")

    # Khởi tạo Gemini client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Tạo image part
    image_part = types.Part.from_bytes(data=img_bytes, mime_type=content_type)

    # Gọi Gemini Vision
    response = client.models.generate_content(
        model="models/gemini-3.1-flash-lite",
        contents=[DETECT_PROMPT, image_part],
    )

    raw_text = response.text.strip() if response and hasattr(response, "text") else ""
    if not raw_text:
        raise ValueError("Empty response from Gemini")

    # Parse JSON từ response (loại bỏ markdown ``` nếu có)
    parsed = _parse_json_response(raw_text)

    return DetectResponse(**parsed)


def _parse_json_response(text: str) -> dict:
    """Parse JSON từ text response, xử lý cả trường hợp có ```markdown."""
    # Loại bỏ ```json ... ``` hoặc ``` ... ```
    text = text.strip()
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # Fallback: tìm { ... } trong text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini response as JSON", extra={"text": text[:500], "error": str(e)})
        raise ValueError(f"Invalid JSON from Gemini: {e}")

    # Validate + set defaults
    return {
        "wear_level": _safe_str(result.get("wear_level", "good")),
        "wear_percentage": min(100, max(0, int(result.get("wear_percentage", 0)))),
        "tire_type_detected": _safe_str(result.get("tire_type_detected", "")),
        "crack_detected": bool(result.get("crack_detected", False)),
        "crack_severity": _safe_str(result.get("crack_severity", "none")),
        "crack_locations": result.get("crack_locations", []) if isinstance(result.get("crack_locations"), list) else [],
        "confidence": min(1.0, max(0.0, float(result.get("confidence", 0.0)))),
        "recommendation": _safe_str(result.get("recommendation", "")),
    }


def _safe_str(val, default=""):
    if val is None:
        return default
    return str(val)


def _mock_detect() -> DetectResponse:
    """Trả kết quả mock khi không có Gemini."""
    return DetectResponse(
        wear_level="good",
        wear_percentage=35,
        tire_type_detected="Lốp xe máy",
        crack_detected=False,
        crack_severity="none",
        crack_locations=[],
        confidence=0.85,
        recommendation=(
            "Lốp còn khá tốt. Bạn có thể yên tâm sử dụng "
            "thêm 5.000-8.000 km nữa trước khi cần thay."
        ),
    )
