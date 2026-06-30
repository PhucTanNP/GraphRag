"""Detect API — endpoint nhận dạng thông số lốp xe từ URL ảnh.

Backend (Node.js) gọi:
  POST /api/v1/detect
  { image_url: "https://..." }

  POST /api/v1/detect/recommend
  { vehicle_name: "Vario 125 Click 125i" }

  POST /api/v1/detect/vehicles-by-pattern
  { pattern: "D119" }

  POST /api/v1/detect/sizes-by-vehicle
  { vehicle_name: "Vario 125 Click 125i" }

Trả về brand/size/pattern + steps chi tiết từng bước.
+ Recommend: trả lốp trước/sau theo xe, tất cả brand × pattern.
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.detect import TireDetector
from app.services.neo4j_service import Neo4jService

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


# ═══════════════════════════════════════════════════════════════════════════
#  Recommend — từ tên xe → gợi ý tất cả lốp trước + sau, mọi brand × pattern
#  ═══════════════════════════════════════════════════════════════════════════

class VehicleBySizeRequest(BaseModel):
    size: str = Field(..., description="Size lốp đã detect, vd: 80/90-14")


class VehicleByPatternRequest(BaseModel):
    pattern: str = Field(..., description="Mã gai đã detect, vd: D119 hoặc 119")


class VehicleInfo(BaseModel):
    name: str


class VehicleByPatternResponse(BaseModel):
    success: bool = False
    pattern: str | None = None
    vehicles: list[VehicleInfo] = []
    error: str | None = None


class SizesByVehicleRequest(BaseModel):
    vehicle_name: str = Field(..., description="Tên xe")


class SizesByVehicleResponse(BaseModel):
    success: bool = False
    vehicle_name: str | None = None
    front_size: str | None = None
    rear_size: str | None = None
    error: str | None = None


class VehicleBySizeResponse(BaseModel):
    success: bool = False
    size: str | None = None
    vehicles: list[VehicleInfo] = []
    error: str | None = None


class RecommendRequest(BaseModel):
    vehicle_name: str = Field(..., description="Tên xe, vd: Vario 125 Click 125i")


class RecommendResponse(BaseModel):
    success: bool = False
    vehicle_name: str | None = None
    front_size: str | None = None
    rear_size: str | None = None
    error: str | None = None


@router.post("/detect/vehicles-by-size", response_model=VehicleBySizeResponse)
def vehicles_by_size(req: VehicleBySizeRequest):
    """Tìm tất cả xe có dùng size lốp này (trước hoặc sau)."""
    try:
        size = req.size.strip()
        neo4j = Neo4jService()
        rows = neo4j.query("""
            MATCH (v:Vehicle)-[:DÙNG_LỐP_TRƯỚC|DÙNG_LỐP_SAU]->(t:Tire)
            WHERE t.tire_size = $size
            RETURN DISTINCT v.name AS name
            ORDER BY v.name
        """, {"size": size})
        vehicles = [VehicleInfo(name=r["name"]) for r in rows]
        return VehicleBySizeResponse(success=True, size=size, vehicles=vehicles)
    except Exception as e:
        logger.exception("vehicles_by_size error")
        return VehicleBySizeResponse(success=False, error=str(e))


@router.post("/detect/recommend", response_model=RecommendResponse)
def recommend_by_vehicle(req: RecommendRequest):
    """Gợi ý size lốp trước/sau cho 1 xe — chỉ query Neo4j lấy size.

    Backend (Node.js) sẽ dùng front_size + rear_size để query Supabase
    lấy dữ liệu sản phẩm (giá, brand, pattern benefit,...).
    """
    try:
        neo4j = Neo4jService()
        vname = req.vehicle_name.strip()

        size_rows = neo4j.query("""
            OPTIONAL MATCH (v:Vehicle {name: $name})-[:DÙNG_LỐP_TRƯỚC]->(ft:Tire)
            WITH COLLECT(DISTINCT ft.tire_size) AS front_sizes
            OPTIONAL MATCH (v:Vehicle {name: $name})-[:DÙNG_LỐP_SAU]->(rt:Tire)
            RETURN front_sizes, COLLECT(DISTINCT rt.tire_size) AS rear_sizes
        """, {"name": vname})

        if not size_rows:
            return RecommendResponse(success=False, error=f"Không tìm thấy xe '{vname}'")

        front_sizes: list = size_rows[0].get("front_sizes") or []
        rear_sizes: list = size_rows[0].get("rear_sizes") or []
        all_sizes = list(set(front_sizes + rear_sizes))

        if not all_sizes:
            return RecommendResponse(
                success=False, vehicle_name=vname,
                error=f"Xe '{vname}' chưa có dữ liệu lốp",
            )

        return RecommendResponse(
            success=True,
            vehicle_name=vname,
            front_size=front_sizes[0] if front_sizes else None,
            rear_size=rear_sizes[0] if rear_sizes else None,
        )

    except Exception as e:
        logger.exception("Recommend error")
        return RecommendResponse(success=False, error=str(e))


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline mới (cho Scan FE) — 3 case
#  ═══════════════════════════════════════════════════════════════════════════

@router.post("/detect/vehicles-by-pattern", response_model=VehicleByPatternResponse)
def vehicles_by_pattern(req: VehicleByPatternRequest):
    """CASE 3: Tìm tất cả xe có dùng pattern này."""
    try:
        raw_pattern = req.pattern.strip()
        # Tự động thêm tiền tố D nếu pattern không có D
        pattern = raw_pattern if raw_pattern.upper().startswith("D") else "D" + raw_pattern
        neo4j = Neo4jService()
        rows = neo4j.query("""
            MATCH (v:Vehicle)-[:DÙNG_LỐP_TRƯỚC|DÙNG_LỐP_SAU]->(t:Tire)
            WHERE t.pattern_code = $pattern
            RETURN DISTINCT v.name AS name
            ORDER BY v.name
        """, {"pattern": pattern})
        vehicles = [VehicleInfo(name=r["name"]) for r in rows]
        return VehicleByPatternResponse(success=True, pattern=pattern, vehicles=vehicles)
    except Exception as e:
        logger.exception("vehicles_by_pattern error")
        return VehicleByPatternResponse(success=False, error=str(e))


@router.post("/detect/sizes-by-vehicle", response_model=SizesByVehicleResponse)
def sizes_by_vehicle(req: SizesByVehicleRequest):
    """CASE 3: Lấy front_size + rear_size của 1 xe."""
    try:
        vname = req.vehicle_name.strip()
        neo4j = Neo4jService()
        rows = neo4j.query("""
            OPTIONAL MATCH (v:Vehicle {name: $name})-[:DÙNG_LỐP_TRƯỚC]->(ft:Tire)
            WITH COLLECT(DISTINCT ft.tire_size) AS front_sizes
            OPTIONAL MATCH (v:Vehicle {name: $name})-[:DÙNG_LỐP_SAU]->(rt:Tire)
            RETURN front_sizes, COLLECT(DISTINCT rt.tire_size) AS rear_sizes
        """, {"name": vname})

        if not rows:
            return SizesByVehicleResponse(success=False, error=f"Không tìm thấy xe '{vname}'")

        front_sizes = rows[0].get("front_sizes") or []
        rear_sizes = rows[0].get("rear_sizes") or []
        front_size = front_sizes[0] if front_sizes else None
        rear_size = rear_sizes[0] if rear_sizes else None

        return SizesByVehicleResponse(
            success=True,
            vehicle_name=vname,
            front_size=front_size,
            rear_size=rear_size,
        )
    except Exception as e:
        logger.exception("sizes_by_vehicle error")
        return SizesByVehicleResponse(success=False, error=str(e))

