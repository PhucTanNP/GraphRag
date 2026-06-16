"""
Tire Recognition API Routes
"""
import os
import cv2
from datetime import datetime

import time
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    TireRecognitionRequest,
    TireRecognitionResponse,
    Step4RecognitionResult
)
from app.services import (
    TireSegmentationService,
    TireUnwrapService,
    ImagePreprocessingService,
    TireRecognitionService
)
from app.utils import decode_base64_to_image, encode_image_to_base64, Neo4jService
from app.logger import logger

router = APIRouter(prefix="/api/v1/tire", tags=["tire-recognition"])

# Initialize services
seg_service = None
unwrap_service = None
preprocess_service = None
recognition_service = None
neo4j_service = None


def init_services():
    global seg_service, unwrap_service, preprocess_service, recognition_service, neo4j_service

    if seg_service is None:
        seg_service = TireSegmentationService()

    if unwrap_service is None:
        unwrap_service = TireUnwrapService()

    if preprocess_service is None:
        preprocess_service = ImagePreprocessingService()

    if recognition_service is None:
        recognition_service = TireRecognitionService()

    if neo4j_service is None:
        neo4j_service = Neo4jService()




@router.post("/recognize", response_model=TireRecognitionResponse)
async def recognize_tire(request: TireRecognitionRequest):
    """
    Full tire recognition pipeline
    
    Steps:
    1. Decode image
    2. Segment tire (YOLO11-seg)
    3. Unwrap & correct distortion
    4. Preprocess (CLAHE)
    5. Recognize attributes (YOLO11 + OCR)
    6. Query Neo4j
    
    Request body:
    ```json
    {
        "image_base64": "base64_encoded_image"
    }
    ```
    
    Returns tire brand, size, pattern
    """
    if recognition_service is None:
        init_services()
    
    start_time = time.time()
    
    try:
        if not recognition_service:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        logger.info("Starting tire recognition pipeline...")
        
        # ========== DECODE IMAGE ==========
        success, image = decode_base64_to_image(request.image_base64)
        if not success or image is None:
            logger.error("Failed to decode input image")
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        logger.info(f"Image decoded successfully: {image.shape}")
        
        # ========== STEP 1: SEGMENTATION ==========
        logger.info("Step 1: Tire segmentation...")
        success, tire_mask, seg_meta = seg_service.segment_tire(image)
        if not success or tire_mask is None:
            logger.warning("Segmentation failed")
            raise HTTPException(status_code=400, detail=f"Tire segmentation failed: {seg_meta.get('error', 'unknown error')}")
        
        center = tuple(seg_meta["center"])
        r_outer = seg_meta["outer_radius"]
        r_inner = seg_meta["inner_radius"]
        
        logger.info(f"Segmentation complete: center={center}, r_outer={r_outer}, r_inner={r_inner}")
        
        # ========== STEP 2: UNWRAP & DISTORTION CORRECTION ==========
        logger.info("Step 2: Tire unwrap & distortion correction...")
        success, unwrapped, unwrap_meta = unwrap_service.process_unwrap_pipeline(
            image, tire_mask, center, r_outer, r_inner
        )
        if not success or unwrapped is None:
            logger.warning("Unwrap pipeline failed")
            raise HTTPException(status_code=400, detail="Unwrap pipeline failed")
        
        logger.info(f"Unwrap complete: {unwrapped.shape}")
        
        # ========== STEP 3: PREPROCESSING ==========
        logger.info("Step 3: Image preprocessing...")
        success, preprocessed, preprocess_meta = preprocess_service.process_preprocessing_pipeline(unwrapped)
        if not success or preprocessed is None:
            logger.warning("Preprocessing failed")
            raise HTTPException(status_code=400, detail="Preprocessing failed")
        
        logger.info(f"Preprocessing complete: {preprocessed.shape}")
        
        # ========== STEP 4: RECOGNITION & OCR ==========
        logger.info("Step 4: Tire recognition & OCR...")
        success, recognition_results, recognition_meta = recognition_service.process_recognition_pipeline(preprocessed)
        if not success:
            logger.warning("Recognition pipeline failed")
            raise HTTPException(status_code=400, detail="Recognition pipeline failed")
        
        logger.info(f"Recognition complete: {recognition_results}")
        
        # ========== QUERY NEO4J ==========
        logger.info("Querying Neo4j database...")
        tire_product = neo4j_service.find_tire_by_attributes(
            brand=recognition_results.get("brand"),
            size=recognition_results.get("size"),
            pattern=recognition_results.get("pattern")
        )
        
        if not tire_product:
            logger.warning("No tire product found in database, trying fuzzy match...")
            tire_product = neo4j_service.find_tire_fuzzy(
                brand=recognition_results.get("brand"),
                size=recognition_results.get("size"),
                pattern=recognition_results.get("pattern")
            )
        
        # ========== PREPARE RESPONSE ==========
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        result = Step4RecognitionResult(
            success=True,
            brand=recognition_results.get("brand"),
            size=recognition_results.get("size"),
            pattern=recognition_results.get("pattern"),
            brand_ocr=recognition_results.get("brand_ocr"),
            size_ocr=recognition_results.get("size_ocr"),
            pattern_ocr=recognition_results.get("pattern_ocr"),
            detections_count=recognition_results.get("detections_count", 0),
            message=f"Tire recognized successfully. Database match: {tire_product is not None}"
        )
        
        response = TireRecognitionResponse(
            success=True,
            data=result,
            processing_time_ms=processing_time
        )
        
        logger.info(f"Pipeline completed in {processing_time:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in recognition pipeline: {e}", exc_info=True)
        processing_time = (time.time() - start_time) * 1000
        
        return TireRecognitionResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "services": {
            "segmentation": seg_service is not None,
            "unwrap": unwrap_service is not None,
            "preprocessing": preprocess_service is not None,
            "recognition": recognition_service is not None,
            "neo4j": neo4j_service is not None
        }
    }

DEBUG_DIR = "debug_outputs"
os.makedirs(DEBUG_DIR, exist_ok=True)

def save_debug(name, img):
    ts = datetime.now().strftime("%H%M%S_%f")
    path = os.path.join(DEBUG_DIR, f"{ts}_{name}.jpg")
    cv2.imwrite(path, img)
    return path