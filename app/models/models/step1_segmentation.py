"""
STEP 1: Tire Segmentation Service
Phân đoạn vành lốp từ ảnh gốc bằng YOLO11-seg
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from app.logger import logger
from app.config import settings
from ultralytics import YOLO


class TireSegmentationService:
    """
    Service xử lý STEP 1: Phân đoạn lốp
    - Load YOLO11-seg model
    - Detect và extract vành lốp
    - Filter background noise
    - Return tire mask và center
    """
    
    def __init__(self):
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading YOLO11-seg model from {settings.yolo_seg_model_path}")
            self.model = YOLO(settings.yolo_seg_model_path)
            logger.info("YOLO11-seg model loaded successfully")
    
    def segment_tire(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[dict]]:
        """
        Segment tire from image using YOLO11-seg
        
        Args:
            image: Input image (BGR)
            
        Returns:
            success: Boolean indicating success
            tire_mask: Binary mask of tire area
            metadata: Dict containing center, radius info
        """
        self._load_model()
        
        try:
            h_orig, w_orig = image.shape[:2]
            logger.info(f"Segmenting tire from image size: {w_orig}x{h_orig}")
            
            # Run YOLO11-seg prediction
            results = self.model.predict(
                image,
                conf=settings.yolo_conf_threshold,
                imgsz=640,
                verbose=False
            )[0]
            
            tire_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            
            if results.masks is None or len(results.boxes) == 0:
                logger.warning("YOLO11-seg could not detect any tire")
                return False, None, {"error": "No tire detected"}
            
            # Step 1: Get raw mask from YOLO11
            mask_raw = results.masks.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_raw, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            raw_mask = (mask_resized * 255).astype(np.uint8)
            
            # Step 2: Get bounding box and limit segmentation region
            box = results.boxes[0].xyxy[0].cpu().numpy().astype(np.int32)
            bx1, by1, bx2, by2 = box[0], box[1], box[2], box[3]
            
            box_mask = np.zeros_like(raw_mask)
            cv2.rectangle(box_mask, (bx1, by1), (bx2, by2), 255, -1)
            raw_mask = cv2.bitwise_and(raw_mask, box_mask)
            
            # Step 3: Filter out bright background (concrete/soil)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, dark_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
            raw_mask = cv2.bitwise_and(raw_mask, dark_mask)
            
            # Step 4: Extract tire geometry (center, radii)
            contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                logger.warning("No contours found after filtering")
                return False, None, {"error": "No valid contours after filtering"}
            
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit ellipse or circle
            if len(largest_contour) >= 5:
                (xc, yc), (d1, d2), angle = cv2.fitEllipse(largest_contour)
                cx, cy = int(xc), int(yc)
                r_outer = int(max(d1, d2) / 2)
            else:
                (xc, yc), radius_out = cv2.minEnclosingCircle(largest_contour)
                cx, cy = int(xc), int(yc)
                r_outer = int(radius_out)
            
            # Copy and create inner hole (donut shape)
            tire_mask = raw_mask.copy()
            r_inner = int(r_outer * 0.64)
            cv2.circle(tire_mask, (cx, cy), r_inner, 0, -1)
            
            # Morphological opening to remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tire_mask = cv2.morphologyEx(tire_mask, cv2.MORPH_OPEN, kernel)
            
            metadata = {
                "center": (cx, cy),
                "outer_radius": r_outer,
                "inner_radius": r_inner,
                "image_size": (w_orig, h_orig),
                "mask_area": int(cv2.countNonZero(tire_mask))
            }
            
            logger.info(f"Tire segmented successfully: center=({cx}, {cy}), r_outer={r_outer}, r_inner={r_inner}")
            
            debug_seg = cv2.bitwise_and(image, image, mask=tire_mask)
            cv2.imwrite("debug_outputs/step1_segmented.jpg", debug_seg)
            
            return True, tire_mask, metadata
            
        except Exception as e:
            logger.error(f"Error in tire segmentation: {e}", exc_info=True)
            return False, None, {"error": str(e)}
