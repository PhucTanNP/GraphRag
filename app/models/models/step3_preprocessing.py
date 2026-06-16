"""
STEP 3: Image Preprocessing Service
Tiền xử lý ảnh: cắt, resize, CLAHE cho OCR input
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from app.logger import logger
from app.config import settings


class ImagePreprocessingService:
    """
    Service xử lý STEP 3: Tiền xử lý ảnh
    - Resize ảnh theo kích thước tiêu chuẩn
    - Cắt phần trên dưới bỏ phần rác
    - Apply CLAHE để nâng cao chất lượng cho OCR
    """
    
    def __init__(self):
        """Initialize preprocessing service"""
        logger.info("ImagePreprocessingService initialized")
    
    def normalize_image_size(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], dict]:
        """
        Normalize image size to fit OCR model requirements
        
        Args:
            image: Input image (BGR)
            
        Returns:
            success, resized_image, metadata
        """
        try:
            h, w = image.shape[:2]
            logger.info(f"Normalizing image size: {w}x{h}")
            
            # Resize if too wide
            if w > settings.max_image_width:
                scale = settings.max_image_width / w
                new_h = int(h * scale)
                image = cv2.resize(image, (settings.max_image_width, new_h), interpolation=cv2.INTER_AREA)
                h, w = image.shape[:2]
                logger.info(f"Resized to max width: {w}x{h}")
            
            # Resize if too short
            if h < settings.min_image_height:
                scale = int(np.ceil(settings.min_image_height / h))
                image = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                h, w = image.shape[:2]
                logger.info(f"Upscaled to min height: {w}x{h}")
            
            return True, image, {"original_size": (w, h)}
            
        except Exception as e:
            logger.error(f"Error in image size normalization: {e}", exc_info=True)
            return False, None, {"error": str(e)}
    
    def crop_and_clean(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], dict]:
        """
        Crop top/bottom garbage, keep center tire text area
        
        Args:
            image: Normalized image
            
        Returns:
            success, cropped_image, metadata
        """
        try:
            h, w = image.shape[:2]
            logger.info(f"Cropping image: removing top/bottom garbage")
            
            # Keep 98% of height (remove 1% top, 1% bottom)
            h_crop = int(h * 0.98)
            image_cropped = image[:h_crop, :]
            
            logger.info(f"Cropped size: {w}x{h_crop}")
            
            return True, image_cropped, {"crop_ratio": 0.98}
            
        except Exception as e:
            logger.error(f"Error in crop and clean: {e}", exc_info=True)
            return False, None, {"error": str(e)}
    
    def apply_clahe(self, image):
        try:
            logger.info("Applying CLAHE for OCR optimization")

            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 4))
            result_clahe = clahe.apply(gray)

            # Convert back to 3-channel
            result_clahe_bgr = cv2.cvtColor(result_clahe, cv2.COLOR_GRAY2BGR)

            logger.info("CLAHE applied successfully")

            return True, result_clahe_bgr, {
                "method": "CLAHE",
                "clipLimit": 6.0,
                "tileGridSize": (8, 4)
            }

        except Exception as e:
            logger.error(f"Error in CLAHE application: {e}", exc_info=True)
            return False, None, {"error": str(e)}
    
    def process_preprocessing_pipeline(
        self,
        image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], dict]:
        """
        Full preprocessing pipeline:
        1. Normalize size
        2. Crop and clean
        3. Apply CLAHE
        4. Upscale
        
        Args:
            image: Input image
            
        Returns:
            success, final_image, combined_metadata
        """
        pipeline_meta = {}
        
        # Step 1: Normalize size
        success, image, meta = self.normalize_image_size(image)
        if not success:
            return False, None, meta
        pipeline_meta.update(meta)
        
        # Step 2: Crop and clean
        success, image, meta = self.crop_and_clean(image)
        if not success:
            return False, None, meta
        pipeline_meta.update(meta)
        
        # Step 3: Apply CLAHE
        success, image, meta = self.apply_clahe(image)
        if not success:
            return False, None, meta
        pipeline_meta.update(meta)
        
        # Step 4: Upscale
        pipeline_meta["upscale_skipped"] = True
        if not success:
            return False, None, meta
        pipeline_meta.update(meta)
        
        logger.info("Step 3 preprocessing pipeline completed successfully")
        
        cv2.imwrite("debug_outputs/step3_preprocessed.jpg", image)
        
        return True, image, pipeline_meta
