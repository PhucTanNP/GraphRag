"""
STEP 2: Tire Unwrap & Distortion Correction Service
Duỗi thẳng vành lốp từ dạng tròn sang dạng phẳng + khử cong
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from app.logger import logger


class TireUnwrapService:
    """
    Service xử lý STEP 2: Duỗi thẳng lốp
    - Convert từ tọa độ cực (polar) sang Cartesian
    - Khử cong đa thức
    - Output ảnh thẳng phẳng
    """
    
    def __init__(self):
        """Initialize unwrap service"""
        self.unwrap_width = 2268  # Standard width for unwrapped tire
        logger.info("TireUnwrapService initialized")
    
    def unwrap_polar(
        self,
        image: np.ndarray,
        tire_mask: np.ndarray,
        center: Tuple[int, int],
        r_outer: int,
        r_inner: int
    ) -> Tuple[bool, Optional[np.ndarray], Optional[dict]]:
        """
        Unwrap tire from polar coordinates to Cartesian (flat) coordinates
        
        Args:
            image: Input image (BGR)
            tire_mask: Binary mask of tire area
            center: (x, y) center of tire
            r_outer: Outer radius
            r_inner: Inner radius
            
        Returns:
            success: Boolean
            unwrapped_image: Unwrapped image
            metadata: Processing info
        """
        try:
            h_orig, w_orig = image.shape[:2]
            xc, yc = center
            
            out_h = r_outer - r_inner
            out_w = self.unwrap_width
            
            logger.info(f"Unwrapping tire: r_outer={r_outer}, r_inner={r_inner}, output_size={out_w}x{out_h}")
            
            # Create unwrap maps (polar to Cartesian)
            theta_raw = np.linspace(2 * np.pi, 0, out_w, dtype=np.float32)
            theta = theta_raw  # Assume circular (not ellipse)
            
            r_arr = np.linspace(r_inner, r_outer, out_h, dtype=np.float32)
            T, R = np.meshgrid(theta, r_arr)
            
            map_x = np.clip((xc + R * np.cos(T)).astype(np.float32), 0, w_orig - 1)
            map_y = np.clip((yc + R * np.sin(T)).astype(np.float32), 0, h_orig - 1)
            
            # Remap image
            img_unwrapped = cv2.remap(
                image,
                map_x,
                map_y,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Rotate 180 degrees
            img_unwrapped = cv2.rotate(img_unwrapped, cv2.ROTATE_180)
            
            logger.info("Polar unwrap completed")
            
            cv2.imwrite("debug_outputs/step2_unwrap_raw.jpg", img_unwrapped)
            
            return True, img_unwrapped, {"step": "polar_unwrap", "output_size": (out_w, out_h)}
            
        except Exception as e:
            logger.error(f"Error in polar unwrap: {e}", exc_info=True)
            return False, None, {"error": str(e)}
    
    def correct_distortion(
        self,
        image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[dict]]:
        """
        Correct polynomial distortion to flatten the tire
        
        Args:
            image: Unwrapped image (from polar step)
            
        Returns:
            success: Boolean
            corrected_image: Distortion-corrected image
            metadata: Processing info
        """
        try:
            h, w = image.shape[:2]
            logger.info(f"Correcting distortion for image size: {w}x{h}")
            
            # Convert to grayscale
            gray_temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_temp, 10, 255, cv2.THRESH_BINARY)
            
            # Find central line curve
            cols = np.linspace(0, w - 1, 30, dtype=np.int32)
            pts_x = []
            pts_y = []
            
            for x in cols:
                column = thresh[:, x]
                nz = np.where(column > 0)[0]
                if len(nz) > 0:
                    pts_x.append(x)
                    pts_y.append(int(np.mean(nz)))
            
            if len(pts_x) < 3:
                logger.warning("Not enough points for polynomial fitting")
                return True, image, {"step": "distortion_correction", "skipped": True}
            
            # Fit polynomial (degree 3)
            poly_coeffs = np.polyfit(pts_x, pts_y, 3)
            poly_func = np.poly1d(poly_coeffs)
            
            # Create correction map
            map_flat_x, map_flat_y = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32)
            )
            
            mid_y = h / 2.0
            curve_y = poly_func(map_flat_x)
            
            # Apply correction
            map_flat_y = np.clip(map_flat_y + (curve_y - mid_y), 0, h - 1).astype(np.float32)
            
            img_flat = cv2.remap(
                image,
                map_flat_x,
                map_flat_y,
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            logger.info("Distortion correction completed")
            
            cv2.imwrite("debug_outputs/step2_corrected.jpg", img_flat)
            
            return True, img_flat, {"step": "distortion_correction", "poly_degree": 3}
            
        except Exception as e:
            logger.error(f"Error in distortion correction: {e}", exc_info=True)
            return False, None, {"error": str(e)}
    
    def process_unwrap_pipeline(
        self,
        image: np.ndarray,
        tire_mask: np.ndarray,
        center: Tuple[int, int],
        r_outer: int,
        r_inner: int
    ) -> Tuple[bool, Optional[np.ndarray], Optional[dict]]:
        """
        Full pipeline: Unwrap + Distortion Correction
        
        Args:
            image: Input image
            tire_mask: Tire mask
            center: Tire center
            r_outer: Outer radius
            r_inner: Inner radius
            
        Returns:
            success, processed_image, metadata
        """
        # Step 1: Polar unwrap
        success, unwrapped, meta1 = self.unwrap_polar(image, tire_mask, center, r_outer, r_inner)
        if not success or unwrapped is None:
            return False, None, meta1
        
        # Step 2: Distortion correction
        success, corrected, meta2 = self.correct_distortion(unwrapped)
        if not success:
            return False, None, meta2
        
        combined_meta = {**meta1, **meta2}
        logger.info("Step 2 pipeline completed successfully")
        
        return True, corrected, combined_meta
