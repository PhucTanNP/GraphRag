"""TireDetector Pipeline — orchestrator cho 4 bước nhận dạng lốp xe.

Kết nối các module:
  models  → load models
  segment → Step 1: YOLO11-seg
  unwrap  → Step 2: Polar unwrap
  preprocess → Step 3: CLAHE
  ocr     → Step 4: YOLO detect + PaddleOCR
"""

import os
import logging
import traceback

import numpy as np

from app.config import YOLO_CONF_THRESHOLD
from app.detect.models import ModelManager
from app.detect.segment import run_segmentation
from app.detect.unwrap import run_unwrap
from app.detect.preprocess import run_preprocess
from app.detect.ocr import run_ocr, CLASS_MAP
from app.detect.utils import download_image, img_to_b64
import cv2

# Fix OpenMP conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger("tire-detector")


class TireDetector:
    """Nhận dạng thông số lốp xe từ URL ảnh.

    Pipeline:
      1. Segment: YOLO11-seg → mask → center + radius
      2. Unwrap: Polar → ảnh phẳng + chỉnh méo
      3. Preprocess: CLAHE + resize
      4. Detect + OCR: YOLO detect → crop → PaddleOCR → brand/size/pattern

    Usage:
        detector = TireDetector()
        result = detector.detect_from_url("https://...")
    """

    def __init__(self):
        self.models = ModelManager()

    # ── Public API ──────────────────────────────────────────────────────

    def detect_from_url(self, image_url: str) -> dict:
        """Nhận dạng lốp từ URL ảnh (tải từ Cloudinary/HTTP)."""
        self.models.load_all()
        try:
            image = download_image(image_url)
            logger.info("Image loaded from URL: %dx%d", image.shape[1], image.shape[0])
            return self._detect_from_image(image)
        except Exception as e:
            logger.error("Download failed: %s", e)
            return {"success": False, "error": f"Download failed: {e}", "steps": []}

    def detect_from_file(self, file_path: str) -> dict:
        """Nhận dạng lốp từ file ảnh local (dùng khi test không cần Cloudinary)."""
        self.models.load_all()
        try:
            image = cv2.imread(file_path)
            if image is None:
                return {"success": False, "error": f"Cannot read file: {file_path}", "steps": []}
            logger.info("Image loaded from file: %dx%d", image.shape[1], image.shape[0])
            return self._detect_from_image(image)
        except Exception as e:
            logger.error("File read error: %s", e)
            return {"success": False, "error": f"File error: {e}", "steps": []}

    def _detect_from_image(self, image: np.ndarray) -> dict:
        """Pipeline xử lý chung cho 1 ảnh (đã load thành numpy array)."""
        steps = []
        os.makedirs("debug_outputs", exist_ok=True)

        logger.info("─" * 60)
        logger.info("▶ _detect_from_image START — processing image...")
        logger.info("Image size: %dx%d", image.shape[1], image.shape[0])

        try:
            # ══════════════════════════════════════════════════════════
            #  STEP 1: SEGMENT
            # ══════════════════════════════════════════════════════════
            logger.info("Step 1/4: Segmenting tire...")
            seg_result = run_segmentation(image, self.models.seg_model)

            if not seg_result["success"]:
                steps.append({
                    "step": 1, "name": "Segment tire",
                    "status": "error",
                    "detail": seg_result.get("error", "Unknown error"),
                })
                return {"success": False, "error": seg_result.get("error"), "steps": steps}

            steps.append({
                "step": 1, "name": "Segment tire",
                "status": "ok",
                "detail": seg_result["detail"],
                "image": seg_result["image_base64"],
            })

            cx, cy = seg_result["cx"], seg_result["cy"]
            r_outer, r_inner = seg_result["r_outer"], seg_result["r_inner"]
            masked_image = seg_result.get("masked_image", image)  # ảnh đã cô lập lốp

            # ══════════════════════════════════════════════════════════
            #  STEP 2: UNWRAP
            # ══════════════════════════════════════════════════════════
            logger.info("Step 2/4: Unwrapping + distortion correction...")
            unwrap_result = run_unwrap(masked_image, cx, cy, r_outer, r_inner)

            steps.append({
                "step": 2, "name": "Unwrap + distortion correction",
                "status": "ok",
                "detail": unwrap_result["detail"],
                "image": unwrap_result["image_base64"],
            })

            # ══════════════════════════════════════════════════════════
            #  STEP 3: PREPROCESS (CLAHE)
            # ══════════════════════════════════════════════════════════
            logger.info("Step 3/4: Preprocessing (CLAHE)...")
            preprocess_result = run_preprocess(unwrap_result["image"])

            steps.append({
                "step": 3, "name": "CLAHE preprocessing",
                "status": "ok",
                "detail": preprocess_result["detail"],
                "image": preprocess_result["image_base64"],
            })

            # ══════════════════════════════════════════════════════════
            #  STEP 4: YOLO DETECT + OCR
            # ══════════════════════════════════════════════════════════
            logger.info("Step 4/4: YOLO detect + OCR...")
            logger.info("  → Running YOLO detect predict (imgsz=1280)... (may take 10-30s on CPU)")

            results = self.models.detect_model.predict(
                preprocess_result["image_resized"],
                imgsz=1280,
                conf=YOLO_CONF_THRESHOLD,
                iou=0.3,
                verbose=False,
            )[0]

            logger.info("  ✓ YOLO detect predict done")

            # Parse detections
            detections = []
            if results.boxes and len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "class_name": CLASS_MAP.get(int(box.cls[0]), "unknown"),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "yolo_confidence": float(box.conf[0]),
                    })

                # Giữ top 3 confidence
                detections.sort(key=lambda d: d["yolo_confidence"], reverse=True)
                detections = detections[:3]

            # Run OCR
            recognition = run_ocr(
                preprocessed=preprocess_result["image"],
                detections=detections,
                scale_factors=preprocess_result["scale_factors"],
                predictor=self.models.predictor,
                input_names=self.models.input_names,
                output_names=self.models.output_names,
                char_dict=self.models.char_dict,
            )

            # Vẽ kết quả step 4
            step4_image = _draw_detections(
                preprocess_result["image"],
                detections,
                preprocess_result["scale_factors"],
            )
            crops_b64 = self._collect_crops(recognition)

            s4_detail = (
                f"Detected {recognition['detections_count']} regions: "
                f"brand={recognition['brand'] or '?'}, "
                f"size={recognition['size'] or '?'}, "
                f"pattern={recognition['pattern'] or '?'} | "
                f"full_size={preprocess_result['image'].shape[1]}x{preprocess_result['image'].shape[0]}"
            )

            steps.append({
                "step": 4,
                "name": "YOLO detect + OCR",
                "status": "ok",
                "detail": s4_detail,
                "image": step4_image,
                "crops": crops_b64,
                "detect_input_image": img_to_b64(preprocess_result["image_resized"]),
            })

            recognition["success"] = True
            recognition["steps"] = steps
            logger.info("  ✓ Recognition done")
            return recognition

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("TireDetector error: %s", e)
            print(f"🔥🔥🔥 TIRE DETECTOR ERROR 🔥🔥🔥\n{tb}\n{'='*60}")
            return {"success": False, "error": f"{type(e).__name__}: {e}", "steps": steps}

    def health(self) -> dict:
        """Health check."""
        return self.models.health()

    def _collect_crops(self, recognition: dict) -> list:
        """Collect ảnh crop từ kết quả OCR."""
        crops = []
        for cn in ["brand", "size", "pattern"]:
            ocr_data = recognition.get(f"{cn}_ocr")
            if ocr_data and ocr_data.get("crop_image"):
                crops.append({"class": cn, "image": ocr_data["crop_image"]})
        return crops


def _draw_detections(
    image, detections: list, scale_factors: tuple[float, float]
) -> str:
    """Vẽ YOLO detections lên ảnh → base64.

    Args:
        image: Ảnh gốc
        detections: Danh sách detection
        scale_factors: (sf_x, sf_y)

    Returns:
        base64 data URI
    """
    from app.detect.utils import img_to_b64, CLASS_COLORS
    import cv2

    sf_x, sf_y = scale_factors
    img_s4 = image.copy()

    for det in detections:
        cn = det["class_name"]
        bbox = det["bbox"]
        x1 = int(bbox["x1"] * sf_x)
        y1 = int(bbox["y1"] * sf_y)
        x2 = int(bbox["x2"] * sf_x)
        y2 = int(bbox["y2"] * sf_y)
        color = CLASS_COLORS.get(cn, (200, 200, 0))
        cv2.rectangle(img_s4, (x1, y1), (x2, y2), color, 3)
        label = f"{cn}: {det['yolo_confidence']:.2f}"
        font_scale = max(0.5, img_s4.shape[1] / 800)
        cv2.putText(img_s4, label, (x1, max(30, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    return img_to_b64(img_s4)
