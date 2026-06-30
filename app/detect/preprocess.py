"""Step 3: CLAHE preprocessing — tăng cường ảnh trước khi detect."""

import logging

import cv2
import numpy as np

from app.detect.utils import img_to_b64

logger = logging.getLogger("tire-detector.preprocess")


DETECT_INPUT_WIDTH = 1280


def run_preprocess(image: np.ndarray) -> dict:
    """CLAHE preprocessing + resize cho YOLO detect.

    Args:
        image: Ảnh BGR đã unwrapped

    Returns:
        dict với keys:
          - image: Ảnh đã preprocess (kích thước gốc)
          - image_resized: Ảnh đã resize cho YOLO detect
          - scale_factors: (sf_x, sf_y) để map bbox về kích thước gốc
          - detail, image_base64
    """
    hu, wu = image.shape[:2]

    # Crop viền đen
    cropped = image[:int(hu * 0.98), :]

    # CLAHE
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    preprocessed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Resize cho YOLO detect
    h_p, w_p = preprocessed.shape[:2]
    if w_p > DETECT_INPUT_WIDTH:
        scale = DETECT_INPUT_WIDTH / w_p
        resized = cv2.resize(preprocessed, (DETECT_INPUT_WIDTH, int(h_p * scale)), cv2.INTER_AREA)
    else:
        resized = preprocessed.copy()

    sf_x = preprocessed.shape[1] / resized.shape[1]
    sf_y = preprocessed.shape[0] / resized.shape[0]

    detail = f"CLAHE applied, size: {w_p}x{h_p}, resized: {resized.shape[1]}x{resized.shape[0]}"
    logger.info("  → %s", detail)

    return {
        "image": preprocessed,
        "image_resized": resized,
        "scale_factors": (sf_x, sf_y),
        "detail": detail,
        "image_base64": img_to_b64(preprocessed),
    }
