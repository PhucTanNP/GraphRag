"""Image utilities — encode/decode, constants."""

import base64
import cv2
import numpy as np


def img_to_b64(img: np.ndarray) -> str:
    """Encode OpenCV image → base64 data URI (JPEG)."""
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def download_image(url: str, timeout: int = 30) -> np.ndarray:
    """Tải ảnh từ URL về numpy array (OpenCV BGR format)."""
    import requests
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    img_bytes = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image from URL")
    return image


# ─── Màu sắc cho từng class YOLO detect ─────────────────────────────────
CLASS_COLORS = {
    "brand": (255, 0, 0),   # Xanh dương
    "size": (0, 200, 0),    # Xanh lá
    "pattern": (0, 0, 255), # Đỏ
}
