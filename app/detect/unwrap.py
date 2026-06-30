"""Step 2: Polar unwrap + distortion correction.

Chuyển ảnh lốp từ dạng tròn sang dạng phẳng ngang.
"""

import logging

import cv2
import numpy as np

from app.detect.utils import img_to_b64

logger = logging.getLogger("tire-detector.unwrap")


def run_unwrap(
    image: np.ndarray,
    cx: int,
    cy: int,
    r_outer: int,
    r_inner: int,
    out_width: int = 2268,
) -> dict:
    """Polar unwrap + distortion correction.

    Args:
        image: Ảnh gốc BGR
        cx, cy: Tâm lốp
        r_outer: Bán kính ngoài
        r_inner: Bán kính trong
        out_width: Chiều rộng ảnh output

    Returns:
        dict với keys: image (ảnh đã unwrap), detail, image_base64
    """
    h_orig, w_orig = image.shape[:2]

    # ── Đảm bảo r_inner < r_outer ───────────────────────────────────────
    # Tránh lỗi np.linspace với số âm
    if r_inner >= r_outer:
        logger.warning("  r_inner (%d) >= r_outer (%d), swapping", r_inner, r_outer)
        r_inner, r_outer = r_outer, r_inner
    if r_inner >= r_outer:
        logger.warning("  still invalid, fallback to ratio 0.35")
        r_inner = int(r_outer * 0.35)

    out_h = r_outer - r_inner

    # Polar unwrap
    theta = np.linspace(2 * np.pi, 0, out_width, dtype=np.float32)
    r_arr = np.linspace(r_inner, r_outer, out_h, dtype=np.float32)
    T, R = np.meshgrid(theta, r_arr)

    map_x = np.clip((cx + R * np.cos(T)).astype(np.float32), 0, w_orig - 1)
    map_y = np.clip((cy + R * np.sin(T)).astype(np.float32), 0, h_orig - 1)

    unwrapped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    unwrapped = cv2.rotate(unwrapped, cv2.ROTATE_180)

    # Distortion correction
    distortion = _apply_distortion_correction(unwrapped)

    h_u, w_u = unwrapped.shape[:2]
    detail = f"Unwrapped: {w_u}x{h_u}, distortion_correction={'yes' if distortion else 'no'}"
    logger.info("  → %s", detail)

    return {
        "image": unwrapped,
        "detail": detail,
        "image_base64": img_to_b64(unwrapped),
    }


def _apply_distortion_correction(unwrapped: np.ndarray) -> bool:
    """Chỉnh méo ảnh unwrapped nếu phát hiện đường cong."""
    h_u, w_u = unwrapped.shape[:2]
    gray_temp = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_temp, 10, 255, cv2.THRESH_BINARY)

    cols = np.linspace(0, w_u - 1, 30, dtype=np.int32)
    pts_x, pts_y = [], []

    for x in cols:
        col = thresh[:, x]
        nz = np.where(col > 0)[0]
        if len(nz) > 0:
            pts_x.append(x)
            pts_y.append(int(np.mean(nz)))

    if len(pts_x) < 3:
        return False

    poly = np.poly1d(np.polyfit(pts_x, pts_y, 3))
    mx, my = np.meshgrid(np.arange(w_u, dtype=np.float32), np.arange(h_u, dtype=np.float32))
    mid_y = h_u / 2.0
    my = np.clip(my + (poly(mx) - mid_y), 0, h_u - 1).astype(np.float32)
    # Gán đè lên ảnh gốc (in-place)
    corrected = cv2.remap(unwrapped, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    unwrapped[:] = corrected
    return True
