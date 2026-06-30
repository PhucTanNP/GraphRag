"""Step 1: Segment tire using YOLO11-seg.

Phân vùng lốp xe từ ảnh gốc, tìm tâm và bán kính.
"""

import logging

import cv2
import numpy as np

from app.detect.utils import img_to_b64

logger = logging.getLogger("tire-detector.segment")


def run_segmentation(image: np.ndarray, seg_model) -> dict:
    """Segment lốp xe bằng YOLO11-seg.

    Args:
        image: Ảnh BGR đầu vào (numpy array).
        seg_model: Model YOLO segmentation đã load.

    Returns:
        dict với các keys:
          - success: bool
          - detail: str mô tả kết quả
          - image_base64: Ảnh segmentation overlay (data URI)
          - cx, cy: Tọa độ tâm lốp (pixel)
          - r_outer: Bán kính ngoài (pixel)
          - r_inner: Bán kính trong (pixel)
          - masked_image: Ảnh đã cô lập lốp (nền đen)
    """
    h, w = image.shape[:2]

    # ── Run YOLO11-seg inference ──────────────────────────────────────
    results = seg_model.predict(image, verbose=False)[0]

    if results.masks is None or len(results.masks) == 0:
        logger.warning("  ✗ No mask found — fallback to full image")
        # Fallback: dùng toàn bộ ảnh, ước lượng lốp chiếm phần lớn
        cx, cy = w // 2, h // 2
        r_outer = min(w, h) // 2
        r_inner = int(r_outer * 0.35)
        masked_image = image.copy()
        return {
            "success": False,
            "detail": "No mask detected, using full image fallback",
            "image_base64": img_to_b64(image),
            "cx": cx,
            "cy": cy,
            "r_outer": r_outer,
            "r_inner": r_inner,
            "masked_image": masked_image,
        }

    # ── Get mask của class đầu tiên (tire) ──────────────────────────
    mask_tensor = results.masks.data[0].cpu().numpy()
    # Resize mask về kích thước ảnh gốc
    mask = cv2.resize(mask_tensor, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.uint8) * 255

    # ── Find contours ────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("  ✗ No contours found in mask — using fallback")
        cx, cy = w // 2, h // 2
        r_outer = min(w, h) // 2
        r_inner = int(r_outer * 0.35)
        masked_image = image.copy()
        return {
            "success": False,
            "detail": "No contours found, using fallback",
            "image_base64": img_to_b64(image),
            "cx": cx,
            "cy": cy,
            "r_outer": r_outer,
            "r_inner": r_inner,
            "masked_image": masked_image,
        }

    # Lấy contour lớn nhất (lốp xe)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit circle bao quanh contour
    ((cx, cy), radius) = cv2.minEnclosingCircle(largest_contour)
    cx, cy = int(cx), int(cy)
    r_outer = int(radius)

    # Tính bán kính trong dựa trên mask
    # Tìm lỗ trống bên trong bằng cách invert mask và tìm contour lỗ
    mask_inv = cv2.bitwise_not(mask)
    contours_inner, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r_inner = int(r_outer * 0.35)  # default ratio

    for c_inner in contours_inner:
        # Chỉ xét contour nằm bên trong vòng tròn ngoài
        M = cv2.moments(c_inner)
        if M["m00"] == 0:
            continue
        c_cx = int(M["m10"] / M["m00"])
        c_cy = int(M["m01"] / M["m00"])
        dist = np.sqrt((c_cx - cx) ** 2 + (c_cy - cy) ** 2)
        if dist < r_outer * 0.3:
            # Contour này là lỗ trong (tâm lốp)
            ((_, _), ir) = cv2.minEnclosingCircle(c_inner)
            r_inner = int(ir)
            break

    # ── Tạo masked image ─────────────────────────────────────────────
    # Dùng mask nhị phân để giữ lại vùng lốp
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    masked_image = (image * mask_3ch).astype(np.uint8)

    # Crop quanh lốp để giảm kích thước
    x_min = max(0, cx - r_outer - 20)
    x_max = min(w, cx + r_outer + 20)
    y_min = max(0, cy - r_outer - 20)
    y_max = min(h, cy + r_outer + 20)
    masked_image = masked_image[y_min:y_max, x_min:x_max]

    # ── Vẽ kết quả segmentation ──────────────────────────────────────
    overlay = image.copy()
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask > 0] = [0, 200, 0]  # green overlay
    overlay = cv2.addWeighted(overlay, 0.6, colored_mask, 0.4, 0)
    # Vẽ vòng tròn ngoài
    cv2.circle(overlay, (cx, cy), r_outer, (0, 255, 0), 3)
    # Vẽ tâm
    cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)

    detail = (
        f"Tire segmented: center=({cx},{cy}), "
        f"r_outer={r_outer}px, r_inner={r_inner}px, "
        f"mask_area={cv2.contourArea(largest_contour)}px²"
    )
    logger.info("  → %s", detail)

    return {
        "success": True,
        "detail": detail,
        "image_base64": img_to_b64(overlay),
        "cx": cx,
        "cy": cy,
        "r_outer": r_outer,
        "r_inner": r_inner,
        "masked_image": masked_image,
    }
