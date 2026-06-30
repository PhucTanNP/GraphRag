"""OCR processing — PaddleOCR inference + CTC decode + fuzzy matching."""

import math
import logging
import re

import cv2
import numpy as np
from paddle import inference
from rapidfuzz import process, fuzz

from app.config import TARGET_OCR_HEIGHT, MAX_OCR_WIDTH
from app.detect.utils import img_to_b64

logger = logging.getLogger("tire-detector.ocr")

# ─── Class map cho YOLO detect ──────────────────────────────────────────
CLASS_MAP = {0: "size", 1: "pattern", 2: "brand"}

# ─── Fuzzy match candidates ────────────────────────────────────────────
BRAND_CANDIDATES = ["DRC", "DPLUS"]

# ─── Brand hard-fix mapping ─────────────────────────────────────────────
BRAND_HARD_FIX = {
    "D": "DRC", "DR": "DRC", "DRI": "DRC", "D-D": "DRC",
    "OR": "DRC", "DRC!": "DRC",
    "DP": "DPLUS", "DPL": "DPLUS", "DPLU": "DPLUS", "DPLS": "DPLUS",
}


def run_ocr(
    preprocessed: np.ndarray,
    detections: list,
    scale_factors: tuple[float, float],
    predictor: inference.Predictor,
    input_names: list[str],
    output_names: list[str],
    char_dict: list[str],
) -> dict:
    """Run OCR on detected regions.

    Args:
        preprocessed: Ảnh đã CLAHE (kích thước gốc)
        detections: Danh sách detection từ YOLO (đã sort top 3)
        scale_factors: (sf_x, sf_y) map từ resized → gốc
        predictor: PaddleOCR predictor
        input_names, output_names: Tên input/output tensor
        char_dict: Dictionary ký tự cho CTC decode

    Returns:
        dict: {brand, size, pattern, brand_ocr, size_ocr, pattern_ocr, detections_count}
    """
    recognition = {
        "brand": None, "size": None, "pattern": None,
        "brand_ocr": None, "size_ocr": None, "pattern_ocr": None,
        "detections_count": 0,
    }

    sf_x, sf_y = scale_factors
    valid_labels = _get_valid_labels()
    crops_b64 = []

    if not detections:
        return recognition

    recognition["detections_count"] = len(detections)
    logger.info("  → %d objects detected (top 3 by confidence)", len(detections))

    for det in detections:
        cn = det["class_name"]
        x1, y1, x2, y2 = det["bbox"]["x1"], det["bbox"]["y1"], det["bbox"]["x2"], det["bbox"]["y2"]

        # Map về kích thước gốc
        fx1 = int(x1 * sf_x)
        fy1 = int(y1 * sf_y)
        fx2 = int(x2 * sf_x)
        fy2 = int(y2 * sf_y)

        # Pad thêm
        pad_y = int((fy2 - fy1) * (0.05 if cn != "pattern" else 0.03))
        fy1p = max(0, fy1 - pad_y)
        fy2p = min(preprocessed.shape[0], fy2 + pad_y)
        crop_full = preprocessed[fy1p:fy2p, fx1:fx2]

        if crop_full.size == 0:
            continue

        # Encode crop
        crops_b64.append({"class": cn, "image": img_to_b64(crop_full)})

        # OCR
        ocr_result = _paddle_ocr_inference(crop_full, predictor, input_names, output_names, char_dict)
        raw = ocr_result["text"]

        # Hard fix brand
        if cn == "brand":
            raw = BRAND_HARD_FIX.get(raw, raw)

        # Fuzzy normalize
        normalized = _fuzzy_normalize(raw, cn, valid_labels)

        # Lưu kết quả
        if recognition[cn] is None:
            recognition[cn] = normalized

        recognition[f"{cn}_ocr"] = {
            "raw_text": raw,
            "normalized_text": normalized,
            "ocr_confidence": ocr_result["confidence"],
            "yolo_confidence": round(det["yolo_confidence"], 4),
            "crop_image": crops_b64[-1]["image"],
            "ocr_input_image": ocr_result["input_image_b64"],
        }
        logger.info("    %s: '%s' → '%s'", cn.upper(), raw, normalized)

    return recognition


def _paddle_ocr_inference(
    crop: np.ndarray,
    predictor, input_names, output_names, char_dict
) -> dict:
    """PaddleOCR inference trên 1 crop ảnh.

    Returns: {text, confidence, input_image_b64}
    """
    h_c, w_c = crop.shape[:2]
    imgH, imgW = TARGET_OCR_HEIGHT, MAX_OCR_WIDTH  # 48, 320

    ratio = w_c / h_c
    max_wh_ratio = imgW / imgH
    resized_w = imgW if math.ceil(imgH * ratio) > imgW else int(math.ceil(imgH * ratio))

    resized = cv2.resize(crop, (resized_w, imgH), interpolation=cv2.INTER_CUBIC)

    # Normalize
    norm_img = resized.astype("float32").transpose((2, 0, 1)) / 255.0
    norm_img -= 0.5
    norm_img /= 0.5

    # Pad → 320
    padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = norm_img

    # Ảnh input cho OCR display
    ocr_display = cv2.resize(crop, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
    input_b64 = img_to_b64(ocr_display)

    # Inference
    img_input = padding_im[np.newaxis, ...]  # (1, 3, 48, 320)
    predictor.get_input_handle(input_names[0]).copy_from_cpu(img_input)
    predictor.run()
    preds = predictor.get_output_handle(output_names[0]).copy_to_cpu()

    # CTC decode
    preds_idx = preds.argmax(axis=2)[0]
    preds_prob = preds.max(axis=2)[0]

    keep = np.ones(len(preds_idx), dtype=bool)
    keep[1:] = preds_idx[1:] != preds_idx[:-1]  # remove duplicates
    keep &= preds_idx != 0                       # remove blanks

    chars = []
    confs = []
    for idx, prob in zip(preds_idx[keep], preds_prob[keep]):
        if (idx - 1) < len(char_dict):
            chars.append(char_dict[idx - 1])
            confs.append(float(prob))

    text = "".join(chars).strip()
    conf = float(np.mean(confs)) if confs else 0.0

    return {"text": text, "confidence": round(conf, 4), "input_image_b64": input_b64}


def _fuzzy_normalize(raw: str, class_name: str, valid_labels: list[str]) -> str:
    """Fuzzy match OCR result với danh sách hợp lệ."""
    if class_name == "brand":
        candidates = BRAND_CANDIDATES
    elif class_name == "pattern":
        candidates = [l for l in valid_labels if re.fullmatch(r"[A-Z]?\d{3,4}", l)]
    elif class_name == "size":
        candidates = [l for l in valid_labels if re.fullmatch(r"\d{2,3}/\d{2,3}-\d{2}", l)]
    else:
        candidates = valid_labels

    match = process.extractOne(raw, candidates, scorer=fuzz.ratio) if candidates else None
    return match[0] if match and match[1] >= 75 else raw


def _get_valid_labels() -> list[str]:
    """Danh sách nhãn hợp lệ."""
    return ["DRC", "DPLUS", "175/65-14", "185/65-15", "195/65-15", "8001", "8002", "8003"]
