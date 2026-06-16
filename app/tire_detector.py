"""
TireDetector — Service duy nhất cho toàn bộ pipeline nhận dạng lốp xe.

Gồm 4 bước trong một class:
  1. Segment tire (YOLO11-seg)
  2. Unwrap polar + distortion correction
  3. Preprocess (CLAHE)
  4. YOLO detect regions + Paddle OCR → brand / size / pattern

Không truy vấn Neo4j. Chỉ trả về những gì detect được.
"""
import os

# Fix xung đột OpenMP giữa PyTorch (MKL) và PaddlePaddle
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
import logging
import math
import os
import re
import uuid
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
from paddle import inference
from rapidfuzz import process, fuzz
from ultralytics import YOLO

from app.config import (
    YOLO_SEG_MODEL_PATH,
    YOLO_DETECT_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    PADDLE_OCR_MODEL_PATH,
    PADDLE_DICT_PATH,
    TARGET_OCR_HEIGHT,
    MAX_OCR_WIDTH,
)

logger = logging.getLogger("tire-detector")


def _img_to_b64(img: np.ndarray) -> str:
    """Encode OpenCV image → base64 data URI (JPEG)."""
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


class TireDetector:
    """
    Nhận dạng thông số lốp xe từ URL ảnh.

    Pipeline:
      1. Tải ảnh từ URL → numpy array
      2. YOLO11-seg → phân đoạn vành lốp → mask
      3. Polar unwrap + distortion correction → ảnh phẳng
      4. CLAHE preprocessing
      5. YOLO detect → crop brand/size/pattern → Paddle OCR
      6. Trả về {brand, size, pattern, ocr_details}
    """

    # ── Class map cho YOLO detect ──────────────────────────────────────
    CLASS_MAP = {0: "size", 1: "pattern", 2: "brand"}

    def __init__(self):
        self.seg_model: Optional[YOLO] = None
        self.detect_model: Optional[YOLO] = None
        self.predictor = None
        self.char_dict: list[str] = []
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self._loaded = False

    # ═══════════════════════════════════════════════════════════════════
    #  LOAD MODELS
    # ═══════════════════════════════════════════════════════════════════

    def load_models(self):
        """Load toàn bộ models. Log từng model khi load xong."""
        if self._loaded:
            return

        # ── 1. YOLO11-seg ──
        logger.info("=" * 50)
        logger.info("Loading models for TireDetector...")
        logger.info(f"  YOLO11-seg  → {YOLO_SEG_MODEL_PATH}")
        self.seg_model = YOLO(YOLO_SEG_MODEL_PATH)
        logger.info("  ✓ YOLO11-seg loaded")

        # ── 2. YOLO detect ──
        logger.info(f"  YOLO detect → {YOLO_DETECT_MODEL_PATH}")
        self.detect_model = YOLO(YOLO_DETECT_MODEL_PATH)
        logger.info("  ✓ YOLO detect loaded")

        # ── 3. Paddle dict ──
        logger.info(f"  Paddle dict  → {PADDLE_DICT_PATH}")
        if os.path.exists(PADDLE_DICT_PATH):
            with open(PADDLE_DICT_PATH, "r", encoding="utf-8") as f:
                self.char_dict = [line.strip() for line in f if line.strip()]
            logger.info(f"  ✓ Paddle dict loaded ({len(self.char_dict)} chars)")
        else:
            logger.warning("  ✗ Paddle dict NOT FOUND")

        # ── 4. Paddle predictor ──
        model_file = os.path.join(PADDLE_OCR_MODEL_PATH, "inference.pdmodel")
        params_file = os.path.join(PADDLE_OCR_MODEL_PATH, "inference.pdiparams")
        logger.info(f"  Paddle OCR   → {model_file}")
        cfg = inference.Config(model_file, params_file)
        cfg.disable_gpu()
        cfg.disable_mkldnn()
        cfg.switch_ir_optim(False)
        cfg.switch_use_feed_fetch_ops(False)
        self.predictor = inference.create_predictor(cfg)
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        logger.info("  ✓ Paddle OCR loaded")

        self._loaded = True
        logger.info("=" * 50)
        logger.info("All models loaded successfully!")
        logger.info("=" * 50)

    # ═══════════════════════════════════════════════════════════════════
    #  MAIN DETECT METHOD
    # ═══════════════════════════════════════════════════════════════════

    def detect_from_url(self, image_url: str) -> dict:
        """
        Nhận dạng lốp từ URL ảnh.

        Returns:
            {
                "success": True/False,
                "brand": "DRC" | None,
                "size": "185/65-15" | None,
                "pattern": "8001" | None,
                "brand_ocr": {...} | None,
                "size_ocr": {...} | None,
                "pattern_ocr": {...} | None,
                "detections_count": 3,
                "error": "..." | None,
            }
        """
        self.load_models()

        steps = []  # accumulate step details

        try:
            logger.info("────────────────────────────────────────────────────────")
            logger.info(f"▶ detect_from_url START — processing image...")
            
            # ── Download image ──
            logger.info(f"Downloading image: {image_url}")
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            img_bytes = np.frombuffer(resp.content, dtype=np.uint8)
            image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if image is None:
                return {"success": False, "error": "Cannot decode image", "steps": steps}

            logger.info(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
            os.makedirs("debug_outputs", exist_ok=True)

            # ══════════════════════════════════════════════════════════
            #  STEP 1: SEGMENT
            # ══════════════════════════════════════════════════════════
            logger.info("Step 1/4: Segmenting tire...")
            h_orig, w_orig = image.shape[:2]
            logger.info("  → Running YOLO11-seg predict (imgsz=640)... (may take 10-30s on CPU)")
            results = self.seg_model.predict(
                image, conf=YOLO_CONF_THRESHOLD, imgsz=640, verbose=False
            )[0]
            logger.info("  ✓ YOLO11-seg predict done")
            logger.info(f"  → boxes={len(results.boxes) if results.boxes else 0}, has_masks={results.masks is not None}")

            if len(results.boxes) == 0:
                steps.append({"step": 1, "name": "Segment tire", "status": "error", "detail": "No tire detected"})
                return {"success": False, "error": "No tire detected in image", "steps": steps}

            if results.masks is not None:
                mask_raw = results.masks.data[0].cpu().numpy()
                logger.info(f"  → mask shape={mask_raw.shape}, max={mask_raw.max():.3f}")
                logger.info("  → Processing contour, ellipse fit...")
                mask_resized = cv2.resize(mask_raw, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                raw_mask = (mask_resized * 255).astype(np.uint8)
                box = results.boxes[0].xyxy[0].cpu().numpy().astype(np.int32)
                bx1, by1, bx2, by2 = box[0], box[1], box[2], box[3]
                box_mask = np.zeros_like(raw_mask)
                cv2.rectangle(box_mask, (bx1, by1), (bx2, by2), 255, -1)
                raw_mask = cv2.bitwise_and(raw_mask, box_mask)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, dark_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
                raw_mask = cv2.bitwise_and(raw_mask, dark_mask)
                contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    steps.append({"step": 1, "name": "Segment tire", "status": "error", "detail": "No contours after filtering"})
                    return {"success": False, "error": "No contours after filtering", "steps": steps}
                largest_contour = max(contours, key=cv2.contourArea)
                if len(largest_contour) >= 5:
                    (xc, yc), (d1, d2), _ = cv2.fitEllipse(largest_contour)
                    cx, cy = int(xc), int(yc)
                    r_outer = int(max(d1, d2) / 2)
                else:
                    (xc, yc), radius_out = cv2.minEnclosingCircle(largest_contour)
                    cx, cy = int(xc), int(yc)
                    r_outer = int(radius_out)
                tire_mask = raw_mask.copy()
                r_inner = int(r_outer * 0.64)
                cv2.circle(tire_mask, (cx, cy), r_inner, 0, -1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                tire_mask = cv2.morphologyEx(tire_mask, cv2.MORPH_OPEN, kernel)
                s1_detail = f"Mask: center=({cx},{cy}), r_outer={r_outer}"
            else:
                logger.info("  → No masks from model, using bounding box estimate")
                box = results.boxes[0].xyxy[0].cpu().numpy().astype(np.int32)
                bx1, by1, bx2, by2 = box[0], box[1], box[2], box[3]
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2
                r_outer = int(max(bx2 - bx1, by2 - by1) * 0.55)
                r_inner = int(r_outer * 0.64)
                s1_detail = f"Box: center=({cx},{cy}), r_outer={r_outer}"

            logger.info(f"  → {s1_detail}")
            # ── Step 1 image: draw mask/ellipse on original ──
            img_s1 = image.copy()
            cv2.circle(img_s1, (cx, cy), r_outer, (0, 255, 0), 3)
            cv2.circle(img_s1, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img_s1, f"r={r_outer}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            step1_img = _img_to_b64(img_s1)
            steps.append({"step": 1, "name": "Segment tire", "status": "ok", "detail": s1_detail, "image": step1_img})

            # ══════════════════════════════════════════════════════════
            #  STEP 2: UNWRAP + DISTORTION CORRECTION
            # ══════════════════════════════════════════════════════════
            logger.info("Step 2/4: Unwrapping + distortion correction...")
            logger.info("  → Polar unwrap (2268px wide)...")
            out_h = r_outer - r_inner
            out_w = 2268
            theta = np.linspace(2 * np.pi, 0, out_w, dtype=np.float32)
            r_arr = np.linspace(r_inner, r_outer, out_h, dtype=np.float32)
            T, R = np.meshgrid(theta, r_arr)
            map_x = np.clip((cx + R * np.cos(T)).astype(np.float32), 0, w_orig - 1)
            map_y = np.clip((cy + R * np.sin(T)).astype(np.float32), 0, h_orig - 1)
            unwrapped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            unwrapped = cv2.rotate(unwrapped, cv2.ROTATE_180)
            h_u, w_u = unwrapped.shape[:2]

            # Distortion correction
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
            distortion = False
            if len(pts_x) >= 3:
                poly = np.poly1d(np.polyfit(pts_x, pts_y, 3))
                mx, my = np.meshgrid(np.arange(w_u, dtype=np.float32), np.arange(h_u, dtype=np.float32))
                mid_y = h_u / 2.0
                my = np.clip(my + (poly(mx) - mid_y), 0, h_u - 1).astype(np.float32)
                unwrapped = cv2.remap(unwrapped, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                distortion = True

            s2_detail = f"Unwrapped: {unwrapped.shape[1]}x{unwrapped.shape[0]}, distortion_correction={'yes' if distortion else 'no'}"
            logger.info(f"  → {s2_detail}")
            step2_img = _img_to_b64(unwrapped)
            steps.append({"step": 2, "name": "Unwrap + distortion correction", "status": "ok", "detail": s2_detail, "image": step2_img})

            # ══════════════════════════════════════════════════════════
            #  STEP 3: PREPROCESS (CLAHE)
            # ══════════════════════════════════════════════════════════
            logger.info("Step 3/4: Preprocessing (CLAHE)...")

            # Crop nhẹ viền đen trước
            hu, wu = unwrapped.shape[:2]
            unwrapped_cropped = unwrapped[:int(hu * 0.98), :]

            # CLAHE trên full-width
            gray_clahe = cv2.cvtColor(unwrapped_cropped, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_clahe)
            preprocessed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            # Encode step 3 image — full width gốc
            s3_detail = f"CLAHE applied, size: {preprocessed.shape[1]}x{preprocessed.shape[0]}"
            logger.info(f"  → {s3_detail}")
            step3_img = _img_to_b64(preprocessed)
            steps.append({"step": 3, "name": "CLAHE preprocessing", "status": "ok", "detail": s3_detail, "image": step3_img})

            # Resize width cho YOLO detect — giữ kích thước đủ lớn (tối đa 1280)
            h_p, w_p = preprocessed.shape[:2]
            DETECT_INPUT_WIDTH = 1280
            if w_p > DETECT_INPUT_WIDTH:
                scale = DETECT_INPUT_WIDTH / w_p
                preprocessed_resized = cv2.resize(preprocessed, (DETECT_INPUT_WIDTH, int(h_p * scale)), cv2.INTER_AREA)
            else:
                preprocessed_resized = preprocessed.copy()
            logger.info(f"  → Resized for detect: {preprocessed_resized.shape[1]}x{preprocessed_resized.shape[0]}")

            # ══════════════════════════════════════════════════════════
            #  STEP 4: YOLO DETECT + OCR
            # ══════════════════════════════════════════════════════════
            logger.info("Step 4/4: YOLO detect + OCR...")
            logger.info("  → Running YOLO detect predict (imgsz=1280)... (may take 10-30s on CPU)")
            results = self.detect_model.predict(preprocessed_resized, imgsz=1280, conf=0.25, iou=0.3, verbose=False)[0]
            logger.info("  ✓ YOLO detect predict done")

            recognition = {
                "brand": None, "size": None, "pattern": None,
                "brand_ocr": None, "size_ocr": None, "pattern_ocr": None,
                "detections_count": 0,
            }
            detections = []
            crops_b64 = []  # ảnh crop từng vùng

            if results.boxes and len(results.boxes) > 0:
                detections = []
                # Scale factors from resized → full
                sf_x = preprocessed.shape[1] / preprocessed_resized.shape[1]
                sf_y = preprocessed.shape[0] / preprocessed_resized.shape[0]

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "class_name": self.CLASS_MAP.get(int(box.cls[0]), "unknown"),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "yolo_confidence": float(box.conf[0]),
                    })

                # Giữ tối đa 3 box có confidence cao nhất
                detections.sort(key=lambda d: d["yolo_confidence"], reverse=True)
                detections = detections[:3]

                recognition["detections_count"] = len(detections)
                logger.info(f"  → {len(detections)} objects detected (top 3 by confidence)")

                for det in detections:
                    cn = det["class_name"]
                    bbox = det["bbox"]
                    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

                    # Map to full-size coords for crop
                    fx1 = int(x1 * sf_x)
                    fy1 = int(y1 * sf_y)
                    fx2 = int(x2 * sf_x)
                    fy2 = int(y2 * sf_y)

                    pad_y = int((fy2 - fy1) * (0.05 if cn != "pattern" else 0.03))
                    fy1p = max(0, fy1 - pad_y)
                    fy2p = min(preprocessed.shape[0], fy2 + pad_y)
                    crop_full = preprocessed[fy1p:fy2p, fx1:fx2]

                    if crop_full.size == 0:
                        continue

                    # Encode crop image
                    crop_b64 = _img_to_b64(crop_full)
                    crops_b64.append({"class": cn, "image": crop_b64})

                    # OCR — SVTRRecResizeImg (giống predict_rec.py)
                    h_c, w_c = crop_full.shape[:2]
                    imgH, imgW = TARGET_OCR_HEIGHT, MAX_OCR_WIDTH  # 48, 320
                    ratio = w_c * 1.0 / h_c
                    max_wh_ratio = imgW * 1.0 / imgH
                    if math.ceil(imgH * ratio) > imgW:
                        resized_w = imgW
                    else:
                        resized_w = int(math.ceil(imgH * ratio))
                    resized = cv2.resize(crop_full, (resized_w, imgH), interpolation=cv2.INTER_CUBIC)
                    # Normalize: /255 → -0.5 → /0.5
                    norm_img = resized.astype("float32").transpose((2, 0, 1)) / 255.0
                    norm_img -= 0.5
                    norm_img /= 0.5
                    # Pad phải → 320
                    padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
                    padding_im[:, :, 0:resized_w] = norm_img

                    # Ảnh input thật cho OCR (đã resize + pad)
                    ocr_display = cv2.resize(crop_full, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
                    ocr_input_b64 = _img_to_b64(ocr_display)

                    # Inference
                    img_input = padding_im[np.newaxis, ...]  # (1, 3, 48, 320)
                    self.predictor.get_input_handle(self.input_names[0]).copy_from_cpu(img_input)
                    self.predictor.run()
                    preds = self.predictor.get_output_handle(self.output_names[0]).copy_to_cpu()

                    # CTC decode — giống CTCLabelDecode (bỏ threshold, đúng blank/duplicate)
                    preds_idx = preds.argmax(axis=2)[0]
                    preds_prob = preds.max(axis=2)[0]
                    keep = np.ones(len(preds_idx), dtype=bool)
                    keep[1:] = preds_idx[1:] != preds_idx[:-1]   # remove duplicates
                    keep &= preds_idx != 0                        # remove blanks (idx 0)
                    chars = []
                    confs = []
                    for idx, prob in zip(preds_idx[keep], preds_prob[keep]):
                        if (idx - 1) < len(self.char_dict):
                            chars.append(self.char_dict[idx - 1])
                            confs.append(float(prob))
                    text = "".join(chars)
                    ocr_conf = float(np.mean(confs)) if confs else 0.0
                    raw = text.strip()

                    # Brand hard fix
                    if cn == "brand":
                        if raw in ("D", "DR", "DRI", "D-D", "OR", "DRC!"):
                            raw = "DRC"
                        elif raw in ("DP", "DPL", "DPLU", "DPLS"):
                            raw = "DPLUS"

                    # Fuzzy normalize
                    if cn == "brand":
                        candidates = ["DRC", "DPLUS"]
                    elif cn == "pattern":
                        candidates = [l for l in self._valid_labels() if re.fullmatch(r"[A-Z]?\d{3,4}", l)]
                    elif cn == "size":
                        candidates = [l for l in self._valid_labels() if re.fullmatch(r"\d{2,3}/\d{2,3}-\d{2}", l)]
                    else:
                        candidates = self._valid_labels()

                    match = process.extractOne(raw, candidates, scorer=fuzz.ratio) if candidates else None
                    normalized = match[0] if match and match[1] >= 75 else raw

                    if recognition[cn] is None:
                        recognition[cn] = normalized
                    recognition[f"{cn}_ocr"] = {
                        "raw_text": raw,
                        "normalized_text": normalized,
                        "ocr_confidence": round(ocr_conf, 4),
                        "yolo_confidence": round(det["yolo_confidence"], 4),
                        "crop_image": crop_b64,      # ảnh crop từ step 3
                        "ocr_input_image": ocr_input_b64,  # ảnh thực tế đưa vào PaddleOCR (48×320, đã pad)
                    }
                    logger.info(f"    {cn.upper()}: '{raw}' → '{normalized}'")

            # ── Step 4 image: draw YOLO boxes with class + confidence (NOT OCR result) ──
            img_s4 = preprocessed.copy()
            colors = {"brand": (255, 0, 0), "size": (0, 200, 0), "pattern": (0, 0, 255)}
            for det in detections:
                cn = det["class_name"]
                bbox = det["bbox"]
                # Scale resized bbox back to full-size
                x1 = int(bbox["x1"] * sf_x)
                y1 = int(bbox["y1"] * sf_y)
                x2 = int(bbox["x2"] * sf_x)
                y2 = int(bbox["y2"] * sf_y)
                color = colors.get(cn, (200, 200, 0))
                cv2.rectangle(img_s4, (x1, y1), (x2, y2), color, 3)
                label = f"{cn}: {det['yolo_confidence']:.2f}"
                font_scale = max(0.5, img_s4.shape[1] / 800)
                cv2.putText(img_s4, label, (x1, max(30, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            step4_img = _img_to_b64(img_s4)

            logger.info("  ✓ Recognition done")
            s4_detail = f"Detected {recognition['detections_count']} regions: brand={recognition['brand'] or '?'}, size={recognition['size'] or '?'}, pattern={recognition['pattern'] or '?'}"
            s4_detail += f" | full_size={preprocessed.shape[1]}x{preprocessed.shape[0]}"
            steps.append({"step": 4, "name": "YOLO detect + OCR", "status": "ok", "detail": s4_detail, "image": step4_img, "crops": crops_b64, "detect_input_image": _img_to_b64(preprocessed_resized)})

            recognition["success"] = True
            recognition["steps"] = steps
            return recognition

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"TireDetector error: {e}")
            print(f"🔥🔥🔥 TIRE DETECTOR ERROR 🔥🔥🔥\n{tb}\n{'='*60}")
            return {"success": False, "error": f"{type(e).__name__}: {e}", "steps": steps}

    def _valid_labels(self) -> list:
        return ["DRC", "DPLUS", "175/65-14", "185/65-15", "195/65-15", "8001", "8002", "8003"]

    def health(self) -> dict:
        return {
            "models_loaded": self._loaded,
            "seg_model": self.seg_model is not None,
            "detect_model": self.detect_model is not None,
            "paddle_ocr": self.predictor is not None,
        }
