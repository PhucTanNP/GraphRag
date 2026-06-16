"""
STEP 4: Tire Recognition & OCR Service
Detect brand/size/pattern boxes, OCR, normalize, query Neo4j
"""

import cv2
import numpy as np
import os
import re
from typing import Tuple, Optional, Dict, List
from rapidfuzz import process, fuzz
from app.logger import logger
from app.config import settings
from ultralytics import YOLO
import uuid
import paddle
from paddle import inference

class TireRecognitionService:
    CLASS_MAP = {
        0: "size",
        1: "pattern",
        2: "brand"
    }

    def __init__(self):
        try:
            # YOLO
            logger.info(f"Loading YOLO11 detection model from {settings.yolo_detect_model_path}")
            self.yolo_model = YOLO(settings.yolo_detect_model_path)
            logger.info("YOLO11 detection model loaded successfully")

            # Paddle dict
            self.char_dict = self._load_paddle_dict()
            self.valid_labels = self._load_valid_labels()

            # Paddle Predictor
            logger.info("Loading Paddle predictor...")

            model_file = os.path.join(
                settings.paddle_ocr_model_path,
                "inference.pdmodel"
            )

            params_file = os.path.join(
                settings.paddle_ocr_model_path,
                "inference.pdiparams"
            )

            config = inference.Config(model_file, params_file)
            config.disable_gpu()

            # TẮT ONEDNN
            config.disable_mkldnn()
            config.switch_ir_optim(False)
            config.switch_use_feed_fetch_ops(False)

            self.predictor = inference.create_predictor(config)

            self.input_names = self.predictor.get_input_names()
            self.output_names = self.predictor.get_output_names()

            logger.info("Paddle predictor loaded successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize TireRecognitionService: {e}",
                exc_info=True
            )
            raise

    def _load_paddle_dict(self) -> List[str]:
        try:
            if not os.path.exists(settings.paddle_dict_path):
                logger.warning(f"PaddleOCR dict not found at {settings.paddle_dict_path}")
                return []

            with open(settings.paddle_dict_path, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]

            logger.info(f"Loaded {len(labels)} characters from PaddleOCR dictionary")
            return labels

        except Exception as e:
            logger.error(f"Error loading PaddleOCR dictionary: {e}")
            return []

    def _load_valid_labels(self) -> List[str]:
        return [
            "DRC", "DPLUS",
            "175/65-14", "185/65-15", "195/65-15",
            "8001", "8002", "8003"
        ]

    def detect_objects(self, image: np.ndarray):
        try:
            h, w = image.shape[:2]
            logger.info(f"Running YOLO detection on {w}x{h}")

            results = self.yolo_model.predict(
                image,
                imgsz=1280,
                conf=0.25,
                iou=0.3,
                verbose=False
            )[0]

            detections = []

            if results.boxes is None or len(results.boxes) == 0:
                logger.warning("No detections found")
                return True, detections, {"count": 0}

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                detections.append({
                    "class_name": self.CLASS_MAP.get(cls_id, "unknown"),
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "yolo_confidence": yolo_conf
                })

            logger.info(f"Detected {len(detections)} objects")
            
            # LƯU ẢNH ĐỂ DEBUG 
            debug_img = image.copy()
            for det in detections:
                b = det["bbox"]

                cv2.rectangle(
                    debug_img,
                    (b["x1"], b["y1"]),
                    (b["x2"], b["y2"]),
                    (0,255,0),
                    2
                )
            cv2.imwrite("debug_outputs/step4_detect.jpg", debug_img)
            
            return True, detections, {"count": len(detections)}

        except Exception as e:
            logger.error(f"YOLO detection error: {e}", exc_info=True)
            return False, [], {"error": str(e)}

    def crop_detection(self, image, bbox, class_name):
        try:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

            w_box = x2 - x1
            h_box = y2 - y1

            if class_name == "brand":
                pad_x = 0
                pad_y = int(h_box * 0.05)

            elif class_name == "size":
                pad_x = 0
                pad_y = int(h_box * 0.05)

            elif class_name == "pattern":
                pad_x = 0
                pad_y = int(h_box * 0.03)

            x1p = max(0, x1 - pad_x)
            y1p = max(0, y1 - pad_y)
            x2p = min(image.shape[1], x2 + pad_x)
            y2p = min(image.shape[0], y2 + pad_y)

            crop = image[y1p:y2p, x1p:x2p]

            cv2.imwrite(
                f"debug_outputs/step4_ocr_input_{uuid.uuid4().hex}.jpg",
                crop
            )
            
            if crop.size == 0:
                return False, None, {"error": "Empty crop"}

            return True, crop, {"crop_size": crop.shape}

        except Exception as e:
            logger.error(f"Crop error: {e}", exc_info=True)
            return False, None, {"error": str(e)}

    def ocr_crop(self, crop, class_name):
        try:
            h, w = crop.shape[:2]

            if h == 0 or w == 0:
                return "", 0.0

            target_h = settings.target_ocr_height
            target_w = settings.max_ocr_width   # 320

            scale = target_h / h
            new_w = int(w * scale)
            new_w = min(new_w, target_w)

            crop = cv2.resize(
                crop,
                (new_w, target_h),
                interpolation=cv2.INTER_CUBIC
            )

            # PAD RIGHT giống predict_rec.py
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            padded[:, :new_w, :] = crop

            # debug
            cv2.imwrite(
                "debug_outputs/step4_ocr_input.jpg",
                padded
            )

            # preprocess
            img = crop.astype("float32")
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)

            # inference
            input_handle = self.predictor.get_input_handle(
                self.input_names[0]
            )
            input_handle.copy_from_cpu(img)

            self.predictor.run()

            output_handle = self.predictor.get_output_handle(
                self.output_names[0]
            )

            preds = output_handle.copy_to_cpu()

            # CTC decode
            text = ""
            confs = []

            pred_indices = preds.argmax(axis=2)[0]
            pred_probs = preds.max(axis=2)[0]

            last_idx = -1

            for idx, prob in zip(pred_indices, pred_probs):
                if idx != 0 and idx != last_idx:   # blank=0
                    if idx - 1 < len(self.char_dict):
                        text += self.char_dict[idx - 1]
                        confs.append(float(prob))
                last_idx = idx

            confidence = np.mean(confs) if confs else 0.0

            return text.strip(), confidence

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0.0

    def normalize_prediction(self, pred, valid_labels, threshold=75):
        if not pred or not valid_labels:
            return pred

        try:
            match = process.extractOne(pred, valid_labels, scorer=fuzz.ratio)

            if match:
                best_label, score, _ = match
                if score >= threshold:
                    return best_label

            return pred

        except Exception:
            return pred

    def process_recognition_pipeline(self, image):
        results = {
            "brand": None,
            "size": None,
            "pattern": None,
            "brand_ocr": None,
            "size_ocr": None,
            "pattern_ocr": None,
            "detections_count": 0
        }

        try:
            success, detections, _ = self.detect_objects(image)

            if not success or len(detections) == 0:
                return True, results, {}

            results["detections_count"] = len(detections)

            for detection in detections:
                class_name = detection["class_name"]
                bbox = detection["bbox"]
                yolo_conf = detection["yolo_confidence"]

                success, crop, _ = self.crop_detection(image, bbox, class_name)
                if not success:
                    continue

                raw_text, ocr_conf = self.ocr_crop(crop, class_name)
                if class_name == "brand":
                    if raw_text in ["D", "DR", "DRI", "D-D", "OR", "DRC!"]:
                        raw_text = "DRC"
                    elif raw_text in ["DP", "DPL", "DPLU", "DPLS"]:
                        raw_text = "DPLUS"
                
                if class_name == "brand":
                    candidate_labels = ["DRC", "DPLUS"]
                elif class_name == "pattern":
                    candidate_labels = [
                        x for x in self.valid_labels
                        if re.fullmatch(r"[A-Z]?\d{3,4}", x)
                    ]
                elif class_name == "size":
                    candidate_labels = [
                        x for x in self.valid_labels
                        if re.fullmatch(r"(\d{2,3}/\d{2,3}-\d{2})", x)
                    ]
                else:
                    candidate_labels = self.valid_labels

                normalized_text = self.normalize_prediction(raw_text, candidate_labels)

                if results[class_name] is None:
                    results[class_name] = normalized_text
                    
                results[f"{class_name}_ocr"] = {
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "ocr_confidence": ocr_conf,
                    "yolo_confidence": yolo_conf
                }

                logger.info(
                    f"{class_name.upper()}: raw='{raw_text}' → normalized='{normalized_text}'"
                )

            return True, results, {"status": "completed"}

        except Exception as e:
            logger.error(f"Recognition pipeline error: {e}", exc_info=True)
            return False, results, {"error": str(e)}