"""Model loading — load và quản lý các model AI (YOLO, PaddleOCR)."""

import os
import logging

from paddle import inference
from ultralytics import YOLO

from app.config import (
    YOLO_SEG_MODEL_PATH,
    YOLO_DETECT_MODEL_PATH,
    PADDLE_OCR_MODEL_PATH,
    PADDLE_DICT_PATH,
)

logger = logging.getLogger("tire-detector.models")


class ModelManager:
    """Quản lý tất cả models: load, health check, singleton."""

    def __init__(self):
        self.seg_model: YOLO | None = None
        self.detect_model: YOLO | None = None
        self.predictor = None
        self.char_dict: list[str] = []
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load_all(self):
        """Load toàn bộ models. Log từng model khi load xong."""
        if self._loaded:
            return

        logger.info("=" * 50)
        logger.info("Loading models for TireDetector...")

        # 1. YOLO11-seg
        logger.info("  YOLO11-seg  → %s", YOLO_SEG_MODEL_PATH)
        self.seg_model = YOLO(YOLO_SEG_MODEL_PATH)
        logger.info("  ✓ YOLO11-seg loaded")

        # 2. YOLO detect
        logger.info("  YOLO detect → %s", YOLO_DETECT_MODEL_PATH)
        self.detect_model = YOLO(YOLO_DETECT_MODEL_PATH)
        logger.info("  ✓ YOLO detect loaded")

        # 3. Paddle dict
        logger.info("  Paddle dict  → %s", PADDLE_DICT_PATH)
        if os.path.exists(PADDLE_DICT_PATH):
            with open(PADDLE_DICT_PATH, "r", encoding="utf-8") as f:
                self.char_dict = [line.strip() for line in f if line.strip()]
            logger.info("  ✓ Paddle dict loaded (%d chars)", len(self.char_dict))
        else:
            logger.warning("  ✗ Paddle dict NOT FOUND")

        # 4. Paddle predictor
        model_file = os.path.join(PADDLE_OCR_MODEL_PATH, "inference.pdmodel")
        params_file = os.path.join(PADDLE_OCR_MODEL_PATH, "inference.pdiparams")
        logger.info("  Paddle OCR   → %s", model_file)
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

    def health(self) -> dict:
        return {
            "models_loaded": self._loaded,
            "seg_model": self.seg_model is not None,
            "detect_model": self.detect_model is not None,
            "paddle_ocr": self.predictor is not None,
        }
