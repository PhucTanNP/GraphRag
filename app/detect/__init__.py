"""Tire Detection Package — Nhận dạng thông số lốp xe từ ảnh.

Gồm 4 bước xử lý:
  1. Segment tire (YOLO11-seg)
  2. Polar unwrap + distortion correction
  3. CLAHE preprocessing
  4. YOLO detect regions + Paddle OCR → brand / size / pattern
"""

from app.detect.pipeline import TireDetector

__all__ = ["TireDetector"]
