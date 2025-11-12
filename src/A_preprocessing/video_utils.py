"""Funciones auxiliares para trabajar con archivos de vídeo."""

from __future__ import annotations

import math
import os
from typing import Dict

import cv2

__all__ = ["validate_video"]


def validate_video(path: str) -> Dict[str, float]:
    """Verifica que el vídeo sea legible y devuelve metadatos esenciales."""

    if not os.path.exists(path):
        raise IOError(f"Video path does not exist: {path}")

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise IOError(f"Could not open the video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid FPS obtained: {fps}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count > 0 else 0.0

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
        }
    finally:
        capture.release()
