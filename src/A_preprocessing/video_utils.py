"""Utility functions for working with video files."""

from __future__ import annotations

import math
import os
from typing import Dict

import cv2

__all__ = ["validate_video"]


def validate_video(path: str) -> Dict[str, float]:
    """Validate that a video file can be opened and extract basic metadata."""

    if not os.path.exists(path):
        raise IOError(f"La ruta del vídeo no existe: {path}")

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise IOError(f"No se pudo abrir el vídeo: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"FPS inválido obtenido: {fps}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count > 0 else 0.0

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
        }
    finally:
        capture.release()
