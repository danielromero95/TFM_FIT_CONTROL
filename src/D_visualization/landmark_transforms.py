"""Transformaciones de vídeo específicas para rotaciones sencillas.
Centraliza la lógica de orientación de los frames para que todos los renderizadores
apliquen las mismas reglas al corregir el ángulo de la cámara."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from src.A_preprocessing.frame_extraction.utils import normalize_rotation_deg

# Indicamos qué elementos forman parte de la API pública del módulo.
__all__ = ["_normalize_rotation_deg", "_rotate_frame"]

# Diccionario con la correspondencia entre ángulos y constantes de OpenCV.
_ROTATE_CODE: Dict[int, int | None] = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _normalize_rotation_deg(value: int) -> int:
    """Alias interno que reutiliza la normalización global de rotaciones."""

    return normalize_rotation_deg(value)


def _rotate_frame(frame: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Devuelve el frame rotado de manera horaria según ``rotation_deg``.
    Al centralizar la rotación garantizamos que el resto del pipeline manipule los
    frames ya orientados, simplificando la lógica de dibujo y recorte."""

    rotation = _normalize_rotation_deg(rotation_deg)
    code = _ROTATE_CODE.get(rotation)
    return cv2.rotate(frame, code) if code is not None else frame
