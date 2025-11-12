"""Cálculo de ángulos articulares a partir de *landmarks* de pose."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from ..constants import JOINT_INDEX_MAP

def _points_finite(landmarks: Sequence[Mapping[str, float]], indices: Iterable[int]) -> bool:
    """Comprobar si los puntos referenciados contienen coordenadas finitas."""
    try:
        return all(
            np.isfinite(landmarks[i]["x"]) and np.isfinite(landmarks[i]["y"])
            for i in indices
        )
    except Exception:
        return False


def calculate_angle(p1: Mapping[str, float], p2: Mapping[str, float], p3: Mapping[str, float]) -> float:
    """Calcular el ángulo en grados formado por tres puntos ordenados."""

    v1 = (p1["x"] - p2["x"], p1["y"] - p2["y"])
    v2 = (p3["x"] - p2["x"], p3["y"] - p2["y"])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
    if mag1 * mag2 == 0:
        return 0.0
    cosine = max(min(dot_product / (mag1 * mag2), 1.0), -1.0)
    return math.degrees(math.acos(cosine))


def extract_joint_angles(landmarks: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Devolver un diccionario con los ángulos relevantes de las articulaciones."""

    angles: Dict[str, float] = {}
    for name, indices in JOINT_INDEX_MAP.items():
        if _points_finite(landmarks, indices):
            p1, p2, p3 = (landmarks[i] for i in indices)
            angles[name] = calculate_angle(p1, p2, p3)
        else:
            angles[name] = np.nan
    return angles


__all__ = ["calculate_angle", "extract_joint_angles"]
