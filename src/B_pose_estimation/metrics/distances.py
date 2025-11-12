"""Métricas basadas en distancias entre puntos clave."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..constants import DISTANCE_PAIRS


def calculate_distances(landmarks: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Obtener las distancias horizontales definidas por los pares de *landmarks*."""

    metrics: Dict[str, float] = {}
    for name, (a_idx, b_idx) in DISTANCE_PAIRS.items():
        try:
            a = landmarks[a_idx]
            b = landmarks[b_idx]
            if np.isfinite(a["x"]) and np.isfinite(b["x"]):
                metrics[name] = abs(float(b["x"]) - float(a["x"]))
            else:
                metrics[name] = np.nan
        except Exception:
            metrics[name] = np.nan
    return metrics


__all__ = ["calculate_distances"]
