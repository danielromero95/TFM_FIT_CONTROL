"""Utilidades para normalizar marcadores corporales."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..constants import HIP_CENTER
from ..types import Landmark


def normalize_landmarks(landmarks: Sequence[Landmark]) -> List[Landmark]:
    """Devuelve los marcadores centrados en el punto medio de las caderas."""

    if len(landmarks) <= max(HIP_CENTER):
        return [Landmark.from_mapping(lm) for lm in landmarks]

    left_idx, right_idx = HIP_CENTER
    hip_left = landmarks[left_idx]
    hip_right = landmarks[right_idx]
    cx = (hip_left["x"] + hip_right["x"]) / 2.0
    cy = (hip_left["y"] + hip_right["y"]) / 2.0

    centred: List[Landmark] = []
    for lm in landmarks:
        centred.append(
            Landmark(
                x=float(lm.get("x", np.nan)) - cx,
                y=float(lm.get("y", np.nan)) - cy,
                z=float(lm.get("z", np.nan)),
                visibility=float(lm.get("visibility", np.nan)),
            )
        )
    return centred


__all__ = ["normalize_landmarks"]
