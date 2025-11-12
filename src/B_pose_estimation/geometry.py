"""Utilidades geométricas compartidas entre los módulos de estimación de pose."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .constants import LANDMARK_COUNT
from .types import Landmark, PoseSequence


def landmarks_from_proto(landmarks: Iterable[object]) -> list[Landmark]:
    """Convierte landmarks de Mediapipe en objetos :class:`Landmark`."""

    converted: list[Landmark] = []
    for lm in landmarks:
        converted.append(
            Landmark(
                x=float(getattr(lm, "x", np.nan)),
                y=float(getattr(lm, "y", np.nan)),
                z=float(getattr(lm, "z", np.nan)),
                visibility=float(getattr(lm, "visibility", np.nan)),
            )
        )
    return converted


def landmarks_to_pixel_xy(landmarks: Sequence[Landmark], width: int, height: int) -> np.ndarray:
    """Devuelve un array ``(N, 2)`` con las coordenadas de píxel resultantes."""

    arr = np.empty((len(landmarks), 2), dtype=float)
    for idx, lm in enumerate(landmarks):
        arr[idx, 0] = float(lm.x) * width
        arr[idx, 1] = float(lm.y) * height
    return arr


def bounding_box_from_landmarks(
    landmarks: Sequence[Landmark], width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    """Obtiene ``(xmin, ymin, xmax, ymax)`` en píxeles o ``None`` si no hay datos."""

    if not landmarks:
        return None
    xy = landmarks_to_pixel_xy(landmarks, width, height)
    if xy.size == 0:
        return None
    x_min, y_min = float(np.min(xy[:, 0])), float(np.min(xy[:, 1]))
    x_max, y_max = float(np.max(xy[:, 0])), float(np.max(xy[:, 1]))
    return x_min, y_min, x_max, y_max


def expand_and_clip_box(
    bbox: Tuple[float, float, float, float], width: int, height: int, margin: float
) -> list[int]:
    """Expande ``bbox`` con el margen dado y recorta al tamaño de la imagen."""

    x_min, y_min, x_max, y_max = bbox
    dx = (x_max - x_min) * float(margin)
    dy = (y_max - y_min) * float(margin)
    x1 = max(int(np.floor(x_min - dx)), 0)
    y1 = max(int(np.floor(y_min - dy)), 0)
    x2 = min(int(np.ceil(x_max + dx)), width)
    y2 = min(int(np.ceil(y_max + dy)), height)

    if x2 <= x1:
        if x1 >= width:
            x1 = max(width - 1, 0)
        x2 = min(width, x1 + 1)
        x1 = max(0, x2 - 1)
    if y2 <= y1:
        if y1 >= height:
            y1 = max(height - 1, 0)
        y2 = min(height, y1 + 1)
        y1 = max(0, y2 - 1)

    return [x1, y1, x2, y2]


def smooth_bounding_box(
    previous: Optional[Tuple[float, float, float, float]],
    new_bbox: Tuple[float, float, float, float],
    *,
    factor: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Suaviza ``new_bbox`` frente a ``previous`` conservando el resultado en pantalla."""

    alpha = float(np.clip(factor, 0.0, 0.99))
    new_arr = np.array(new_bbox, dtype=float)
    new_arr[0::2] = np.clip(new_arr[0::2], 0.0, float(max(width, 1)))
    new_arr[1::2] = np.clip(new_arr[1::2], 0.0, float(max(height, 1)))

    if previous is None:
        blended = new_arr
    else:
        prev_arr = np.array(previous, dtype=float)
        prev_arr[0::2] = np.clip(prev_arr[0::2], 0.0, float(max(width, 1)))
        prev_arr[1::2] = np.clip(prev_arr[1::2], 0.0, float(max(height, 1)))
        blended = alpha * prev_arr + (1.0 - alpha) * new_arr

    # Garantizar un tamaño estrictamente positivo antes de convertir a enteros.
    x1, y1, x2, y2 = blended.tolist()
    if x2 <= x1:
        center_x = (x1 + x2) * 0.5
        half_width = max(1.0, (x2 - x1) * 0.5)
        x1 = np.clip(center_x - half_width, 0.0, float(width - 1 if width > 1 else 0))
        x2 = np.clip(center_x + half_width, x1 + 1.0, float(width))
    if y2 <= y1:
        center_y = (y1 + y2) * 0.5
        half_height = max(1.0, (y2 - y1) * 0.5)
        y1 = np.clip(center_y - half_height, 0.0, float(height - 1 if height > 1 else 0))
        y2 = np.clip(center_y + half_height, y1 + 1.0, float(height))

    return float(x1), float(y1), float(x2), float(y2)


def sequence_to_coordinate_arrays(
    sequence: PoseSequence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transforma una secuencia de pose en arrays ``(x, y, z, visibility)``."""

    T = len(sequence)
    n = LANDMARK_COUNT
    xs = np.full((T, n), np.nan, dtype=float)
    ys = np.full((T, n), np.nan, dtype=float)
    zs = np.full((T, n), np.nan, dtype=float)
    vs = np.full((T, n), np.nan, dtype=float)

    for t, frame in enumerate(sequence):
        if frame is None:
            continue
        for j, lm in enumerate(frame):
            try:
                xs[t, j] = float(lm.get("x", np.nan))
                ys[t, j] = float(lm.get("y", np.nan))
                zs[t, j] = float(lm.get("z", np.nan))
                visibility = lm.get("visibility", np.nan)
                vs[t, j] = float(visibility) if np.isfinite(visibility) else np.nan
            except Exception:
                continue

    return xs, ys, zs, vs


def angle_abc_deg(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> np.ndarray:
    """Calcula en vector el ángulo ABC en grados para cada fila de datos."""

    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    dot = v1x * v2x + v1y * v2y
    n1 = np.hypot(v1x, v1y)
    n2 = np.hypot(v2x, v2y)
    denom = n1 * n2
    cos = np.where(denom > 0, dot / denom, np.nan)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


__all__ = [
    "angle_abc_deg",
    "bounding_box_from_landmarks",
    "expand_and_clip_box",
    "smooth_bounding_box",
    "landmarks_from_proto",
    "landmarks_to_pixel_xy",
    "sequence_to_coordinate_arrays",
]
