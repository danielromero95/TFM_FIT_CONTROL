"""Utilidades de suavizado de señales utilizadas en el clasificador de ejercicios."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from .constants import (
    DEFAULT_SAMPLING_RATE,
    SMOOTHING_POLY_ORDER,
    SMOOTHING_WINDOW_SECONDS,
)


def linear_fill_nan(series: np.ndarray) -> np.ndarray:
    """Rellena los NaN mediante interpolación lineal conservando los extremos faltantes."""

    if series.size == 0:
        return series.copy()

    arr = np.asarray(series, dtype=float)
    mask = np.isfinite(arr)
    if mask.all():
        return arr.copy()
    if not mask.any():
        return arr.copy()

    indices = np.arange(arr.size)
    filled = arr.copy()
    filled[~mask] = np.interp(indices[~mask], indices[mask], arr[mask])
    return filled


def smooth(series: np.ndarray, sr: Optional[float] = None) -> np.ndarray:
    """Aplica suavizado de Savitzky–Golay ajustando la ventana de forma adaptativa."""

    arr = np.asarray(series, dtype=float)
    if arr.size < 3:
        return arr.copy()

    sampling_rate = float(sr or DEFAULT_SAMPLING_RATE)
    window = int(max(5, round(sampling_rate * SMOOTHING_WINDOW_SECONDS)))
    if window % 2 == 0:
        window += 1
    window = min(window, arr.size - (1 - arr.size % 2))
    if window <= SMOOTHING_POLY_ORDER:
        window = SMOOTHING_POLY_ORDER + 2 + (SMOOTHING_POLY_ORDER + 2) % 2
    window = min(window, arr.size - (1 - arr.size % 2))
    if window < 3:
        return arr.copy()

    mask = np.isfinite(arr)
    if mask.sum() < 3:
        return arr.copy()

    filled = linear_fill_nan(arr)
    smoothed = savgol_filter(filled, window_length=window, polyorder=SMOOTHING_POLY_ORDER, mode="interp")
    smoothed[~mask] = np.nan
    return smoothed

