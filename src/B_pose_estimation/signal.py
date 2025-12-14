"""Utilidades de procesamiento de señales robustas y seguras frente a NaN."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from src.config.constants import (
    ANALYSIS_MAX_GAP_FRAMES,
    ANALYSIS_SAVGOL_POLYORDER,
    ANALYSIS_SAVGOL_WINDOW_SEC,
    ANALYSIS_SMOOTH_METHOD,
)


def _as_array(series: Iterable[float]) -> np.ndarray:
    return np.asarray(list(series), dtype=float)


def interpolate_small_gaps(series: Iterable[float], max_gap_frames: int = ANALYSIS_MAX_GAP_FRAMES) -> np.ndarray:
    """Rellena huecos cortos (NaN) mediante interpolación lineal.

    Si el hueco supera ``max_gap_frames`` se conserva como NaN para evitar
    suposiciones excesivas.
    """

    values = _as_array(series)
    n = values.size
    if n == 0 or max_gap_frames <= 0:
        return values

    isnan = ~np.isfinite(values)
    if not isnan.any():
        return values

    idx = np.arange(n)
    finite_idx = idx[~isnan]
    if finite_idx.size == 0:
        return values

    allowed = np.zeros_like(isnan, dtype=bool)
    run_start = None
    for i, flag in enumerate(isnan):
        if flag and run_start is None:
            run_start = i
        if (not flag or i == n - 1) and run_start is not None:
            end = i if not flag else i + 1
            length = end - run_start
            if length <= max_gap_frames:
                allowed[run_start:end] = True
            run_start = None

    filled = values.copy()
    if allowed.any():
        filled[allowed] = np.interp(idx[allowed], finite_idx, values[finite_idx])

    return filled


def smooth_series(
    series: Iterable[float],
    fps: float,
    *,
    method: str = ANALYSIS_SMOOTH_METHOD,
    window_seconds: float = ANALYSIS_SAVGOL_WINDOW_SEC,
    polyorder: int = ANALYSIS_SAVGOL_POLYORDER,
) -> np.ndarray:
    """Suaviza la serie conservando NaN en huecos largos.

    El tamaño de ventana se ajusta automáticamente al FPS y se limita a un
    mínimo seguro para evitar *overfitting* o desplazamientos de fase.
    """

    values = _as_array(series)
    n = values.size
    if n == 0:
        return values

    mask = np.isfinite(values)
    if mask.sum() < 3 or fps <= 0:
        return values

    filled = values.copy()
    idx = np.arange(n)
    filled[~mask] = np.interp(idx[~mask], idx[mask], values[mask])

    window = max(3, int(round(max(fps * window_seconds, 3))))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 3:
        return values

    if method == "savgol":
        if window < polyorder + 2:
            return values
        smoothed = savgol_filter(
            filled,
            window_length=int(window),
            polyorder=int(min(polyorder, window - 1)),
            mode="interp",
        )
    elif method == "butter":
        nyq = 0.5 * fps
        cutoff = min(2.0, nyq * 0.8)
        b, a = butter(N=2, Wn=cutoff / nyq, btype="low")
        smoothed = filtfilt(b, a, filled, method="gust")
    else:
        smoothed = filled

    smoothed = np.where(mask, smoothed, np.nan)
    return smoothed


def derivative(series: Iterable[float], fps: float) -> np.ndarray:
    """Calcula derivada primera robusta respetando NaN y FPS."""

    values = _as_array(series)
    n = values.size
    if n == 0:
        return values
    if fps <= 0:
        return np.zeros_like(values)

    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.zeros_like(values)

    idx = np.arange(n)
    filled = values.copy()
    filled[~mask] = np.interp(idx[~mask], idx[mask], values[mask])
    dt = 1.0 / float(fps)
    grad = np.gradient(filled, dt)
    grad = np.where(mask, grad, np.nan)
    return grad


__all__ = ["interpolate_small_gaps", "smooth_series", "derivative"]
