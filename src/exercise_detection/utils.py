"""Utilidades compartidas para el módulo de detección de ejercicios."""

from __future__ import annotations

import numpy as np


def nanmean_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 and right.size == 0:
        return np.array([], dtype=float)
    if left.size == 0:
        return np.asarray(right, dtype=float)
    if right.size == 0:
        return np.asarray(left, dtype=float)
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    stacked = np.vstack([left_arr, right_arr])
    finite_mask = np.isfinite(stacked)
    counts = finite_mask.sum(axis=0)
    if not counts.any():
        return np.full(stacked.shape[1], np.nan)
    sums = np.nansum(stacked, axis=0)
    result = np.full(stacked.shape[1], np.nan)
    valid = counts > 0
    result[valid] = sums[valid] / counts[valid]
    return result
