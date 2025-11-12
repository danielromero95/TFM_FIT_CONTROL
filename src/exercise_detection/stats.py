"""Utilidades numéricas que evitan *RuntimeWarnings* cuando todo son NaN."""

from __future__ import annotations

import numpy as np


def _as_float_array(arr: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(arr, dtype=float)


def _all_nan(arr: np.ndarray) -> bool:
    return arr.size == 0 or not np.isfinite(arr).any()


def safe_nanmedian(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanmedian(array))


def safe_nanmean(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanmean(array))


def safe_nanstd(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanstd(array))


def safe_nanpercentile(arr: np.ndarray | list[float] | tuple[float, ...], q: float) -> float:
    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanpercentile(array, q))


def safe_nanmax(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanmax(array))


def safe_nanmin(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanmin(array))


__all__ = [
    "safe_nanmedian",
    "safe_nanmean",
    "safe_nanstd",
    "safe_nanpercentile",
    "safe_nanmax",
    "safe_nanmin",
]
