"""Estadísticos robustos que evitan *warnings* con entradas llenas de NaN."""

from __future__ import annotations

import numpy as np


def _as_float_array(arr: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convierte cualquier contenedor numérico en un ``ndarray`` de ``float``."""

    return np.asarray(arr, dtype=float)


def _all_nan(arr: np.ndarray) -> bool:
    """Indica si el array está vacío o no contiene valores finitos."""

    return arr.size == 0 or not np.isfinite(arr).any()


def safe_nanmedian(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """Calcula la mediana ignorando NaN y devuelve NaN si la entrada es inválida."""

    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanmedian(array))


def safe_nanmean(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """Calcula la media sin NaN y conserva NaN si no hay datos válidos."""

    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanmean(array))


def safe_nanstd(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """Calcula la desviación típica evitando *warnings* por columnas vacías."""

    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanstd(array))


def safe_nanpercentile(arr: np.ndarray | list[float] | tuple[float, ...], q: float) -> float:
    """Obtiene el percentil ``q`` sin fallar cuando la serie sólo contiene NaN."""

    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanpercentile(array, q))


def safe_nanmax(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """Devuelve el máximo ignorando NaN, o NaN si no hay valores válidos."""

    array = _as_float_array(arr)
    if _all_nan(array):
        return float("nan")
    return float(np.nanmax(array))


def safe_nanmin(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """Devuelve el mínimo ignorando NaN, o NaN si la serie es inválida."""

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
