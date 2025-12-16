from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def maybe_convert_radians_to_degrees(values: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Convert series expressed in radians to degrees when it is unambiguous.

    The heuristics only trigger when the magnitude clearly fits a radian range
    (roughly [-pi, pi]) and the values are not trivially close to zero. This
    avoids altering genuine degree measurements that happen to be small.
    """

    arr = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return arr, False

    finite_values = arr[finite_mask]
    max_abs = float(np.nanmax(np.abs(finite_values)))
    median_abs = float(np.nanmedian(np.abs(finite_values)))

    if (
        max_abs <= math.pi + 0.1
        and max_abs >= 0.35
        and median_abs <= (math.pi / 2 + 0.1)
    ):
        return np.degrees(arr), True

    return arr, False


def apply_warmup_mask(values: np.ndarray, warmup_frames: int) -> Tuple[np.ndarray, int]:
    """Mask the first ``warmup_frames`` entries with ``NaN``.

    Returns the masked array and the count of values that were overwritten.
    """

    arr = np.asarray(values, dtype=float).copy()
    if warmup_frames <= 0:
        return arr, 0

    applied = min(int(warmup_frames), arr.size)
    arr[:applied] = np.nan
    return arr, applied


def suppress_spikes(values: np.ndarray, threshold_deg: float) -> Tuple[np.ndarray, int]:
    """Remove isolated spikes that deviate more than ``threshold_deg`` degrees.

    A spike is defined as a value that differs from the average of its immediate
    neighbours by more than the threshold. The function returns the cleaned
    array and the number of spikes that were replaced with ``NaN``.
    """

    arr = np.asarray(values, dtype=float).copy()
    if arr.size < 3:
        return arr, 0

    removed = 0
    finite = np.isfinite(arr)
    for idx in range(1, arr.size - 1):
        if not (finite[idx - 1] and finite[idx] and finite[idx + 1]):
            continue

        neighbour_mean = (arr[idx - 1] + arr[idx + 1]) * 0.5
        neighbour_gap = abs(arr[idx - 1] - arr[idx + 1])
        if abs(arr[idx] - neighbour_mean) > threshold_deg and neighbour_gap < threshold_deg:
            arr[idx] = np.nan
            removed += 1

    return arr, removed


__all__ = [
    "maybe_convert_radians_to_degrees",
    "apply_warmup_mask",
    "suppress_spikes",
]
