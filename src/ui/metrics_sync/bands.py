"""Helpers for building rep phase bands in the metrics sync chart."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


def _min_phase_seconds(fps: float) -> float:
    fps_value = fps if fps > 0 else 1.0
    return max(1.0 / fps_value, 0.08)


def build_phase_bands(
    rep_splits: Iterable[Tuple[float, float, float]],
    fps: float,
    *,
    x_mode: str = "time",
) -> Tuple[List[dict], float | None]:
    """Build phase band ranges from repetition split times.

    Returns a list of dicts with ``phase``, ``x0`` and ``x1`` in axis units, plus the max
    band end (or ``None`` when no bands are present).
    """

    if not rep_splits:
        return [], None

    fps_value = fps if fps > 0 else 1.0
    min_phase_s = _min_phase_seconds(fps_value)
    min_phase_axis = min_phase_s if x_mode == "time" else min_phase_s * fps_value

    bands: List[dict] = []
    max_end: float | None = None

    durations: List[float] = []
    normalized_splits: List[Tuple[float, float, float]] = []

    for start_s, split_s, end_s in rep_splits:
        if not np.isfinite(start_s) or not np.isfinite(split_s) or not np.isfinite(end_s):
            continue
        start = start_s if x_mode == "time" else start_s * fps_value
        split = split_s if x_mode == "time" else split_s * fps_value
        end = end_s if x_mode == "time" else end_s * fps_value
        if not np.isfinite(start) or not np.isfinite(split) or not np.isfinite(end):
            continue
        if end < start:
            start, end = end, start
        split = min(max(split, start), end)

        duration = end - start
        if duration > 0:
            durations.append(duration)
        normalized_splits.append((start, split, end))

    median_duration = float(np.nanmedian(durations)) if durations else np.nan
    min_rep_axis = 2 * min_phase_axis
    if np.isfinite(median_duration):
        min_rep_axis = max(min_rep_axis, 0.25 * median_duration)

    for start, split, end in normalized_splits:
        if end - start < min_rep_axis:
            continue

        if split - start >= min_phase_axis:
            bands.append({"phase": "down", "x0": start, "x1": split})
        if end - split >= min_phase_axis:
            bands.append({"phase": "up", "x0": split, "x1": end})

        max_end = end if max_end is None else max(max_end, end)

    return bands, max_end
