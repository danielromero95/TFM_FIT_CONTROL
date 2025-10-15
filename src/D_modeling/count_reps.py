"""Utilities for counting repetitions from biomechanical metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src import config

logger = logging.getLogger(__name__)


@dataclass
class CountingDebugInfo:
    """Debug payload returned by the repetition counter."""

    valley_indices: List[int]
    prominences: List[float]


def count_reps_from_angles(
    angle_sequence: Sequence[float],
    *,
    low_thresh: float,
    high_thresh: float,
) -> int:
    """Simple state-machine counter used by legacy unit tests."""
    if not angle_sequence:
        return 0

    state = "up" if angle_sequence[0] >= high_thresh else "down"
    reps = 0
    has_reached_bottom = False

    for angle in angle_sequence:
        if state == "up":
            if angle < low_thresh:
                state = "down"
                has_reached_bottom = True
        else:  # state == "down"
            if angle >= high_thresh and has_reached_bottom:
                reps += 1
                state = "up"
                has_reached_bottom = False
    return reps


def count_reps_by_valleys(
    angle_sequence: Sequence[float],
    *,
    prominence: float,
    distance: int,
) -> Tuple[int, CountingDebugInfo]:
    """Count repetitions by detecting valleys in the angle sequence."""
    if not angle_sequence:
        return 0, CountingDebugInfo([], [])

    inverted_angles = -np.asarray(angle_sequence)
    valleys, properties = find_peaks(
        inverted_angles,
        prominence=prominence,
        distance=distance,
    )

    prominences = properties.get("prominences", [])
    debug = CountingDebugInfo(valley_indices=[int(idx) for idx in valleys], prominences=list(map(float, prominences)))
    logger.info("Detección de picos encontró %s valles válidos.", len(debug.valley_indices))
    return len(debug.valley_indices), debug


def count_repetitions_with_config(
    df_metrics: pd.DataFrame,
    counting_cfg: config.CountingConfig,
    fps: float,
) -> Tuple[int, CountingDebugInfo]:
    """Count repetitions using the configuration-driven parameters."""
    angle_column = counting_cfg.primary_angle

    if df_metrics.empty or angle_column not in df_metrics.columns:
        logger.warning(
            "La columna '%s' no se encontró o el DataFrame está vacío. Se devuelven 0 repeticiones.",
            angle_column,
        )
        return 0, CountingDebugInfo([], [])

    angles = df_metrics[angle_column].ffill().bfill().dropna().tolist()
    if not angles:
        return 0, CountingDebugInfo([], [])

    fps = fps if fps > 0 else 1.0
    distance_frames = max(1, int(round(counting_cfg.min_distance_sec * fps)))

    reps, debug = count_reps_by_valleys(
        angle_sequence=angles,
        prominence=float(counting_cfg.min_prominence),
        distance=distance_frames,
    )

    if counting_cfg.refractory_sec > 0 and debug.valley_indices:
        refractory_frames = max(1, int(round(counting_cfg.refractory_sec * fps)))
        filtered: List[int] = []
        for idx in debug.valley_indices:
            if filtered and idx - filtered[-1] < refractory_frames:
                continue
            filtered.append(idx)
        debug = CountingDebugInfo(valley_indices=filtered, prominences=debug.prominences[: len(filtered)])
        reps = len(filtered)

    return reps, debug


def count_repetitions_from_df(
    df_metrics: pd.DataFrame,
    angle_column: str = "rodilla_izq",
    low_thresh: float = config.SQUAT_LOW_THRESH,
) -> int:
    """Legacy helper preserved for backwards compatibility."""
    if df_metrics.empty or angle_column not in df_metrics.columns:
        logger.warning(
            "La columna '%s' no se encontró o el DataFrame está vacío. Se devuelven 0 repeticiones.",
            angle_column,
        )
        return 0

    angles = df_metrics[angle_column].ffill().bfill().tolist()
    if not angles:
        return 0

    reps, _ = count_reps_by_valleys(
        angle_sequence=angles,
        prominence=float(config.PEAK_PROMINENCE),
        distance=int(config.PEAK_DISTANCE),
    )
    return reps
