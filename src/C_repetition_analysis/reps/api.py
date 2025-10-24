"""Repetition counting via valley detection with refractory-window consolidation."""

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


def _count_reps_by_valleys(
    angle_sequence: Sequence[float],
    *,
    prominence: float,
    distance: int,
) -> Tuple[int, CountingDebugInfo]:
    """Detect valleys in the angle sequence by finding peaks on the inverted signal."""
    if not angle_sequence:
        return 0, CountingDebugInfo([], [])

    inverted = -np.asarray(angle_sequence, dtype=float)
    valleys, properties = find_peaks(
        inverted,
        prominence=float(prominence),
        distance=int(max(1, distance)),
    )

    prominences = properties.get("prominences", np.array([], dtype=float))
    debug = CountingDebugInfo(
        valley_indices=[int(i) for i in valleys.tolist()],
        prominences=[float(p) for p in prominences.tolist()],
    )
    logger.debug("Valley detection found %d candidates.", len(debug.valley_indices))
    return len(debug.valley_indices), debug


def _apply_refractory_filter(
    indices: List[int],
    prominences: List[float],
    refractory_frames: int,
) -> Tuple[List[int], List[float]]:
    """Cluster valleys closer than ``refractory_frames`` and keep the most prominent in each cluster."""
    if refractory_frames <= 0 or len(indices) <= 1:
        return indices, prominences

    kept_idx: List[int] = []
    kept_prom: List[float] = []

    # Work over pairs to preserve alignment
    pairs = list(zip(indices, prominences))
    current_cluster: List[Tuple[int, float]] = [pairs[0]]

    for idx, prom in pairs[1:]:
        last_idx = current_cluster[-1][0]
        if idx - last_idx < refractory_frames:
            current_cluster.append((idx, prom))
        else:
            # flush cluster: keep the most prominent
            best = max(current_cluster, key=lambda t: t[1])
            kept_idx.append(best[0])
            kept_prom.append(best[1])
            current_cluster = [(idx, prom)]

    # flush final cluster
    if current_cluster:
        best = max(current_cluster, key=lambda t: t[1])
        kept_idx.append(best[0])
        kept_prom.append(best[1])

    return kept_idx, kept_prom


def count_repetitions_with_config(
    df_metrics: pd.DataFrame,
    counting_cfg: config.CountingConfig,
    fps: float,
) -> Tuple[int, CountingDebugInfo]:
    """
    Count repetitions using configuration-driven parameters.

    Args:
        df_metrics: DataFrame with biomechanical metrics; must contain ``counting_cfg.primary_angle``.
        counting_cfg: instance of ``src.config.models.CountingConfig``.
        fps: effective frames-per-second of the metric sequence.

    Returns:
        (repetition_count, CountingDebugInfo)
    """
    angle_column = counting_cfg.primary_angle

    if df_metrics.empty or angle_column not in df_metrics.columns:
        logger.warning(
            "Column '%s' was not found or the DataFrame is empty. Returning 0 repetitions.",
            angle_column,
        )
        return 0, CountingDebugInfo([], [])

    # Forward/back-fill to mitigate short NaN spans, then drop any remaining NaNs
    angles = df_metrics[angle_column].ffill().bfill().dropna().tolist()
    if not angles:
        return 0, CountingDebugInfo([], [])

    fps_safe = float(fps) if fps and fps > 0 else 1.0
    distance_frames = max(1, int(round(counting_cfg.min_distance_sec * fps_safe)))
    prominence_thr = float(counting_cfg.min_prominence)

    reps, debug = _count_reps_by_valleys(
        angle_sequence=angles,
        prominence=prominence_thr,
        distance=distance_frames,
    )

    refractory_frames = max(0, int(round(float(counting_cfg.refractory_sec) * fps_safe)))
    if refractory_frames > 0 and debug.valley_indices:
        filtered_idx, filtered_prom = _apply_refractory_filter(
            debug.valley_indices, debug.prominences, refractory_frames
        )
        debug = CountingDebugInfo(valley_indices=filtered_idx, prominences=filtered_prom)
        reps = len(filtered_idx)

    logger.debug(
        "Repetition count=%d (distance=%d frames, prominence>=%.3f, refractory=%d frames).",
        reps, distance_frames, prominence_thr, refractory_frames
    )
    return reps, debug
