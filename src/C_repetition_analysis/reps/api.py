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

    # Work with numpy arrays; inputs are expected in ascending order.
    idx = np.asarray(indices, dtype=np.int64)
    prom = np.asarray(prominences, dtype=np.float64)

    # Identify cluster starts where the gap is >= refractory_frames
    diffs = np.diff(idx)
    starts = np.r_[0, np.flatnonzero(diffs >= int(refractory_frames)) + 1]
    ends = np.r_[starts[1:], idx.size]

    kept_idx: list[int] = []
    kept_prom: list[float] = []

    # Iterate per cluster (number of clusters << number of points typically)
    for s, e in zip(starts, ends):
        # Argmax returns first occurrence on ties → deterministic earliest index in cluster
        local = s + int(np.argmax(prom[s:e]))
        kept_idx.append(int(idx[local]))
        kept_prom.append(float(prom[local]))

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
