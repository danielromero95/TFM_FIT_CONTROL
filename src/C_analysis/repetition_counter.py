"""Conteo de repeticiones mediante detección de valles y consolidación por periodo refractario."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src import config
from src.B_pose_estimation.signal import derivative, interpolate_small_gaps, smooth_series
from src.config.constants import (
    ANALYSIS_MAX_GAP_FRAMES,
    ANALYSIS_SAVGOL_POLYORDER,
    ANALYSIS_SAVGOL_WINDOW_SEC,
    ANALYSIS_SMOOTH_METHOD,
)

logger = logging.getLogger(__name__)
EPS_DEG = 0.0


@dataclass
class CountingDebugInfo:
    """Carga de depuración que devuelve el contador de repeticiones."""
    valley_indices: List[int]
    prominences: List[float]
    raw_count: int = 0
    reps_rejected_threshold: int = 0
    rejection_reasons: List[str] | None = None


def _quality_mask_from_df(df_metrics: pd.DataFrame, column: str) -> np.ndarray:
    mask = np.isfinite(pd.to_numeric(df_metrics[column], errors="coerce"))
    if "pose_ok" in df_metrics.columns:
        pose_ok = pd.to_numeric(df_metrics["pose_ok"], errors="coerce") >= 0.5
        mask &= pose_ok.to_numpy(dtype=bool, na_value=False)
    return mask.to_numpy(dtype=bool) if hasattr(mask, "to_numpy") else np.asarray(mask, dtype=bool)


def _prepare_primary_series(
    df_metrics: pd.DataFrame, column: str, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    series = pd.to_numeric(df_metrics[column], errors="coerce")
    quality_mask = _quality_mask_from_df(df_metrics, column)
    gated = series.where(quality_mask, np.nan)
    reference = gated.to_numpy(dtype=float, copy=True)
    interpolated = interpolate_small_gaps(gated, ANALYSIS_MAX_GAP_FRAMES)
    smoothed = smooth_series(
        interpolated,
        fps,
        method=ANALYSIS_SMOOTH_METHOD,
        window_seconds=ANALYSIS_SAVGOL_WINDOW_SEC,
        polyorder=ANALYSIS_SAVGOL_POLYORDER,
    )
    smoothed = np.where(quality_mask, smoothed, np.nan)
    velocity = derivative(smoothed, fps)
    return smoothed, velocity, quality_mask, reference


def _state_machine_reps(
    angles: np.ndarray,
    velocities: np.ndarray,
    *,
    refractory_frames: int,
    min_excursion: float,
    min_prominence: float,
    min_distance_frames: int,
    reference: np.ndarray | None = None,
) -> tuple[list[int], list[float], bool]:
    finite_angles = angles[np.isfinite(angles)]
    if finite_angles.size == 0:
        return [], [], False

    angle_range = float(np.nanmax(finite_angles) - np.nanmin(finite_angles))
    if not np.isfinite(angle_range) or angle_range < min_excursion:
        return [], [], False

    top_thr = float(np.nanpercentile(finite_angles, 70))
    bottom_thr = float(np.nanpercentile(finite_angles, 30))
    vel_valid = np.abs(velocities[np.isfinite(velocities)])
    vel_thr = max(2.0, float(np.nanpercentile(vel_valid, 65))) if vel_valid.size else 2.0

    state = "IDLE"
    last_rep_frame = -10 * refractory_frames
    rep_indices: list[int] = []
    prominences: list[float] = []
    bottom_value = np.nan
    bottom_idx = -1
    invalid_run = 0
    long_gap_seen = False

    for idx, (angle, vel) in enumerate(zip(angles, velocities)):
        if not np.isfinite(angle):
            invalid_run += 1
            if invalid_run > ANALYSIS_MAX_GAP_FRAMES:
                state = "IDLE"
                bottom_value = np.nan
                bottom_idx = -1
                long_gap_seen = True
            continue

        invalid_run = 0
        descending = (np.isfinite(vel) and vel < -vel_thr) or (np.isfinite(angle) and angle < top_thr)
        ascending = (np.isfinite(vel) and vel > vel_thr) or (np.isfinite(angle) and angle > bottom_thr)

        if state == "IDLE":
            if angle >= top_thr:
                state = "TOP"
            elif angle <= bottom_thr:
                state = "BOTTOM"
        elif state == "TOP":
            if descending:
                state = "ECCENTRIC"
        elif state == "ECCENTRIC":
            if angle <= bottom_thr:
                bottom_value = angle
                bottom_idx = idx
                state = "BOTTOM"
        elif state == "BOTTOM":
            if ascending:
                state = "CONCENTRIC"
        elif state == "CONCENTRIC":
            if angle >= top_thr:
                if idx - last_rep_frame >= max(refractory_frames, min_distance_frames):
                    prominence = float(np.nanmax([top_thr - bottom_value, angle_range]))
                    if prominence >= min_prominence:
                        source = reference if reference is not None else angles
                        start = max(0, last_rep_frame + 1)
                        bottom_window = source[start : idx + 1]
                        if bottom_window.size:
                            local_min = int(np.nanargmin(bottom_window))
                            valley_idx = start + local_min
                        else:
                            valley_idx = bottom_idx if bottom_idx >= 0 else idx
                        rep_indices.append(valley_idx)
                        prominences.append(prominence)
                        last_rep_frame = idx
                state = "TOP"

    if long_gap_seen:
        return [], [], True
    return rep_indices, prominences, False


def _as_optional_float(value: object) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _rep_intervals_from_valleys(series_length: int, valley_indices: list[int]) -> list[tuple[int, int]]:
    if series_length <= 0 or not valley_indices:
        return []

    bounded = [max(0, min(int(idx), series_length - 1)) for idx in valley_indices]
    if not bounded:
        return []

    intervals: list[tuple[int, int]] = []
    for i, valley in enumerate(bounded):
        start = 0 if i == 0 else int(round((bounded[i - 1] + valley) / 2))
        end = series_length - 1 if i == len(bounded) - 1 else int(round((valley + bounded[i + 1]) / 2))
        start = max(0, min(start, series_length - 1))
        end = max(start, min(end, series_length - 1))
        intervals.append((start, end))
    return intervals


def _filter_reps_by_thresholds(
    angles: np.ndarray,
    valley_indices: list[int],
    prominences: list[float],
    *,
    low_thresh: float | None,
    high_thresh: float | None,
    enforce_low: bool = True,
    enforce_high: bool = True,
) -> tuple[list[int], list[float], int, list[str]]:
    apply_low = enforce_low and low_thresh is not None
    apply_high = enforce_high and high_thresh is not None
    if not apply_low and not apply_high:
        return valley_indices, prominences, 0, []

    intervals = _rep_intervals_from_valleys(len(angles), valley_indices)
    filtered_indices: list[int] = []
    filtered_proms: list[float] = []
    rejection_reasons: list[str] = []

    for idx, prominence, interval in zip(valley_indices, prominences, intervals):
        start, end = interval
        window = angles[start : end + 1]
        finite = window[np.isfinite(window)]
        if finite.size < 2:
            rejection_reasons.append(
                f"Rep at {idx} discarded: insufficient finite samples between {start} and {end}."
            )
            continue

        min_angle = float(np.nanmin(finite))
        max_angle = float(np.nanmax(finite))

        low_ok = (not apply_low) or (min_angle <= low_thresh)  # type: ignore[operator]
        high_ok = (not apply_high) or (max_angle >= high_thresh)  # type: ignore[operator]

        if low_ok and high_ok:
            filtered_indices.append(idx)
            filtered_proms.append(prominence)
            continue

        if not low_ok and not high_ok:
            reason = (
                f"Rep at {idx} discarded: min {min_angle:.1f}° above low_thresh {low_thresh}°"
                f" and max {max_angle:.1f}° below high_thresh {high_thresh}°."
            )
        elif not low_ok:
            reason = f"Rep at {idx} discarded: min {min_angle:.1f}° above low_thresh {low_thresh}°."
        else:
            reason = f"Rep at {idx} discarded: max {max_angle:.1f}° below high_thresh {high_thresh}°."
        rejection_reasons.append(reason)

    rejected = len(valley_indices) - len(filtered_indices)
    return filtered_indices, filtered_proms, rejected, rejection_reasons


def count_repetitions_with_config(
    df_metrics: pd.DataFrame,
    counting_cfg: config.CountingConfig,
    fps: float,
    *,
    faults_cfg: config.FaultConfig | None = None,
    overrides: dict[str, float] | None = None,
) -> Tuple[int, CountingDebugInfo]:
    """
    Cuenta repeticiones empleando los parámetros definidos en la configuración.

    Args:
        df_metrics: DataFrame con las métricas biomecánicas; debe contener ``counting_cfg.primary_angle``.
        counting_cfg: instancia de ``src.config.models.CountingConfig`` con los umbrales vigentes.
        fps: fotogramas por segundo efectivos de la serie sobre la que se cuenta.

    Returns:
        Tupla ``(repetition_count, CountingDebugInfo)`` con el total y la información de depuración.

    Keyword Args:
        overrides: diccionario opcional con claves ``min_angle_excursion_deg`` y ``refractory_sec`` para
            ajustar temporalmente los umbrales sin mutar la configuración original.
    """
    angle_column = counting_cfg.primary_angle

    if df_metrics.empty or angle_column not in df_metrics.columns:
        logger.warning(
            "Column '%s' was not found or the DataFrame is empty. Returning 0 repetitions.",
            angle_column,
        )
        return 0, CountingDebugInfo([], [])

    overrides = overrides or {}

    fps_safe = float(fps) if fps and fps > 0 else 1.0
    refractory_sec = float(overrides.get("refractory_sec", counting_cfg.refractory_sec))
    refractory_frames = max(1, int(round(refractory_sec * fps_safe)))
    min_excursion = float(overrides.get("min_angle_excursion_deg", counting_cfg.min_angle_excursion_deg))
    min_prominence = float(overrides.get("min_prominence", counting_cfg.min_prominence))
    min_distance_sec = float(overrides.get("min_distance_sec", counting_cfg.min_distance_sec))
    min_distance_frames = max(1, int(round(min_distance_sec * fps_safe)))

    angles, velocities, _, reference = _prepare_primary_series(df_metrics, angle_column, fps_safe)
    if np.isfinite(angles).sum() < 3:
        return 0, CountingDebugInfo([], [])

    rep_indices, prominences, long_gap_seen = _state_machine_reps(
        angles,
        velocities,
        refractory_frames=refractory_frames,
        min_excursion=min_excursion,
        min_prominence=min_prominence,
        min_distance_frames=min_distance_frames,
        reference=reference,
    )

    raw_count = len(rep_indices)
    low_thresh = _as_optional_float(getattr(faults_cfg, "low_thresh", None) if faults_cfg else None)
    high_thresh = _as_optional_float(getattr(faults_cfg, "high_thresh", None) if faults_cfg else None)
    enforce_low = bool(getattr(counting_cfg, "enforce_low_thresh", False))
    enforce_high = bool(getattr(counting_cfg, "enforce_high_thresh", False))

    filtered_indices, filtered_proms, rejected, reasons = _filter_reps_by_thresholds(
        reference if reference is not None else angles,
        rep_indices,
        prominences,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        enforce_low=enforce_low,
        enforce_high=enforce_high,
    )

    debug = CountingDebugInfo(
        valley_indices=filtered_indices,
        prominences=filtered_proms,
        raw_count=raw_count,
        reps_rejected_threshold=rejected,
        rejection_reasons=reasons,
    )

    if long_gap_seen:
        return 0, debug

    reps = len(filtered_indices)

    logger.debug(
        "Repetition count=%d (raw=%d, refractory=%d frames ≈ %.2fs, min_excursion=%.1f). Valid frames=%d/%d",
        reps,
        raw_count,
        refractory_frames,
        refractory_frames / fps_safe,
        min_excursion,
        int(np.isfinite(angles).sum()),
        len(angles),
    )
    return reps, debug
