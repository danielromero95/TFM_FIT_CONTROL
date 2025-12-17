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


@dataclass
class CountingDebugInfo:
    """Carga de depuración que devuelve el contador de repeticiones."""
    valley_indices: List[int]
    prominences: List[float]


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
) -> Tuple[int, CountingDebugInfo]:
    finite_angles = angles[np.isfinite(angles)]
    if finite_angles.size == 0:
        return 0, CountingDebugInfo([], [])

    angle_range = float(np.nanmax(finite_angles) - np.nanmin(finite_angles))
    if not np.isfinite(angle_range) or angle_range < min_excursion:
        return 0, CountingDebugInfo([], [])

    top_thr = float(np.nanpercentile(finite_angles, 70))
    bottom_thr = float(np.nanpercentile(finite_angles, 30))
    vel_valid = np.abs(velocities[np.isfinite(velocities)])
    vel_thr = max(2.0, float(np.nanpercentile(vel_valid, 65))) if vel_valid.size else 2.0

    state = "IDLE"
    last_rep_frame = -10 * max(refractory_frames, min_distance_frames)
    last_valley_rep = -10 * max(refractory_frames, min_distance_frames)
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
            if angle <= bottom_thr:
                bottom_value = angle
                bottom_idx = idx
                state = "BOTTOM"
            elif angle >= top_thr:
                state = "TOP"
        elif state == "TOP":
            if descending:
                bottom_value = np.nan
                bottom_idx = -1
                state = "ECCENTRIC"
        elif state == "ECCENTRIC":
            if np.isnan(bottom_value) or angle < bottom_value:
                bottom_value = angle
                bottom_idx = idx
            if angle <= bottom_thr:
                state = "BOTTOM"
        elif state == "BOTTOM":
            if np.isnan(bottom_value) or angle < bottom_value:
                bottom_value = angle
                bottom_idx = idx
            if ascending:
                state = "CONCENTRIC"
        elif state == "CONCENTRIC":
            if descending and angle <= bottom_thr:
                state = "BOTTOM"

        recovered = (
            bottom_idx >= 0
            and np.isfinite(bottom_value)
            and ((angle - bottom_value >= min_prominence) or (angle >= top_thr))
        )

        if recovered and bottom_value - bottom_thr <= angle_range:
            prominence = float(np.nanmax([angle - bottom_value, top_thr - bottom_value, angle_range]))
            if prominence >= min_prominence:
                source = reference if reference is not None else angles
                search_radius = max(min_distance_frames, refractory_frames)
                start = max(0, bottom_idx - search_radius)
                end = min(len(source), bottom_idx + search_radius + 1)
                bottom_slice = source[start:end]
                if bottom_slice.size and np.isfinite(bottom_slice).any():
                    local_min = int(np.nanargmin(bottom_slice))
                    valley_idx = start + local_min
                else:
                    valley_idx = bottom_idx

                spacing_ok = (
                    idx - last_rep_frame >= max(refractory_frames, min_distance_frames)
                    and valley_idx - last_valley_rep >= min_distance_frames
                )

                if spacing_ok:
                    rep_indices.append(valley_idx)
                    prominences.append(prominence)
                    last_rep_frame = idx
                    last_valley_rep = valley_idx
            bottom_value = np.nan
            bottom_idx = -1
            state = "TOP"

    if rep_indices:
        consolidated: list[tuple[int, float]] = []
        for valley_idx, prom in sorted(zip(rep_indices, prominences), key=lambda x: x[0]):
            if consolidated:
                distance = valley_idx - consolidated[-1][0]
                similar_prominence = (
                    abs(prom - consolidated[-1][1])
                    <= 0.1 * max(prom, consolidated[-1][1], 1e-6)
                )
                if distance < min_distance_frames or (distance < 2 * min_distance_frames and similar_prominence):
                    if prom > consolidated[-1][1]:
                        consolidated[-1] = (valley_idx, prom)
                    continue
            consolidated.append((valley_idx, prom))
        rep_indices = [idx for idx, _ in consolidated]
        prominences = [prom for _, prom in consolidated]

    debug = CountingDebugInfo(valley_indices=rep_indices, prominences=prominences)
    if long_gap_seen:
        return 0, debug
    return len(rep_indices), debug


def count_repetitions_with_config(
    df_metrics: pd.DataFrame,
    counting_cfg: config.CountingConfig,
    fps: float,
    *,
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

    reps, debug = _state_machine_reps(
        angles,
        velocities,
        refractory_frames=refractory_frames,
        min_excursion=min_excursion,
        min_prominence=min_prominence,
        min_distance_frames=min_distance_frames,
        reference=reference,
    )

    logger.debug(
        "Repetition count=%d (refractory=%d frames ≈ %.2fs, min_excursion=%.1f). Valid frames=%d/%d",
        reps,
        refractory_frames,
        refractory_frames / fps_safe,
        min_excursion,
        int(np.isfinite(angles).sum()),
        len(angles),
    )
    return reps, debug
