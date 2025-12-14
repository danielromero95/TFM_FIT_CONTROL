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


def _prepare_primary_series(df_metrics: pd.DataFrame, column: str, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = pd.to_numeric(df_metrics[column], errors="coerce")
    quality_mask = _quality_mask_from_df(df_metrics, column)
    gated = series.where(quality_mask, np.nan)
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
    return smoothed, velocity, quality_mask


def _state_machine_reps(
    angles: np.ndarray,
    velocities: np.ndarray,
    *,
    refractory_frames: int,
    min_excursion: float,
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
    vel_thr = max(5.0, float(np.nanpercentile(vel_valid, 65))) if vel_valid.size else 5.0

    state = "IDLE"
    last_rep_frame = -10 * refractory_frames
    rep_indices: list[int] = []
    prominences: list[float] = []
    bottom_value = np.nan
    invalid_run = 0

    for idx, (angle, vel) in enumerate(zip(angles, velocities)):
        if not np.isfinite(angle):
            invalid_run += 1
            if invalid_run > ANALYSIS_MAX_GAP_FRAMES:
                state = "IDLE"
            continue

        invalid_run = 0
        descending = np.isfinite(vel) and vel < -vel_thr
        ascending = np.isfinite(vel) and vel > vel_thr

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
                state = "BOTTOM"
        elif state == "BOTTOM":
            if ascending:
                state = "CONCENTRIC"
        elif state == "CONCENTRIC":
            if angle >= top_thr:
                if idx - last_rep_frame >= refractory_frames:
                    rep_indices.append(idx)
                    prominence = float(np.nanmax([top_thr - bottom_value, angle_range]))
                    prominences.append(prominence)
                    last_rep_frame = idx
                state = "TOP"

    debug = CountingDebugInfo(valley_indices=rep_indices, prominences=prominences)
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

    angles, velocities, _ = _prepare_primary_series(df_metrics, angle_column, fps_safe)
    if np.isfinite(angles).sum() < 3:
        return 0, CountingDebugInfo([], [])

    reps, debug = _state_machine_reps(
        angles,
        velocities,
        refractory_frames=refractory_frames,
        min_excursion=min_excursion,
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
