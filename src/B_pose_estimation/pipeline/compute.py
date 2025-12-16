"""Cálculo de métricas biomecánicas a partir de secuencias de marcadores corporales."""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from ..constants import (
    HIP_CENTER,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_ANKLE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_ANKLE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from ..geometry import angle_abc_deg, sequence_to_coordinate_arrays
from ..signal import derivative, interpolate_small_gaps, smooth_series
from src.config.constants import (
    ANALYSIS_MAX_GAP_FRAMES,
    ANALYSIS_SAVGOL_POLYORDER,
    ANALYSIS_SAVGOL_WINDOW_SEC,
    ANALYSIS_SMOOTH_METHOD,
)
from ..types import PoseSequence
from src.utils.angles import maybe_convert_radians_to_degrees, suppress_spikes

logger = logging.getLogger(__name__)


ANGLE_COLUMNS = [
    "left_knee",
    "right_knee",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
]


def calculate_metrics_from_sequence(
    sequence: PoseSequence,
    fps: float,
    *,
    smooth_window: int | None = None,
    sg_poly: int | None = None,
    vel_method: str = "forward",
    quality_mask: Optional[Iterable[bool]] = None,
    warmup_seconds: float | None = None,
    warmup_frames: int | None = None,
    spike_threshold_deg: float = 55.0,
) -> pd.DataFrame:
    """Calcula métricas biomecánicas para la secuencia de marcadores proporcionada."""

    logger.info("Computing metrics for a sequence of %d frames.", len(sequence))
    if len(sequence) == 0:
        return pd.DataFrame([])

    xs, ys, _zs, _vs = sequence_to_coordinate_arrays(sequence)

    hip_cx = (xs[:, HIP_CENTER[0]] + xs[:, HIP_CENTER[1]]) / 2.0
    hip_cy = (ys[:, HIP_CENTER[0]] + ys[:, HIP_CENTER[1]]) / 2.0
    x_norm = xs - hip_cx[:, None]
    y_norm = ys - hip_cy[:, None]

    left_hip_angle = angle_abc_deg(
        x_norm[:, LEFT_SHOULDER],
        y_norm[:, LEFT_SHOULDER],
        x_norm[:, LEFT_HIP],
        y_norm[:, LEFT_HIP],
        x_norm[:, LEFT_KNEE],
        y_norm[:, LEFT_KNEE],
    )
    right_hip_angle = angle_abc_deg(
        x_norm[:, RIGHT_SHOULDER],
        y_norm[:, RIGHT_SHOULDER],
        x_norm[:, RIGHT_HIP],
        y_norm[:, RIGHT_HIP],
        x_norm[:, RIGHT_KNEE],
        y_norm[:, RIGHT_KNEE],
    )

    shoulder_mid_x = (x_norm[:, LEFT_SHOULDER] + x_norm[:, RIGHT_SHOULDER]) * 0.5
    shoulder_mid_y = (y_norm[:, LEFT_SHOULDER] + y_norm[:, RIGHT_SHOULDER]) * 0.5
    hip_mid_x = (x_norm[:, HIP_CENTER[0]] + x_norm[:, HIP_CENTER[1]]) * 0.5
    hip_mid_y = (y_norm[:, HIP_CENTER[0]] + y_norm[:, HIP_CENTER[1]]) * 0.5
    dx = np.abs(shoulder_mid_x - hip_mid_x)
    dy = np.abs(shoulder_mid_y - hip_mid_y)
    trunk_inclination_deg = np.degrees(np.arctan2(dx, dy + 1e-6))

    left_knee_angle = angle_abc_deg(
        x_norm[:, LEFT_HIP],
        y_norm[:, LEFT_HIP],
        x_norm[:, LEFT_KNEE],
        y_norm[:, LEFT_KNEE],
        x_norm[:, LEFT_ANKLE],
        y_norm[:, LEFT_ANKLE],
    )
    right_knee_angle = angle_abc_deg(
        x_norm[:, RIGHT_HIP],
        y_norm[:, RIGHT_HIP],
        x_norm[:, RIGHT_KNEE],
        y_norm[:, RIGHT_KNEE],
        x_norm[:, RIGHT_ANKLE],
        y_norm[:, RIGHT_ANKLE],
    )
    left_elbow_angle = angle_abc_deg(
        x_norm[:, LEFT_SHOULDER],
        y_norm[:, LEFT_SHOULDER],
        x_norm[:, LEFT_ELBOW],
        y_norm[:, LEFT_ELBOW],
        x_norm[:, LEFT_WRIST],
        y_norm[:, LEFT_WRIST],
    )
    right_elbow_angle = angle_abc_deg(
        x_norm[:, RIGHT_SHOULDER],
        y_norm[:, RIGHT_SHOULDER],
        x_norm[:, RIGHT_ELBOW],
        y_norm[:, RIGHT_ELBOW],
        x_norm[:, RIGHT_WRIST],
        y_norm[:, RIGHT_WRIST],
    )

    shoulder_width = np.abs(x_norm[:, RIGHT_SHOULDER] - x_norm[:, LEFT_SHOULDER])
    foot_separation = np.abs(x_norm[:, RIGHT_ANKLE] - x_norm[:, LEFT_ANKLE])

    frame_idx = np.arange(len(sequence), dtype=int)
    dfm = pd.DataFrame(
        {
            "frame_idx": frame_idx,
            "left_knee": left_knee_angle,
            "right_knee": right_knee_angle,
            "left_elbow": left_elbow_angle,
            "right_elbow": right_elbow_angle,
            "left_hip": left_hip_angle,
            "right_hip": right_hip_angle,
            "trunk_inclination_deg": trunk_inclination_deg,
            "shoulder_width": shoulder_width,
            "foot_separation": foot_separation,
        }
    )

    cleaning_meta: Dict[str, Dict[str, float | bool]] = {}
    effective_warmup = 0
    if warmup_frames is not None:
        effective_warmup = max(0, int(warmup_frames))
    elif warmup_seconds is not None and fps > 0:
        effective_warmup = max(0, int(math.ceil(fps * warmup_seconds)))

    for column in ANGLE_COLUMNS:
        cleaned, spikes_removed = suppress_spikes(dfm[column], spike_threshold_deg)

        dfm[column] = cleaned
        cleaning_meta[column] = {
            "converted_from_rad": False,
            "warmup_masked": int(effective_warmup),
            "spikes_removed": int(spikes_removed),
        }

    trunk_cleaned, converted = maybe_convert_radians_to_degrees(dfm["trunk_inclination_deg"])
    trunk_cleaned, spikes_removed = suppress_spikes(trunk_cleaned, spike_threshold_deg)
    dfm["trunk_inclination_deg"] = trunk_cleaned
    cleaning_meta["trunk_inclination_deg"] = {
        "converted_from_rad": bool(converted),
        "warmup_masked": int(effective_warmup),
        "spikes_removed": int(spikes_removed),
    }

    valid_mask = np.ones(len(dfm), dtype=bool)
    if hasattr(sequence, "__len__") and len(dfm) != len(sequence):
        valid_mask[:] = False

    def _normalize_mask(mask: Optional[Iterable[bool]]) -> np.ndarray:
        if mask is None:
            return valid_mask
        try:
            arr = np.asarray(list(mask), dtype=bool)
        except Exception:
            return valid_mask
        if arr.size != len(dfm):
            return valid_mask
        return arr

    preferred_mask = quality_mask if quality_mask is not None else getattr(sequence, "quality_mask", None)
    valid_mask = _normalize_mask(preferred_mask)

    if effective_warmup > 0 and len(valid_mask):
        warmup = min(int(effective_warmup), valid_mask.size)
        valid_mask[:warmup] = False

    raw_angles = dfm[ANGLE_COLUMNS].copy()
    for column in ANGLE_COLUMNS:
        dfm[f"raw_{column}"] = np.where(valid_mask, raw_angles[column], np.nan)

    window_seconds = ANALYSIS_SAVGOL_WINDOW_SEC if smooth_window is None else max(smooth_window / max(fps, 1e-6), 0.0)
    polyorder = ANALYSIS_SAVGOL_POLYORDER if sg_poly is None else sg_poly
    smoothing_meta: Dict[str, float | int | str | None] = {
        "method": ANALYSIS_SMOOTH_METHOD,
        "window": int(max(3, round(fps * window_seconds))) if fps > 0 else None,
        "poly": polyorder,
        "vel_method": vel_method,
    }

    for column in ANGLE_COLUMNS:
        series = pd.Series(np.where(valid_mask, dfm[column], np.nan), index=dfm.index)
        interpolated = interpolate_small_gaps(series, ANALYSIS_MAX_GAP_FRAMES)
        smoothed = smooth_series(
            interpolated,
            fps,
            method=ANALYSIS_SMOOTH_METHOD,
            window_seconds=window_seconds,
            polyorder=int(polyorder),
        )
        smoothed = np.where(valid_mask, smoothed, np.nan)
        dfm[column] = smoothed
        if vel_method == "diff":
            diffs = np.diff(smoothed, prepend=np.nan)
            dfm[f"ang_vel_{column}"] = diffs * float(fps)
        else:
            dfm[f"ang_vel_{column}"] = derivative(smoothed, fps)

    dfm.attrs["smoothing"] = smoothing_meta
    dfm.attrs["angle_cleaning"] = cleaning_meta
    dfm.attrs["warmup_frames_applied"] = int(max(0, effective_warmup))

    def _symmetry(col_left: str, col_right: str) -> np.ndarray:
        L = raw_angles[col_left].to_numpy()
        R = raw_angles[col_right].to_numpy()
        max_lr = np.maximum(np.abs(L), np.abs(R))
        nan_mask = ~np.isfinite(L) | ~np.isfinite(R)
        both_zero = max_lr == 0
        return np.where(
            nan_mask,
            np.nan,
            np.where(both_zero, 1.0, 1.0 - (np.abs(L - R) / max_lr)),
        )

    dfm["knee_symmetry"] = _symmetry("left_knee", "right_knee")
    dfm["elbow_symmetry"] = _symmetry("left_elbow", "right_elbow")

    dfm["pose_ok"] = valid_mask.astype(float)

    dfm.loc[~valid_mask, ["trunk_inclination_deg", "shoulder_width", "foot_separation"]] = np.nan

    return dfm


__all__ = ["calculate_metrics_from_sequence"]
