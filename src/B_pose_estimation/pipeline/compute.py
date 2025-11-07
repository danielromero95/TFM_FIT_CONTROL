"""Computation of biomechanical metrics from pose sequences."""

from __future__ import annotations

import logging
from typing import Dict

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
from ..metrics.timeseries import angular_velocity
from ..types import PoseSequence

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
    smooth_window: int = 7,
    sg_poly: int = 2,
    vel_method: str = "forward",
) -> pd.DataFrame:
    """Compute biomechanical metrics for the provided landmark sequence."""

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

    raw_angles = dfm[ANGLE_COLUMNS].copy()
    for column in ANGLE_COLUMNS:
        dfm[f"raw_{column}"] = raw_angles[column]

    smoothing_meta: Dict[str, float | int | str | None] = {
        "method": "none",
        "window": None,
        "poly": None,
    }

    savgol_available = False
    savgol_filter = None
    if smooth_window >= 5:
        try:
            from scipy.signal import savgol_filter as _savgol_filter

            savgol_filter = _savgol_filter
            savgol_available = True
        except Exception:  # pragma: no cover
            savgol_available = False

    def _maybe_apply_savgol(series: pd.Series) -> tuple[pd.Series, tuple[str, int, int] | None]:
        if not savgol_available or savgol_filter is None:
            return series, None

        values = series.to_numpy(dtype=float)
        n = values.size
        if n < 5:
            return series, None

        window = min(smooth_window, n if n % 2 == 1 else n - 1)
        if window < 5:
            return series, None
        if window % 2 == 0:
            window = max(5, window - 1)
        if window > n:
            window = n if n % 2 == 1 else n - 1
        if window < 5:
            return series, None

        poly = min(max(1, sg_poly), window - 1)
        if poly >= window:
            poly = window - 1
        if poly < 1:
            return series, None

        try:
            filtered = savgol_filter(values, window_length=window, polyorder=poly, mode="interp")
        except ValueError:
            return series, None

        return pd.Series(filtered, index=series.index), ("savgol", int(window), int(poly))

    def _apply_smoothing(series: pd.Series) -> tuple[pd.Series, tuple[str, int, int] | None]:
        if smooth_window >= 5:
            smoothed, desc = _maybe_apply_savgol(series)
            if desc is not None:
                return smoothed, desc
        if smooth_window >= 3:
            rolled = series.rolling(smooth_window, center=True, min_periods=1).mean()
            return rolled, ("rolling_mean", int(smooth_window), 0)
        return series, None

    for column in ANGLE_COLUMNS:
        interpolated = dfm[column].interpolate(method="linear", limit_direction="both")
        smoothed_series, desc = _apply_smoothing(interpolated)
        dfm[column] = smoothed_series
        if desc is not None:
            smoothing_meta = {
                "method": desc[0],
                "window": desc[1],
                "poly": desc[2] if desc[0] == "savgol" else None,
            }
        dfm[f"ang_vel_{column}"] = angular_velocity(dfm[column].to_numpy(), fps, method=vel_method)

    dfm.attrs["smoothing"] = smoothing_meta

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

    return dfm


__all__ = ["calculate_metrics_from_sequence"]
