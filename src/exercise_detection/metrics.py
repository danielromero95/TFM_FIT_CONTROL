"""Cálculo de características por repetición y estadísticas agregadas."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .constants import ARM_ABOVE_HIP_THRESH, BAR_NEAR_SHOULDER_THRESH
from .stats import safe_nanmax, safe_nanmedian, safe_nanmin, safe_nanstd
from .types import AggregateMetrics, RepMetrics, RepSlice
from .utils import nanmean_pair


BOTTOM_WINDOW_DEG = 10.0
BOTTOM_MIN_FRAMES = 5
BOTTOM_FRACTION = 0.15


def compute_metrics(
    slices: Sequence[RepSlice],
    series: Dict[str, np.ndarray],
    torso_scale: float,
    sampling_rate: float,
) -> AggregateMetrics:
    """Calcula métricas por repetición y estadísticas robustas agregadas."""

    per_rep: List[RepMetrics] = []
    scale = torso_scale if np.isfinite(torso_scale) and torso_scale > 1e-6 else np.nan

    for rep_slice in slices:
        metrics = _compute_single_rep(rep_slice, series, scale, sampling_rate)
        if metrics is not None:
            per_rep.append(metrics)

    if not per_rep:
        return AggregateMetrics(per_rep=tuple(), rep_count=0)

    agg = _aggregate(per_rep, series, scale)
    return agg


def _compute_single_rep(
    rep_slice: RepSlice,
    series: Dict[str, np.ndarray],
    torso_scale: float,
    sampling_rate: float,
) -> RepMetrics | None:
    start, end = rep_slice.start, rep_slice.end
    if end - start < 2:
        return None

    knee = _safe_slice(series.get("knee_angle"), start, end)
    hip = _safe_slice(series.get("hip_angle"), start, end)
    elbow = _safe_slice(series.get("elbow_angle"), start, end)
    torso = _safe_slice(series.get("torso_tilt"), start, end)
    wrist_y = _safe_slice(series.get("wrist_y"), start, end)
    wrist_x = _safe_slice(series.get("wrist_x"), start, end)
    shoulder_y = _safe_slice(series.get("shoulder_y"), start, end)
    hip_y = _safe_slice(series.get("hip_y"), start, end)
    hip_left_y = _safe_slice(series.get("hip_left_y"), start, end)
    hip_right_y = _safe_slice(series.get("hip_right_y"), start, end)
    knee_x = _safe_slice(series.get("knee_x"), start, end)
    knee_y = _safe_slice(series.get("knee_y"), start, end)
    ankle_x = _safe_slice(series.get("ankle_x"), start, end)
    ankle_y = _safe_slice(series.get("ankle_y"), start, end)
    bar_y = _safe_slice(series.get("bar_y"), start, end)
    bar_x = _safe_slice(series.get("bar_x"), start, end)

    if not _has_enough_finite(knee):
        return None

    knee_min_val = np.nanmin(knee)
    bottom_mask = knee <= (knee_min_val + BOTTOM_WINDOW_DEG)
    bottom_indices = np.flatnonzero(bottom_mask & np.isfinite(knee))

    if bottom_indices.size == 0:
        if np.isfinite(knee).sum() == 0:
            return None
        center = int(np.nanargmin(knee))
        half_window = max(BOTTOM_MIN_FRAMES // 2, int(round((end - start) * BOTTOM_FRACTION / 2)))
        lo = max(0, center - half_window)
        hi = min(knee.size, center + half_window + 1)
        bottom_indices = np.arange(lo, hi)

    bottom_mask = np.zeros_like(knee, dtype=bool)
    bottom_mask[bottom_indices] = True

    knee_bottom = safe_nanmedian(knee[bottom_mask])
    hip_bottom = safe_nanmedian(hip[bottom_mask])
    elbow_bottom = safe_nanmedian(elbow[bottom_mask])
    torso_bottom = safe_nanmedian(torso[bottom_mask])

    wrist_y_bottom = safe_nanmedian(wrist_y[bottom_mask])
    shoulder_bottom = safe_nanmedian(shoulder_y[bottom_mask])
    hip_y_bottom = safe_nanmedian(hip_y[bottom_mask])
    hip_left_bottom = safe_nanmedian(hip_left_y[bottom_mask])
    hip_right_bottom = safe_nanmedian(hip_right_y[bottom_mask])
    hip_y_mean_bottom = safe_nanmedian([hip_left_bottom, hip_right_bottom])
    if not np.isfinite(hip_y_mean_bottom):
        hip_y_mean_bottom = hip_y_bottom
    ankle_x_bottom = safe_nanmedian(ankle_x[bottom_mask])
    knee_x_bottom = safe_nanmedian(knee_x[bottom_mask])
    ankle_y_bottom = safe_nanmedian(ankle_y[bottom_mask])
    knee_y_bottom = safe_nanmedian(knee_y[bottom_mask])
    bar_x_bottom = safe_nanmedian(bar_x[bottom_mask])
    bar_y_bottom = safe_nanmedian(bar_y[bottom_mask])

    wrist_shoulder_diff_norm = _normed_difference(wrist_y_bottom, shoulder_bottom, torso_scale)
    wrist_hip_diff_norm = _normed_difference(wrist_y_bottom, hip_y_bottom, torso_scale)
    bar_above_hip_norm = _normed_difference(hip_y_mean_bottom, bar_y_bottom, torso_scale)
    knee_forward_norm = _normed_difference(knee_x_bottom, ankle_x_bottom, torso_scale)

    delta_x = np.abs(knee_x_bottom - ankle_x_bottom)
    delta_y = np.abs(knee_y_bottom - ankle_y_bottom)
    tibia_angle = np.degrees(np.arctan2(delta_x, delta_y)) if np.isfinite(delta_x) and np.isfinite(delta_y) else np.nan
    bar_ankle_diff_norm = _normed_difference(bar_x_bottom, ankle_x_bottom, torso_scale, absolute=True)

    knee_rom = safe_nanmax(knee) - safe_nanmin(knee)
    hip_rom = safe_nanmax(hip) - safe_nanmin(hip)
    elbow_rom = safe_nanmax(elbow) - safe_nanmin(elbow)
    bar_range_norm = _normed_difference(safe_nanmax(bar_y), safe_nanmin(bar_y), torso_scale, absolute=True)

    duration_s = max(0.0, (end - start) / max(sampling_rate, 1e-6))

    return RepMetrics(
        slice=rep_slice,
        knee_min=knee_bottom,
        hip_min=hip_bottom,
        elbow_bottom=elbow_bottom,
        torso_tilt_bottom=torso_bottom,
        wrist_shoulder_diff_norm=wrist_shoulder_diff_norm,
        wrist_hip_diff_norm=wrist_hip_diff_norm,
        bar_above_hip_norm=bar_above_hip_norm,
        knee_forward_norm=knee_forward_norm,
        tibia_angle_deg=tibia_angle,
        bar_ankle_diff_norm=bar_ankle_diff_norm,
        knee_rom=knee_rom,
        hip_rom=hip_rom,
        elbow_rom=elbow_rom,
        bar_range_norm=bar_range_norm,
        duration_s=duration_s,
        bottom_frame_count=int(bottom_indices.size),
    )


def _aggregate(per_rep: Sequence[RepMetrics], series: Dict[str, np.ndarray], torso_scale: float) -> AggregateMetrics:
    def med(name: str) -> float:
        values = [getattr(rep, name) for rep in per_rep]
        return safe_nanmedian(values)

    hip_series = series.get("hip_y", np.array([]))
    hip_left_y = series.get("hip_left_y", np.array([]))
    hip_right_y = series.get("hip_right_y", np.array([]))
    shoulder_left_y = series.get("shoulder_left_y", np.array([]))
    shoulder_right_y = series.get("shoulder_right_y", np.array([]))
    shoulder_series = series.get("shoulder_y", np.array([]))
    bar_y_series = series.get("bar_y", np.array([]))
    bar_x_series = series.get("bar_x", np.array([]))
    arm_y_series = series.get("arm_y")
    if arm_y_series is None or np.asarray(arm_y_series, dtype=float).size == 0:
        arm_y_series = bar_y_series

    hip_mean = nanmean_pair(hip_left_y, hip_right_y)
    if hip_mean.size == 0 or not np.isfinite(hip_mean).any():
        hip_mean = np.asarray(hip_series, dtype=float)

    shoulder_mean = nanmean_pair(shoulder_left_y, shoulder_right_y)
    if shoulder_mean.size == 0 or not np.isfinite(shoulder_mean).any():
        shoulder_mean = np.asarray(shoulder_series, dtype=float)

    hip_range_norm = _series_range_norm(hip_series, torso_scale)
    bar_vertical_range_norm = _series_range_norm(bar_y_series, torso_scale)
    bar_horizontal_std_norm = _series_std_norm(bar_x_series, torso_scale)
    arms_above_hip_fraction = _series_fraction_above(
        hip_mean, arm_y_series, torso_scale, ARM_ABOVE_HIP_THRESH
    )
    bar_near_shoulders_fraction = _series_fraction_below(
        bar_y_series, shoulder_mean, torso_scale, BAR_NEAR_SHOULDER_THRESH
    )

    return AggregateMetrics(
        per_rep=tuple(per_rep),
        knee_min=med("knee_min"),
        hip_min=med("hip_min"),
        elbow_bottom=med("elbow_bottom"),
        torso_tilt_bottom=med("torso_tilt_bottom"),
        wrist_shoulder_diff_norm=med("wrist_shoulder_diff_norm"),
        wrist_hip_diff_norm=med("wrist_hip_diff_norm"),
        bar_above_hip_norm=med("bar_above_hip_norm"),
        knee_forward_norm=med("knee_forward_norm"),
        tibia_angle_deg=med("tibia_angle_deg"),
        bar_ankle_diff_norm=med("bar_ankle_diff_norm"),
        knee_rom=med("knee_rom"),
        hip_rom=med("hip_rom"),
        elbow_rom=med("elbow_rom"),
        bar_range_norm=med("bar_range_norm"),
        hip_range_norm=hip_range_norm,
        bar_vertical_range_norm=bar_vertical_range_norm,
        bar_horizontal_std_norm=bar_horizontal_std_norm,
        arms_high_fraction=arms_above_hip_fraction,
        arms_above_hip_fraction=arms_above_hip_fraction,
        bar_near_shoulders_fraction=bar_near_shoulders_fraction,
        duration_s=safe_nanmedian([rep.duration_s for rep in per_rep]),
        rep_count=len(per_rep),
    )


def _safe_slice(series: np.ndarray | None, start: int, end: int) -> np.ndarray:
    if series is None or series.size == 0:
        return np.full(end - start, np.nan)
    if end > series.size:
        pad = np.full(end - series.size, np.nan)
        trimmed = np.concatenate([series, pad])
    else:
        trimmed = series
    return np.asarray(trimmed[start:end], dtype=float)


def _has_enough_finite(series: np.ndarray) -> bool:
    return np.isfinite(series).sum() >= max(BOTTOM_MIN_FRAMES, 3)


def _normed_difference(a: float, b: float, scale: float, *, absolute: bool = False) -> float:
    if not np.isfinite(scale) or scale <= 1e-6:
        return float("nan")
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    diff = a - b
    if absolute:
        diff = np.abs(diff)
    return float(diff / scale)


def _series_range_norm(series: np.ndarray, torso_scale: float) -> float:
    array = np.asarray(series, dtype=float)
    if array.size == 0 or not np.isfinite(array).any():
        return float("nan")
    if not np.isfinite(torso_scale) or torso_scale <= 1e-6:
        return float("nan")
    rng = safe_nanmax(array) - safe_nanmin(array)
    return float(rng / torso_scale)


def _series_std_norm(series: np.ndarray, torso_scale: float) -> float:
    array = np.asarray(series, dtype=float)
    if array.size == 0 or not np.isfinite(array).any():
        return float("nan")
    if not np.isfinite(torso_scale) or torso_scale <= 1e-6:
        return float("nan")
    return float(safe_nanstd(array) / torso_scale)


def _series_fraction_above(
    upper: np.ndarray,
    lower: np.ndarray,
    torso_scale: float,
    threshold: float,
) -> float:
    if not np.isfinite(torso_scale) or torso_scale <= 1e-6:
        return float("nan")
    upper_arr = np.asarray(upper, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    if upper_arr.size == 0 or lower_arr.size == 0:
        return float("nan")
    length = max(upper_arr.size, lower_arr.size)
    if upper_arr.size != length:
        upper_arr = np.pad(upper_arr, (0, length - upper_arr.size), constant_values=np.nan)
    if lower_arr.size != length:
        lower_arr = np.pad(lower_arr, (0, length - lower_arr.size), constant_values=np.nan)
    valid = np.isfinite(upper_arr) & np.isfinite(lower_arr)
    if not valid.any():
        return float("nan")
    diff = (upper_arr[valid] - lower_arr[valid]) / torso_scale
    return float(np.mean(diff > threshold))


def _series_fraction_below(
    primary: np.ndarray,
    reference: np.ndarray,
    torso_scale: float,
    threshold: float,
) -> float:
    if not np.isfinite(torso_scale) or torso_scale <= 1e-6:
        return float("nan")
    primary_arr = np.asarray(primary, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    if primary_arr.size == 0 or reference_arr.size == 0:
        return float("nan")
    length = max(primary_arr.size, reference_arr.size)
    if primary_arr.size != length:
        primary_arr = np.pad(primary_arr, (0, length - primary_arr.size), constant_values=np.nan)
    if reference_arr.size != length:
        reference_arr = np.pad(reference_arr, (0, length - reference_arr.size), constant_values=np.nan)
    valid = np.isfinite(primary_arr) & np.isfinite(reference_arr)
    if not valid.any():
        return float("nan")
    diff = np.abs(primary_arr[valid] - reference_arr[valid]) / torso_scale
    return float(np.mean(diff < threshold))
