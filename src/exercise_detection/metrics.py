"""Cálculo de características por repetición y estadísticas agregadas."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .constants import (
    DEADLIFT_ELBOW_MIN_DEG,
    DEADLIFT_WRIST_HIP_DIFF_MIN_NORM,
    SQUAT_ELBOW_BOTTOM_MAX_DEG,
    SQUAT_KNEE_DEEP_THRESHOLD_DEG,
    SQUAT_WRIST_SHOULDER_DIFF_MAX_NORM,
)
from .stats import safe_nanmax, safe_nanmedian, safe_nanmin, safe_nanstd
from .types import AggregateMetrics, RepMetrics, RepSlice


BOTTOM_WINDOW_DEG = 10.0
BOTTOM_MIN_FRAMES = 5
BOTTOM_FRACTION = 0.15
KNEE_IQR_PERCENTILES = (25, 75)


def compute_metrics(
    slices: Sequence[RepSlice],
    series: Dict[str, np.ndarray],
    torso_scale: float,
    sampling_rate: float,
    *,
    view_label: str | None = None,
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

    agg = _aggregate(per_rep, series, scale, view_label=view_label)
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
    ankle_x_bottom = safe_nanmedian(ankle_x[bottom_mask])
    knee_x_bottom = safe_nanmedian(knee_x[bottom_mask])
    ankle_y_bottom = safe_nanmedian(ankle_y[bottom_mask])
    knee_y_bottom = safe_nanmedian(knee_y[bottom_mask])
    bar_x_bottom = safe_nanmedian(bar_x[bottom_mask])

    wrist_shoulder_diff_norm = _normed_difference(wrist_y_bottom, shoulder_bottom, torso_scale)
    wrist_hip_diff_norm = _normed_difference(wrist_y_bottom, hip_y_bottom, torso_scale)
    knee_forward_norm = _normed_difference(knee_x_bottom, ankle_x_bottom, torso_scale)

    delta_x = np.abs(knee_x_bottom - ankle_x_bottom)
    delta_y = np.abs(knee_y_bottom - ankle_y_bottom)
    tibia_angle = np.degrees(np.arctan2(delta_x, delta_y)) if np.isfinite(delta_x) and np.isfinite(delta_y) else np.nan
    bar_ankle_diff_norm = _normed_difference(bar_x_bottom, ankle_x_bottom, torso_scale, absolute=True)

    knee_rom = safe_nanmax(knee) - safe_nanmin(knee)
    hip_rom = safe_nanmax(hip) - safe_nanmin(hip)
    elbow_rom = safe_nanmax(elbow) - safe_nanmin(elbow)
    knee_finite = knee[np.isfinite(knee)]
    knee_iqr = float("nan")
    knee_deep_fraction = float("nan")
    if knee_finite.size:
        knee_iqr = float(np.subtract(*np.nanpercentile(knee_finite, KNEE_IQR_PERCENTILES[::-1])))
        deep_mask = knee_finite <= SQUAT_KNEE_DEEP_THRESHOLD_DEG
        knee_deep_fraction = float(np.mean(deep_mask)) if deep_mask.size else float("nan")
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
        knee_forward_norm=knee_forward_norm,
        tibia_angle_deg=tibia_angle,
        bar_ankle_diff_norm=bar_ankle_diff_norm,
        knee_rom=knee_rom,
        hip_rom=hip_rom,
        elbow_rom=elbow_rom,
        knee_iqr=knee_iqr,
        knee_deep_fraction=knee_deep_fraction,
        hip_knee_rom_ratio=(hip_rom / knee_rom) if np.isfinite(knee_rom) and knee_rom > 1e-6 else float("nan"),
        bar_range_norm=bar_range_norm,
        duration_s=duration_s,
        bottom_frame_count=int(bottom_indices.size),
    )


def _aggregate(
    per_rep: Sequence[RepMetrics],
    series: Dict[str, np.ndarray],
    torso_scale: float,
    *,
    view_label: str | None,
) -> AggregateMetrics:
    def med(name: str) -> float:
        values = [getattr(rep, name) for rep in per_rep]
        return safe_nanmedian(values)

    hip_series = series.get("hip_y", np.array([]))
    bar_y_series = series.get("bar_y", np.array([]))
    bar_x_series = series.get("bar_x", np.array([]))

    hip_range_norm = _series_range_norm(hip_series, torso_scale)
    pelvis_range_norm = _series_range_norm(series.get("pelvis_y", np.array([])), torso_scale)
    bar_vertical_range_norm = _series_range_norm(bar_y_series, torso_scale)
    bar_horizontal_std_norm = _series_std_norm(bar_x_series, torso_scale)

    wrist_shoulder_diff = _series_normed_diff(series.get("wrist_y"), series.get("shoulder_y"), torso_scale)
    wrist_hip_diff = _series_normed_diff(series.get("wrist_y"), series.get("hip_y"), torso_scale)
    elbow_series = _as_array(series.get("elbow_angle"))

    bar_high_fraction = _finite_fraction(np.abs(wrist_shoulder_diff) <= SQUAT_WRIST_SHOULDER_DIFF_MAX_NORM)
    bar_low_fraction = _finite_fraction(wrist_hip_diff >= DEADLIFT_WRIST_HIP_DIFF_MIN_NORM)
    elbow_extended_fraction = _finite_fraction(elbow_series >= DEADLIFT_ELBOW_MIN_DEG)
    elbow_flexed_fraction = _finite_fraction(elbow_series <= SQUAT_ELBOW_BOTTOM_MAX_DEG)

    return AggregateMetrics(
        per_rep=tuple(per_rep),
        knee_min=med("knee_min"),
        hip_min=med("hip_min"),
        elbow_bottom=med("elbow_bottom"),
        torso_tilt_bottom=med("torso_tilt_bottom"),
        wrist_shoulder_diff_norm=med("wrist_shoulder_diff_norm"),
        wrist_hip_diff_norm=med("wrist_hip_diff_norm"),
        knee_forward_norm=med("knee_forward_norm"),
        tibia_angle_deg=med("tibia_angle_deg"),
        bar_ankle_diff_norm=med("bar_ankle_diff_norm"),
        knee_rom=med("knee_rom"),
        hip_rom=med("hip_rom"),
        elbow_rom=med("elbow_rom"),
        knee_iqr=med("knee_iqr"),
        knee_deep_fraction=med("knee_deep_fraction"),
        hip_knee_rom_ratio=med("hip_knee_rom_ratio"),
        bar_range_norm=med("bar_range_norm"),
        hip_range_norm=hip_range_norm,
        pelvis_range_norm=pelvis_range_norm,
        bar_vertical_range_norm=bar_vertical_range_norm,
        bar_horizontal_std_norm=bar_horizontal_std_norm,
        bar_high_fraction=bar_high_fraction,
        bar_low_fraction=bar_low_fraction,
        elbow_extended_fraction=elbow_extended_fraction,
        elbow_flexed_fraction=elbow_flexed_fraction,
        duration_s=safe_nanmedian([rep.duration_s for rep in per_rep]),
        rep_count=len(per_rep),
        view_label=view_label or "unknown",
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


def _series_normed_diff(
    a_series: np.ndarray | None, b_series: np.ndarray | None, torso_scale: float, *, absolute: bool = False
) -> np.ndarray:
    a_array = _as_array(a_series)
    b_array = _as_array(b_series)
    if a_array.size == 0 or b_array.size == 0:
        return np.full(max(a_array.size, b_array.size, 1), np.nan)
    length = max(a_array.size, b_array.size)
    if a_array.size < length:
        a_array = np.pad(a_array, (0, length - a_array.size), constant_values=np.nan)
    if b_array.size < length:
        b_array = np.pad(b_array, (0, length - b_array.size), constant_values=np.nan)
    if not np.isfinite(torso_scale) or torso_scale <= 1e-6:
        return np.full(length, np.nan)
    diff = a_array - b_array
    if absolute:
        diff = np.abs(diff)
    return np.asarray(diff / torso_scale, dtype=float)


def _as_array(series: np.ndarray | None) -> np.ndarray:
    if series is None:
        return np.array([], dtype=float)
    return np.asarray(series, dtype=float)


def _finite_fraction(mask: np.ndarray | float) -> float:
    array = np.asarray(mask, dtype=float)
    finite_mask = np.isfinite(array)
    if finite_mask.sum() == 0:
        return float("nan")
    return float(np.mean(array[finite_mask] > 0.5))

