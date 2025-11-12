"""Feature calculations per repetition and aggregated statistics."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .types import AggregateMetrics, RepMetrics, RepSlice


BOTTOM_WINDOW_DEG = 10.0
BOTTOM_MIN_FRAMES = 5
BOTTOM_FRACTION = 0.15


def compute_metrics(
    slices: Sequence[RepSlice],
    series: Dict[str, np.ndarray],
    torso_scale: float,
    sampling_rate: float,
) -> AggregateMetrics:
    """Compute per-repetition metrics and aggregate robust statistics."""

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

    knee_bottom = _nanmedian(knee[bottom_mask])
    hip_bottom = _nanmedian(hip[bottom_mask])
    elbow_bottom = _nanmedian(elbow[bottom_mask])
    torso_bottom = _nanmedian(torso[bottom_mask])

    wrist_y_bottom = _nanmedian(wrist_y[bottom_mask])
    shoulder_bottom = _nanmedian(shoulder_y[bottom_mask])
    hip_y_bottom = _nanmedian(hip_y[bottom_mask])
    ankle_x_bottom = _nanmedian(ankle_x[bottom_mask])
    knee_x_bottom = _nanmedian(knee_x[bottom_mask])
    ankle_y_bottom = _nanmedian(ankle_y[bottom_mask])
    knee_y_bottom = _nanmedian(knee_y[bottom_mask])
    bar_x_bottom = _nanmedian(bar_x[bottom_mask])

    wrist_shoulder_diff_norm = _normed_difference(wrist_y_bottom, shoulder_bottom, torso_scale)
    wrist_hip_diff_norm = _normed_difference(wrist_y_bottom, hip_y_bottom, torso_scale)
    knee_forward_norm = _normed_difference(knee_x_bottom, ankle_x_bottom, torso_scale)

    delta_x = np.abs(knee_x_bottom - ankle_x_bottom)
    delta_y = np.abs(knee_y_bottom - ankle_y_bottom)
    tibia_angle = np.degrees(np.arctan2(delta_x, delta_y)) if np.isfinite(delta_x) and np.isfinite(delta_y) else np.nan
    bar_ankle_diff_norm = _normed_difference(bar_x_bottom, ankle_x_bottom, torso_scale, absolute=True)

    knee_rom = np.nanmax(knee) - np.nanmin(knee)
    hip_rom = np.nanmax(hip) - np.nanmin(hip)
    elbow_rom = np.nanmax(elbow) - np.nanmin(elbow)
    bar_range_norm = _normed_difference(_nanmax(bar_y), _nanmin(bar_y), torso_scale, absolute=True)

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
        bar_range_norm=bar_range_norm,
        duration_s=duration_s,
        bottom_frame_count=int(bottom_indices.size),
    )


def _aggregate(per_rep: Sequence[RepMetrics], series: Dict[str, np.ndarray], torso_scale: float) -> AggregateMetrics:
    def med(name: str) -> float:
        values = [getattr(rep, name) for rep in per_rep]
        return float(np.nanmedian(values))

    hip_series = series.get("hip_y", np.array([]))
    bar_y_series = series.get("bar_y", np.array([]))
    bar_x_series = series.get("bar_x", np.array([]))

    hip_range_norm = _series_range_norm(hip_series, torso_scale)
    bar_vertical_range_norm = _series_range_norm(bar_y_series, torso_scale)
    bar_horizontal_std_norm = _series_std_norm(bar_x_series, torso_scale)

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
        bar_range_norm=med("bar_range_norm"),
        hip_range_norm=hip_range_norm,
        bar_vertical_range_norm=bar_vertical_range_norm,
        bar_horizontal_std_norm=bar_horizontal_std_norm,
        duration_s=float(np.nanmedian([rep.duration_s for rep in per_rep])),
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


def _nanmedian(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmedian(values))


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
    if series.size == 0 or not np.isfinite(series).any():
        return float("nan")
    if not np.isfinite(torso_scale) or torso_scale <= 1e-6:
        return float("nan")
    rng = np.nanmax(series) - np.nanmin(series)
    return float(rng / torso_scale)


def _series_std_norm(series: np.ndarray, torso_scale: float) -> float:
    if series.size == 0 or not np.isfinite(series).any():
        return float("nan")
    if not np.isfinite(torso_scale) or torso_scale <= 1e-6:
        return float("nan")
    return float(np.nanstd(series) / torso_scale)


def _nanmax(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmax(values))


def _nanmin(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmin(values))

