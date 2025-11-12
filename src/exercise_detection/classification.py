"""Orquestación de la clasificación de ejercicio y vista a partir de rasgos."""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Tuple

import numpy as np

from .classifier import classify_exercise
from .constants import DEFAULT_SAMPLING_RATE, FEATURE_NAMES, MIN_VALID_FRAMES
from .metrics import compute_metrics
from .segmentation import segment_reps
from .smoothing import smooth
from .types import AggregateMetrics, ClassificationScores, FeatureSeries
from .view import classify_view

logger = logging.getLogger(__name__)

SMOOTH_KEYS = {
    "knee_angle_left",
    "knee_angle_right",
    "hip_angle_left",
    "hip_angle_right",
    "elbow_angle_left",
    "elbow_angle_right",
    "torso_tilt_deg",
    "wrist_left_x",
    "wrist_left_y",
    "wrist_right_x",
    "wrist_right_y",
    "hip_left_y",
    "hip_right_y",
    "shoulder_width_norm",
    "shoulder_yaw_deg",
    "shoulder_z_delta_abs",
    "ankle_width_norm",
    "torso_length",
    "torso_length_world",
}

_SIDE_VISIBILITY_KEYS = (
    "hip_{side}_x",
    "hip_{side}_y",
    "knee_{side}_x",
    "knee_{side}_y",
    "ankle_{side}_x",
    "ankle_{side}_y",
)

_BOTH_SIDES_VISIBLE_MIN_RATIO = 0.35


def classify_features(features: FeatureSeries) -> Tuple[str, str, float]:
    """Clasifica el ejercicio y la vista de cámara a partir de rasgos preprocesados."""

    if features.valid_frames < MIN_VALID_FRAMES:
        return "unknown", "unknown", 0.0

    series = _prepare_series(features)
    sampling_rate = float(features.sampling_rate or DEFAULT_SAMPLING_RATE)

    smoothed = {key: smooth(values, sampling_rate) for key, values in series.items() if key in SMOOTH_KEYS}

    def get_series(name: str) -> np.ndarray:
        if name in smoothed:
            return smoothed[name]
        return series.get(name, np.full(series["_length"], np.nan))

    torso_scale = _resolve_torso_scale(get_series("torso_length"), get_series("torso_length_world"))

    view_result = classify_view(
        shoulder_yaw=get_series("shoulder_yaw_deg"),
        shoulder_z_delta=get_series("shoulder_z_delta_abs"),
        shoulder_width=get_series("shoulder_width_norm"),
        ankle_width=get_series("ankle_width_norm"),
    )

    visibility = _compute_side_visibility(series)
    side = _select_visible_side(series, visibility=visibility)
    both_sides_visible = _both_sides_visible(visibility)

    knee_angle = get_series(f"knee_angle_{side}")
    hip_angle = get_series(f"hip_angle_{side}")
    elbow_angle = get_series(f"elbow_angle_{side}")
    torso_tilt = get_series("torso_tilt_deg")

    if _is_invalid(knee_angle) or _is_invalid(hip_angle) or _is_invalid(elbow_angle):
        return "unknown", view_result.label, 0.0

    wrist_y_side = get_series(f"wrist_{side}_y")
    wrist_x_side = get_series(f"wrist_{side}_x")
    shoulder_y_side = get_series(f"shoulder_{side}_y")
    hip_y_side = get_series(f"hip_{side}_y")
    knee_x_side = get_series(f"knee_{side}_x")
    knee_y_side = get_series(f"knee_{side}_y")
    ankle_x_side = get_series(f"ankle_{side}_x")
    ankle_y_side = get_series(f"ankle_{side}_y")

    wrist_y_mean = _nanmean_pair(get_series("wrist_left_y"), get_series("wrist_right_y"))
    wrist_x_mean = _nanmean_pair(get_series("wrist_left_x"), get_series("wrist_right_x"))

    rep_slices = segment_reps(knee_angle, wrist_y_mean, torso_scale, sampling_rate)

    metrics_series = {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "elbow_angle": elbow_angle,
        "torso_tilt": torso_tilt,
        "wrist_y": wrist_y_side,
        "wrist_x": wrist_x_side,
        "shoulder_y": shoulder_y_side,
        "hip_y": hip_y_side,
        "knee_x": knee_x_side,
        "knee_y": knee_y_side,
        "ankle_x": ankle_x_side,
        "ankle_y": ankle_y_side,
        "bar_y": wrist_y_mean,
        "bar_x": wrist_x_mean,
    }

    metrics = compute_metrics(rep_slices, metrics_series, torso_scale, sampling_rate)
    if metrics.rep_count == 0:
        return "unknown", view_result.label, 0.0

    label, confidence, scores, _ = classify_exercise(metrics)
    view_label = view_result.label

    if label == "squat" and view_label == "side" and both_sides_visible:
        view_label = "front"

    if label == "unknown":
        view_label = "unknown"
        confidence = 0.0

    _log_summary(side, view_label, metrics, scores)
    return label, view_label, float(confidence)


def _prepare_series(features: FeatureSeries) -> Dict[str, np.ndarray]:
    """Normaliza las series de rasgos a un mismo número de fotogramas."""

    data = {key: np.asarray(value, dtype=float) for key, value in features.data.items() if isinstance(value, (list, tuple, np.ndarray))}
    frame_count = int(features.total_frames or max((len(arr) for arr in data.values()), default=0))
    if frame_count <= 0:
        frame_count = int(max(features.valid_frames, 1))

    for key in FEATURE_NAMES:
        if key not in data:
            data[key] = np.full(frame_count, np.nan)
        else:
            arr = data[key]
            if arr.size < frame_count:
                pad = np.full(frame_count - arr.size, np.nan)
                data[key] = np.concatenate([arr, pad])
            elif arr.size > frame_count:
                data[key] = arr[:frame_count]

    data["_length"] = frame_count
    return data


def _resolve_torso_scale(torso_length: np.ndarray, torso_world: np.ndarray) -> float:
    """Selecciona una escala de torso combinando señales en píxeles y mundo real."""

    candidates = np.concatenate([torso_length[np.isfinite(torso_length)], torso_world[np.isfinite(torso_world)]])
    if candidates.size == 0:
        return float("nan")
    median = float(np.median(candidates))
    return median if np.isfinite(median) and median > 1e-6 else float("nan")


def _select_visible_side(
    series: Dict[str, np.ndarray], *, visibility: Mapping[str, Tuple[int, float]] | None = None
) -> str:
    """Escoge el lado (izquierdo/derecho) con mayor cobertura de frames visibles."""

    if visibility is None:
        visibility = _compute_side_visibility(series)

    left_count = visibility.get("left", (0, 0.0))[0]
    right_count = visibility.get("right", (0, 0.0))[0]
    return "left" if left_count >= right_count else "right"


def _compute_side_visibility(series: Dict[str, np.ndarray]) -> Dict[str, Tuple[int, float]]:
    """Calcula cuántos puntos visibles aporta cada lado y su ratio normalizado."""

    frame_count = int(series.get("_length", 0) or 0)
    per_side: Dict[str, Tuple[int, float]] = {}
    total_possible = frame_count * len(_SIDE_VISIBILITY_KEYS)

    for side in ("left", "right"):
        count = 0
        for key_template in _SIDE_VISIBILITY_KEYS:
            key = key_template.format(side=side)
            values = series.get(key)
            if values is None:
                continue
            arr = np.asarray(values, dtype=float)
            count += int(np.isfinite(arr).sum())

        ratio = (count / total_possible) if total_possible > 0 else 0.0
        per_side[side] = (count, ratio)

    return per_side


def _both_sides_visible(
    visibility: Mapping[str, Tuple[int, float]],
    *,
    min_ratio: float = _BOTH_SIDES_VISIBLE_MIN_RATIO,
) -> bool:
    """Determina si ambos lados superan el ratio mínimo de visibilidad."""

    left_ratio = visibility.get("left", (0, 0.0))[1]
    right_ratio = visibility.get("right", (0, 0.0))[1]
    if left_ratio <= 0.0 or right_ratio <= 0.0:
        return False
    return min(left_ratio, right_ratio) >= min_ratio


def _is_invalid(series: np.ndarray) -> bool:
    """Indica si la serie tiene menos de tres valores finitos."""

    return series.size == 0 or np.isfinite(series).sum() < 3


def _nanmean_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Fusiona dos series tomando la media ignorando valores faltantes."""

    if left.size == 0 and right.size == 0:
        return np.array([], dtype=float)
    if left.size == 0:
        return np.asarray(right, dtype=float)
    if right.size == 0:
        return np.asarray(left, dtype=float)
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    stacked = np.vstack([left_arr, right_arr])
    finite_mask = np.isfinite(stacked)
    counts = finite_mask.sum(axis=0)
    if not counts.any():
        return np.full(stacked.shape[1], np.nan)
    sums = np.nansum(stacked, axis=0)
    result = np.full(stacked.shape[1], np.nan)
    valid = counts > 0
    result[valid] = sums[valid] / counts[valid]
    return result


def _log_summary(side: str, view_label: str, metrics: AggregateMetrics, scores: ClassificationScores) -> None:
    """Escribe en el log los valores clave obtenidos durante la clasificación."""

    logger.info(
        "side=%s view=%s knee_min=%.1f hip_min=%.1f elbow_bottom=%.1f torso_tilt_bottom=%.1f "
        "wrist_shoulder_diff=%.3f wrist_hip_diff=%.3f knee_rom=%.1f hip_rom=%.1f elbow_rom=%.1f "
        "knee_forward=%.3f tibia_angle=%.1f bar_range=%.3f bar_ankle_diff=%.3f hip_range=%.3f bar_horizontal_std=%.3f "
        "duration=%.2f scores_raw=%s scores_adj=%s veto=%s",
        side,
        view_label,
        metrics.knee_min,
        metrics.hip_min,
        metrics.elbow_bottom,
        metrics.torso_tilt_bottom,
        metrics.wrist_shoulder_diff_norm,
        metrics.wrist_hip_diff_norm,
        metrics.knee_rom,
        metrics.hip_rom,
        metrics.elbow_rom,
        metrics.knee_forward_norm,
        metrics.tibia_angle_deg,
        metrics.bar_range_norm,
        metrics.bar_ankle_diff_norm,
        metrics.hip_range_norm,
        metrics.bar_horizontal_std_norm,
        metrics.duration_s,
        {k: round(v, 3) for k, v in scores.raw.items()},
        {k: round(v, 3) for k, v in scores.adjusted.items()},
        scores.deadlift_veto,
    )

    for idx, rep in enumerate(metrics.per_rep):
        logger.debug(
            "rep=%d frames=%d knee_min=%.1f hip_min=%.1f elbow_bottom=%.1f torso=%.1f",  # noqa: G004
            idx,
            rep.slice.end - rep.slice.start,
            rep.knee_min,
            rep.hip_min,
            rep.elbow_bottom,
            rep.torso_tilt_bottom,
        )


__all__ = ["classify_features"]

