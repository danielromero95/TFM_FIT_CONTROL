"""Punto de entrada público que conecta el preprocesado con los clasificadores."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .classifier import classify_exercise
from .constants import DEFAULT_SAMPLING_RATE, FEATURE_NAMES, MIN_VALID_FRAMES
from .metrics import compute_metrics
from .segmentation import segment_reps
from .smoothing import smooth
from .types import AggregateMetrics, ClassificationScores, FeatureSeries, ViewResult
from .view import classify_view
from src.utils.angles import apply_warmup_mask, maybe_convert_radians_to_degrees, suppress_spikes
from src.config.settings import DEFAULT_WARMUP_SECONDS

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

ANGLE_LIKE_KEYS = {
    "knee_angle_left",
    "knee_angle_right",
    "hip_angle_left",
    "hip_angle_right",
    "elbow_angle_left",
    "elbow_angle_right",
    "torso_tilt_deg",
    "shoulder_yaw_deg",
}

RADIAN_PRONE_KEYS = {"torso_tilt_deg", "shoulder_yaw_deg"}

DETECTION_WARMUP_SECONDS = DEFAULT_WARMUP_SECONDS
DETECTION_WARMUP_MAX_FRAMES = 3
DETECTION_SPIKE_THRESHOLD_DEG = 55.0


def _compute_warmup_frames(
    sampling_rate: float,
    *,
    warmup_seconds: float | None,
    override: int | None,
    max_frames: int = DETECTION_WARMUP_MAX_FRAMES,
) -> int:
    if override is not None:
        return max(0, int(override))
    if warmup_seconds is None or sampling_rate <= 0:
        return 0
    computed = int(math.ceil(sampling_rate * warmup_seconds))
    if max_frames > 0:
        return max(0, min(computed, int(max_frames)))
    return max(0, computed)


def _build_frame_mask(length: int, reliability: np.ndarray | None, warmup_frames: int) -> np.ndarray:
    frame_count = int(length or 0)
    if frame_count <= 0:
        return np.array([], dtype=bool)

    mask = np.ones(frame_count, dtype=bool)
    if warmup_frames > 0:
        mask[: min(frame_count, int(warmup_frames))] = False

    if reliability is not None:
        rel_arr = np.asarray(reliability, dtype=bool)
        if rel_arr.size:
            limit = min(frame_count, rel_arr.size)
            mask[:limit] &= rel_arr[:limit]

    return mask

_SIDE_VISIBILITY_KEYS = (
    "hip_{side}_x",
    "hip_{side}_y",
    "knee_{side}_x",
    "knee_{side}_y",
    "ankle_{side}_x",
    "ankle_{side}_y",
)

_BOTH_SIDES_VISIBLE_MIN_RATIO = 0.35


def classify_features(
    features: FeatureSeries,
    *,
    return_metadata: bool = False,
    warmup_seconds: float | None = None,
    warmup_frames_override: int | None = None,
) -> Tuple[str, str, float] | Tuple[str, str, float, Dict[str, Any]]:
    """Clasifica el ejercicio y la vista de cámara a partir de rasgos de pose.

    Args:
        features: serie de características agregadas.
        return_metadata: cuando es ``True`` devuelve un cuarto elemento con
            detalles útiles para depuración (vista y fiabilidad).
    """

    if features.valid_frames < MIN_VALID_FRAMES:
        return "unknown", "unknown", 0.0

    series = _prepare_series(features)
    cleaning_debug: Dict[str, Dict[str, float | bool]] = {}

    sampling_rate = float(features.sampling_rate or DEFAULT_SAMPLING_RATE)
    warmup_frames = _compute_warmup_frames(
        sampling_rate,
        warmup_seconds=DETECTION_WARMUP_SECONDS if warmup_seconds is None else warmup_seconds,
        override=warmup_frames_override,
    )

    view_reliability = getattr(features, "view_reliability", None)
    reliability_mask = view_reliability.get("frame_reliability") if isinstance(view_reliability, dict) else None

    frame_mask = _build_frame_mask(series.get("_length", 0), reliability_mask, warmup_frames)

    for key in ANGLE_LIKE_KEYS:
        if key not in series:
            continue

        converted = False
        cleaned = series[key]
        if key in RADIAN_PRONE_KEYS:
            cleaned, converted = maybe_convert_radians_to_degrees(series[key])

        cleaned, warmup_applied = apply_warmup_mask(cleaned, warmup_frames)
        cleaned, spikes_removed = suppress_spikes(cleaned, DETECTION_SPIKE_THRESHOLD_DEG)
        series[key] = cleaned

        if converted or spikes_removed > 0 or warmup_applied > 0:
            cleaning_debug[key] = {
                "converted_from_rad": bool(converted),
                "warmup_masked": int(warmup_applied),
                "spikes_removed": int(spikes_removed),
            }

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
        reliability=getattr(features, "view_reliability", None),
    )

    view_cues_present = any(
        np.isfinite(get_series(name)).any()
        for name in ("shoulder_yaw_deg", "shoulder_z_delta_abs", "shoulder_width_norm", "ankle_width_norm")
    )
    if view_result.label == "unknown" and not view_cues_present:
        return "unknown", "unknown", 0.0

    visibility = _compute_side_visibility(series)
    side = _select_visible_side(series, visibility=visibility)
    both_sides_visible = _both_sides_visible(visibility)

    knee_angle = get_series(f"knee_angle_{side}")
    hip_angle = get_series(f"hip_angle_{side}")
    elbow_angle_raw = get_series(f"elbow_angle_{side}")
    elbow_angle = _apply_frame_mask(elbow_angle_raw, frame_mask)
    torso_tilt = get_series("torso_tilt_deg")

    if _is_invalid(knee_angle) or _is_invalid(hip_angle) or _is_invalid(elbow_angle_raw):
        return "unknown", view_result.label, 0.0

    wrist_y_mean = _apply_frame_mask(_nanmean_pair(get_series("wrist_left_y"), get_series("wrist_right_y")), frame_mask)
    wrist_x_mean = _apply_frame_mask(_nanmean_pair(get_series("wrist_left_x"), get_series("wrist_right_x")), frame_mask)
    shoulder_y_mean = _apply_frame_mask(
        _nanmean_pair(get_series("shoulder_left_y"), get_series("shoulder_right_y")), frame_mask
    )
    hip_y_mean = _apply_frame_mask(_nanmean_pair(get_series("hip_left_y"), get_series("hip_right_y")), frame_mask)
    knee_x_side = get_series(f"knee_{side}_x")
    knee_y_side = get_series(f"knee_{side}_y")
    ankle_x_side = get_series(f"ankle_{side}_x")
    ankle_y_side = get_series(f"ankle_{side}_y")

    rep_slices = segment_reps(knee_angle, wrist_y_mean, torso_scale, sampling_rate)

    metrics_series = {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "elbow_angle": elbow_angle,
        "torso_tilt": torso_tilt,
        "wrist_y": wrist_y_mean,
        "wrist_x": wrist_x_mean,
        "shoulder_y": shoulder_y_mean,
        "hip_y": hip_y_mean,
        "knee_x": knee_x_side,
        "knee_y": knee_y_side,
        "ankle_x": ankle_x_side,
        "ankle_y": ankle_y_side,
        "bar_y": wrist_y_mean,
        "bar_x": wrist_x_mean,
        "pelvis_y": _apply_frame_mask(get_series("pelvis_y"), frame_mask),
    }

    metrics = compute_metrics(
        rep_slices,
        metrics_series,
        torso_scale,
        sampling_rate,
        view_label=view_result.label,
    )
    if metrics.rep_count == 0:
        return "unknown", view_result.label, 0.0

    label, confidence, scores, _ = classify_exercise(metrics)
    view_label = view_result.label

    if (
        label == "squat"
        and view_label == "side"
        and both_sides_visible
        and view_result.votes.get("front", 0) > view_result.votes.get("side", 0)
    ):
        view_label = "front"

    if label == "unknown":
        view_label = "unknown"
        confidence = 0.0

    if np.isfinite(view_result.confidence):
        confidence = float(min(confidence, view_result.confidence))

    _log_summary(side, view_label, metrics, scores, view_result)

    if not return_metadata:
        return label, view_label, float(confidence)

    view_stats_raw = getattr(view_result, "stats", {}) or {}

    def _jsonable(value: Any) -> Any:
        try:
            if np.isfinite(value):
                return float(value)
            if isinstance(value, (int, float, np.generic)):
                return None
        except Exception:
            return value
        if isinstance(value, (int, float, np.generic)):
            return None
        return value

    view_stats = {key: _jsonable(value) for key, value in view_stats_raw.items()}
    def _feature_summary(values: np.ndarray) -> Dict[str, Any]:
        total = int(values.size)
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        valid_count = int(finite_mask.sum())
        valid_fraction = (valid_count / total) if total > 0 else None
        if valid_count == 0:
            return {"valid_fraction": valid_fraction, "count": total}

        stats = {
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
            "mean": float(np.mean(finite_values)),
            "std": float(np.std(finite_values)),
        }
        stats["range"] = stats["max"] - stats["min"]
        try:
            percentiles = np.percentile(finite_values, [10, 50, 90])
            stats.update(
                {
                    "p10": float(percentiles[0]),
                    "p50": float(percentiles[1]),
                    "p90": float(percentiles[2]),
                }
            )
        except Exception:
            pass

        stats["valid_fraction"] = valid_fraction
        stats["count"] = total
        return stats

    SUMMARY_FEATURES = (
        "pelvis_y",
        "torso_tilt_deg",
        "knee_angle_left",
        "knee_angle_right",
        "hip_angle_left",
        "hip_angle_right",
        "elbow_angle_left",
        "elbow_angle_right",
        "shoulder_width_norm",
        "shoulder_yaw_deg",
        "shoulder_z_delta_abs",
    )

    features_summary: Dict[str, Any] = {}
    for name in SUMMARY_FEATURES:
        values = get_series(name)
        if values.size:
            features_summary[name] = _feature_summary(values)

    if is_dataclass(scores):
        classification_scores: Dict[str, Any] = asdict(scores)
    elif hasattr(scores, "__dict__"):
        classification_scores = dict(getattr(scores, "__dict__", {}))
    else:
        classification_scores = {"value": scores}

    vetoes = {
        key: bool(value)
        for key, value in classification_scores.items()
        if isinstance(value, bool) or str(key).endswith("_veto")
    }

    cleaning_summary: Dict[str, Any] = {}
    if cleaning_debug:
        unit_fixes = [name for name, meta in cleaning_debug.items() if meta.get("converted_from_rad")]
        spikes = {name: meta["spikes_removed"] for name, meta in cleaning_debug.items() if meta.get("spikes_removed")}
        warmup_applied = any(meta.get("warmup_masked") for meta in cleaning_debug.values())

        if unit_fixes:
            cleaning_summary["unit_fixes_applied"] = unit_fixes
        if spikes:
            cleaning_summary["outliers_masked"] = spikes
        if warmup_applied:
            cleaning_summary["warmup_frames_dropped"] = warmup_frames

    metadata: Dict[str, Any] = {
        "view_label": view_label,
        "view_confidence": float(view_result.confidence),
        "view_side": getattr(view_result, "side", None),
        "view_stats": view_stats,
        "both_sides_visible": bool(both_sides_visible),
        "view_cues_present": bool(view_cues_present),
        "features_summary_source": "mixed",
        "features_summary": features_summary,
        "classification_scores": classification_scores,
        "vetoes": vetoes,
        "rep_count": int(metrics.rep_count),
    }
    if cleaning_summary:
        metadata["angle_cleaning"] = cleaning_summary
    return label, view_label, float(confidence), metadata


def _prepare_series(features: FeatureSeries) -> Dict[str, np.ndarray]:
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
    candidates = np.concatenate([torso_length[np.isfinite(torso_length)], torso_world[np.isfinite(torso_world)]])
    if candidates.size == 0:
        return float("nan")
    median = float(np.median(candidates))
    return median if np.isfinite(median) and median > 1e-6 else float("nan")


def _select_visible_side(
    series: Dict[str, np.ndarray], *, visibility: Mapping[str, Tuple[int, float]] | None = None
) -> str:
    if visibility is None:
        visibility = _compute_side_visibility(series)

    left_count = visibility.get("left", (0, 0.0))[0]
    right_count = visibility.get("right", (0, 0.0))[0]
    return "left" if left_count >= right_count else "right"


def _compute_side_visibility(series: Dict[str, np.ndarray]) -> Dict[str, Tuple[int, float]]:
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
    left_ratio = visibility.get("left", (0, 0.0))[1]
    right_ratio = visibility.get("right", (0, 0.0))[1]
    if left_ratio <= 0.0 or right_ratio <= 0.0:
        return False
    return min(left_ratio, right_ratio) >= min_ratio


def _is_invalid(series: np.ndarray) -> bool:
    return series.size == 0 or np.isfinite(series).sum() < 3


def _apply_frame_mask(series: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return arr

    if mask.size == 0:
        return arr

    applied = mask
    if mask.size != arr.size:
        applied = np.ones(arr.size, dtype=bool)
        applied[: min(arr.size, mask.size)] = mask[: min(arr.size, mask.size)]

    masked = arr.copy()
    masked[~applied] = np.nan
    return masked


def _nanmean_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def _log_summary(
    side: str, view_label: str, metrics: AggregateMetrics, scores: ClassificationScores, view_result: ViewResult
) -> None:
    stats = getattr(view_result, "stats", {}) or {}
    rel_used = int(stats.get("reliable_frames", 0))
    rel_total = int(stats.get("total_frames_sampled", 0))
    width_med = stats.get("width_med", float("nan"))
    z_med = stats.get("z_med", float("nan"))
    lateral_score = stats.get("lateral_score", float("nan"))
    side_hint = getattr(view_result, "side", None)
    logger.info(
        "side=%s view=%s knee_min=%.1f hip_min=%.1f elbow_bottom=%.1f torso_tilt_bottom=%.1f "
        "wrist_shoulder_diff=%.3f wrist_hip_diff=%.3f knee_rom=%.1f hip_rom=%.1f elbow_rom=%.1f "
        "knee_forward=%.3f tibia_angle=%.1f bar_range=%.3f bar_ankle_diff=%.3f hip_range=%.3f bar_horizontal_std=%.3f "
        "duration=%.2f bar_high_frac=%.2f bar_low_frac=%.2f elbow_ext_frac=%.2f elbow_flex_frac=%.2f "
        "scores_raw=%s scores_adj=%s veto=%s view_rel=%d/%d width_med=%.3f z_med=%.3f lateral=%.3f view_side=%s",
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
        metrics.bar_high_fraction,
        metrics.bar_low_fraction,
        metrics.elbow_extended_fraction,
        metrics.elbow_flexed_fraction,
        {k: round(v, 3) for k, v in scores.raw.items()},
        {k: round(v, 3) for k, v in scores.adjusted.items()},
        scores.deadlift_veto,
        rel_used,
        rel_total,
        width_med,
        z_med,
        lateral_score,
        side_hint,
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

