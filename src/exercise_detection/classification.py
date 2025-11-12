"""Heurísticas de clasificación para identificar ejercicio y vista de cámara.

El módulo replica la lógica artesanal del detector original pero la organiza
en funciones autocontenidas.  Así aislamos las transformaciones puras y
facilitamos su mantenimiento, dejando claro que los umbrales viven en
``constants.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.signal import savgol_filter

from .constants import (
    ANKLE_FRONT_WIDTH_THRESHOLD,
    ANKLE_SIDE_WIDTH_MAX,
    ANKLE_WIDTH_STD_THRESHOLD,
    BAR_DROP_MIN_NORM,
    BENCH_BAR_HORIZONTAL_STD_MAX,
    BENCH_BAR_RANGE_MIN_NORM,
    BENCH_ELBOW_ROM_MIN_DEG,
    BENCH_HIP_RANGE_MAX_NORM,
    BENCH_HIP_ROM_MAX_DEG,
    BENCH_KNEE_ROM_MAX_DEG,
    BENCH_TORSO_HORIZONTAL_DEG,
    CLASSIFICATION_MARGIN,
    DEADLIFT_BAR_ANKLE_MAX_NORM,
    DEADLIFT_BAR_RANGE_MIN_NORM,
    DEADLIFT_ELBOW_MIN_DEG,
    DEADLIFT_HIP_ROM_MIN_DEG,
    DEADLIFT_KNEE_BOTTOM_MIN_DEG,
    DEADLIFT_KNEE_FORWARD_MAX_NORM,
    DEADLIFT_TORSO_TILT_MIN_DEG,
    DEADLIFT_WRIST_HIP_DIFF_MIN_NORM,
    DEFAULT_SAMPLING_RATE,
    EVENT_MIN_GAP_SECONDS,
    MIN_CONFIDENCE_SCORE,
    MIN_VALID_FRAMES,
    SIDE_WIDTH_MAX,
    SQUAT_ELBOW_BOTTOM_MAX_DEG,
    SQUAT_ELBOW_BOTTOM_MIN_DEG,
    SQUAT_HIP_BOTTOM_MAX_DEG,
    SQUAT_KNEE_BOTTOM_MAX_DEG,
    SQUAT_KNEE_FORWARD_MIN_NORM,
    SQUAT_MIN_ROM_DEG,
    SQUAT_TIBIA_MAX_DEG,
    SQUAT_TORSO_TILT_MAX_DEG,
    SQUAT_WRIST_SHOULDER_DIFF_MAX_NORM,
    VIEW_FRONT_FALLBACK_YAW_DEG,
    VIEW_FRONT_WIDTH_THRESHOLD,
    VIEW_MARGIN_PER_EVIDENCE_THRESHOLD,
    VIEW_SCORE_PER_EVIDENCE_THRESHOLD,
    VIEW_SIDE_FALLBACK_YAW_DEG,
    VIEW_WIDTH_STD_THRESHOLD,
    YAW_FRONT_MAX_DEG,
    YAW_SIDE_MIN_DEG,
    Z_DELTA_FRONT_MAX,
    KNEE_DOWN_THRESHOLD_DEG,
    KNEE_UP_THRESHOLD_DEG,
)
from .types import FeatureSeries

logger = logging.getLogger(__name__)


def classify_features(features: FeatureSeries) -> Tuple[str, str, float]:
    """Clasificar ejercicio y vista con las heurísticas de la vista lateral."""

    if features.valid_frames < MIN_VALID_FRAMES:
        return "unknown", "unknown", 0.0

    data = features.data
    sr = max(1.0, float(features.sampling_rate or DEFAULT_SAMPLING_RATE))

    smooth_keys = {
        "knee_angle_left",
        "knee_angle_right",
        "hip_angle_left",
        "hip_angle_right",
        "elbow_angle_left",
        "elbow_angle_right",
        "torso_tilt_deg",
        "wrist_left_y",
        "wrist_right_y",
        "wrist_left_x",
        "wrist_right_x",
        "hip_left_y",
        "hip_right_y",
        "shoulder_width_norm",
        "shoulder_yaw_deg",
        "shoulder_z_delta_abs",
        "ankle_width_norm",
        "torso_length",
    }

    smoothed: Dict[str, np.ndarray] = {}
    for key in smooth_keys:
        raw_series = data.get(key)
        if raw_series is None:
            continue
        arr = np.asarray(raw_series, dtype=float)
        if arr.size == 0:
            continue
        smoothed_arr = _smooth(arr, sr=sr)
        if smoothed_arr.size:
            smoothed[key] = smoothed_arr

    def _series(name: str) -> np.ndarray:
        """Recuperar una serie privilegiando la versión suavizada."""

        candidate = smoothed.get(name)
        if candidate is not None and candidate.size:
            return candidate
        raw = data.get(name)
        if raw is None:
            return np.array([])
        return np.asarray(raw, dtype=float)

    torso_length_series = _series("torso_length")
    torso_length_world_series = _series("torso_length_world")
    torso_scale = _resolve_torso_scale(torso_length_series, torso_length_world_series)

    shoulder_width_series = _series("shoulder_width_norm")
    shoulder_yaw_series = _series("shoulder_yaw_deg")
    shoulder_z_series = _series("shoulder_z_delta_abs")
    ankle_width_series = _series("ankle_width_norm")

    view = _classify_view(
        shoulder_width_series=shoulder_width_series,
        shoulder_yaw_series=shoulder_yaw_series,
        shoulder_z_delta_series=shoulder_z_series,
        ankle_width_series=ankle_width_series,
    )

    side = _select_visible_side(data)

    knee_series = _series(f"knee_angle_{side}")
    hip_series = _series(f"hip_angle_{side}")
    elbow_series = _series(f"elbow_angle_{side}")

    if knee_series.size == 0 or hip_series.size == 0 or elbow_series.size == 0:
        return "unknown", view, 0.0

    torso_tilt_series = _series("torso_tilt_deg")
    wrist_left_y = _series("wrist_left_y")
    wrist_right_y = _series("wrist_right_y")
    wrist_left_x = _series("wrist_left_x")
    wrist_right_x = _series("wrist_right_x")
    shoulder_side_y = _series(f"shoulder_{side}_y")
    hip_side_y = _series(f"hip_{side}_y")
    hip_left_y = _series("hip_left_y")
    hip_right_y = _series("hip_right_y")
    knee_side_x = _series(f"knee_{side}_x")
    knee_side_y = _series(f"knee_{side}_y")
    ankle_side_x = _series(f"ankle_{side}_x")
    ankle_side_y = _series(f"ankle_{side}_y")

    bar_y_series = _mean_series([wrist_left_y, wrist_right_y])
    bar_x_series = _mean_series([wrist_left_x, wrist_right_x])
    hip_y_avg_series = _mean_series([hip_left_y, hip_right_y])

    seg_scale = torso_scale if np.isfinite(torso_scale) and torso_scale > 1e-6 else 1.0

    rep_slices = _segment_reps_state(
        knee_series=knee_series,
        bar_series=bar_y_series,
        sr=sr,
        torso_scale=seg_scale,
    )
    if not rep_slices:
        total_len = int(
            max(
                knee_series.size,
                hip_series.size,
                elbow_series.size,
                torso_tilt_series.size,
                bar_y_series.size if bar_y_series.size else 0,
            )
        )
        if total_len <= 0:
            return "unknown", view, 0.0
        rep_slices = [slice(0, total_len)]

    rep_stats = [
        _compute_rep_metrics(
            rep,
            sr=sr,
            torso_scale=torso_scale,
            knee=knee_series,
            hip=hip_series,
            elbow=elbow_series,
            torso_tilt=torso_tilt_series,
            wrist_y=bar_y_series,
            shoulder_y=shoulder_side_y,
            hip_y=hip_side_y,
            knee_x=knee_side_x,
            knee_y=knee_side_y,
            ankle_x=ankle_side_x,
            ankle_y=ankle_side_y,
            bar_x=bar_x_series,
        )
        for rep in rep_slices
    ]

    def _aggregate(values: Iterable[float], reducer=np.median, default=float("nan")) -> float:
        arr = _finite(np.asarray(list(values), dtype=float))
        if arr.size == 0:
            return float(default)
        return float(reducer(arr))

    knee_rom_med = _aggregate((rep.knee_rom for rep in rep_stats))
    hip_rom_med = _aggregate((rep.hip_rom for rep in rep_stats))
    elbow_rom_med = _aggregate((rep.elbow_rom for rep in rep_stats))
    knee_bottom_med = _aggregate((rep.knee_min for rep in rep_stats))
    hip_bottom_med = _aggregate((rep.hip_min for rep in rep_stats))
    elbow_bottom_med = _aggregate((rep.elbow_bottom for rep in rep_stats))
    torso_bottom_med = _aggregate((rep.torso_tilt_bottom for rep in rep_stats))
    wrist_shoulder_norm_med = _aggregate((rep.wrist_shoulder_diff_norm for rep in rep_stats))
    wrist_hip_norm_med = _aggregate((rep.wrist_hip_diff_norm for rep in rep_stats))
    knee_forward_norm_med = _aggregate((rep.knee_forward_norm for rep in rep_stats))
    tibia_angle_med = _aggregate((rep.tibia_angle_deg for rep in rep_stats))
    bar_ankle_norm_med = _aggregate((rep.bar_ankle_diff_norm for rep in rep_stats))
    bar_range_norm_med = _aggregate((rep.bar_range_norm for rep in rep_stats))
    duration_med = _aggregate((rep.duration_s for rep in rep_stats))

    hip_range_norm = _normalized_range(hip_y_avg_series, torso_scale)
    bar_vertical_range_norm = _normalized_range(bar_y_series, torso_scale)
    bar_horizontal_std_norm = _normalized_std(bar_x_series, torso_scale)

    abs_knee_forward = (
        abs(knee_forward_norm_med) if np.isfinite(knee_forward_norm_med) else float("nan")
    )

    bench_score = 0.0
    bench_posture = np.isfinite(torso_bottom_med) and torso_bottom_med >= BENCH_TORSO_HORIZONTAL_DEG
    if bench_posture:
        bench_score += 1.4
        if np.isfinite(elbow_rom_med) and elbow_rom_med >= BENCH_ELBOW_ROM_MIN_DEG:
            bench_score += 1.0
        if np.isfinite(knee_rom_med) and knee_rom_med <= BENCH_KNEE_ROM_MAX_DEG:
            bench_score += 0.7
        if np.isfinite(hip_rom_med) and hip_rom_med <= BENCH_HIP_ROM_MAX_DEG:
            bench_score += 0.7
        if np.isfinite(bar_range_norm_med) and bar_range_norm_med >= BENCH_BAR_RANGE_MIN_NORM:
            bench_score += 1.0
        if np.isfinite(hip_range_norm) and hip_range_norm <= BENCH_HIP_RANGE_MAX_NORM:
            bench_score += 0.6
        if np.isfinite(bar_horizontal_std_norm) and bar_horizontal_std_norm <= BENCH_BAR_HORIZONTAL_STD_MAX:
            bench_score += 0.6
        if np.isfinite(bar_vertical_range_norm) and bar_vertical_range_norm >= BENCH_BAR_RANGE_MIN_NORM:
            bench_score += 0.4

    squat_score = 0.0
    squat_posture = (
        (np.isfinite(knee_bottom_med) and knee_bottom_med <= SQUAT_KNEE_BOTTOM_MAX_DEG)
        or (np.isfinite(hip_bottom_med) and hip_bottom_med <= SQUAT_HIP_BOTTOM_MAX_DEG)
    )
    if squat_posture:
        if np.isfinite(knee_bottom_med) and knee_bottom_med <= SQUAT_KNEE_BOTTOM_MAX_DEG:
            squat_score += 1.0
        if np.isfinite(hip_bottom_med) and hip_bottom_med <= SQUAT_HIP_BOTTOM_MAX_DEG:
            squat_score += 0.6
        if np.isfinite(torso_bottom_med) and torso_bottom_med <= SQUAT_TORSO_TILT_MAX_DEG:
            squat_score += 0.6
        if np.isfinite(wrist_shoulder_norm_med) and abs(wrist_shoulder_norm_med) <= SQUAT_WRIST_SHOULDER_DIFF_MAX_NORM:
            squat_score += 0.6
        if (
            np.isfinite(elbow_bottom_med)
            and SQUAT_ELBOW_BOTTOM_MIN_DEG <= elbow_bottom_med <= SQUAT_ELBOW_BOTTOM_MAX_DEG
        ):
            squat_score += 0.5
        if np.isfinite(abs_knee_forward) and abs_knee_forward >= SQUAT_KNEE_FORWARD_MIN_NORM:
            squat_score += 0.6
        if np.isfinite(knee_rom_med) and knee_rom_med >= SQUAT_MIN_ROM_DEG:
            squat_score += 0.6
        if np.isfinite(tibia_angle_med) and tibia_angle_med <= SQUAT_TIBIA_MAX_DEG:
            squat_score += 0.4

    deadlift_score = 0.0
    deadlift_posture = (
        (np.isfinite(torso_bottom_med) and torso_bottom_med >= DEADLIFT_TORSO_TILT_MIN_DEG)
        or (np.isfinite(wrist_hip_norm_med) and wrist_hip_norm_med >= DEADLIFT_WRIST_HIP_DIFF_MIN_NORM)
    )
    if deadlift_posture:
        if np.isfinite(knee_bottom_med) and knee_bottom_med >= DEADLIFT_KNEE_BOTTOM_MIN_DEG:
            deadlift_score += 1.0
        if np.isfinite(wrist_hip_norm_med) and wrist_hip_norm_med >= DEADLIFT_WRIST_HIP_DIFF_MIN_NORM:
            deadlift_score += 0.8
        if np.isfinite(elbow_bottom_med) and elbow_bottom_med >= DEADLIFT_ELBOW_MIN_DEG:
            deadlift_score += 0.6
        if np.isfinite(abs_knee_forward) and abs_knee_forward <= DEADLIFT_KNEE_FORWARD_MAX_NORM:
            deadlift_score += 0.6
        if np.isfinite(hip_rom_med) and hip_rom_med >= DEADLIFT_HIP_ROM_MIN_DEG:
            deadlift_score += 0.6
        if np.isfinite(bar_ankle_norm_med) and bar_ankle_norm_med <= DEADLIFT_BAR_ANKLE_MAX_NORM:
            deadlift_score += 0.6
        if np.isfinite(bar_range_norm_med) and bar_range_norm_med >= DEADLIFT_BAR_RANGE_MIN_NORM:
            deadlift_score += 0.5
        if np.isfinite(bar_horizontal_std_norm) and bar_horizontal_std_norm <= BENCH_BAR_HORIZONTAL_STD_MAX:
            deadlift_score += 0.4

    scores = {
        "squat": float(squat_score),
        "bench_press": float(bench_score),
        "deadlift": float(deadlift_score),
    }

    label, confidence = _pick_label(scores)

    logger.info(
        "DET señales — side=%s view=%s kneeMin=%.1f hipMin=%.1f elbowBottom=%.1f torsoBottom=%.1f "
        "wristShoulder=%.3f wristHip=%.3f kneeROM=%.1f hipROM=%.1f elbowROM=%.1f kneeForward=%.3f "
        "tibia=%.1f barRange=%.3f barAnkle=%.3f hipRange=%.3f barStd=%.3f dur=%.2f scores=%s",
        side,
        view,
        knee_bottom_med,
        hip_bottom_med,
        elbow_bottom_med,
        torso_bottom_med,
        wrist_shoulder_norm_med,
        wrist_hip_norm_med,
        knee_rom_med,
        hip_rom_med,
        elbow_rom_med,
        abs_knee_forward,
        tibia_angle_med,
        bar_range_norm_med,
        bar_ankle_norm_med,
        hip_range_norm,
        bar_horizontal_std_norm,
        duration_med,
        scores,
    )

    if label == "unknown":
        view = "unknown"

    return label, view, confidence
@dataclass
class RepMetrics:
    knee_min: float
    hip_min: float
    elbow_bottom: float
    torso_tilt_bottom: float
    wrist_shoulder_diff_norm: float
    wrist_hip_diff_norm: float
    knee_forward_norm: float
    tibia_angle_deg: float
    bar_ankle_diff_norm: float
    knee_rom: float
    hip_rom: float
    elbow_rom: float
    bar_range_norm: float
    duration_s: float


def _mean_series(series_list: Iterable[np.ndarray | None]) -> np.ndarray:
    """Promediar series homólogas ignorando huecos con ``NaN``."""

    arrays: list[np.ndarray] = []
    max_len = 0
    for series in series_list:
        if series is None:
            continue
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            continue
        arrays.append(arr)
        if arr.size > max_len:
            max_len = arr.size
    if not arrays or max_len == 0:
        return np.array([])
    stacked = np.full((len(arrays), max_len), np.nan, dtype=float)
    for idx, arr in enumerate(arrays):
        length = min(arr.size, max_len)
        stacked[idx, :length] = arr[:length]
    return np.nanmean(stacked, axis=0)


def _normalized_range(series: np.ndarray | None, scale: float) -> float:
    if series is None:
        return float("nan")
    arr = np.asarray(series, dtype=float)
    if arr.size == 0 or not np.isfinite(scale) or scale <= 1e-6:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float((finite.max() - finite.min()) / scale)


def _normalized_std(series: np.ndarray | None, scale: float) -> float:
    if series is None:
        return float("nan")
    arr = np.asarray(series, dtype=float)
    if arr.size == 0 or not np.isfinite(scale) or scale <= 1e-6:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite) / scale)


def _resolve_torso_scale(torso_length: np.ndarray | None, torso_world: np.ndarray | None) -> float:
    """Seleccionar una longitud de torso robusta para normalizar distancias."""

    candidate = _finite_stat(torso_length, np.median)
    if np.isfinite(candidate) and candidate > 1e-6:
        return float(candidate)
    candidate = _finite_stat(torso_world, np.median)
    if np.isfinite(candidate) and candidate > 1e-6:
        return float(candidate)
    return float("nan")


def _select_visible_side(data: Dict[str, np.ndarray]) -> str:
    """Escoger el lado con más articulaciones visibles según la consigna."""

    def _score(side: str) -> int:
        total = 0
        for joint in ("hip", "knee", "ankle"):
            for axis in ("x", "y"):
                key = f"{joint}_{side}_{axis}"
                series = data.get(key)
                if series is None:
                    continue
                arr = np.asarray(series, dtype=float)
                if arr.size == 0:
                    continue
                total += int(np.isfinite(arr).sum())
        return total

    left_score = _score("left")
    right_score = _score("right")
    return "left" if left_score >= right_score else "right"


def _segment_reps_state(
    *,
    knee_series: np.ndarray,
    bar_series: np.ndarray | None,
    sr: float,
    torso_scale: float,
) -> list[slice]:
    """Detectar repeticiones con histéresis en el ángulo de rodilla."""

    if knee_series is None:
        return []
    knee = np.asarray(knee_series, dtype=float)
    if knee.size == 0:
        return []

    bar: np.ndarray | None = None
    if bar_series is not None:
        bar_arr = np.asarray(bar_series, dtype=float)
        if bar_arr.size:
            bar = bar_arr

    min_gap_frames = max(1, int(round(max(1.0, sr * EVENT_MIN_GAP_SECONDS))))
    knee_down_threshold = KNEE_DOWN_THRESHOLD_DEG
    knee_up_threshold = KNEE_UP_THRESHOLD_DEG
    scale_value = float(torso_scale) if np.isfinite(torso_scale) and torso_scale > 1e-6 else 1e-3
    bar_threshold = BAR_DROP_MIN_NORM * scale_value

    state = "up"
    down_idx: int | None = None
    last_transition = -min_gap_frames * 2
    bar_high = None
    bar_low = None
    reps: list[slice] = []
    total = knee.size

    for idx in range(total):
        angle = knee[idx]
        if not np.isfinite(angle):
            continue

        bar_value = None
        if bar is not None and idx < bar.size:
            bar_value = bar[idx]

        if state == "up":
            if bar_value is not None and (bar_high is None or bar_value > bar_high):
                bar_high = bar_value
            cond = angle < knee_down_threshold
            if not cond and bar_value is not None and bar_high is not None:
                cond = (bar_high - bar_value) > bar_threshold
            if cond and (idx - last_transition) > min_gap_frames:
                state = "down"
                down_idx = idx
                last_transition = idx
                bar_low = bar_value if bar_value is not None else bar_low
        else:
            if bar_value is not None and (bar_low is None or bar_value < bar_low):
                bar_low = bar_value
            cond = angle > knee_up_threshold
            if not cond and bar_value is not None and bar_low is not None:
                cond = (bar_value - bar_low) > bar_threshold
            if cond and (idx - last_transition) > min_gap_frames:
                state = "up"
                last_transition = idx
                start = down_idx if down_idx is not None else max(0, idx - int(sr * 0.5))
                stop = idx + int(sr * 0.25)
                start = max(0, start - int(sr * 0.1))
                stop = min(total, max(start + 3, stop))
                reps.append(slice(start, stop))
                down_idx = None
                bar_high = bar_value if bar_value is not None else bar_high

    return reps


def _compute_rep_metrics(
    rep: slice,
    *,
    sr: float,
    torso_scale: float,
    knee: np.ndarray,
    hip: np.ndarray,
    elbow: np.ndarray,
    torso_tilt: np.ndarray,
    wrist_y: np.ndarray,
    shoulder_y: np.ndarray,
    hip_y: np.ndarray,
    knee_x: np.ndarray,
    knee_y: np.ndarray,
    ankle_x: np.ndarray,
    ankle_y: np.ndarray,
    bar_x: np.ndarray,
) -> RepMetrics:
    """Calcular métricas cinemáticas para una repetición individual."""

    start = max(0, int(rep.start or 0))
    stop = max(start, int(rep.stop or start))
    if stop - start < 2:
        return RepMetrics(*(float("nan") for _ in range(13)))

    def _segment(series: np.ndarray | None) -> np.ndarray:
        if series is None:
            return np.array([])
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            return np.array([])
        start_idx = max(0, min(start, arr.size))
        stop_idx = max(start_idx, min(stop, arr.size))
        return arr[start_idx:stop_idx]

    knee_seg = _segment(knee)
    hip_seg = _segment(hip)
    elbow_seg = _segment(elbow)
    torso_seg = _segment(torso_tilt)
    wrist_seg = _segment(wrist_y)
    shoulder_seg = _segment(shoulder_y)
    hip_y_seg = _segment(hip_y)
    knee_x_seg = _segment(knee_x)
    knee_y_seg = _segment(knee_y)
    ankle_x_seg = _segment(ankle_x)
    ankle_y_seg = _segment(ankle_y)
    bar_x_seg = _segment(bar_x)

    torso_norm = float(torso_scale) if np.isfinite(torso_scale) and torso_scale > 1e-6 else float("nan")

    knee_min = _finite_stat(knee_seg, np.min)
    hip_min = _finite_stat(hip_seg, np.min)
    elbow_min = _finite_stat(elbow_seg, np.min)

    if knee_seg.size:
        finite_knee = knee_seg[np.isfinite(knee_seg)]
        threshold = finite_knee.min() + 10.0 if finite_knee.size else float("inf")
        mask = np.isfinite(knee_seg) & (knee_seg <= threshold)
    else:
        mask = np.array([], dtype=bool)

    def _masked_stat(values: np.ndarray, default: float = float("nan")) -> float:
        if values.size == 0 or mask.size == 0:
            return float(default)
        masked = values[mask & np.isfinite(values)]
        if masked.size == 0:
            return float(default)
        return float(np.median(masked))

    wrist_shoulder = wrist_seg - shoulder_seg
    wrist_hip = wrist_seg - hip_y_seg
    knee_forward = knee_x_seg - ankle_x_seg
    tibia_angles = np.degrees(
        np.arctan2(np.abs(knee_x_seg - ankle_x_seg), np.abs(knee_y_seg - ankle_y_seg) + 1e-6)
    )
    bar_ankle = np.abs(bar_x_seg - ankle_x_seg)

    wrist_shoulder_norm = _masked_stat(wrist_shoulder / torso_norm)
    wrist_hip_norm = _masked_stat(wrist_hip / torso_norm)
    knee_forward_norm = _masked_stat(knee_forward / torso_norm)
    tibia_angle = _masked_stat(tibia_angles)
    bar_ankle_norm = _masked_stat(bar_ankle / torso_norm)

    knee_rom = _range_of(knee_seg)
    hip_rom = _range_of(hip_seg)
    elbow_rom = _range_of(elbow_seg)
    bar_range_norm = _normalized_range(wrist_seg, torso_norm)
    torso_bottom = _masked_stat(torso_seg)

    duration_s = (stop - start) / sr if sr > 0 else 0.0

    return RepMetrics(
        knee_min=float(knee_min),
        hip_min=float(hip_min),
        elbow_bottom=float(elbow_min),
        torso_tilt_bottom=float(torso_bottom),
        wrist_shoulder_diff_norm=float(wrist_shoulder_norm),
        wrist_hip_diff_norm=float(wrist_hip_norm),
        knee_forward_norm=float(knee_forward_norm),
        tibia_angle_deg=float(tibia_angle),
        bar_ankle_diff_norm=float(bar_ankle_norm),
        knee_rom=float(knee_rom),
        hip_rom=float(hip_rom),
        elbow_rom=float(elbow_rom),
        bar_range_norm=float(bar_range_norm),
        duration_s=float(duration_s),
    )


def _range_of(series: np.ndarray | None) -> float:
    if series is None or series.size == 0:
        return 0.0
    valid = series[~np.isnan(series)]
    if valid.size == 0:
        return 0.0
    return float(valid.max() - valid.min())


def _finite_stat(series: np.ndarray | None, reducer) -> float:
    if series is None or series.size == 0:
        return float("nan")
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return float("nan")
    return float(reducer(finite))


def _finite_percentile(series: np.ndarray | None, percentile: float) -> float:
    if series is None or series.size == 0:
        return float("nan")
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return float("nan")
    percentile = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(finite, percentile))


def _score(value: float, threshold: float, scale: float = 1.5) -> float:
    if threshold <= 0:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    if value <= 0:
        return 0.0
    ratio = value / threshold
    ratio = max(0.0, min(ratio, scale * 2.0))
    return ratio


def _score_inverse(value: float, threshold: float, scale: float = 1.5) -> float:
    if threshold <= 0:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    if value <= 0:
        return 0.0
    headroom = threshold - value
    if headroom <= 0:
        return 0.0
    ratio = headroom / threshold
    ratio = max(0.0, min(ratio, scale * 2.0))
    return ratio


def _pick_label(scores: Dict[str, float]) -> Tuple[str, float]:
    values = np.array(list(scores.values()), dtype=float)
    labels = list(scores.keys())
    max_index = int(np.argmax(values))
    max_value = float(values[max_index])
    sorted_indices = np.argsort(values)
    second_best = float(values[sorted_indices[-2]]) if values.size > 1 else 0.0

    if max_value < MIN_CONFIDENCE_SCORE or (max_value - second_best) < CLASSIFICATION_MARGIN:
        return "unknown", 0.0

    exp_scores = np.exp(values - np.max(values))
    probs = exp_scores / exp_scores.sum()
    confidence = float(probs[max_index])
    return labels[max_index], confidence


def _pick_view_label(scores: Dict[str, float], evidence: int) -> str:
    if not scores:
        return "unknown"

    labels = list(scores.keys())
    values = np.array(list(scores.values()), dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return "unknown"

    values = values[finite_mask]
    labels = [label for label, mask in zip(labels, finite_mask) if mask]

    max_index = int(np.argmax(values))
    max_value = float(values[max_index])
    if max_value <= 0:
        return "unknown"

    if values.size > 1:
        sorted_values = np.sort(values)
        second_best = float(sorted_values[-2])
    else:
        second_best = 0.0

    evidence = max(1, int(evidence))
    min_required = VIEW_SCORE_PER_EVIDENCE_THRESHOLD * evidence
    margin_required = VIEW_MARGIN_PER_EVIDENCE_THRESHOLD * evidence

    if max_value < min_required:
        return "unknown"

    if (max_value - second_best) < margin_required:
        return "unknown"

    return labels[max_index]


def _classify_view(
    shoulder_width_series: np.ndarray | None,
    shoulder_yaw_series: np.ndarray | None = None,
    shoulder_z_delta_series: np.ndarray | None = None,
    ankle_width_series: np.ndarray | None = None,
) -> str:
    yaw_med = _finite_stat(shoulder_yaw_series, np.median)
    yaw_p75 = _finite_percentile(shoulder_yaw_series, 75)
    z_med = _finite_stat(shoulder_z_delta_series, np.median)
    width_mean = _finite_stat(shoulder_width_series, np.mean)
    width_std = _finite_stat(shoulder_width_series, np.std)
    width_p10 = _finite_percentile(shoulder_width_series, 10)
    ankle_mean = _finite_stat(ankle_width_series, np.mean)
    ankle_std = _finite_stat(ankle_width_series, np.std)
    ankle_p10 = _finite_percentile(ankle_width_series, 10)

    front_score = 0.0
    side_score = 0.0
    evidence = 0

    front_votes = 0
    side_votes = 0

    yaw_frontish = np.isfinite(yaw_med) and yaw_med <= YAW_FRONT_MAX_DEG * 1.05
    yaw_strong_front = np.isfinite(yaw_med) and yaw_med <= YAW_FRONT_MAX_DEG * 0.85
    yaw_strong_side = np.isfinite(yaw_p75) and yaw_p75 >= YAW_SIDE_MIN_DEG * 1.05
    z_frontish = np.isfinite(z_med) and z_med <= Z_DELTA_FRONT_MAX * 1.2

    if np.isfinite(yaw_med):
        yaw_weight = 1.4 if yaw_strong_front else 1.0
        front_score += yaw_weight * _score_inverse(yaw_med, YAW_FRONT_MAX_DEG, scale=2.0)
        if yaw_med <= YAW_FRONT_MAX_DEG * 1.05:
            front_votes += 1
        if yaw_med >= YAW_SIDE_MIN_DEG * 0.9:
            side_votes += 1
        evidence += 1
    if np.isfinite(yaw_p75):
        side_score += _score(yaw_p75, YAW_SIDE_MIN_DEG, scale=2.0)
        if yaw_p75 >= YAW_SIDE_MIN_DEG:
            side_votes += 1

    if np.isfinite(z_med):
        front_score += _score_inverse(z_med, Z_DELTA_FRONT_MAX, scale=2.0)
        side_score += _score(z_med, Z_DELTA_FRONT_MAX * 1.6, scale=2.0)
        if z_med <= Z_DELTA_FRONT_MAX * 1.05:
            front_votes += 1
        if z_med >= Z_DELTA_FRONT_MAX * 1.6:
            side_votes += 1
        evidence += 1

    if np.isfinite(width_mean):
        front_score += _score(width_mean, VIEW_FRONT_WIDTH_THRESHOLD, scale=2.0)
        side_weight = 0.4 if yaw_frontish and z_frontish else 1.0
        side_score += side_weight * _score_inverse(width_mean, SIDE_WIDTH_MAX, scale=2.0)
        if width_mean >= VIEW_FRONT_WIDTH_THRESHOLD * 0.96:
            front_votes += 1
        if width_mean <= SIDE_WIDTH_MAX * 1.04:
            side_votes += 1
        evidence += 1

    if np.isfinite(width_std):
        front_score += _score_inverse(width_std, VIEW_WIDTH_STD_THRESHOLD, scale=2.0)
        side_weight = 0.45 if yaw_frontish and z_frontish else 1.0
        side_score += side_weight * _score(width_std, VIEW_WIDTH_STD_THRESHOLD * 0.9, scale=2.0)
        if width_std <= VIEW_WIDTH_STD_THRESHOLD * 0.9:
            front_votes += 1
        if width_std >= VIEW_WIDTH_STD_THRESHOLD * 1.05:
            side_votes += 1
        evidence += 1

    if np.isfinite(width_p10):
        front_score += _score(width_p10, VIEW_FRONT_WIDTH_THRESHOLD * 0.9, scale=1.8)
        side_weight = 0.45 if yaw_frontish and z_frontish else 1.0
        side_score += side_weight * _score_inverse(width_p10, SIDE_WIDTH_MAX * 1.1, scale=1.8)
        if width_p10 >= VIEW_FRONT_WIDTH_THRESHOLD * 0.85:
            front_votes += 1
        if width_p10 <= SIDE_WIDTH_MAX * 1.1:
            side_votes += 1

    if np.isfinite(ankle_mean):
        front_score += 0.8 * _score(ankle_mean, ANKLE_FRONT_WIDTH_THRESHOLD * 0.95, scale=2.0)
        side_weight = 0.5 if yaw_frontish else 1.0
        side_score += 0.8 * side_weight * _score_inverse(
            ankle_mean, ANKLE_SIDE_WIDTH_MAX * 1.05, scale=2.0
        )
        if ankle_mean >= ANKLE_FRONT_WIDTH_THRESHOLD * 0.9:
            front_votes += 1
        if ankle_mean <= ANKLE_SIDE_WIDTH_MAX * 1.05:
            side_votes += 1
        evidence += 1

    if np.isfinite(ankle_std):
        front_score += 0.6 * _score_inverse(ankle_std, ANKLE_WIDTH_STD_THRESHOLD * 1.1, scale=2.0)
        side_weight = 0.5 if yaw_frontish else 1.0
        side_score += 0.6 * side_weight * _score(ankle_std, ANKLE_WIDTH_STD_THRESHOLD, scale=2.0)
        if ankle_std <= ANKLE_WIDTH_STD_THRESHOLD:
            front_votes += 1
        if ankle_std >= ANKLE_WIDTH_STD_THRESHOLD * 1.05:
            side_votes += 1
        evidence += 1

    if np.isfinite(ankle_p10):
        front_score += 0.6 * _score(ankle_p10, ANKLE_FRONT_WIDTH_THRESHOLD * 0.85, scale=1.8)
        side_weight = 0.5 if yaw_frontish else 1.0
        side_score += 0.6 * side_weight * _score_inverse(
            ankle_p10, ANKLE_SIDE_WIDTH_MAX * 1.15, scale=1.8
        )
        if ankle_p10 >= ANKLE_FRONT_WIDTH_THRESHOLD * 0.82:
            front_votes += 1
        if ankle_p10 <= ANKLE_SIDE_WIDTH_MAX * 1.1:
            side_votes += 1

    scores = {"front": float(front_score), "side": float(side_score)}
    view = _pick_view_label(scores, evidence)

    logger.info(
        "VIEW DEBUG — scores=%s evidence=%d yawMed=%.1f yawP75=%.1f zMed=%.3f widthMean=%.3f "
        "widthStd=%.3f widthP10=%.3f ankleMean=%.3f ankleStd=%.3f ankleP10=%.3f "
        "frontVotes=%d sideVotes=%d",
        scores,
        evidence,
        float(yaw_med) if np.isfinite(yaw_med) else float("nan"),
        float(yaw_p75) if np.isfinite(yaw_p75) else float("nan"),
        float(z_med) if np.isfinite(z_med) else float("nan"),
        float(width_mean) if np.isfinite(width_mean) else float("nan"),
        float(width_std) if np.isfinite(width_std) else float("nan"),
        float(width_p10) if np.isfinite(width_p10) else float("nan"),
        float(ankle_mean) if np.isfinite(ankle_mean) else float("nan"),
        float(ankle_std) if np.isfinite(ankle_std) else float("nan"),
        float(ankle_p10) if np.isfinite(ankle_p10) else float("nan"),
        front_votes,
        side_votes,
    )

    if view == "side" and yaw_frontish and (front_votes >= side_votes or z_frontish):
        return "front"
    if view == "front" and yaw_strong_side and side_votes > front_votes:
        return "side"

    if view == "unknown":
        if np.isfinite(yaw_med):
            if yaw_med <= VIEW_FRONT_FALLBACK_YAW_DEG:
                return "front"
            if yaw_med >= VIEW_SIDE_FALLBACK_YAW_DEG:
                return "side"
        if np.isfinite(z_med):
            if z_med <= Z_DELTA_FRONT_MAX:
                return "front"
            if z_med >= Z_DELTA_FRONT_MAX * 1.8:
                return "side"
        if np.isfinite(ankle_mean):
            if ankle_mean >= ANKLE_FRONT_WIDTH_THRESHOLD * 0.90:
                return "front"
            if ankle_mean <= ANKLE_SIDE_WIDTH_MAX * 1.10:
                return "side"
        if np.isfinite(width_mean):
            return "front" if width_mean >= VIEW_FRONT_WIDTH_THRESHOLD else "side"

    return view


def _smooth(
    series: np.ndarray,
    sr: float | None = None,
    cadence_hz: float | None = None,
    min_ms: float = 180.0,
    max_ms: float = 300.0,
    poly: int = 2,
) -> np.ndarray:
    """Devolver una copia suavizada con una ventana Savitzky–Golay adaptativa.

    Ajustamos la ventana en función de la cadencia estimada para conservar la
    forma de las repeticiones rápidas sin introducir demasiado retraso ni
    atenuar picos relevantes.
    """

    if series is None:
        return np.array([])

    x = np.asarray(series, dtype=float)
    if x.size == 0:
        return np.array([])

    mask = np.isfinite(x)
    finite_count = int(mask.sum())
    if finite_count < 5:
        return x

    if sr is None or sr <= 0:
        window = 11
    else:
        if cadence_hz is not None and cadence_hz > 0:
            period_s = 1.0 / cadence_hz
            target_ms = float(np.clip(0.20 * period_s * 1000.0, min_ms, max_ms))
        else:
            target_ms = float(np.clip(250.0, min_ms, max_ms))
        window = int(round(sr * target_ms / 1000.0))
        window = max(5, min(window, 51))

    if window % 2 == 0:
        window += 1

    if window > x.size:
        window = x.size if x.size % 2 == 1 else x.size - 1

    if window < 5:
        return x

    if not mask.all():
        finite_idx = np.flatnonzero(mask)
        if finite_idx.size == 0:
            return x
        x[~mask] = np.interp(np.flatnonzero(~mask), finite_idx, x[mask])

    if finite_count < window:
        window = finite_count if finite_count % 2 == 1 else finite_count - 1
        if window < 5:
            return x

    polyorder = int(min(poly, max(1, window - 1)))

    try:
        return savgol_filter(x, window_length=int(window), polyorder=polyorder)
    except Exception:
        return x


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / (s + 1e-6)


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


__all__ = ["classify_features"]
