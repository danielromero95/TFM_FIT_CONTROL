"""Conteo de repeticiones mediante detección de valles y consolidación por periodo refractario."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

from src import config
from src.B_pose_estimation.signal import derivative, interpolate_small_gaps, smooth_series
from src.C_analysis.rep_candidates import (
    EXERCISE_SPECS,
    ExerciseRepSpec,
    RepCandidate,
    RejectionReason,
    Zone,
    detect_rep_candidates,
)
from src.config.constants import (
    ANALYSIS_MAX_GAP_FRAMES,
    ANALYSIS_SAVGOL_POLYORDER,
    ANALYSIS_SAVGOL_WINDOW_SEC,
    ANALYSIS_SMOOTH_METHOD,
)

logger = logging.getLogger(__name__)
EPS_DEG = 0.0


@dataclass
class CountingDebugInfo:
    """Carga de depuración que devuelve el contador de repeticiones."""
    valley_indices: List[int]
    prominences: List[float]
    raw_count: int = 0
    reps_rejected_threshold: int = 0
    rejection_reasons: List[str] | None = None
    rep_candidates: list[dict] = field(default_factory=list)
    rep_intervals: list[tuple[int, int]] = field(default_factory=list)


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


def _frame_index_bounds(length: int, start: int | None, end: int | None) -> tuple[int, int]:
    start_idx = 0 if start is None else max(0, min(int(start), length - 1))
    end_idx = length - 1 if end is None else max(start_idx, min(int(end), length - 1))
    return start_idx, end_idx


def _close_deadlift_candidates_with_fallback(
    rep_candidates: list,
    df_metrics: pd.DataFrame,
    fps: float,
    *,
    fallback_metric: str = "trunk_inclination_deg",
) -> None:
    """Try to close incomplete deadlift reps using a fallback metric."""

    if df_metrics is None or df_metrics.empty or fallback_metric not in df_metrics.columns:
        return

    try:
        fallback_smoothed, _, fallback_quality, _ = _prepare_primary_series(
            df_metrics, fallback_metric, fps
        )
    except Exception:
        return

    if fallback_smoothed.size == 0:
        return

    bottom_is_max = True
    metric_name = fallback_metric.lower()
    if "trunk_inclination" not in metric_name and not metric_name.startswith("trunk_"):
        bottom_is_max = False

    length = len(fallback_smoothed)

    for cand in rep_candidates:
        if getattr(cand, "end_frame", None) is not None:
            continue
        start_frame = getattr(cand, "start_frame", None)
        if start_frame is None:
            continue

        start_idx, _ = _frame_index_bounds(length, start_frame, None)
        segment = fallback_smoothed[start_idx:]
        quality = fallback_quality[start_idx:]
        if segment.size == 0:
            continue

        finite_mask = np.isfinite(segment) & quality
        if not finite_mask.any():
            continue

        try:
            turning_rel = int(np.nanargmin(segment[finite_mask])) if bottom_is_max else int(np.nanargmax(segment[finite_mask]))
        except Exception:
            turning_rel = None

        if turning_rel is not None:
            finite_indices = np.nonzero(finite_mask)[0]
            turning_idx = start_idx + finite_indices[min(max(turning_rel, 0), finite_indices.size - 1)]
        else:
            turning_idx = getattr(cand, "turning_frame", start_idx)

        tail_start = max(turning_idx, start_idx)
        tail_segment = fallback_smoothed[tail_start:]
        tail_quality = fallback_quality[tail_start:]
        if tail_segment.size == 0:
            continue

        tail_mask = np.isfinite(tail_segment) & tail_quality
        if not tail_mask.any():
            continue

        try:
            end_rel = int(np.nanargmax(tail_segment[tail_mask])) if bottom_is_max else int(np.nanargmin(tail_segment[tail_mask]))
        except Exception:
            continue

        tail_indices = np.nonzero(tail_mask)[0]
        end_idx = tail_start + tail_indices[min(max(end_rel, 0), tail_indices.size - 1)]

        if end_idx <= turning_idx:
            continue

        cand.turning_frame = cand.turning_frame or turning_idx
        cand.end_frame = end_idx


def _state_machine_reps(
    angles: np.ndarray,
    velocities: np.ndarray,
    *,
    refractory_frames: int,
    min_excursion: float,
    min_prominence: float,
    min_distance_frames: int,
    reference: np.ndarray | None = None,
) -> tuple[list[int], list[float], bool]:
    finite_angles = angles[np.isfinite(angles)]
    if finite_angles.size == 0:
        return [], [], False

    angle_range = float(np.nanmax(finite_angles) - np.nanmin(finite_angles))
    if not np.isfinite(angle_range) or angle_range < min_excursion:
        return [], [], False

    top_thr = float(np.nanpercentile(finite_angles, 70))
    bottom_thr = float(np.nanpercentile(finite_angles, 30))
    vel_valid = np.abs(velocities[np.isfinite(velocities)])
    vel_thr = max(2.0, float(np.nanpercentile(vel_valid, 65))) if vel_valid.size else 2.0

    state = "IDLE"
    last_rep_frame = -10 * refractory_frames
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
                bottom_idx = idx
                state = "BOTTOM"
        elif state == "BOTTOM":
            if ascending:
                state = "CONCENTRIC"
        elif state == "CONCENTRIC":
            if angle >= top_thr:
                if idx - last_rep_frame >= max(refractory_frames, min_distance_frames):
                    prominence = float(np.nanmax([top_thr - bottom_value, angle_range]))
                    if prominence >= min_prominence:
                        source = reference if reference is not None else angles
                        start = max(0, last_rep_frame + 1)
                        bottom_window = source[start : idx + 1]
                        if bottom_window.size:
                            local_min = int(np.nanargmin(bottom_window))
                            valley_idx = start + local_min
                        else:
                            valley_idx = bottom_idx if bottom_idx >= 0 else idx
                        rep_indices.append(valley_idx)
                        prominences.append(prominence)
                        last_rep_frame = idx
                state = "TOP"

    if long_gap_seen:
        return [], [], True
    return rep_indices, prominences, False


def _as_optional_float(value: object) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _rep_intervals_from_valleys(series_length: int, valley_indices: list[int]) -> list[tuple[int, int]]:
    if series_length <= 0 or not valley_indices:
        return []

    bounded = [max(0, min(int(idx), series_length - 1)) for idx in valley_indices]
    if not bounded:
        return []

    intervals: list[tuple[int, int]] = []
    for i, valley in enumerate(bounded):
        start = 0 if i == 0 else int(round((bounded[i - 1] + valley) / 2))
        end = series_length - 1 if i == len(bounded) - 1 else int(round((valley + bounded[i + 1]) / 2))
        start = max(0, min(start, series_length - 1))
        end = max(start, min(end, series_length - 1))
        intervals.append((start, end))
    return intervals


def _filter_reps_by_thresholds(
    angles: np.ndarray,
    valley_indices: list[int],
    prominences: list[float],
    *,
    low_thresh: float | None,
    high_thresh: float | None,
    enforce_low: bool = True,
    enforce_high: bool = True,
) -> tuple[list[int], list[float], int, list[str]]:
    apply_low = enforce_low and low_thresh is not None
    apply_high = enforce_high and high_thresh is not None
    if not apply_low and not apply_high:
        return valley_indices, prominences, 0, []

    intervals = _rep_intervals_from_valleys(len(angles), valley_indices)
    filtered_indices: list[int] = []
    filtered_proms: list[float] = []
    rejection_reasons: list[str] = []

    for idx, prominence, interval in zip(valley_indices, prominences, intervals):
        start, end = interval
        window = angles[start : end + 1]
        finite = window[np.isfinite(window)]
        if finite.size < 2:
            rejection_reasons.append(
                f"Rep at {idx} discarded: insufficient finite samples between {start} and {end}."
            )
            continue

        min_angle = float(np.nanmin(finite))
        max_angle = float(np.nanmax(finite))

        low_ok = (not apply_low) or (min_angle <= low_thresh)  # type: ignore[operator]
        high_ok = (not apply_high) or (max_angle >= high_thresh)  # type: ignore[operator]

        if low_ok and high_ok:
            filtered_indices.append(idx)
            filtered_proms.append(prominence)
            continue

        if not low_ok and not high_ok:
            reason = (
                f"Rep at {idx} discarded: min {min_angle:.1f}° above low_thresh {low_thresh}°"
                f" and max {max_angle:.1f}° below high_thresh {high_thresh}°."
            )
        elif not low_ok:
            reason = f"Rep at {idx} discarded: min {min_angle:.1f}° above low_thresh {low_thresh}°."
        else:
            reason = f"Rep at {idx} discarded: max {max_angle:.1f}° below high_thresh {high_thresh}°."
        rejection_reasons.append(reason)

    rejected = len(valley_indices) - len(filtered_indices)
    return filtered_indices, filtered_proms, rejected, rejection_reasons


def _threshold_zone_reps(
    values: np.ndarray,
    *,
    low_thresh: float,
    high_thresh: float,
    min_distance_frames: int,
) -> list[tuple[int, int, int]]:
    reps: list[tuple[int, int, int]] = []
    if values.size == 0:
        return reps

    last_high_idx: int | None = None
    start_high_idx: int | None = None
    low_idx: int | None = None
    low_value = np.inf
    last_completed_end = -min_distance_frames
    state = "WAIT_LOW"

    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue

        if value >= high_thresh:
            last_high_idx = idx

        if state == "WAIT_LOW":
            if value <= low_thresh and last_high_idx is not None:
                start_high_idx = last_high_idx
                low_idx = idx
                low_value = value
                state = "WAIT_HIGH"
        else:
            if value <= low_thresh:
                if value < low_value:
                    low_value = value
                    low_idx = idx
                continue
            if value >= high_thresh and start_high_idx is not None and low_idx is not None:
                if idx - last_completed_end >= min_distance_frames:
                    reps.append((start_high_idx, low_idx, idx))
                    last_completed_end = idx
                start_high_idx = idx
                low_idx = None
                low_value = np.inf
                state = "WAIT_LOW"

    return reps


def _build_edge_candidate(
    values: np.ndarray,
    *,
    start_idx: int,
    low_idx: int,
    end_idx: int,
    exercise_key: str,
) -> RepCandidate | None:
    if (
        start_idx < 0
        or end_idx <= start_idx
        or end_idx >= len(values)
        or low_idx < start_idx
        or low_idx > end_idx
    ):
        return None

    segment = values[start_idx : end_idx + 1]
    finite_mask = np.isfinite(segment)
    if not finite_mask.any():
        return None

    finite_values = segment[finite_mask]
    min_angle = float(np.nanmin(finite_values))
    max_angle = float(np.nanmax(finite_values))
    turning_idx = low_idx

    candidate = RepCandidate(
        rep_index=0,
        start_frame=start_idx,
        turning_frame=turning_idx,
        end_frame=end_idx,
        min_angle=min_angle,
        max_angle=max_angle,
    )

    spec = EXERCISE_SPECS.get(exercise_key.lower(), EXERCISE_SPECS["squat"])
    _apply_rep_phases(candidate, spec)
    return candidate


def _apply_rep_phases(candidate: RepCandidate, spec: ExerciseRepSpec) -> None:
    if spec.phases[0] == "Down":
        candidate.down_start = candidate.start_frame
        candidate.down_end = candidate.turning_frame
        candidate.up_start = candidate.turning_frame
        candidate.up_end = candidate.end_frame
    else:
        candidate.up_start = candidate.start_frame
        candidate.up_end = candidate.turning_frame
        candidate.down_start = candidate.turning_frame
        candidate.down_end = candidate.end_frame


def _reindex_candidates(candidates: list[RepCandidate]) -> None:
    candidates.sort(key=lambda cand: (cand.start_frame, cand.end_frame or -1))
    for idx, cand in enumerate(candidates):
        cand.rep_index = idx


def _count_passing_threshold_reps(
    candidates: list[RepCandidate],
    values: np.ndarray,
    *,
    low_thresh: float,
    high_thresh: float,
) -> int:
    count = 0
    for cand in candidates:
        if cand.rejection_reason == RejectionReason.INCOMPLETE:
            continue
        if cand.start_frame is None or cand.end_frame is None:
            continue
        start = max(0, min(int(cand.start_frame), len(values) - 1))
        end = max(start, min(int(cand.end_frame), len(values) - 1))
        segment = values[start : end + 1]
        finite = segment[np.isfinite(segment)]
        if finite.size == 0:
            continue
        min_angle = float(np.nanmin(finite))
        max_angle = float(np.nanmax(finite))
        if min_angle <= low_thresh and max_angle >= high_thresh:
            count += 1
    return count


def count_repetitions_with_config(
    df_metrics: pd.DataFrame,
    counting_cfg: config.CountingConfig,
    fps: float,
    *,
    faults_cfg: config.FaultConfig | None = None,
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

    state_valleys, state_proms, long_gap = _state_machine_reps(
        angles,
        velocities,
        refractory_frames=refractory_frames,
        min_excursion=min_excursion,
        min_prominence=min_prominence,
        min_distance_frames=min_distance_frames,
        reference=reference,
    )

    finite_angles = angles[np.isfinite(angles)]
    auto_low = float(np.nanpercentile(finite_angles, 20)) if finite_angles.size else 0.0
    auto_high = float(np.nanpercentile(finite_angles, 80)) if finite_angles.size else 0.0
    low_thresh_cfg = _as_optional_float(getattr(faults_cfg, "low_thresh", None) if faults_cfg else None)
    high_thresh_cfg = _as_optional_float(getattr(faults_cfg, "high_thresh", None) if faults_cfg else None)

    pass_low = low_thresh_cfg if low_thresh_cfg is not None else auto_low
    pass_high = high_thresh_cfg if high_thresh_cfg is not None else auto_high

    detect_low = auto_low
    detect_high = auto_high

    enforce_low_flag = bool(getattr(counting_cfg, "enforce_low_thresh", True))
    enforce_high_flag = bool(getattr(counting_cfg, "enforce_high_thresh", True))

    exercise_key = getattr(counting_cfg, "exercise", "squat")
    rep_candidates = detect_rep_candidates(
        reference if reference is not None else angles,
        low_thresh=detect_low,
        high_thresh=detect_high,
        exercise_key=exercise_key,
        enforce_low=False,
        enforce_high=False,
    )

    if long_gap:
        rep_candidates = []

    exercise_key = exercise_key.lower()

    if (
        enforce_low_flag
        and enforce_high_flag
        and low_thresh_cfg is not None
        and high_thresh_cfg is not None
    ):
        spec = EXERCISE_SPECS.get(exercise_key, EXERCISE_SPECS["squat"])
        if spec.start_zone == Zone.HIGH and spec.completion_zone == Zone.HIGH:
            values = reference if reference is not None else angles
            threshold_reps = _threshold_zone_reps(
                values,
                low_thresh=low_thresh_cfg,
                high_thresh=high_thresh_cfg,
                min_distance_frames=min_distance_frames,
            )
            if threshold_reps:
                current_pass_count = _count_passing_threshold_reps(
                    rep_candidates,
                    values,
                    low_thresh=low_thresh_cfg,
                    high_thresh=high_thresh_cfg,
                )
                tolerance = 3
                short_threshold = max(2, int(round(min_distance_frames / 2)))
                matched_canonical_count = 0
                used_candidates: set[int] = set()
                has_deviation = False

                for start_idx, low_idx, end_idx in threshold_reps:
                    matched_idx = None
                    for idx, cand in enumerate(rep_candidates):
                        if idx in used_candidates:
                            continue
                        if cand.turning_frame is None:
                            continue
                        if abs(cand.turning_frame - low_idx) <= tolerance:
                            matched_idx = idx
                            break
                    if matched_idx is None:
                        continue
                    used_candidates.add(matched_idx)
                    matched_canonical_count += 1
                    cand = rep_candidates[matched_idx]
                    if cand.start_frame is None or cand.end_frame is None:
                        has_deviation = True
                    else:
                        if (
                            abs(cand.start_frame - start_idx) > tolerance
                            or abs(cand.end_frame - end_idx) > tolerance
                        ):
                            has_deviation = True

                suspicious_candidates = matched_canonical_count != len(rep_candidates)
                last_canonical_end = threshold_reps[-1][2]
                last_end_early = True
                max_end = None
                for cand in rep_candidates:
                    if cand.start_frame is None or cand.end_frame is None or cand.turning_frame is None:
                        suspicious_candidates = True
                        continue
                    duration = cand.end_frame - cand.start_frame
                    if duration < short_threshold:
                        suspicious_candidates = True
                    if max_end is None or cand.end_frame > max_end:
                        max_end = cand.end_frame
                if max_end is not None:
                    last_end_early = max_end < last_canonical_end - tolerance
                if max_end is None:
                    last_end_early = True

                should_reconcile = (
                    matched_canonical_count < len(threshold_reps)
                    or has_deviation
                    or suspicious_candidates
                    or last_end_early
                )

                if should_reconcile:
                    reconciled: list[RepCandidate] = []
                    used_candidates.clear()

                    for start_idx, low_idx, end_idx in threshold_reps:
                        matched_idx = None
                        for idx, cand in enumerate(rep_candidates):
                            if idx in used_candidates:
                                continue
                            if cand.turning_frame is None:
                                continue
                            if abs(cand.turning_frame - low_idx) <= tolerance:
                                matched_idx = idx
                                break

                        if matched_idx is not None:
                            cand = rep_candidates[matched_idx]
                            used_candidates.add(matched_idx)
                            cand.start_frame = start_idx
                            cand.turning_frame = low_idx
                            cand.end_frame = end_idx
                            segment = values[start_idx : end_idx + 1]
                            finite = segment[np.isfinite(segment)]
                            if finite.size:
                                cand.min_angle = float(np.nanmin(finite))
                                cand.max_angle = float(np.nanmax(finite))
                            else:
                                cand.min_angle = None
                                cand.max_angle = None
                            _apply_rep_phases(cand, spec)
                            reconciled.append(cand)
                        else:
                            new_candidate = _build_edge_candidate(
                                values,
                                start_idx=start_idx,
                                low_idx=low_idx,
                                end_idx=end_idx,
                                exercise_key=exercise_key,
                            )
                            if new_candidate is not None:
                                reconciled.append(new_candidate)

                    if reconciled:
                        reconciled_pass_count = _count_passing_threshold_reps(
                            reconciled,
                            values,
                            low_thresh=low_thresh_cfg,
                            high_thresh=high_thresh_cfg,
                        )
                        if reconciled_pass_count >= current_pass_count:
                            rep_candidates = reconciled
                            _reindex_candidates(rep_candidates)

    if not rep_candidates and state_valleys:
        for i, valley in enumerate(state_valleys):
            end_frame = min(len(angles) - 1, valley + 1)
            cand = RepCandidate(
                rep_index=i,
                start_frame=valley,
                turning_frame=valley,
                end_frame=end_frame,
                min_angle=float(reference[valley]) if reference is not None and np.isfinite(reference[valley]) else float(auto_low),
                max_angle=float(reference[valley]) if reference is not None and np.isfinite(reference[valley]) else float(auto_high),
            )
            rep_candidates.append(cand)

    for cand in rep_candidates:
        cand.passed_low = bool(cand.min_angle is not None and cand.min_angle <= pass_low)
        cand.passed_high = bool(cand.max_angle is not None and cand.max_angle >= pass_high)
        if exercise_key == "deadlift" and not enforce_low_flag:
            cand.passed_low = True

    if exercise_key == "deadlift":
        _close_deadlift_candidates_with_fallback(rep_candidates, df_metrics, fps_safe)

    for cand in rep_candidates:
        if cand.rejection_reason == RejectionReason.INCOMPLETE:
            continue
        cand.accepted = True
        cand.rejection_reason = RejectionReason.NONE
        if enforce_low_flag and not cand.passed_low:
            cand.accepted = False
            cand.rejection_reason = RejectionReason.LOW_THRESH
        if enforce_high_flag and not cand.passed_high:
            cand.accepted = False
            if cand.rejection_reason == RejectionReason.NONE:
                cand.rejection_reason = RejectionReason.HIGH_THRESH

    for cand in rep_candidates:
        if cand.rejection_reason == RejectionReason.INCOMPLETE and (
            not enforce_low_flag or cand.passed_low
        ):
            if enforce_high_flag and not cand.passed_high:
                continue
            cand.accepted = True
            cand.rejection_reason = RejectionReason.NONE
            if cand.end_frame is None:
                cand.end_frame = len(angles) - 1

    raw_count = len(rep_candidates)
    reps = sum(1 for r in rep_candidates if r.accepted)
    threshold_rejected = sum(
        1
        for r in rep_candidates
        if r.rejection_reason
        in (RejectionReason.LOW_THRESH, RejectionReason.HIGH_THRESH)
    )

    valley_for_debug = state_valleys if state_valleys else [c.turning_frame or c.start_frame for c in rep_candidates]
    debug = CountingDebugInfo(
        valley_indices=valley_for_debug,
        prominences=state_proms if state_proms else [0.0 for _ in rep_candidates],
        raw_count=raw_count,
        reps_rejected_threshold=threshold_rejected,
        rejection_reasons=[r.rejection_reason.value for r in rep_candidates],
        rep_candidates=[r.as_dict() for r in rep_candidates],
        rep_intervals=[
            (c.start_frame, c.end_frame)
            for c in rep_candidates
            if c.end_frame is not None and c.start_frame is not None
        ],
    )

    logger.debug(
        "Repetition count=%d (raw=%d, refractory=%d frames ≈ %.2fs, min_excursion=%.1f). Valid frames=%d/%d",
        reps,
        raw_count,
        refractory_frames,
        refractory_frames / fps_safe,
        min_excursion,
        int(np.isfinite(angles).sum()),
        len(angles),
    )
    return reps, debug
