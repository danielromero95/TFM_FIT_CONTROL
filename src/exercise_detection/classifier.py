"""Cálculo de puntuaciones por ejercicio, lógica de vetos y confianza final."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

from .constants import (
    BENCH_BAR_HORIZONTAL_PENALTY_WEIGHT,
    BENCH_BAR_HORIZONTAL_STD_MAX,
    BENCH_BAR_RANGE_MIN_NORM,
    BENCH_BAR_RANGE_WEIGHT,
    BENCH_ELBOW_ROM_GATE_FACTOR,
    BENCH_ELBOW_ROM_MIN_DEG,
    BENCH_ELBOW_ROM_WEIGHT,
    BENCH_GATE_BONUS,
    BENCH_HIP_RANGE_MAX_NORM,
    BENCH_HIP_ROM_MAX_DEG,
    BENCH_KNEE_ROM_MAX_DEG,
    BENCH_POSTURE_WEIGHT,
    BENCH_ROM_PENALTY_WEIGHT,
    BENCH_TORSO_HORIZONTAL_DEG,
    CLASSIFICATION_MARGIN,
    DEADLIFT_BAR_ANKLE_MAX_NORM,
    DEADLIFT_BAR_ANKLE_WEIGHT,
    DEADLIFT_BAR_HORIZONTAL_STD_MAX,
    DEADLIFT_BAR_HORIZONTAL_WEIGHT,
    DEADLIFT_BAR_RANGE_MIN_NORM,
    DEADLIFT_BAR_RANGE_WEIGHT,
    DEADLIFT_BAR_SHOULDER_MIN_NORM,
    DEADLIFT_BAR_SHOULDER_PENALTY_WEIGHT,
    DEADLIFT_BENCH_PENALTY_WEIGHT,
    DEADLIFT_ELBOW_MIN_DEG,
    DEADLIFT_ELBOW_WEIGHT,
    DEADLIFT_HIP_ROM_MIN_DEG,
    DEADLIFT_KNEE_BOTTOM_MIN_DEG,
    DEADLIFT_KNEE_FORWARD_MAX_NORM,
    DEADLIFT_KNEE_PENALTY_WEIGHT,
    DEADLIFT_LOW_MOVEMENT_CAP,
    DEADLIFT_ROM_WEIGHT,
    DEADLIFT_SQUAT_PENALTY_WEIGHT,
    DEADLIFT_SQUAT_ROM_THRESHOLD_DEG,
    DEADLIFT_TORSO_TILT_MIN_DEG,
    DEADLIFT_FRONT_TORSO_WEIGHT_FACTOR,
    DEADLIFT_HIP_KNEE_RATIO_MIN,
    DEADLIFT_KNEE_DEEP_VETO_FRACTION,
    DEADLIFT_TORSO_WEIGHT,
    DEADLIFT_VETO_MOVEMENT_MIN,
    DEADLIFT_VETO_SCORE_CLAMP,
    DEADLIFT_FRONT_VETO_SCORE_CLAMP,
    DEADLIFT_WRIST_HIP_DIFF_MIN_NORM,
    DEADLIFT_WRIST_HIP_WEIGHT,
    DEADLIFT_BAR_LOW_FRACTION_MIN,
    DEADLIFT_ELBOW_EXTENDED_FRACTION_MIN,
    DEADLIFT_FRONT_ARM_WEIGHT,
    DEADLIFT_FRONT_BAR_BONUS_WEIGHT,
    MIN_CONFIDENCE_SCORE,
    SQUAT_ARM_BONUS_WEIGHT,
    SQUAT_ARM_PENALTY_FACTOR,
    SQUAT_BAR_HIGH_BONUS_WEIGHT,
    SQUAT_BAR_HIGH_FRACTION_MIN,
    SQUAT_BAR_SHOULDER_MAX_NORM,
    SQUAT_DEPTH_WEIGHT,
    SQUAT_ELBOW_BOTTOM_MAX_DEG,
    SQUAT_ELBOW_BOTTOM_MIN_DEG,
    SQUAT_ELBOW_FLEXED_FRACTION_MIN,
    SQUAT_HINGE_PENALTY_WEIGHT,
    SQUAT_HIP_BOTTOM_MAX_DEG,
    SQUAT_DEEP_FRACTION_MIN,
    SQUAT_KNEE_BOTTOM_MAX_DEG,
    SQUAT_KNEE_DEEP_THRESHOLD_DEG,
    SQUAT_KNEE_FORWARD_MIN_NORM,
    SQUAT_KNEE_FORWARD_WEIGHT,
    SQUAT_MIN_ROM_DEG,
    SQUAT_ROM_WEIGHT,
    SQUAT_TIBIA_MAX_DEG,
    SQUAT_TIBIA_PENALTY_WEIGHT,
    SQUAT_TORSO_TILT_MAX_DEG,
    SQUAT_TORSO_WEIGHT,
    SQUAT_WRIST_SHOULDER_DIFF_MAX_NORM,
)
from .types import AggregateMetrics, ClassificationScores

LABELS = ("squat", "deadlift", "bench_press")


def classify_exercise(agg: AggregateMetrics) -> Tuple[str, float, ClassificationScores, Mapping[str, float]]:
    """Calcula la etiqueta, la confianza y las puntuaciones intermedias del ejercicio."""

    raw_scores: Dict[str, float] = {label: 0.0 for label in LABELS}
    penalties: Dict[str, float] = {label: 0.0 for label in LABELS}
    adjusted: Dict[str, float] = {label: 0.0 for label in LABELS}

    bench_score = _bench_score(agg)
    raw_scores["bench_press"] = bench_score
    adjusted["bench_press"] = max(0.0, bench_score)

    squat_raw, squat_penalty = _squat_score(agg)
    raw_scores["squat"] = squat_raw
    penalties["squat"] = squat_penalty
    squat_adjusted = max(0.0, squat_raw - squat_penalty)

    view_label = getattr(agg, "view_label", "unknown")
    deadlift_raw, deadlift_penalty = _deadlift_score(agg, view_label=view_label)
    raw_scores["deadlift"] = deadlift_raw
    penalties["deadlift"] = deadlift_penalty
    deadlift_adjusted = max(0.0, deadlift_raw - deadlift_penalty)

    deadlift_blocked = _apply_deadlift_veto(agg, bench_score, deadlift_adjusted, view_label=view_label)
    if deadlift_blocked:
        clamp = DEADLIFT_FRONT_VETO_SCORE_CLAMP if view_label == "front" else DEADLIFT_VETO_SCORE_CLAMP
        deadlift_adjusted *= clamp

    adjusted["squat"] = squat_adjusted
    adjusted["deadlift"] = deadlift_adjusted

    label, confidence, probabilities = _pick_label(adjusted, agg)

    scores = ClassificationScores(
        raw=raw_scores,
        adjusted=adjusted,
        penalties=penalties,
        deadlift_veto=deadlift_blocked,
        deadlift_blocked=deadlift_blocked,
    )
    return label, confidence, scores, probabilities


def _bench_score(agg: AggregateMetrics) -> float:
    torso_margin = _margin_above(agg.torso_tilt_bottom, BENCH_TORSO_HORIZONTAL_DEG)
    if torso_margin <= 0:
        return 0.0

    elbow_margin = _margin_above(agg.elbow_rom, BENCH_ELBOW_ROM_MIN_DEG)
    bar_range_margin = _margin_above(agg.bar_range_norm, BENCH_BAR_RANGE_MIN_NORM)
    knee_penalty = _margin_above(agg.knee_rom, BENCH_KNEE_ROM_MAX_DEG)
    hip_penalty = _margin_above(agg.hip_rom, BENCH_HIP_ROM_MAX_DEG)
    hip_range_penalty = _margin_above(agg.hip_range_norm, BENCH_HIP_RANGE_MAX_NORM)
    bar_horizontal_penalty = _margin_above(agg.bar_horizontal_std_norm, BENCH_BAR_HORIZONTAL_STD_MAX)

    score = 0.0
    score += BENCH_POSTURE_WEIGHT * torso_margin
    score += BENCH_ELBOW_ROM_WEIGHT * elbow_margin
    score += BENCH_BAR_RANGE_WEIGHT * bar_range_margin
    score -= BENCH_ROM_PENALTY_WEIGHT * knee_penalty
    score -= BENCH_ROM_PENALTY_WEIGHT * hip_penalty
    score -= BENCH_ROM_PENALTY_WEIGHT * hip_range_penalty
    score -= BENCH_BAR_HORIZONTAL_PENALTY_WEIGHT * bar_horizontal_penalty

    if _bench_gate(agg):
        score += BENCH_GATE_BONUS
        score = max(score, MIN_CONFIDENCE_SCORE)
    return max(0.0, score)


def _squat_score(agg: AggregateMetrics) -> Tuple[float, float]:
    knee_depth = _margin_below(agg.knee_min, SQUAT_KNEE_BOTTOM_MAX_DEG)
    hip_depth = _margin_below(agg.hip_min, SQUAT_HIP_BOTTOM_MAX_DEG)
    depth = max(knee_depth, hip_depth)
    torso_upright = _margin_below(agg.torso_tilt_bottom, SQUAT_TORSO_TILT_MAX_DEG)

    wrist_diff = np.abs(agg.wrist_shoulder_diff_norm)
    arm_ok = (
        np.isfinite(wrist_diff)
        and wrist_diff <= SQUAT_WRIST_SHOULDER_DIFF_MAX_NORM
        and np.isfinite(agg.elbow_bottom)
        and SQUAT_ELBOW_BOTTOM_MIN_DEG <= agg.elbow_bottom <= SQUAT_ELBOW_BOTTOM_MAX_DEG
    )

    knee_forward = _margin_above(np.abs(agg.knee_forward_norm), SQUAT_KNEE_FORWARD_MIN_NORM)
    tibia_penalty = _margin_above(agg.tibia_angle_deg, SQUAT_TIBIA_MAX_DEG)
    knee_rom_margin = _margin_above(agg.knee_rom, SQUAT_MIN_ROM_DEG)
    knee_deep_margin = _margin_above(agg.knee_deep_fraction, SQUAT_DEEP_FRACTION_MIN)

    squat_posture_ok = np.isfinite(agg.torso_tilt_bottom) and agg.torso_tilt_bottom <= (
        SQUAT_TORSO_TILT_MAX_DEG + 5.0
    )
    bar_high_bonus = 0.0
    if squat_posture_ok and np.isfinite(agg.wrist_shoulder_diff_norm):
        bar_high_bonus = _margin_below(
            np.abs(agg.wrist_shoulder_diff_norm), SQUAT_BAR_SHOULDER_MAX_NORM
        )

    bar_high_fraction = _margin_above(agg.bar_high_fraction, SQUAT_BAR_HIGH_FRACTION_MIN)
    elbow_flexion_fraction = _margin_above(agg.elbow_flexed_fraction, SQUAT_ELBOW_FLEXED_FRACTION_MIN)

    score = 0.0
    score += SQUAT_DEPTH_WEIGHT * depth
    score += SQUAT_TORSO_WEIGHT * torso_upright
    score += SQUAT_KNEE_FORWARD_WEIGHT * knee_forward
    score += SQUAT_ROM_WEIGHT * knee_rom_margin
    score += 0.25 * SQUAT_DEPTH_WEIGHT * knee_deep_margin
    score += SQUAT_BAR_HIGH_BONUS_WEIGHT * bar_high_bonus
    score += 0.5 * SQUAT_BAR_HIGH_BONUS_WEIGHT * bar_high_fraction
    score += 0.35 * SQUAT_ARM_BONUS_WEIGHT * elbow_flexion_fraction

    if arm_ok:
        score += SQUAT_ARM_BONUS_WEIGHT
    else:
        score *= SQUAT_ARM_PENALTY_FACTOR

    penalty = SQUAT_TIBIA_PENALTY_WEIGHT * max(0.0, tibia_penalty)

    hinge_cues = [
        _margin_above(agg.elbow_bottom, DEADLIFT_ELBOW_MIN_DEG),
        _margin_above(agg.wrist_hip_diff_norm, DEADLIFT_WRIST_HIP_DIFF_MIN_NORM),
        _margin_above(agg.torso_tilt_bottom, DEADLIFT_TORSO_TILT_MIN_DEG),
        _margin_below(np.abs(agg.knee_forward_norm), DEADLIFT_KNEE_FORWARD_MAX_NORM),
    ]
    penalty += SQUAT_HINGE_PENALTY_WEIGHT * sum(max(0.0, cue) for cue in hinge_cues)

    if np.isfinite(agg.torso_tilt_bottom) and agg.torso_tilt_bottom >= BENCH_TORSO_HORIZONTAL_DEG:
        score *= 0.35

    return score, penalty


def _deadlift_score(agg: AggregateMetrics, *, view_label: str = "unknown") -> Tuple[float, float]:
    hinge_posture = _margin_above(agg.torso_tilt_bottom, DEADLIFT_TORSO_TILT_MIN_DEG)
    wrist_drop = _margin_above(agg.wrist_hip_diff_norm, DEADLIFT_WRIST_HIP_DIFF_MIN_NORM)
    elbow_extension = _margin_above(agg.elbow_bottom, DEADLIFT_ELBOW_MIN_DEG)
    knee_penalty = _margin_below(agg.knee_min, DEADLIFT_KNEE_BOTTOM_MIN_DEG)

    if view_label == "front":
        hinge_posture *= DEADLIFT_FRONT_TORSO_WEIGHT_FACTOR

    bar_alignment = _margin_below(agg.bar_ankle_diff_norm, DEADLIFT_BAR_ANKLE_MAX_NORM)
    bar_vertical = _margin_above(agg.bar_range_norm, DEADLIFT_BAR_RANGE_MIN_NORM)
    hip_rom_margin = _margin_above(agg.hip_rom, DEADLIFT_HIP_ROM_MIN_DEG)
    bar_horizontal_penalty = _margin_above(agg.bar_horizontal_std_norm, DEADLIFT_BAR_HORIZONTAL_STD_MAX)
    bar_shoulder_penalty = _margin_below(
        np.abs(agg.wrist_shoulder_diff_norm), DEADLIFT_BAR_SHOULDER_MIN_NORM
    )

    score = 0.0
    score += DEADLIFT_TORSO_WEIGHT * hinge_posture
    score += DEADLIFT_WRIST_HIP_WEIGHT * wrist_drop
    score += DEADLIFT_ELBOW_WEIGHT * elbow_extension
    score += DEADLIFT_BAR_ANKLE_WEIGHT * bar_alignment
    score += DEADLIFT_BAR_RANGE_WEIGHT * bar_vertical
    score += DEADLIFT_ROM_WEIGHT * hip_rom_margin

    if view_label == "front":
        arm_extension = _margin_above(agg.elbow_extended_fraction, DEADLIFT_ELBOW_EXTENDED_FRACTION_MIN)
        bar_low = _margin_above(agg.bar_low_fraction, DEADLIFT_BAR_LOW_FRACTION_MIN)
        score += DEADLIFT_FRONT_ARM_WEIGHT * arm_extension
        score += DEADLIFT_FRONT_BAR_BONUS_WEIGHT * bar_low

    movement_factor = max(
        agg.hip_rom / max(DEADLIFT_HIP_ROM_MIN_DEG, 1e-6),
        agg.bar_range_norm / max(DEADLIFT_BAR_RANGE_MIN_NORM, 1e-6),
    )
    movement_factor = max(0.0, movement_factor)
    score = min(score, DEADLIFT_LOW_MOVEMENT_CAP * movement_factor)

    penalty = 0.0
    penalty += DEADLIFT_KNEE_PENALTY_WEIGHT * max(0.0, knee_penalty)
    penalty += DEADLIFT_BAR_HORIZONTAL_WEIGHT * max(0.0, bar_horizontal_penalty)
    penalty += DEADLIFT_BAR_SHOULDER_PENALTY_WEIGHT * max(0.0, bar_shoulder_penalty)
    if np.isfinite(agg.hip_knee_rom_ratio) and agg.hip_knee_rom_ratio < DEADLIFT_HIP_KNEE_RATIO_MIN:
        ratio_penalty = _margin_below(agg.hip_knee_rom_ratio, DEADLIFT_HIP_KNEE_RATIO_MIN)
        penalty += DEADLIFT_SQUAT_PENALTY_WEIGHT * ratio_penalty

    squat_excursion = _margin_above(agg.knee_rom, DEADLIFT_SQUAT_ROM_THRESHOLD_DEG)
    squat_depth_fraction = _margin_above(agg.knee_deep_fraction, DEADLIFT_KNEE_DEEP_VETO_FRACTION)
    penalty += DEADLIFT_SQUAT_PENALTY_WEIGHT * (squat_excursion + 0.5 * squat_depth_fraction)

    if _bench_gate(agg):
        penalty += DEADLIFT_BENCH_PENALTY_WEIGHT

    squat_cues = [
        _margin_below(np.abs(agg.wrist_shoulder_diff_norm), SQUAT_WRIST_SHOULDER_DIFF_MAX_NORM),
        _margin_below(agg.elbow_bottom, SQUAT_ELBOW_BOTTOM_MAX_DEG),
        _margin_below(agg.torso_tilt_bottom, SQUAT_TORSO_TILT_MAX_DEG),
        _margin_above(agg.bar_high_fraction, SQUAT_BAR_HIGH_FRACTION_MIN),
        _margin_above(agg.elbow_flexed_fraction, SQUAT_ELBOW_FLEXED_FRACTION_MIN),
    ]
    penalty += DEADLIFT_SQUAT_PENALTY_WEIGHT * sum(max(0.0, cue) for cue in squat_cues)

    return score, penalty


def _apply_deadlift_veto(
    agg: AggregateMetrics,
    bench_score: float,
    deadlift_score: float,
    *,
    view_label: str = "unknown",
) -> bool:
    if bench_score > deadlift_score:
        return False

    hinge_evidence = [
        _margin_above(agg.torso_tilt_bottom, DEADLIFT_TORSO_TILT_MIN_DEG),
        _margin_above(agg.wrist_hip_diff_norm, DEADLIFT_WRIST_HIP_DIFF_MIN_NORM),
        _margin_above(agg.elbow_bottom, DEADLIFT_ELBOW_MIN_DEG),
    ]

    movement = max(agg.hip_rom, agg.bar_range_norm)
    squat_rom = np.isfinite(agg.knee_rom) and agg.knee_rom >= DEADLIFT_SQUAT_ROM_THRESHOLD_DEG
    squat_deep = np.isfinite(agg.knee_deep_fraction) and agg.knee_deep_fraction >= DEADLIFT_KNEE_DEEP_VETO_FRACTION
    pelvis_drop = np.isfinite(agg.pelvis_range_norm) and agg.pelvis_range_norm >= 0.08
    hip_ratio_low = np.isfinite(agg.hip_knee_rom_ratio) and agg.hip_knee_rom_ratio < DEADLIFT_HIP_KNEE_RATIO_MIN
    bar_high = np.isfinite(agg.bar_high_fraction) and agg.bar_high_fraction >= SQUAT_BAR_HIGH_FRACTION_MIN
    elbows_flexed = np.isfinite(agg.elbow_flexed_fraction) and agg.elbow_flexed_fraction >= SQUAT_ELBOW_FLEXED_FRACTION_MIN

    squat_like = sum([squat_rom, squat_deep, hip_ratio_low, pelvis_drop, bar_high, elbows_flexed])

    hinge_like = sum(margin > 0.0 for margin in hinge_evidence)
    if view_label != "front" and hinge_like >= 2 and movement >= DEADLIFT_VETO_MOVEMENT_MIN:
        return True

    if view_label == "front":
        if squat_deep or squat_like >= 2:
            if movement >= DEADLIFT_VETO_MOVEMENT_MIN or squat_deep:
                return True

    if squat_like >= 3 and movement >= DEADLIFT_VETO_MOVEMENT_MIN and deadlift_score > 0:
        return True
    return False


def _pick_label(adjusted: Mapping[str, float], agg: AggregateMetrics) -> Tuple[str, float, Mapping[str, float]]:
    scores = np.array([max(0.0, adjusted[label]) for label in LABELS], dtype=float)
    if not np.isfinite(scores).any():
        return "unknown", 0.0, {label: 0.0 for label in LABELS}

    clipped = np.clip(scores, 0.0, None)
    if clipped.sum() == 0:
        return "unknown", 0.0, {label: 0.0 for label in LABELS}

    exp_scores = np.exp(clipped - clipped.max())
    probabilities = exp_scores / exp_scores.sum()
    prob_map = {label: float(probabilities[idx]) for idx, label in enumerate(LABELS)}

    best_idx = int(np.argmax(clipped))
    best_label = LABELS[best_idx]
    best_score = clipped[best_idx]
    sorted_scores = np.sort(clipped)
    second_best = sorted_scores[-2] if clipped.size >= 2 else 0.0

    if best_score < MIN_CONFIDENCE_SCORE:
        return "unknown", 0.0, prob_map

    if best_score - second_best < CLASSIFICATION_MARGIN:
        resolved = _tiebreak(adjusted, agg)
        if resolved == "unknown":
            return "unknown", 0.0, prob_map
        best_label = resolved

    confidence = prob_map.get(best_label, 0.0)
    if confidence < MIN_CONFIDENCE_SCORE:
        return "unknown", 0.0, prob_map
    return best_label, confidence, prob_map


def _tiebreak(adjusted: Mapping[str, float], agg: AggregateMetrics) -> str:
    squat_score = max(0.0, adjusted.get("squat", 0.0))
    deadlift_score = max(0.0, adjusted.get("deadlift", 0.0))
    bench_score = max(0.0, adjusted.get("bench_press", 0.0))

    if _bench_gate(agg) and bench_score >= MIN_CONFIDENCE_SCORE:
        return "bench_press"

    if bench_score > max(squat_score, deadlift_score) and bench_score >= MIN_CONFIDENCE_SCORE:
        return "bench_press"

    if np.isfinite(agg.wrist_hip_diff_norm) and agg.wrist_hip_diff_norm >= DEADLIFT_WRIST_HIP_DIFF_MIN_NORM:
        return "deadlift"
    if np.isfinite(agg.elbow_bottom) and agg.elbow_bottom >= DEADLIFT_ELBOW_MIN_DEG:
        return "deadlift"
    if np.isfinite(agg.torso_tilt_bottom) and agg.torso_tilt_bottom >= DEADLIFT_TORSO_TILT_MIN_DEG:
        return "deadlift"
    if np.isfinite(agg.knee_forward_norm) and abs(agg.knee_forward_norm) >= SQUAT_KNEE_FORWARD_MIN_NORM:
        return "squat"

    squat_posture = _margin_below(agg.torso_tilt_bottom, SQUAT_TORSO_TILT_MAX_DEG)
    deadlift_posture = _margin_above(agg.torso_tilt_bottom, DEADLIFT_TORSO_TILT_MIN_DEG)
    if deadlift_posture > squat_posture:
        return "deadlift"
    if squat_posture > deadlift_posture:
        return "squat"

    return "unknown"


def _margin_above(value: float, threshold: float) -> float:
    if not np.isfinite(value) or not np.isfinite(threshold):
        return 0.0
    return max(0.0, (value - threshold) / max(abs(threshold), 1e-6))


def _margin_below(value: float, threshold: float) -> float:
    if not np.isfinite(value) or not np.isfinite(threshold):
        return 0.0
    return max(0.0, (threshold - value) / max(abs(threshold), 1e-6))


def _bench_gate(agg: AggregateMetrics) -> bool:
    return (
        np.isfinite(agg.torso_tilt_bottom)
        and agg.torso_tilt_bottom >= BENCH_TORSO_HORIZONTAL_DEG
        and np.isfinite(agg.elbow_rom)
        and agg.elbow_rom >= BENCH_ELBOW_ROM_MIN_DEG
        and (not np.isfinite(agg.knee_rom) or agg.knee_rom <= BENCH_KNEE_ROM_MAX_DEG)
        and (not np.isfinite(agg.hip_rom) or agg.hip_rom <= BENCH_HIP_ROM_MAX_DEG)
        and (
            not np.isfinite(agg.bar_range_norm)
            or agg.bar_range_norm >= BENCH_BAR_RANGE_MIN_NORM
            or agg.elbow_rom >= BENCH_ELBOW_ROM_MIN_DEG * BENCH_ELBOW_ROM_GATE_FACTOR
        )
    )

