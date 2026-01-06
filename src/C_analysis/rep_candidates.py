"""Exercise-aware repetition detection and phase extraction.

This module exposes a canonical representation for detected repetitions
(`RepCandidate`) alongside a simple hysteresis-based finite state machine
that is aware of the semantics of each supported lift (squat, deadlift,
bench press).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional

import numpy as np


class Zone(str, Enum):
    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"


class RejectionReason(str, Enum):
    NONE = "NONE"
    LOW_THRESH = "LOW_THRESH"
    HIGH_THRESH = "HIGH_THRESH"
    INCOMPLETE = "INCOMPLETE"
    NOISY = "NOISY"


@dataclass
class ExerciseRepSpec:
    """Defines the expected semantics for a lift."""

    name: str
    start_zone: Zone
    completion_zone: Zone
    phases: tuple[str, str]
    require_low: bool = True
    require_high: bool = True


EXERCISE_SPECS = {
    "squat": ExerciseRepSpec(
        name="squat",
        start_zone=Zone.HIGH,
        completion_zone=Zone.HIGH,
        phases=("Down", "Up"),
        require_low=True,
        require_high=True,
    ),
    "bench": ExerciseRepSpec(
        name="bench",
        start_zone=Zone.HIGH,
        completion_zone=Zone.HIGH,
        phases=("Down", "Up"),
        require_low=True,
        require_high=True,
    ),
    "bench_press": ExerciseRepSpec(
        name="bench_press",
        start_zone=Zone.HIGH,
        completion_zone=Zone.HIGH,
        phases=("Down", "Up"),
        require_low=True,
        require_high=True,
    ),
    "deadlift": ExerciseRepSpec(
        name="deadlift",
        start_zone=Zone.LOW,
        completion_zone=Zone.LOW,
        phases=("Up", "Down"),
        require_low=True,
        require_high=True,
    ),
}


@dataclass
class RepCandidate:
    rep_index: int
    start_frame: int
    turning_frame: Optional[int]
    end_frame: Optional[int]
    min_angle: float | None
    max_angle: float | None
    down_start: Optional[int] = None
    down_end: Optional[int] = None
    up_start: Optional[int] = None
    up_end: Optional[int] = None
    passed_low: bool = False
    passed_high: bool = False
    accepted: bool = False
    rejection_reason: RejectionReason = RejectionReason.NONE

    def as_dict(self) -> dict:
        return {
            "rep_index": self.rep_index,
            "start_frame": self.start_frame,
            "turning_frame": self.turning_frame,
            "end_frame": self.end_frame,
            "min_angle": self.min_angle,
            "max_angle": self.max_angle,
            "down_start": self.down_start,
            "down_end": self.down_end,
            "up_start": self.up_start,
            "up_end": self.up_end,
            "passed_low": self.passed_low,
            "passed_high": self.passed_high,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason.value,
        }


def _majority_zone(zones: Iterable[Zone]) -> Zone:
    counts = {Zone.LOW: 0, Zone.MID: 0, Zone.HIGH: 0}
    for z in zones:
        counts[z] += 1
    return max(counts, key=counts.get)


def _zone_from_angle(angle: float, low_thresh: float, high_thresh: float) -> Zone:
    if not np.isfinite(angle):
        return Zone.MID
    if angle <= low_thresh:
        return Zone.LOW
    if angle >= high_thresh:
        return Zone.HIGH
    return Zone.MID


def detect_rep_candidates(
    angles: np.ndarray,
    *,
    low_thresh: float,
    high_thresh: float,
    exercise_key: str,
    frame_offset: int = 0,
    enforce_low: bool = True,
    enforce_high: bool = True,
) -> List[RepCandidate]:
    """Detect repetitions using hysteresis zones and an exercise-aware FSM."""

    spec = EXERCISE_SPECS.get(exercise_key.lower(), EXERCISE_SPECS["squat"])
    if angles.size == 0:
        return []

    finite_mask = np.isfinite(angles)
    if not finite_mask.any():
        return []

    zones = [_zone_from_angle(a, low_thresh, high_thresh) for a in angles]
    initial_zone = _majority_zone(zones[: max(3, min(10, len(zones)))])
    last_zone = initial_zone
    rep_start_idx: Optional[int] = None
    turning_idx: Optional[int] = None
    min_angle = np.inf
    max_angle = -np.inf
    candidates: list[RepCandidate] = []
    rep_index = 0

    def _flush_incomplete():
        nonlocal rep_index, rep_start_idx, turning_idx, min_angle, max_angle
        if rep_start_idx is None:
            return
        candidate = RepCandidate(
            rep_index=rep_index,
            start_frame=rep_start_idx + frame_offset,
            turning_frame=turning_idx + frame_offset if turning_idx is not None else None,
            end_frame=None,
            min_angle=min_angle if np.isfinite(min_angle) else None,
            max_angle=max_angle if np.isfinite(max_angle) else None,
            rejection_reason=RejectionReason.INCOMPLETE,
            accepted=False,
        )
        candidate.passed_low = bool(candidate.min_angle is not None and candidate.min_angle <= low_thresh)
        candidate.passed_high = bool(
            candidate.max_angle is not None and candidate.max_angle >= high_thresh
        )
        candidates.append(candidate)
        rep_index += 1
        rep_start_idx = None
        turning_idx = None
        min_angle = np.inf
        max_angle = -np.inf

    apply_low = spec.require_low and enforce_low
    apply_high = spec.require_high and enforce_high

    for idx, (angle, zone) in enumerate(zip(angles, zones)):
        if np.isfinite(angle):
            min_angle = min(min_angle, angle)
            max_angle = max(max_angle, angle)

        if rep_start_idx is None:
            if last_zone == spec.start_zone and zone != spec.start_zone:
                rep_start_idx = idx
                turning_idx = None
                min_angle = angle if np.isfinite(angle) else min_angle
                max_angle = angle if np.isfinite(angle) else max_angle
        else:
            if turning_idx is None and zone != spec.start_zone:
                turning_idx = idx
            if zone == spec.completion_zone and turning_idx is not None:
                end_idx = idx
                candidate = RepCandidate(
                    rep_index=rep_index,
                    start_frame=rep_start_idx + frame_offset,
                    turning_frame=turning_idx + frame_offset,
                    end_frame=end_idx + frame_offset,
                    min_angle=min_angle if np.isfinite(min_angle) else None,
                    max_angle=max_angle if np.isfinite(max_angle) else None,
                )
                candidate.passed_low = bool(candidate.min_angle is not None and candidate.min_angle <= low_thresh)
                candidate.passed_high = bool(
                    candidate.max_angle is not None and candidate.max_angle >= high_thresh
                )
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

                candidate.accepted = True
                candidate.rejection_reason = RejectionReason.NONE
                if apply_low and not candidate.passed_low:
                    candidate.accepted = False
                    candidate.rejection_reason = RejectionReason.LOW_THRESH
                if apply_high and not candidate.passed_high:
                    candidate.accepted = False
                    if candidate.rejection_reason == RejectionReason.NONE:
                        candidate.rejection_reason = RejectionReason.HIGH_THRESH
                candidates.append(candidate)
                rep_index += 1
                rep_start_idx = None
                turning_idx = None
                min_angle = np.inf
                max_angle = -np.inf

        last_zone = zone

    _flush_incomplete()

    return candidates

