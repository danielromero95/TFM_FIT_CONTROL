"""Shared dataclasses for the exercise recognition pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

from src.core.types import ExerciseType, ViewType, as_exercise, as_view


@dataclass(frozen=True)
class DetectionResult:
    """Normalized output returned by the detection pipeline."""

    label: ExerciseType
    view: ViewType
    confidence: float


def make_detection_result(label: str, view: str, confidence: float) -> "DetectionResult":
    """Convert legacy text identifiers into ``DetectionResult`` with enums."""

    return DetectionResult(as_exercise(label), as_view(view), float(confidence))


@dataclass
class FeatureSeries:
    """Container for Pose-derived time series and basic metadata."""

    data: Dict[str, np.ndarray]
    sampling_rate: float
    valid_frames: int
    total_frames: int


@dataclass(frozen=True)
class RepSlice:
    """Inclusive-exclusive slice describing the frame span of a repetition."""

    start: int
    end: int

    def clamp(self, total: int) -> "RepSlice":
        """Clamp the slice to ``total`` frames to avoid overflow issues."""

        start = max(0, min(self.start, total))
        end = max(start + 1, min(self.end, total))
        return RepSlice(start=start, end=end)


@dataclass(frozen=True)
class RepMetrics:
    """Summary statistics describing a single repetition."""

    slice: RepSlice
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
    bottom_frame_count: int


@dataclass(frozen=True)
class AggregateMetrics:
    """Robust aggregation across repetitions and clip-wide measurements."""

    per_rep: Sequence[RepMetrics] = field(default_factory=tuple)
    knee_min: float = np.nan
    hip_min: float = np.nan
    elbow_bottom: float = np.nan
    torso_tilt_bottom: float = np.nan
    wrist_shoulder_diff_norm: float = np.nan
    wrist_hip_diff_norm: float = np.nan
    knee_forward_norm: float = np.nan
    tibia_angle_deg: float = np.nan
    bar_ankle_diff_norm: float = np.nan
    knee_rom: float = np.nan
    hip_rom: float = np.nan
    elbow_rom: float = np.nan
    bar_range_norm: float = np.nan
    hip_range_norm: float = np.nan
    bar_vertical_range_norm: float = np.nan
    bar_horizontal_std_norm: float = np.nan
    duration_s: float = np.nan
    rep_count: int = 0


@dataclass(frozen=True)
class ClassificationScores:
    """Intermediate classification artefacts useful for debugging/tests."""

    raw: Mapping[str, float]
    adjusted: Mapping[str, float]
    penalties: Mapping[str, float]
    deadlift_veto: bool


@dataclass(frozen=True)
class ViewResult:
    """Scores and votes gathered while classifying the camera view."""

    label: str
    scores: Mapping[str, float]
    votes: Mapping[str, int]

