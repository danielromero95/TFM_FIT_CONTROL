"""Dataclasses compartidas por la *pipeline* de reconocimiento de ejercicios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

from src.core.types import ExerciseType, ViewType, as_exercise, as_view


@dataclass(frozen=True)
class DetectionResult:
    """Resultado normalizado que devuelve la pipeline de detección."""

    label: ExerciseType
    view: ViewType
    confidence: float
    side: str | None = None
    diagnostics: Mapping[str, object] | None = None


def make_detection_result(label: str, view: str, confidence: float) -> "DetectionResult":
    """Convierte identificadores de texto legados en ``DetectionResult`` usando enumeraciones."""

    return DetectionResult(as_exercise(label), as_view(view), float(confidence))


@dataclass
class FeatureSeries:
    """Contenedor de series temporales derivadas de la pose y metadatos básicos."""

    data: Dict[str, np.ndarray]
    sampling_rate: float
    valid_frames: int
    total_frames: int
    view_reliability: Dict[str, np.ndarray | float | int] = field(default_factory=dict)


@dataclass(frozen=True)
class RepSlice:
    """Segmento inclusivo-exclusivo que describe el rango de fotogramas de una repetición."""

    start: int
    end: int

    def clamp(self, total: int) -> "RepSlice":
        """Ajusta el segmento al rango ``total`` de fotogramas para evitar desbordes."""

        start = max(0, min(self.start, total))
        end = max(start + 1, min(self.end, total))
        return RepSlice(start=start, end=end)


@dataclass(frozen=True)
class RepMetrics:
    """Estadísticas resumidas que describen una repetición individual."""

    slice: RepSlice
    knee_min: float
    hip_min: float
    elbow_bottom: float
    torso_tilt_bottom: float
    wrist_shoulder_diff_norm: float
    wrist_hip_diff_norm: float
    bar_above_hip_norm: float
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
    """Agregación robusta entre repeticiones y métricas a nivel de clip."""

    per_rep: Sequence[RepMetrics] = field(default_factory=tuple)
    knee_min: float = np.nan
    hip_min: float = np.nan
    elbow_bottom: float = np.nan
    torso_tilt_bottom: float = np.nan
    wrist_shoulder_diff_norm: float = np.nan
    wrist_hip_diff_norm: float = np.nan
    bar_above_hip_norm: float = np.nan
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
    arms_high_fraction: float = np.nan
    bar_near_shoulders_fraction: float = np.nan
    duration_s: float = np.nan
    rep_count: int = 0


@dataclass(frozen=True)
class ClassificationScores:
    """Artefactos intermedios de clasificación útiles para depurar o probar."""

    raw: Mapping[str, float]
    adjusted: Mapping[str, float]
    penalties: Mapping[str, float]
    deadlift_veto: bool


@dataclass(frozen=True)
class ViewResult:
    """Puntuaciones y votos recopilados durante la clasificación de la vista de cámara."""

    label: str
    scores: Mapping[str, float]
    votes: Mapping[str, int]
    confidence: float = 0.0
    lateral_score: float = float("nan")
    stats: Mapping[str, float] = field(default_factory=dict)
    side: str | None = None
