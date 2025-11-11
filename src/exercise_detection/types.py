"""Modelos de datos compartidos por el pipeline de detección de ejercicio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.core.types import ExerciseType, ViewType, as_exercise, as_view


@dataclass(frozen=True)
class DetectionResult:
    """Salida normalizada que devuelve el pipeline de detección."""

    label: ExerciseType
    view: ViewType
    confidence: float


def make_detection_result(label: str, view: str, confidence: float) -> "DetectionResult":
    """Convertir identificadores de texto legados en ``DetectionResult`` con enums."""

    return DetectionResult(as_exercise(label), as_view(view), float(confidence))


@dataclass
class FeatureSeries:
    """Contenedor de series temporales extraídas y su metainformación."""

    data: Dict[str, np.ndarray]
    sampling_rate: float
    valid_frames: int
    total_frames: int
