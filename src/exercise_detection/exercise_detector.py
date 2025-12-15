"""Punto de entrada público para las heurísticas de detección de ejercicio."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from src.core.types import ExerciseType, ViewType

from .classification import classify_features
from .constants import MIN_VALID_FRAMES
from .extraction import extract_features, extract_features_from_frames
from .incremental import IncrementalExerciseFeatureExtractor
from .types import DetectionResult, FeatureSeries, make_detection_result  # noqa: F401

logger = logging.getLogger(__name__)


def detect_exercise(video_path: str, max_frames: int = 300) -> DetectionResult:
    """Detectar ejercicio, vista y confianza para un vídeo en disco."""

    try:
        features = extract_features(video_path, max_frames=max_frames)
    except Exception:  # pragma: no cover - guardia defensiva
        logger.exception("Fallo al extraer características para la detección")
        return DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)

    if features.total_frames == 0 or features.valid_frames < MIN_VALID_FRAMES:
        logger.info(
            "Detección inconclusa: fotogramas válidos %d de %d",
            features.valid_frames,
            features.total_frames,
        )
        return DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)

    try:
        result = classify_features(features, return_metadata=True)
        if isinstance(result, tuple) and len(result) == 4:
            label, view, confidence, metadata = result
        else:  # pragma: no cover - defensive fallback
            label, view, confidence = result  # type: ignore[misc]
            metadata = {}
    except Exception:  # pragma: no cover - guardia defensiva
        logger.exception("Fallo al clasificar las características extraídas")
        return DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)

    logger.info(
        "Ejercicio detectado: label=%s view=%s confidence=%.2f (frames=%d/%d)",
        label,
        view,
        confidence,
        features.valid_frames,
        features.total_frames,
    )
    return make_detection_result(
        label,
        view,
        confidence,
        debug=metadata,
        side=metadata.get("view_side") if isinstance(metadata, dict) else None,
        view_stats=metadata.get("view_stats") if isinstance(metadata, dict) else None,
    )


def detect_exercise_from_frames(
    frames: Iterable[np.ndarray], *, fps: float, max_frames: int = 300
) -> DetectionResult:
    """Detectar el ejercicio a partir de un iterable de fotogramas ya precalculados."""

    try:
        features = extract_features_from_frames(frames, fps=fps, max_frames=max_frames)
    except Exception:  # pragma: no cover - guardia defensiva
        logger.exception("Fallo al extraer características para la detección (streaming)")
        return DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)

    if features.total_frames == 0 or features.valid_frames < MIN_VALID_FRAMES:
        logger.info(
            "Detección inconclusa (streaming): fotogramas válidos %d de %d",
            features.valid_frames,
            features.total_frames,
        )
        return DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)

    try:
        result = classify_features(features, return_metadata=True)
        if isinstance(result, tuple) and len(result) == 4:
            label, view, confidence, metadata = result
        else:  # pragma: no cover - defensive fallback
            label, view, confidence = result  # type: ignore[misc]
            metadata = {}
    except Exception:  # pragma: no cover - guardia defensiva
        logger.exception("Fallo al clasificar las características en streaming")
        return DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)

    logger.info(
        "Ejercicio detectado (streaming): label=%s view=%s confidence=%.2f (frames=%d/%d)",
        label,
        view,
        confidence,
        features.valid_frames,
        features.total_frames,
    )
    return make_detection_result(
        label,
        view,
        confidence,
        debug=metadata,
        side=metadata.get("view_side") if isinstance(metadata, dict) else None,
        view_stats=metadata.get("view_stats") if isinstance(metadata, dict) else None,
    )


__all__ = [
    "DetectionResult",
    "FeatureSeries",
    "IncrementalExerciseFeatureExtractor",
    "classify_features",
    "detect_exercise",
    "detect_exercise_from_frames",
    "extract_features",
    "extract_features_from_frames",
    "make_detection_result",
]
