"""Punto de entrada público para las heurísticas de detección de ejercicio."""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np

from src.core.types import as_exercise, as_view
from .classification import classify_features, classify_features_with_diagnostics
from .constants import MIN_VALID_FRAMES
from .extraction import extract_features, extract_features_from_frames
from .incremental import IncrementalExerciseFeatureExtractor
from .types import DetectionResult, FeatureSeries, make_detection_result  # noqa: F401

logger = logging.getLogger(__name__)


def detect_exercise(video_path: str, max_frames: int = 300) -> Tuple[str, str, float]:
    """Detectar ejercicio, vista y confianza para un vídeo en disco."""

    try:
        features = extract_features(video_path, max_frames=max_frames)
    except Exception:  # pragma: no cover - guardia defensiva
        logger.exception("Fallo al extraer características para la detección")
        return "unknown", "unknown", 0.0

    if features.total_frames == 0 or features.valid_frames < MIN_VALID_FRAMES:
        logger.info(
            "Detección inconclusa: fotogramas válidos %d de %d",
            features.valid_frames,
            features.total_frames,
        )
        return "unknown", "unknown", 0.0

    try:
        label, view, confidence, _diagnostics = classify_features_with_diagnostics(features)
    except Exception:  # pragma: no cover - guardia defensiva
        logger.exception("Fallo al clasificar las características extraídas")
        return "unknown", "unknown", 0.0

    logger.info(
        "Ejercicio detectado: label=%s view=%s confidence=%.2f (frames=%d/%d)",
        label,
        view,
        confidence,
        features.valid_frames,
        features.total_frames,
    )
    return label, view, confidence


def detect_exercise_from_frames(
    frames: Iterable[np.ndarray], *, fps: float, max_frames: int = 300
) -> Tuple[str, str, float]:
    """Detectar el ejercicio a partir de un iterable de fotogramas ya precalculados."""

    try:
        features = extract_features_from_frames(frames, fps=fps, max_frames=max_frames)
    except Exception:  # pragma: no cover - guardia defensiva
        logger.exception("Fallo al extraer características para la detección (streaming)")
        return "unknown", "unknown", 0.0

    if features.total_frames == 0 or features.valid_frames < MIN_VALID_FRAMES:
        logger.info(
            "Detección inconclusa (streaming): fotogramas válidos %d de %d",
            features.valid_frames,
            features.total_frames,
        )
        return "unknown", "unknown", 0.0

    try:
        label, view, confidence, _diagnostics = classify_features_with_diagnostics(features)
    except Exception:  # pragma: no cover - guardia defensiva
        logger.exception("Fallo al clasificar las características en streaming")
        return "unknown", "unknown", 0.0

    logger.info(
        "Ejercicio detectado (streaming): label=%s view=%s confidence=%.2f (frames=%d/%d)",
        label,
        view,
        confidence,
        features.valid_frames,
        features.total_frames,
    )
    return label, view, confidence


def detect_exercise_with_diagnostics(video_path: str, max_frames: int = 300) -> DetectionResult:
    """Detectar ejercicio y devolver un ``DetectionResult`` con diagnósticos."""

    try:
        features = extract_features(video_path, max_frames=max_frames)
    except Exception:  # pragma: no cover - guardia defensiva
        logger.exception("Fallo al extraer características para la detección")
        return DetectionResult(as_exercise("unknown"), as_view("unknown"), 0.0, diagnostics=None)

    if features.total_frames == 0 or features.valid_frames < MIN_VALID_FRAMES:
        logger.info(
            "Detección inconclusa: fotogramas válidos %d de %d",
            features.valid_frames,
            features.total_frames,
        )
        return DetectionResult(as_exercise("unknown"), as_view("unknown"), 0.0, diagnostics=None)

    try:
        label, view, confidence, diagnostics = classify_features_with_diagnostics(features)
    except Exception:  # pragma: no cover - guardia defensiva
        logger.exception("Fallo al clasificar las características extraídas")
        return DetectionResult(as_exercise("unknown"), as_view("unknown"), 0.0, diagnostics=None)

    logger.info(
        "Ejercicio detectado: label=%s view=%s confidence=%.2f (frames=%d/%d)",
        label,
        view,
        confidence,
        features.valid_frames,
        features.total_frames,
    )
    return DetectionResult(
        label=as_exercise(label),
        view=as_view(view),
        confidence=float(confidence),
        diagnostics=diagnostics,
    )


__all__ = [
    "DetectionResult",
    "FeatureSeries",
    "IncrementalExerciseFeatureExtractor",
    "classify_features",
    "detect_exercise",
    "detect_exercise_with_diagnostics",
    "detect_exercise_from_frames",
    "extract_features",
    "extract_features_from_frames",
    "make_detection_result",
]
