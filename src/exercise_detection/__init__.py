"""Utilidades de detección para identificar ejercicios y vistas de cámara."""

from .exercise_detector import (
    DetectionResult,
    FeatureSeries,
    classify_features,
    detect_exercise,
    detect_exercise_with_diagnostics,
    detect_exercise_from_frames,
    extract_features,
    extract_features_from_frames,
    make_detection_result,
)
from .incremental import IncrementalExerciseFeatureExtractor

__all__ = [
    "DetectionResult",
    "FeatureSeries",
    "classify_features",
    "detect_exercise",
    "detect_exercise_with_diagnostics",
    "detect_exercise_from_frames",
    "extract_features",
    "extract_features_from_frames",
    "make_detection_result",
    "IncrementalExerciseFeatureExtractor",
]
