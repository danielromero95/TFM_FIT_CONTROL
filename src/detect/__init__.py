"""Detection utilities for identifying exercises and camera views."""

from .exercise_detector import (
    DetectionResult,
    FeatureSeries,
    classify_features,
    detect_exercise,
    extract_features,
)

__all__ = [
    "DetectionResult",
    "FeatureSeries",
    "classify_features",
    "detect_exercise",
    "extract_features",
]
