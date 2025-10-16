"""Detection utilities for identifying exercises and camera views."""

from .exercise_detector import FeatureSeries, classify_features, detect_exercise, extract_features

__all__ = [
    "FeatureSeries",
    "classify_features",
    "detect_exercise",
    "extract_features",
]
