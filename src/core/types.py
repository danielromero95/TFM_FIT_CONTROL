from __future__ import annotations

from enum import Enum
from typing import Union


class ExerciseType(str, Enum):
    UNKNOWN = "unknown"
    SQUAT = "squat"
    BENCH_PRESS = "bench_press"
    DEADLIFT = "deadlift"


class ViewType(str, Enum):
    UNKNOWN = "unknown"
    FRONT = "front"
    SIDE = "side"


_EXERCISE_ALIAS_MAP = {
    "bench": ExerciseType.BENCH_PRESS.value,
    "benchpress": ExerciseType.BENCH_PRESS.value,
}


def _normalize_label(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def as_exercise(value: Union[str, "ExerciseType", None]) -> "ExerciseType":
    if isinstance(value, ExerciseType):
        return value
    if not value:
        return ExerciseType.UNKNOWN
    normalized = _normalize_label(str(value))
    mapped = _EXERCISE_ALIAS_MAP.get(normalized, normalized)
    try:
        return ExerciseType(mapped)
    except ValueError:
        return ExerciseType.UNKNOWN


EXERCISE_HUMAN_LABEL = {
    ExerciseType.UNKNOWN: "Unknown",
    ExerciseType.SQUAT: "Squat",
    ExerciseType.BENCH_PRESS: "Bench Press",
    ExerciseType.DEADLIFT: "Deadlift",
}


def as_view(value: Union[str, "ViewType", None]) -> "ViewType":
    if isinstance(value, ViewType):
        return value
    if not value:
        return ViewType.UNKNOWN
    try:
        return ViewType(str(value).lower())
    except ValueError:
        return ViewType.UNKNOWN
