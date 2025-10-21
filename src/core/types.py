from __future__ import annotations

from enum import Enum
from typing import Union


class ExerciseType(str, Enum):
    UNKNOWN = "unknown"
    SQUAT = "squat"
    BENCH = "bench"
    DEADLIFT = "deadlift"


class ViewType(str, Enum):
    UNKNOWN = "unknown"
    FRONT = "front"
    SIDE = "side"


def as_exercise(value: Union[str, "ExerciseType", None]) -> "ExerciseType":
    if isinstance(value, ExerciseType):
        return value
    if not value:
        return ExerciseType.UNKNOWN
    try:
        return ExerciseType(str(value).lower())
    except ValueError:
        return ExerciseType.UNKNOWN


def as_view(value: Union[str, "ViewType", None]) -> "ViewType":
    if isinstance(value, ViewType):
        return value
    if not value:
        return ViewType.UNKNOWN
    try:
        return ViewType(str(value).lower())
    except ValueError:
        return ViewType.UNKNOWN
