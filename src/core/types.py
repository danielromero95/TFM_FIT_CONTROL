"""Definiciones de tipos centrales utilizados en todo el proyecto."""

from __future__ import annotations

from enum import Enum
from typing import Union


class ExerciseType(str, Enum):
    """Catálogo de ejercicios soportados por el sistema."""
    UNKNOWN = "unknown"
    SQUAT = "squat"
    BENCH_PRESS = "bench_press"
    DEADLIFT = "deadlift"


class ViewType(str, Enum):
    """Perspectivas de cámara esperadas al analizar un levantamiento."""
    UNKNOWN = "unknown"
    FRONT = "front"
    SIDE = "side"


_EXERCISE_ALIAS_MAP = {
    "bench": ExerciseType.BENCH_PRESS.value,
    "benchpress": ExerciseType.BENCH_PRESS.value,
}

# Diccionario para traducir alias comunes a la etiqueta oficial del ejercicio.


def _normalize_label(value: str) -> str:
    """Limpia y homogeneiza una etiqueta para facilitar su comparación."""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def as_exercise(value: Union[str, "ExerciseType", None]) -> "ExerciseType":
    """Convierte valores variados en una instancia válida de ``ExerciseType``."""
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

# Etiquetas legibles por humanos utilizadas en la interfaz de usuario.


def as_view(value: Union[str, "ViewType", None]) -> "ViewType":
    """Normaliza un valor arbitrario a la enumeración ``ViewType``."""
    if isinstance(value, ViewType):
        return value
    if not value:
        return ViewType.UNKNOWN
    try:
        return ViewType(str(value).lower())
    except ValueError:
        return ViewType.UNKNOWN
