"""Tipos y utilidades comunes para etiquetar ejercicios y vistas de cámara.

El objetivo del módulo es normalizar las etiquetas que circulan entre modelos,
configuraciones y la interfaz, evitando condicionales repetidos. Así se
documenta cómo se desambiguan etiquetas libres provenientes de archivos o
entradas de usuario."""

from __future__ import annotations

from enum import Enum
from typing import Union


class ExerciseType(str, Enum):
    """Catálogo normalizado de ejercicios que reconoce la aplicación."""

    UNKNOWN = "unknown"
    SQUAT = "squat"
    BENCH_PRESS = "bench_press"
    DEADLIFT = "deadlift"


class ViewType(str, Enum):
    """Perspectivas de cámara admitidas por los detectores."""

    UNKNOWN = "unknown"
    FRONT = "front"
    SIDE = "side"


_EXERCISE_ALIAS_MAP = {
    # Unificamos alias comunes para que "bench" y "benchpress" signifiquen lo mismo.
    "bench": ExerciseType.BENCH_PRESS.value,
    "benchpress": ExerciseType.BENCH_PRESS.value,
}


def _normalize_label(value: str) -> str:
    """Limpiar una etiqueta textual para compararla de forma consistente."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def as_exercise(value: Union[str, "ExerciseType", None]) -> "ExerciseType":
    """Convertir una entrada libre en un ``ExerciseType`` reconocido.

    Se aceptan cadenas provenientes de archivos de configuración, argumentos en
    CLI o incluso instancias del propio ``ExerciseType``. Cualquier valor no
    reconocido se degrada a ``UNKNOWN`` para evitar excepciones aguas abajo."""

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
    """Normalizar una etiqueta de vista de cámara hacia ``ViewType``.

    Esta función se usa al importar configuraciones o resultados donde la vista
    puede expresarse como texto libre; cualquier error se traduce en ``UNKNOWN``
    para que el resto del sistema decida cómo proceder."""

    if isinstance(value, ViewType):
        return value
    if not value:
        return ViewType.UNKNOWN
    try:
        return ViewType(str(value).lower())
    except ValueError:
        return ViewType.UNKNOWN
