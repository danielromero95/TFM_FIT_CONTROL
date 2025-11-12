"""Validación de parámetros para los modos de muestreo de fotogramas."""

from __future__ import annotations

from typing import Literal, Optional


def _validate_sampling_args(
    sampling: Literal["auto", "time", "index"],
    target_fps: Optional[float],
    every_n: Optional[int],
    start_time: Optional[float],
    end_time: Optional[float],
) -> tuple[Literal["auto", "time", "index"], Optional[float], Optional[int]]:
    """Normaliza y valida los argumentos de muestreo devolviendo valores seguros."""
    if sampling not in ("auto", "time", "index"):
        raise ValueError('sampling debe ser "auto", "time" o "index"')

    normalized_target_fps = float(target_fps) if target_fps else None
    if sampling in ("auto", "time") and not (normalized_target_fps and normalized_target_fps > 0):
        if sampling == "time":
            raise ValueError("target_fps debe ser > 0 en el muestreo basado en tiempo")

    normalized_every_n = int(every_n) if every_n is not None else None
    if sampling == "index":
        if normalized_every_n is None or normalized_every_n <= 0:
            raise ValueError("every_n debe ser un entero positivo en el muestreo por índice")

    if start_time is not None and end_time is not None and end_time <= start_time:
        raise ValueError("end_time debe ser mayor que start_time")

    return sampling, normalized_target_fps, normalized_every_n
