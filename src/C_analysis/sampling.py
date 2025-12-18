"""Planificación de muestreo de vídeo.

Aquí se definen las heurísticas que deciden cada cuántos fotogramas extraer
cuadros del vídeo (``sample_rate``) y qué FPS efectivo usar durante la
inferencia, según los metadatos del archivo y la configuración elegida.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2

from src import config
from src.A_preprocessing.video_metadata import VideoInfo, read_video_file_info
from src.exercise_detection.exercise_detector import DetectionResult
from src.core.types import ExerciseType, ViewType, as_exercise, as_view

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingPlan:
    """Parámetros derivados para decidir cada cuántos cuadros muestrear."""

    fps_base: float
    sample_rate: int
    fps_effective: float
    warnings: list[str]


def compute_sample_rate(fps: float, cfg: config.Config) -> int:
    """Calcular el *stride* inicial a partir del FPS original y la configuración."""

    # Si el usuario fija un muestreo manual, se respeta por encima de todo.
    if cfg.video.manual_sample_rate and cfg.video.manual_sample_rate > 0:
        return max(1, int(cfg.video.manual_sample_rate))
    # De lo contrario, se calcula el stride necesario para aproximar el FPS objetivo.
    if fps > 0 and cfg.video.target_fps and cfg.video.target_fps > 0:
        return max(1, int(round(fps / cfg.video.target_fps)))
    return 1


def normalize_detection(
    detection: Union[DetectionResult, Tuple[str, str, float]]
) -> Tuple[ExerciseType, ViewType, float]:
    """Normalizar tuplas de detección a enumeraciones fuertes."""

    if isinstance(detection, DetectionResult):
        label, view, confidence = detection.label, detection.view, float(detection.confidence)
    else:
        label, view, confidence = detection
    return as_exercise(label), as_view(view), float(confidence)


def make_sampling_plan(
    *,
    fps_metadata: float,
    fps_from_reader: float,
    prefer_reader_fps: bool,
    initial_sample_rate: int,
    cfg: config.Config,
    fps_warning: Optional[str],
) -> SamplingPlan:
    """Construir un plan consistente con las heurísticas heredadas de la aplicación."""

    warnings: list[str] = []

    fps_base = float(fps_metadata)
    # Preferimos el FPS del lector cuando los metadatos son poco fiables
    # (por ejemplo, tras reencodificaciones) o cuando las heurísticas lo indican.
    if (fps_base <= 0.0 or prefer_reader_fps) and fps_from_reader > 0.0:
        fps_base = float(fps_from_reader)
    if fps_base <= 0.0 and fps_from_reader <= 0.0:
        warnings.append(
            "Unable to determine a valid FPS from metadata or reader. Falling back to 1 FPS."
        )
        fps_base = 1.0

    sample_rate = int(initial_sample_rate)
    # Si el stride inicial no limita, recalculamos para acercarnos al FPS objetivo.
    if initial_sample_rate == 1 and cfg.video.target_fps and cfg.video.target_fps > 0:
        recomputed = compute_sample_rate(fps_base, cfg)
        if recomputed > 1:
            sample_rate = recomputed

    fps_effective = fps_base / sample_rate if sample_rate > 0 else fps_base

    # Propagamos advertencias para mostrarlas en la UI y bitácoras.
    if fps_warning:
        message = f"{fps_warning} Using FPS value: {fps_base:.2f}."
        logger.warning(message)
        warnings.append(message)

    return SamplingPlan(
        fps_base=float(fps_base),
        sample_rate=int(sample_rate),
        fps_effective=float(fps_effective),
        warnings=warnings,
    )


def open_video_cap(video_path: str) -> cv2.VideoCapture:
    """Abrir un `VideoCapture` y validar que el manejador sea válido."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        from .errors import VideoOpenError

        raise VideoOpenError(f"Could not open the video: {video_path}")
    return cap


def read_info_and_initial_sampling(
    cap: cv2.VideoCapture,
    video_path: str,
) -> tuple[VideoInfo, float, Optional[str], bool]:
    """Leer metadatos y derivar las heurísticas usadas para planificar el muestreo."""

    info = read_video_file_info(video_path, cap=cap)

    fps_original = float(info.fps or 0.0)
    fps_warning: Optional[str] = None
    prefer_reader_fps = False

    if info.fps_source == "estimated":
        fps_warning = f"Invalid metadata FPS. Estimated from video duration ({fps_original:.2f} fps)."
    elif info.fps_source == "reader":
        fps_warning = (
            "Invalid metadata FPS and unreliable duration. Falling back to the reader-reported FPS."
        )
        prefer_reader_fps = True

    return info, fps_original, fps_warning, prefer_reader_fps
