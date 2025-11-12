"""Dataclasses y utilidades base para la extracción de fotogramas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass(slots=True)
class FrameInfo:
    """Metadatos básicos asociados a cada fotograma extraído del vídeo."""

    index: int
    timestamp_sec: float
    array: np.ndarray
    width: int
    height: int


_ROTATE_MAP = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _apply_rotation(frame: np.ndarray, rotation_deg: Optional[int]) -> np.ndarray:
    """Gira ``frame`` si la rotación solicitada coincide con un giro conocido."""

    rot = _ROTATE_MAP.get(int(rotation_deg or 0))
    return cv2.rotate(frame, rot) if rot is not None else frame


@dataclass(slots=True)
class _FrameProcessor:
    """Transforma fotogramas aplicando rotación, *resize* y paso a escala de grises."""

    rotate_deg: int
    resize_to: Optional[tuple[int, int]]
    to_gray: bool

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Aplica las transformaciones configuradas al fotograma recibido."""

        if self.rotate_deg:
            frame = _apply_rotation(frame, self.rotate_deg)
        if self.resize_to:
            target_width, target_height = self.resize_to
            height, width = frame.shape[:2]
            if width != target_width or height != target_height:
                interpolation = (
                    cv2.INTER_AREA
                    if width > target_width or height > target_height
                    else cv2.INTER_LINEAR
                )
                frame = cv2.resize(frame, self.resize_to, interpolation=interpolation)
        if self.to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame


@dataclass(slots=True)
class _ProgressHandler:
    """Gestiona las notificaciones de progreso sin repetir porcentajes."""

    callback: Callable[[int], None] | None
    frame_count: int
    last_progress: int = -1

    def report(self, index: int) -> None:
        """Publica el avance aproximado usando el índice de fotograma actual."""

        if not self.callback or self.frame_count <= 0:
            return
        percent = int((index / self.frame_count) * 100)
        if percent > self.last_progress:
            self.callback(percent)
            self.last_progress = percent

    def finalize(self) -> None:
        """Envía el 100 % si el proceso termina sin alcanzar el umbral final."""

        if self.callback and self.frame_count > 0 and self.last_progress < 100:
            self.callback(100)


@dataclass(slots=True)
class _IteratorContext:
    """Estado mutable compartido por los iteradores de extracción de fotogramas."""

    cap: cv2.VideoCapture
    fps_base: float
    frame_count: int
    start_time: float
    end_time: Optional[float]
    max_frames: Optional[int]
    processor: _FrameProcessor
    progress: _ProgressHandler
    read_idx: int
    produced: int = 0

    def effective_timestamp(self, ts_ms: float, idx: int) -> float:
        """Devuelve el instante en segundos usando ``ts_ms`` o el índice + FPS base."""

        if ts_ms > 0:
            return ts_ms / 1000.0
        if self.fps_base > 0:
            return idx / self.fps_base
        return 0.0

    def limit_reached(self) -> bool:
        """Indica si se alcanzó el máximo de fotogramas a producir."""

        return bool(self.max_frames and self.produced >= self.max_frames)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Aplica el procesador configurado al fotograma recibido."""

        return self.processor(frame)

    def report_progress(self, index: int) -> None:
        """Actualiza el progreso notificando el índice de lectura actual."""

        self.progress.report(index)

    def increment_produced(self) -> None:
        """Incrementa el contador interno de fotogramas entregados."""

        self.produced += 1

    def reset_read_idx(self, value: int) -> None:
        """Reinicia el índice de lectura para reintentos o saltos explícitos."""

        self.read_idx = value
