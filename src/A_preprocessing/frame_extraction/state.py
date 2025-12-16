"""Dataclasses centrales y utilidades para procesar fotogramas extraídos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .utils import normalize_rotation_deg


@dataclass(slots=True)
class FrameInfo:
    """Metadatos asociados a un fotograma extraído del vídeo.

    Incluimos información de letterbox para poder reconstruir coordenadas en
    el espacio original cuando los frames se reescalan manteniendo el aspect
    ratio.
    """

    index: int
    timestamp_sec: float
    array: np.ndarray
    width: int
    height: int
    source_width: int | None = None
    source_height: int | None = None
    letterbox_scale: float | None = None
    letterbox_pad: tuple[int, int, int, int] | None = None


_ROTATE_MAP = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _apply_rotation(frame: np.ndarray, rotation_deg: Optional[int]) -> np.ndarray:
    """Rota el fotograma cuando el ángulo solicitado está soportado."""

    rot = _ROTATE_MAP.get(normalize_rotation_deg(int(rotation_deg or 0)))
    return cv2.rotate(frame, rot) if rot is not None else frame


@dataclass(slots=True)
class _FrameProcessor:
    """Transforma fotogramas aplicando rotación, escalado y conversión a gris."""

    rotate_deg: int
    resize_to: Optional[tuple[int, int]]
    to_gray: bool

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, float, tuple[int, int, int, int]] | None]:
        if self.rotate_deg:
            frame = _apply_rotation(frame, self.rotate_deg)

        letterbox_info: tuple[int, int, float, tuple[int, int, int, int]] | None = None
        if self.resize_to:
            target_width, target_height = self.resize_to
            height, width = frame.shape[:2]
            if width != target_width or height != target_height:
                scale = min(target_width / float(width), target_height / float(height))
                new_w = max(1, int(round(width * scale)))
                new_h = max(1, int(round(height * scale)))
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

                pad_x = target_width - new_w
                pad_y = target_height - new_h
                pad_left = pad_x // 2
                pad_right = pad_x - pad_left
                pad_top = pad_y // 2
                pad_bottom = pad_y - pad_top
                frame = cv2.copyMakeBorder(
                    resized,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=0,
                )
                letterbox_info = (width, height, scale, (pad_left, pad_top, pad_right, pad_bottom))

        if self.to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, letterbox_info


@dataclass(slots=True)
class _ProgressHandler:
    """Controla y notifica avances sin repetir porcentajes."""

    callback: Callable[[int], None] | None
    frame_count: int
    last_progress: int = -1

    def report(self, index: int) -> None:
        if not self.callback or self.frame_count <= 0:
            return
        percent = int((index / self.frame_count) * 100)
        if percent > self.last_progress:
            self.callback(percent)
            self.last_progress = percent

    def finalize(self) -> None:
        if self.callback and self.frame_count > 0 and self.last_progress < 100:
            self.callback(100)


@dataclass(slots=True)
class _IteratorContext:
    """Estado mutable compartido entre las implementaciones de iteradores."""

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
        if ts_ms > 0:
            return ts_ms / 1000.0
        if self.fps_base > 0:
            return idx / self.fps_base
        return 0.0

    def limit_reached(self) -> bool:
        return bool(self.max_frames and self.produced >= self.max_frames)

    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int, float, tuple[int, int, int, int]] | None]:
        return self.processor(frame)

    def report_progress(self, index: int) -> None:
        self.progress.report(index)

    def increment_produced(self) -> None:
        self.produced += 1

    def reset_read_idx(self, value: int) -> None:
        self.read_idx = value
