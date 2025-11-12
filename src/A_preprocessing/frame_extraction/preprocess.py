"""Tareas de extracción de fotogramas a resolución completa para el pipeline."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import cv2
import numpy as np

from src import config

from ..video_metadata import read_video_file_info
from .core import extract_frames_stream

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ..video_metadata import VideoInfo



def extract_and_preprocess_frames(
    video_path: str,
    sample_rate: int = 1,
    rotate: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
    *,
    cap: Optional[cv2.VideoCapture] = None,
    prefetched_info: Optional["VideoInfo"] = None,
) -> Tuple[List, float]:
    """Extrae fotogramas, aplica rotación automática y devuelve imágenes completas."""

    logger.info("Starting extraction for: %s", video_path)

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in config.VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: '{ext}'.")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    cap_obj = cap if cap is not None else cv2.VideoCapture(video_path)
    own_cap = cap is None

    if not cap_obj.isOpened():
        if own_cap:
            cap_obj.release()
        raise IOError(f"Could not open the video: {video_path}")

    try:
        info = prefetched_info or read_video_file_info(video_path, cap=cap_obj)
        fps_from_metadata = float(info.fps or 0.0)

        if rotate is None:
            rotate = int(info.rotation or 0)

        logger.info(
            "Metadata summary: fps=%.2f rotation=%s", fps_from_metadata, rotate
        )

        fps = cap_obj.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Video properties: %d frames, %.2f FPS", frame_count, fps)

        original_frames: List[np.ndarray] = []
        for finfo in extract_frames_stream(
            video_path,
            sampling="index",
            every_n=sample_rate,
            rotate=rotate,
            progress_callback=progress_callback,
            cap=cap_obj,
            prefetched_info=info,
        ):
            original_frames.append(finfo.array)

        logger.info(
            "Process complete. Extracted %d frames into memory.", len(original_frames)
        )
        return original_frames, fps
    finally:
        if own_cap:
            cap_obj.release()
