"""Frame extraction helpers with optional auto-rotation."""

import logging
import os
from typing import Callable, List, Tuple

import cv2

from src import config
from .video_metadata import read_video_file_info

logger = logging.getLogger(__name__)


def extract_and_preprocess_frames(
    video_path: str,
    sample_rate: int = 1,
    rotate: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> Tuple[List, float]:
    """Extract frames, apply rotation automatically and return full-resolution images."""
    logger.info("Starting extraction for: %s", video_path)

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in config.VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: '{ext}'.")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    info = read_video_file_info(video_path)
    fps_from_metadata = float(info.fps or 0.0)

    # Auto-detect rotation when not provided explicitly.
    if rotate is None:
        rotate = int(info.rotation or 0)

    logger.info(
        "Metadata summary: fps=%.2f rotation=%s", fps_from_metadata, rotate
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open the video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video properties: %d frames, %.2f FPS", frame_count, fps)

    original_frames: List = []
    idx = 0
    last_percent_done = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if progress_callback and frame_count > 0:
            percent_done = int((idx / frame_count) * 100)
            if percent_done > last_percent_done:
                progress_callback(percent_done)
                last_percent_done = percent_done

        if idx % sample_rate == 0:
            if rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            original_frames.append(frame)

        idx += 1

    cap.release()
    if progress_callback and frame_count > 0 and last_percent_done < 100:
        progress_callback(100)
    logger.info("Process complete. Extracted %d frames into memory.", len(original_frames))
    return original_frames, fps
