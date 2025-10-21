"""Utilities to query rotation metadata from video files."""

from __future__ import annotations

from typing import Optional
import logging
import math
import shutil
import subprocess
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def _normalize_rotation(value: int) -> int:
    """Return the closest value among 0/90/180/270 for ``value``."""
    value = int(value) % 360
    candidates = [0, 90, 180, 270]
    return min(candidates, key=lambda candidate: abs(candidate - value))


def get_video_rotation(video_path: str) -> int:
    """Read the video rotation metadata using ``ffprobe`` when available."""
    try:
        if shutil.which("ffprobe") is None:
            logger.info("ffprobe not found on PATH; assuming rotation 0°.")
            return 0

        # 1) Read the classic ``rotate`` tag when present.
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream_tags=rotate",
            "-of",
            "default=nw=1:nk=1",
            str(Path(video_path)),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = (res.stdout or "").strip()
        if output:
            rotation = _normalize_rotation(int(float(output)))
            if rotation:
                logger.info("Rotation detected by ffprobe (tags): %d°", rotation)
            return rotation

        # 2) If no rotation metadata is present, default to 0°.
        logger.info("No rotation detected in metadata; assuming 0°.")
        return 0

    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Unable to read rotation with ffprobe (%s); assuming 0°.", exc)
        return 0


def probe_video_metadata(video_path: str) -> tuple[float, int, Optional[str], bool]:
    """
    Inspect the video file to determine a reliable FPS and frame count.
    Falls back to estimating FPS from duration when metadata is invalid.
    Returns:
        fps (float): base FPS (metadata/estimated/0.0)
        frame_count (int)
        warning (Optional[str]): human-readable warning if we had to estimate
        prefer_reader_fps (bool): hint that downstream should trust reader FPS
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"Could not open the video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fallback_warning: Optional[str] = None
    prefer_reader_fps = False

    if fps <= 1.0 or not math.isfinite(fps):
        duration_sec = _estimate_duration_seconds(capture, frame_count)
        if duration_sec > 0 and frame_count > 0:
            estimated_fps = frame_count / duration_sec
            fallback_warning = (
                "Invalid metadata FPS. Estimated from video duration "
                f"({estimated_fps:.2f} fps)."
            )
            fps = estimated_fps
        else:
            fallback_warning = (
                "Invalid metadata FPS and unreliable duration. Falling back to the reader-reported FPS."
            )
            fps = 0.0
            prefer_reader_fps = True

    capture.release()
    return float(fps), frame_count, fallback_warning, prefer_reader_fps


def _estimate_duration_seconds(capture: cv2.VideoCapture, frame_count: int) -> float:
    """Best-effort duration inference from the last frame timestamp."""
    if frame_count <= 1:
        return 0.0
    try:
        last_frame_index = max(frame_count - 1, 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
        if capture.grab():
            duration_msec = capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0
            return float(duration_msec) / 1000.0 if duration_msec > 0 else 0.0
    except Exception:  # pragma: no cover - backend dependent fallbacks
        return 0.0
    return 0.0
