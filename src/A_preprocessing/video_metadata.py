"""Unified video metadata reader with optional ffprobe support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import math
import shutil
import subprocess

import cv2

logger = logging.getLogger(__name__)

__all__ = ["VideoInfo", "read_video_file_info"]


@dataclass(slots=True)
class VideoInfo:
    """Structured video metadata."""
    path: Path
    width: int | None
    height: int | None
    fps: float | None              # None if invalid/unavailable
    frame_count: int | None        # None if unavailable
    duration_sec: float | None     # None if unavailable
    rotation: int | None           # 0/90/180/270 or None
    codec: str | None              # best-effort from FOURCC
    fps_source: str | None         # "metadata" | "estimated" | "reader" | None


def _normalize_rotation(value: int) -> int:
    """Return the closest value among 0/90/180/270 for ``value``."""
    value = int(value) % 360
    candidates = [0, 90, 180, 270]
    return min(candidates, key=lambda c: abs(c - value))


def _read_rotation_ffprobe(path: Path) -> Optional[int]:
    """Read rotation from ffprobe tags if available; return None if not found/available."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "default=nw=1:nk=1",
            str(path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = (res.stdout or "").strip()
        if output:
            return _normalize_rotation(int(float(output)))
    except Exception as exc:  # pragma: no cover
        logger.warning("ffprobe rotation check failed: %s", exc)
    return None


def _estimate_duration_seconds(cap: cv2.VideoCapture, frame_count: int) -> float:
    """Best-effort duration inference from the last frame timestamp."""
    if frame_count <= 1:
        return 0.0
    try:
        last_frame_index = max(frame_count - 1, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
        if cap.grab():
            duration_msec = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
            return float(duration_msec) / 1000.0 if duration_msec > 0 else 0.0
    except Exception:  # pragma: no cover - backend dependent fallbacks
        return 0.0
    return 0.0


def read_video_file_info(path: str | Path) -> VideoInfo:
    """
    Read robust video metadata using OpenCV and (optionally) ffprobe.

    Rules:
    - If OpenCV-reported FPS is invalid (<=1 or non-finite), try to estimate using (frame_count / duration).
    - If estimation is impossible, leave fps=None and mark fps_source="reader".
    - Rotation is read via ffprobe if available, else defaults to 0 (common convention for pipelines).
    """
    p = Path(path)
    if not p.exists():
        raise IOError(f"Video path does not exist: {p}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Could not open the video: {p}")

    try:
        # Dimensions
        width_val = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height_val = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        width = width_val if width_val > 0 else None
        height = height_val if height_val > 0 else None

        # FPS + frame count
        fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = frame_count_raw if frame_count_raw > 0 else None

        fps: float | None
        duration: float | None
        fps_source: str | None

        fps_valid = math.isfinite(fps_raw) and fps_raw > 1.0

        if fps_valid:
            fps = fps_raw
            fps_source = "metadata"
            duration = (frame_count_raw / fps_raw) if frame_count_raw > 0 else None
        else:
            # Try estimating duration from last frame timestamp, then FPS = frames/duration
            duration_est = _estimate_duration_seconds(cap, frame_count_raw)
            if duration_est > 0.0 and frame_count_raw > 0:
                fps = frame_count_raw / duration_est
                duration = duration_est
                fps_source = "estimated"
                logger.warning(
                    "Invalid metadata FPS. Estimated from duration: %.2f fps.", fps
                )
            else:
                fps = None
                duration = None
                fps_source = "reader"
                logger.warning(
                    "Invalid metadata FPS and unreliable duration. Falling back to reader."
                )

        # Rotation (best-effort)
        rotation = _read_rotation_ffprobe(p)
        if rotation is None:
            rotation = 0

        # Codec (best-effort from FOURCC)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        if fourcc_int:
            # Convert int FOURCC to 4-char code
            codec_chars = [
                chr((fourcc_int & 0xFF)),
                chr((fourcc_int >> 8) & 0xFF),
                chr((fourcc_int >> 16) & 0xFF),
                chr((fourcc_int >> 24) & 0xFF),
            ]
            codec = "".join(codec_chars).strip() or None
        else:
            codec = None

        logger.info(
            "VideoInfo: path=%s size=%sx%s frames=%s fps=%s (source=%s) rot=%s codec=%s",
            p.name, width, height, frame_count, fps, fps_source, rotation, codec
        )

        return VideoInfo(
            path=p,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
            rotation=rotation,
            codec=codec,
            fps_source=fps_source,
        )

    finally:
        cap.release()
