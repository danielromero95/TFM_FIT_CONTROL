"""Utilities to query rotation metadata from video files."""

import logging
import shutil
import subprocess
from pathlib import Path

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
