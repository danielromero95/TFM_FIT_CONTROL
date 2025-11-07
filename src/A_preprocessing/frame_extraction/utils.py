"""Utility helpers for interacting with OpenCV captures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from ..video_metadata import VideoInfo, read_video_file_info


def _seek_to_msec(cap: cv2.VideoCapture, msec: float) -> None:
    """Attempt to seek the capture to the provided millisecond offset."""

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(msec))
    except Exception:
        # Silencioso: algunos backends no soportan seek fiable.
        pass


def _open_video_capture(
    video_path: str | Path, cap: Optional[cv2.VideoCapture]
) -> tuple[cv2.VideoCapture, bool]:
    cap_obj = cap if cap is not None else cv2.VideoCapture(str(video_path))
    own_cap = cap is None

    if not cap_obj.isOpened():
        if own_cap:
            cap_obj.release()
        raise IOError(f"Could not open the video: {video_path}")
    return cap_obj, own_cap


def _load_video_info(
    path_obj: Path, cap_obj: cv2.VideoCapture, prefetched_info: Optional[VideoInfo]
) -> VideoInfo:
    info = prefetched_info
    if info is None:
        info = read_video_file_info(path_obj, cap=cap_obj)
    return info


def _resolve_rotation(rotate: int | str | None, info: VideoInfo) -> int:
    if rotate == "auto" or rotate is None:
        return int(info.rotation or 0)
    return int(rotate)
