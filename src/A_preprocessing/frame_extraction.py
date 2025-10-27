"""Frame extraction helpers with optional auto-rotation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Literal, Optional, Tuple

import cv2
import numpy as np

from src import config
from .video_metadata import read_video_file_info

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameInfo:
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
    rot = _ROTATE_MAP.get(int(rotation_deg or 0))
    return cv2.rotate(frame, rot) if rot is not None else frame


def _seek_to_msec(cap: cv2.VideoCapture, msec: float) -> None:
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(msec))
    except Exception:
        # Silencioso: algunos backends no soportan seek fiable.
        pass


def extract_frames_stream(
    video_path: str | Path,
    *,
    sampling: Literal["auto", "time", "index"] = "auto",
    target_fps: Optional[float] = None,
    every_n: Optional[int] = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    max_frames: Optional[int] = None,
    resize_to: Optional[tuple[int, int]] = None,
    to_gray: bool = False,
    rotate: int | str | None = "auto",
    progress_callback: Callable[[int], None] | None = None,
    cap: Optional[cv2.VideoCapture] = None,
    prefetched_info: Optional["VideoInfo"] = None,
) -> Iterator[FrameInfo]:
    """Stream video frames according to the requested sampling strategy.

    When ``cap`` is supplied the existing ``VideoCapture`` instance is reused and
    not released by this function. ``prefetched_info`` allows callers to provide
    metadata obtained elsewhere to avoid redundant probing.
    """

    if sampling not in ("auto", "time", "index"):
        raise ValueError('sampling must be "auto", "time", or "index"')
    if sampling in ("auto", "time") and (not target_fps or target_fps <= 0):
        if sampling == "time":
            raise ValueError("target_fps must be > 0 for time-based sampling")
    if sampling == "index":
        if every_n is None or int(every_n) <= 0:
            raise ValueError("every_n must be a positive integer for index-based sampling")
    if start_time is not None and end_time is not None and end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    path_obj = Path(video_path)
    p = str(path_obj)

    cap_obj = cap if cap is not None else cv2.VideoCapture(p)
    own_cap = cap is None

    if not cap_obj.isOpened():
        if own_cap:
            cap_obj.release()
        raise IOError(f"Could not open the video: {p}")

    info = prefetched_info
    frame_count = 0
    last_progress = -1

    try:
        if info is None:
            info = read_video_file_info(path_obj, cap=cap_obj)

        fps_base = float(info.fps or 0.0)
        frame_count = int(info.frame_count or 0)

        if rotate == "auto" or rotate is None:
            rotate_deg = int(info.rotation or 0)
        else:
            rotate_deg = int(rotate)

        cap = cap_obj
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:  # pragma: no cover - backend dependent fallbacks
            pass

        t_start = float(start_time or 0.0)
        read_idx = int(round(t_start * fps_base)) if fps_base > 0 else 0
        produced = 0

        if t_start > 0:
            _seek_to_msec(cap, t_start * 1000.0)

        def _process_frame(frame: np.ndarray) -> np.ndarray:
            if rotate_deg:
                frame = _apply_rotation(frame, rotate_deg)
            if resize_to:
                target_width, target_height = resize_to
                height, width = frame.shape[:2]
                if width != target_width or height != target_height:
                    interpolation = (
                        cv2.INTER_AREA if width > target_width or height > target_height else cv2.INTER_LINEAR
                    )
                    frame = cv2.resize(frame, resize_to, interpolation=interpolation)
            if to_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame

        def _effective_timestamp(ts_ms: float, idx: int) -> float:
            if ts_ms > 0:
                return ts_ms / 1000.0
            if fps_base > 0:
                return idx / fps_base
            return 0.0

        def time_mode_iterator() -> Iterator[FrameInfo]:
            nonlocal produced, read_idx, last_progress
            interval_ms = 1000.0 / float(target_fps)
            next_target_ms = (t_start * 1000.0) if start_time else 0.0
            bad_ts_streak = 0
            max_bad_ts = 5

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if progress_callback and frame_count > 0:
                    percent = int((read_idx / frame_count) * 100)
                    if percent > last_progress:
                        progress_callback(percent)
                        last_progress = percent

                ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                effective_sec = _effective_timestamp(ts_ms, read_idx)

                if ts_ms <= 0.0 or (
                    end_time is not None and ts_ms > end_time * 1000.0 + 1.0
                ):
                    bad_ts_streak += 1
                else:
                    bad_ts_streak = 0

                if end_time is not None and effective_sec > end_time:
                    break

                if ts_ms >= next_target_ms:
                    frame = _process_frame(frame)
                    h, w = frame.shape[:2]
                    yield FrameInfo(
                        index=read_idx,
                        timestamp_sec=_effective_timestamp(ts_ms, read_idx),
                        array=frame,
                        width=w,
                        height=h,
                    )
                    produced += 1
                    next_target_ms = (
                        (ts_ms if ts_ms > 0 else next_target_ms) + interval_ms
                    )
                    if max_frames and produced >= max_frames:
                        break

                read_idx += 1

                if bad_ts_streak >= max_bad_ts:
                    stride = (
                        max(1, int(round(fps_base / float(target_fps))))
                        if fps_base > 0
                        else 1
                    )
                    logger.warning(
                        "Timestamps unreliable. Falling back to index-based sampling with every_n=%d",
                        stride,
                    )
                    yield from index_mode_iterator(fallback_stride=stride, start_idx_offset=read_idx)
                    return

            return

        def index_mode_iterator(
            *, fallback_stride: Optional[int] = None, start_idx_offset: Optional[int] = None
        ) -> Iterator[FrameInfo]:
            nonlocal produced, read_idx, last_progress
            stride = int(fallback_stride or every_n or 1)
            idx_local = 0

            if start_idx_offset is not None:
                read_idx = start_idx_offset

            start_threshold = t_start if t_start > 0 else None

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if progress_callback and frame_count > 0:
                    percent = int((read_idx / frame_count) * 100)
                    if percent > last_progress:
                        progress_callback(percent)
                        last_progress = percent

                ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                effective_sec = _effective_timestamp(ts_ms, read_idx)

                if end_time is not None and effective_sec > end_time:
                    break

                if start_threshold is not None:
                    if effective_sec + 1e-6 < start_threshold:
                        read_idx += 1
                        idx_local += 1
                        continue

                if idx_local % stride == 0:
                    frame = _process_frame(frame)
                    h, w = frame.shape[:2]
                    yield FrameInfo(
                        index=read_idx,
                        timestamp_sec=_effective_timestamp(ts_ms, read_idx),
                        array=frame,
                        width=w,
                        height=h,
                    )
                    produced += 1
                    if max_frames and produced >= max_frames:
                        break

                read_idx += 1
                idx_local += 1

            return

        if sampling == "index" or (sampling == "auto" and not (target_fps and target_fps > 0)):
            yield from index_mode_iterator()
        else:
            yield from time_mode_iterator()

    finally:
        if own_cap:
            cap_obj.release()
        if progress_callback and frame_count > 0 and last_progress < 100:
            progress_callback(100)


def extract_processed_frames_stream(
    *,
    video_path: str,
    every_n: int,
    rotate: int,
    resize_to: tuple[int, int],
    cap: Optional[cv2.VideoCapture],
    prefetched_info: Optional["VideoInfo"],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Iterator[np.ndarray]:
    """Stream frames with index-based sampling (every_n), applying rotation and resizing on the fly.

    Yields np.ndarray frames already in the target size/orientation.
    """

    for finfo in extract_frames_stream(
        video_path=video_path,
        sampling="index",
        every_n=every_n,
        rotate=rotate,
        resize_to=resize_to,
        progress_callback=progress_callback,
        cap=cap,
        prefetched_info=prefetched_info,
    ):
        yield finfo.array


def extract_and_preprocess_frames(
    video_path: str,
    sample_rate: int = 1,
    rotate: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
    *,
    cap: Optional[cv2.VideoCapture] = None,
    prefetched_info: Optional["VideoInfo"] = None,
) -> Tuple[List, float]:
    """Extract frames, apply rotation automatically and return full-resolution images.

    Both ``cap`` and ``prefetched_info`` let callers reuse already-open captures
    and metadata for efficiency; ownership of ``cap`` remains with the caller.
    """
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

        # Auto-detect rotation when not provided explicitly.
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
