"""APIs públicas para extraer y preprocesar fotogramas de vídeo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Literal, Optional

import cv2
import numpy as np

from ..video_metadata import read_video_file_info
from .index_sampling import _index_mode_iterator
from .state import FrameInfo, _FrameProcessor, _IteratorContext, _ProgressHandler
from .time_sampling import _time_mode_iterator
from .utils import _load_video_info, _open_video_capture, _resolve_rotation, _seek_to_msec
from .validation import _validate_sampling_args

if TYPE_CHECKING:
    from ..video_metadata import VideoInfo

logger = logging.getLogger(__name__)



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
    """Genera fotogramas aplicando la estrategia de muestreo solicitada."""

    sampling, normalized_target_fps, normalized_every_n = _validate_sampling_args(
        sampling, target_fps, every_n, start_time, end_time
    )

    path_obj = Path(video_path)
    cap_obj, own_cap = _open_video_capture(path_obj, cap)

    progress = _ProgressHandler(progress_callback, frame_count=0)
    info: Optional["VideoInfo"] = prefetched_info

    try:
        info = _load_video_info(path_obj, cap_obj, info)

        fps_base = float(info.fps or 0.0)
        frame_count = int(info.frame_count or 0)
        progress.frame_count = frame_count

        rotate_deg = _resolve_rotation(rotate, info)

        try:
            cap_obj.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:  # pragma: no cover - backend dependent fallbacks
            pass

        start_time_value = float(start_time or 0.0)
        read_idx = int(round(start_time_value * fps_base)) if fps_base > 0 else 0

        if start_time_value > 0:
            _seek_to_msec(cap_obj, start_time_value * 1000.0)

        context = _IteratorContext(
            cap=cap_obj,
            fps_base=fps_base,
            frame_count=frame_count,
            start_time=start_time_value,
            end_time=end_time,
            max_frames=max_frames,
            processor=_FrameProcessor(rotate_deg, resize_to, to_gray),
            progress=progress,
            read_idx=read_idx,
        )

        start_threshold = start_time_value if start_time_value > 0 else None

        if sampling == "index" or (
            sampling == "auto" and not (normalized_target_fps and normalized_target_fps > 0)
        ):
            stride = normalized_every_n if normalized_every_n else 1
            yield from _index_mode_iterator(
                context,
                stride=stride,
                start_threshold=start_threshold,
            )
        else:
            yield from _time_mode_iterator(
                context,
                target_fps=float(normalized_target_fps),
            )

    finally:
        if own_cap:
            cap_obj.release()
        progress.finalize()



def extract_processed_frames_stream(
    *,
    video_path: str,
    rotate: int,
    resize_to: tuple[int, int],
    cap: Optional[cv2.VideoCapture],
    prefetched_info: Optional["VideoInfo"],
    progress_callback: Optional[Callable[[int], None]] = None,
    every_n: Optional[int] = None,
    target_fps: Optional[float] = None,
) -> Iterator[np.ndarray]:
    """Produce fotogramas ya rotados/escalados respetando los filtros de muestreo."""

    sampling_kwargs: dict[str, object]
    if target_fps and target_fps > 0:
        sampling_mode: Literal["time", "index"] = "time"
        sampling_kwargs = {"target_fps": float(target_fps)}
    else:
        sampling_mode = "index"
        stride = int(every_n) if every_n is not None else 1
        sampling_kwargs = {"every_n": max(1, stride)}

    for finfo in extract_frames_stream(
        video_path=video_path,
        sampling=sampling_mode,
        rotate=rotate,
        resize_to=resize_to,
        progress_callback=progress_callback,
        cap=cap,
        prefetched_info=prefetched_info,
        **sampling_kwargs,
    ):
        yield finfo.array
