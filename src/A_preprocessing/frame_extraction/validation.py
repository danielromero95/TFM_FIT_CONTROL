"""Argument validation for frame extraction sampling modes."""

from __future__ import annotations

from typing import Literal, Optional


def _validate_sampling_args(
    sampling: Literal["auto", "time", "index"],
    target_fps: Optional[float],
    every_n: Optional[int],
    start_time: Optional[float],
    end_time: Optional[float],
) -> tuple[Literal["auto", "time", "index"], Optional[float], Optional[int]]:
    if sampling not in ("auto", "time", "index"):
        raise ValueError('sampling must be "auto", "time", or "index"')

    normalized_target_fps = float(target_fps) if target_fps else None
    if sampling in ("auto", "time") and not (normalized_target_fps and normalized_target_fps > 0):
        if sampling == "time":
            raise ValueError("target_fps must be > 0 for time-based sampling")

    normalized_every_n = int(every_n) if every_n is not None else None
    if sampling == "index":
        if normalized_every_n is None or normalized_every_n <= 0:
            raise ValueError("every_n must be a positive integer for index-based sampling")

    if start_time is not None and end_time is not None and end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    return sampling, normalized_target_fps, normalized_every_n
