"""Segmentación de repeticiones con histéresis y validación por caída de la barra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .constants import (
    BAR_DROP_MIN_NORM,
    EVENT_MIN_GAP_SECONDS,
    KNEE_DOWN_THRESHOLD_DEG,
    KNEE_UP_THRESHOLD_DEG,
)
from .types import RepSlice


@dataclass(frozen=True)
class SegmentationDebug:
    """Información que describe cómo se decidieron los límites de cada repetición."""

    slices: Sequence[RepSlice]
    knee_events: Sequence[int]
    bar_drop_events: Sequence[int]


def segment_reps(
    knee_angle: np.ndarray,
    bar_y: np.ndarray,
    torso_scale: float,
    sampling_rate: float,
) -> List[RepSlice]:
    """Segmenta el clip en repeticiones usando histéresis de rodilla y detección de caída de barra."""

    n = int(max(knee_angle.size, bar_y.size))
    if n == 0:
        return [RepSlice(0, 1)]

    knee = np.asarray(knee_angle, dtype=float)
    bar = np.asarray(bar_y, dtype=float)

    if knee.size != n:
        knee = _pad_to_length(knee, n)
    if bar.size != n:
        bar = _pad_to_length(bar, n)

    finite_mask = np.isfinite(knee)
    if finite_mask.sum() < 3:
        return [RepSlice(0, n)]

    knee = knee.copy()
    knee[~finite_mask] = np.interp(
        np.flatnonzero(~finite_mask),
        np.flatnonzero(finite_mask),
        knee[finite_mask],
    )

    min_gap = int(round(sampling_rate * EVENT_MIN_GAP_SECONDS))

    down_candidates = np.flatnonzero(knee <= KNEE_DOWN_THRESHOLD_DEG)
    up_candidates = np.flatnonzero(knee >= KNEE_UP_THRESHOLD_DEG)

    down_frames: List[int] = []
    up_frames: List[int] = []

    if down_candidates.size == 0 or up_candidates.size == 0:
        return [RepSlice(0, n)]

    down_idx = 0
    up_idx = 0
    last_event = -min_gap
    looking_down = True

    while down_idx < down_candidates.size or up_idx < up_candidates.size:
        if looking_down:
            if down_idx >= down_candidates.size:
                break
            candidate = int(down_candidates[down_idx])
            down_idx += 1
            if candidate - last_event < min_gap:
                continue
            down_frames.append(candidate)
            last_event = candidate
            looking_down = False
        else:
            if up_idx >= up_candidates.size:
                break
            candidate = int(up_candidates[up_idx])
            up_idx += 1
            if candidate <= down_frames[-1] or candidate - last_event < min_gap:
                continue
            up_frames.append(candidate)
            last_event = candidate
            looking_down = True

    if not down_frames or not up_frames:
        return [RepSlice(0, n)]

    if up_frames[0] < down_frames[0]:
        up_frames = up_frames[1:]
    if len(up_frames) < len(down_frames):
        down_frames = down_frames[: len(up_frames)]
    else:
        up_frames = up_frames[: len(down_frames)]

    if not down_frames:
        return [RepSlice(0, n)]

    slices: List[RepSlice] = []
    bar_drop_threshold = BAR_DROP_MIN_NORM * max(torso_scale, 1e-5)
    for down, up in zip(down_frames, up_frames):
        start = max(0, down - int(0.4 * (up - down)))
        end = min(n, up + int(0.4 * (up - down)))

        if np.isfinite(bar_drop_threshold) and bar_drop_threshold > 0 and np.isfinite(bar).any():
            segment_bar = bar[start:end]
            finite_bar = segment_bar[np.isfinite(segment_bar)]
            if finite_bar.size:
                drop = float(finite_bar.max() - finite_bar.min())
                if drop < bar_drop_threshold:
                    continue

        slices.append(RepSlice(start, end))

    if not slices:
        return [RepSlice(0, n)]

    merged: List[RepSlice] = []
    last_end = -1
    for rep_slice in slices:
        if last_end >= 0 and rep_slice.start <= last_end:
            merged[-1] = RepSlice(merged[-1].start, max(merged[-1].end, rep_slice.end))
        else:
            merged.append(rep_slice)
        last_end = merged[-1].end

    return merged


def _pad_to_length(series: np.ndarray, length: int) -> np.ndarray:
    if series.size >= length:
        return series[:length]
    pad = np.full(length - series.size, np.nan)
    return np.concatenate([series, pad])

