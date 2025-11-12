"""Iteradores basados en tiempo para muestrear fotogramas uniformemente."""

from __future__ import annotations

import logging
from typing import Iterator

import cv2

from .state import FrameInfo, _IteratorContext
from .utils import _seek_to_msec
from .index_sampling import _index_mode_iterator

logger = logging.getLogger(__name__)


SEEK_SLACK_MS = 40.0  # Tolerancia en ms antes de considerar que vamos retrasados.
_SEEK_RETRY_INTERVAL = 3  # Cada cuántos intentos se reintenta un ``seek`` duro.
_MAX_GRABS_PER_LOOP = 12  # Límite de ``grab`` consecutivos antes de forzar revisión.


def _time_mode_iterator(
    context: _IteratorContext,
    *,
    target_fps: float,
) -> Iterator[FrameInfo]:
    """Produce fotogramas espaciados por tiempo objetivo usando heurísticas de seek."""
    cap = context.cap
    interval_ms = 1000.0 / float(target_fps)
    next_target_ms = context.start_time * 1000.0 if context.start_time else 0.0
    bad_ts_streak = 0
    max_bad_ts = 5
    behind_counter = 0
    grabs_total = 0
    retrieves_total = 0
    seeks_total = 1 if context.start_time > 0 else 0
    fallbacks_to_index = 0

    def _log_telemetry(fallback_used: bool) -> None:
        logger.debug(
            "time-mode counters: grabs=%d retrieves=%d seeks=%d fallbacks_to_index=%d",
            grabs_total,
            retrieves_total,
            seeks_total,
            fallbacks_to_index,
        )
        logger.debug(
            "time-mode: grabs=%d retrieves=%d seeks=%d fallback=%s",
            grabs_total,
            retrieves_total,
            seeks_total,
            fallback_used,
        )

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            grabs_total += 1

            current_idx = context.read_idx
            context.read_idx += 1
            context.report_progress(current_idx)

            ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            effective_sec = context.effective_timestamp(ts_ms, current_idx)

            if ts_ms <= 0.0 or (
                context.end_time is not None and ts_ms > context.end_time * 1000.0 + 1.0
            ):
                bad_ts_streak += 1
            else:
                bad_ts_streak = 0

            if bad_ts_streak >= max_bad_ts:
                stride = (
                    max(1, int(round(context.fps_base / float(target_fps))))
                    if context.fps_base > 0
                    else 1
                )
                logger.warning(
                    "Timestamps unreliable. Falling back to index-based sampling with every_n=%d",
                    stride,
                )
                fallbacks_to_index += 1
                yield from _index_mode_iterator(
                    context,
                    stride=stride,
                    start_threshold=context.start_time if context.start_time > 0 else None,
                    start_idx_override=context.read_idx,
                )
                return

            if context.end_time is not None and effective_sec > context.end_time:
                break

            if ts_ms > 0.0 and ts_ms + SEEK_SLACK_MS < next_target_ms:
                behind_counter += 1
                grabs_performed = 0
                while grabs_performed < _MAX_GRABS_PER_LOOP:
                    ok = cap.grab()
                    if not ok:
                        return
                    grabs_total += 1
                    current_idx = context.read_idx
                    context.read_idx += 1
                    context.report_progress(current_idx)
                    ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                    effective_sec = context.effective_timestamp(ts_ms, current_idx)
                    grabs_performed += 1
                    if not (ts_ms > 0.0 and ts_ms + SEEK_SLACK_MS < next_target_ms):
                        break

                if ts_ms > 0.0 and ts_ms + SEEK_SLACK_MS < next_target_ms:
                    if behind_counter % _SEEK_RETRY_INTERVAL == 0:
                        _seek_to_msec(cap, next_target_ms)
                        seeks_total += 1
                        if context.fps_base > 0:
                            approx_idx = int(round((next_target_ms / 1000.0) * context.fps_base))
                            context.read_idx = max(context.read_idx, approx_idx)
                    continue
            else:
                behind_counter = 0

            if ts_ms < next_target_ms:
                continue

            ok, frame = cap.retrieve()
            retrieves_total += 1
            if not ok:
                continue

            frame = context.process_frame(frame)
            h, w = frame.shape[:2]
            yield FrameInfo(
                index=current_idx,
                timestamp_sec=context.effective_timestamp(ts_ms, current_idx),
                array=frame,
                width=w,
                height=h,
            )
            context.increment_produced()
            next_target_ms = (ts_ms if ts_ms > 0 else next_target_ms) + interval_ms
            if context.limit_reached():
                break

    finally:
        _log_telemetry(bool(fallbacks_to_index))
