"""Muestreo de fotogramas guiado por índices consecutivos."""

from __future__ import annotations

from typing import Iterator, Optional

import cv2

from .state import FrameInfo, _IteratorContext


def _index_mode_iterator(
    context: _IteratorContext,
    *,
    stride: int,
    start_threshold: Optional[float],
    start_idx_override: Optional[int] = None,
) -> Iterator[FrameInfo]:
    """Entrega fotogramas según un paso ``stride`` fijo y umbrales temporales."""
    cap = context.cap
    stride = max(1, stride)
    idx_local = 0

    if start_idx_override is not None:
        context.reset_read_idx(start_idx_override)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        context.report_progress(context.read_idx)

        ts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        effective_sec = context.effective_timestamp(ts_ms, context.read_idx)

        if context.end_time is not None and effective_sec > context.end_time:
            break

        if start_threshold is not None:
            if effective_sec + 1e-6 < start_threshold:
                context.read_idx += 1
                idx_local += 1
                continue

        if idx_local % stride == 0:
            processed = context.process_frame(frame)
            h, w = processed.shape[:2]
            yield FrameInfo(
                index=context.read_idx,
                timestamp_sec=context.effective_timestamp(ts_ms, context.read_idx),
                array=processed,
                width=w,
                height=h,
            )
            context.increment_produced()
            if context.limit_reached():
                break

        context.read_idx += 1
        idx_local += 1

    return
