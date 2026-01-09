"""Helpers for retiming video writes to match source timestamps."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetimeWriter:
    """Track frame duplication to preserve timestamp-based timing."""

    fps_out: float
    base_ts: float | None = None
    last_ts: float | None = None
    frames_written: int = 0
    max_repeat: int = field(init=False)

    def __post_init__(self) -> None:
        fps_value = float(self.fps_out) if self.fps_out and self.fps_out > 0 else 30.0
        self.fps_out = fps_value
        self.max_repeat = int(math.ceil(self.fps_out * 2.0))

    def prime(self, ts: float, fallback_ts: float | None = None) -> float:
        """Initialize the base timestamp without writing frames."""

        return self._normalize_ts(ts, fallback_ts)

    def repeats_for_next(self, ts: float, fallback_ts: float | None = None) -> int:
        """Compute how many times to write the previous frame."""

        ts_value = self._normalize_ts(ts, fallback_ts)
        if self.base_ts is None or not np.isfinite(ts_value):
            repeat = 1
        else:
            desired_total = max(
                1,
                int(math.floor((ts_value - self.base_ts) * self.fps_out + 1e-6)),
            )
            repeat = max(0, desired_total - self.frames_written)
        repeat = int(min(self.max_repeat, repeat))
        self.frames_written += repeat
        return repeat

    def _normalize_ts(self, ts: float, fallback_ts: float | None) -> float:
        if np.isfinite(ts):
            ts_value = float(ts)
        elif fallback_ts is not None and np.isfinite(fallback_ts):
            ts_value = float(fallback_ts)
        else:
            ts_value = float("nan")
        if self.base_ts is None and np.isfinite(ts_value):
            self.base_ts = float(ts_value)

        if self.last_ts is not None and np.isfinite(self.last_ts):
            if not np.isfinite(ts_value) or ts_value <= self.last_ts:
                ts_value = self.last_ts + max(1.0 / self.fps_out, 0.001)

        if np.isfinite(ts_value):
            self.last_ts = float(ts_value)
        return float(ts_value)
