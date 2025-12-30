"""Estado reutilizable para gestionar el ROI de recorte en vídeo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


def _clamp_roi(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> list[int]:
    x1_i = int(np.clip(np.floor(x1), 0, max(width, 1)))
    y1_i = int(np.clip(np.floor(y1), 0, max(height, 1)))
    x2_i = int(np.clip(np.ceil(x2), x1_i + 1, max(width, 1)))
    y2_i = int(np.clip(np.ceil(y2), y1_i + 1, max(height, 1)))
    return [x1_i, y1_i, x2_i, y2_i]


def _expand_roi(
    roi: Sequence[int], *, factor: float, width: int, height: int, min_ratio: float
) -> list[int]:
    x1, y1, x2, y2 = map(float, roi)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    half_w = (w * factor) * 0.5
    half_h = (h * factor) * 0.5
    min_half = max(min_ratio * min(width, height) * 0.5, 1.0)
    half_w = max(half_w, min_half)
    half_h = max(half_h, min_half)
    new_x1, new_x2 = cx - half_w, cx + half_w
    new_y1, new_y2 = cy - half_h, cy + half_h
    return _clamp_roi(new_x1, new_y1, new_x2, new_y2, width, height)


@dataclass
class RoiDebugRecorder:
    path: Optional[str] = None
    max_frames: int = 50
    records: list[dict[str, object]] = field(default_factory=list)

    def record(self, data: dict[str, object]) -> None:
        if self.path and len(self.records) < self.max_frames:
            self.records.append(data)

    def finalize(self) -> None:
        if not self.path or not self.records:
            return
        import json

        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.records, fh, ensure_ascii=False, indent=2)


@dataclass
class RoiState:
    """Control robusto del ROI para los estimadores con recorte."""

    warmup_frames: int = 10
    fallback_misses: int = 2
    expansion_factor: float = 1.8
    min_roi_ratio: float = 0.4
    min_fullframe_ratio: float = 1.0
    recorder: Optional[RoiDebugRecorder] = None

    last_roi: Optional[list[int]] = None
    fail_streak: int = 0
    frame_idx: int = 0
    force_full_next: bool = True
    has_pose: bool = False

    def _full_frame_roi(self, width: int, height: int) -> list[int]:
        min_w = int(width * self.min_fullframe_ratio)
        min_h = int(height * self.min_fullframe_ratio)
        return _clamp_roi((width - min_w) / 2.0, (height - min_h) / 2.0, (width + min_w) / 2.0, (height + min_h) / 2.0, width, height)

    def next_roi(self, width: int, height: int) -> tuple[list[int], bool]:
        warmup_active = (not self.has_pose) or (self.frame_idx < self.warmup_frames)
        use_full = warmup_active or self.force_full_next or self.last_roi is None
        roi = self._full_frame_roi(width, height) if use_full else list(self.last_roi)
        return roi, use_full or warmup_active

    def update_success(self, bbox: Sequence[float], width: int, height: int) -> None:
        x_min, y_min, x_max, y_max = bbox
        self.last_roi = _expand_roi((x_min, y_min, x_max, y_max), factor=1.0, width=width, height=height, min_ratio=self.min_roi_ratio)
        self.fail_streak = 0
        self.force_full_next = False
        self.has_pose = True
        self.frame_idx += 1

    def update_failure(self, width: int, height: int) -> None:
        self.fail_streak += 1
        if self.last_roi is not None:
            self.last_roi = _expand_roi(self.last_roi, factor=self.expansion_factor, width=width, height=height, min_ratio=self.min_roi_ratio)
        if self.fail_streak >= self.fallback_misses:
            self.last_roi = None
            self.force_full_next = True
        self.frame_idx += 1

    def emit_debug(
        self,
        *,
        frame_idx: int,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
        crop_used: bool,
        roi: Sequence[int],
        pose_ok: bool,
        warmup_active: bool,
        fallback_to_full: bool,
        letterbox: Optional[dict[str, object]] = None,
    ) -> None:
        if not self.recorder:
            return
        self.recorder.record(
            {
                "frame_idx": int(frame_idx),
                "input_size": list(map(int, input_size)),
                "output_size": list(map(int, output_size)),
                "crop_used": bool(crop_used),
                "roi": [int(v) for v in roi],
                "pose_ok": bool(pose_ok),
                "fail_streak": int(self.fail_streak),
                "warmup_active": bool(warmup_active),
                "fallback_to_full": bool(fallback_to_full),
                "letterbox": letterbox,
            }
        )

