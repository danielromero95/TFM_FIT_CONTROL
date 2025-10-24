from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence, Tuple, Optional
import logging
import math

import cv2
import numpy as np

from src import config

try:
    # Prefer central style if available
    from src.config.pose_visualization import LANDMARK_COLOR as _LM, CONNECTION_COLOR as _CN
except Exception:
    _LM, _CN = config.LANDMARK_COLOR, config.CONNECTION_COLOR


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverlayStyle:
    connection_thickness: int = 2
    landmark_radius: int = 4
    connection_bgr: Tuple[int, int, int] = tuple(_CN)  # type: ignore
    landmark_bgr: Tuple[int, int, int] = tuple(_LM)  # type: ignore


@dataclass(frozen=True)
class RenderStats:
    frames_in: int
    frames_written: int
    skipped_empty: int
    duration_ms: float
    used_fourcc: str


def _normalize_points_for_frame(
    frame_landmarks,  # iterable of dict-like {"x","y"} or None
    crop_box: Optional[Sequence[float]],  # [x1_p,y1_p,x2_p,y2_p] in processed pixels or None
    orig_w: int,
    orig_h: int,
    proc_w: int,
    proc_h: int,
) -> dict[int, tuple[int, int]]:
    pts: dict[int, tuple[int, int]] = {}
    if frame_landmarks is None:
        return pts
    try:
        if all((math.isnan(lm["x"]) for lm in frame_landmarks)):
            return pts
    except Exception:
        pass
    sx, sy = (orig_w / float(proc_w)), (orig_h / float(proc_h))
    for idx, lm in enumerate(frame_landmarks):
        try:
            x, y = float(lm["x"]), float(lm["y"])
            if math.isnan(x) or math.isnan(y):
                continue
        except Exception:
            continue
        if crop_box is not None:
            x1_p, y1_p, x2_p, y2_p = map(float, crop_box)
            abs_x_p = x1_p + x * (x2_p - x1_p)
            abs_y_p = y1_p + y * (y2_p - y1_p)
            final_x = int(round(abs_x_p * sx))
            final_y = int(round(abs_y_p * sy))
        else:
            final_x = int(round(x * orig_w))
            final_y = int(round(y * orig_h))
        final_x = max(0, min(orig_w - 1, final_x))
        final_y = max(0, min(orig_h - 1, final_y))
        pts[idx] = (final_x, final_y)
    return pts


def draw_pose_on_frame(
    frame: np.ndarray,
    points_xy: Mapping[int, tuple[int, int]],
    *,
    connections: Sequence[tuple[int, int]] = tuple(config.POSE_CONNECTIONS),
    style: OverlayStyle = OverlayStyle(),
) -> None:
    for a, b in connections:
        if a in points_xy and b in points_xy:
            cv2.line(frame, points_xy[a], points_xy[b], style.connection_bgr, style.connection_thickness)
    for p in points_xy.values():
        cv2.circle(frame, p, style.landmark_radius, style.landmark_bgr, -1)


def _open_writer(path: str, fps: float, size: tuple[int, int], prefs: Sequence[str]) -> tuple[cv2.VideoWriter, str]:
    for code in prefs:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), max(fps, 1.0), size)
        if writer.isOpened():
            return writer, code
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for path={path} with prefs={prefs}")


def render_landmarks_video(
    frames: Iterable[np.ndarray],  # original-resolution frames
    landmarks_sequence,  # per-frame sequence
    crop_boxes,  # per-frame sequence or None
    out_path: str,
    fps: float,
    *,
    processed_size: tuple[int, int],  # REQUIRED to fix scaling bug
    style: OverlayStyle | None = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,  # (written, total)
    cancelled: Optional[Callable[[], bool]] = None,
    codec_preference: Sequence[str] = ("avc1", "mp4v", "H264"),
) -> RenderStats:
    """Stream frames -> overlay -> write. UI-agnostic."""
    import time

    t0 = time.perf_counter()
    frames_in = frames_written = skipped = 0

    frames_iter = iter(frames)
    first = next(frames_iter, None)
    if first is None:
        return RenderStats(0, 0, 0, 0.0, "")
    orig_h, orig_w = first.shape[:2]
    writer, used_code = _open_writer(out_path, fps if fps > 0 else 1.0, (orig_w, orig_h), codec_preference)

    try:
        def handle_one(idx: int, fr: np.ndarray):
            nonlocal frames_written, skipped
            h, w = fr.shape[:2]
            if (w, h) != (orig_w, orig_h):
                fr = cv2.resize(fr, (orig_w, orig_h))
            crop = None
            if crop_boxes is not None and idx < len(crop_boxes):
                cb = crop_boxes[idx]
                if cb is not None and not (np.isnan(cb).all() if hasattr(cb, "all") else False):
                    crop = cb
            pts = _normalize_points_for_frame(
                landmarks_sequence[idx] if idx < len(landmarks_sequence) else None,
                crop,
                orig_w,
                orig_h,
                processed_size[0],
                processed_size[1],
            )
            if not pts:
                skipped += 1
                writer.write(fr)
                frames_written += 1
                return
            fr_ann = fr.copy()
            draw_pose_on_frame(fr_ann, pts, style=style or OverlayStyle())
            writer.write(fr_ann)
            frames_written += 1

        handle_one(0, first)
        frames_in += 1
        if progress_cb:
            try:
                progress_cb(frames_written, frames_in)
            except Exception:
                pass

        for idx, fr in enumerate(frames_iter, start=1):
            if cancelled and cancelled():
                break
            frames_in += 1
            handle_one(idx, fr)
            if progress_cb and (idx % 10 == 0):
                try:
                    progress_cb(frames_written, frames_in)
                except Exception:
                    pass
    finally:
        writer.release()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return RenderStats(frames_in, frames_written, skipped, dt_ms, used_code)
