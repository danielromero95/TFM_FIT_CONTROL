from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import video_landmarks_visualization as vlv


logger = logging.getLogger(__name__)


__all__ = [
    "OverlayStyle",
    "RenderStats",
    "draw_pose_on_frame",
    "render_landmarks_video",
    "render_landmarks_video_streaming",
]


@dataclass(frozen=True)
class OverlayStyle:
    connection_thickness: int = vlv.THICKNESS_DEFAULT
    landmark_radius: int = vlv.RADIUS_DEFAULT
    connection_bgr: Tuple[int, int, int] = tuple(vlv.CONNECTION_COLOR)
    landmark_bgr: Tuple[int, int, int] = tuple(vlv.LANDMARK_COLOR)


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
    connections: Sequence[tuple[int, int]] = tuple(vlv.POSE_CONNECTIONS),
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
            logger.info(
                "Opened VideoWriter fourcc=%s size=%s fps=%.2f path=%s",
                code,
                size,
                max(fps, 1.0),
                path,
            )
            return writer, code
        logger.info("VideoWriter open failed with fourcc=%s; trying next...", code)
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for path={path} with prefs={prefs}")


def render_landmarks_video_streaming(
    video_path: str,
    landmarks_sequence,
    crop_boxes,
    out_path: str,
    fps: float,
    *,
    processed_size: tuple[int, int],
    sample_rate: int = 1,
    rotate: int = 0,  # 0|90|180|270
    style: OverlayStyle | None = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    codec_preference: Sequence[str] = ("avc1", "mp4v", "H264"),
) -> RenderStats:
    """
    Stream frames from `video_path` applying `sample_rate` and `rotate`, overlay landmarks, and write `out_path`.
    The i-th element of `landmarks_sequence` and `crop_boxes` corresponds to the i-th kept (sampled) frame.
    `processed_size` is the (width, height) used during pose processing (needed for scaling back).
    """
    import time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open the video for streaming: {video_path}")

    def _rotate(f, deg):
        if not deg:
            return f
        if deg == 90:
            return cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
        if deg == 180:
            return cv2.rotate(f, cv2.ROTATE_180)
        if deg == 270:
            return cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return f

    ok, first_raw = cap.read()
    if not ok:
        cap.release()
        return RenderStats(0, 0, 0, 0.0, "")

    first = _rotate(first_raw, rotate)
    orig_h, orig_w = first.shape[:2]
    try:
        writer, used_fourcc = _open_writer(
            out_path, fps if fps > 0 else 1.0, (orig_w, orig_h), codec_preference
        )
    except Exception:
        cap.release()
        raise

    t0 = time.perf_counter()
    frames_in = frames_written = skipped = 0
    style_obj = style or OverlayStyle()

    def _get(seq, i):
        try:
            return seq[i]
        except Exception:
            return None

    def _handle_one(idx, fr):
        nonlocal frames_written, skipped
        h, w = fr.shape[:2]
        if (w, h) != (orig_w, orig_h):
            fr = cv2.resize(fr, (orig_w, orig_h))
        cb = _get(crop_boxes, idx) if crop_boxes is not None else None
        if cb is not None and hasattr(cb, "all") and np.isnan(cb).all():
            cb = None
        pts = _normalize_points_for_frame(
            _get(landmarks_sequence, idx), cb, orig_w, orig_h, processed_size[0], processed_size[1]
        )
        if not pts:
            skipped += 1
            writer.write(fr)
            frames_written += 1
            return
        fr_ann = fr.copy()
        draw_pose_on_frame(fr_ann, pts, style=style_obj)
        writer.write(fr_ann)
        frames_written += 1

    try:
        # Sampled frame index 0
        _handle_one(0, first)
        frames_in += 1
        if progress_cb:
            try:
                progress_cb(frames_written, frames_in)
            except Exception:
                pass

        # Keep every `sample_rate`-th raw frame
        raw_skip = max(0, int(sample_rate) - 1)
        while True:
            if cancelled and cancelled():
                break
            for _ in range(raw_skip):
                ok, _ = cap.read()
                if not ok:
                    raw_skip = 0
                    break
            ok, raw = cap.read()
            if not ok:
                break
            fr = _rotate(raw, rotate)
            frames_in += 1
            _handle_one(frames_in - 1, fr)
            if progress_cb and (frames_in % 10 == 0):
                try:
                    progress_cb(frames_written, frames_in)
                except Exception:
                    pass
    finally:
        writer.release()
        cap.release()

    dt_ms = (time.perf_counter() - t0) * 1000.0
    if progress_cb:
        try:
            progress_cb(frames_written, frames_in)
        except Exception:
            pass
    logger.info(
        "Rendered debug video (streaming): frames_in=%d written=%d skipped=%d fourcc=%s ms=%.1f path=%s",
        frames_in,
        frames_written,
        skipped,
        used_fourcc,
        dt_ms,
        out_path,
    )
    return RenderStats(
        frames_in=frames_in,
        frames_written=frames_written,
        skipped_empty=skipped,
        duration_ms=dt_ms,
        used_fourcc=used_fourcc,
    )


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
    """Iterate in-memory frames -> overlay -> write. UI-agnostic.

    `landmarks_sequence[i]` and `crop_boxes[i]` correspond to the i-th kept (sampled) frame.
    `processed_size` is the (width, height) used during pose processing (needed for scaling back).
    """
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
    if progress_cb:
        try:
            progress_cb(frames_written, frames_in)
        except Exception:
            pass
    logger.info(
        "Rendered debug video (in-memory): frames_in=%d written=%d skipped=%d fourcc=%s ms=%.1f path=%s",
        frames_in,
        frames_written,
        skipped,
        used_code,
        dt_ms,
        out_path,
    )
    return RenderStats(
        frames_in=frames_in,
        frames_written=frames_written,
        skipped_empty=skipped,
        duration_ms=dt_ms,
        used_fourcc=used_code,
    )
