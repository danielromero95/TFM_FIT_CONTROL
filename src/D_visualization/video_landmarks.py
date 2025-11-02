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
    "transcode_video",
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
    crop_vals: Optional[tuple[float, float, float, float]] = None
    if crop_box is not None:
        try:
            crop_vals = tuple(map(float, crop_box))  # type: ignore[arg-type]
        except Exception:
            crop_vals = None

    treat_as_global = False
    if crop_vals is not None and max(crop_vals) > 1.0:
        for lm in frame_landmarks:
            try:
                x = float(lm["x"])
                y = float(lm["y"])
            except Exception:
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                treat_as_global = True
                break

    sx, sy = (orig_w / float(proc_w)), (orig_h / float(proc_h))
    for idx, lm in enumerate(frame_landmarks):
        try:
            x, y = float(lm["x"]), float(lm["y"])
            if math.isnan(x) or math.isnan(y):
                continue
        except Exception:
            continue
        if not treat_as_global and crop_vals is not None:
            x1_p, y1_p, x2_p, y2_p = crop_vals
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


def _estimate_subject_bbox(
    pts: dict[int, tuple[int, int]],
    frame_w: int,
    frame_h: int,
    *,
    margin: float = 0.12,
) -> tuple[int, int, int, int] | None:
    """
    A partir de los puntos ya pasados a píxeles, devuelve un bbox (x1,y1,x2,y2)
    ampliado con un pequeño margen. Si no hay puntos, devuelve None.
    """

    if not pts:
        return None

    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    if not xs or not ys:
        return None

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    base = min(frame_w, frame_h)
    margin_px = float(max(0.0, margin)) * float(base)

    x1 = int(math.floor(min_x - margin_px))
    y1 = int(math.floor(min_y - margin_px))
    x2 = int(math.ceil(max_x + margin_px + 1.0))
    y2 = int(math.ceil(max_y + margin_px + 1.0))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, max(x2, x1 + 1))
    y2 = min(frame_h, max(y2, y1 + 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _adaptive_style_for_region(width: int, height: int) -> OverlayStyle:
    """
    Devuelve un OverlayStyle con thickness/radius escalados al tamaño del recorte.
    Usa como referencia un lado corto de 320px: por debajo de eso, usa 2px/4px,
    por encima escala linealmente.
    """

    base = max(1, min(int(width), int(height)))
    thickness = max(2, round(base / 320 * 2))
    radius = max(3, round(base / 320 * 4))
    return OverlayStyle(connection_thickness=thickness, landmark_radius=radius)


_ROTATE_CODE = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _normalize_rotation_deg(value: int) -> int:
    """Clamp ``value`` to the closest multiple of 90 degrees."""

    value = int(value) % 360
    if value in _ROTATE_CODE:
        return value
    candidates = tuple(_ROTATE_CODE.keys())
    return min(candidates, key=lambda cand: min((value - cand) % 360, (cand - value) % 360))


def _rotate_frame(frame: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Return ``frame`` rotated by ``rotation_deg`` (clockwise)."""

    rotation = _normalize_rotation_deg(rotation_deg)
    code = _ROTATE_CODE.get(rotation)
    return cv2.rotate(frame, code) if code is not None else frame


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


def transcode_video(
    src_path: str,
    dst_path: str,
    *,
    fps: float,
    codec_preference: Sequence[str] = ("avc1", "H264"),
) -> tuple[bool, str]:
    """Re-encode ``src_path`` using the requested codecs.

    Returns ``(success, codec)`` and leaves ``dst_path`` on disk only when ``success``
    is ``True``.
    """

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False, ""

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return False, ""

    try:
        writer, used_code = _open_writer(
            dst_path,
            fps if fps > 0 else 1.0,
            (width, height),
            codec_preference,
        )
    except Exception:
        cap.release()
        return False, ""

    frames_written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frames_written += 1
    finally:
        writer.release()
        cap.release()

    if frames_written <= 0:
        try:
            Path(dst_path).unlink(missing_ok=True)
        except Exception:
            pass
        return False, ""

    return True, used_code


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
    output_rotate: int = 0,
    tighten_to_subject: bool = False,
    subject_margin: float = 0.12,
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
    rotation_out = _normalize_rotation_deg(output_rotate)
    base_style = style or OverlayStyle()
    writer: Optional[cv2.VideoWriter] = None
    used_code = ""
    writer_size: Optional[tuple[int, int]] = None
    subject_region_size: Optional[tuple[int, int]] = None

    def _safe_get(seq, index):
        try:
            return seq[index]
        except Exception:
            return None

    def _process_frame(idx: int, fr: np.ndarray) -> np.ndarray:
        nonlocal skipped, subject_region_size
        frame = fr
        h, w = frame.shape[:2]
        if (w, h) != (orig_w, orig_h):
            frame = cv2.resize(frame, (orig_w, orig_h))

        crop = None
        if crop_boxes is not None:
            cb = _safe_get(crop_boxes, idx)
            if cb is not None and not (np.isnan(cb).all() if hasattr(cb, "all") else False):
                crop = cb

        pts = _normalize_points_for_frame(
            _safe_get(landmarks_sequence, idx),
            crop,
            orig_w,
            orig_h,
            processed_size[0],
            processed_size[1],
        )

        if not pts:
            skipped += 1
            frame_to_write = frame
            if rotation_out:
                frame_to_write = _rotate_frame(frame_to_write, rotation_out)
            return frame_to_write

        frame_region = frame
        points_for_draw = pts
        style_for_draw = base_style

        if tighten_to_subject:
            bbox = _estimate_subject_bbox(pts, orig_w, orig_h, margin=subject_margin)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                if subject_region_size is not None:
                    target_w, target_h = subject_region_size
                    target_w = max(1, int(target_w))
                    target_h = max(1, int(target_h))
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    x1 = int(round(cx - target_w / 2))
                    y1 = int(round(cy - target_h / 2))
                    x2 = x1 + target_w
                    y2 = y1 + target_h
                    if x1 < 0:
                        x2 -= x1
                        x1 = 0
                    if x2 > orig_w:
                        shift = x2 - orig_w
                        x1 = max(0, x1 - shift)
                        x2 = orig_w
                    if y1 < 0:
                        y2 -= y1
                        y1 = 0
                    if y2 > orig_h:
                        shift = y2 - orig_h
                        y1 = max(0, y1 - shift)
                        y2 = orig_h
                if x2 > x1 and y2 > y1:
                    if subject_region_size is None:
                        subject_region_size = (x2 - x1, y2 - y1)
                    frame_region = frame[y1:y2, x1:x2]
                    points_for_draw = {k: (px - x1, py - y1) for k, (px, py) in pts.items()}
                    adaptive = _adaptive_style_for_region(frame_region.shape[1], frame_region.shape[0])
                    style_for_draw = OverlayStyle(
                        connection_thickness=adaptive.connection_thickness,
                        landmark_radius=adaptive.landmark_radius,
                        connection_bgr=base_style.connection_bgr,
                        landmark_bgr=base_style.landmark_bgr,
                    )

        frame_to_annotate = frame_region.copy()
        draw_pose_on_frame(frame_to_annotate, points_for_draw, style=style_for_draw)
        frame_to_write = frame_to_annotate

        if rotation_out:
            frame_to_write = _rotate_frame(frame_to_write, rotation_out)
        return frame_to_write

    try:
        frame_to_write = _process_frame(0, first)
        writer_size = (frame_to_write.shape[1], frame_to_write.shape[0])
        writer, used_code = _open_writer(
            out_path, fps if fps > 0 else 1.0, writer_size, codec_preference
        )
        writer.write(frame_to_write)
        frames_written += 1
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
            frame_to_write = _process_frame(idx, fr)
            if writer_size and (frame_to_write.shape[1], frame_to_write.shape[0]) != writer_size:
                frame_to_write = cv2.resize(frame_to_write, writer_size)
            writer.write(frame_to_write)
            frames_written += 1
            if progress_cb and (idx % 10 == 0):
                try:
                    progress_cb(frames_written, frames_in)
                except Exception:
                    pass
    finally:
        if writer:
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
