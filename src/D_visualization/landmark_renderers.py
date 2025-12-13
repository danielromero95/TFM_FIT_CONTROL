"""Renderizadores que generan vídeos con anotaciones de marcadores.
Organiza flujos en vivo y en memoria para compartir utilidades sin duplicación y
facilitar diagnósticos cuando el modelo de pose produce resultados inesperados."""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

from .landmark_drawing import _adaptive_style_for_region, draw_pose_on_frame
from .landmark_geometry import _estimate_subject_bbox, _normalize_points_for_frame
from .landmark_overlay_styles import OverlayStyle, RenderStats
from .landmark_transforms import _normalize_rotation_deg, _rotate_frame
from .landmark_video_io import DEFAULT_CODEC_PREFERENCE, _open_writer

logger = logging.getLogger(__name__)

# Exponemos las funciones principales del módulo.
__all__ = [
    "render_landmarks_video",
    "render_landmarks_video_streaming",
]


def render_landmarks_video_streaming(
    video_path: str,
    landmarks_sequence,
    crop_boxes,
    out_path: str,
    fps: float,
    *,
    processed_size: tuple[int, int],
    sample_rate: int = 1,
    rotate: int = 0,
    style: OverlayStyle | None = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    codec_preference: Sequence[str] = DEFAULT_CODEC_PREFERENCE,
) -> RenderStats:
    """Lee frames de disco, superpone marcadores y escribe un nuevo archivo de vídeo.
    Procesamos en streaming para monitorizar sesiones largas sin agotar memoria y
    poder revisar rápidamente grabaciones almacenadas en disco."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open the video for streaming: {video_path}")

    ok, first_raw = cap.read()
    if not ok:
        cap.release()
        return RenderStats(0, 0, 0, 0.0, "")

    rotation_in = _normalize_rotation_deg(rotate)
    first = _rotate_frame(first_raw, rotation_in)
    orig_h, orig_w = first.shape[:2]
    try:
        writer, used_fourcc = _open_writer(
            out_path,
            fps if fps > 0 else 1.0,
            (orig_w, orig_h),
            codec_preference,
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
            _get(landmarks_sequence, idx),
            cb,
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
        draw_pose_on_frame(fr_ann, pts, style=style_obj)
        writer.write(fr_ann)
        frames_written += 1

    try:
        _handle_one(0, first)
        frames_in += 1
        if progress_cb:
            try:
                progress_cb(frames_written, frames_in)
            except Exception:
                pass

        # Saltamos frames cuando ``sample_rate`` > 1 para acelerar visualizaciones largas.
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
            fr = _rotate_frame(raw, rotation_in)
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
    frames: Iterable[np.ndarray],
    landmarks_sequence,
    crop_boxes,
    out_path: str,
    fps: float,
    *,
    processed_size: tuple[int, int],
    style: OverlayStyle | None = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    codec_preference: Sequence[str] = DEFAULT_CODEC_PREFERENCE,
    output_rotate: int = 0,
    tighten_to_subject: bool = False,
    subject_margin: float = 0.12,
) -> RenderStats:
    """Procesa frames en memoria aplicando anotaciones y rotaciones opcionales.
    Este modo evita E/S innecesaria cuando los frames ya están precargados y permite
    experimentar con el pipeline sin depender de archivos temporales."""

    t0 = time.perf_counter()
    frames_in = frames_written = skipped = 0

    frames_iter = iter(frames)
    first = next(frames_iter, None)
    if first is None:
        return RenderStats(0, 0, 0, 0.0, "")

    orig_h, orig_w = first.shape[:2]
    base_style = style or OverlayStyle()
    subject_region_size: Optional[tuple[int, int]] = None
    rotation_out = _normalize_rotation_deg(output_rotate)

    def _safe_get(seq, idx):
        try:
            return seq[idx]
        except Exception:
            return None

    def _process_frame(idx: int, frame: np.ndarray) -> np.ndarray:
        """Genera un frame anotado teniendo en cuenta el recorte del sujeto.
        Esta separación permite reutilizar el cálculo tanto para escritura como para
        previsualizaciones in-memory."""

        nonlocal skipped, subject_region_size

        crop = _safe_get(crop_boxes, idx) if crop_boxes is not None else None
        if crop is not None and hasattr(crop, "all") and np.isnan(crop).all():
            crop = None
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
            # Reencuadramos para generar clips enfocados en la persona y mantenerla centrada.
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

    writer = None
    used_code = ""

    try:
        frame_to_write = _process_frame(0, first)
        writer_size = (frame_to_write.shape[1], frame_to_write.shape[0])
        writer, used_code = _open_writer(
            out_path,
            fps if fps > 0 else 1.0,
            writer_size,
            codec_preference,
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
