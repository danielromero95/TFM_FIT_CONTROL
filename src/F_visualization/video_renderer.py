"""High-quality video renderer for drawing pose landmarks."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src import config

logger = logging.getLogger(__name__)


def render_landmarks_on_video_hq(
    frames: list[np.ndarray],
    sequence,
    crop_boxes,
    out_path: str,
    fps: float,
    *,
    processed_size: tuple[int, int] | None = None,
    landmarks_normalized: bool = True,
    clear_rotation_tag: bool = True,
) -> None:
    """Renderiza el vídeo de debug sin aplicar rotaciones adicionales.
    Si los landmarks provienen de frames redimensionados/rotados previamente (processed_frames),
    re-proyecta a las dimensiones reales de cada frame de salida.
    """

    logger.info("Starting HQ video rendering at: %s", out_path)
    if not frames:
        logger.warning("No frames available to render.")
        return

    dst_h, dst_w = frames[0].shape[:2]
    if processed_size is None:
        proc_w, proc_h = dst_w, dst_h
    else:
        proc_w, proc_h = processed_size

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (dst_w, dst_h))
    if not writer.isOpened():
        logger.warning("Falling back to mp4v VideoWriter for: %s", out_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (dst_w, dst_h))
    if not writer.isOpened():
        logger.error("Could not open VideoWriter for path: %s", out_path)
        return

    scale_x = dst_w / proc_w if proc_w else 1.0
    scale_y = dst_h / proc_h if proc_h else 1.0

    for index, frame in enumerate(frames):
        frame_h, frame_w = frame.shape[:2]
        if frame_h != dst_h or frame_w != dst_w:
            logger.warning(
                "Frame %d has shape (%d, %d) different from expected (%d, %d). Resizing.",
                index,
                frame_h,
                frame_w,
                dst_h,
                dst_w,
            )
            frame = cv2.resize(frame, (dst_w, dst_h))
        annotated_frame = frame.copy()

        if index < len(sequence) and sequence[index] is not None:
            frame_landmarks = sequence[index]
            def _coord(lm, k):
                try:
                    return lm.get(k) if isinstance(lm, dict) else lm[k]
                except Exception:
                    return None

            if all(
                lm is None or _coord(lm, "x") is None or not np.isfinite(_coord(lm, "x"))
                for lm in frame_landmarks
            ):
                writer.write(annotated_frame)
                continue

            points_to_draw = {}
            crop_box = None
            if crop_boxes is not None and index < len(crop_boxes):
                raw_crop = crop_boxes[index]
                if raw_crop is not None:
                    crop_arr = np.asarray(raw_crop, dtype=float).reshape(-1)
                    if crop_arr.size >= 4 and not np.isnan(crop_arr[:4]).all():
                        crop_box = _normalize_crop_box(crop_arr[:4], proc_w, proc_h)

            for landmark_index, landmark in enumerate(frame_landmarks):
                if landmark is None:
                    continue
                x_value = _coord(landmark, "x")
                y_value = _coord(landmark, "y")
                if x_value is None or y_value is None:
                    continue
                if not np.isfinite(x_value) or not np.isfinite(y_value):
                    continue

                if landmarks_normalized:
                    if crop_box is not None:
                        crop_x, crop_y, crop_w, crop_h = crop_box
                        x_proc = crop_x + x_value * crop_w
                        y_proc = crop_y + y_value * crop_h
                    else:
                        x_proc = x_value * proc_w
                        y_proc = y_value * proc_h
                else:
                    x_proc = x_value
                    y_proc = y_value

                x_draw = int(round(x_proc * scale_x))
                y_draw = int(round(y_proc * scale_y))

                points_to_draw[landmark_index] = (x_draw, y_draw)

            for start_idx, end_idx in config.POSE_CONNECTIONS:
                if start_idx in points_to_draw and end_idx in points_to_draw:
                    cv2.line(
                        annotated_frame,
                        points_to_draw[start_idx],
                        points_to_draw[end_idx],
                        config.CONNECTION_COLOR,
                        2,
                    )
            for point in points_to_draw.values():
                cv2.circle(annotated_frame, point, 4, config.LANDMARK_COLOR, -1)

        writer.write(annotated_frame)

    writer.release()
    logger.info("HQ debug video rendered successfully.")

    if clear_rotation_tag:
        logger.debug(
            "clear_rotation_tag requested, but OpenCV VideoWriter does not embed rotation metadata."
        )


def _normalize_crop_box(
    crop_values: np.ndarray, proc_w: float, proc_h: float
) -> tuple[float, float, float, float] | None:
    """Interpret crop box coordinates as (x, y, w, h) within the processed space."""

    if crop_values.size < 4:
        return None

    cx, cy, third, fourth = map(float, crop_values[:4])

    def _is_within(x: float, limit: float) -> bool:
        return -1e-3 <= x <= limit + 1e-3

    # Candidate 1: (x1, y1, x2, y2)
    width_candidate = third - cx
    height_candidate = fourth - cy
    candidate1_valid = width_candidate > 0 and height_candidate > 0
    if proc_w and proc_h:
        candidate1_valid = (
            candidate1_valid
            and _is_within(cx, proc_w)
            and _is_within(cy, proc_h)
            and _is_within(cx + width_candidate, proc_w)
            and _is_within(cy + height_candidate, proc_h)
        )

    # Candidate 2: (x, y, width, height)
    width_candidate2 = third
    height_candidate2 = fourth
    candidate2_valid = width_candidate2 > 0 and height_candidate2 > 0
    if proc_w and proc_h:
        candidate2_valid = (
            candidate2_valid
            and _is_within(cx, proc_w)
            and _is_within(cy, proc_h)
            and _is_within(cx + width_candidate2, proc_w)
            and _is_within(cy + height_candidate2, proc_h)
        )

    if candidate1_valid and not candidate2_valid:
        return cx, cy, width_candidate, height_candidate
    if candidate2_valid and not candidate1_valid:
        return cx, cy, width_candidate2, height_candidate2
    if candidate1_valid:
        return cx, cy, width_candidate, height_candidate
    if candidate2_valid:
        return cx, cy, width_candidate2, height_candidate2

    return None
