"""High-quality video renderer for drawing pose landmarks."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src import config
from src.config.settings import DEFAULT_TARGET_HEIGHT, DEFAULT_TARGET_WIDTH

logger = logging.getLogger(__name__)


def render_landmarks_on_video_hq(
    original_frames: list,
    landmarks_sequence: np.ndarray,
    crop_boxes: np.ndarray,
    output_path: str,
    fps: float,
) -> None:
    """Draw landmarks onto original frames and persist a high-quality video."""
    logger.info("Starting HQ video rendering at: %s", output_path)
    if not original_frames:
        logger.warning("No frames available to render.")
        return

    orig_h, orig_w, _ = original_frames[0].shape
    proc_w, proc_h = DEFAULT_TARGET_WIDTH, DEFAULT_TARGET_HEIGHT

    # Scaling factors from the 256x256 processed space to the original resolution.
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (orig_w, orig_h))
    if not writer.isOpened():
        logger.error("Could not open VideoWriter for path: %s", output_path)
        return

    for index, frame in enumerate(original_frames):
        annotated_frame = frame.copy()

        if index < len(landmarks_sequence) and landmarks_sequence[index] is not None:
            frame_landmarks = landmarks_sequence[index]
            if all(np.isnan(landmark["x"]) for landmark in frame_landmarks):
                writer.write(annotated_frame)
                continue

            points_to_draw = {}
            crop_box = (
                crop_boxes[index]
                if crop_boxes is not None and index < len(crop_boxes) and not np.isnan(crop_boxes[index]).all()
                else None
            )

            for landmark_index, landmark in enumerate(frame_landmarks):
                if np.isnan(landmark["x"]):
                    continue

                # --- Correct coordinate transformation ---
                if crop_box is not None:
                    # Crop present: convert from [0, 1] relative to crop into processed-image pixels.
                    x1_p, y1_p, x2_p, y2_p = crop_box
                    crop_w_p = x2_p - x1_p
                    crop_h_p = y2_p - y1_p

                    abs_x_p = x1_p + landmark["x"] * crop_w_p
                    abs_y_p = y1_p + landmark["y"] * crop_h_p

                    # Then scale from processed image to original resolution.
                    final_x = int(abs_x_p * scale_x)
                    final_y = int(abs_y_p * scale_y)
                else:
                    # No crop: landmarks are relative to the processed image, scale directly.
                    final_x = int(landmark["x"] * orig_w)
                    final_y = int(landmark["y"] * orig_h)

                points_to_draw[landmark_index] = (final_x, final_y)

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
