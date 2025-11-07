"""Pose landmark extraction pipeline stage."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence, Type

import numpy as np
import pandas as pd

from src.config.constants import MIN_DETECTION_CONFIDENCE

from ..constants import LANDMARK_COUNT
from ..estimators import CroppedPoseEstimator, PoseEstimator, PoseEstimatorBase, RoiPoseEstimator
from ..types import PoseResult

logger = logging.getLogger(__name__)


def extract_landmarks_from_frames(
    frames: Iterable[np.ndarray],
    use_crop: bool = False,
    *,
    use_roi_tracking: bool = False,
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
    min_visibility: float = 0.5,
) -> pd.DataFrame:
    """Extract pose landmarks frame by frame and return a raw DataFrame."""

    logger.info("Extracting landmarks from streaming frames. Using crop: %s", use_crop)
    estimator_cls: Type[PoseEstimatorBase]
    if use_crop and use_roi_tracking:
        estimator_cls = RoiPoseEstimator
    elif use_crop:
        estimator_cls = CroppedPoseEstimator
    else:
        estimator_cls = PoseEstimator

    rows: list[dict[str, float]] = []
    with estimator_cls(min_detection_confidence=min_detection_confidence) as estimator:
        for index, image in enumerate(frames):
            height, width = image.shape[:2]
            result: PoseResult = estimator.estimate(image)
            landmarks = result.landmarks
            crop_box = result.crop_box
            if landmarks and crop_box is None:
                crop_box = (0, 0, width, height)

            row: dict[str, float] = {"frame_idx": int(index)}
            if landmarks:
                crop_values: Sequence[float]
                if crop_box is None:
                    crop_values = (0.0, 0.0, float(width), float(height))
                else:
                    crop_values = tuple(float(v) for v in crop_box)
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_values
                crop_width = max(crop_x2 - crop_x1, 0.0)
                crop_height = max(crop_y2 - crop_y1, 0.0)

                for landmark_index, point in enumerate(landmarks):
                    visibility = float(point.get("visibility", np.nan))
                    x_value = float(point.get("x", np.nan))
                    y_value = float(point.get("y", np.nan))

                    if use_crop and not use_roi_tracking:
                        if crop_width > 0.0 and crop_height > 0.0:
                            x_value = (x_value * crop_width + crop_x1) / float(width)
                            y_value = (y_value * crop_height + crop_y1) / float(height)
                        else:
                            x_value = np.nan
                            y_value = np.nan

                    if visibility < min_visibility:
                        x_value = np.nan
                        y_value = np.nan

                    row[f"x{landmark_index}"] = x_value
                    row[f"y{landmark_index}"] = y_value
                    row[f"z{landmark_index}"] = float(point.get("z", np.nan))
                    row[f"v{landmark_index}"] = visibility

                row.update(
                    {
                        "crop_x1": float(crop_x1),
                        "crop_y1": float(crop_y1),
                        "crop_x2": float(crop_x2),
                        "crop_y2": float(crop_y2),
                    }
                )
            else:
                for landmark_index in range(LANDMARK_COUNT):
                    row[f"x{landmark_index}"] = np.nan
                    row[f"y{landmark_index}"] = np.nan
                    row[f"z{landmark_index}"] = np.nan
                    row[f"v{landmark_index}"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)


__all__ = ["extract_landmarks_from_frames"]
