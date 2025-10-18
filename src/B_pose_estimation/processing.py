"""Pose landmark extraction and biomechanical metric utilities."""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .estimators import CroppedPoseEstimator, PoseEstimator
from .metrics import (
    calculate_angular_velocity,
    calculate_distances,
    calculate_symmetry,
    extract_joint_angles,
    normalize_landmarks,
)

logger = logging.getLogger(__name__)


def extract_landmarks_from_frames(
    frames: Iterable[np.ndarray],
    use_crop: bool = False,
    visibility_threshold: float = 0.5,
) -> pd.DataFrame:
    """Extract pose landmarks frame by frame and return a raw DataFrame."""
    frames = list(frames)
    logger.info("Extracting landmarks from %d frames. Using crop: %s", len(frames), use_crop)
    estimator = (
        CroppedPoseEstimator(min_detection_confidence=visibility_threshold)
        if use_crop
        else PoseEstimator(min_detection_confidence=visibility_threshold)
    )

    rows: list[dict[str, float]] = []
    for index, image in enumerate(frames):
        crop_box = None
        if use_crop:
            landmarks, _, crop_box = estimator.estimate(image)
        else:
            landmarks, _ = estimator.estimate(image)
            # When no crop is used the crop box corresponds to the full image.
            # This ensures the renderer always receives bounding box metadata.
            if landmarks:
                height, width, _ = image.shape
                crop_box = [0, 0, width, height]

        row: dict[str, float] = {"frame_idx": index}
        if landmarks:
            for landmark_index, point in enumerate(landmarks):
                row.update(
                    {
                        f"x{landmark_index}": point["x"],
                        f"y{landmark_index}": point["y"],
                        f"z{landmark_index}": point["z"],
                        f"v{landmark_index}": point["visibility"],
                    }
                )

            if crop_box:
                row.update(
                    {
                        "crop_x1": crop_box[0],
                        "crop_y1": crop_box[1],
                        "crop_x2": crop_box[2],
                        "crop_y2": crop_box[3],
                    }
                )
        else:
            for landmark_index in range(33):
                row.update(
                    {
                        f"x{landmark_index}": np.nan,
                        f"y{landmark_index}": np.nan,
                        f"z{landmark_index}": np.nan,
                        f"v{landmark_index}": np.nan,
                    }
                )
        rows.append(row)

    estimator.close()
    return pd.DataFrame(rows)


def filter_and_interpolate_landmarks(
    df_raw: pd.DataFrame, min_confidence: float = 0.5
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Filter landmarks below ``min_confidence`` and interpolate gaps."""
    logger.info("Filtering and interpolating %d landmark frames.", len(df_raw))
    n_frames, n_points = len(df_raw), 33
    arr = np.full((n_frames, n_points, 4), np.nan, dtype=float)

    for time_index, (_, row) in enumerate(df_raw.iterrows()):
        for point_index in range(n_points):
            visibility = row.get(f"v{point_index}", np.nan)
            if pd.notna(visibility) and visibility >= min_confidence:
                arr[time_index, point_index, 0] = row.get(f"x{point_index}")
                arr[time_index, point_index, 1] = row.get(f"y{point_index}")
                arr[time_index, point_index, 2] = row.get(f"z{point_index}")
                arr[time_index, point_index, 3] = visibility

    for point_index in range(n_points):
        valid_mask = ~np.isnan(arr[:, point_index, 0])
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 1:
            interp_indices = np.arange(n_frames)
            for axis in range(3):
                arr[:, point_index, axis] = np.interp(
                    interp_indices, valid_indices, arr[valid_indices, point_index, axis]
                )

    filtered_sequence = []
    for time_index in range(n_frames):
        frame_landmarks = [
            {
                "x": arr[time_index, point_index, 0],
                "y": arr[time_index, point_index, 1],
                "z": arr[time_index, point_index, 2],
                "visibility": arr[time_index, point_index, 3]
                if pd.notna(arr[time_index, point_index, 3])
                else 0.0,
            }
            for point_index in range(n_points)
        ]
        filtered_sequence.append(frame_landmarks)

    crop_coords = (
        df_raw[["crop_x1", "crop_y1", "crop_x2", "crop_y2"]].to_numpy()
        if "crop_x1" in df_raw.columns
        else None
    )
    return np.array(filtered_sequence, dtype=object), crop_coords


def calculate_metrics_from_sequence(sequence: np.ndarray, fps: float) -> pd.DataFrame:
    """Compute biomechanical metrics for the provided landmark sequence."""
    logger.info("Computing metrics for a sequence of %d frames.", len(sequence))
    all_metrics: list[dict[str, float]] = []
    for index, frame_landmarks in enumerate(sequence):
        row: dict[str, float] = {"frame_idx": index}
        if frame_landmarks is None or any(np.isnan(lm["x"]) for lm in frame_landmarks):
            row.update(
                {
                    "left_knee": np.nan,
                    "right_knee": np.nan,
                    "left_elbow": np.nan,
                    "right_elbow": np.nan,
                    "shoulder_width": np.nan,
                    "foot_separation": np.nan,
                }
            )
        else:
            norm_landmarks = normalize_landmarks(frame_landmarks)
            angles = extract_joint_angles(norm_landmarks)
            distances = calculate_distances(norm_landmarks)
            row.update(angles)
            row.update(distances)
        all_metrics.append(row)

    dfm = pd.DataFrame(all_metrics)
    if dfm.empty:
        return dfm

    dfm_filled = dfm.ffill().bfill()
    for column in ["left_knee", "right_knee", "left_elbow", "right_elbow"]:
        dfm[f"ang_vel_{column}"] = calculate_angular_velocity(dfm_filled[column].tolist(), fps)

    dfm["knee_symmetry"] = dfm.apply(
        lambda row: calculate_symmetry(row["left_knee"], row["right_knee"]), axis=1
    )
    dfm["elbow_symmetry"] = dfm.apply(
        lambda row: calculate_symmetry(row["left_elbow"], row["right_elbow"]), axis=1
    )
    return dfm
