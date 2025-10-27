"""Pose landmark extraction and biomechanical metric utilities."""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .estimators import CroppedPoseEstimator, PoseEstimator, RoiPoseEstimator
from .metrics import (
    angular_velocity,
    calculate_distances,
    calculate_symmetry,
    extract_joint_angles,
    normalize_landmarks,
)
from src.config.constants import MIN_DETECTION_CONFIDENCE

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
    estimator_cls = (
        RoiPoseEstimator
        if (use_crop and use_roi_tracking)
        else (CroppedPoseEstimator if use_crop else PoseEstimator)
    )

    rows: list[dict[str, float]] = []
    with estimator_cls(min_detection_confidence=min_detection_confidence) as estimator:
        for index, image in enumerate(frames):
            height, width = image.shape[0], image.shape[1]
            landmarks, _annotated, crop_box = estimator.estimate(image)
            if landmarks and crop_box is None:
                crop_box = [0, 0, width, height]

            row: dict[str, float] = {"frame_idx": index}
            if landmarks:
                crop_values = crop_box if crop_box else [0, 0, width, height]
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_values
                crop_width = max(crop_x2 - crop_x1, 0)
                crop_height = max(crop_y2 - crop_y1, 0)

                for landmark_index, point in enumerate(landmarks):
                    visibility = point["visibility"]
                    x_value = point["x"]
                    y_value = point["y"]

                    if use_crop and not use_roi_tracking:
                        if crop_width > 0 and crop_height > 0:
                            x_value = (x_value * crop_width + crop_x1) / width
                            y_value = (y_value * crop_height + crop_y1) / height
                        else:
                            x_value = np.nan
                            y_value = np.nan

                    if visibility < min_visibility:
                        x_value = np.nan
                        y_value = np.nan

                    row.update(
                        {
                            f"x{landmark_index}": x_value,
                            f"y{landmark_index}": y_value,
                            f"z{landmark_index}": point["z"],
                            f"v{landmark_index}": visibility,
                        }
                    )

                row.update(
                    {
                        "crop_x1": float(crop_x1),
                        "crop_y1": float(crop_y1),
                        "crop_x2": float(crop_x2),
                        "crop_y2": float(crop_y2),
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

    return pd.DataFrame(rows)


def filter_and_interpolate_landmarks(
    df_raw: pd.DataFrame, min_confidence: float = 0.5
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Filter landmarks below ``min_confidence`` and interpolate gaps."""
    logger.info("Filtering and interpolating %d landmark frames.", len(df_raw))
    n_frames, n_points = len(df_raw), 33

    crop_coords = (
        df_raw[["crop_x1", "crop_y1", "crop_x2", "crop_y2"]].to_numpy()
        if "crop_x1" in df_raw.columns
        else None
    )
    if n_frames == 0:
        return np.array([], dtype=object), crop_coords

    cols_x = [f"x{idx}" for idx in range(n_points)]
    cols_y = [f"y{idx}" for idx in range(n_points)]
    cols_z = [f"z{idx}" for idx in range(n_points)]
    cols_v = [f"v{idx}" for idx in range(n_points)]

    xs = df_raw[cols_x].to_numpy(dtype=float, copy=True)
    ys = df_raw[cols_y].to_numpy(dtype=float, copy=True)
    zs = df_raw[cols_z].to_numpy(dtype=float, copy=True)
    vs = df_raw[cols_v].to_numpy(dtype=float, copy=False)

    mask_valid = (vs >= min_confidence) & np.isfinite(xs) & np.isfinite(ys)
    xs[~mask_valid] = np.nan
    ys[~mask_valid] = np.nan
    zs[~mask_valid] = np.nan

    def _interp_columns(arr: np.ndarray) -> np.ndarray:
        """Interpolate each landmark column independently."""
        out = arr.copy()
        if out.size == 0:
            return out
        idx = np.arange(out.shape[0])
        for column_index in range(out.shape[1]):
            column = out[:, column_index]
            valid = np.isfinite(column)
            if valid.sum() >= 2:
                out[:, column_index] = np.interp(idx, idx[valid], column[valid])
        return out

    xs_interp = _interp_columns(xs)
    ys_interp = _interp_columns(ys)
    zs_interp = _interp_columns(zs)

    vis_output = np.where(np.isfinite(vs), vs, 0.0)

    def _build_landmark(x: float, y: float, z: float, v: float) -> dict[str, float]:
        return {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "visibility": float(v),
        }

    vectorized_builder = np.vectorize(_build_landmark, otypes=[object])
    landmarks_array = vectorized_builder(xs_interp, ys_interp, zs_interp, vis_output)
    frames_list = landmarks_array.tolist()
    filtered_sequence = np.empty(n_frames, dtype=object)
    filtered_sequence[:] = frames_list
    return filtered_sequence, crop_coords


def calculate_metrics_from_sequence(
    sequence: np.ndarray,
    fps: float,
    *,
    smooth_window: int = 0,
    vel_method: str = "forward",
) -> pd.DataFrame:
    """Compute biomechanical metrics for the provided landmark sequence."""
    logger.info("Computing metrics for a sequence of %d frames.", len(sequence))
    all_metrics: list[dict[str, float]] = []
    for index, frame_landmarks in enumerate(sequence):
        row: dict[str, float] = {"frame_idx": index}
        if frame_landmarks is None:
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

    angle_columns = ["left_knee", "right_knee", "left_elbow", "right_elbow"]
    raw_angles = dfm[angle_columns].copy()
    for column in angle_columns:
        dfm[column] = dfm[column].interpolate(method="linear", limit_direction="both")
        if smooth_window >= 3:
            dfm[column] = dfm[column].rolling(smooth_window, center=True, min_periods=1).mean()
        dfm[f"ang_vel_{column}"] = angular_velocity(
            dfm[column].tolist(), fps, method=vel_method
        )

    dfm["knee_symmetry"] = raw_angles.apply(
        lambda row: calculate_symmetry(row["left_knee"], row["right_knee"]), axis=1
    )
    dfm["elbow_symmetry"] = raw_angles.apply(
        lambda row: calculate_symmetry(row["left_elbow"], row["right_elbow"]), axis=1
    )
    return dfm
