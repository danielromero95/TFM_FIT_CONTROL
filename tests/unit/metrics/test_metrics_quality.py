from __future__ import annotations

import numpy as np

from src.B_pose_estimation.constants import (
    LANDMARK_COUNT,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from src.B_pose_estimation.pipeline import calculate_metrics_from_sequence
from src.B_pose_estimation.types import Landmark


def _frame_with_offset(offset: float) -> list[Landmark]:
    base = [Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(LANDMARK_COUNT)]
    base[LEFT_SHOULDER] = Landmark(x=-0.1 + offset, y=0.2, z=0.0, visibility=1.0)
    base[RIGHT_SHOULDER] = Landmark(x=0.1 + offset, y=0.2, z=0.0, visibility=1.0)
    base[LEFT_HIP] = Landmark(x=-0.1 + offset, y=0.6, z=0.0, visibility=1.0)
    base[RIGHT_HIP] = Landmark(x=0.1 + offset, y=0.6, z=0.0, visibility=1.0)
    base[LEFT_KNEE] = Landmark(x=-0.1 + offset, y=0.9, z=0.0, visibility=1.0)
    base[RIGHT_KNEE] = Landmark(x=0.1 + offset, y=0.9, z=0.0, visibility=1.0)
    base[LEFT_ANKLE] = Landmark(x=-0.12 + offset, y=1.1, z=0.0, visibility=1.0)
    base[RIGHT_ANKLE] = Landmark(x=0.12 + offset, y=1.1, z=0.0, visibility=1.0)
    base[LEFT_ELBOW] = Landmark(x=-0.15 + offset, y=0.4, z=0.0, visibility=1.0)
    base[RIGHT_ELBOW] = Landmark(x=0.15 + offset, y=0.4, z=0.0, visibility=1.0)
    base[LEFT_WRIST] = Landmark(x=-0.18 + offset, y=0.5, z=0.0, visibility=1.0)
    base[RIGHT_WRIST] = Landmark(x=0.18 + offset, y=0.5, z=0.0, visibility=1.0)
    return base


def test_metrics_mask_outliers_and_remain_finite():
    sequence = [_frame_with_offset(0.0), _frame_with_offset(0.02), _frame_with_offset(0.5), _frame_with_offset(-0.02)]
    quality_mask = [True, True, False, True]

    metrics = calculate_metrics_from_sequence(sequence, fps=30.0, quality_mask=quality_mask)

    assert metrics["pose_ok"].sum() == 3
    assert np.isfinite(metrics.loc[metrics["pose_ok"] > 0.5, "trunk_inclination_deg"]).all()
    assert metrics.loc[metrics["pose_ok"] <= 0.5, "trunk_inclination_deg"].isna().all()
    assert metrics[["left_knee", "right_knee", "left_elbow", "right_elbow"]].max().max() < 200
