from __future__ import annotations

import numpy as np

from exercise_detection.metrics import compute_metrics
from exercise_detection.types import RepSlice


def test_arms_above_hip_fraction_uses_fallback_scale_when_torso_nan() -> None:
    frames = 10
    series = {
        "knee_angle": np.linspace(170.0, 140.0, frames),
        "hip_angle": np.linspace(160.0, 120.0, frames),
        "elbow_angle": np.full(frames, 100.0),
        "torso_tilt": np.linspace(10.0, 20.0, frames),
        "wrist_y": np.full(frames, 0.4),
        "wrist_x": np.full(frames, 0.3),
        "shoulder_y": np.full(frames, 0.3),
        "hip_y": np.full(frames, 0.6),
        "hip_left_y": np.full(frames, 0.6),
        "hip_right_y": np.full(frames, 0.6),
        "knee_x": np.full(frames, 0.2),
        "knee_y": np.full(frames, 0.7),
        "ankle_x": np.full(frames, 0.2),
        "ankle_y": np.full(frames, 0.9),
        "bar_y": np.full(frames, 0.4),
        "bar_x": np.full(frames, 0.3),
        "arm_y": np.full(frames, 0.4),
        "shoulder_left_y": np.full(frames, 0.3),
        "shoulder_right_y": np.full(frames, 0.3),
    }
    rep_slices = [RepSlice(start=0, end=frames)]

    metrics = compute_metrics(rep_slices, series, float("nan"), sampling_rate=30.0)

    assert metrics.rep_count == 1
    assert np.isfinite(metrics.arms_above_hip_fraction)
