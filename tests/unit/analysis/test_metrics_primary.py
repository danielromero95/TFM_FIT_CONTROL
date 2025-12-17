"""Pruebas para la selección del ángulo principal en vista frontal de peso muerto."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.C_analysis import metrics


def test_frontal_deadlift_prefers_bilateral_hips_mean() -> None:
    df = pd.DataFrame(
        {
            "left_hip": [10.0, 20.0, np.nan, 30.0],
            "right_hip": [20.0, np.nan, 30.0, 40.0],
            "pose_ok": [1.0, 1.0, 1.0, 1.0],
        }
    )

    metrics._maybe_add_bilateral_hip_mean(df, "deadlift", "front")
    chosen = metrics.choose_primary_angle("deadlift", "front", df)

    assert "hips_mean" in df.columns
    assert chosen == "hips_mean"
    assert df["hips_mean"].tolist() == [15.0, 20.0, 30.0, 35.0]


def test_side_deadlift_keeps_single_sided_choice() -> None:
    df = pd.DataFrame({
        "left_hip": [10.0, 20.0, 30.0],
        "right_hip": [30.0, 25.0, 20.0],
        "pose_ok": [1.0, 1.0, 1.0],
    })

    metrics._maybe_add_bilateral_hip_mean(df, "deadlift", "lateral")
    chosen = metrics.choose_primary_angle("deadlift", "lateral", df)

    assert chosen in {"left_hip", "right_hip"}
    assert "hips_mean" not in df.columns


def test_frontal_deadlift_prefers_hips_mean_with_raw_values() -> None:
    df = pd.DataFrame(
        {
            "left_hip": [np.nan, 15.0, np.nan, 25.0],
            "right_hip": [12.0, np.nan, 28.0, 32.0],
            "raw_left_hip": [np.nan, 15.0, 20.0, np.nan],
            "raw_right_hip": [10.0, 20.0, np.nan, 30.0],
            "pose_ok": [1.0, 1.0, 1.0, 1.0],
        }
    )

    metrics._maybe_add_bilateral_hip_mean(df, "deadlift", "front")
    chosen = metrics.choose_primary_angle("deadlift", "front", df)

    assert "raw_hips_mean" in df.columns
    assert "hips_mean" in df.columns
    assert chosen == "hips_mean"
