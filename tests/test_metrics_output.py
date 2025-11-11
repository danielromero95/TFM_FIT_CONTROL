"""Smoke tests for the metrics CSV schema using synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

EXPECTED_COLUMNS = {
    "frame_idx",
    "left_knee",
    "right_knee",
    "left_elbow",
    "right_elbow",
    "shoulder_width",
    "foot_separation",
    "ang_vel_left_knee",
    "ang_vel_right_knee",
    "ang_vel_left_elbow",
    "ang_vel_right_elbow",
    "knee_symmetry",
    "elbow_symmetry",
}


@pytest.fixture
def metrics_csv(tmp_path):
    """Create a representative metrics CSV for portable regression tests."""

    rows = [{col: float(idx) for col in EXPECTED_COLUMNS} for idx in range(5)]
    df = pd.DataFrame(rows)
    path = tmp_path / "metrics.csv"
    df.to_csv(path, index=False)
    return path


def test_metrics_csv_exists(metrics_csv):
    """Verify that the synthetic metrics CSV fixture is created."""

    assert metrics_csv.exists(), f"Metrics file not created: {metrics_csv}"


def test_metrics_csv_columns(metrics_csv):
    """Ensure the metrics CSV exposes the expected minimum set of columns."""

    df = pd.read_csv(metrics_csv)
    actual_cols = set(df.columns)
    missing = EXPECTED_COLUMNS - actual_cols
    assert not missing, f"Missing columns in metrics CSV: {missing}"
    assert len(df) == 5, "Synthetic metrics CSV size changed unexpectedly."
