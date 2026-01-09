import numpy as np
import pandas as pd

from src.ui.metrics_catalog import is_user_facing_metric
from src.ui.steps.results.results import _prepare_metrics_df_for_display


def _metric_options_from_df(metrics_df: pd.DataFrame) -> list[str]:
    numeric_candidates = [
        c
        for c in metrics_df.columns
        if metrics_df[c].dtype.kind in "fi" and c != "frame_idx"
    ]
    return [c for c in numeric_candidates if is_user_facing_metric(c)]


def test_metric_filter_for_selector() -> None:
    metrics_df = pd.DataFrame(
        {
            "analysis_frame_idx": [1, 2],
            "source_frame_idx": [1, 2],
            "source_time_s": [0.0, 0.1],
            "time_s": [0.0, 0.1],
            "pose_ok": [1, 0],
            "frame_idx": [1, 2],
            "left_knee": [10.0, 20.0],
            "right_knee": [11.0, 21.0],
            "trunk_inclination_deg": [5.0, 6.0],
        }
    )

    metric_options = _metric_options_from_df(metrics_df)

    assert "left_knee" in metric_options
    assert "right_knee" in metric_options
    assert "trunk_inclination_deg" in metric_options
    assert "analysis_frame_idx" not in metric_options
    assert "source_frame_idx" not in metric_options
    assert "source_time_s" not in metric_options
    assert "time_s" not in metric_options
    assert "pose_ok" not in metric_options
    assert "frame_idx" not in metric_options


def test_prepare_metrics_df_for_display_normalizes_low_scale_trunk() -> None:
    metrics_df = pd.DataFrame(
        {"trunk_inclination_deg": [0.0, 5.0, np.nan, 10.0]}
    )
    metrics_df_original = metrics_df.copy()

    display_df = _prepare_metrics_df_for_display(metrics_df)

    expected = 90.0 - metrics_df_original["trunk_inclination_deg"]
    pd.testing.assert_series_equal(
        display_df["trunk_inclination_deg"], expected, check_names=False
    )
    pd.testing.assert_frame_equal(metrics_df, metrics_df_original)


def test_prepare_metrics_df_for_display_keeps_high_scale_trunk() -> None:
    metrics_df = pd.DataFrame(
        {"trunk_inclination_deg": [80.0, 85.0, 90.0]}
    )
    metrics_df_original = metrics_df.copy()

    display_df = _prepare_metrics_df_for_display(metrics_df)

    pd.testing.assert_frame_equal(display_df, metrics_df_original)
    pd.testing.assert_frame_equal(metrics_df, metrics_df_original)
