import pandas as pd

from src.pipeline_data import RunStats
from src.ui.steps.results.results import _build_rep_speed_chart_df, _compute_rep_speeds


def _dummy_stats() -> RunStats:
    return RunStats(
        config_sha1="x",
        fps_original=30.0,
        fps_effective=30.0,
        frames=10,
        exercise_selected=None,
        exercise_detected="squat",
        view_detected="front",
        view_selected=None,
        detection_confidence=1.0,
        primary_angle="knee",
        angle_range_deg=0.0,
        min_prominence=0.0,
        min_distance_sec=0.0,
        refractory_sec=0.0,
    )


def test_rep_speed_only_counts_completed_and_accepted_candidates():
    stats = _dummy_stats()
    frame_values = pd.Series([0, 1, 2, 3])
    metrics_df = pd.DataFrame({"frame_idx": frame_values, "knee": [0.0, 1.0, 0.5, 1.5]})

    rep_candidates = [
        {"start_frame": 0, "end_frame": 2, "accepted": True},
        {"start_frame": 2, "end_frame": None, "accepted": True},
        {"start_frame": 3, "end_frame": 4, "accepted": False},
    ]

    rep_speeds_df = _compute_rep_speeds(
        [],
        stats,
        exercise_key="squat",
        valley_frames=[],
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric="knee",
        rep_candidates=rep_candidates,
    )

    assert rep_speeds_df["Repetition"].tolist() == [1]
    assert rep_speeds_df["Accepted"].tolist() == [True]

    rep_chart_df = _build_rep_speed_chart_df(rep_speeds_df, ("Down", "Up"))
    assert sorted(rep_chart_df["Repetition"].unique().tolist()) == [1]
    assert len(rep_chart_df) == 2
