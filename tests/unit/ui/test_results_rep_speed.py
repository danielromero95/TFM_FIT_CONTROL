from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.pipeline_data import Report, RunStats

from src.ui.steps.results.results import (
    _build_rep_speed_chart_df,
    _compute_rep_intervals,
    _compute_rep_speeds,
    phase_order_for_exercise,
)


def _dummy_stats(primary_angle: str = "angle", fps: float = 30.0, exercise: str = "deadlift") -> RunStats:
    return RunStats(
        config_sha1="sha1",
        fps_original=fps,
        fps_effective=fps,
        frames=0,
        exercise_selected=None,
        exercise_detected=exercise,
        view_detected="side",
        detection_confidence=1.0,
        primary_angle=primary_angle,
        angle_range_deg=0.0,
        min_prominence=0.0,
        min_distance_sec=0.0,
        refractory_sec=0.0,
    )


def _dummy_report(stats: RunStats, exercise: str, repetitions: int, metrics: pd.DataFrame) -> Report:
    config_used = SimpleNamespace(counting=None, faults=None)
    return Report(
        repetitions=repetitions,
        metrics=metrics,
        stats=stats,
        config_used=config_used,
    )


def test_phase_order_for_exercise_mapping():
    assert phase_order_for_exercise("deadlift") == ("Up", "Down")
    assert phase_order_for_exercise("squat") == ("Down", "Up")
    assert phase_order_for_exercise("bench_press") == ("Down", "Up")
    assert phase_order_for_exercise(None) == ("Down", "Up")


def test_rep_speed_chart_uses_deadlift_phase_order_first():
    rep_speeds_df = pd.DataFrame(
        [
            {
                "Repetition": 1,
                "Down speed (deg/s)": 1.0,
                "Down duration (s)": 0.5,
                "Up speed (deg/s)": 2.0,
                "Up duration (s)": 0.4,
                "Cadence (reps/min)": 60.0,
            }
        ]
    )

    chart_df = _build_rep_speed_chart_df(rep_speeds_df, ("Up", "Down"))

    assert list(chart_df["Phase"]) == ["Up", "Down"]
    assert chart_df.iloc[0]["Speed"] == 2.0
    assert chart_df.iloc[0]["Phase duration (s)"] == 0.4
    assert chart_df.iloc[1]["Speed"] == 1.0
    assert chart_df.iloc[1]["Phase duration (s)"] == 0.5


def test_rep_speed_chart_default_phase_mapping():
    rep_speeds_df = pd.DataFrame(
        [
            {
                "Repetition": 1,
                "Down speed (deg/s)": 1.0,
                "Down duration (s)": 0.5,
                "Up speed (deg/s)": 2.0,
                "Up duration (s)": 0.4,
                "Cadence (reps/min)": 60.0,
            }
        ]
    )

    chart_df = _build_rep_speed_chart_df(rep_speeds_df, ("Down", "Up"))

    assert list(chart_df["Phase"]) == ["Down", "Up"]
    assert chart_df.iloc[0]["Speed"] == 1.0
    assert chart_df.iloc[0]["Phase duration (s)"] == 0.5
    assert chart_df.iloc[1]["Speed"] == 2.0
    assert chart_df.iloc[1]["Phase duration (s)"] == 0.4


def test_deadlift_rep_intervals_and_speeds_cover_all_phases():
    angles = []
    for _ in range(3):
        up = np.linspace(0.0, 10.0, 6)
        down = np.linspace(10.0, 0.0, 5)[1:]
        angles.extend(up)
        angles.extend(down)

    frame_idx = np.arange(len(angles))
    metrics_df = pd.DataFrame({"frame_idx": frame_idx, "angle": angles})
    stats = _dummy_stats(exercise="deadlift")
    report = _dummy_report(stats, "deadlift", repetitions=3, metrics=metrics_df)

    rep_intervals, valley_frames, frame_values, primary_metric = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="deadlift",
    )

    assert len(rep_intervals) > 0
    if len(rep_intervals) < report.repetitions:
        assert getattr(report.stats, "counting_accuracy_warning", None)

    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="deadlift",
        valley_frames=valley_frames,
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric=primary_metric,
    )

    assert rep_speeds_df.shape[0] == len(rep_intervals)
    assert rep_speeds_df[["Down duration (s)", "Up duration (s)"]].isna().sum().sum() == 0

    chart_df = _build_rep_speed_chart_df(
        rep_speeds_df, phase_order_for_exercise("deadlift")
    )
    assert len(chart_df) == 2 * len(rep_intervals)
    assert set(chart_df["Phase"]) == {"Up", "Down"}
    assert chart_df["Repetition"].nunique() == len(rep_intervals)


def test_squat_phase_mapping_preserved():
    metrics_df = pd.DataFrame({"frame_idx": [0, 1, 2], "angle": [10.0, 0.0, 10.0]})
    stats = _dummy_stats(exercise="squat")
    report = _dummy_report(stats, "squat", repetitions=1, metrics=metrics_df)

    rep_intervals, valley_frames, frame_values, primary_metric = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="squat",
    )

    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="squat",
        valley_frames=valley_frames,
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric=primary_metric,
    )

    assert list(rep_speeds_df.columns).index("Down duration (s)") < list(
        rep_speeds_df.columns
    ).index("Up duration (s)")
    assert rep_speeds_df.iloc[0]["Down speed (deg/s)"] > 0
    assert rep_speeds_df.iloc[0]["Up speed (deg/s)"] > 0


def test_squat_midpoint_valley_intervals_fill_rep_speed_chart():
    angles: list[float] = []
    for _ in range(5):
        angles.extend([10.0, 5.0, 0.0, 5.0])

    frame_idx = np.arange(len(angles))
    metrics_df = pd.DataFrame({"frame_idx": frame_idx, "angle": angles})
    stats = _dummy_stats(exercise="squat")
    report = _dummy_report(stats, "squat", repetitions=5, metrics=metrics_df)

    rep_intervals, valley_frames, frame_values, primary_metric = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="squat",
    )

    assert len(valley_frames) == 5
    assert len(rep_intervals) == report.repetitions

    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="squat",
        valley_frames=valley_frames,
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric=primary_metric,
    )

    chart_df = _build_rep_speed_chart_df(
        rep_speeds_df, phase_order_for_exercise("squat")
    )

    assert chart_df.shape[0] == 2 * report.repetitions
    assert chart_df["Phase"].value_counts().to_dict() == {
        "Down": report.repetitions,
        "Up": report.repetitions,
    }
    assert chart_df["Repetition"].nunique() == report.repetitions
    assert not chart_df[["Speed", "Phase duration (s)"]].isna().any().any()


def test_squat_midpoint_fallback_when_debug_intervals_undercount(monkeypatch):
    angles: list[float] = []
    for _ in range(5):
        angles.extend([10.0, 5.0, 0.0, 5.0])

    valley_indices = [2, 6, 10, 14, 18]
    debug = SimpleNamespace(
        valley_indices=valley_indices,
        rep_intervals=list(zip(valley_indices[:-1], valley_indices[1:])),
    )

    def _fake_count_reps(metrics_df, counting_cfg, fps, faults_cfg=None):
        return len(valley_indices), debug

    monkeypatch.setattr(
        "src.ui.steps.results.results.count_repetitions_with_config", _fake_count_reps
    )

    frame_idx = np.arange(len(angles))
    metrics_df = pd.DataFrame({"frame_idx": frame_idx, "angle": angles})
    stats = _dummy_stats(exercise="squat")
    report = _dummy_report(stats, "squat", repetitions=5, metrics=metrics_df)

    rep_intervals, valley_frames, frame_values, primary_metric = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="squat",
    )

    assert len(valley_frames) == 5
    assert len(rep_intervals) == report.repetitions

    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="squat",
        valley_frames=valley_frames,
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric=primary_metric,
    )

    chart_df = _build_rep_speed_chart_df(
        rep_speeds_df, phase_order_for_exercise("squat")
    )

    assert chart_df.shape[0] == 2 * report.repetitions
    assert chart_df["Repetition"].nunique() == report.repetitions
    assert not chart_df[["Speed", "Phase duration (s)"]].isna().any().any()


def test_deadlift_with_boundary_nans_keeps_all_phases():
    angles = []
    for _ in range(3):
        up = np.linspace(0.0, 10.0, 6)
        down = np.linspace(10.0, 0.0, 5)[1:]
        angles.extend(up)
        angles.extend(down)

    angles[0] = np.nan
    frame_idx = np.arange(len(angles))
    pose_ok = np.ones_like(frame_idx, dtype=float)
    pose_ok[0] = 0.0
    metrics_df = pd.DataFrame({"frame_idx": frame_idx, "angle": angles, "pose_ok": pose_ok})
    stats = _dummy_stats(exercise="deadlift")
    report = _dummy_report(stats, "deadlift", repetitions=3, metrics=metrics_df)

    rep_intervals, valley_frames, frame_values, primary_metric = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="deadlift",
    )

    assert len(rep_intervals) > 0

    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="deadlift",
        valley_frames=valley_frames,
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric=primary_metric,
    )

    assert rep_speeds_df.shape[0] == len(rep_intervals)

    rep_chart_df = _build_rep_speed_chart_df(
        rep_speeds_df, phase_order_for_exercise("deadlift")
    )

    assert rep_chart_df["Repetition"].nunique() == len(rep_intervals)
    assert rep_chart_df.shape[0] == 2 * len(rep_intervals)
    assert not rep_chart_df[["Speed", "Phase duration (s)"]].isna().any().any()
    assert (rep_chart_df["Phase duration (s)"] > 0).all()


def test_nonconsecutive_index_samples_nearest_frames():
    metrics_df = pd.DataFrame(
        {
            "frame_idx": [0, 2, 4, 6, 8],
            "angle": [10.0, 0.0, 10.0, 0.0, 10.0],
        },
        index=[100, 101, 104, 105, 110],
    )
    stats = _dummy_stats(exercise="squat")
    report = _dummy_report(stats, "squat", repetitions=2, metrics=metrics_df)

    rep_intervals, valley_frames, frame_values, primary_metric = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="squat",
    )

    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="squat",
        valley_frames=valley_frames,
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric=primary_metric,
    )

    chart_df = _build_rep_speed_chart_df(
        rep_speeds_df, phase_order_for_exercise("squat")
    )

    assert chart_df["Repetition"].nunique() == len(rep_speeds_df)
    assert chart_df["Phase"].value_counts().to_dict() == {"Down": len(rep_speeds_df), "Up": len(rep_speeds_df)}
    assert not chart_df["Speed"].isna().any()


def test_trunk_inclination_uses_max_turning_point():
    metrics_df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "trunk_inclination_deg": [0.0, 20.0, 0.0],
        }
    )
    stats = _dummy_stats(primary_angle="trunk_inclination_deg", exercise="squat")
    report = _dummy_report(stats, "squat", repetitions=1, metrics=metrics_df)

    rep_intervals = [(0, 2)]
    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="squat",
        valley_frames=[],
        frame_values=metrics_df["frame_idx"],
        metrics_df=metrics_df,
        primary_metric="trunk_inclination_deg",
    )

    assert rep_speeds_df.iloc[0]["Bottom frame"] == 1


def test_deadlift_infers_start_bottom_and_keeps_phases_non_nan():
    angles = []
    for _ in range(3):
        up = np.linspace(0.0, 10.0, 5)
        down = np.linspace(10.0, 0.0, 5)[1:]
        angles.extend(up)
        angles.extend(down)

    angles = [angles[0]] + angles  # start at boundary bottom
    frame_idx = np.arange(len(angles))
    metrics_df = pd.DataFrame({"frame_idx": frame_idx, "angle": angles})
    stats = _dummy_stats(exercise="deadlift")
    report = _dummy_report(stats, "deadlift", repetitions=3, metrics=metrics_df)

    rep_intervals, _, frame_values, primary_metric = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="deadlift",
    )

    assert len(rep_intervals) > 0
    if len(rep_intervals) < report.repetitions:
        assert getattr(report.stats, "counting_accuracy_warning", None)

    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="deadlift",
        valley_frames=[],
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric=primary_metric,
    )

    chart_df = _build_rep_speed_chart_df(
        rep_speeds_df, phase_order_for_exercise("deadlift")
    )

    assert chart_df.shape[0] == 2 * len(rep_intervals)
    assert not chart_df[["Speed", "Phase duration (s)"]].isna().any().any()


def test_deadlift_boundary_pose_gaps_still_produce_phases():
    angles = []
    for _ in range(2):
        up = np.linspace(0.0, 10.0, 5)
        down = np.linspace(10.0, 0.0, 5)[1:]
        angles.extend(up)
        angles.extend(down)

    frame_idx = np.arange(len(angles))
    pose_ok = np.ones_like(frame_idx, dtype=float)
    pose_ok[0] = 0.0
    angles[0] = np.nan
    metrics_df = pd.DataFrame({"frame_idx": frame_idx, "angle": angles, "pose_ok": pose_ok})
    stats = _dummy_stats(exercise="deadlift")
    report = _dummy_report(stats, "deadlift", repetitions=2, metrics=metrics_df)

    rep_intervals, _, frame_values, primary_metric = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="deadlift",
    )

    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="deadlift",
        valley_frames=[],
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric=primary_metric,
    )

    chart_df = _build_rep_speed_chart_df(
        rep_speeds_df, phase_order_for_exercise("deadlift")
    )

    assert chart_df.shape[0] == 2 * len(rep_speeds_df)
    assert not chart_df[["Speed", "Phase duration (s)"]].isna().any().any()


def test_fewer_bottoms_emit_warning_and_limit_intervals():
    metrics_df = pd.DataFrame({"frame_idx": [0, 1, 2], "angle": [1.0, 0.5, 1.0]})
    stats = _dummy_stats(exercise="deadlift")
    report = _dummy_report(stats, "deadlift", repetitions=3, metrics=metrics_df)

    rep_intervals, _, _, _ = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="deadlift",
    )

    assert len(rep_intervals) <= report.repetitions
    assert getattr(report.stats, "counting_accuracy_warning", None)


def test_deadlift_threshold_intervals_match_reported_reps(monkeypatch):
    monkeypatch.setattr("src.ui.steps.results.results.find_peaks", None)

    angles = []
    bottoms = [0.0, 0.5, 1.5]
    for bottom in bottoms:
        up = np.linspace(bottom, 10.0, 5)
        down = np.linspace(10.0, bottom, 5)[1:]
        angles.extend(up)
        angles.extend(down)

    frame_idx = np.arange(len(angles))
    metrics_df = pd.DataFrame({"frame_idx": frame_idx, "angle": angles})
    stats = _dummy_stats(exercise="deadlift")
    stats.lower_threshold = 2.0
    stats.upper_threshold = 8.0
    report = _dummy_report(stats, "deadlift", repetitions=3, metrics=metrics_df)

    rep_intervals, _, frame_values, primary_metric = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="deadlift",
    )

    assert len(rep_intervals) == report.repetitions
    assert not getattr(report.stats, "counting_accuracy_warning", None)

    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="deadlift",
        valley_frames=[],
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric=primary_metric,
    )

    chart_df = _build_rep_speed_chart_df(
        rep_speeds_df, phase_order_for_exercise("deadlift")
    )

    assert chart_df.shape[0] == 2 * len(rep_speeds_df)
    assert not chart_df[["Speed", "Phase duration (s)"]].isna().any().any()


def test_deadlift_threshold_intervals_use_faults_config(monkeypatch):
    monkeypatch.setattr("src.ui.steps.results.results.find_peaks", None)

    angles = []
    bottoms = [0.0, 0.5, 1.5]
    for bottom in bottoms:
        up = np.linspace(bottom, 10.0, 5)
        down = np.linspace(10.0, bottom, 5)[1:]
        angles.extend(up)
        angles.extend(down)

    frame_idx = np.arange(len(angles))
    metrics_df = pd.DataFrame({"frame_idx": frame_idx, "angle": angles})
    stats = _dummy_stats(exercise="deadlift")
    report = _dummy_report(stats, "deadlift", repetitions=3, metrics=metrics_df)
    report.config_used.faults = SimpleNamespace(low_thresh=2.0, high_thresh=8.0)

    rep_intervals, _, frame_values, primary_metric = _compute_rep_intervals(
        metrics_df=metrics_df,
        report=report,
        stats=stats,
        numeric_columns=["angle"],
        exercise_key="deadlift",
    )

    assert len(rep_intervals) == report.repetitions
    assert not getattr(report.stats, "counting_accuracy_warning", None)

    rep_speeds_df = _compute_rep_speeds(
        rep_intervals,
        stats,
        exercise_key="deadlift",
        valley_frames=[],
        frame_values=frame_values,
        metrics_df=metrics_df,
        primary_metric=primary_metric,
    )

    chart_df = _build_rep_speed_chart_df(
        rep_speeds_df, phase_order_for_exercise("deadlift")
    )

    assert chart_df.shape[0] == 2 * len(rep_speeds_df)
    assert not chart_df[["Speed", "Phase duration (s)"]].isna().any().any()
