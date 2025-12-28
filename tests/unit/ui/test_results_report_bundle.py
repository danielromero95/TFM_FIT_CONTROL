import io
import json
import zipfile

import numpy as np
import pandas as pd

from src.pipeline_data import Report, RunStats
from src.ui.steps.results.results import (
    _build_debug_report_bundle,
    _build_rep_speed_chart_df,
    _compute_rep_intervals,
    _compute_rep_speeds,
    phase_order_for_exercise,
)


class _DummyConfig:
    def __init__(self, counting=None, faults=None):
        self.counting = counting
        self.faults = faults

    def to_serializable_dict(self):
        return {"counting": self.counting or {}, "faults": self.faults or {}}


def _dummy_stats(primary_angle: str = "angle", exercise: str = "squat") -> RunStats:
    return RunStats(
        config_sha1="sha1",
        fps_original=30.0,
        fps_effective=30.0,
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


def _dummy_report(stats: RunStats, repetitions: int, metrics: pd.DataFrame | None) -> Report:
    return Report(
        repetitions=repetitions,
        metrics=metrics,
        stats=stats,
        config_used=_DummyConfig(),
    )


def test_debug_bundle_contains_rep_speed_artifacts():
    angles: list[float] = []
    for _ in range(3):
        angles.extend([10.0, 5.0, 0.0, 5.0])

    frame_idx = np.arange(len(angles))
    metrics_df = pd.DataFrame({"frame_idx": frame_idx, "angle": angles})
    stats = _dummy_stats(exercise="squat")
    report = _dummy_report(stats, repetitions=3, metrics=metrics_df)

    (
        rep_intervals,
        valley_frames,
        frame_values,
        primary_metric,
        interval_strategy,
        thresholds_used,
    ) = _compute_rep_intervals(
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

    phase_order = phase_order_for_exercise("squat")
    rep_chart_df = _build_rep_speed_chart_df(rep_speeds_df, phase_order)

    metrics_csv = metrics_df.to_csv(index=False)
    stats_df = pd.DataFrame([
        {"Field": "metric", "Value": "angle"},
    ])

    bundle_bytes = _build_debug_report_bundle(
        report=report,
        stats_df=stats_df,
        metrics_df=metrics_df,
        metrics_csv=metrics_csv,
        effective_config_bytes=None,
        video_name="video.mp4",
        rep_intervals=rep_intervals,
        valley_frames=valley_frames,
        rep_speeds_df=rep_speeds_df,
        rep_chart_df=rep_chart_df,
        exercise_key="squat",
        primary_metric=primary_metric,
        phase_order=phase_order,
        interval_strategy=interval_strategy,
        thresholds_used=thresholds_used,
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zf:
        names = set(zf.namelist())
        assert {
            "report.json",
            "run_stats.csv",
            "effective_config.json",
            "metrics.csv",
            "rep_intervals.csv",
            "rep_speeds.csv",
            "rep_speed_long.csv",
            "rep_speed_meta.json",
        }.issubset(names)

        rep_intervals_df = pd.read_csv(zf.open("rep_intervals.csv"))
        assert not rep_intervals_df.empty
        assert {"rep", "start_frame", "end_frame"}.issubset(rep_intervals_df.columns)

        rep_speeds_loaded = pd.read_csv(zf.open("rep_speeds.csv"))
        assert rep_speeds_loaded.shape[0] == len(rep_intervals)

        rep_speed_long = pd.read_csv(zf.open("rep_speed_long.csv"))
        assert rep_speed_long["Repetition"].nunique() == len(rep_intervals)

        meta = json.loads(zf.read("rep_speed_meta.json"))
        assert meta["expected_reps"] == report.repetitions
        assert meta["interval_count"] == len(rep_intervals)
        assert meta["interval_strategy"] == interval_strategy


def test_debug_bundle_handles_missing_metrics():
    stats = _dummy_stats(exercise="deadlift")
    report = _dummy_report(stats, repetitions=0, metrics=None)
    stats_df = pd.DataFrame([
        {"Field": "metric", "Value": "angle"},
    ])

    bundle_bytes = _build_debug_report_bundle(
        report=report,
        stats_df=stats_df,
        metrics_df=None,
        metrics_csv=None,
        effective_config_bytes=None,
        video_name=None,
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zf:
        names = set(zf.namelist())
        assert "report.json" in names
        assert "run_stats.csv" in names
        assert "effective_config.json" in names

        # New artifacts should still be present even if empty
        assert "rep_intervals.csv" in names
        assert "rep_speeds.csv" in names
        assert "rep_speed_long.csv" in names
        assert "rep_speed_meta.json" in names
