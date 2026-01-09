import math

import pandas as pd

from src.ui.steps.results.results import _compute_rep_splits
from src.ui.steps.results.results import _metric_extreme


def test_rep_splits_use_min_for_squat():
    metrics_df = pd.DataFrame({"primary": [5, 4, 3, 1, 2, 3, 4, 5, 6, 7, 8]})
    rep_splits = _compute_rep_splits(
        metrics_df,
        [(0, 10)],
        primary_metric="primary",
        exercise_key="squat",
        fps_effective=10,
    )

    assert rep_splits == [(0.0, 0.3, 1.0)]


def test_rep_splits_use_max_for_deadlift():
    metrics_df = pd.DataFrame({"primary": [1, 2, 3, 4, 5, 6, 9, 4, 3, 2, 1]})
    rep_splits = _compute_rep_splits(
        metrics_df,
        [(0, 10)],
        primary_metric="primary",
        exercise_key="deadlift",
        fps_effective=10,
    )

    assert rep_splits == [(0.0, 0.6, 1.0)]


def test_rep_splits_trunk_extremes_flip_by_exercise():
    metrics_df = pd.DataFrame(
        {"trunk_inclination_deg": [1, 2, 3, 9, 8, 7, 6, 5, 4, 3, 2]}
    )
    assert _metric_extreme("trunk_inclination_deg") == "max"

    rep_splits_squat = _compute_rep_splits(
        metrics_df,
        [(0, 10)],
        primary_metric="trunk_inclination_deg",
        exercise_key="squat",
        fps_effective=10,
    )
    assert rep_splits_squat == [(0.0, 0.3, 1.0)]

    rep_splits_deadlift = _compute_rep_splits(
        metrics_df,
        [(0, 10)],
        primary_metric="trunk_inclination_deg",
        exercise_key="deadlift",
        fps_effective=10,
    )
    assert rep_splits_deadlift == [(0.0, 0.0, 1.0)]


def test_rep_splits_fallback_to_midpoint_when_nan():
    metrics_df = pd.DataFrame({"primary": [math.nan] * 11})
    rep_splits = _compute_rep_splits(
        metrics_df,
        [(0, 10)],
        primary_metric="primary",
        exercise_key="squat",
        fps_effective=10,
    )

    start, split, end = rep_splits[0]
    assert start == 0.0
    assert end == 1.0
    assert split == 0.5
