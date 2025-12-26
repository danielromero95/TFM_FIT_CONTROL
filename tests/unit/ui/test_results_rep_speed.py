import pandas as pd

from src.ui.steps.results.results import (
    _build_rep_speed_chart_df,
    phase_order_for_exercise,
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
