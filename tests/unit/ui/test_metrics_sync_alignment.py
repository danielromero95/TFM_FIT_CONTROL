import numpy as np
import pandas as pd
import pytest

from src.ui.metrics_sync.component import _build_payload


def test_axis_times_rebased_from_frame_idx_start():
    frame_idx = np.arange(10, 15)
    df = pd.DataFrame({"frame_idx": frame_idx, "metric": np.linspace(0, 1, frame_idx.size)})

    payload = _build_payload(df, ["metric"], fps=30.0, max_points=100)

    expected_start = frame_idx[0] / 30.0
    expected_span = (frame_idx[-1] - frame_idx[0]) / 30.0
    assert payload["axis_times"][0] == pytest.approx(expected_start)
    assert payload["axis_times"][-1] == pytest.approx(expected_start + expected_span)
    assert payload["time_offset_s"] == pytest.approx(0.0)


def test_axis_times_are_strictly_increasing_with_duplicates():
    times = [1.0, 1.0, 1.002, 1.004]
    df = pd.DataFrame({"source_time_s": times, "source_frame_idx": [0, 1, 2, 3], "metric": [0.0, 0.1, 0.2, 0.3]})

    payload = _build_payload(df, ["metric"], fps=30.0, max_points=10)

    axis_times = payload["axis_times"]
    assert axis_times[0] == pytest.approx(0.0)
    assert all(b > a for a, b in zip(axis_times, axis_times[1:]))
