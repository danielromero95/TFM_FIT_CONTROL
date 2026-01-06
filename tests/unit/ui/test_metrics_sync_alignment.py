import numpy as np
import pandas as pd
import pytest

from src.ui.metrics_sync.component import _build_payload, nearest_time_index


def test_axis_times_preserve_full_resolution():
    df = pd.DataFrame(
        {
            "time_s": np.linspace(0, 5, 50),
            "frame_idx": np.arange(50),
            "metric_a": np.linspace(0.0, 1.0, 50),
        }
    )

    payload = _build_payload(df, ["metric_a"], fps=25.0, max_points=10)

    assert len(payload["times"]) < len(payload["axis_times"])
    assert pytest.approx(payload["axis_times"][0]) == 0.0
    assert pytest.approx(payload["axis_times"][-1]) == 5.0
    assert len(payload["axis_frames"]) == len(df)


def test_frame_idx_only_converted_to_seconds():
    frame_idx = np.arange(120)
    df = pd.DataFrame(
        {
            "frame_idx": frame_idx,
            "metric_a": np.linspace(0.0, 1.0, len(frame_idx)),
        }
    )

    payload = _build_payload(df, ["metric_a"], fps=30.0, max_points=10)

    assert len(payload["axis_times"]) == len(df)
    assert payload["axis_times"][-1] == pytest.approx(frame_idx[-1] / 30.0)
    assert payload["axis_times"][-1] != pytest.approx(float(frame_idx[-1]))
    assert len(payload["times"]) < len(payload["axis_times"])


def test_nearest_time_index_binary_search():
    times = [0.0, 0.1, 0.25, 0.4, 0.55]
    assert nearest_time_index(times, 0.26) == 2
    assert nearest_time_index(times, 0.52) == 4
    assert nearest_time_index(times, -0.1) == 0
