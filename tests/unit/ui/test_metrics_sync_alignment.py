import numpy as np
import pandas as pd
import pytest

from src.ui.metrics_sync.component import _build_payload


def test_axis_times_rebased_from_frame_idx_start():
    frame_idx = np.arange(10, 15)
    df = pd.DataFrame({"frame_idx": frame_idx, "metric": np.linspace(0, 1, frame_idx.size)})

    payload = _build_payload(df, ["metric"], fps=30.0, max_points=100)

    assert payload["axis_times"][0] == pytest.approx(0.0)
    expected_span = (frame_idx[-1] - frame_idx[0]) / 30.0
    assert payload["axis_times"][-1] == pytest.approx(expected_span)
