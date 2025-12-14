from __future__ import annotations

import numpy as np
import pandas as pd

from src.B_pose_estimation.signal import derivative, interpolate_small_gaps, smooth_series


def test_interpolate_small_gaps_respects_limit():
    series = [1.0, np.nan, 3.0, np.nan, np.nan, 7.0]
    filled_short = interpolate_small_gaps(series, max_gap_frames=1)
    assert np.isnan(filled_short[1]) is False
    assert np.isnan(filled_short[3]) is True and np.isnan(filled_short[4]) is True

    filled_longer = interpolate_small_gaps(series, max_gap_frames=3)
    assert np.allclose(filled_longer[[1, 3, 4]], [2.0, 5.0, 6.0], equal_nan=False)


def test_smooth_series_preserves_nans_and_shape():
    base = np.sin(np.linspace(0, np.pi, 11))
    base[5:] = np.nan
    smoothed = smooth_series(base, fps=30.0, method="savgol", window_seconds=0.2)
    assert len(smoothed) == len(base)
    assert np.isnan(smoothed[7])  # largo hueco permanece NaN
    assert smoothed[2] > smoothed[1]  # sin desplazamiento significativo


def test_derivative_is_nan_safe():
    series = pd.Series([0.0, 1.0, np.nan, 4.0])
    vel = derivative(series, fps=10.0)
    assert np.isnan(vel[2])
    assert vel[1] > 0
    assert vel.shape == series.shape
