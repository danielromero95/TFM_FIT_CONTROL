import numpy as np

from src.B_pose_estimation.metrics import angular_velocity


def test_forward_nan_behavior_and_length():
    # Mix of finite and NaN values
    seq = [10.0, 12.0, float("nan"), 15.0, 18.0]
    fps = 50.0  # dt = 0.02
    out = angular_velocity(seq, fps, method="forward")
    assert isinstance(out, list)
    assert len(out) == len(seq)

    dt = 1.0 / fps
    expected = [
        0.0,                    # padded
        abs(12.0 - 10.0) / dt,  # finite-finite
        0.0,                    # 12 -> NaN
        0.0,                    # NaN -> 15
        abs(18.0 - 15.0) / dt,  # finite-finite
    ]

    # Exact equality is fine here (no floating noise introduced)
    assert out == expected


def test_forward_empty_and_zero_fps():
    assert angular_velocity([], 30.0, method="forward") == []
    seq = [1.0, 2.0, 3.0]
    assert angular_velocity(seq, 0.0, method="forward") == [0.0] * len(seq)


def test_central_smoke_no_nan_matches_gradient():
    # For clean data, central difference should match np.gradient over dt
    seq = [0.0, 1.0, 3.0, 6.0, 10.0]
    fps = 20.0
    dt = 1.0 / fps
    out = angular_velocity(seq, fps, method="central")
    expected = np.gradient(np.asarray(seq, dtype=float), dt)
    assert np.allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-12)
    assert len(out) == len(seq)
