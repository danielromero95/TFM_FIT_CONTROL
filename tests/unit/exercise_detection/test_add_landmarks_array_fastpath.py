import pytest

np = pytest.importorskip("numpy")

from src.exercise_detection.exercise_detector import (
    _build_features_from_landmark_array,
    _build_features_from_landmarks,
)


def test_feature_parity_list_vs_array_minimal_case():
    """Ensure list[dict] and ndarray landmark inputs produce identical features."""

    lms = []
    for i in range(33):
        lms.append(
            {
                "x": 0.1 * i,
                "y": 0.2 * i,
                "z": 0.0,
                "visibility": 1.0,
            }
        )

    arr = np.zeros((33, 4), dtype=float)
    for i in range(33):
        arr[i, 0] = 0.1 * i
        arr[i, 1] = 0.2 * i
        arr[i, 2] = 0.0
        arr[i, 3] = 1.0

    feat_list = _build_features_from_landmarks(lms)
    feat_arr = _build_features_from_landmark_array(arr)

    assert set(feat_list.keys()) == set(feat_arr.keys())
    for key in feat_list.keys():
        a = float(feat_list[key])
        b = float(feat_arr[key])
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b)
        else:
            assert np.isclose(a, b, rtol=1e-12, atol=1e-12)
