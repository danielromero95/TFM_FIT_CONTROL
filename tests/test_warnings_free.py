"""Ensure the classification pipeline executes without RuntimeWarnings."""

from __future__ import annotations

import warnings
from typing import Iterable

import pytest

np = pytest.importorskip("numpy")

from exercise_detection import FeatureSeries, classify_features

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


class _NoWarningsContext:
    def __enter__(self) -> Iterable[warnings.WarningMessage]:
        self._catch = warnings.catch_warnings(record=True)
        self._records = self._catch.__enter__()
        warnings.simplefilter("always")
        return self._records

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._catch.__exit__(exc_type, exc, tb)
        if exc_type is not None:
            return False
        if self._records:
            unexpected = ", ".join(sorted({rec.category.__name__ for rec in self._records}))
            raise AssertionError(f"Unexpected warnings captured: {unexpected}")
        return False


try:
    pytest.warns(None)
except TypeError:  # pragma: no cover - pytest>=8 removed the None shortcut
    _orig_warns = pytest.warns

    def _warns(expected_warning=Warning, *args, **kwargs):  # type: ignore[override]
        if expected_warning is None:
            return _NoWarningsContext()
        return _orig_warns(expected_warning, *args, **kwargs)

    pytest.warns = _warns  # type: ignore[assignment]


def _base_feature_data(length: int) -> dict[str, np.ndarray]:
    const_angle = np.full(length, 90.0)
    const_coord = np.full(length, 0.5)
    left_x = np.full(length, 0.4)
    right_x = np.full(length, 0.6)
    hip_y = np.full(length, 0.65)
    knee_y = np.full(length, 0.8)
    ankle_y = np.full(length, 0.92)
    return {
        "knee_angle_left": const_angle.copy(),
        "knee_angle_right": const_angle.copy(),
        "hip_angle_left": const_angle.copy(),
        "hip_angle_right": const_angle.copy(),
        "elbow_angle_left": const_angle.copy(),
        "elbow_angle_right": const_angle.copy(),
        "shoulder_angle_left": const_angle.copy(),
        "shoulder_angle_right": const_angle.copy(),
        "pelvis_y": const_coord.copy(),
        "torso_length": np.full(length, 0.52),
        "torso_length_world": np.full(length, 0.52),
        "wrist_left_x": const_coord.copy(),
        "wrist_left_y": const_coord.copy(),
        "wrist_right_x": const_coord.copy(),
        "wrist_right_y": const_coord.copy(),
        "shoulder_width_norm": np.full(length, 0.6),
        "shoulder_yaw_deg": np.full(length, np.nan),
        "shoulder_z_delta_abs": np.full(length, np.nan),
        "torso_tilt_deg": np.full(length, 15.0),
        "shoulder_left_x": left_x.copy(),
        "shoulder_left_y": np.full(length, 0.45),
        "shoulder_right_x": right_x.copy(),
        "shoulder_right_y": np.full(length, 0.45),
        "hip_left_x": np.full(length, 0.42),
        "hip_left_y": hip_y.copy(),
        "hip_right_x": np.full(length, 0.58),
        "hip_right_y": hip_y.copy(),
        "knee_left_x": np.full(length, 0.44),
        "knee_left_y": knee_y.copy(),
        "knee_right_x": np.full(length, 0.56),
        "knee_right_y": knee_y.copy(),
        "ankle_left_x": np.full(length, 0.46),
        "ankle_left_y": ankle_y.copy(),
        "ankle_right_x": np.full(length, 0.54),
        "ankle_right_y": ankle_y.copy(),
        "elbow_left_x": np.full(length, 0.38),
        "elbow_left_y": np.full(length, 0.52),
        "elbow_right_x": np.full(length, 0.62),
        "elbow_right_y": np.full(length, 0.52),
    }


def _make_feature_series(data: dict[str, np.ndarray]) -> FeatureSeries:
    length = len(next(iter(data.values())))
    return FeatureSeries(
        data=data,
        sampling_rate=30.0,
        valid_frames=length,
        total_frames=length,
    )


def test_classify_features_warning_free_synthetic_fixture() -> None:
    length = 40
    data = _base_feature_data(length)
    data["knee_angle_left"] = np.linspace(80.0, 150.0, length)
    data["knee_angle_right"] = np.linspace(82.0, 148.0, length)
    data["hip_angle_left"] = np.linspace(60.0, 110.0, length)
    data["hip_angle_right"] = np.linspace(58.0, 108.0, length)
    data["pelvis_y"] = np.linspace(0.45, 0.60, length)
    data["shoulder_width_norm"] = np.full(length, 0.65)
    data["shoulder_yaw_deg"] = np.full(length, 8.0)
    data["shoulder_z_delta_abs"] = np.full(length, 0.01)
    data["torso_tilt_deg"] = np.full(length, 12.0)
    data["knee_left_x"] = np.full(length, 0.56)
    data["knee_right_x"] = np.full(length, 0.44)
    data["ankle_left_x"] = np.full(length, 0.46)
    data["ankle_right_x"] = np.full(length, 0.54)

    features = _make_feature_series(data)

    with pytest.warns(None):
        label, view, confidence = classify_features(features)

    assert label == "squat"
    assert view == "front"
    assert confidence >= 0.6


def test_classify_features_warning_free_sparse_clip() -> None:
    length = 30
    data = {
        "knee_angle_left": np.concatenate([
            np.linspace(80.0, 120.0, length - 5),
            np.full(5, np.nan),
        ]),
        "hip_angle_left": np.concatenate([
            np.linspace(50.0, 70.0, 6),
            np.full(length - 6, np.nan),
        ]),
        "elbow_angle_left": np.concatenate([
            np.linspace(40.0, 80.0, 6),
            np.full(length - 6, np.nan),
        ]),
    }

    features = FeatureSeries(
        data=data,
        sampling_rate=30.0,
        valid_frames=length,
        total_frames=length,
    )

    with pytest.warns(None):
        label, view, confidence = classify_features(features)

    assert label == "unknown"
    assert view == "unknown"
    assert confidence == 0.0
