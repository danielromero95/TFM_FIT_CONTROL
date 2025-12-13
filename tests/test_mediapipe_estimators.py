from __future__ import annotations

import math

import numpy as np
import pytest

from src.B_pose_estimation.estimators.mediapipe_estimators import (
    _rescale_landmarks_from_crop,
)


class _FakeNormalizedLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class _FakeNormalizedLandmarkList:
    def __init__(self, landmark):
        self.landmark = list(landmark)


class _FakePb2:
    NormalizedLandmark = _FakeNormalizedLandmark
    NormalizedLandmarkList = _FakeNormalizedLandmarkList


@pytest.mark.parametrize(
    "crop_box, lm, expected",
    [
        (
            (100, 50, 300, 250),
            _FakeNormalizedLandmark(0.25, 0.75, z=0.1, visibility=0.9),
            (0.375, 2 / 3, 0.1, 0.9),
        ),
        (
            (0, 0, 200, 200),
            _FakeNormalizedLandmark(1.2, -0.1, z=-0.5, visibility=2.0),
            (0.6, 0.0, -0.5, 2.0),
        ),
    ],
)
def test_rescale_landmarks_from_crop_respects_bounds_and_scaling(crop_box, lm, expected):
    landmarks, proto_list = _rescale_landmarks_from_crop(
        [lm],
        crop_box,
        image_width=400,
        image_height=300,
        landmark_pb2_module=_FakePb2,
    )

    (x, y, z, visibility) = expected
    assert math.isclose(landmarks[0].x, x, rel_tol=1e-9)
    assert math.isclose(landmarks[0].y, y, rel_tol=1e-9)
    assert math.isclose(landmarks[0].z, z, rel_tol=1e-9)
    assert math.isclose(landmarks[0].visibility, visibility, rel_tol=1e-9)
    assert isinstance(proto_list, _FakeNormalizedLandmarkList)
    assert len(proto_list.landmark) == 1
    assert np.isfinite(proto_list.landmark[0].x)
    assert 0.0 <= proto_list.landmark[0].x <= 1.0
    assert 0.0 <= proto_list.landmark[0].y <= 1.0


def test_rescale_landmarks_from_crop_requires_complete_box():
    with pytest.raises(ValueError):
        _rescale_landmarks_from_crop(
            [],
            (10, 20, 30),
            image_width=100,
            image_height=100,
            landmark_pb2_module=_FakePb2,
        )
