from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from src.B_pose_estimation.estimators.mediapipe_estimators import (
    CroppedPoseEstimator,
    _rescale_landmarks_from_crop,
)
from src.B_pose_estimation.estimators.mediapipe_pool import PoseGraphPool


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


def _install_fake_mediapipe(monkeypatch):
    mp_module = types.ModuleType("mediapipe")
    framework = types.ModuleType("mediapipe.framework")
    formats = types.ModuleType("mediapipe.framework.formats")
    landmark_pb2 = types.ModuleType("mediapipe.framework.formats.landmark_pb2")
    landmark_pb2.NormalizedLandmark = _FakeNormalizedLandmark
    landmark_pb2.NormalizedLandmarkList = _FakeNormalizedLandmarkList
    formats.landmark_pb2 = landmark_pb2
    framework.formats = formats
    mp_module.framework = framework
    solutions = types.ModuleType("mediapipe.python.solutions")
    mp_module.python = types.SimpleNamespace(solutions=solutions)

    monkeypatch.setitem(sys.modules, "mediapipe", mp_module)
    monkeypatch.setitem(sys.modules, "mediapipe.framework", framework)
    monkeypatch.setitem(sys.modules, "mediapipe.framework.formats", formats)
    monkeypatch.setitem(sys.modules, "mediapipe.framework.formats.landmark_pb2", landmark_pb2)


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


class _FakePose:
    def __init__(self, landmarks):
        self._landmarks = landmarks

    def process(self, _image):
        class _Result:
            def __init__(self, pose_landmarks):
                self.pose_landmarks = pose_landmarks

        return _Result(_FakeNormalizedLandmarkList(self._landmarks) if self._landmarks else None)


class _FakeDrawing:
    def __init__(self):
        self.calls = []

    def draw_landmarks(self, image, landmark_list, _connections):
        self.calls.append((image.copy(), landmark_list))


def test_cropped_estimator_draws_full_frame_landmarks(monkeypatch):
    _install_fake_mediapipe(monkeypatch)

    drawing = _FakeDrawing()
    poses = [
        _FakePose(
            [
                _FakeNormalizedLandmark(0.25, 0.25, visibility=1.0),
                _FakeNormalizedLandmark(0.75, 0.75, visibility=1.0),
            ]
        ),
        _FakePose([_FakeNormalizedLandmark(0.5, 0.25, visibility=1.0)]),
    ]

    def _fake_acquire(**_kwargs):
        pose = poses.pop(0)
        return pose, (False, 2, 0.5, 0.5)

    monkeypatch.setattr(
        PoseGraphPool, "acquire", classmethod(lambda cls, **kwargs: _fake_acquire(**kwargs))
    )
    monkeypatch.setattr(PoseGraphPool, "release", classmethod(lambda cls, inst, key: None))
    monkeypatch.setattr(PoseGraphPool, "_imported", True)
    monkeypatch.setattr(PoseGraphPool, "_ensure_imports", classmethod(lambda cls: None))
    monkeypatch.setattr(PoseGraphPool, "mp_pose", types.SimpleNamespace(Pose=None))
    monkeypatch.setattr(PoseGraphPool, "mp_drawing", drawing)

    estimator = CroppedPoseEstimator(target_size=(4, 4), crop_margin=0.0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    result = estimator.estimate(image)

    assert result.annotated_image is not None
    assert result.annotated_image.shape == image.shape
    assert drawing.calls, "Landmarks should be drawn on the full frame"

    _, drawn_landmarks = drawing.calls[0]
    assert isinstance(drawn_landmarks, _FakeNormalizedLandmarkList)
    assert pytest.approx(drawn_landmarks.landmark[0].x) == pytest.approx(0.5)
    assert pytest.approx(drawn_landmarks.landmark[0].y) == pytest.approx(0.375)


def test_cropped_estimator_falls_back_to_full_pose_when_crop_fails(monkeypatch):
    _install_fake_mediapipe(monkeypatch)

    drawing = _FakeDrawing()
    poses = [
        _FakePose([_FakeNormalizedLandmark(0.25, 0.5, visibility=1.0)]),
        _FakePose([]),  # recorte sin detección
    ]

    def _fake_acquire(**_kwargs):
        pose = poses.pop(0)
        return pose, (False, 2, 0.5, 0.5)

    monkeypatch.setattr(
        PoseGraphPool, "acquire", classmethod(lambda cls, **kwargs: _fake_acquire(**kwargs))
    )
    monkeypatch.setattr(PoseGraphPool, "release", classmethod(lambda cls, inst, key: None))
    monkeypatch.setattr(PoseGraphPool, "_imported", True)
    monkeypatch.setattr(PoseGraphPool, "_ensure_imports", classmethod(lambda cls: None))
    monkeypatch.setattr(PoseGraphPool, "mp_pose", types.SimpleNamespace(Pose=None))
    monkeypatch.setattr(PoseGraphPool, "mp_drawing", drawing)

    estimator = CroppedPoseEstimator(target_size=(4, 4), crop_margin=0.0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    result = estimator.estimate(image)

    assert result.landmarks is not None
    assert result.landmarks[0].get("x") == pytest.approx(0.25)
    assert drawing.calls, "Debe dibujar los landmarks del fotograma completo si el recorte falla"
