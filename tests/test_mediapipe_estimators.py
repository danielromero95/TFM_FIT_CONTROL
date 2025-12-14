from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from src.B_pose_estimation.estimators.mediapipe_estimators import (
    CroppedPoseEstimator,
    LandmarkSmoother,
    PoseEstimator,
    _rescale_landmarks_from_crop,
    pose_reliable,
    reliability_score,
)
from src.B_pose_estimation.estimators.mediapipe_pool import PoseGraphPool
from src.B_pose_estimation.types import Landmark


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


def test_pose_estimator_reuses_pose_instance(monkeypatch):
    _install_fake_mediapipe(monkeypatch)
    drawing = _FakeDrawing()
    calls: list[dict[str, object]] = []

    class _FakePose:
        def __init__(self):
            self.calls = 0

        def process(self, _image):
            self.calls += 1

            return types.SimpleNamespace(
                pose_landmarks=_FakeNormalizedLandmarkList(
                    [_FakeNormalizedLandmark(0.1, 0.2, visibility=0.9)]
                )
            )

    def _fake_acquire(**kwargs):
        calls.append(kwargs)
        return _FakePose(), (False, 2, 0.6, 0.6, True, False)

    monkeypatch.setattr(PoseGraphPool, "acquire", classmethod(lambda cls, **kwargs: _fake_acquire(**kwargs)))
    monkeypatch.setattr(PoseGraphPool, "release", classmethod(lambda cls, inst, key: None))
    monkeypatch.setattr(PoseGraphPool, "_imported", True)
    monkeypatch.setattr(PoseGraphPool, "_ensure_imports", classmethod(lambda cls: None))
    monkeypatch.setattr(PoseGraphPool, "mp_pose", types.SimpleNamespace(Pose=None))
    monkeypatch.setattr(PoseGraphPool, "mp_drawing", drawing)

    estimator = PoseEstimator()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    estimator.estimate(image)
    estimator.estimate(image)
    estimator.close()

    assert len(calls) == 1


def test_pose_estimator_resets_smoother_on_miss(monkeypatch):
    _install_fake_mediapipe(monkeypatch)
    drawing = _FakeDrawing()

    sequences = [
        [_FakeNormalizedLandmark(0.1, 0.2, visibility=1.0)],
        [],
    ]

    class _FakePose:
        def process(self, _image):
            landmarks = sequences.pop(0)
            return types.SimpleNamespace(
                pose_landmarks=
                _FakeNormalizedLandmarkList(landmarks) if landmarks else None
            )

    def _fake_acquire(**kwargs):
        return _FakePose(), (False, 2, 0.6, 0.6, True, False)

    monkeypatch.setattr(PoseGraphPool, "acquire", classmethod(lambda cls, **kwargs: _fake_acquire(**kwargs)))
    monkeypatch.setattr(PoseGraphPool, "release", classmethod(lambda cls, inst, key: None))
    monkeypatch.setattr(PoseGraphPool, "_imported", True)
    monkeypatch.setattr(PoseGraphPool, "_ensure_imports", classmethod(lambda cls: None))
    monkeypatch.setattr(PoseGraphPool, "mp_pose", types.SimpleNamespace(Pose=None))
    monkeypatch.setattr(PoseGraphPool, "mp_drawing", drawing)

    estimator = PoseEstimator(landmark_smoothing_alpha=0.5)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    estimator.estimate(image)
    assert estimator._smoother is not None
    assert estimator._smoother._state is not None

    result = estimator.estimate(image)
    assert result.pose_ok is False
    assert estimator._smoother._state is None


def test_pose_reliable_requires_visible_core_landmarks():
    landmarks = [Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(33)]

    assert pose_reliable(landmarks, visibility_threshold=0.5)

    landmarks[11] = Landmark(x=0.0, y=0.0, z=0.0, visibility=0.1)
    assert not pose_reliable(landmarks, visibility_threshold=0.5)
    assert not pose_reliable(landmarks[:4], visibility_threshold=0.5)


def test_reliability_score_returns_mean_visibility():
    landmarks = [Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(33)]

    assert reliability_score(landmarks) == pytest.approx(1.0)

    landmarks[12] = Landmark(x=0.0, y=0.0, z=0.0, visibility=0.2)
    assert reliability_score(landmarks) == pytest.approx(0.8)

    landmarks[24] = Landmark(x=0.0, y=0.0, z=0.0, visibility=float("nan"))
    assert np.isnan(reliability_score(landmarks))


def test_landmark_smoother_is_deterministic(monkeypatch):
    smoother = LandmarkSmoother(alpha=0.5, visibility_threshold=0.5)
    frame1 = [Landmark(0.0, 0.0, 0.0, visibility=1.0)]
    frame2 = [Landmark(1.0, 1.0, 1.0, visibility=1.0)]
    low_vis = [Landmark(10.0, 10.0, 10.0, visibility=0.0)]

    first = smoother(frame1)
    assert first is not None
    assert first[0].x == pytest.approx(0.0)

    second = smoother(frame2)
    assert second is not None
    assert second[0].x == pytest.approx(0.5)
    assert second[0].y == pytest.approx(0.5)
    assert second[0].z == pytest.approx(0.5)

    third = smoother(low_vis)
    assert third is not None
    assert third[0].x == pytest.approx(second[0].x)
    assert third[0].y == pytest.approx(second[0].y)
    assert third[0].z == pytest.approx(second[0].z)

    fourth = smoother(low_vis)
    assert fourth is not None
    assert fourth[0].x == pytest.approx(second[0].x)


def test_landmark_smoother_skips_non_finite_values():
    smoother = LandmarkSmoother(alpha=0.3, visibility_threshold=0.5)
    baseline = [Landmark(0.0, 0.0, 0.0, visibility=1.0)]
    noisy = [Landmark(float("nan"), 5.0, 5.0, visibility=float("nan"))]

    first = smoother(baseline)
    assert first is not None
    assert first[0].x == pytest.approx(0.0)

    second = smoother(noisy)
    assert second is not None
    assert second[0].x == pytest.approx(first[0].x)
    assert second[0].y == pytest.approx(first[0].y)
    assert second[0].z == pytest.approx(first[0].z)
