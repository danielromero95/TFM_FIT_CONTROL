from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from src.B_pose_estimation.estimators.mediapipe_estimators import RoiPoseEstimator
from src.B_pose_estimation.estimators import mediapipe_pool


class _DummyNormalizedLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class _DummyNormalizedLandmarkList:
    def __init__(self, landmark):
        self.landmark = list(landmark)


class _RecordingPose:
    def __init__(self, sequences):
        self._sequences = list(sequences)

    def process(self, _image):
        try:
            lms = self._sequences.pop(0)
        except IndexError:
            lms = None
        if lms is None:
            return SimpleNamespace(pose_landmarks=None)
        return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=lms))


@pytest.fixture(autouse=True)
def _restore_pool(monkeypatch):
    monkeypatch.setattr(mediapipe_pool.PoseGraphPool, "_imported", True)
    monkeypatch.setattr(mediapipe_pool.PoseGraphPool, "_all", [])
    monkeypatch.setattr(mediapipe_pool.PoseGraphPool, "_free", {})
    monkeypatch.setattr(mediapipe_pool.PoseGraphPool, "_lock", mediapipe_pool.threading.Lock())
    yield


def test_roi_estimator_scales_and_draws_cropped_landmarks(monkeypatch):
    drawn = {}

    def _draw_landmarks(image, landmark_list, _connections):
        drawn["shape"] = tuple(image.shape)
        drawn["list_type"] = type(landmark_list)

    fake_pb2 = SimpleNamespace(
        NormalizedLandmark=_DummyNormalizedLandmark, NormalizedLandmarkList=_DummyNormalizedLandmarkList
    )

    # Intercept Mediapipe imports so that the estimator uses lightweight doubles.
    fake_mediapipe = ModuleType("mediapipe")
    fake_framework = ModuleType("mediapipe.framework")
    fake_formats = ModuleType("mediapipe.framework.formats")
    fake_formats.landmark_pb2 = fake_pb2
    fake_framework.formats = fake_formats
    fake_mediapipe.framework = fake_framework

    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)
    monkeypatch.setitem(sys.modules, "mediapipe.framework", fake_framework)
    monkeypatch.setitem(sys.modules, "mediapipe.framework.formats", fake_formats)
    monkeypatch.setitem(sys.modules, "mediapipe.framework.formats.landmark_pb2", fake_pb2)
    monkeypatch.setattr(mediapipe_pool.PoseGraphPool, "_ensure_imports", lambda: None)
    monkeypatch.setattr(mediapipe_pool.PoseGraphPool, "mp_drawing", SimpleNamespace(draw_landmarks=_draw_landmarks))
    monkeypatch.setattr(mediapipe_pool.PoseGraphPool, "mp_pose", SimpleNamespace())

    full_landmarks = [[_DummyNormalizedLandmark(0.25, 0.25), _DummyNormalizedLandmark(0.75, 0.75)]]
    crop_landmarks = [[_DummyNormalizedLandmark(0.0, 0.0), _DummyNormalizedLandmark(1.0, 1.0)]]
    poses = [_RecordingPose(full_landmarks), _RecordingPose(crop_landmarks)]

    def _fake_acquire(**_kwargs):
        if not poses:
            pytest.fail("No more fake poses available")
        return poses.pop(0), (False, len(poses), 0.0, 0.0)

    monkeypatch.setattr(mediapipe_pool.PoseGraphPool, "acquire", _fake_acquire)

    estimator = RoiPoseEstimator(crop_margin=0.0, refresh_period=10, max_misses=2)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    first = estimator.estimate(frame)
    assert first.crop_box is not None

    second = estimator.estimate(frame)
    assert second.landmarks is not None
    assert len(second.landmarks) == 2

    assert second.landmarks[0].x == pytest.approx(0.25)
    assert second.landmarks[0].y == pytest.approx(0.25)
    assert second.landmarks[1].x == pytest.approx(0.75)
    assert second.landmarks[1].y == pytest.approx(0.75)

    assert drawn["shape"] == frame.shape
    assert drawn["list_type"] is _DummyNormalizedLandmarkList
