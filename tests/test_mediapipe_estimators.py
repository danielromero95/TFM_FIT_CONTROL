from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

import src.B_pose_estimation.estimators.mediapipe_estimators as mediapipe_estimators
from src.B_pose_estimation.estimators.mediapipe_estimators import (
    CroppedPoseEstimator,
    LandmarkSmoother,
    LetterboxTransform,
    PoseEstimator,
    _resize_with_letterbox,
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


def test_resize_with_letterbox_returns_image_and_transform():
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    resized, transform = _resize_with_letterbox(image, (4, 4))

    assert resized.shape == (4, 4, 3)
    assert transform is not None
    assert transform.target_width == 4
    assert transform.target_height == 4
    assert math.isclose(transform.scale, 2.0)
    assert transform.pad_x == 0
    assert transform.pad_y == 0


def test_rescale_landmarks_from_letterboxed_crop():
    transform = LetterboxTransform(
        target_width=256,
        target_height=256,
        scale=1.28,
        pad_x=0,
        pad_y=64,
    )
    crop_box = (10, 20, 210, 120)
    landmark = _FakeNormalizedLandmark(0.5, 0.5, z=0.2, visibility=0.8)

    landmarks, _ = _rescale_landmarks_from_crop(
        [landmark],
        crop_box,
        image_width=400,
        image_height=300,
        landmark_pb2_module=_FakePb2,
        letterbox_transform=transform,
    )

    assert math.isclose(landmarks[0].x, 110 / 400)
    assert math.isclose(landmarks[0].y, 70 / 300)
    assert math.isclose(landmarks[0].z, 0.2)
    assert math.isclose(landmarks[0].visibility, 0.8)


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


def test_cropped_estimator_passes_letterbox_transform(monkeypatch):
    _install_fake_mediapipe(monkeypatch)

    dummy_transform = LetterboxTransform(
        target_width=4, target_height=2, scale=0.5, pad_x=1, pad_y=0
    )
    captured: dict[str, object] = {}

    def _fake_resize(image, target_size):
        return np.zeros((target_size[1], target_size[0], 3), dtype=image.dtype), dummy_transform

    def _fake_rescale(mp_landmarks, crop_box, image_width, image_height, landmark_pb2_module, letterbox_transform=None):
        captured["letterbox_transform"] = letterbox_transform
        return [Landmark(0.0, 0.0, 0.0, visibility=1.0)], _FakeNormalizedLandmarkList(
            [_FakeNormalizedLandmark(0.0, 0.0)]
        )

    poses = [_FakePose([_FakeNormalizedLandmark(0.25, 0.25, visibility=1.0)]), _FakePose([_FakeNormalizedLandmark(0.5, 0.5, visibility=1.0)])]

    monkeypatch.setattr(mediapipe_estimators, "_resize_with_letterbox", _fake_resize)
    monkeypatch.setattr(mediapipe_estimators, "_rescale_landmarks_from_crop", _fake_rescale)
    monkeypatch.setattr(
        mediapipe_estimators, "bounding_box_from_landmarks", lambda *_args, **_kwargs: (0, 0, 4, 2)
    )
    monkeypatch.setattr(mediapipe_estimators, "smooth_bounding_box", lambda *_args, **_kwargs: (0, 0, 4, 2))
    monkeypatch.setattr(mediapipe_estimators, "expand_and_clip_box", lambda bbox, *_args, **_kwargs: bbox)

    def _fake_acquire(**_kwargs):
        return poses.pop(0), (False, 2, 0.5, 0.5)

    drawing = _FakeDrawing()
    monkeypatch.setattr(PoseGraphPool, "acquire", classmethod(lambda cls, **kwargs: _fake_acquire(**kwargs)))
    monkeypatch.setattr(PoseGraphPool, "release", classmethod(lambda cls, inst, key: None))
    monkeypatch.setattr(PoseGraphPool, "_imported", True)
    monkeypatch.setattr(PoseGraphPool, "_ensure_imports", classmethod(lambda cls: None))
    monkeypatch.setattr(PoseGraphPool, "mp_pose", types.SimpleNamespace(Pose=None))
    monkeypatch.setattr(PoseGraphPool, "mp_drawing", drawing)

    estimator = CroppedPoseEstimator(target_size=(4, 2), crop_margin=0.0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    estimator.estimate(image)

    assert captured["letterbox_transform"] is dummy_transform


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


def test_roi_estimator_passes_letterbox_transform(monkeypatch):
    _install_fake_mediapipe(monkeypatch)

    dummy_transform = LetterboxTransform(
        target_width=6, target_height=4, scale=0.5, pad_x=1, pad_y=1
    )
    captured: dict[str, object] = {}

    def _fake_resize(image, target_size):
        return np.zeros((target_size[1], target_size[0], 3), dtype=image.dtype), dummy_transform

    def _fake_rescale(mp_landmarks, crop_box, image_width, image_height, landmark_pb2_module, letterbox_transform=None):
        captured["letterbox_transform"] = letterbox_transform
        return [Landmark(0.0, 0.0, 0.0, visibility=1.0)], _FakeNormalizedLandmarkList(
            [_FakeNormalizedLandmark(0.0, 0.0)]
        )

    poses = [_FakePose([_FakeNormalizedLandmark(0.2, 0.2, visibility=1.0)]), _FakePose([_FakeNormalizedLandmark(0.4, 0.4, visibility=1.0)])]

    monkeypatch.setattr(mediapipe_estimators, "_resize_with_letterbox", _fake_resize)
    monkeypatch.setattr(mediapipe_estimators, "_rescale_landmarks_from_crop", _fake_rescale)
    monkeypatch.setattr(mediapipe_estimators, "_process_with_recovery", lambda pose_graph, *_args, **_kwargs: pose_graph.process(None))
    monkeypatch.setattr(PoseGraphPool, "acquire", classmethod(lambda cls, **kwargs: (poses.pop(0), (False, 2, 0.5, 0.5))))
    monkeypatch.setattr(PoseGraphPool, "release", classmethod(lambda cls, inst, key: None))
    monkeypatch.setattr(PoseGraphPool, "_imported", True)
    monkeypatch.setattr(PoseGraphPool, "_ensure_imports", classmethod(lambda cls: None))
    monkeypatch.setattr(PoseGraphPool, "mp_pose", types.SimpleNamespace(Pose=None))
    monkeypatch.setattr(PoseGraphPool, "mp_drawing", _FakeDrawing())

    estimator = mediapipe_estimators.RoiPoseEstimator(target_size=(6, 4), refresh_period=100, warmup_frames=0, max_misses=1)
    estimator.frame_idx = 1
    estimator.roi_state.next_roi = lambda _w, _h: ([0, 0, 3, 3], False)

    image = np.zeros((6, 6, 3), dtype=np.uint8)

    estimator.estimate(image)

    assert captured["letterbox_transform"] is dummy_transform


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


def test_pose_estimator_draws_smoothed_landmarks_when_smoothing_enabled(monkeypatch):
    _install_fake_mediapipe(monkeypatch)

    drawing = _FakeDrawing()

    class _SequentialPose:
        def __init__(self):
            self._frames = [
                [_FakeNormalizedLandmark(0.0, 0.0, visibility=1.0)],
                [_FakeNormalizedLandmark(1.0, 1.0, visibility=1.0)],
            ]

        def process(self, _image):
            landmarks = self._frames.pop(0)
            return types.SimpleNamespace(
                pose_landmarks=_FakeNormalizedLandmarkList(landmarks)
            )

    def _fake_acquire(**kwargs):
        return _SequentialPose(), (False, 2, 0.6, 0.6, True, False)

    monkeypatch.setattr(PoseGraphPool, "acquire", classmethod(lambda cls, **kwargs: _fake_acquire(**kwargs)))
    monkeypatch.setattr(PoseGraphPool, "release", classmethod(lambda cls, inst, key: None))
    monkeypatch.setattr(PoseGraphPool, "_imported", True)
    monkeypatch.setattr(PoseGraphPool, "_ensure_imports", classmethod(lambda cls: None))
    monkeypatch.setattr(PoseGraphPool, "mp_pose", types.SimpleNamespace(Pose=None))
    monkeypatch.setattr(PoseGraphPool, "mp_drawing", drawing)

    estimator = PoseEstimator(landmark_smoothing_alpha=0.5)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    estimator.estimate(image)
    estimator.estimate(image)

    assert drawing.calls, "Landmarks should be drawn"
    smoothed_landmarks = drawing.calls[-1][1]
    assert isinstance(smoothed_landmarks, _FakeNormalizedLandmarkList)
    assert smoothed_landmarks.landmark[0].x == pytest.approx(0.5)
    assert smoothed_landmarks.landmark[0].y == pytest.approx(0.5)


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


def test_roi_estimator_draws_smoothed_landmarks_when_smoothing_enabled(monkeypatch):
    _install_fake_mediapipe(monkeypatch)

    drawing = _FakeDrawing()

    class _SequentialPose:
        def __init__(self, sequence):
            self._sequence = list(sequence)

        def process(self, _image):
            landmarks = self._sequence.pop(0)
            return types.SimpleNamespace(
                pose_landmarks=_FakeNormalizedLandmarkList(landmarks)
            )

    full_pose = _SequentialPose(
        [[_FakeNormalizedLandmark(0.25, 0.25, visibility=1.0)]]
    )
    crop_pose = _SequentialPose(
        [
            [_FakeNormalizedLandmark(0.0, 0.0, visibility=1.0)],
            [_FakeNormalizedLandmark(1.0, 1.0, visibility=1.0)],
        ]
    )

    poses = [full_pose, crop_pose]

    def _fake_acquire(**_kwargs):
        return poses.pop(0), (False, 2, 0.5, 0.5)

    monkeypatch.setattr(PoseGraphPool, "acquire", classmethod(lambda cls, **kwargs: _fake_acquire(**kwargs)))
    monkeypatch.setattr(PoseGraphPool, "release", classmethod(lambda cls, inst, key: None))
    monkeypatch.setattr(PoseGraphPool, "_imported", True)
    monkeypatch.setattr(PoseGraphPool, "_ensure_imports", classmethod(lambda cls: None))
    monkeypatch.setattr(PoseGraphPool, "mp_pose", types.SimpleNamespace(Pose=None))
    monkeypatch.setattr(PoseGraphPool, "mp_drawing", drawing)
    monkeypatch.setattr(
        mediapipe_estimators, "_process_with_recovery", lambda pose_graph, *_args, **_kwargs: pose_graph.process(None)
    )
    monkeypatch.setattr(
        mediapipe_estimators, "bounding_box_from_landmarks", lambda *_args, **_kwargs: (0, 0, 4, 4)
    )
    monkeypatch.setattr(
        mediapipe_estimators, "smooth_bounding_box", lambda _prev, bbox, *_args, **_kwargs: bbox
    )
    monkeypatch.setattr(mediapipe_estimators, "expand_and_clip_box", lambda bbox, *_args, **_kwargs: bbox)
    monkeypatch.setattr(
        mediapipe_estimators,
        "_resize_with_letterbox",
        lambda image, target_size: (np.zeros((target_size[1], target_size[0], 3), dtype=image.dtype), None),
    )

    estimator = mediapipe_estimators.RoiPoseEstimator(
        target_size=(4, 4), refresh_period=100, warmup_frames=0, max_misses=1, landmark_smoothing_alpha=0.5
    )
    estimator.frame_idx = 1
    estimator.roi_state.next_roi = lambda _w, _h: ([0, 0, 4, 4], False)

    image = np.zeros((4, 4, 3), dtype=np.uint8)

    estimator.estimate(image)
    estimator.estimate(image)

    assert len(drawing.calls) == 2
    smoothed_landmarks = drawing.calls[-1][1]
    assert isinstance(smoothed_landmarks, _FakeNormalizedLandmarkList)
    assert smoothed_landmarks.landmark[0].x == pytest.approx(0.5)
    assert smoothed_landmarks.landmark[0].y == pytest.approx(0.5)
