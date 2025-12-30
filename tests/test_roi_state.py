import numpy as np

from types import SimpleNamespace

import numpy as np

from src.B_pose_estimation.estimators.mediapipe_estimators import (
    _resize_with_letterbox,
    _rescale_landmarks_from_crop,
)
from src.B_pose_estimation.roi_state import RoiState


class DummyNormalizedLandmark:
    def __init__(self, x: float, y: float, z: float, visibility: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class DummyLandmarkList:
    def __init__(self, landmark):
        self.landmark = landmark


def test_roi_state_uses_full_frame_initially():
    state = RoiState(warmup_frames=3, min_fullframe_ratio=1.0)
    roi, warmup = state.next_roi(100, 200)
    assert warmup is True
    assert roi == [0, 0, 100, 200]


def test_roi_state_fallback_after_consecutive_misses():
    state = RoiState(fallback_misses=2, min_fullframe_ratio=1.0)
    state.update_failure(100, 100)
    assert state.last_roi is None or len(state.last_roi) == 4
    state.update_failure(100, 100)
    assert state.last_roi is None
    roi, use_full = state.next_roi(100, 100)
    assert use_full is True
    assert roi == [0, 0, 100, 100]


def test_roi_state_expands_after_failure():
    state = RoiState(fallback_misses=3, expansion_factor=2.0)
    state.update_success((20, 20, 40, 60), width=200, height=200)
    original_roi = state.last_roi
    state.update_failure(200, 200)
    expanded_roi = state.last_roi
    assert expanded_roi is not None and original_roi is not None
    assert expanded_roi[2] - expanded_roi[0] > original_roi[2] - original_roi[0]
    assert expanded_roi[3] - expanded_roi[1] > original_roi[3] - original_roi[1]


def test_vertical_video_roi_has_minimum_size():
    state = RoiState(min_roi_ratio=0.4, min_fullframe_ratio=1.0)
    roi, _ = state.next_roi(120, 240)
    assert roi[2] - roi[0] >= 120
    assert roi[3] - roi[1] >= 240


def test_letterbox_preserves_aspect_ratio():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    image[:, :] = 255
    resized, transform = _resize_with_letterbox(image, (256, 256))
    assert resized.shape == (256, 256, 3)
    non_zero_mask = np.any(resized != 0, axis=2)
    ys, xs = np.where(non_zero_mask)
    height = ys.max() - ys.min() + 1
    width = xs.max() - xs.min() + 1
    assert abs(width / height - 2.0) < 0.1
    assert transform.pad_y > 0 and transform.pad_x == 0


def test_letterbox_inverse_mapping_restores_points():
    width, height = 100, 200
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    crop_box = (0, 0, width, height)
    resized, transform = _resize_with_letterbox(crop, (256, 256))
    assert resized.shape == (256, 256, 3)

    points_crop = [(0.0, 0.0), (50.0, 100.0), (90.0, 150.0)]
    mp_landmarks = []
    for x_crop, y_crop in points_crop:
        x_lb = (x_crop * transform.scale + transform.pad_x) / transform.target_width
        y_lb = (y_crop * transform.scale + transform.pad_y) / transform.target_height
        mp_landmarks.append(SimpleNamespace(x=x_lb, y=y_lb, z=0.1, visibility=1.0))

    landmarks_full, _ = _rescale_landmarks_from_crop(
        mp_landmarks,
        crop_box,
        width,
        height,
        landmark_pb2_module=SimpleNamespace(
            NormalizedLandmark=DummyNormalizedLandmark,
            NormalizedLandmarkList=DummyLandmarkList,
        ),
        letterbox_transform=transform,
    )

    for original, recovered in zip(points_crop, landmarks_full):
        expected_x = original[0] / width
        expected_y = original[1] / height
        assert np.isclose(recovered.x, expected_x, atol=1e-3)
        assert np.isclose(recovered.y, expected_y, atol=1e-3)
        assert recovered.visibility == 1.0
