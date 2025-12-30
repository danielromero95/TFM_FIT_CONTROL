import numpy as np

from src.B_pose_estimation.estimators.mediapipe_estimators import _resize_with_letterbox
from src.B_pose_estimation.roi_state import RoiState


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
    resized = _resize_with_letterbox(image, (256, 256))
    assert resized.shape == (256, 256, 3)
    non_zero_mask = np.any(resized != 0, axis=2)
    ys, xs = np.where(non_zero_mask)
    height = ys.max() - ys.min() + 1
    width = xs.max() - xs.min() + 1
    assert abs(width / height - 2.0) < 0.1
