from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.B_pose_estimation.estimators.mediapipe_estimators import (
    _remap_landmarks_from_padded_result,
    _resize_with_padding,
)


def test_resize_with_padding_preserves_aspect_ratio() -> None:
    crop = np.zeros((200, 100, 3), dtype=np.uint8)
    target_size = (256, 256)

    padded, scale, pad_x, pad_y = _resize_with_padding(crop, target_size)

    assert padded.shape[:2] == target_size[::-1]
    assert scale == pytest.approx(1.28)
    assert pad_x == 64
    assert pad_y == 0


def test_remap_landmarks_to_crop_space_undoes_padding() -> None:
    crop = np.zeros((200, 100, 3), dtype=np.uint8)
    target_size = (256, 256)
    padded, scale, pad_x, pad_y = _resize_with_padding(crop, target_size)

    assert padded.shape[:2] == (256, 256)

    lm = SimpleNamespace(x=0.5, y=0.5, z=0.25, visibility=0.9)
    remapped = _remap_landmarks_from_padded_result(
        [lm],
        crop_box=(10, 20, 110, 220),
        target_size=target_size,
        pad_x=pad_x,
        pad_y=pad_y,
        scale=scale,
        frame_width=640,
        frame_height=480,
        to_frame=False,
    )

    assert len(remapped) == 1
    assert remapped[0].x == pytest.approx(0.5)
    assert remapped[0].y == pytest.approx(0.5)
    # z debe reescalar en proporción al factor aplicado al recorte original.
    assert remapped[0].z == pytest.approx(lm.z / scale)


def test_remap_landmarks_to_frame_space() -> None:
    crop = np.zeros((200, 100, 3), dtype=np.uint8)
    target_size = (256, 256)
    _, scale, pad_x, pad_y = _resize_with_padding(crop, target_size)

    lm = SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0)
    mapped = _remap_landmarks_from_padded_result(
        [lm],
        crop_box=(100, 50, 200, 250),
        target_size=target_size,
        pad_x=pad_x,
        pad_y=pad_y,
        scale=scale,
        frame_width=400,
        frame_height=400,
        to_frame=True,
    )

    assert len(mapped) == 1
    expected_x_crop = ((lm.x * target_size[0]) - pad_x) / scale
    expected_y_crop = ((lm.y * target_size[1]) - pad_y) / scale
    expected_x_full = (expected_x_crop + 100) / 400.0
    expected_y_full = (expected_y_crop + 50) / 400.0
    assert mapped[0].x == pytest.approx(expected_x_full)
    assert mapped[0].y == pytest.approx(expected_y_full)

