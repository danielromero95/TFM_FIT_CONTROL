import pytest

pytest.importorskip("numpy")

from B_pose_estimation.geometry import smooth_bounding_box


def test_smooth_bounding_box_initializes_without_previous() -> None:
    bbox = (10.0, 20.0, 60.0, 80.0)
    smoothed = smooth_bounding_box(None, bbox, factor=0.5, width=200, height=200)

    assert smoothed[0] == pytest.approx(bbox[0])
    assert smoothed[1] == pytest.approx(bbox[1])
    assert smoothed[2] == pytest.approx(bbox[2])
    assert smoothed[3] == pytest.approx(bbox[3])


def test_smooth_bounding_box_blends_with_previous() -> None:
    previous = (20.0, 30.0, 100.0, 140.0)
    new = (40.0, 60.0, 160.0, 180.0)

    smoothed = smooth_bounding_box(previous, new, factor=0.5, width=200, height=200)

    assert smoothed[0] == pytest.approx(30.0)
    assert smoothed[1] == pytest.approx(45.0)
    assert smoothed[2] == pytest.approx(130.0)
    assert smoothed[3] == pytest.approx(160.0)


def test_smooth_bounding_box_clips_to_frame() -> None:
    previous = (180.0, 170.0, 240.0, 260.0)
    new = (220.0, 210.0, 280.0, 310.0)

    smoothed = smooth_bounding_box(previous, new, factor=0.5, width=256, height=256)

    assert 0.0 <= smoothed[0] <= 256.0
    assert 0.0 <= smoothed[1] <= 256.0
    assert 0.0 <= smoothed[2] <= 256.0
    assert 0.0 <= smoothed[3] <= 256.0
    assert smoothed[2] > smoothed[0]
    assert smoothed[3] > smoothed[1]
