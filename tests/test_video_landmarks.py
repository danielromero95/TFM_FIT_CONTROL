import pytest

np = pytest.importorskip("numpy")

from src.D_visualization.video_landmarks import (
    _normalize_points_for_frame,
    render_landmarks_video,
    transcode_video,
)


def test_mapping_without_crop_matches_expected_pixels():
    landmarks = [
        {"x": 0.0, "y": 0.0},
        {"x": 0.5, "y": 0.5},
        {"x": 1.0, "y": 1.0},
    ]
    points = _normalize_points_for_frame(
        landmarks,
        None,
        orig_w=640,
        orig_h=360,
        proc_w=256,
        proc_h=256,
    )
    assert points[0] == (0, 0)
    assert points[1] == (320, 180)
    assert points[2] == (639, 359)


def test_mapping_with_crop_scales_correctly():
    landmarks = [{"x": 0.5, "y": 0.5}]
    crop_box = [64, 64, 192, 192]
    points = _normalize_points_for_frame(
        landmarks,
        crop_box,
        orig_w=640,
        orig_h=360,
        proc_w=256,
        proc_h=256,
    )
    assert points[0] == (320, 180)


def _make_test_frames(count: int, width: int = 640, height: int = 360):
    for i in range(count):
        frame = np.full((height, width, 3), i * 40, dtype=np.uint8)
        yield frame


def test_writer_fps_fallback_and_counts(tmp_path):
    output = tmp_path / "out.mp4"
    frames = list(_make_test_frames(3))
    stats = render_landmarks_video(
        frames,
        [None, None, None],
        None,
        str(output),
        fps=0.0,
        processed_size=(256, 256),
    )
    assert stats.frames_in == 3
    assert stats.frames_written == 3
    assert stats.skipped_empty == 3
    assert output.exists()
    assert output.stat().st_size > 0


def test_writer_codec_fallback(tmp_path):
    output = tmp_path / "fallback.mp4"
    frames = list(_make_test_frames(2))
    stats = render_landmarks_video(
        frames,
        [None, None],
        None,
        str(output),
        fps=25.0,
        processed_size=(256, 256),
        codec_preference=("ZZZZ", "mp4v"),
    )
    assert stats.used_fourcc == "mp4v"
    assert output.exists()
    assert output.stat().st_size > 0


def test_transcode_video_rewrites_file_when_possible(tmp_path):
    src = tmp_path / "source.mp4"
    frames = list(_make_test_frames(2))
    render_landmarks_video(
        frames,
        [None, None],
        None,
        str(src),
        fps=25.0,
        processed_size=(256, 256),
        codec_preference=("mp4v",),
    )
    dst = tmp_path / "h264.mp4"
    success, codec = transcode_video(
        str(src),
        str(dst),
        fps=25.0,
        codec_preference=("mp4v",),
    )
    assert success
    assert codec == "mp4v"
    assert dst.exists()
    assert dst.stat().st_size > 0
