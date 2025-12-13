from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

np = pytest.importorskip("numpy")

from src.D_visualization import render_landmarks_video, transcode_video
from src.D_visualization.landmark_video_io import WEB_SAFE_SUFFIX, make_web_safe_h264
from src.D_visualization.landmark_geometry import _normalize_points_for_frame


class _RecordingWriter:
    def __init__(self, output_path: Path, code: str) -> None:
        self.output_path = Path(output_path)
        self.output_path.write_bytes(b"stub")
        self.code = code
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame))

    def release(self) -> None:  # pragma: no cover - nada que limpiar
        pass


class _WriterFactory:
    def __init__(self, *, supported: set[str] | None = None) -> None:
        self.supported = supported or {"mp4v"}
        self.created: list[_RecordingWriter] = []

    def open(self, path: str, fps: float, size, prefs):
        for code in prefs:
            if code not in self.supported:
                continue
            writer = _RecordingWriter(Path(path), code)
            self.created.append(writer)
            return writer, code
        raise RuntimeError(f"No supported codec in {prefs}")


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


def test_pixel_coordinates_without_crop_are_scaled():
    landmarks = [{"x": 128.0, "y": 90.0}]
    points = _normalize_points_for_frame(
        landmarks,
        None,
        orig_w=640,
        orig_h=360,
        proc_w=256,
        proc_h=256,
    )

    assert points[0] == (320, 127)


def _make_test_frames(count: int, width: int = 640, height: int = 360):
    for i in range(count):
        frame = np.full((height, width, 3), i * 40, dtype=np.uint8)
        yield frame


def test_writer_fps_fallback_and_counts(tmp_path, monkeypatch):
    output = tmp_path / "out.mp4"
    frames = list(_make_test_frames(3))
    factory = _WriterFactory()
    monkeypatch.setattr(
        "src.D_visualization.landmark_renderers._open_writer",
        lambda path, fps, size, prefs: factory.open(path, fps, size, prefs),
    )

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
    assert factory.created[0].frames and len(factory.created[0].frames) == 3


def test_writer_codec_fallback(tmp_path, monkeypatch):
    output = tmp_path / "fallback.mp4"
    frames = list(_make_test_frames(2))
    factory = _WriterFactory(supported={"mp4v"})
    monkeypatch.setattr(
        "src.D_visualization.landmark_renderers._open_writer",
        lambda path, fps, size, prefs: factory.open(path, fps, size, prefs),
    )

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
    assert factory.created[0].code == "mp4v"


def test_transcode_video_rewrites_file_when_possible(tmp_path, monkeypatch):
    src = tmp_path / "source.mp4"
    frames = list(_make_test_frames(2))
    render_factory = _WriterFactory(supported={"mp4v"})
    monkeypatch.setattr(
        "src.D_visualization.landmark_renderers._open_writer",
        lambda path, fps, size, prefs: render_factory.open(path, fps, size, prefs),
    )
    render_landmarks_video(
        frames,
        [None, None],
        None,
        str(src),
        fps=25.0,
        processed_size=(256, 256),
        codec_preference=("mp4v",),
    )

    class _FakeCapture:
        def __init__(self, _path: str) -> None:
            self._frames = iter(frames)

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float(frames[0].shape[1])
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(frames[0].shape[0])
            return 0.0

        def read(self):
            try:
                frame = next(self._frames)
            except StopIteration:
                return False, None
            return True, frame

        def release(self) -> None:  # pragma: no cover - nada que limpiar
            pass

    transcode_factory = _WriterFactory(supported={"mp4v"})
    monkeypatch.setattr(
        "src.D_visualization.landmark_video_io._open_writer",
        lambda path, fps, size, prefs: transcode_factory.open(path, fps, size, prefs),
    )
    monkeypatch.setattr("src.D_visualization.landmark_video_io.cv2.VideoCapture", _FakeCapture)

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
    assert transcode_factory.created[0].frames


def test_make_web_safe_h264_generates_copy_when_ffmpeg_succeeds(tmp_path, monkeypatch):
    source = tmp_path / "overlay.mp4"
    source.write_bytes(b"raw")

    created_commands: list[list[str]] = []

    def _fake_run(cmd, check, capture_output):
        created_commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"websafe")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        "src.D_visualization.landmark_video_io.subprocess.run",
        _fake_run,
    )

    result = make_web_safe_h264(source)
    expected_path = source.with_name(f"{source.stem}{WEB_SAFE_SUFFIX}.mp4")

    assert result.ok
    assert result.output_path == expected_path
    assert expected_path.exists() and expected_path.read_bytes() == b"websafe"
    assert created_commands and created_commands[0][-1].endswith(".tmp")
    assert created_commands[0][-3:-1] == ["-f", "mp4"]


def test_make_web_safe_h264_handles_missing_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "overlay.mp4"
    source.write_bytes(b"raw")

    def _missing_run(*_args, **_kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(
        "src.D_visualization.landmark_video_io.subprocess.run",
        _missing_run,
    )

    result = make_web_safe_h264(source)
    expected_path = source.with_name(f"{source.stem}{WEB_SAFE_SUFFIX}.mp4")

    assert not result.ok
    assert result.output_path is None
    assert not expected_path.exists()
