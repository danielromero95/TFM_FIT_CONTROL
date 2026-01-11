import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.A_preprocessing.frame_extraction.core import extract_frames_stream
from src.A_preprocessing.frame_extraction.preprocess import extract_and_preprocess_frames
from src.A_preprocessing.frame_extraction.state import FrameInfo
from src.A_preprocessing.video_metadata import (
    VideoInfo,
    _normalize_rotation,
    _parse_ffprobe_rotation,
    get_video_metadata,
    read_video_file_info,
)


class _DummyCap:
    def __init__(self):
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self):
        return True

    def release(self):
        self.released = True

    def get(self, prop_id):  # pragma: no cover - overwritten in subclasses
        return 0.0

    def set(self, prop_id, value):
        self.set_calls.append((prop_id, value))


class _InfoCap(_DummyCap):
    def __init__(self, width=1280, height=720, fps=0.0, frame_count=0, fourcc=0):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.fourcc = fourcc

    def get(self, prop_id):
        mapping = {
            cv2.CAP_PROP_FRAME_WIDTH: float(self.width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self.height),
            cv2.CAP_PROP_FPS: float(self.fps),
            cv2.CAP_PROP_FRAME_COUNT: float(self.frame_count),
            cv2.CAP_PROP_FOURCC: float(self.fourcc),
        }
        return mapping.get(prop_id, 0.0)


class _IteratorCap(_DummyCap):
    def __init__(self):
        super().__init__()
        self.reset_to_start = False

    def set(self, prop_id, value):
        super().set(prop_id, value)
        if prop_id == cv2.CAP_PROP_POS_FRAMES and value == 0:
            self.reset_to_start = True


class _PrefetchedCap(_InfoCap):
    def __init__(self):
        super().__init__(fps=30.0, frame_count=4)
        self.release_calls = 0

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return 0.0
        return super().get(prop_id)

    def release(self):
        self.release_calls += 1
        return super().release()


@pytest.mark.parametrize(
    "value,expected",
    [(-90, 270), (450, 90), (179, 180), (225, 180)],
)
def test_normalize_rotation_wraps_values(value, expected):
    assert _normalize_rotation(value) == expected


def test_read_video_file_info_reader_fallback(monkeypatch, tmp_path):
    dummy_path = tmp_path / "no_meta.mp4"
    dummy_path.write_text("placeholder")

    cap = _InfoCap(fps=0.0, frame_count=0)
    monkeypatch.setattr(
        "src.A_preprocessing.video_metadata.cv2.VideoCapture", lambda path: cap
    )
    monkeypatch.setattr(
        "src.A_preprocessing.video_metadata._estimate_duration_seconds", lambda *_: 0.0
    )
    monkeypatch.setattr("src.A_preprocessing.video_metadata._read_rotation_ffprobe", lambda _p: None)

    info = read_video_file_info(dummy_path)

    assert info.fps is None
    assert info.duration_sec is None
    assert info.fps_source == "reader"
    assert info.rotation == 0
    assert cap.released is True


@pytest.mark.parametrize(
    "stream,expected",
    [
        ({"tags": {"rotate": "90"}}, 90),
        ({"side_data_list": [{"rotation": "-90"}]}, 270),
        ({}, None),
    ],
)
def test_parse_ffprobe_rotation_variants(stream, expected):
    assert _parse_ffprobe_rotation(stream) == expected


def test_get_video_metadata_merges_ffprobe(monkeypatch, tmp_path):
    dummy_path = tmp_path / "ffprobe.mp4"
    dummy_path.write_text("placeholder")

    video_info = VideoInfo(
        path=dummy_path,
        width=320,
        height=240,
        fps=29.97,
        frame_count=90,
        duration_sec=3.0,
        rotation=90,
        codec="avc1",
        fps_source="metadata",
    )

    monkeypatch.setattr(
        "src.A_preprocessing.video_metadata.read_video_file_info", lambda path: video_info
    )
    monkeypatch.setattr(
        "src.A_preprocessing.video_metadata._run_ffprobe",
        lambda _p: {
            "format": {
                "format_name": "mov,mp4,m4a",
                "duration": "4.0",
                "size": "1234",
                "tags": {"creation_time": "2023-12-01T00:00:00Z"},
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "codec_long_name": "H.264",
                    "profile": "High",
                    "pix_fmt": "yuv420p",
                    "width": 640,
                    "height": 480,
                    "nb_frames": "100",
                    "r_frame_rate": "30000/1001",
                    "avg_frame_rate": "30/1",
                    "tags": {"rotate": "-90", "creation_time": "2023-12-02T00:00:00Z"},
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        },
    )

    metadata = get_video_metadata(dummy_path)

    assert metadata["video_codec"] == "h264"
    assert metadata["audio_present"] is True
    assert metadata["rotation"] == 270
    assert metadata["fps_r_frame_rate_float"] == pytest.approx(29.97, rel=1e-3)
    assert metadata["fps_avg_frame_rate_float"] == pytest.approx(30.0)
    assert metadata["total_frames_estimated"] == 100
    assert isinstance(metadata["total_frames_estimated"], int)
    assert metadata["file_size_bytes"] == 1234
    assert metadata["creation_time"] == "2023-12-02T00:00:00Z"


def test_extract_frames_stream_prefers_time_mode(monkeypatch, tmp_path):
    dummy_path = tmp_path / "frames.mp4"
    dummy_path.write_text("placeholder")

    cap = _IteratorCap()
    video_info = VideoInfo(
        path=dummy_path,
        width=640,
        height=360,
        fps=25.0,
        frame_count=5,
        duration_sec=0.2,
        rotation=0,
        codec=None,
        fps_source="metadata",
    )

    monkeypatch.setattr(
        "src.A_preprocessing.frame_extraction.core._open_video_capture",
        lambda path_obj, _cap: (cap, True),
    )
    monkeypatch.setattr(
        "src.A_preprocessing.frame_extraction.core._load_video_info", lambda *_: video_info
    )

    yielded = []

    def fake_iterator(context, target_fps):
        yielded.append((context.processor.rotate_deg, target_fps))
        array = np.zeros((1, 1, 3), dtype=np.uint8)
        yield FrameInfo(index=0, timestamp_sec=0.0, array=array, width=1, height=1)

    monkeypatch.setattr(
        "src.A_preprocessing.frame_extraction.core._time_mode_iterator", fake_iterator
    )

    frames = list(
        extract_frames_stream(
            video_path=str(dummy_path),
            sampling="auto",
            target_fps=12.5,
            rotate=0,
            cap=cap,
        )
    )

    assert yielded == [(0, 12.5)]
    assert len(frames) == 1
    assert cap.reset_to_start is True


def test_extract_frames_stream_releases_on_failure(monkeypatch, tmp_path):
    dummy_path = tmp_path / "error.mp4"
    dummy_path.write_text("placeholder")

    cap = _IteratorCap()

    monkeypatch.setattr(
        "src.A_preprocessing.frame_extraction.core._open_video_capture",
        lambda path_obj, _cap: (cap, True),
    )
    monkeypatch.setattr(
        "src.A_preprocessing.frame_extraction.core._load_video_info",
        lambda *_: (_ for _ in ()).throw(ValueError("bad info")),
    )

    with pytest.raises(ValueError):
        list(
            extract_frames_stream(
                video_path=str(dummy_path),
                sampling="index",
                every_n=1,
                rotate=0,
            )
        )

    assert cap.released is True


def test_extract_and_preprocess_frames_prefetched(monkeypatch, tmp_path):
    dummy_path = tmp_path / "clip.mp4"
    dummy_path.write_text("placeholder")

    cap = _PrefetchedCap()
    prefetched = VideoInfo(
        path=dummy_path,
        width=800,
        height=600,
        fps=30.0,
        frame_count=4,
        duration_sec=0.12,
        rotation=0,
        codec="avc1",
        fps_source="metadata",
    )

    arrays = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
    frame_stream = (
        FrameInfo(index=i, timestamp_sec=0.0, array=arr, width=2, height=2)
        for i, arr in enumerate(arrays)
    )

    monkeypatch.setattr(
        "src.A_preprocessing.frame_extraction.preprocess.extract_frames_stream",
        lambda *args, **kwargs: frame_stream,
    )

    frames, fps = extract_and_preprocess_frames(
        str(dummy_path),
        sample_rate=1,
        rotate=None,
        cap=cap,
        prefetched_info=prefetched,
    )

    assert fps == pytest.approx(30.0)
    assert len(frames) == 2
    assert cap.release_calls == 0
