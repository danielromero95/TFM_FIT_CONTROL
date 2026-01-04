import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.A_preprocessing.frame_extraction.state import _FrameProcessor, _IteratorContext, _ProgressHandler
from src.A_preprocessing.frame_extraction.time_sampling import _time_mode_iterator
from src.A_preprocessing.frame_extraction.validation import _validate_sampling_args
from src.A_preprocessing.video_metadata import read_video_file_info


class _FakeCap:
    def __init__(self, grab_timestamps: list[float], read_timestamps: list[float]):
        self._grab_ts = grab_timestamps
        self._read_ts = read_timestamps
        self.grab_idx = 0
        self.read_idx = 0
        self.last_ts = 0.0
        self.set_calls: list[tuple[int | float, float]] = []

    def isOpened(self) -> bool:  # pragma: no cover - trivial
        return True

    def release(self) -> None:  # pragma: no cover - trivial
        return None

    def grab(self) -> bool:
        if self.grab_idx < len(self._grab_ts):
            self.last_ts = self._grab_ts[self.grab_idx]
            self.grab_idx += 1
            return True
        return False

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        return False, None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.read_idx < len(self._read_ts):
            self.last_ts = self._read_ts[self.read_idx]
            self.read_idx += 1
            return True, np.ones((3, 3, 3), dtype=np.uint8)
        return False, None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return float(self.last_ts)
        return 0.0

    def set(self, prop_id: int, value: float) -> None:
        self.set_calls.append((prop_id, value))


class _MetadataCap:
    def __init__(self, fps: float, frame_count: int, width: int = 640, height: int = 480):
        self.fps = fps
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        return None

    def get(self, prop_id: int) -> float:
        mapping: dict[int, float] = {
            cv2.CAP_PROP_FRAME_WIDTH: float(self.width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self.height),
            cv2.CAP_PROP_FPS: float(self.fps),
            cv2.CAP_PROP_FRAME_COUNT: float(self.frame_count),
            cv2.CAP_PROP_POS_FRAMES: 0.0,
        }
        return mapping.get(prop_id, 0.0)

    def set(self, prop_id: int, value: float) -> None:
        self.set_calls.append((prop_id, value))

    def grab(self) -> bool:
        return False


def test_time_mode_iterator_falls_back_to_index():
    fake_cap = _FakeCap(grab_timestamps=[0.0] * 6, read_timestamps=[0.0, 33.0])
    processor = _FrameProcessor(rotate_deg=0, resize_to=None, to_gray=False)
    context = _IteratorContext(
        cap=fake_cap,  # type: ignore[arg-type]
        fps_base=30.0,
        frame_count=10,
        start_time=0.0,
        end_time=None,
        max_frames=1,
        processor=processor,
        progress=_ProgressHandler(callback=None, frame_count=10),
        read_idx=0,
    )

    frames = list(_time_mode_iterator(context, target_fps=30.0))

    assert len(frames) == 1
    assert frames[0].index >= 5
    assert fake_cap.grab_idx >= 5  # consumed enough bad timestamps before falling back
    assert fake_cap.read_idx == 1  # index iterator retrieved the first available frame


def test_read_video_file_info_uses_metadata(monkeypatch, tmp_path):
    dummy_path = tmp_path / "dummy.mp4"
    dummy_path.write_text("placeholder")

    cap = _MetadataCap(fps=25.0, frame_count=50)
    monkeypatch.setattr("src.A_preprocessing.video_metadata.cv2.VideoCapture", lambda path: cap)
    monkeypatch.setattr("src.A_preprocessing.video_metadata._read_rotation_ffprobe", lambda _path: 0)

    info = read_video_file_info(dummy_path)

    assert info.fps == 25.0
    assert info.frame_count == 50
    assert info.width == 640
    assert info.height == 480
    assert info.fps_source == "metadata"


def test_read_video_file_info_estimates_duration(monkeypatch, tmp_path):
    dummy_path = tmp_path / "needs_estimate.mp4"
    dummy_path.write_text("placeholder")

    cap = _MetadataCap(fps=0.5, frame_count=100)
    monkeypatch.setattr("src.A_preprocessing.video_metadata.cv2.VideoCapture", lambda path: cap)
    monkeypatch.setattr("src.A_preprocessing.video_metadata._read_rotation_ffprobe", lambda _path: 0)
    monkeypatch.setattr("src.A_preprocessing.video_metadata._estimate_duration_seconds", lambda _cap, _fc: 5.0)

    info = read_video_file_info(dummy_path)

    assert info.fps_source == "estimated"
    assert info.duration_sec == pytest.approx(5.0)
    assert info.fps == pytest.approx(20.0)


def test_validate_sampling_args_requires_target_fps():
    with pytest.raises(ValueError):
        _validate_sampling_args("time", target_fps=None, every_n=None, start_time=None, end_time=None)

    sampling, target_fps, every_n = _validate_sampling_args(
        "index", target_fps=0, every_n=2, start_time=0.0, end_time=None
    )
    assert sampling == "index"
    assert target_fps is None
    assert every_n == 2
