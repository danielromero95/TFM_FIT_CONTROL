# tests/test_pipeline_prefetch.py

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.C_repetition_analysis.repetition_counter import CountingDebugInfo
from src.services.analysis_service import run_pipeline
from src.A_preprocessing.video_metadata import VideoInfo


def test_run_pipeline_uses_prefetched_detection(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video content")

    cfg = config.load_default()
    cfg.output.base_dir = tmp_path / "outputs"
    cfg.output.counts_dir = tmp_path / "counts"
    cfg.output.poses_dir = tmp_path / "poses"
    cfg.debug.generate_debug_video = False
    cfg.debug.debug_mode = False
    cfg.counting.exercise = "auto"

    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(20)]

    monkeypatch.setattr(
        "src.services.analysis_service.read_video_file_info",
        lambda *args, **kwargs: VideoInfo(
            path=Path(args[0] if args else kwargs.get("path", video_path)),
            width=640,
            height=480,
            fps=30.0,
            frame_count=600,
            duration_sec=20.0,
            rotation=0,
            codec="H264",
            fps_source="metadata",
        ),
    )

    monkeypatch.setattr(
        "src.services.analysis_service.extract_and_preprocess_frames",
        lambda **kwargs: (frames, 30.0),
    )

    def _fake_extract_landmarks(
        *, frames, use_crop, min_detection_confidence, min_visibility
    ):  # type: ignore[override]
        length = len(frames)
        return pd.DataFrame({"frame": range(length)})

    monkeypatch.setattr(
        "src.services.analysis_service.extract_landmarks_from_frames",
        _fake_extract_landmarks,
    )

    def _fake_filter(df_raw_landmarks):  # type: ignore[override]
        return df_raw_landmarks.copy(), []

    monkeypatch.setattr(
        "src.services.analysis_service.filter_and_interpolate_landmarks",
        _fake_filter,
    )

    def _fake_metrics(_sequence, _fps):
        return pd.DataFrame({cfg.counting.primary_angle: [90.0, 80.0, 105.0]})

    monkeypatch.setattr(
        "src.services.analysis_service.calculate_metrics_from_sequence",
        _fake_metrics,
    )

    monkeypatch.setattr(
        "src.services.analysis_service.count_repetitions_with_config",
        lambda df, counting_cfg, fps: (2, CountingDebugInfo(valley_indices=[1], prominences=[1.0])),
    )

    def _fail_detect(*args, **kwargs):  # pragma: no cover - should not be invoked
        raise AssertionError("detect_exercise should not be called when prefetched")

    monkeypatch.setattr("src.services.analysis_service.detect_exercise", _fail_detect)

    report = run_pipeline(
        str(video_path),
        cfg,
        prefetched_detection=("squat", "front", 0.9),
    )

    assert report.stats.exercise_detected == "squat"
    assert report.stats.view_detected == "front"
    assert report.stats.detection_confidence == 0.9
    assert report.stats.exercise_selected == cfg.counting.exercise
