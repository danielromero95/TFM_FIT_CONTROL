# tests/test_pipeline_prefetch.py

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.C_analysis.repetition_counter import CountingDebugInfo

from src import config
from src.A_preprocessing.video_metadata import VideoInfo
from src.C_analysis import run_pipeline
from src.C_analysis.streaming import StreamingPoseResult


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

    class _DummyCap:
        def release(self) -> None:  # pragma: no cover - limpieza de mejor esfuerzo
            pass

        def get(self, _prop) -> float:  # pragma: no cover - stub sencillo
            return 30.0

    monkeypatch.setattr("src.C_analysis.pipeline.open_video_cap", lambda _path: _DummyCap())

    monkeypatch.setattr(
        "src.C_analysis.pipeline.read_info_and_initial_sampling",
        lambda _cap, _video_path: (
            VideoInfo(
                path=Path(_video_path),
                width=640,
                height=480,
                fps=30.0,
                frame_count=600,
                duration_sec=20.0,
                rotation=0,
                codec="H264",
                fps_source="metadata",
            ),
            30.0,
            None,
            False,
        ),
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.extract_frames_stream",
        lambda **_: frames,
    )

    def _fake_stream_pose_and_detection(*_args, **kwargs):
        assert kwargs.get("detection_enabled") is False
        return StreamingPoseResult(
            df_landmarks=pd.DataFrame({"frame_idx": range(len(frames))}),
            frames_processed=len(frames),
            detection=None,
            debug_video_path=None,
            processed_size=(2, 2),
        )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.stream_pose_and_detection",
        _fake_stream_pose_and_detection,
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.filter_landmarks",
        lambda df: (df.copy(), [], pd.Series([True] * len(df)), pd.Series(range(len(df)))),
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.infer_upright_quadrant_from_sequence",
        lambda _sequence: 0,
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.compute_metrics_and_angle",
        lambda *_args, **_kwargs: (
            pd.DataFrame({cfg.counting.primary_angle: [90.0, 80.0, 105.0]}),
            25.0,
            [],
            None,
            cfg.counting.primary_angle,
            pd.Series([0, 1, 2]),
        ),
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.auto_counting_params",
        lambda *_args, **_kwargs: SimpleNamespace(
            min_prominence=1.0,
            min_distance_sec=1.0,
            refractory_sec=0.5,
            cadence_period_sec=None,
            iqr_deg=None,
            multipliers={},
        ),
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.maybe_count_reps",
        lambda *_args, **_kwargs: (2, [], CountingDebugInfo([], [])),
    )

    report = run_pipeline(
        str(video_path),
        cfg,
        prefetched_detection=("squat", "front", 0.9),
    )

    assert report.stats.exercise_detected == "squat"
    assert report.stats.view_detected == "front"
    assert report.stats.detection_confidence == 0.9
