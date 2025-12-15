from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import config
from src.A_preprocessing.video_metadata import VideoInfo
from src.C_analysis import run_pipeline
from src.C_analysis.repetition_counter import CountingDebugInfo
from src.C_analysis.streaming import StreamingPoseResult
from src.exercise_detection.types import DetectionResult


def test_run_pipeline_exposes_debug_summary(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video content")

    cfg = config.load_default()
    cfg.output.base_dir = tmp_path / "outputs"
    cfg.output.counts_dir = tmp_path / "counts"
    cfg.output.poses_dir = tmp_path / "poses"
    cfg.debug.generate_debug_video = False
    cfg.debug.debug_mode = False
    cfg.counting.exercise = "auto"

    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(10)]

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
        "src.C_analysis.pipeline.extract_processed_frames_stream",
        lambda **_: frames,
    )

    detection = DetectionResult(
        label="squat",
        view="front",
        confidence=0.85,
        side="left",
        view_stats={
            "reliable_frames": np.int64(8),
            "total_frames_sampled": np.int64(9),
            "lateral_score": np.float64(0.2),
        },
        debug={
            "features_summary": {
                "knee_angle_left": {"min": np.float64(1.0), "valid_fraction": np.float32(1.0)}
            },
            "classification_scores": {
                "raw": {"squat": np.float32(0.2), "deadlift": np.float32(0.1)},
                "adjusted": {"squat": np.float64(0.3)},
                "penalties": {"squat": np.float32(0.1)},
                "deadlift_veto": np.bool_(False),
            },
            "vetoes": {"deadlift_veto": np.bool_(False)},
            "rep_count": np.int32(2),
        },
    )

    def _fake_stream_pose_and_detection(*_args, **kwargs):
        assert kwargs.get("detection_enabled") is True
        return StreamingPoseResult(
            df_landmarks=pd.DataFrame({"frame_idx": range(len(frames))}),
            frames_processed=len(frames),
            detection=detection,
            debug_video_path=None,
            processed_size=(2, 2),
        )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.stream_pose_and_detection",
        _fake_stream_pose_and_detection,
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.filter_landmarks",
        lambda df: (df.copy(), [], pd.Series([True] * len(df))),
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.infer_upright_quadrant_from_sequence",
        lambda _sequence: 0,
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.compute_metrics_and_angle",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    cfg.counting.primary_angle: [90.0, 80.0, 105.0, np.nan],
                    "pose_ok": [0.6, 0.4, 0.9, 0.2],
                }
            ),
            25.0,
            [],
            None,
            cfg.counting.primary_angle,
        ),
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.auto_counting_params",
        lambda *_args, **_kwargs: SimpleNamespace(
            min_prominence=np.float64(1.0),
            min_distance_sec=np.float64(1.0),
            refractory_sec=np.float64(0.5),
            cadence_period_sec=np.float64(2.0),
            iqr_deg=np.float64(10.0),
            multipliers={"pose_ok": np.float64(1.0)},
        ),
    )

    debug_info = CountingDebugInfo(valley_indices=[1, 3], prominences=[2.0, 2.5])

    monkeypatch.setattr(
        "src.C_analysis.pipeline.maybe_count_reps",
        lambda *_args, **kwargs: (2, [], debug_info) if kwargs.get("return_debug") else (2, []),
    )

    report = run_pipeline(
        str(video_path),
        cfg,
    )

    assert report.debug_summary is not None
    json.dumps(report.debug_summary)  # debe ser serializable

    exercise = report.debug_summary.get("exercise", {})
    assert exercise.get("label") == "squat"
    assert exercise.get("view") == "front"
    assert exercise.get("confidence") == 0.85
    assert exercise.get("view_side") == "left"

    detection_summary = report.debug_summary.get("detection", {})
    assert detection_summary.get("source") == "incremental"
    assert detection_summary.get("view_side") == "left"
    exercise_detection_debug = detection_summary.get("exercise_detection_debug", {})
    assert isinstance(exercise_detection_debug, dict)
    features_summary = exercise_detection_debug.get("features_summary")
    assert isinstance(features_summary, dict)
    summary_values = features_summary.get("knee_angle_left", {})
    assert all(
        isinstance(val, (float, int, bool, type(None))) for val in summary_values.values()
    )
    classification_scores = exercise_detection_debug.get("classification_scores", {})
    assert isinstance(classification_scores, dict)
    assert all(
        isinstance(val, (float, int, bool, type(None), dict))
        for val in classification_scores.get("raw", {}).values()
    )
    assert isinstance(exercise_detection_debug.get("vetoes", {}), dict)

    counting_debug = report.debug_summary.get("counting_debug", {})
    assert counting_debug.get("valley_indices") == debug_info.valley_indices
    assert counting_debug.get("prominences") == debug_info.prominences

    counting_params = report.debug_summary.get("counting_params", {})
    assert isinstance(counting_params.get("multipliers", {}).get("pose_ok"), float)

    view_stats = report.debug_summary.get("view_stats", {})
    assert isinstance(view_stats.get("lateral_score"), float)
    assert isinstance(view_stats.get("reliable_frames"), int)

    quality = report.debug_summary.get("quality", {})
    assert quality.get("frames") == 4
    assert quality.get("primary_angle") == cfg.counting.primary_angle
    assert quality.get("pose_ok_fraction") == 0.5

