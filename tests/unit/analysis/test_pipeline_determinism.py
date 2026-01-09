from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import config
from src.A_preprocessing.video_metadata import VideoInfo
from src.B_pose_estimation.estimators.mediapipe_pool import PoseGraphPool
from src.C_analysis import run_pipeline
from src.C_analysis.repetition_counter import CountingDebugInfo


def test_pose_graph_pool_isolated_per_run(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video content")

    cfg = config.load_default()
    cfg.output.base_dir = tmp_path / "outputs"
    cfg.debug.generate_debug_video = False
    cfg.debug.debug_mode = False
    cfg.pose.use_crop = False
    cfg.pose.use_roi_tracking = False

    class _FakePose:
        instances: list["_FakePose"] = []

        def __init__(self, **_kwargs) -> None:
            self.counter = 0
            self.__class__.instances.append(self)

        def process(self, _rgb_image):
            self.counter += 1
            value = 0.1 + self.counter * 0.01
            landmarks = [
                SimpleNamespace(x=value, y=0.2, z=0.0, visibility=1.0) for _ in range(33)
            ]
            return SimpleNamespace(
                pose_landmarks=SimpleNamespace(landmark=landmarks),
                pose_world_landmarks=None,
            )

        def close(self) -> None:
            return None

    def _fake_draw_landmarks(*_args, **_kwargs) -> None:
        return None

    def _fake_ensure_imports(cls) -> None:
        cls.mp_pose = SimpleNamespace(Pose=_FakePose)
        cls.mp_drawing = SimpleNamespace(draw_landmarks=_fake_draw_landmarks)
        cls._imported = True

    monkeypatch.setattr(PoseGraphPool, "_ensure_imports", classmethod(_fake_ensure_imports))

    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]

    class _DummyCap:
        def release(self) -> None:  # pragma: no cover - cleanup best effort
            return None

        def get(self, _prop) -> float:  # pragma: no cover - stub
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
                frame_count=2,
                duration_sec=0.1,
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

    monkeypatch.setattr(
        "src.C_analysis.pipeline.filter_landmarks",
        lambda df: (df.copy(), [], pd.Series([True] * len(df)), pd.Series(range(len(df)))),
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.infer_upright_quadrant_from_sequence",
        lambda _sequence: 0,
    )

    def _compute_metrics(filtered_sequence, *_args, **_kwargs):
        return (
            pd.DataFrame({cfg.counting.primary_angle: filtered_sequence["x0"].to_numpy()}),
            25.0,
            [],
            None,
            cfg.counting.primary_angle,
            pd.Series(range(len(filtered_sequence))),
        )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.compute_metrics_and_angle",
        _compute_metrics,
    )

    def _auto_params(_exercise, _df_metrics, _primary, _fps, counting_cfg, **_kwargs):
        return SimpleNamespace(
            min_prominence=counting_cfg.min_prominence + 1.0,
            min_distance_sec=counting_cfg.min_distance_sec + 0.1,
            refractory_sec=counting_cfg.refractory_sec + 0.05,
            cadence_period_sec=None,
            iqr_deg=None,
            multipliers={},
        )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.auto_counting_params",
        _auto_params,
    )

    monkeypatch.setattr(
        "src.C_analysis.pipeline.maybe_count_reps",
        lambda *_args, **_kwargs: (1, [], CountingDebugInfo([], [])),
    )

    report_one = run_pipeline(
        str(video_path),
        cfg,
        prefetched_detection=("squat", "front", 0.9),
    )

    report_two = run_pipeline(
        str(video_path),
        cfg,
        prefetched_detection=("squat", "front", 0.9),
    )

    assert report_one.metrics.equals(report_two.metrics)
    assert len(_FakePose.instances) == 2
