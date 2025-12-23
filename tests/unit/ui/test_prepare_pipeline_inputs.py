from __future__ import annotations

from pathlib import Path

from src.ui.state import EXERCISE_THRESHOLDS, default_configure_values
from src.ui.steps.utils.pipeline import prepare_pipeline_inputs
from src.ui.state.model import AppState


def test_prepare_pipeline_inputs_propagates_thresholds_enable(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")

    cfg_values = {
        **default_configure_values(),
        "thresholds_enable": False,
        "thresholds_by_exercise": {
            "squat": {"low": 30.0, "high": 120.0, "custom": True},
        },
    }
    state = AppState(video_path=str(video_path), configure_values=cfg_values)

    _, cfg, _ = prepare_pipeline_inputs(state)

    assert cfg.faults.low_thresh == 30.0
    assert cfg.faults.high_thresh == 120.0
    assert cfg.counting.enforce_low_thresh is False
    assert cfg.counting.enforce_high_thresh is False


def test_prepare_pipeline_inputs_defaults_thresholds_enable_true(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")

    state = AppState(video_path=str(video_path))

    _, cfg, _ = prepare_pipeline_inputs(state)

    assert cfg.counting.enforce_low_thresh is True
    assert cfg.counting.enforce_high_thresh is True
    assert Path(state.video_path).exists()


def test_prepare_pipeline_inputs_apply_deadlift_defaults_when_not_custom(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")

    state = AppState(
        video_path=str(video_path),
        exercise="Deadlift",
        configure_values=default_configure_values(),
    )

    _, cfg, _ = prepare_pipeline_inputs(state)

    assert cfg.faults.low_thresh == EXERCISE_THRESHOLDS["deadlift"]["low"]
    assert cfg.faults.high_thresh == EXERCISE_THRESHOLDS["deadlift"]["high"]


def test_prepare_pipeline_inputs_preserves_custom_thresholds(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")

    state = AppState(
        video_path=str(video_path),
        exercise="Deadlift",
        configure_values={
            **default_configure_values(),
            "thresholds_by_exercise": {
                "deadlift": {"low": 75.0, "high": 155.0, "custom": True},
            },
        },
    )

    _, cfg, _ = prepare_pipeline_inputs(state)

    assert cfg.faults.low_thresh == 75.0
    assert cfg.faults.high_thresh == 155.0


def test_switching_exercises_keeps_custom_per_exercise(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")

    custom_squat = {"low": 110.0, "high": 150.0, "custom": True}
    state = AppState(
        video_path=str(video_path),
        exercise="Deadlift",
        configure_values={
            **default_configure_values(),
            "thresholds_by_exercise": {"squat": custom_squat},
        },
    )

    _, cfg, _ = prepare_pipeline_inputs(state)

    assert cfg.faults.low_thresh == EXERCISE_THRESHOLDS["deadlift"]["low"]
    assert cfg.faults.high_thresh == EXERCISE_THRESHOLDS["deadlift"]["high"]


def test_custom_thresholds_persist_per_exercise(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")

    custom_squat = {"low": 95.0, "high": 165.0, "custom": True}
    state = AppState(
        video_path=str(video_path),
        exercise="Squat",
        configure_values={
            **default_configure_values(),
            "thresholds_by_exercise": {"squat": custom_squat},
        },
    )

    _, cfg, _ = prepare_pipeline_inputs(state)

    assert cfg.faults.low_thresh == 95.0
    assert cfg.faults.high_thresh == 165.0
