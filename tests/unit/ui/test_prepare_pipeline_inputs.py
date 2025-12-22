from __future__ import annotations

from pathlib import Path

from src.ui.steps.utils.pipeline import prepare_pipeline_inputs
from src.ui.state import CONFIG_DEFAULTS
from src.ui.state.model import AppState


def test_prepare_pipeline_inputs_propagates_strict_high(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")

    cfg_values = {**CONFIG_DEFAULTS, "strict_high": False, "low": 30.0, "high": 120.0}
    state = AppState(video_path=str(video_path), configure_values=cfg_values)

    _, cfg, _ = prepare_pipeline_inputs(state)

    assert cfg.faults.low_thresh == 30.0
    assert cfg.faults.high_thresh == 120.0
    assert cfg.counting.enforce_high_thresh is False


def test_prepare_pipeline_inputs_defaults_strict_high_true(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")

    state = AppState(video_path=str(video_path))

    _, cfg, _ = prepare_pipeline_inputs(state)

    assert cfg.counting.enforce_high_thresh is True
    assert Path(state.video_path).exists()
