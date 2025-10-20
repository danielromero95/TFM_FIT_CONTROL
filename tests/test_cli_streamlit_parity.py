"""Integration-level parity test between CLI and Streamlit usage."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from src import config
from src.services.analysis_service import run_pipeline
from src.run_pipeline import main as cli_main


def _generate_smoke_video(path: Path, *, fps: int = 30, seconds: int = 3) -> None:
    width, height = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - defensive guard
        pytest.skip("OpenCV could not initialise the test video writer")

    total_frames = fps * seconds
    for index in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        center_x = int(width * (0.1 + 0.8 * index / max(total_frames - 1, 1)))
        center_y = height // 2
        cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"{index:03d}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()


def test_cli_and_streamlit_calls_are_in_sync(tmp_path: Path) -> None:
    video_path = tmp_path / "smoke_streamlit_cli.mp4"
    _generate_smoke_video(video_path)

    streamlit_cfg = config.load_default()
    streamlit_cfg.output.base_dir = tmp_path / "streamlit"
    streamlit_cfg.output.counts_dir = streamlit_cfg.output.base_dir / "counts"
    streamlit_cfg.output.poses_dir = streamlit_cfg.output.base_dir / "poses"

    report_streamlit = run_pipeline(str(video_path), streamlit_cfg)

    cli_output_dir = tmp_path / "cli"
    exit_code = cli_main(
        [
            "--video",
            str(video_path),
            "--output_dir",
            str(cli_output_dir),
        ]
    )
    assert exit_code == 0

    cli_cfg = config.load_default()
    cli_cfg.output.base_dir = cli_output_dir
    cli_cfg.output.counts_dir = cli_output_dir / "counts"
    cli_cfg.output.poses_dir = cli_output_dir / "poses"

    report_cli = run_pipeline(str(video_path), cli_cfg)

    assert report_cli.repetitions == report_streamlit.repetitions
    assert report_cli.stats.config_sha1 == report_streamlit.stats.config_sha1
    assert pytest.approx(report_cli.stats.fps_effective, abs=0.1) == report_streamlit.stats.fps_effective
