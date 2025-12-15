from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

from src import config
from src.debug_report import build_debug_report_bundle
from src.pipeline_data import Report, RunStats


def test_build_debug_report_bundle_serializes_strict_json(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"binary video content")

    cfg = config.load_default()

    config_used_path = tmp_path / "config_used.json"
    config_used_path.write_text("{}", encoding="utf-8")
    effective_config_path = tmp_path / "config_effective.json"
    effective_config_path.write_text("{}", encoding="utf-8")

    stats = RunStats(
        config_sha1="deadbeef",
        fps_original=30.0,
        fps_effective=30.0,
        frames=3,
        exercise_selected="auto",
        exercise_detected="squat",
        view_detected="front",
        detection_confidence=0.9,
        primary_angle="left_knee",
        angle_range_deg=20.0,
        min_prominence=1.0,
        min_distance_sec=1.0,
        refractory_sec=0.5,
        warnings=["sample warning"],
        config_path=config_used_path,
    )

    df_metrics = pd.DataFrame(
        {
            "left_knee": [1.0, np.nan, np.inf],
            "pose_ok": [0.6, 0.4, 0.8],
        }
    )

    debug_summary = {
        "quality": {"pose_ok_fraction": np.nan},
        "counting_params": {"multipliers": {"pose_ok": np.float32(1.5)}},
        "detection": {
            "source": "incremental",
            "exercise_detection_debug": {
                "classification_scores": {"deadlift_veto": np.bool_(False)},
                "features_summary": {
                    "torso_tilt_deg": {"min": np.float64(1.0), "valid_fraction": np.float32(0.5)}
                },
            },
        },
    }

    report = Report(
        repetitions=1,
        metrics=df_metrics,
        stats=stats,
        config_used=cfg,
        effective_config_path=effective_config_path,
        debug_summary=debug_summary,
    )

    bundle_bytes, bundle_name = build_debug_report_bundle(
        report,
        video_path=str(video_path),
        cfg=cfg,
        df_metrics=df_metrics,
    )

    assert bundle_name.startswith("debug_report_video_")

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
        names = set(archive.namelist())
        assert "debug_report.json" in names
        assert config_used_path.name in names
        assert effective_config_path.name in names

        metrics_files = [n for n in names if n.endswith("_metrics.csv")]
        assert metrics_files, "metrics.csv should be included when under the limit"
        metrics_content = archive.read(metrics_files[0]).decode()
        assert "left_knee" in metrics_content

        payload = json.loads(archive.read("debug_report.json"))

    assert payload["metrics_summary"]["row_count"] == len(df_metrics)
    assert payload["metrics_summary"]["nan_ratio"]["left_knee"] > 0
    assert payload["counting_params"]["multipliers"]["pose_ok"] == 1.5
    assert payload["quality"]["pose_ok_fraction"] is None
    assert payload["warnings"] == ["sample warning"]
    detection_block = payload["detection"]
    assert detection_block["source"] == "incremental"
    assert detection_block["exercise_detection_debug"]["classification_scores"]["deadlift_veto"] is False
    assert detection_block["exercise_detection_debug"]["features_summary"]["torso_tilt_deg"]["min"] == 1.0

