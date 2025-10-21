from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

from src import config
from src.ui.state import AppState, CONFIG_DEFAULTS, DEFAULT_EXERCISE_LABEL, get_state
from src.ui.steps.detect import EXERCISE_TO_CONFIG


def ensure_video_path() -> None:
    state = get_state()
    upload_data = state.upload_data
    if not upload_data:
        return

    old_path = state.video_path
    if old_path:
        try:
            Path(old_path).unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:
            try:
                Path(old_path).unlink()
            except FileNotFoundError:
                pass
        except OSError:
            pass

    suffix = Path(upload_data["name"]).suffix or ".mp4"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(upload_data["bytes"])
    tmp_file.flush()
    tmp_file.close()

    state.video_path = tmp_file.name
    state.detect_result = None
    state.upload_data = None


def prepare_pipeline_inputs(
    state: AppState,
) -> Tuple[str, config.Config, Optional[Tuple[str, str, float]]]:
    video_path = state.video_path
    if not video_path:
        raise ValueError("The video to process was not found.")

    cfg = config.load_default()
    cfg_values = state.configure_values or CONFIG_DEFAULTS

    cfg.faults.low_thresh = float(cfg_values.get("low", CONFIG_DEFAULTS["low"]))
    cfg.faults.high_thresh = float(cfg_values.get("high", CONFIG_DEFAULTS["high"]))
    cfg.counting.primary_angle = str(
        cfg_values.get("primary_angle", CONFIG_DEFAULTS["primary_angle"])
    )
    cfg.counting.min_prominence = float(
        cfg_values.get("min_prominence", CONFIG_DEFAULTS["min_prominence"])
    )
    cfg.counting.min_distance_sec = float(
        cfg_values.get("min_distance_sec", CONFIG_DEFAULTS["min_distance_sec"])
    )
    cfg.debug.generate_debug_video = bool(cfg_values.get("debug_video", True))
    cfg.pose.use_crop = True

    exercise_label = state.exercise or DEFAULT_EXERCISE_LABEL
    cfg.counting.exercise = EXERCISE_TO_CONFIG.get(
        exercise_label,
        EXERCISE_TO_CONFIG.get(DEFAULT_EXERCISE_LABEL, "squat"),
    )

    det = state.detect_result
    prefetched_detection: Optional[Tuple[str, str, float]] = None
    if det:
        prefetched_detection = (
            det.get("label", "unknown"),
            det.get("view", "unknown"),
            float(det.get("confidence", 0.0)),
        )

    return str(video_path), cfg, prefetched_detection
