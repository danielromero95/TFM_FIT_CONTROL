"""Funciones auxiliares para preparar las entradas del pipeline de análisis."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from src import config
from src.exercise_detection.types import DetectionResult, make_detection_result
from src.ui.state import (
    AppState,
    CONFIG_DEFAULTS,
    DEFAULT_EXERCISE_LABEL,
    get_state,
)


def ensure_video_path() -> None:
    """Persiste el archivo subido en disco temporal y limpia restos previos."""

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
) -> tuple[str, config.Config, Optional[DetectionResult]]:
    """Prepara video, configuración y detecciones previas para ``run_pipeline``."""

    from src.ui.steps.detect import EXERCISE_TO_CONFIG

    video_path = state.video_path
    if not video_path:
        raise ValueError("The video to process was not found.")

    cfg = config.load_default()
    cfg_values = state.configure_values or CONFIG_DEFAULTS

    cfg.faults.low_thresh = float(cfg_values.get("low", CONFIG_DEFAULTS["low"]))
    cfg.faults.high_thresh = float(cfg_values.get("high", CONFIG_DEFAULTS["high"]))
    # Siempre ejecutamos en modo auto para que el pipeline elija el ángulo ideal.
    cfg.counting.primary_angle = "auto"
    cfg.debug.generate_debug_video = bool(cfg_values.get("debug_video", True))
    cfg.pose.use_crop = True

    exercise_label = state.exercise or DEFAULT_EXERCISE_LABEL
    cfg.counting.exercise = EXERCISE_TO_CONFIG.get(
        exercise_label,
        EXERCISE_TO_CONFIG.get(DEFAULT_EXERCISE_LABEL, "squat"),
    )

    det = state.detect_result
    prefetched_detection: Optional[DetectionResult] = None
    if isinstance(det, DetectionResult):
        prefetched_detection = det
    elif isinstance(det, dict):
        prefetched_detection = make_detection_result(
            det.get("label", "unknown"),
            det.get("view", "unknown"),
            float(det.get("confidence", 0.0)),
            side=det.get("side"),
            view_stats=det.get("view_stats"),
            debug=det.get("debug"),
        )

    return str(video_path), cfg, prefetched_detection
