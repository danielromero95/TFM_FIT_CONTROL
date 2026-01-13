"""Funciones auxiliares para preparar las entradas del pipeline de análisis."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

from src import config
from src.core.types import ExerciseType, ViewType, as_exercise, as_view
from src.exercise_detection.types import DetectionResult
from src.ui.state import (
    AppState,
    CONFIG_DEFAULTS,
    EXERCISE_THRESHOLDS,
    get_state,
    default_configure_values,
    migrate_thresholds_config,
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
    state.video_original_name = upload_data.get("name") or Path(tmp_file.name).name
    state.detect_result = None
    state.upload_data = None


def prepare_pipeline_inputs(
    state: AppState,
) -> Tuple[str, config.Config, Optional[Union[Tuple[str, str, float], DetectionResult]]]:
    """Prepara video, configuración y detecciones previas para ``run_pipeline``."""

    video_path = state.video_path
    if not video_path:
        raise ValueError("The video to process was not found.")

    cfg = config.load_default()
    cfg_values = default_configure_values()
    cfg_values.update(state.configure_values or {})

    selected_key = state.exercise_selected
    if selected_key and as_exercise(selected_key) is ExerciseType.UNKNOWN:
        selected_key = None

    detected_key = None
    if state.detect_result:
        detected_key = state.detect_result.get("label")
        if detected_key and as_exercise(detected_key) is ExerciseType.UNKNOWN:
            detected_key = None

    ex_key = selected_key or detected_key
    if not ex_key:
        raise ValueError("Please select an exercise before continuing.")
    cfg_values = migrate_thresholds_config(cfg_values, ex_key)

    defaults = EXERCISE_THRESHOLDS.get(ex_key, EXERCISE_THRESHOLDS["squat"])
    thresholds_by_exercise = cfg_values.get("thresholds_by_exercise") or {}
    exercise_thresholds = thresholds_by_exercise.get(ex_key) or {
        "low": defaults["low"],
        "high": defaults["high"],
        "custom": False,
    }

    cfg.faults.low_thresh = float(exercise_thresholds.get("low", defaults["low"]))
    cfg.faults.high_thresh = float(exercise_thresholds.get("high", defaults["high"]))
    thresholds_enable = bool(cfg_values.get("thresholds_enable", True))
    cfg.counting.enforce_low_thresh = thresholds_enable
    cfg.counting.enforce_high_thresh = thresholds_enable
    cfg.video.target_fps = float(cfg_values.get("target_fps", CONFIG_DEFAULTS["target_fps"]))
    cfg.pose.model_complexity = int(
        cfg_values.get("model_complexity", CONFIG_DEFAULTS["model_complexity"])
    )
    # Siempre ejecutamos en modo auto para que el pipeline elija el ángulo ideal.
    cfg.counting.primary_angle = "auto"
    cfg.debug.generate_debug_video = bool(cfg_values.get("debug_video", True))
    cfg.debug.debug_mode = bool(cfg_values.get("debug_mode", CONFIG_DEFAULTS.get("debug_mode", True)))
    cfg.pose.use_crop = True

    cfg.counting.exercise = ex_key

    det = state.detect_result
    prefetched_detection: Optional[Union[Tuple[str, str, float], DetectionResult]] = None
    if det:
        label = det.get("label", "unknown")
        view = det.get("view", "unknown")
        confidence = float(det.get("confidence", 0.0))
        diagnostics = det.get("diagnostics")
        if diagnostics:
            prefetched_detection = DetectionResult(
                as_exercise(label),
                as_view(view),
                confidence,
                diagnostics=diagnostics,
            )
        else:
            prefetched_detection = (label, view, confidence)

    effective_view = None
    if state.view_selected and as_view(state.view_selected) is not ViewType.UNKNOWN:
        effective_view = state.view_selected
    elif det:
        detected_view = det.get("view")
        if detected_view and as_view(detected_view) is not ViewType.UNKNOWN:
            effective_view = detected_view
    if not effective_view:
        raise ValueError("Please select a view before continuing.")

    return str(video_path), cfg, prefetched_detection
