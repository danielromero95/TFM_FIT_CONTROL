"""Modelos de datos inmutables que describen el estado de la aplicación."""

from __future__ import annotations

from concurrent.futures import Future
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from src.config.settings import (
    DEFAULT_DEBUG_MODE,
    DEFAULT_GENERATE_VIDEO,
    DEFAULT_PREVIEW_FPS,
    DEFAULT_USE_CROP,
    MODEL_COMPLEXITY,
    SQUAT_HIGH_THRESH,
    SQUAT_LOW_THRESH,
)


class Step(str, Enum):
    """Enumera los pasos visibles del asistente en Streamlit."""

    UPLOAD = "upload"
    DETECT = "detect"
    CONFIGURE = "configure"
    RUNNING = "running"
    RESULTS = "results"


DEFAULT_EXERCISE_LABEL = "Auto-Detect"
EXERCISE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "squat": {"low": float(SQUAT_LOW_THRESH), "high": float(SQUAT_HIGH_THRESH)},
    "deadlift": {"low": 150.0, "high": 170.0},
    "bench_press": {"low": 90.0, "high": 160.0},
}

CONFIG_DEFAULTS: Dict[str, Any] = {
    "low": EXERCISE_THRESHOLDS["squat"]["low"],
    "high": EXERCISE_THRESHOLDS["squat"]["high"],
    "primary_angle": "auto",
    "debug_video": bool(DEFAULT_GENERATE_VIDEO),
    "debug_mode": bool(DEFAULT_DEBUG_MODE),
    "use_crop": bool(DEFAULT_USE_CROP),
    "target_fps": 10.0,
    "model_complexity": int(MODEL_COMPLEXITY),
    "thresholds_enable": True,
    "thresholds_by_exercise": {
        exercise: {"low": values["low"], "high": values["high"], "custom": False}
        for exercise, values in EXERCISE_THRESHOLDS.items()
    },
}


def default_configure_values() -> Dict[str, Any]:
    """Return a deep copy of the default configure values."""

    return copy.deepcopy(CONFIG_DEFAULTS)


def migrate_thresholds_config(
    cfg_values: Dict[str, Any], ex_key: str
) -> Dict[str, Any]:
    """Migrate legacy threshold keys into the per-exercise store."""

    thresholds_by_exercise = cfg_values.get("thresholds_by_exercise") or {}
    if not isinstance(thresholds_by_exercise, dict):
        thresholds_by_exercise = {}

    legacy_keys_present = any(
        key in cfg_values for key in ("thresholds_custom", "thresholds_exercise", "strict_high")
    )
    if not legacy_keys_present:
        cfg_values["thresholds_by_exercise"] = thresholds_by_exercise
        return cfg_values

    legacy_low = cfg_values.get("low")
    legacy_high = cfg_values.get("high")
    legacy_exercise = cfg_values.get("thresholds_exercise") or ex_key
    flagged_custom = cfg_values.get("thresholds_custom")

    # Consider older strict_high flag as an indicator that the user configured thresholds.
    strict_high = cfg_values.get("strict_high")

    if legacy_low is not None or legacy_high is not None:
        current_entry = thresholds_by_exercise.get(legacy_exercise)
        defaults = EXERCISE_THRESHOLDS.get(legacy_exercise, EXERCISE_THRESHOLDS["squat"])
        low = float(legacy_low) if legacy_low is not None else defaults["low"]
        high = float(legacy_high) if legacy_high is not None else defaults["high"]
        custom = bool(flagged_custom) or low != defaults["low"] or high != defaults["high"] or bool(strict_high)

        if current_entry is None or not current_entry.get("custom"):
            thresholds_by_exercise = {
                **thresholds_by_exercise,
                legacy_exercise: {"low": low, "high": high, "custom": custom},
            }

    cfg_values["thresholds_by_exercise"] = thresholds_by_exercise
    return cfg_values


@dataclass
class AppState:
    """Estado centralizado que se guarda en ``st.session_state``."""

    step: Step = Step.UPLOAD
    upload_data: Optional[Dict[str, Any]] = None
    upload_token: Optional[Tuple[str, int, str]] = None
    active_upload_token: Optional[Tuple[str, int, str]] = None
    ui_rev: int = 0
    video_path: Optional[str] = None
    video_original_name: Optional[str] = None
    exercise: str = DEFAULT_EXERCISE_LABEL
    exercise_pending_update: Optional[str] = None
    view: str = ""  # "", "front", "side"
    view_pending_update: Optional[str] = None
    detect_result: Optional[Dict[str, Any]] = None
    configure_values: Dict[str, Any] = field(default_factory=default_configure_values)
    report: Optional[Any] = None
    pipeline_error: Optional[str] = None
    metrics_path: Optional[str] = None
    cfg_fingerprint: Optional[str] = None
    last_run_success: bool = False
    video_uploader: Any = None
    analysis_future: Future | None = None
    progress_value_from_cb: int = 0
    phase_text_from_cb: str = "Preparando..."
    run_id: str | None = None
    preview_enabled: bool = False
    preview_fps: float = float(DEFAULT_PREVIEW_FPS)
    preview_frame_count: int = 0
    preview_last_ts_ms: float = 0.0
    overlay_video_stream_path: Optional[str] = None
    overlay_video_download_path: Optional[str] = None
