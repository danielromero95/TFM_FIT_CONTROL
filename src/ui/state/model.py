"""Modelos de datos inmutables que describen el estado de la aplicación."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from src.exercise_detection.types import DetectionResult

from src.config.settings import (
    DEFAULT_GENERATE_VIDEO,
    DEFAULT_PREVIEW_FPS,
    DEFAULT_USE_CROP,
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
CONFIG_DEFAULTS: Dict[str, float | str | bool] = {
    "low": float(SQUAT_LOW_THRESH),
    "high": float(SQUAT_HIGH_THRESH),
    "primary_angle": "auto",
    "debug_video": bool(DEFAULT_GENERATE_VIDEO),
    "use_crop": bool(DEFAULT_USE_CROP),
}


@dataclass
class AppState:
    """Estado centralizado que se guarda en ``st.session_state``."""

    step: Step = Step.UPLOAD
    upload_data: Optional[Dict[str, Any]] = None
    upload_token: Optional[Tuple[str, int, str]] = None
    active_upload_token: Optional[Tuple[str, int, str]] = None
    ui_rev: int = 0
    video_path: Optional[str] = None
    exercise: str = DEFAULT_EXERCISE_LABEL
    exercise_pending_update: Optional[str] = None
    view: str = ""  # "", "front", "side"
    view_pending_update: Optional[str] = None
    detect_result: Optional[DetectionResult] = None
    configure_values: Dict[str, float | str | bool] = field(
        default_factory=lambda: CONFIG_DEFAULTS.copy()
    )
    report: Optional[Any] = None
    pipeline_error: Optional[str] = None
    count_path: Optional[str] = None
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
