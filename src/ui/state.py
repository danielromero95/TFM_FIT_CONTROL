from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from concurrent.futures import Future

import streamlit as st

from src.config.settings import (
    DEFAULT_GENERATE_VIDEO,
    DEFAULT_PREVIEW_FPS,
    DEFAULT_USE_CROP,
    SQUAT_HIGH_THRESH,
    SQUAT_LOW_THRESH,
)


class Step(str, Enum):
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
    detect_result: Optional[Dict[str, Any]] = None
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
    overlay_video_path: Optional[str] = None


def get_state() -> AppState:
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    state: AppState = st.session_state.app_state
    if "video_uploader" in st.session_state:
        state.video_uploader = st.session_state.get("video_uploader")
    else:
        state.video_uploader = "__unset__"
    return state


def go_to(step: Step) -> None:
    st.session_state.app_state.step = step


def reset_state(*, preserve_upload: bool = False) -> None:
    state = get_state()
    video_path = state.video_path
    if video_path:
        try:
            Path(video_path).unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:
            try:
                Path(video_path).unlink()
            except FileNotFoundError:
                pass
        except OSError:
            pass

    state.video_path = None
    state.report = None
    state.pipeline_error = None
    state.count_path = None
    state.metrics_path = None
    state.cfg_fingerprint = None
    state.last_run_success = False
    state.analysis_future = None
    state.progress_value_from_cb = 0
    state.phase_text_from_cb = "Preparando..."
    state.run_id = None
    state.preview_enabled = False
    state.preview_fps = float(DEFAULT_PREVIEW_FPS)
    state.preview_frame_count = 0
    state.preview_last_ts_ms = 0.0
    state.overlay_video_path = None
    state.exercise = DEFAULT_EXERCISE_LABEL
    state.exercise_pending_update = None
    state.view = ""
    state.view_pending_update = None
    state.detect_result = None
    state.configure_values = CONFIG_DEFAULTS.copy()
    state.ui_rev = 0
    state.step = Step.UPLOAD

    for key in list(st.session_state.keys()):
        if key.startswith("metrics_multiselect_") or key.startswith("metrics_default_"):
            del st.session_state[key]

    if not preserve_upload:
        state.upload_data = None
        state.upload_token = None
    state.active_upload_token = None


def safe_rerun() -> None:
    """Attempt to rerun the Streamlit script using the stable API first."""

    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
