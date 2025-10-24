from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from concurrent.futures import Future

import streamlit as st


class Step(str, Enum):
    UPLOAD = "upload"
    DETECT = "detect"
    CONFIGURE = "configure"
    RUNNING = "running"
    RESULTS = "results"


DEFAULT_EXERCISE_LABEL = "Auto-Detect"
CONFIG_DEFAULTS: Dict[str, float | str | bool] = {
    "low": 80,
    "high": 150,
    "primary_angle": "left_knee",
    "min_prominence": 10.0,
    "min_distance_sec": 0.5,
    "debug_video": True,
    "use_crop": True,
}


@dataclass
class AppState:
    step: Step = Step.UPLOAD
    upload_data: Optional[Dict[str, Any]] = None
    upload_token: Optional[Tuple[str, int, str]] = None
    active_upload_token: Optional[Tuple[str, int, str]] = None
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
    state.exercise = DEFAULT_EXERCISE_LABEL
    state.exercise_pending_update = None
    state.view = ""
    state.view_pending_update = None
    state.detect_result = None
    state.configure_values = CONFIG_DEFAULTS.copy()
    state.step = Step.UPLOAD
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
