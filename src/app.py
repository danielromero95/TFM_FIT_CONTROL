# src/app.py
"""Streamlit front-end for the analysis pipeline."""

from __future__ import annotations

import atexit
import hashlib
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError
from queue import SimpleQueue, Empty
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Gym Performance Analysis")

# --- Ensure Windows loads codec DLLs from the active conda env first -----------
if sys.platform.startswith("win"):
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        dll_dir = Path(conda_prefix) / "Library" / "bin"
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

# --- Tame noisy logs from TF/MediaPipe -----------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# Ensure the project root is available on the import path when Streamlit executes the app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.pipeline import Report

from src.services.analysis_service import run_pipeline
from src.detect.exercise_detector import detect_exercise

EXERCISE_CHOICES = [
    ("Auto-Detect", "auto"),
    ("Squat", "squat"),
]

DEFAULT_EXERCISE_LABEL = "Auto-Detect"
EXERCISE_LABELS = [lbl for (lbl, _) in EXERCISE_CHOICES]
VALID_EXERCISE_LABELS = set(EXERCISE_LABELS)
EXERCISE_TO_CONFIG = {lbl: key for (lbl, key) in EXERCISE_CHOICES}
CONFIG_TO_LABEL = {key: lbl for (lbl, key) in EXERCISE_CHOICES}
EXERCISE_WIDGET_KEY = "exercise_select_value"
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
    step: str = "upload"
    upload_data: Optional[Dict[str, Any]] = None
    upload_token: Optional[Tuple[str, int, str]] = None
    active_upload_token: Optional[Tuple[str, int, str]] = None
    video_path: Optional[str] = None
    exercise: str = field(default_factory=lambda: DEFAULT_EXERCISE_LABEL)
    exercise_pending_update: Optional[str] = None
    detect_result: Optional[Dict[str, Any]] = None
    configure_values: Dict[str, float | str | bool] = field(
        default_factory=lambda: CONFIG_DEFAULTS.copy()
    )
    report: Optional[Report] = None
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


@st.cache_resource
def get_thread_pool_executor() -> ThreadPoolExecutor:
    executor = ThreadPoolExecutor(max_workers=1)
    atexit.register(executor.shutdown, wait=False, cancel_futures=True)
    return executor


@st.cache_resource
def get_progress_queue() -> SimpleQueue:
    return SimpleQueue()


def _drain_queue(queue: SimpleQueue) -> None:
    while True:
        try:
            queue.get_nowait()
        except Empty:
            break


def _get_state(*, inject_css: bool = True) -> AppState:
    if inject_css and threading.current_thread() is threading.main_thread():
        _inject_css()
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    state: AppState = st.session_state.app_state
    if "video_uploader" in st.session_state:
        state.video_uploader = st.session_state.get("video_uploader")
    else:
        state.video_uploader = "__unset__"
    return state


def _inject_css() -> None:
    st.markdown(
        """
    <style>
      /* --- Top header with inline title --- */
      header[data-testid="stHeader"] {
        background: #0f172a;
        border-bottom: 1px solid #1f2937;
        height: 48px;
        padding-right: 140px;
        position: relative;
      }
      header[data-testid="stHeader"]::before {
        content: "Gym Performance Analysis";
        position: absolute;
        left: 16px;
        top: 50%;
        transform: translateY(-50%);
        color: #e5e7eb;
        font-weight: 700;
        font-size: 18px;
        letter-spacing: .2px;
        pointer-events: none;
        max-width: calc(100% - 180px);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .step-detect .form-label {
        color: #e5e7eb;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: .25rem;
      }

      /* Target the Streamlit button that follows our marker */
      .step-detect .btn-danger + div .stButton > button,
      .step-detect .btn-danger + div button {
        border-radius: 12px !important;
        min-height: 40px;
        min-width: 140px;
        background: transparent !important;
        color: #ef4444 !important;
        border: 1px solid rgba(239,68,68,.6) !important;
        transition: background .15s ease, border-color .15s ease, transform .15s ease, box-shadow .15s ease;
      }
      .step-detect .btn-danger + div .stButton > button:hover,
      .step-detect .btn-danger + div button:hover {
        background: rgba(239,68,68,.10) !important;
        border-color: rgba(239,68,68,.9) !important;
        transform: translateY(-1px);
      }

      .step-detect .btn-success + div .stButton > button,
      .step-detect .btn-success + div button {
        border-radius: 12px !important;
        min-height: 40px;
        min-width: 140px;
        background: linear-gradient(135deg, rgba(34,197,94,.95), rgba(16,185,129,.95)) !important;
        color: #ecfdf5 !important;
        border: 1px solid rgba(34,197,94,.8) !important;
        box-shadow: 0 12px 24px rgba(16,185,129,.35);
        transition: transform .15s ease, box-shadow .15s ease;
        margin-left: auto;
        display: block;
      }
      .step-detect .btn-success + div .stButton > button:hover,
      .step-detect .btn-success + div button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 28px rgba(16,185,129,.45);
      }

      /* Disabled state */
      .step-detect .btn-danger + div .stButton > button[disabled],
      .step-detect .btn-success + div .stButton > button[disabled] {
        opacity: .55 !important;
        transform: none !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
      }

      .chip {display:inline-block;padding:.2rem .5rem;border-radius:9999px;font-size:.8rem;
             margin-right:.25rem}
      .chip.ok {background:#065f46;color:#ecfdf5;}      /* green */
      .chip.warn {background:#7c2d12;color:#ffedd5;}    /* amber/brown */
      .chip.info {background:#1e293b;color:#e2e8f0;}    /* slate */

      .spacer-sm { height: .5rem; }

      /* Step 2: keep a consistent preview area regardless of the source video */
      .step-detect [data-testid="stVideo"] {
        width: 100%;
        max-width: 100%;
        margin: 0 auto .75rem auto;
      }
      .step-detect [data-testid="stVideo"] video {
        width: 100% !important;
        max-width: 100% !important;
        height: 360px !important;
        background: #000;
        object-fit: contain;
      }

      /* Keep title above content and safe on very narrow viewports */
      header[data-testid="stHeader"] { z-index: 1000; }
      @media (max-width: 520px) {
        header[data-testid="stHeader"]::before { font-size: 16px; }
      }
      @media (max-width: 420px) {
        header[data-testid="stHeader"]::before { display: none; }
      }

      /* Slightly larger click targets for nav buttons */
      .btn-danger > button, .btn-success > button { min-height: 40px; min-width: 140px; }

      /* Pull content closer to the toolbar */
      section[data-testid="stMain"],
      main {
        padding-top: 0 !important;
      }
      section[data-testid="stMain"] > div:first-child,
      main > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
      }
      main .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
      }
      main .block-container > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
      }
      main [data-testid="stToolbar"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
      }
      main [data-testid="stToolbar"] + div {
        margin-top: 0 !important;
        padding-top: 0 !important;
      }
      main [data-testid="stVerticalBlock"] {
        padding-top: 0 !important;
      }
      main [data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0 !important;
      }
      main [data-testid="stHorizontalBlock"] {
        margin-top: 0 !important;
        align-items: flex-start !important;
      }
      main [data-testid="column"] {
        padding-top: 0 !important;
      }
      main [data-testid="column"] > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
      }

      /* Ensure the results column aligns with step 4 content */
      .results-panel {
        margin-top: 0 !important;
        display: flex;
        flex-direction: column;
        gap: .75rem;
      }
      .results-panel h3:first-child {
        margin-top: 0 !important;
      }
      .results-panel .stDataFrame {
        margin-bottom: .25rem;
      }
      .results-panel video {
        border-radius: 12px;
      }
      .results-panel [data-testid="stHorizontalBlock"] {
        gap: .5rem !important;
      }
      .results-panel [data-testid="column"] .stButton > button {
        width: 100%;
      }
    </style>
    """,
        unsafe_allow_html=True,
    )
def _reset_state(*, preserve_upload: bool = False) -> None:
    state = _get_state()
    video_path = state.video_path
    if video_path:
        try:
            Path(video_path).unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:
            # Python < 3.8 compatibility – ignore missing files silently
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
    state.detect_result = None
    state.configure_values = CONFIG_DEFAULTS.copy()
    state.step = "upload"
    if not preserve_upload:
        state.upload_data = None
        state.upload_token = None
    state.active_upload_token = None


def _reset_app() -> None:
    _reset_state()
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def _ensure_video_path() -> None:
    state = _get_state()
    upload_data = state.upload_data
    if not upload_data:
        return
    # If we're replacing an existing temp file, delete it first
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
    # Free large byte payload from session to reduce memory usage once persisted
    state.upload_data = None


def _prepare_pipeline_inputs(state: AppState) -> Tuple[str, config.Config, Optional[Tuple[str, str, float]]]:
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


def _run_pipeline(
    *,
    video_path: str,
    cfg: config.Config,
    prefetched_detection: Optional[Tuple[str, str, float]] = None,
    progress_cb=None,
) -> Report:
    return run_pipeline(
        str(video_path),
        cfg,
        progress_callback=progress_cb,
        prefetched_detection=prefetched_detection,
    )


def _upload_step() -> None:
    st.markdown("### 1. Upload your video")
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv", "mpg", "mpeg", "wmv"],
        key="video_uploader",
        label_visibility="collapsed",
    )
    state = _get_state()
    previous_token = state.active_upload_token
    if uploaded is not None:
        data_bytes = uploaded.getvalue()
        new_token = (
            uploaded.name,
            len(data_bytes),
            hashlib.md5(data_bytes).hexdigest(),
        )
        if previous_token != new_token:
            _reset_state(preserve_upload=True)
            state = _get_state()
            state.upload_data = {
                "name": uploaded.name,
                "bytes": data_bytes,
            }
            state.upload_token = new_token
            state.active_upload_token = new_token
            _ensure_video_path()
            state = _get_state()
            if state.video_path:
                state.step = "detect"
        else:
            state.active_upload_token = new_token
    else:
        uploader_state = state.video_uploader
        if (
            previous_token is not None
            and state.video_path
            and uploader_state in (None, "")
        ):
            _reset_state()
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()


def _detect_step() -> None:
    st.markdown("### 2. Detect the exercise")
    st.markdown('<div class="step step-detect">', unsafe_allow_html=True)
    state = _get_state()
    video_path = state.video_path
    if video_path:
        st.video(str(video_path))

    step = state.step or "upload"
    is_active = step == "detect" and video_path is not None

    token = state.upload_token
    detect_result = state.detect_result
    if detect_result is not None and detect_result.get("token") != token:
        state.detect_result = None
        detect_result = None

    pending_exercise = state.exercise_pending_update
    state.exercise_pending_update = None
    if pending_exercise in VALID_EXERCISE_LABELS:
        state.exercise = pending_exercise

    current_exercise = state.exercise or DEFAULT_EXERCISE_LABEL
    if current_exercise not in VALID_EXERCISE_LABELS:
        current_exercise = DEFAULT_EXERCISE_LABEL
        state.exercise = current_exercise

    widget_value = st.session_state.get(EXERCISE_WIDGET_KEY)
    if widget_value not in VALID_EXERCISE_LABELS:
        widget_value = current_exercise
    if widget_value != current_exercise:
        current_exercise = widget_value
        state.exercise = current_exercise
    st.session_state[EXERCISE_WIDGET_KEY] = current_exercise

    if current_exercise != DEFAULT_EXERCISE_LABEL and detect_result is not None:
        state.detect_result = None
        detect_result = None

    select_col_label, select_col_control = st.columns([1, 2])
    with select_col_label:
        st.markdown(
            '<div class="form-label">Select the exercise</div>',
            unsafe_allow_html=True,
        )
    with select_col_control:
        select_index = EXERCISE_LABELS.index(current_exercise)
        selected_exercise = st.selectbox(
            "Select the exercise",
            options=EXERCISE_LABELS,
            index=select_index,
            key=EXERCISE_WIDGET_KEY,
            label_visibility="collapsed",
            disabled=not is_active,
        )
        if selected_exercise != current_exercise:
            state.exercise = selected_exercise
            current_exercise = selected_exercise

    detect_result = state.detect_result
    if (
        detect_result is not None
        and detect_result.get("token") != token
    ):
        detect_result = None
        state.detect_result = None

    info_container = st.container()
    if detect_result:
        if detect_result.get("error"):
            info_container.error(
                "Automatic exercise detection failed: "
                f"{detect_result.get('error')}"
            )
            info_container.info(
                "You can adjust the selection manually or try detecting again."
            )
        else:
            label_key = detect_result.get("label", "unknown")
            label_display = CONFIG_TO_LABEL.get(
                label_key,
                label_key.replace("_", " ").title(),
            )
            view_display = detect_result.get("view", "unknown").replace("_", " ").title()
            confidence = float(detect_result.get("confidence", 0.0))
            info_container.success(
                f"Detected exercise: {label_display} – {view_display} view"
                f" ({confidence:.0%} confidence)."
            )
            if not detect_result.get("accepted"):
                info_container.info(
                    "Click Continue to accept this detection or choose an exercise manually."
                )

    actions_placeholder = st.empty()
    if is_active:
        with actions_placeholder.container():
            back_col, continue_col = st.columns([1, 2])
            with back_col:
                st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
                if st.button("Back", key="detect_back"):
                    state.detect_result = None
                    state.step = "upload"
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with continue_col:
                st.markdown('<div class="btn-success">', unsafe_allow_html=True)
                if st.button("Continue", key="detect_continue"):
                    if current_exercise == DEFAULT_EXERCISE_LABEL:
                        detect_result = state.detect_result
                        if (
                            detect_result
                            and not detect_result.get("error")
                            and detect_result.get("label")
                            and detect_result.get("token") == token
                            and detect_result.get("accepted")
                        ):
                            state.step = "configure"
                        elif (
                            detect_result
                            and not detect_result.get("error")
                            and detect_result.get("label")
                            and detect_result.get("token") == token
                        ):
                            mapped_label = CONFIG_TO_LABEL.get(
                                detect_result.get("label", ""),
                                current_exercise,
                            )
                            state.exercise_pending_update = mapped_label
                            detect_result["accepted"] = True
                            state.step = "configure"
                        else:
                            if not video_path:
                                st.warning("Please upload a video before continuing.")
                            else:
                                with st.spinner("Detecting exercise…"):
                                    try:
                                        label_key, detected_view, confidence = detect_exercise(
                                            str(video_path)
                                        )
                                    except Exception as exc:  # pragma: no cover - UI feedback
                                        state.detect_result = {
                                            "error": str(exc),
                                            "accepted": False,
                                            "token": token,
                                        }
                                    else:
                                        state.detect_result = {
                                            "label": label_key,
                                            "view": detected_view,
                                            "confidence": float(confidence),
                                            "accepted": False,
                                            "token": token,
                                        }
                                try:
                                    st.rerun()
                                except Exception:
                                    st.experimental_rerun()
                    else:
                        state.detect_result = None
                        state.step = "configure"
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        actions_placeholder.empty()
    st.markdown("</div>", unsafe_allow_html=True)


def _configure_step(*, disabled: bool = False, show_actions: bool = True) -> None:
    st.markdown("### 3. Configure the analysis")
    state = _get_state()
    stored_cfg = state.configure_values
    if stored_cfg is None:
        cfg_values = CONFIG_DEFAULTS.copy()
    else:
        cfg_values = {**CONFIG_DEFAULTS, **dict(stored_cfg)}
    cfg_values.pop("target_fps", None)
    cfg_values["use_crop"] = True

    if disabled:
        current_step = state.step
        if current_step == "running":
            st.info("The configuration is displayed for reference while the analysis runs.")
        elif current_step == "results":
            st.info("Configuration values used for the analysis are shown below.")
        else:
            st.info("Configuration is read-only at this stage.")

    col1, col2 = st.columns(2)
    with col1:
        low = st.number_input(
            "Lower threshold (°)",
            min_value=0,
            max_value=180,
            value=int(cfg_values.get("low", CONFIG_DEFAULTS["low"])),
            disabled=disabled,
            key="cfg_low",
        )
    with col2:
        high = st.number_input(
            "Upper threshold (°)",
            min_value=0,
            max_value=180,
            value=int(cfg_values.get("high", CONFIG_DEFAULTS["high"])),
            disabled=disabled,
            key="cfg_high",
        )

    primary_angle = st.text_input(
        "Primary angle",
        value=str(cfg_values.get("primary_angle", CONFIG_DEFAULTS["primary_angle"])),
        disabled=disabled,
        key="cfg_primary_angle",
    )

    col3, col4 = st.columns(2)
    with col3:
        min_prominence = st.number_input(
            "Minimum prominence",
            min_value=0.0,
            value=float(cfg_values.get("min_prominence", CONFIG_DEFAULTS["min_prominence"])),
            step=0.5,
            disabled=disabled,
            key="cfg_min_prominence",
        )
    with col4:
        min_distance_sec = st.number_input(
            "Minimum distance (s)",
            min_value=0.0,
            value=float(cfg_values.get("min_distance_sec", CONFIG_DEFAULTS["min_distance_sec"])),
            step=0.1,
            disabled=disabled,
            key="cfg_min_distance",
        )

    debug_video = st.checkbox(
        "Generate debug video",
        value=bool(cfg_values.get("debug_video", CONFIG_DEFAULTS["debug_video"])),
        disabled=disabled,
        key="cfg_debug_video",
    )

    current_values = {
        "low": float(low),
        "high": float(high),
        "primary_angle": primary_angle,
        "min_prominence": float(min_prominence),
        "min_distance_sec": float(min_distance_sec),
        "debug_video": bool(debug_video),
        "use_crop": True,
    }
    if not disabled:
        state.configure_values = current_values

    if show_actions and not disabled:
        run_active = bool(state.analysis_future and not state.analysis_future.done())
        col_back, col_forward = st.columns(2)
        with col_back:
            st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
            if st.button("Back", key="configure_back", disabled=run_active):
                state.step = "detect"
            st.markdown("</div>", unsafe_allow_html=True)
        with col_forward:
            st.markdown('<div class="btn-success">', unsafe_allow_html=True)
            if st.button(
                "Continue",
                key="configure_continue",
                disabled=run_active,
            ):
                state.configure_values = current_values
                state.step = "running"
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def _running_step() -> None:
    st.markdown("### 4. Running the analysis")
    executor = get_thread_pool_executor()
    progress_queue = get_progress_queue()

    state = _get_state()
    if state.analysis_future and state.analysis_future.done():
        state.analysis_future = None
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
        return

    cancel_disabled = not state.analysis_future or state.analysis_future.done()
    if st.button("Cancelar análisis", disabled=cancel_disabled):
        if state.analysis_future and state.analysis_future.cancel():
            pass
        state.run_id = None
        state.last_run_success = False
        state.pipeline_error = "Análisis cancelado por el usuario."

    debug_enabled = bool((state.configure_values or {}).get("debug_video", True))

    state.last_run_success = False

    def _phase_for(p: int, *, debug_enabled: bool) -> str:
        if p < 10:
            return "Preparing…"
        if p < 25:
            return "Extracting frames…"
        if p < 50:
            return "Estimating pose…"
        if p < 65:
            return "Filtering and interpolating…"
        if debug_enabled and p < 75:
            return "Rendering debug video…"
        if debug_enabled:
            if p < 90:
                return "Computing metrics…"
        else:
            if p < 85:
                return "Computing metrics…"
        if p < 100:
            return "Counting repetitions…"
        return "Finishing up…"

    def make_cb(queue: SimpleQueue, run_id: str, debug_enabled: bool):
        def _cb(p: int) -> None:
            value = max(0, min(100, int(p)))
            phase = _phase_for(value, debug_enabled=debug_enabled)
            try:
                queue.put((run_id, value, phase))
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Error putting progress in queue: {exc}")

        return _cb

    _ensure_video_path()
    state = _get_state()
    if not state.video_path:
        state.pipeline_error = "The video to process was not found."
        state.step = "results"
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
        return

    if state.analysis_future is None:
        try:
            video_path, cfg, prefetched_detection = _prepare_pipeline_inputs(state)
        except ValueError as exc:
            state.pipeline_error = str(exc)
            state.step = "results"
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
            return

        state.pipeline_error = None
        state.report = None
        state.count_path = None
        state.metrics_path = None
        state.cfg_fingerprint = None
        state.progress_value_from_cb = 0
        state.phase_text_from_cb = _phase_for(0, debug_enabled=debug_enabled)

        _drain_queue(progress_queue)

        run_id = uuid4().hex
        state.run_id = run_id
        callback = make_cb(progress_queue, run_id, debug_enabled)

        def _job() -> tuple[str, Report]:
            report = _run_pipeline(
                video_path=video_path,
                cfg=cfg,
                prefetched_detection=prefetched_detection,
                progress_cb=callback,
            )
            return run_id, report

        state.analysis_future = executor.submit(_job)
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
        return

    future = state.analysis_future
    with st.status("Analizando vídeo...", expanded=True) as status:
        latest_progress = getattr(state, "progress_value_from_cb", 0)
        latest_phase = getattr(
            state,
            "phase_text_from_cb",
            _phase_for(latest_progress, debug_enabled=debug_enabled),
        )
        bar = st.progress(latest_progress)
        progress_message = status.empty()

        status.update(
            label=f"Analizando vídeo... {latest_progress}%",
            state="running",
            expanded=True,
        )
        bar.progress(latest_progress)
        progress_message.markdown(
            f"**Fase actual:** {latest_phase} ({latest_progress}%)"
        )

        while future and not future.done():
            state = _get_state()
            if state.run_id is None:
                state.report = None
                state.count_path = None
                state.metrics_path = None
                state.cfg_fingerprint = None
                state.last_run_success = False
                state.progress_value_from_cb = latest_progress
                state.phase_text_from_cb = latest_phase
                state.analysis_future = None
                state.pipeline_error = "Análisis cancelado por el usuario."
                state.step = "results"
                _drain_queue(progress_queue)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
                return
            while True:
                try:
                    msg_run_id, progress, phase = progress_queue.get_nowait()
                except Empty:
                    break
                if msg_run_id != state.run_id:
                    continue
                clamped_progress = max(latest_progress, progress)
                if clamped_progress != progress:
                    phase = _phase_for(
                        clamped_progress, debug_enabled=debug_enabled
                    )
                latest_progress = clamped_progress
                latest_phase = phase
                state.progress_value_from_cb = clamped_progress
                state.phase_text_from_cb = phase

            status.update(
                label=f"Analizando vídeo... {latest_progress}%",
                state="running",
                expanded=True,
            )
            bar.progress(latest_progress)
            progress_message.markdown(
                f"**Fase actual:** {latest_phase} ({latest_progress}%)"
            )
            time.sleep(0.2)
            future = state.analysis_future

        state = _get_state()
        current_future = state.analysis_future
        if not current_future:
            latest_phase = "Cancelado"
            state.pipeline_error = "Análisis cancelado por el usuario."
            state.report = None
            state.count_path = None
            state.metrics_path = None
            state.cfg_fingerprint = None
            state.last_run_success = False
            state.progress_value_from_cb = latest_progress
            state.phase_text_from_cb = latest_phase
            status.update(label="Análisis cancelado", state="error", expanded=True)
            bar.progress(latest_progress)
            progress_message.markdown(
                f"**Fase actual:** {latest_phase} ({latest_progress}%)"
            )
            _drain_queue(progress_queue)
            state.run_id = None
            state.step = "results"
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
            return

        try:
            while True:
                try:
                    msg_run_id, progress, phase = progress_queue.get_nowait()
                except Empty:
                    break
                if msg_run_id != state.run_id:
                    continue
                clamped_progress = max(latest_progress, progress)
                if clamped_progress != progress:
                    phase = _phase_for(
                        clamped_progress, debug_enabled=debug_enabled
                    )
                latest_progress = clamped_progress
                latest_phase = phase
            state.progress_value_from_cb = latest_progress
            state.phase_text_from_cb = latest_phase

            try:
                completed_run_id, report = current_future.result()
                _drain_queue(progress_queue)
            except CancelledError:
                latest_phase = "Cancelado"
                state.pipeline_error = "Análisis cancelado por el usuario."
                state.report = None
                state.count_path = None
                state.metrics_path = None
                state.cfg_fingerprint = None
                state.last_run_success = False
                state.progress_value_from_cb = latest_progress
                state.phase_text_from_cb = latest_phase
                state.run_id = None
                status.update(
                    label="Análisis cancelado",
                    state="error",
                    expanded=True,
                )
                bar.progress(latest_progress)
                progress_message.markdown(
                    f"**Fase actual:** {latest_phase} ({latest_progress}%)"
                )
                _drain_queue(progress_queue)
                state.analysis_future = None
                state.step = "results"
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
                return

            if state.run_id != completed_run_id:
                latest_phase = "Cancelado"
                state.pipeline_error = "Análisis cancelado por el usuario."
                state.report = None
                state.count_path = None
                state.metrics_path = None
                state.cfg_fingerprint = None
                state.last_run_success = False
                state.progress_value_from_cb = latest_progress
                state.phase_text_from_cb = latest_phase
                state.run_id = None
                status.update(
                    label="Análisis cancelado",
                    state="error",
                    expanded=False,
                )
                bar.progress(latest_progress)
                progress_message.markdown(
                    f"**Fase actual:** {latest_phase} ({latest_progress}%)"
                )
                _drain_queue(progress_queue)
                state.analysis_future = None
                state.step = "results"
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
                return

            state.pipeline_error = None
            state.report = report
            state.cfg_fingerprint = report.stats.config_sha1

            file_errors: list[str] = []
            video_path = state.video_path
            if video_path:
                counts_dir = Path(report.config_used.output.counts_dir)
                poses_dir = Path(report.config_used.output.poses_dir)
                try:
                    counts_dir.mkdir(parents=True, exist_ok=True)
                    poses_dir.mkdir(parents=True, exist_ok=True)
                except OSError as io_exc:
                    file_errors.append(f"Could not prepare output directories: {io_exc}")

                video_stem = Path(video_path).stem
                try:
                    count_path = counts_dir / f"{video_stem}_count.txt"
                    count_path.write_text(
                        f"{report.repetitions}\n", encoding="utf-8"
                    )
                    state.count_path = str(count_path)
                except OSError as io_exc:
                    state.count_path = None
                    file_errors.append(f"Could not write repetition count: {io_exc}")

                metrics_df = report.metrics
                if metrics_df is not None:
                    try:
                        metrics_path = poses_dir / f"{video_stem}_metrics.csv"
                        metrics_df.to_csv(metrics_path, index=False)
                        state.metrics_path = str(metrics_path)
                    except OSError as io_exc:
                        state.metrics_path = None
                        file_errors.append(f"Could not write metrics: {io_exc}")
                else:
                    state.metrics_path = None
            else:
                state.count_path = None
                state.metrics_path = None

            latest_progress = 100
            latest_phase = _phase_for(100, debug_enabled=debug_enabled)
            state.progress_value_from_cb = latest_progress
            state.phase_text_from_cb = latest_phase
            bar.progress(latest_progress)

            if file_errors:
                state.pipeline_error = "\n".join(file_errors)
                state.last_run_success = False
                status.update(
                    label="Análisis completado con errores",
                    state="error",
                    expanded=True,
                )
            else:
                state.last_run_success = True
                status.update(
                    label="Análisis completado!",
                    state="complete",
                    expanded=False,
                )

            progress_message.markdown(
                f"**Fase actual:** {latest_phase} ({latest_progress}%)"
            )
        except Exception as exc:
            state.pipeline_error = f"Error en el hilo de análisis: {exc}"
            state.report = None
            state.count_path = None
            state.metrics_path = None
            state.cfg_fingerprint = None
            state.progress_value_from_cb = latest_progress
            state.phase_text_from_cb = latest_phase
            state.last_run_success = False
            status.update(label="Error en el análisis", state="error", expanded=True)
        finally:
            _drain_queue(progress_queue)
            state = _get_state()
            state.run_id = None
            if state.analysis_future is current_future:
                state.analysis_future = None
                state.step = "results"
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
            return


def _results_panel() -> Dict[str, bool]:
    st.markdown('<div class="results-panel">', unsafe_allow_html=True)
    st.markdown("### 5. Results")

    actions: Dict[str, bool] = {"adjust": False, "reset": False}

    state = _get_state()

    if state.pipeline_error:
        st.error("An error occurred during the analysis")
        st.code(str(state.pipeline_error))
    elif state.report is not None:
        report: Report = state.report
        stats = report.stats
        repetitions = report.repetitions
        metrics_df = report.metrics

        st.markdown(f"**Detected repetitions:** {repetitions}")

        if report.debug_video_path and bool(
            (state.configure_values or {}).get("debug_video", True)
        ):
            st.video(str(report.debug_video_path))

        stats_rows = [
            {"Field": "CONFIG_SHA1", "Value": stats.config_sha1},
            {"Field": "fps_original", "Value": f"{stats.fps_original:.2f}"},
            {"Field": "fps_effective", "Value": f"{stats.fps_effective:.2f}"},
            {"Field": "frames", "Value": stats.frames},
            {
                "Field": "exercise_selected",
                "Value": stats.exercise_selected or "N/A",
            },
            {"Field": "exercise_detected", "Value": stats.exercise_detected},
            {"Field": "view_detected", "Value": stats.view_detected},
            {
                "Field": "detection_confidence",
                "Value": f"{stats.detection_confidence:.0%}",
            },
            {"Field": "primary_angle", "Value": stats.primary_angle or "N/A"},
            {"Field": "angle_range_deg", "Value": f"{stats.angle_range_deg:.2f}"},
            {"Field": "min_prominence", "Value": f"{stats.min_prominence:.2f}"},
            {"Field": "min_distance_sec", "Value": f"{stats.min_distance_sec:.2f}"},
            {"Field": "refractory_sec", "Value": f"{stats.refractory_sec:.2f}"},
        ]
        stats_df = pd.DataFrame(stats_rows, columns=["Field", "Value"]).astype({"Value": "string"})

        if metrics_df is not None:
            st.markdown("#### Calculated metrics")
            st.dataframe(metrics_df, use_container_width=True)
            numeric_columns = [
                col
                for col in metrics_df.columns
                if metrics_df[col].dtype.kind in "fi"
            ]
            if numeric_columns:
                default_selection = numeric_columns[:3]
                selected_metrics = st.multiselect(
                    "View metrics",
                    options=numeric_columns,
                    default=default_selection,
                )
                if selected_metrics:
                    st.line_chart(metrics_df[selected_metrics])

        if stats.warnings:
            st.warning("\n".join(f"• {msg}" for msg in stats.warnings))

        if stats.skip_reason:
            st.error(f"Repetition counting skipped: {stats.skip_reason}")

        with st.expander("Run statistics (optional)", expanded=False):
            try:
                st.dataframe(stats_df, use_container_width=True)
            except Exception:
                st.json({row["Field"]: row["Value"] for row in stats_rows})

            if state.video_path:
                st.markdown("### Original video")
                st.video(str(state.video_path))

        if state.metrics_path is not None:
            metrics_data = None
            try:
                metrics_data = Path(state.metrics_path).read_text(
                    encoding="utf-8"
                )
            except FileNotFoundError:
                st.error("The metrics file for download was not found.")
            except OSError as exc:
                st.error(f"Could not read metrics: {exc}")
            else:
                st.download_button(
                    "Download metrics",
                    data=metrics_data,
                    file_name=f"{Path(state.video_path).stem}_metrics.csv",
                    mime="text/csv",
                )

        if state.count_path is not None:
            count_data = None
            try:
                count_data = Path(state.count_path).read_text(
                    encoding="utf-8"
                )
            except FileNotFoundError:
                st.error("The repetition file for download was not found.")
            except OSError as exc:
                st.error(f"Could not read the repetition file: {exc}")
            else:
                st.download_button(
                    "Download repetition count",
                    data=count_data,
                    file_name=f"{Path(state.video_path).stem}_count.txt",
                    mime="text/plain",
                )

    else:
        st.info("No results found to display.")

    adjust_col, reset_col = st.columns(2)
    with adjust_col:
        if st.button("Adjust configuration and re-run", key="results_adjust"):
            actions["adjust"] = True
    with reset_col:
        if st.button("Back to start", key="results_reset"):
            actions["reset"] = True

    st.markdown("</div>", unsafe_allow_html=True)

    return actions


def _results_summary() -> None:
    st.markdown("### 4. Run the analysis")
    state = _get_state()
    if state.last_run_success:
        st.success("Analysis complete ✅")


def main() -> None:
    state = _get_state()

    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        _upload_step()
        state = _get_state()
        if (
            state.step in ("detect", "configure", "running", "results")
            and state.video_path
        ):
            _detect_step()

    results_actions: Dict[str, bool] = {"adjust": False, "reset": False}

    with col_mid:
        state = _get_state()
        step = state.step
        if step in ("configure", "running", "results"):
            disabled = step != "configure"
            show_actions = step == "configure"
            _configure_step(disabled=disabled, show_actions=show_actions)
            if step == "running":
                _running_step()
            elif step == "results":
                _results_summary()
        else:
            st.empty()

    with col_right:
        state = _get_state()
        if state.step == "results":
            results_actions = _results_panel()
        else:
            st.empty()

    if results_actions.get("adjust"):
        state = _get_state()
        state.step = "configure"
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    if results_actions.get("reset"):
        _reset_app()


if __name__ == "__main__":
    main()
