# src/app.py
"""Streamlit front-end for the analysis pipeline."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

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
from src.detect.exercise_detector import detect_exercise
from src.pipeline import Report, run_pipeline

EXERCISE_CHOICES = [
    ("Auto-Detect", "auto", True),
    ("Squat", "squat", True),
    ("Deadlift", "deadlift", False),
    ("Bench Press", "benchpress", False),
]

DEFAULT_EXERCISE_LABEL = "Auto-Detect"
ENABLED_EXERCISE_LABELS = [lbl for (lbl, _, en) in EXERCISE_CHOICES if en]
EXERCISE_LABELS = [lbl for (lbl, _, _) in EXERCISE_CHOICES]
EXERCISE_TO_CONFIG = {lbl: key for (lbl, key, _) in EXERCISE_CHOICES}
CONFIG_TO_EXERCISE = {key: lbl for (lbl, key, _) in EXERCISE_CHOICES}
CONFIG_DEFAULTS: Dict[str, float | str | bool | None] = {
    "low": 80,
    "high": 150,
    "primary_angle": "left_knee",
    "min_prominence": 10.0,
    "min_distance_sec": 0.5,
    "debug_video": True,
    "use_crop": True,
    "target_fps": None,
}


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

      .btn-danger > button, .btn-success > button {
        border-radius: 12px !important;
        border-width: 1px !important;
        background: transparent !important;
        transition: background .15s ease, border-color .15s ease;
      }
      .btn-danger > button {
        color: #ef4444 !important;
        border-color: rgba(239,68,68,.6) !important;
      }
      .btn-danger > button:hover {
        background: rgba(239,68,68,.10) !important;
        border-color: rgba(239,68,68,.9) !important;
      }

      .btn-success > button {
        color: #22c55e !important;
        border-color: rgba(34,197,94,.6) !important;
      }
      .btn-success > button:hover {
        background: rgba(34,197,94,.10) !important;
        border-color: rgba(34,197,94,.9) !important;
      }

      .btn-danger > button[disabled], .btn-success > button[disabled] {
        opacity: .6 !important;
        cursor: not-allowed !important;
      }

      .chip {display:inline-block;padding:.2rem .5rem;border-radius:9999px;font-size:.8rem;
             margin-right:.25rem}
      .chip.ok {background:#065f46;color:#ecfdf5;}      /* green */
      .chip.warn {background:#7c2d12;color:#ffedd5;}    /* amber/brown */
      .chip.info {background:#1e293b;color:#e2e8f0;}    /* slate */

      .spacer-sm { height: .5rem; }

      /* Step 2: fix preview width, keep aspect and black bars when needed */
      .step-detect [data-testid="stVideo"] {
        max-width: 720px;
        margin: 0 auto .5rem auto;
      }
      .step-detect [data-testid="stVideo"] video {
        width: min(720px, 100%) !important;
        aspect-ratio: 16 / 9;
        background: #000;
        object-fit: contain;
      }

      .step-detect .readonly-hint {
        color: #94a3b8;
        font-size: .85rem;
      }

      /* Keep title above content and safe on very narrow viewports */
      header[data-testid="stHeader"] { z-index: 1000; }
      @media (max-width: 520px) {
        header[data-testid="stHeader"]::before { font-size: 16px; }
      }
      @media (max-width: 420px) {
        header[data-testid="stHeader"]::before { display: none; }
      }

      /* Make the select a little wider than the button for visual balance */
      .step-detect .row-detect { display: grid; grid-template-columns: 1fr 2.25fr; gap: 1rem; }
      .step-detect .stSelect { margin-top: .25rem; }

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

    # build a CSS selector for disabled radio items based on EXERCISE_CHOICES
    disabled_indices = [i + 1 for i, (_, _, en) in enumerate(EXERCISE_CHOICES) if not en]
    disabled_css = ""
    if disabled_indices:
        selector = ", ".join(
            f'.step-detect [data-testid="stRadio"] div[role="radiogroup"] > label:nth-child({i})'
            for i in disabled_indices
        )
        disabled_css = f"""
{selector} {{
  pointer-events: none;
  opacity: .45;
  border-style: dashed;
  border-color: rgba(148,163,184,.18);
}}
{selector} input {{ pointer-events: none; }}
"""
    st.markdown(f"<style>{disabled_css}</style>", unsafe_allow_html=True)

    # segmented radio visuals + compact spacing under the video
    st.markdown(
        """
<style>
.step-detect [data-testid="stRadio"] { margin-top: .25rem; }
.step-detect [data-testid="stRadio"] > div[role="radiogroup"] {
  display:flex; flex-wrap:wrap; gap:.5rem;
}
.step-detect [data-testid="stRadio"] label[data-baseweb="radio"]{
  flex:1 1 120px; border-radius:999px; border:1px solid rgba(148,163,184,.25);
  background:rgba(15,23,42,.78); color:#e2e8f0; padding:.5rem 1rem; font-weight:600;
}
.step-detect [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child{ display:none; }
.step-detect [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked){
  background:linear-gradient(135deg, rgba(59,130,246,.85), rgba(37,99,235,.85));
  border-color:rgba(59,130,246,.75); color:#f8fafc;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    _inject_css()
    if "step" not in st.session_state:
        st.session_state.step = "upload"
    if "upload_data" not in st.session_state:
        st.session_state.upload_data = None
    if "upload_token" not in st.session_state:
        st.session_state.upload_token = None
    if "active_upload_token" not in st.session_state:
        st.session_state.active_upload_token = None
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "exercise" not in st.session_state:
        st.session_state.exercise = DEFAULT_EXERCISE_LABEL
    if "last_valid_exercise" not in st.session_state:
        st.session_state.last_valid_exercise = DEFAULT_EXERCISE_LABEL
    if "detect_result" not in st.session_state:
        st.session_state.detect_result = None
    if "configure_values" not in st.session_state:
        st.session_state.configure_values = CONFIG_DEFAULTS.copy()
    if "report" not in st.session_state:
        st.session_state.report = None
    if "pipeline_error" not in st.session_state:
        st.session_state.pipeline_error = None
    if "count_path" not in st.session_state:
        st.session_state.count_path = None
    if "metrics_path" not in st.session_state:
        st.session_state.metrics_path = None
    if "cfg_fingerprint" not in st.session_state:
        st.session_state.cfg_fingerprint = None
    if "last_run_success" not in st.session_state:
        st.session_state.last_run_success = False


def _reset_state(*, preserve_upload: bool = False) -> None:
    video_path = st.session_state.get("video_path")
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
    st.session_state.video_path = None
    st.session_state.report = None
    st.session_state.pipeline_error = None
    st.session_state.count_path = None
    st.session_state.metrics_path = None
    st.session_state.cfg_fingerprint = None
    st.session_state.last_run_success = False
    st.session_state.exercise = DEFAULT_EXERCISE_LABEL
    st.session_state.last_valid_exercise = DEFAULT_EXERCISE_LABEL
    st.session_state.detect_result = None
    st.session_state.configure_values = CONFIG_DEFAULTS.copy()
    st.session_state.step = "upload"
    if not preserve_upload:
        st.session_state.upload_data = None
        st.session_state.upload_token = None
    st.session_state.active_upload_token = None


def _reset_app() -> None:
    _reset_state()
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def _ensure_video_path() -> None:
    upload_data = st.session_state.upload_data
    if not upload_data:
        return
    # If we're replacing an existing temp file, delete it first
    old_path = st.session_state.get("video_path")
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
    st.session_state.video_path = tmp_file.name
    st.session_state.detect_result = None
    # Free large byte payload from session to reduce memory usage once persisted
    st.session_state.upload_data = None


def _run_pipeline(progress_cb=None) -> None:
    st.session_state.pipeline_error = None
    st.session_state.report = None
    st.session_state.count_path = None
    st.session_state.metrics_path = None
    st.session_state.cfg_fingerprint = None

    video_path = st.session_state.get("video_path")
    if not video_path:
        st.session_state.pipeline_error = "The video to process was not found."
        return

    cfg = config.load_default()

    cfg_values = st.session_state.get("configure_values", CONFIG_DEFAULTS)
    cfg.faults.low_thresh = float(cfg_values.get("low", CONFIG_DEFAULTS["low"]))
    cfg.faults.high_thresh = float(cfg_values.get("high", CONFIG_DEFAULTS["high"]))
    cfg.counting.primary_angle = str(cfg_values.get("primary_angle", CONFIG_DEFAULTS["primary_angle"]))
    cfg.counting.min_prominence = float(cfg_values.get("min_prominence", CONFIG_DEFAULTS["min_prominence"]))
    cfg.counting.min_distance_sec = float(cfg_values.get("min_distance_sec", CONFIG_DEFAULTS["min_distance_sec"]))
    cfg.debug.generate_debug_video = bool(cfg_values.get("debug_video", True))
    cfg.pose.use_crop = bool(cfg_values.get("use_crop", True))
    tfps = cfg_values.get("target_fps")
    if tfps not in (None, "", 0) and hasattr(cfg, "video"):
        try:
            cfg.video.target_fps = float(tfps)
            if hasattr(cfg.video, "manual_sample_rate"):
                cfg.video.manual_sample_rate = None
        except (TypeError, ValueError):
            pass

    exercise_label = st.session_state.get("exercise", DEFAULT_EXERCISE_LABEL)
    cfg.counting.exercise = EXERCISE_TO_CONFIG.get(
        exercise_label,
        EXERCISE_TO_CONFIG.get(DEFAULT_EXERCISE_LABEL, "squat"),
    )

    det = st.session_state.get("detect_result")
    prefetched_detection = None
    if det:
        prefetched_detection = (
            det.get("label", "unknown"),
            det.get("view", "unknown"),
            float(det.get("confidence", 0.0)),
        )

    try:
        report: Report = run_pipeline(
            str(video_path),
            cfg,
            progress_callback=progress_cb,
            prefetched_detection=prefetched_detection,
        )
    except Exception as exc:  # pragma: no cover - surfaced to the UI user
        st.session_state.pipeline_error = str(exc)
        return

    st.session_state.report = report
    st.session_state.cfg_fingerprint = report.stats.config_sha1

    counts_dir = Path(cfg.output.counts_dir)
    poses_dir = Path(cfg.output.poses_dir)
    counts_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem
    count_path = counts_dir / f"{video_stem}_count.txt"
    count_path.write_text(f"{report.repetitions}\n", encoding="utf-8")
    st.session_state.count_path = str(count_path)

    metrics_df = report.metrics
    if metrics_df is not None:
        metrics_path = poses_dir / f"{video_stem}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        st.session_state.metrics_path = str(metrics_path)
    else:
        st.session_state.metrics_path = None


def _upload_step() -> None:
    st.markdown("### 1. Upload your video")
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv", "mpg", "mpeg", "wmv"],
        key="video_uploader",
        label_visibility="collapsed",
    )
    previous_token = st.session_state.get("active_upload_token")
    if uploaded is not None:
        data_bytes = uploaded.getvalue()
        new_token = (
            uploaded.name,
            len(data_bytes),
            hashlib.md5(data_bytes).hexdigest(),
        )
        if previous_token != new_token:
            _reset_state(preserve_upload=True)
            st.session_state.upload_data = {
                "name": uploaded.name,
                "bytes": data_bytes,
            }
            st.session_state.upload_token = new_token
            st.session_state.active_upload_token = new_token
            _ensure_video_path()
            if st.session_state.video_path:
                st.session_state.step = "detect"
        else:
            st.session_state.active_upload_token = new_token


def _detect_step() -> None:
    st.markdown("### 2. Detect the exercise")
    st.markdown('<div class="step step-detect">', unsafe_allow_html=True)
    video_path = st.session_state.get("video_path")
    if video_path:
        st.video(str(video_path))

    detect_readonly = (
        st.session_state.step != "detect" or video_path is None
    )

    st.markdown('<div class="row-detect">', unsafe_allow_html=True)
    col_select, col_detect = st.columns([3, 1])

    with col_select:
        st.caption("Select the exercise")
        st.radio(
            "",
            options=EXERCISE_LABELS,
            key="exercise",
            label_visibility="collapsed",
            horizontal=True,
            disabled=detect_readonly,
        )
        # Enforce disabled choices and remember the last valid one
        current = st.session_state.get("exercise", DEFAULT_EXERCISE_LABEL)
        last_valid = st.session_state.get("last_valid_exercise", DEFAULT_EXERCISE_LABEL)
        if current in ENABLED_EXERCISE_LABELS:
            st.session_state.last_valid_exercise = current
        else:
            st.session_state.exercise = last_valid

    with col_detect:
        st.markdown('<div class="btn-success">', unsafe_allow_html=True)
        if st.button("Detect", key="detect_run", disabled=detect_readonly):
            vp = st.session_state.get("video_path")
            if not vp:
                st.warning("Upload a video before detecting the exercise.")
            else:
                with st.spinner("Analyzing the video…"):
                    label, view, confidence = detect_exercise(str(vp))
                if label == "unknown" and view == "unknown" and confidence <= 0.0:
                    st.session_state.detect_result = None
                    st.warning(
                        "Could not determine the exercise automatically. Select it manually."
                    )
                else:
                    st.session_state.detect_result = {
                        "label": label,
                        "view": view,
                        "confidence": float(confidence),
                    }
                    # Map to visible label; fallback to Auto-Detect if not enabled
                    mapped = CONFIG_TO_EXERCISE.get(label, DEFAULT_EXERCISE_LABEL)
                    if mapped not in ENABLED_EXERCISE_LABELS:
                        mapped = DEFAULT_EXERCISE_LABEL
                    st.session_state.exercise = mapped
                    st.session_state.last_valid_exercise = mapped
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    detect_result = st.session_state.get("detect_result")
    if detect_result:
        label = detect_result["label"]
        view = detect_result["view"]
        confidence_pct = int(round(detect_result["confidence"] * 100))
        st.info(
            f"Detected exercise: {label} · View: {view} · Confidence: {confidence_pct}%"
        )

    if detect_readonly:
        st.markdown(
            '<div class="readonly-hint">Detection controls are read-only during this step.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

    if st.session_state.step == "detect":
        col_back, col_forward = st.columns(2)
        with col_back:
            st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
            if st.button("Back", key="detect_back"):
                st.session_state.step = "upload"
            st.markdown("</div>", unsafe_allow_html=True)
        with col_forward:
            st.markdown('<div class="btn-success">', unsafe_allow_html=True)
            if st.button("Continue", key="detect_continue"):
                st.session_state.step = "configure"
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _configure_step(*, disabled: bool = False, show_actions: bool = True) -> None:
    st.markdown("### 3. Configure the analysis")
    cfg_values = st.session_state.get("configure_values", CONFIG_DEFAULTS.copy())

    if disabled:
        st.info("The configuration is displayed for reference only while the analysis runs.")

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
    use_crop = st.checkbox(
        "Use automatic crop (MediaPipe)",
        value=bool(cfg_values.get("use_crop", CONFIG_DEFAULTS["use_crop"])),
        disabled=disabled,
        key="cfg_use_crop",
    )

    target_fps_current = cfg_values.get("target_fps", CONFIG_DEFAULTS.get("target_fps"))
    target_fps_default = "" if target_fps_current in (None, "") else str(target_fps_current)
    target_fps_raw = st.text_input(
        "Target FPS after resampling",
        value=target_fps_default,
        help="Leave it empty to disable resampling.",
        disabled=disabled,
        key="cfg_target_fps",
    )
    target_fps_error = False
    if disabled:
        raw_value = cfg_values.get("target_fps", None)
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
        if raw_value in (None, "", 0):
            target_fps_value = None
        else:
            try:
                target_fps_value = float(raw_value)
            except (TypeError, ValueError):
                target_fps_value = None
    elif target_fps_raw.strip():
        try:
            parsed_target_fps = float(target_fps_raw)
        except ValueError:
            target_fps_error = True
            target_fps_value = cfg_values.get("target_fps", None)
            st.warning("Enter a valid target FPS or leave the field empty.")
        else:
            target_fps_value = parsed_target_fps
    else:
        target_fps_value = None

    current_values = {
        "low": float(low),
        "high": float(high),
        "primary_angle": primary_angle,
        "min_prominence": float(min_prominence),
        "min_distance_sec": float(min_distance_sec),
        "debug_video": bool(debug_video),
        "use_crop": bool(use_crop),
        "target_fps": target_fps_value,
    }
    if not disabled and not target_fps_error:
        st.session_state.configure_values = current_values

    if show_actions and not disabled:
        col_back, col_forward = st.columns(2)
        with col_back:
            st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
            if st.button("Back", key="configure_back"):
                st.session_state.step = "detect"
            st.markdown("</div>", unsafe_allow_html=True)
        with col_forward:
            st.markdown('<div class="btn-success">', unsafe_allow_html=True)
            if st.button("Continue", key="configure_continue"):
                if target_fps_error:
                    st.warning("Fix the target FPS value before continuing.")
                else:
                    st.session_state.configure_values = current_values
                    st.session_state.step = "running"
            st.markdown("</div>", unsafe_allow_html=True)


def _running_step() -> None:
    st.markdown("### 4. Running the analysis")
    progress_placeholder = st.progress(0)
    phase_placeholder = st.empty()
    debug_enabled = bool(
        st.session_state.get("configure_values", {}).get("debug_video", True)
    )

    st.session_state.last_run_success = False

    def _phase_for(p: int) -> str:
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

    def _cb(p: int) -> None:
        p = max(0, min(100, int(p)))
        progress_placeholder.progress(p)
        phase_placeholder.text(_phase_for(p))

    phase_placeholder.text(_phase_for(0))

    _ensure_video_path()
    if not st.session_state.video_path:
        st.session_state.pipeline_error = "The video to process was not found."
        st.session_state.step = "results"
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
        return

    with st.spinner("Processing video…"):
        _run_pipeline(progress_cb=_cb)

    st.session_state.last_run_success = (
        st.session_state.pipeline_error is None
        and st.session_state.report is not None
    )
    st.session_state.step = "results"
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def _results_panel() -> Dict[str, bool]:
    st.markdown('<div class="results-panel">', unsafe_allow_html=True)
    st.markdown("### 5. Results")

    actions: Dict[str, bool] = {"adjust": False, "reset": False}

    if st.session_state.pipeline_error:
        st.error("An error occurred during the analysis")
        st.code(str(st.session_state.pipeline_error))
    elif st.session_state.report is not None:
        report: Report = st.session_state.report
        stats = report.stats
        repetitions = report.repetitions
        metrics_df = report.metrics

        st.markdown(f"**Detected repetitions:** {repetitions}")

        if report.debug_video_path and bool(
            st.session_state.get("configure_values", {}).get("debug_video", True)
        ):
            st.video(str(report.debug_video_path))
            st.caption("Video with landmarks")

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

            if st.session_state.video_path:
                st.markdown("### Original video")
                st.video(str(st.session_state.video_path))

        if st.session_state.metrics_path is not None:
            metrics_data = None
            try:
                metrics_data = Path(st.session_state.metrics_path).read_text(
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
                    file_name=f"{Path(st.session_state.video_path).stem}_metrics.csv",
                    mime="text/csv",
                )

        if st.session_state.count_path is not None:
            count_data = None
            try:
                count_data = Path(st.session_state.count_path).read_text(
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
                    file_name=f"{Path(st.session_state.video_path).stem}_count.txt",
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
    if st.session_state.get("last_run_success"):
        st.success("Analysis complete ✅")


def main() -> None:
    _init_session_state()

    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        _upload_step()
        if (
            st.session_state.step in ("detect", "configure", "running", "results")
            and st.session_state.video_path
        ):
            _detect_step()

    results_actions: Dict[str, bool] = {"adjust": False, "reset": False}

    with col_mid:
        step = st.session_state.step
        if step in ("configure", "running"):
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
        if st.session_state.step == "results":
            results_actions = _results_panel()
        else:
            st.empty()

    if results_actions.get("adjust"):
        st.session_state.step = "configure"
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    if results_actions.get("reset"):
        _reset_app()


if __name__ == "__main__":
    main()
