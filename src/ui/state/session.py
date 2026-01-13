"""Operaciones con efectos secundarios sobre el estado almacenado en Streamlit."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config.settings import DEFAULT_PREVIEW_FPS

from .model import AppState, CONFIG_DEFAULTS, Step


def get_state() -> AppState:
    """Obtiene o crea la instancia de ``AppState`` guardada en la sesión."""

    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    state: AppState = st.session_state.app_state
    if "video_uploader" in st.session_state:
        state.video_uploader = st.session_state.get("video_uploader")
    else:
        state.video_uploader = "__unset__"
    return state


def go_to(step: Step) -> None:
    """Actualiza el paso activo del asistente."""

    st.session_state.app_state.step = step


def reset_state(*, preserve_upload: bool = False) -> None:
    """Restablece el ``AppState`` manteniendo opcionalmente la subida actual."""

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
    state.metrics_path = None
    state.arm_debug_timeseries_path = None
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
    state.overlay_video_stream_path = None
    state.overlay_video_download_path = None
    state.exercise_selected = None
    state.exercise_pending_update = None
    state.view_selected = None
    state.view_pending_update = None
    state.detect_result = None
    state.configure_values = CONFIG_DEFAULTS.copy()
    state.ui_rev = 0
    state.step = Step.UPLOAD
    state.video_original_name = None

    for key in list(st.session_state.keys()):
        if key.startswith("metrics_multiselect_") or key.startswith("metrics_default_"):
            del st.session_state[key]

    if not preserve_upload:
        state.upload_data = None
        state.upload_token = None
    state.active_upload_token = None


def safe_rerun() -> None:
    """Intenta relanzar el script con la API estable y cae al método legacy."""

    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()
