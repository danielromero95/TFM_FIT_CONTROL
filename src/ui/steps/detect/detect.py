"""Pantalla responsable de seleccionar o detectar automáticamente el ejercicio."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from src.core.types import ExerciseType
from src.exercise_detection.exercise_detector import detect_exercise
from src.ui.state import DEFAULT_EXERCISE_LABEL, Step, get_state, go_to, safe_rerun
from src.ui.video import VIDEO_VIEWPORT_HEIGHT_PX, render_uniform_video

from ..utils import step_container


def _exercise_display_name(ex_type: ExerciseType) -> str:
    """Formatea un ``ExerciseType`` en un nombre legible."""

    return ex_type.value.replace("_", " ").title()


EXERCISE_ITEMS: List[Tuple[str, str]] = [
    (_exercise_display_name(ex_type), ex_type.value)
    for ex_type in ExerciseType
    if ex_type is not ExerciseType.UNKNOWN
]
EX_LABELS: List[str] = [""] + [label for (label, _) in EXERCISE_ITEMS]
EX_TO_KEY: Dict[str, str] = {label: key for (label, key) in EXERCISE_ITEMS}
# Mapear Auto-Detect a un ejercicio real para no romper el pipeline.
EXERCISE_TO_CONFIG: Dict[str, str] = {DEFAULT_EXERCISE_LABEL: ExerciseType.SQUAT.value} | EX_TO_KEY

VIEW_LABELS = ["", "Front", "Lateral"]

EX_WIDGET_KEY = "detect_exercise_value"
VIEW_WIDGET_KEY = "detect_view_value"


def _resolve_overlay_video(state, *, debug_requested: bool) -> tuple[str | None, str | None, bool]:
    """Determina qué video mostrar y si falta la sobreimpresión de depuración."""

    overlay_missing = False
    overlay_source: str | None = None

    def _resolve_candidate(value) -> tuple[str | None, bool]:
        if not value:
            return None, False
        try:
            candidate = Path(str(value)).expanduser()
        except Exception:
            return None, debug_requested
        if candidate.exists() and candidate.is_file():
            return str(candidate), False
        return None, debug_requested

    for value in (
        getattr(state, "overlay_video_stream_path", None),
        getattr(state, "overlay_video_download_path", None),
    ):
        resolved, missing = _resolve_candidate(value)
        if resolved:
            overlay_source = resolved
            break
        if missing:
            overlay_missing = True

    if overlay_source is None:
        candidate_value = None
        report = getattr(state, "report", None)
        if report is not None:
            attr_groups = [
                ("overlay_video_stream_path", "overlay_video_stream"),
                (
                    "overlay_video_path",
                    "overlay_video",
                    "debug_video_path",
                    "debug_video",
                ),
            ]
            for group in attr_groups:
                for attr in group:
                    if hasattr(report, attr):
                        value = getattr(report, attr)
                        if value:
                            candidate_value = value
                            break
                if candidate_value:
                    break
        if candidate_value:
            resolved, missing = _resolve_candidate(candidate_value)
            overlay_source = resolved
            if missing:
                overlay_missing = True

    base_video = str(state.video_path) if state.video_path else None
    return base_video, overlay_source, overlay_missing


def _build_sync_channel(state) -> str | None:
    """Genera la clave de sincronización para vídeo y gráfica de métricas."""

    report = getattr(state, "report", None)
    stats = getattr(report, "stats", None)
    if stats and getattr(stats, "config_sha1", None):
        frames_val = getattr(stats, "frames", None)
        if frames_val is not None:
            return f"vmx-sync-{stats.config_sha1}-{frames_val}"
    return None


def _current_widget_labels(state) -> tuple[str, str]:
    """Obtiene las etiquetas iniciales de ejercicio y vista."""

    current_ex_label = "" if (state.exercise == DEFAULT_EXERCISE_LABEL) else (state.exercise or "")
    view_key_to_label = {"front": "Front", "side": "Lateral"}
    current_view_label = view_key_to_label.get(getattr(state, "view", "") or "", "")
    return current_ex_label, current_view_label


def _render_autodetect_button(
    *,
    state,
    video_path: str | None,
    current_ex_label: str,
    current_view_label: str,
) -> tuple[str, str]:
    """Ejecuta la detección automática y devuelve las etiquetas resultantes."""

    st.markdown('<div class="autodetect-btn">', unsafe_allow_html=True)
    autodetect_clicked = st.button("Auto-Detect", key="btn_autodetect", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    if not autodetect_clicked:
        return current_ex_label, current_view_label

    if not video_path:
        st.warning("Please upload a video first.")
        return current_ex_label, current_view_label

    with st.spinner("Detecting exercise…"):
        try:
            label_key, detected_view, confidence = detect_exercise(str(video_path))
        except Exception as exc:
            st.error(f"Automatic exercise detection failed: {exc}")
            return current_ex_label, current_view_label

    detected_label = next((lbl for (lbl, key) in EXERCISE_ITEMS if key == label_key), "")
    view_label = {"front": "Front", "side": "Lateral"}.get(detected_view, "")

    state.exercise = detected_label or DEFAULT_EXERCISE_LABEL
    state.view = detected_view if view_label else ""
    state.detect_result = {
        "label": label_key,
        "view": detected_view or "",
        "confidence": float(confidence),
    }
    state.ui_rev += 1

    if detected_label and view_label:
        st.success(f"Detected: **{detected_label}** — **{view_label}** view.")

    return detected_label or "", view_label or ""


def _render_selectors(
    *, state, current_ex_label: str, current_view_label: str
) -> tuple[str, str]:
    """Pinta los ``selectbox`` y devuelve el valor elegido por la persona usuaria."""

    ex_key = f"{EX_WIDGET_KEY}_{state.ui_rev}"
    view_key = f"{VIEW_WIDGET_KEY}_{state.ui_rev}"

    row1c1, row1c2 = st.columns([1, 2], gap="small")
    with row1c1:
        st.markdown('<div class="form-label">Exercise</div>', unsafe_allow_html=True)
    with row1c2:
        ex_index = EX_LABELS.index(current_ex_label) if current_ex_label in EX_LABELS else 0
        selected_exercise = st.selectbox(
            "Exercise",
            options=EX_LABELS,
            index=ex_index,
            key=ex_key,
            label_visibility="collapsed",
        )

    row2c1, row2c2 = st.columns([1, 2], gap="small")
    with row2c1:
        st.markdown('<div class="form-label">View</div>', unsafe_allow_html=True)
    with row2c2:
        view_index = VIEW_LABELS.index(current_view_label) if current_view_label in VIEW_LABELS else 0
        selected_view = st.selectbox(
            "View",
            options=VIEW_LABELS,
            index=view_index,
            key=view_key,
            label_visibility="collapsed",
        )

    return selected_exercise, selected_view


def _persist_manual_selection(state, *, selected_exercise: str, selected_view: str) -> None:
    """Sincroniza la selección manual con el ``AppState``."""

    state.exercise = selected_exercise or DEFAULT_EXERCISE_LABEL
    state.view = {"Front": "front", "Lateral": "side"}.get(selected_view or "", "")
    if state.detect_result:
        new_label_key = EX_TO_KEY.get(selected_exercise)
        if new_label_key:
            state.detect_result["label"] = new_label_key
        if state.view:
            state.detect_result["view"] = state.view


def _render_navigation(selected_exercise: str, selected_view: str) -> None:
    """Construye los botones de navegación inferiores del paso."""

    state = get_state()
    st.markdown('<div class="app-nav-buttons">', unsafe_allow_html=True)
    back_col, continue_col = st.columns(2, gap="small")
    with back_col:
        st.markdown('<div class="btn--back">', unsafe_allow_html=True)
        if st.button("Back", key="detect_back", width="stretch"):
            state.detect_result = None
            go_to(Step.UPLOAD)
            safe_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with continue_col:
        can_continue = bool(selected_exercise and selected_view)
        st.markdown('<div class="btn--continue">', unsafe_allow_html=True)
        if st.button(
            "Continue",
            key="detect_continue",
            width="stretch",
            disabled=not can_continue,
        ):
            go_to(Step.CONFIGURE)
            safe_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _detect_step() -> None:
    with step_container("detect"):
        st.markdown("### 2. Detect the exercise")

        state = get_state()
        debug_requested = bool((state.configure_values or {}).get("debug_video", True))

        base_video, overlay_source, overlay_missing = _resolve_overlay_video(
            state, debug_requested=debug_requested
        )
        video_to_show = overlay_source or base_video
        sync_channel = _build_sync_channel(state)

        if video_to_show:
            render_uniform_video(
                video_to_show,
                key="detect_primary_video",
                fixed_height_px=VIDEO_VIEWPORT_HEIGHT_PX,
                bottom_margin=0.0,
                sync_channel=sync_channel,
            )

        if (
            state.step == Step.RESULTS
            and debug_requested
            and overlay_source is None
            and overlay_missing
        ):
            st.warning(
                "Debug overlay video was not found. Showing the original upload instead.",
                icon="⚠️",
            )

        current_ex_label, current_view_label = _current_widget_labels(state)
        current_ex_label, current_view_label = _render_autodetect_button(
            state=state,
            video_path=base_video,
            current_ex_label=current_ex_label,
            current_view_label=current_view_label,
        )

        selected_exercise, selected_view = _render_selectors(
            state=state,
            current_ex_label=current_ex_label,
            current_view_label=current_view_label,
        )

        _persist_manual_selection(
            state,
            selected_exercise=selected_exercise,
            selected_view=selected_view,
        )

        _render_navigation(selected_exercise, selected_view)
