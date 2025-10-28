from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from src.core.types import ExerciseType
from src.exercise_detection.exercise_detector import detect_exercise
from src.ui.state import DEFAULT_EXERCISE_LABEL, Step, get_state, go_to, safe_rerun
from src.ui.video import render_uniform_video
from ..utils import step_container


def _exercise_display_name(ex_type: ExerciseType) -> str:
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


def _detect_step() -> None:
    with step_container("detect"):
        st.markdown("### 2. Detect the exercise")

        state = get_state()
        video_path = state.video_path

        # --- Video: ancho completo, altura fija más pequeña y sin espacio inferior
        if video_path:
            render_uniform_video(
                str(video_path),
                key="detect_video",
                bottom_margin=0.0,
                fixed_height_px=320,  # ↓ antes 400
            )

        # --- Valor actual desde AppState (NO tocar session_state de los widgets)
        # Exercise: vacío si está en Auto-Detect
        current_ex_label = "" if (state.exercise == DEFAULT_EXERCISE_LABEL) else (state.exercise or "")
        # View: convertir 'front'/'side' a etiqueta
        view_key_to_label = {"front": "Front", "side": "Lateral"}
        current_view_label = view_key_to_label.get(getattr(state, "view", "") or "", "")

        # --- Auto-Detect (actualiza AppState y NO st.session_state de los widgets)
        st.markdown('<div class="autodetect-btn">', unsafe_allow_html=True)
        autodetect_clicked = st.button("Auto-Detect", key="btn_autodetect", width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

        if autodetect_clicked:
            if not video_path:
                st.warning("Please upload a video first.")
            else:
                with st.spinner("Detecting exercise…"):
                    try:
                        label_key, detected_view, confidence = detect_exercise(str(video_path))
                    except Exception as exc:
                        st.error(f"Automatic exercise detection failed: {exc}")
                    else:
                        # Mapear clave → etiqueta visible
                        detected_label = next((lbl for (lbl, key) in EXERCISE_ITEMS if key == label_key), "")
                        view_label = {"front": "Front", "side": "Lateral"}.get(detected_view, "")

                        # Persistir SOLO en AppState
                        state.exercise = detected_label or DEFAULT_EXERCISE_LABEL
                        state.view = detected_view if view_label else ""
                        state.detect_result = {
                            "label": label_key,
                            "view": detected_view or "",
                            "confidence": float(confidence),
                        }

                        # Actualizar "current_*" para que el select se pinte ya con el nuevo valor
                        current_ex_label = detected_label or ""
                        current_view_label = view_label or ""

                        # Fuerza refresco de widgets SIN tocar session_state:
                        state.ui_rev += 1

                        if current_ex_label and current_view_label:
                            st.success(f"Detected: **{current_ex_label}** — **{current_view_label}** view.")

        # --- Form rows (usar index derivado del estado; NO escribir a session_state)
        # Claves versionadas para recrear selectboxes tras Auto-Detect
        ex_key = f"{EX_WIDGET_KEY}_{state.ui_rev}"
        view_key = f"{VIEW_WIDGET_KEY}_{state.ui_rev}"

        # Exercise row
        row1c1, row1c2 = st.columns([1, 2], gap="small")
        with row1c1:
            st.markdown('<div class="form-label">Exercise</div>', unsafe_allow_html=True)
        with row1c2:
            ex_index = EX_LABELS.index(current_ex_label) if current_ex_label in EX_LABELS else 0
            selected_exercise = st.selectbox(
                "Exercise",
                options=EX_LABELS,
                index=ex_index,
                key=ex_key,  # clave versionada; no usamos session_state
                label_visibility="collapsed",
            )

        # View row
        row2c1, row2c2 = st.columns([1, 2], gap="small")
        with row2c1:
            st.markdown('<div class="form-label">View</div>', unsafe_allow_html=True)
        with row2c2:
            view_index = VIEW_LABELS.index(current_view_label) if current_view_label in VIEW_LABELS else 0
            selected_view = st.selectbox(
                "View",
                options=VIEW_LABELS,
                index=view_index,
                key=view_key,  # idem
                label_visibility="collapsed",
            )

        # --- Persistir selección manual en AppState
        state.exercise = selected_exercise or DEFAULT_EXERCISE_LABEL
        state.view = {"Front": "front", "Lateral": "side"}.get(selected_view or "", "")
        if state.detect_result:
            new_label_key = EX_TO_KEY.get(selected_exercise)
            if new_label_key:
                state.detect_result["label"] = new_label_key
            if state.view:
                state.detect_result["view"] = state.view

        # --- Navegación: Back | Continue
        st.markdown('<div class="app-nav-buttons">', unsafe_allow_html=True)
        back_col, continue_col = st.columns(2, gap="small")
        with back_col:
            if st.button("Back", key="detect_back", width='stretch'):
                state.detect_result = None
                go_to(Step.UPLOAD)
                safe_rerun()
        with continue_col:
            can_continue = bool(selected_exercise and selected_view)
            if st.button(
                "Continue",
                key="detect_continue",
                width='stretch',
                disabled=not can_continue,
            ):
                go_to(Step.CONFIGURE)
                safe_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

