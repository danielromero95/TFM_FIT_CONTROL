from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from src.core.types import ExerciseType, ViewType
from src.exercise_detection.exercise_detector import detect_exercise
from src.ui.state import DEFAULT_EXERCISE_LABEL, Step, get_state, go_to, safe_rerun
from src.ui.video import render_uniform_video


def _exercise_display_name(ex_type: ExerciseType) -> str:
    return ex_type.value.replace("_", " ").title()


EXERCISE_ITEMS: List[Tuple[str, str]] = [
    (_exercise_display_name(ex_type), ex_type.value)
    for ex_type in ExerciseType
    if ex_type is not ExerciseType.UNKNOWN
]
EX_LABELS: List[str] = [""] + [label for (label, _) in EXERCISE_ITEMS]
EX_TO_KEY: Dict[str, str] = {label: key for (label, key) in EXERCISE_ITEMS}
EXERCISE_TO_CONFIG: Dict[str, str] = {DEFAULT_EXERCISE_LABEL: "auto"} | EX_TO_KEY

VIEW_LABELS = ["", "Front", "Lateral"]
VIEW_TO_KEY = {"Front": "front", "Lateral": "side"}
KEY_TO_VIEW = {value: key for key, value in VIEW_TO_KEY.items()}

EX_WIDGET_KEY = "detect_exercise_value"
VIEW_WIDGET_KEY = "detect_view_value"


def _detect_step() -> None:
    state = get_state()

    st.markdown('<div class="app-step app-step-detect">', unsafe_allow_html=True)
    st.markdown("### 2. Detect the exercise")

    video_path = state.video_path
    if video_path:
        render_uniform_video(
            str(video_path),
            key="detect_video",
            bottom_margin=0.0,
            fixed_height_px=400,
        )

    ex_value = st.session_state.get(EX_WIDGET_KEY, "")
    view_value = st.session_state.get(VIEW_WIDGET_KEY, "")

    if state.exercise and state.exercise != DEFAULT_EXERCISE_LABEL:
        if state.exercise in EX_LABELS:
            ex_value = state.exercise
    else:
        ex_value = ""

    if getattr(state, "view", ""):
        view_label = KEY_TO_VIEW.get(state.view or "", "")
        if view_label in VIEW_LABELS:
            view_value = view_label
    else:
        view_value = ""

    st.session_state[EX_WIDGET_KEY] = ex_value
    st.session_state[VIEW_WIDGET_KEY] = view_value

    st.markdown('<div class="autodetect-btn">', unsafe_allow_html=True)
    autodetect_clicked = st.button(
        "Auto-Detect",
        key="detect_autodetect",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if autodetect_clicked:
        state.detect_result = None
        if not video_path:
            st.warning("Please upload a video first.")
        else:
            with st.spinner("Detecting exercise…"):
                try:
                    label_key, detected_view, confidence = detect_exercise(str(video_path))
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Automatic exercise detection failed: {exc}")
                else:
                    detected_label = next(
                        (label for (label, key) in EXERCISE_ITEMS if key == label_key),
                        "",
                    )
                    try:
                        view_enum = ViewType(detected_view)
                    except ValueError:
                        view_enum = ViewType.UNKNOWN
                    detected_view_label = KEY_TO_VIEW.get(view_enum.value, "")

                    ex_value = detected_label
                    view_value = detected_view_label

                    st.session_state[EX_WIDGET_KEY] = ex_value
                    st.session_state[VIEW_WIDGET_KEY] = view_value

                    state.exercise = ex_value or DEFAULT_EXERCISE_LABEL
                    normalized_view = VIEW_TO_KEY.get(view_value, "")
                    state.view = normalized_view
                    state.detect_result = {
                        "label": label_key,
                        "view": normalized_view or view_enum.value,
                        "confidence": float(confidence),
                    }

                    if ex_value and view_value:
                        st.success(
                            f"Detected: **{ex_value}** — **{view_value}** view."
                        )

    st.markdown('<div class="form-row">', unsafe_allow_html=True)
    label_col, control_col = st.columns([1, 2], gap="small")
    with label_col:
        st.markdown('<div class="form-label">Exercise</div>', unsafe_allow_html=True)
    with control_col:
        selected_exercise = st.selectbox(
            "Exercise",
            options=EX_LABELS,
            index=EX_LABELS.index(ex_value) if ex_value in EX_LABELS else 0,
            key=EX_WIDGET_KEY,
            label_visibility="collapsed",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    state.exercise = selected_exercise or DEFAULT_EXERCISE_LABEL

    st.markdown('<div class="form-row">', unsafe_allow_html=True)
    view_label_col, view_control_col = st.columns([1, 2], gap="small")
    with view_label_col:
        st.markdown('<div class="form-label">View</div>', unsafe_allow_html=True)
    with view_control_col:
        selected_view = st.selectbox(
            "View",
            options=VIEW_LABELS,
            index=VIEW_LABELS.index(view_value) if view_value in VIEW_LABELS else 0,
            key=VIEW_WIDGET_KEY,
            label_visibility="collapsed",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    state.view = VIEW_TO_KEY.get(selected_view, "")

    if state.detect_result and selected_exercise and selected_view:
        state.detect_result.update(
            {
                "label": EX_TO_KEY.get(selected_exercise, state.detect_result.get("label", "unknown")),
                "view": state.view or state.detect_result.get("view", ""),
            }
        )

    st.markdown('<div class="app-nav-buttons">', unsafe_allow_html=True)
    back_col, continue_col = st.columns(2, gap="small")
    with back_col:
        if st.button("Back", key="detect_back", use_container_width=True):
            go_to(Step.UPLOAD)
            safe_rerun()
    with continue_col:
        disabled = not (selected_exercise and selected_view)
        if st.button(
            "Continue",
            key="detect_continue",
            use_container_width=True,
            disabled=disabled,
        ):
            go_to(Step.CONFIGURE)
            safe_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

