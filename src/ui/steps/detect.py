from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from src.core.types import ExerciseType
from src.exercise_detection.exercise_detector import detect_exercise
from src.ui.state import DEFAULT_EXERCISE_LABEL, Step, get_state, go_to, trigger_rerun
from src.ui.video import render_uniform_video


def _exercise_display_name(ex_type: ExerciseType) -> str:
    return ex_type.value.replace("_", " ").title()


EXERCISE_CHOICES: List[Tuple[str, str]] = [
    (DEFAULT_EXERCISE_LABEL, "auto"),
] + [
    (_exercise_display_name(ex_type), ex_type.value)
    for ex_type in ExerciseType
    if ex_type is not ExerciseType.UNKNOWN
]

EXERCISE_LABELS: List[str] = [lbl for (lbl, _) in EXERCISE_CHOICES]
VALID_EXERCISE_LABELS = set(EXERCISE_LABELS)
EXERCISE_TO_CONFIG: Dict[str, str] = {lbl: key for (lbl, key) in EXERCISE_CHOICES}
CONFIG_TO_LABEL: Dict[str, str] = {key: lbl for (lbl, key) in EXERCISE_CHOICES}
EXERCISE_WIDGET_KEY = "exercise_select_value"


def _detect_step() -> None:
    st.markdown("### 2. Detect the exercise")
    state = get_state()
    video_path = state.video_path
    if video_path:
        render_uniform_video(
            str(video_path),
            key="detect_video",
            bottom_margin=0.15,
        )

    step = state.step or Step.UPLOAD
    is_active = (step == Step.DETECT) and (video_path is not None)

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

    select_col_label, select_col_control = st.columns([1, 2], gap="small")
    with select_col_label:
        st.markdown(
            '<div class="form-label form-label--inline">Select the exercise</div>',
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
    if detect_result is not None and detect_result.get("token") != token:
        detect_result = None
        state.detect_result = None

    info_container = st.container()
    if detect_result:
        if detect_result.get("error"):
            info_container.error(
                "Automatic exercise detection failed: " f"{detect_result.get('error')}"
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
                "Detected exercise: "
                f"{label_display} – {view_display} view ({confidence:.0%} confidence)."
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
                if st.button(
                    "Back",
                    key="detect_back",
                    use_container_width=True,
                ):
                    state.detect_result = None
                    go_to(Step.UPLOAD)
                    trigger_rerun()
            with continue_col:
                if st.button(
                    "Continue",
                    key="detect_continue",
                    use_container_width=True,
                ):
                    if current_exercise == DEFAULT_EXERCISE_LABEL:
                        detect_result = state.detect_result
                        if (
                            detect_result
                            and not detect_result.get("error")
                            and detect_result.get("label")
                            and detect_result.get("token") == token
                            and detect_result.get("accepted")
                        ):
                            go_to(Step.CONFIGURE)
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
                            go_to(Step.CONFIGURE)
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
                                trigger_rerun()
                    else:
                        state.detect_result = None
                        go_to(Step.CONFIGURE)
    else:
        actions_placeholder.empty()
