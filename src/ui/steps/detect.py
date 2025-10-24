from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from src.core.types import ExerciseType
from src.exercise_detection.exercise_detector import detect_exercise
from src.ui.state import (
    DEFAULT_EXERCISE_LABEL,
    Step,
    get_state,
    go_to,
    safe_rerun,
)
from src.ui.video import render_uniform_video


def _exercise_display_name(ex_type: ExerciseType) -> str:
    return ex_type.value.replace("_", " ").title()


EXERCISE_EMPTY_LABEL = ""
EXERCISE_CHOICES: List[Tuple[str, str]] = [
    (EXERCISE_EMPTY_LABEL, ""),
] + [
    (_exercise_display_name(ex_type), ex_type.value)
    for ex_type in ExerciseType
    if ex_type is not ExerciseType.UNKNOWN
]
EXERCISE_LABELS: List[str] = [lbl for (lbl, _) in EXERCISE_CHOICES]
VALID_EXERCISE_LABELS = set(EXERCISE_LABELS + [DEFAULT_EXERCISE_LABEL])
EXERCISE_TO_CONFIG: Dict[str, str] = {lbl: key for (lbl, key) in EXERCISE_CHOICES if key}
CONFIG_TO_LABEL: Dict[str, str] = {key: lbl for (lbl, key) in EXERCISE_CHOICES if key}
FALLBACK_MANUAL_CONFIG = next((key for (_, key) in EXERCISE_CHOICES if key), "squat")
EXERCISE_WIDGET_KEY = "exercise_select_value"

VIEW_EMPTY_LABEL = ""
VIEW_CHOICES: List[Tuple[str, str]] = [
    (VIEW_EMPTY_LABEL, "unknown"),
    ("Front View", "front"),
    ("Lateral View", "lateral"),
]
VIEW_LABELS: List[str] = [lbl for (lbl, _) in VIEW_CHOICES]
VIEW_TO_KEY: Dict[str, str] = {lbl: key for (lbl, key) in VIEW_CHOICES}
KEY_TO_VIEW_LABEL: Dict[str, str] = {key: lbl for (lbl, key) in VIEW_CHOICES}
VIEW_WIDGET_KEY = "view_select_value"


def _view_display(key: str | None) -> str:
    normalized = (key or "").strip().lower()
    if normalized in ("front", "frontal"):
        return "Front View"
    if normalized in ("side", "lateral"):
        return "Lateral View"
    if not normalized or normalized == "unknown":
        return "Unknown View"
    return normalized.replace("_", " ").title()


def _detect_step() -> None:
    st.markdown('<div class="app-step app-step-detect">', unsafe_allow_html=True)
    st.markdown("### 2. Detect the exercise")

    state = get_state()
    video_path = state.video_path
    step = state.step or Step.UPLOAD
    is_active = (step == Step.DETECT) and (video_path is not None)

    token = state.upload_token
    detect_result = state.detect_result
    if detect_result is not None and detect_result.get("token") != token:
        state.detect_result = None
        detect_result = None

    pending_exercise = state.exercise_pending_update
    state.exercise_pending_update = None
    if pending_exercise and pending_exercise in VALID_EXERCISE_LABELS:
        state.exercise = pending_exercise

    current_exercise_state = state.exercise or DEFAULT_EXERCISE_LABEL
    if current_exercise_state not in VALID_EXERCISE_LABELS:
        current_exercise_state = DEFAULT_EXERCISE_LABEL
        state.exercise = current_exercise_state

    default_ui_exercise = (
        current_exercise_state
        if current_exercise_state in EXERCISE_LABELS
        else EXERCISE_EMPTY_LABEL
    )

    session_exercise = st.session_state.get(EXERCISE_WIDGET_KEY)
    if session_exercise not in EXERCISE_LABELS:
        session_exercise = default_ui_exercise
        st.session_state[EXERCISE_WIDGET_KEY] = session_exercise

    detected_view_label = VIEW_LABELS[0]
    if detect_result and detect_result.get("view"):
        detected_view_key = str(detect_result.get("view", "")).lower()
        detected_view_label = KEY_TO_VIEW_LABEL.get(detected_view_key, detected_view_label)

    if detect_result and not detect_result.get("accepted"):
        st.session_state[VIEW_WIDGET_KEY] = detected_view_label
    else:
        session_view = st.session_state.get(VIEW_WIDGET_KEY)
        if session_view not in VIEW_LABELS:
            st.session_state[VIEW_WIDGET_KEY] = detected_view_label

    session_view_label = st.session_state.get(VIEW_WIDGET_KEY, VIEW_LABELS[0])
    if session_view_label not in VIEW_LABELS:
        session_view_label = VIEW_LABELS[0]
        st.session_state[VIEW_WIDGET_KEY] = session_view_label

    if video_path:
        render_uniform_video(
            str(video_path),
            key="detect_video",
            bottom_margin=0.0,
            max_width=None,
            viewport_height=360,
        )
    else:
        st.info("Upload a workout video to enable detection.")

    auto_clicked = st.button(
        "Auto-Detect",
        key="detect_auto",
        use_container_width=True,
        type="primary",
        disabled=not is_active,
    )

    exercise_label_col, exercise_input_col = st.columns([1, 4], gap="medium")
    with exercise_label_col:
        st.markdown('<div class="form-label form-label--inline">Exercise</div>', unsafe_allow_html=True)
    with exercise_input_col:
        selected_exercise_label = st.selectbox(
            "Select exercise",
            options=EXERCISE_LABELS,
            index=EXERCISE_LABELS.index(session_exercise),
            key=EXERCISE_WIDGET_KEY,
            label_visibility="collapsed",
            disabled=not is_active,
        )

    view_label_col, view_input_col = st.columns([1, 4], gap="medium")
    with view_label_col:
        st.markdown('<div class="form-label form-label--inline">View</div>', unsafe_allow_html=True)
    with view_input_col:
        selected_view_label = st.selectbox(
            "Select view",
            options=VIEW_LABELS,
            index=VIEW_LABELS.index(session_view_label),
            key=VIEW_WIDGET_KEY,
            label_visibility="collapsed",
            disabled=not is_active,
        )

    state.exercise = (
        selected_exercise_label if selected_exercise_label else DEFAULT_EXERCISE_LABEL
    )

    selected_view_key = VIEW_TO_KEY.get(selected_view_label, "unknown")

    detect_result_local = detect_result
    if state.exercise != DEFAULT_EXERCISE_LABEL and detect_result_local is not None:
        state.detect_result = None
        detect_result_local = None
        detect_result = None

    feedback_container = st.container()
    if detect_result_local:
        if detect_result_local.get("error"):
            feedback_container.error(
                "Automatic exercise detection failed: "
                f"{detect_result_local.get('error')}"
            )
            feedback_container.info(
                "You can adjust the selection manually or try detecting again."
            )
        else:
            label_key = detect_result_local.get("label", "unknown")
            label_display = CONFIG_TO_LABEL.get(
                label_key,
                label_key.replace("_", " ").title(),
            )
            view_display = _view_display(detect_result_local.get("view"))
            confidence = float(detect_result_local.get("confidence", 0.0))
            feedback_container.success(
                f"Detected exercise: {label_display} – {view_display}"
                f" ({confidence:.0%} confidence)."
            )
            if not detect_result_local.get("accepted"):
                feedback_container.info(
                    "Click Continue to accept this detection or choose an exercise manually."
                )

    if video_path and auto_clicked:
        with st.spinner("Detecting exercise…"):
            try:
                label_key, detected_view, confidence = detect_exercise(str(video_path))
            except Exception as exc:  # pragma: no cover - UI feedback
                state.detect_result = {
                    "error": str(exc),
                    "accepted": False,
                    "token": token,
                }
            else:
                detected_view_key = (detected_view or "unknown").lower()
                state.detect_result = {
                    "label": label_key,
                    "view": detected_view_key,
                    "confidence": float(confidence),
                    "accepted": False,
                    "token": token,
                }
                st.session_state[VIEW_WIDGET_KEY] = KEY_TO_VIEW_LABEL.get(
                    detected_view_key,
                    VIEW_LABELS[0],
                )
            st.session_state[EXERCISE_WIDGET_KEY] = EXERCISE_EMPTY_LABEL
            state.exercise = DEFAULT_EXERCISE_LABEL
        safe_rerun()

    nav_container = st.container()
    with nav_container:
        back_col, continue_col = st.columns(2, gap="large")
        with back_col:
            if st.button(
                "Back",
                key="detect_back",
                use_container_width=True,
                disabled=not is_active,
            ):
                state.detect_result = None
                go_to(Step.UPLOAD)
                safe_rerun()
        with continue_col:
            if st.button(
                "Continue",
                key="detect_continue",
                use_container_width=True,
                disabled=not is_active,
            ):
                current_exercise_label = state.exercise or DEFAULT_EXERCISE_LABEL
                if current_exercise_label == DEFAULT_EXERCISE_LABEL:
                    detect_result = state.detect_result
                    if (
                        detect_result
                        and not detect_result.get("error")
                        and detect_result.get("label")
                        and detect_result.get("token") == token
                    ):
                        mapped_label = CONFIG_TO_LABEL.get(
                            detect_result.get("label", ""),
                            current_exercise_label,
                        )
                        state.exercise_pending_update = mapped_label
                        if selected_view_key != "unknown":
                            detect_result["view"] = selected_view_key
                        detect_result["accepted"] = True
                        go_to(Step.CONFIGURE)
                        safe_rerun()
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
                                    detected_view_key = (detected_view or "unknown").lower()
                                    state.detect_result = {
                                        "label": label_key,
                                        "view": detected_view_key,
                                        "confidence": float(confidence),
                                        "accepted": False,
                                        "token": token,
                                    }
                                    st.session_state[VIEW_WIDGET_KEY] = KEY_TO_VIEW_LABEL.get(
                                        detected_view_key,
                                        VIEW_LABELS[0],
                                    )
                            safe_rerun()
                else:
                    state.detect_result = {
                        "label": EXERCISE_TO_CONFIG.get(
                            current_exercise_label,
                            FALLBACK_MANUAL_CONFIG,
                        ),
                        "view": selected_view_key,
                        "confidence": 1.0,
                        "accepted": True,
                        "token": token,
                        "source": "manual",
                    }
                    go_to(Step.CONFIGURE)
                    safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)
