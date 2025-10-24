from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from src.core.types import ExerciseType, ViewType, as_view
from src.exercise_detection.exercise_detector import detect_exercise
from src.ui.state import (
    DEFAULT_EXERCISE_LABEL,
    Step,
    get_state,
    go_to,
    safe_rerun,
)
from src.ui.video import (
    detect_video_orientation,
    generate_portrait_preview,
    render_uniform_video,
)


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

VIEW_CHOICES: List[Tuple[str, str]] = [
    ("Auto-Detect", "auto"),
    ("Front view", ViewType.FRONT.value),
    ("Lateral view", ViewType.SIDE.value),
]
VIEW_LABELS: List[str] = [lbl for (lbl, _) in VIEW_CHOICES]
VIEW_TO_KEY: Dict[str, str] = {lbl: key for (lbl, key) in VIEW_CHOICES}
VIEW_FROM_KEY: Dict[str, str] = {key: lbl for (lbl, key) in VIEW_CHOICES}
VIEW_WIDGET_KEY = "detect_view_select_value"


def _resolve_orientation_pill(video_path: str | None) -> str:
    if not video_path:
        return "<span class=\"detect-pill detect-pill--unknown\">No video</span>"

    orientation, (width, height) = detect_video_orientation(video_path)
    details = f"{width}×{height}" if width and height else ""
    if orientation == "horizontal":
        label = "Horizontal"
        pill_class = "detect-pill--ok"
    elif orientation == "vertical":
        label = "Vertical"
        pill_class = "detect-pill--info"
    elif orientation == "square":
        label = "Square"
        pill_class = "detect-pill--info"
    else:
        label = "Unknown"
        pill_class = "detect-pill--unknown"

    suffix = f" · {details}" if details else ""
    return f"<span class=\"detect-pill {pill_class}\">{label}{suffix}</span>"


def _view_label_from_detection(detect_result: Dict[str, object] | None) -> str:
    if not detect_result:
        return VIEW_LABELS[0]
    view_key = str(detect_result.get("view", "")).strip().lower()
    view_type = as_view(view_key)
    if view_type in (ViewType.FRONT, ViewType.SIDE):
        return VIEW_FROM_KEY.get(view_type.value, VIEW_LABELS[0])
    return VIEW_LABELS[0]


def _prepare_manual_detection(
    *,
    state,
    current_exercise: str,
    view_value: str,
    token,
) -> None:
    manual_result = dict(state.detect_result or {})
    if not manual_result.get("label"):
        manual_result["label"] = EXERCISE_TO_CONFIG.get(current_exercise, "unknown")
    manual_result["view"] = view_value
    manual_result["confidence"] = float(manual_result.get("confidence") or 0.0)
    manual_result["accepted"] = True
    manual_result["token"] = token
    manual_result["source"] = "manual"
    state.detect_result = manual_result


def _reset_manual_detection(state) -> None:
    if state.detect_result and state.detect_result.get("source") == "manual":
        state.detect_result = None


def _render_detection_feedback(info_container, detect_result) -> None:
    if not detect_result:
        return

    if detect_result.get("error"):
        info_container.error(
            "Automatic exercise detection failed: " f"{detect_result.get('error')}"
        )
        info_container.info(
            "You can adjust the selection manually or try detecting again."
        )
        return

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
    if pending_exercise in VALID_EXERCISE_LABELS:
        state.exercise = pending_exercise

    current_exercise = state.exercise or DEFAULT_EXERCISE_LABEL
    if current_exercise not in VALID_EXERCISE_LABELS:
        current_exercise = DEFAULT_EXERCISE_LABEL
        state.exercise = current_exercise

    if (
        current_exercise != DEFAULT_EXERCISE_LABEL
        and detect_result is not None
        and detect_result.get("source") != "manual"
    ):
        state.detect_result = None
        detect_result = None

    widget_value = st.session_state.get(EXERCISE_WIDGET_KEY)
    if widget_value not in VALID_EXERCISE_LABELS:
        widget_value = current_exercise
    if widget_value != current_exercise:
        current_exercise = widget_value
        state.exercise = current_exercise
    st.session_state[EXERCISE_WIDGET_KEY] = current_exercise

    current_view_label = _view_label_from_detection(detect_result)
    view_widget_value = st.session_state.get(VIEW_WIDGET_KEY)
    if view_widget_value not in VIEW_LABELS:
        view_widget_value = current_view_label
    if view_widget_value != current_view_label:
        current_view_label = view_widget_value
    st.session_state[VIEW_WIDGET_KEY] = current_view_label

    st.markdown('<div class="detect-layout">', unsafe_allow_html=True)
    if video_path:
        video_col, controls_col = st.columns([3, 2], gap="large")
    else:
        controls_col = st.container()
        video_col = None

    if video_col and video_path:
        with video_col:
            st.markdown('<div class="detect-video">', unsafe_allow_html=True)
            render_uniform_video(
                str(video_path),
                key="detect_video",
                bottom_margin=0.0,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with controls_col:
        st.markdown('<div class="detect-control-card">', unsafe_allow_html=True)
        pill_html = _resolve_orientation_pill(video_path)
        st.markdown(
            f'<div class="detect-orientation">{pill_html}</div>',
            unsafe_allow_html=True,
        )

        auto_disabled = not (is_active and video_path)
        auto_clicked = st.button(
            "Auto-Detect",
            key="detect_auto",
            use_container_width=True,
            disabled=auto_disabled,
            type="primary",
        )

        select_index = EXERCISE_LABELS.index(current_exercise)
        selected_exercise = st.selectbox(
            "Select exercise",
            options=EXERCISE_LABELS,
            index=select_index,
            key=EXERCISE_WIDGET_KEY,
            disabled=not is_active,
        )
        if selected_exercise != current_exercise:
            state.exercise = selected_exercise
            current_exercise = selected_exercise

        view_index = VIEW_LABELS.index(current_view_label)
        selected_view = st.selectbox(
            "Select view",
            options=VIEW_LABELS,
            index=view_index,
            key=VIEW_WIDGET_KEY,
            disabled=not is_active,
        )

        if selected_view != current_view_label:
            selected_key = VIEW_TO_KEY.get(selected_view, "auto")
            if selected_key == "auto":
                _reset_manual_detection(state)
                detect_result = state.detect_result
            else:
                _prepare_manual_detection(
                    state=state,
                    current_exercise=current_exercise,
                    view_value=selected_key,
                    token=token,
                )
                detect_result = state.detect_result
            st.session_state[VIEW_WIDGET_KEY] = selected_view
            current_view_label = selected_view

        st.markdown("</div>", unsafe_allow_html=True)

        if auto_clicked and video_path:
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
                        "source": "auto",
                    }
                else:
                    state.detect_result = {
                        "label": label_key,
                        "view": detected_view,
                        "confidence": float(confidence),
                        "accepted": False,
                        "token": token,
                        "source": "auto",
                    }
            safe_rerun()

        if video_path:
            preview = generate_portrait_preview(str(video_path), max_height=640)
        else:
            preview = None
        if preview is not None:
            st.markdown('<div class="detect-preview-card">', unsafe_allow_html=True)
            st.caption("Portrait preview (9:16)")
            st.image(preview, use_column_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    detect_result = state.detect_result
    info_container = st.container()
    _render_detection_feedback(info_container, detect_result)

    actions_placeholder = st.empty()
    if is_active:
        with actions_placeholder.container():
            back_col, continue_col = st.columns(2)
            with back_col:
                back_clicked = st.button(
                    "Back",
                    key="detect_back",
                    use_container_width=True,
                )
            with continue_col:
                continue_clicked = st.button(
                    "Continue",
                    key="detect_continue",
                    use_container_width=True,
                )

        if back_clicked:
            state.detect_result = None
            go_to(Step.UPLOAD)
            safe_rerun()

        if continue_clicked:
            current_exercise = state.exercise or DEFAULT_EXERCISE_LABEL
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
                    safe_rerun()
                elif (
                    detect_result
                    and not detect_result.get("error")
                    and detect_result.get("label")
                    and detect_result.get("token") == token
                ):
                    detect_result["accepted"] = True
                    go_to(Step.CONFIGURE)
                    safe_rerun()
                else:
                    info_container.warning(
                        "Run auto-detect before continuing or select an exercise manually."
                    )
            else:
                if selected_view != VIEW_LABELS[0]:
                    selected_key = VIEW_TO_KEY.get(selected_view, "auto")
                    if selected_key != "auto":
                        _prepare_manual_detection(
                            state=state,
                            current_exercise=current_exercise,
                            view_value=selected_key,
                            token=token,
                        )
                else:
                    _reset_manual_detection(state)
                state.detect_result = state.detect_result
                go_to(Step.CONFIGURE)
                safe_rerun()
    else:
        actions_placeholder.empty()

    st.markdown("</div>", unsafe_allow_html=True)
