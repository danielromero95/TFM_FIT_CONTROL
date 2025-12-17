from __future__ import annotations

import streamlit as st

from src.ui.state import CONFIG_DEFAULTS, DEFAULT_EXERCISE_LABEL, Step, get_state, go_to
from src.ui.steps.detect import EXERCISE_TO_CONFIG
from src.ui.metrics_sync.descriptions import describe_primary_angle
from ..utils import step_container


def _primary_candidates_for(ex_key: str) -> list[str]:
    return {
        "squat": ["left_knee", "right_knee"],
        "bench_press": ["left_elbow", "right_elbow"],
        "deadlift": ["left_hip", "right_hip"],
    }.get(ex_key, [])


def _configure_step(*, disabled: bool = False, show_actions: bool = True) -> None:
    with step_container("configure"):
        st.markdown("### 3. Configure the analysis")
        state = get_state()
        stored_cfg = state.configure_values
        if stored_cfg is None:
            cfg_values = CONFIG_DEFAULTS.copy()
        else:
            cfg_values = {**CONFIG_DEFAULTS, **dict(stored_cfg)}
        cfg_values.pop("target_fps", None)
        cfg_values["use_crop"] = True

        # Primary angle is auto-selected downstream; show as read-only
        chosen_primary = None
        if state.step == Step.RESULTS and state.report is not None and getattr(
            state.report, "stats", None
        ):
            chosen_primary = getattr(state.report.stats, "primary_angle", None)

        exercise_label = state.exercise or DEFAULT_EXERCISE_LABEL
        detected_label = None
        if state.detect_result is not None:
            detected_label = getattr(state.detect_result.label, "value", None) or getattr(
                state.detect_result, "label", None
            )
        ex_key = EXERCISE_TO_CONFIG.get(
            exercise_label,
            detected_label or EXERCISE_TO_CONFIG.get(DEFAULT_EXERCISE_LABEL, "squat"),
        )
        candidates = _primary_candidates_for(ex_key)
        primary_display_value = (
            str(chosen_primary) if chosen_primary else (" / ".join(candidates) if candidates else "auto")
        )

        st.text_input(
            "Primary angle (auto)",
            value=primary_display_value,
            disabled=True,
            key="cfg_primary_angle",
        )

        if chosen_primary:
            desc = describe_primary_angle(chosen_primary)
            if desc:
                st.caption(f"Primary angle used for rep counting: **{chosen_primary}** — {desc}")
            else:
                st.caption(f"Primary angle used for rep counting: **{chosen_primary}**")

        if ex_key == "squat":
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
        else:
            st.markdown("#### Deadlift counting thresholds")
            col1, col2 = st.columns(2)
            with col1:
                min_prominence = st.number_input(
                    "Minimum hip excursion (°)",
                    min_value=1.0,
                    max_value=90.0,
                    value=float(cfg_values.get("min_prominence", CONFIG_DEFAULTS["min_prominence"])),
                    step=0.5,
                    disabled=disabled,
                    key="cfg_min_prominence",
                    help="Minimum difference between the bottom of the hinge and the recovery phase to count a rep.",
                )
                min_excursion = st.number_input(
                    "Minimum usable ROM (°)",
                    min_value=1.0,
                    max_value=90.0,
                    value=float(cfg_values.get("min_excursion_deg", CONFIG_DEFAULTS["min_excursion_deg"])),
                    step=0.5,
                    disabled=disabled,
                    key="cfg_min_excursion",
                    help="Frames with less range than this are ignored to avoid false positives.",
                )
            with col2:
                min_distance = st.number_input(
                    "Min distance between reps (s)",
                    min_value=0.2,
                    max_value=3.0,
                    value=float(cfg_values.get("min_distance_sec", CONFIG_DEFAULTS["min_distance_sec"])),
                    step=0.05,
                    disabled=disabled,
                    key="cfg_min_distance",
                    help="Cooldown between reps to avoid counting tiny bounces as separate lifts.",
                )
                refractory = st.number_input(
                    "Refractory period (s)",
                    min_value=0.2,
                    max_value=3.0,
                    value=float(cfg_values.get("refractory_sec", CONFIG_DEFAULTS["refractory_sec"])),
                    step=0.05,
                    disabled=disabled,
                    key="cfg_refractory",
                    help="Time after a detected rep during which new valleys are ignored.",
                )
            low = float(cfg_values.get("low", CONFIG_DEFAULTS["low"]))
            high = float(cfg_values.get("high", CONFIG_DEFAULTS["high"]))

        debug_video = st.checkbox(
            "Generate debug video",
            value=bool(cfg_values.get("debug_video", CONFIG_DEFAULTS["debug_video"])),
            disabled=disabled,
            key="cfg_debug_video",
        )

        current_values = {
            "low": float(low),
            "high": float(high),
            "min_prominence": float(cfg_values.get("min_prominence", CONFIG_DEFAULTS["min_prominence"]))
            if ex_key == "squat"
            else float(min_prominence),
            "min_distance_sec": float(cfg_values.get("min_distance_sec", CONFIG_DEFAULTS["min_distance_sec"]))
            if ex_key == "squat"
            else float(min_distance),
            "refractory_sec": float(cfg_values.get("refractory_sec", CONFIG_DEFAULTS["refractory_sec"]))
            if ex_key == "squat"
            else float(refractory),
            "min_excursion_deg": float(cfg_values.get("min_excursion_deg", CONFIG_DEFAULTS["min_excursion_deg"]))
            if ex_key == "squat"
            else float(min_excursion),
            "primary_angle": "auto",
            "debug_video": bool(debug_video),
            "use_crop": True,
        }
        if not disabled:
            state.configure_values = current_values

        if show_actions and not disabled:
            run_active = bool(state.analysis_future and not state.analysis_future.done())
            col_back, col_forward = st.columns(2)
            with col_back:
                st.markdown('<div class="btn--back">', unsafe_allow_html=True)
                back_clicked = st.button(
                    "Back",
                    key="configure_back",
                    disabled=run_active,
                    width='stretch',
                )
                st.markdown('</div>', unsafe_allow_html=True)
                if back_clicked:
                    go_to(Step.DETECT)
            with col_forward:
                st.markdown('<div class="btn--continue">', unsafe_allow_html=True)
                forward_clicked = st.button(
                    "Continue",
                    key="configure_continue",
                    disabled=run_active,
                    width='stretch',
                )
                st.markdown('</div>', unsafe_allow_html=True)
                if forward_clicked:
                    state.configure_values = current_values
                    go_to(Step.RUNNING)
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
