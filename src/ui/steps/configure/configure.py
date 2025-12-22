from __future__ import annotations

import streamlit as st

from src.ui.metrics_catalog import human_metric_name, metric_base_description
from src.ui.state import CONFIG_DEFAULTS, DEFAULT_EXERCISE_LABEL, Step, get_state, go_to
from src.ui.steps.detect import EXERCISE_TO_CONFIG
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
        st.markdown(
            """
            <style>
            div[data-testid="stColumn"] label {
                display: flex;
                justify-content: center;
                align-items: center;
                width: 100%;
            }

            div[data-testid="stColumn"] label p {
                width: 100%;
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        state = get_state()
        stored_cfg = state.configure_values
        if stored_cfg is None:
            cfg_values = CONFIG_DEFAULTS.copy()
        else:
            cfg_values = {**CONFIG_DEFAULTS, **dict(stored_cfg)}
        cfg_values["use_crop"] = True

        col1, col2, col3 = st.columns(3)
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
        with col3:
            thresholds_enable = st.checkbox(
                "Enable",
                value=bool(
                    cfg_values.get(
                        "thresholds_enable", CONFIG_DEFAULTS["thresholds_enable"]
                    )
                ),
                disabled=disabled,
                key="cfg_thresholds_enable",
                help=(
                    "Apply both upper and lower thresholds to filter repetitions."
                    " Turn off to ignore threshold filtering while counting."
                ),
            )

        # Primary angle is auto-selected downstream; show as read-only
        chosen_primary = None
        if state.step == Step.RESULTS and state.report is not None and getattr(
            state.report, "stats", None
        ):
            chosen_primary = getattr(state.report.stats, "primary_angle", None)

        exercise_label = state.exercise or DEFAULT_EXERCISE_LABEL
        ex_key = EXERCISE_TO_CONFIG.get(
            exercise_label, EXERCISE_TO_CONFIG.get(DEFAULT_EXERCISE_LABEL, "squat")
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

        candidate_labels = [human_metric_name(metric) for metric in candidates]
        if len(candidate_labels) > 1:
            readable_candidates = ", ".join(candidate_labels[:-1]) + f" and {candidate_labels[-1]}"
        elif candidate_labels:
            readable_candidates = candidate_labels[0]
        else:
            readable_candidates = "the available primary angles"

        st.caption(
            "We’ll automatically pick the most reliable primary angle for rep counting "
            f"from {readable_candidates}, prioritizing visibility, stability, and overall signal quality."
        )

        if chosen_primary:
            chosen_label = human_metric_name(chosen_primary)
            st.caption(
                f"Primary angle used for rep counting: **{chosen_label}** (auto-selected for the cleanest signal)."
            )
            primary_desc = metric_base_description(chosen_primary, exercise_label)
            if primary_desc:
                st.caption(primary_desc)

        debug_video = st.checkbox(
            "Generate debug video",
            value=bool(cfg_values.get("debug_video", CONFIG_DEFAULTS["debug_video"])),
            disabled=disabled,
            key="cfg_debug_video",
        )

        col_fps, col_complexity, col_debug = st.columns(3)
        with col_fps:
            target_fps = st.number_input(
                "Target FPS",
                min_value=1,
                max_value=60,
                value=int(cfg_values.get("target_fps", CONFIG_DEFAULTS["target_fps"])),
                disabled=disabled,
                key="cfg_target_fps",
                help=(
                    "Frames per second used during processing.\n\n"
                    "Lower values reduce processing time while higher values capture more motion detail but may take longer."
                ),
            )
        with col_complexity:
            complexity_options = [0, 1, 2]
            stored_complexity = int(
                cfg_values.get("model_complexity", CONFIG_DEFAULTS["model_complexity"])
            )
            selected_index = complexity_options.index(stored_complexity) if stored_complexity in complexity_options else 0
            model_complexity = st.selectbox(
                "Pose model complexity",
                options=complexity_options,
                index=selected_index,
                disabled=disabled,
                key="cfg_model_complexity",
                help=(
                    "Higher complexity improves pose quality at the cost of speed.\n\n"
                    "0 = fastest with lower accuracy.\n\n"
                    "1 = balanced.\n\n"
                    "2 = most accurate but heavier."
                ),
            )
        with col_debug:
            debug_mode = st.checkbox(
                "Debug mode",
                value=bool(cfg_values.get("debug_mode", CONFIG_DEFAULTS["debug_mode"])),
                disabled=disabled,
                key="cfg_debug_mode",
                help=(
                    "Toggle verbose diagnostics during processing to inspect detailed logs and checks."
                ),
            )

        current_values = {
            "low": float(low),
            "high": float(high),
            "thresholds_enable": bool(thresholds_enable),
            "primary_angle": "auto",
            "debug_video": bool(debug_video),
            "debug_mode": bool(debug_mode),
            "use_crop": True,
            "target_fps": float(target_fps),
            "model_complexity": int(model_complexity),
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
