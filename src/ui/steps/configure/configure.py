from __future__ import annotations

import streamlit as st

from src.ui.state import CONFIG_DEFAULTS, Step, get_state, go_to
from ..utils import step_container


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

        if disabled:
            current_step = state.step
            if current_step == Step.RUNNING:
                st.info("The configuration is displayed for reference while the analysis runs.")
            elif current_step == Step.RESULTS:
                st.info("Configuration values used for the analysis are shown below.")
            else:
                st.info("Configuration is read-only at this stage.")

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

        current_values = {
            "low": float(low),
            "high": float(high),
            "primary_angle": primary_angle,
            "min_prominence": float(min_prominence),
            "min_distance_sec": float(min_distance_sec),
            "debug_video": bool(debug_video),
            "use_crop": True,
        }
        if not disabled:
            state.configure_values = current_values

        if show_actions and not disabled:
            run_active = bool(state.analysis_future and not state.analysis_future.done())
            col_back, col_forward = st.columns(2)
            with col_back:
                if st.button(
                    "Back",
                    key="configure_back",
                    disabled=run_active,
                    width='stretch',
                ):
                    go_to(Step.DETECT)
            with col_forward:
                if st.button(
                    "Continue",
                    key="configure_continue",
                    disabled=run_active,
                    width='stretch',
                ):
                    state.configure_values = current_values
                    go_to(Step.RUNNING)
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
