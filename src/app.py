# src/app.py
"""Streamlit front-end for the analysis pipeline."""

from __future__ import annotations

import os
import sys
import threading
from enum import Enum
from pathlib import Path

import streamlit as st


# Ensure Streamlit's configuration is applied before any other interaction.
st.set_page_config(
    page_title="FIT CONTROL v2.3",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# importa después de set_page_config cualquier módulo que haga llamadas a st.*
from src.ui.assets import ensure_toolbar_title


def inject_styles() -> None:
    """Inject custom CSS rules for Streamlit widgets."""

    css_path = Path("src/ui/styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# Apply custom styles immediately after configuring the page.
inject_styles()

# Ensure the project root is available on the import path when Streamlit executes the app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import configure_environment
from src.ui.assets import inject_css, inject_js
from src.ui.state import AppState, Step, get_state, go_to, reset_state, safe_rerun
from src.ui.steps.configure import _configure_step
from src.ui.steps.detect import _detect_step
from src.ui.steps.results import _results_panel, _results_summary
from src.ui.steps.running import _running_step
from src.ui.steps.upload import _upload_step

configure_environment()

ENABLE_JS_ENHANCEMENTS = os.getenv("ENABLE_JS_ENHANCEMENTS", "1") == "1"

class AppAction(Enum):
    """Typed representation of the actions emitted by the results panel."""

    NONE = "none"
    ADJUST = "adjust"
    RESET = "reset"


def should_render_detect(state: AppState) -> bool:
    """Return True when the detect column should be displayed."""

    return bool(
        state.step in (Step.DETECT, Step.CONFIGURE, Step.RUNNING, Step.RESULTS)
        and state.video_path
    )


def middle_mode(state: AppState) -> str:
    """Determine the rendering mode for the middle column."""

    if state.step == Step.CONFIGURE:
        return "configure"
    if state.step == Step.RUNNING:
        return "running"
    if state.step == Step.RESULTS:
        return "results"
    return "empty"


def _results_action_from_payload(payload: dict[str, bool] | None) -> AppAction:
    """Translate the results panel payload into a typed action."""

    if not payload:
        return AppAction.NONE
    if payload.get("reset"):
        return AppAction.RESET
    if payload.get("adjust"):
        return AppAction.ADJUST
    return AppAction.NONE


def handle_results_action(action: AppAction) -> None:
    """Execute the side-effects associated with a results action."""

    if action is AppAction.ADJUST:
        go_to(Step.CONFIGURE)
        safe_rerun()
    elif action is AppAction.RESET:
        _reset_app()


def _reset_app() -> None:
    reset_state()
    safe_rerun()


def main() -> None:
    # Inserta el título en la barra superior (visible)
    ensure_toolbar_title("FIT CONTROL v2.3")

    if threading.current_thread() is threading.main_thread():
        inject_css()
        inject_js(enable=ENABLE_JS_ENHANCEMENTS)

    results_action = AppAction.NONE
    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        _upload_step()
        left_state = get_state()
        if should_render_detect(left_state):
            _detect_step()

    with col_mid:
        mid_state = get_state()
        mode = middle_mode(mid_state)
        if mode in {"configure", "running", "results"}:
            _configure_step(disabled=mode != "configure", show_actions=mode == "configure")
            if mode == "running":
                _running_step()
            elif mode == "results":
                _results_summary()
        else:
            st.empty()

    with col_right:
        right_state = get_state()
        if right_state.step == Step.RESULTS:
            results_action = _results_action_from_payload(_results_panel())
        else:
            st.empty()

    handle_results_action(results_action)

if __name__ == "__main__":
    main()
