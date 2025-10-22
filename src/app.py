# src/app.py
"""Streamlit front-end for the analysis pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
import threading

# Ensure the project root is available on the import path when Streamlit executes the app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.core.runtime import configure_environment
from src.ui.assets import inject_css, inject_js, is_js_feature_enabled
from src.ui.state import AppState, Step, get_state, go_to, reset_state, trigger_rerun
from src.ui.steps.configure import _configure_step
from src.ui.steps.detect import _detect_step
from src.ui.steps.results import ResultsActions, _results_panel, _results_summary
from src.ui.steps.running import _running_step
from src.ui.steps.upload import _upload_step

st.set_page_config(layout="wide", page_title="Exercise Performance Analyzer")

configure_environment()

def _reset_app() -> None:
    reset_state()
    trigger_rerun()


def _should_render_detect_step(state: AppState) -> bool:
    return (
        state.step in (Step.DETECT, Step.CONFIGURE, Step.RUNNING, Step.RESULTS)
        and bool(state.video_path)
    )


def _should_render_configure_step(step: Step) -> bool:
    return step in (Step.CONFIGURE, Step.RUNNING, Step.RESULTS)


def _should_render_results(step: Step) -> bool:
    return step == Step.RESULTS


def _inject_assets() -> None:
    if threading.current_thread() is not threading.main_thread():
        return

    inject_css()
    inject_js(enable=is_js_feature_enabled())


def main() -> None:
    _inject_assets()

    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        _upload_step()
        left_state = get_state()
        if _should_render_detect_step(left_state):
            _detect_step()

    results_actions = ResultsActions()

    with col_mid:
        mid_state = get_state()
        step = mid_state.step
        if _should_render_configure_step(step):
            disabled = step != Step.CONFIGURE
            show_actions = step == Step.CONFIGURE
            _configure_step(disabled=disabled, show_actions=show_actions)
            if step == Step.RUNNING:
                _running_step()
            elif step == Step.RESULTS:
                _results_summary()
        else:
            st.empty()

    with col_right:
        right_state = get_state()
        if _should_render_results(right_state.step):
            results_actions = _results_panel()
        else:
            st.empty()

    if results_actions.adjust:
        go_to(Step.CONFIGURE)
        trigger_rerun()

    if results_actions.reset:
        _reset_app()

if __name__ == "__main__":
    main()
