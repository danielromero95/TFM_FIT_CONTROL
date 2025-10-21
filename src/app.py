# src/app.py
"""Streamlit front-end for the analysis pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import streamlit as st

st.set_page_config(layout="wide", page_title="Exercise Performance Analyzer")

# --- Ensure Windows loads codec DLLs from the active conda env first -----------
if sys.platform.startswith("win"):
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        dll_dir = Path(conda_prefix) / "Library" / "bin"
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

# --- Tame noisy logs from TF/MediaPipe -----------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# Ensure the project root is available on the import path when Streamlit executes the app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.state import Step, get_state, go_to, reset_state
from src.ui.steps.configure import _configure_step
from src.ui.steps.detect import _detect_step
from src.ui.steps.results import _results_panel, _results_summary
from src.ui.steps.running import _running_step
from src.ui.steps.upload import _upload_step
from src.ui.theme import inject_css_js


def _reset_app() -> None:
    reset_state()
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def main() -> None:
    inject_css_js()
    state = get_state()

    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        _upload_step()
        state = get_state()
        if (
            state.step in (Step.DETECT, Step.CONFIGURE, Step.RUNNING, Step.RESULTS)
            and state.video_path
        ):
            _detect_step()

    results_actions: Dict[str, bool] = {"adjust": False, "reset": False}

    with col_mid:
        state = get_state()
        step = state.step
        if step in (Step.CONFIGURE, Step.RUNNING, Step.RESULTS):
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
        state = get_state()
        if state.step == Step.RESULTS:
            results_actions = _results_panel()
        else:
            st.empty()

    if results_actions.get("adjust"):
        state = get_state()
        go_to(Step.CONFIGURE)
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    if results_actions.get("reset"):
        _reset_app()


if __name__ == "__main__":
    main()
