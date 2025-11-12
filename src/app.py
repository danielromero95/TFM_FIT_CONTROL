# src/app.py
"""Interfaz Streamlit que orquesta el pipeline de análisis."""

from __future__ import annotations

import os
import sys
import threading
from enum import Enum
from pathlib import Path

import streamlit as st

# Garantizar que la raíz del proyecto esté en ``sys.path`` cuando Streamlit ejecute la app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import APP_NAME
from src.config.settings import configure_environment
from src.ui.assets import inject_css, inject_js
from src.ui.state import AppState, Step, get_state, go_to, reset_state, safe_rerun
from src.ui.steps.configure import _configure_step
from src.ui.steps.detect import _detect_step
from src.ui.steps.results import _results_panel, _results_summary
from src.ui.steps.running import _running_step
from src.ui.steps.upload import _upload_step

configure_environment()
st.set_page_config(layout="wide", page_title=APP_NAME)

ENABLE_JS_ENHANCEMENTS = os.getenv("ENABLE_JS_ENHANCEMENTS", "1") == "1"

class AppAction(Enum):
    """Acciones posibles emitidas por el panel de resultados."""

    NONE = "none"
    ADJUST = "adjust"
    RESET = "reset"


def should_render_detect(state: AppState) -> bool:
    """Indica si debe mostrarse la columna del paso de detección."""

    return bool(
        state.step in (Step.DETECT, Step.CONFIGURE, Step.RUNNING, Step.RESULTS)
        and state.video_path
    )


def middle_mode(state: AppState) -> str:
    """Determina qué vista debe renderizarse en la columna central."""

    if state.step == Step.CONFIGURE:
        return "configure"
    if state.step == Step.RUNNING:
        return "running"
    if state.step == Step.RESULTS:
        return "results"
    return "empty"


def _results_action_from_payload(payload: dict[str, bool] | None) -> AppAction:
    """Convierte la respuesta del panel de resultados en una acción tipada."""

    if not payload:
        return AppAction.NONE
    if payload.get("reset"):
        return AppAction.RESET
    if payload.get("adjust"):
        return AppAction.ADJUST
    return AppAction.NONE


def handle_results_action(action: AppAction) -> None:
    """Ejecuta los efectos secundarios asociados a cada acción."""

    if action is AppAction.ADJUST:
        go_to(Step.CONFIGURE)
        safe_rerun()
    elif action is AppAction.RESET:
        _reset_app()


def _reset_app() -> None:
    """Restablece el estado global y fuerza un rerun limpio de la interfaz."""

    reset_state()
    safe_rerun()


def main() -> None:
    """Punto de entrada principal de Streamlit que dibuja las tres columnas."""

    if threading.current_thread() is threading.main_thread():
        inject_css()
        inject_js(title=APP_NAME, enable=ENABLE_JS_ENHANCEMENTS)

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
