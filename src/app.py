# src/app.py
"""Streamlit front-end for the analysis pipeline."""

from __future__ import annotations

import os
import sys
import threading
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


def _inject_css_from_file() -> None:
    """Reads the CSS file and injects it into the app."""
    css_path = Path(__file__).parent / "ui" / "styles.css"
    try:
        css = css_path.read_text()
    except FileNotFoundError:
        st.error("Custom CSS file not found. Please check the path.")
        return

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(
        """
    <style>
      /* --- Top header with inline title --- */
      header[data-testid="stHeader"] {
        background: #0f172a;
        border-bottom: 1px solid #1f2937;
        height: 48px;
        padding-right: 140px;
        position: relative;
        display: flex;
        align-items: center;
      }
      header[data-testid="stHeader"]::before {
        content: "Exercise Performance Analyzer" !important;
        position: absolute;
        left: 16px;
        top: 50%;
        transform: translateY(-50%);
        color: #e5e7eb !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        letter-spacing: .2px !important;
        pointer-events: none;
        max-width: calc(100% - 180px);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
        z-index: 2;
      }

      .form-label {
        color: #e5e7eb;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: .25rem;
      }

      .form-label--inline {
        margin-bottom: 0;
        display: flex;
        align-items: center;
        min-height: 38px;
      }

      /* Target the Streamlit button that follows our marker */
      .btn-danger + div .stButton > button,
      .btn-danger + div button {
        border-radius: 12px !important;
        min-height: 40px;
        min-width: 140px;
        background: transparent !important;
        color: #ef4444 !important;
        border: 1px solid rgba(239, 68, 68, .6) !important;
        transition: background .15s ease, border-color .15s ease, transform .15s ease, box-shadow .15s ease;
      }
      .btn-danger + div .stButton > button:hover,
      .btn-danger + div button:hover {
        background: rgba(239,68,68,.10) !important;
        border-color: rgba(239,68,68,.9) !important;
        transform: translateY(-1px);
      }

      .btn-success + div .stButton > button,
      .btn-success + div button {
        border-radius: 12px !important;
        min-height: 40px;
        min-width: 140px;
        background: linear-gradient(135deg, rgba(34, 197, 94, .95), rgba(16, 185, 129, .95)) !important;
        color: #ecfdf5 !important;
        border: 1px solid rgba(34, 197, 94, .8) !important;
        box-shadow: 0 12px 24px rgba(16, 185, 129, .35);
        transition: transform .15s ease, box-shadow .15s ease;
      }
      .btn-success + div .stButton > button:hover,
      .btn-success + div button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 28px rgba(16, 185, 129, .45);
      }

      /* Disabled state */
      .btn-danger + div .stButton > button[disabled],
      .btn-success + div .stButton > button[disabled] {
        opacity: .55 !important;
        transform: none !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
      }

      .chip {display:inline-block;padding:.2rem .5rem;border-radius:9999px;font-size:.8rem;
             margin-right:.25rem}
      .chip.ok {background:#065f46;color:#ecfdf5;}      /* green */
      .chip.warn {background:#7c2d12;color:#ffedd5;}    /* amber/brown */
      .chip.info {background:#1e293b;color:#e2e8f0;}    /* slate */

      .spacer-sm { height: .5rem; }

      /* Keep title above content and safe on very narrow viewports */
      header[data-testid="stHeader"] { z-index: 1000; }
      @media (max-width: 520px) {
        header[data-testid="stHeader"] .app-toolbar-title { font-size: 16px; }
      }
      @media (max-width: 420px) {
        header[data-testid="stHeader"] .app-toolbar-title { display: none; }
      }

      /* Pull content closer to the toolbar */
      section[data-testid="stMain"],
      main {
        padding-top: 0 !important;
      }
      section[data-testid="stMain"] > div:first-child,
      main > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
      }
      main .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
      }
      main .block-container > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
      }
      main [data-testid="stToolbar"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
      }
      main [data-testid="stToolbar"] + div {
        margin-top: 0 !important;
        padding-top: 0 !important;
      }
      main [data-testid="stVerticalBlock"] {
        padding-top: 0 !important;
      }
      main [data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0 !important;
      }
      main [data-testid="stHorizontalBlock"] {
        margin-top: 0 !important;
        align-items: flex-start !important;
      }
      main [data-testid="column"] {
        padding-top: 0 !important;
      }
      main [data-testid="column"] > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
      }

      /* Ensure the results column aligns with step 4 content */
      .results-panel {
        margin-top: 0 !important;
        display: flex;
        flex-direction: column;
        gap: .75rem;
      }
      .results-panel h3:first-child {
        margin-top: 0 !important;
      }
      .results-panel .stDataFrame {
        margin-bottom: .25rem;
      }
      .results-panel video {
        border-radius: 12px;
      }
      .results-panel [data-testid="stHorizontalBlock"] {
        gap: .5rem !important;
      }
      .results-panel [data-testid="column"] .stButton > button {
        width: 100%;
      }
    </style>
    <script>
      (() => {
        const TITLE = "Exercise Performance Analyzer";
        const ENHANCER_KEY = '__appEnhancer';
        const doc = (window.parent && window.parent.document) ? window.parent.document : document;
        if (!doc) {
          return;
        }

        const existingEnhancer = doc[ENHANCER_KEY];
        if (existingEnhancer && typeof existingEnhancer.init === 'function') {
          existingEnhancer.init();
          return;
        }

        const headerObserver = new MutationObserver(() => {
          ensureToolbarTitle(false);
        });

        const mainObserver = new MutationObserver(() => {
          scheduleEnhancements();
        });

        let enhancementFrame = null;

        function ensureToolbarTitle(reattach = true) {
          const header = doc.querySelector('header[data-testid="stHeader"]');
          if (!header) {
            return false;
          }
          let title = header.querySelector('.app-toolbar-title');
          if (!title) {
            title = doc.createElement('div');
            title.className = 'app-toolbar-title';
            header.insertBefore(title, header.firstChild);
          }
          if (title.textContent !== TITLE) {
            title.textContent = TITLE;
          }
          if (reattach) {
            attachHeaderObserver();
          }
          return true;
        }

        function ensureToolbarTitleWithRetry() {
          if (ensureToolbarTitle()) {
            return;
          }
          const retryInterval = setInterval(() => {
            if (ensureToolbarTitle()) {
              clearInterval(retryInterval);
            }
          }, 150);
          setTimeout(() => clearInterval(retryInterval), 5000);
        }

        function attachHeaderObserver() {
          const header = doc.querySelector('header[data-testid="stHeader"]');
          if (!header) {
            headerObserver.disconnect();
            return false;
          }
          headerObserver.disconnect();
          headerObserver.observe(header, { childList: true });
          return true;
        }

        function attachMainObserver() {
          const main = doc.querySelector('main');
          if (!main) {
            mainObserver.disconnect();
            return false;
          }
          mainObserver.disconnect();
          mainObserver.observe(main, { childList: true, subtree: true });
          return true;
        }

        function scheduleEnhancements() {
          if (enhancementFrame !== null) {
            return;
          }
          const requestFrame = window.requestAnimationFrame || function (cb) { return setTimeout(cb, 16); };
          enhancementFrame = requestFrame(() => {
            enhancementFrame = null;
            applyEnhancements();
          });
        }

        function applyEnhancements() {
          ensureToolbarTitle(false);
          applyNavButtonClasses();
          tagDetectStep();
        }

        function applyNavButtonClasses() {
          const wrappers = doc.querySelectorAll('div[data-testid="stButton"]');
          wrappers.forEach((wrapper) => {
            wrapper.classList.remove('app-button-wrapper');
          });

          const buttons = doc.querySelectorAll('div[data-testid="stButton"] button');
          buttons.forEach((button) => {
            const text = (button.textContent || '').trim().toLowerCase();
            button.classList.remove('app-button', 'app-button--back', 'app-button--continue', 'app-button--disabled');
            if (!text) {
              return;
            }

            if (text === 'back') {
              button.classList.add('app-button', 'app-button--back');
              const wrapper = button.closest('div[data-testid="stButton"]');
              if (wrapper) {
                wrapper.classList.add('app-button-wrapper');
              }
            }

            if (text === 'continue') {
              button.classList.add('app-button', 'app-button--continue');
              const wrapper = button.closest('div[data-testid="stButton"]');
              if (wrapper) {
                wrapper.classList.add('app-button-wrapper');
              }
            }

            if (button.disabled) {
              button.classList.add('app-button--disabled');
            }
          });
        }

        function tagDetectStep() {
          doc.querySelectorAll('.app-step-detect').forEach((node) => {
            node.classList.remove('app-step-detect');
          });
          const headings = Array.from(doc.querySelectorAll('h3'));
          for (const heading of headings) {
            const label = (heading.textContent || '').trim().toLowerCase();
            if (!label.startsWith('2. detect the exercise')) {
              continue;
            }
            const block = heading.closest('[data-testid="stVerticalBlock"]');
            if (block) {
              block.classList.add('app-step-detect');
            }
          }
        }

        function init() {
          ensureToolbarTitleWithRetry();
          attachHeaderObserver();
          attachMainObserver();
          applyEnhancements();
        }

        if (doc.readyState === 'loading') {
          doc.addEventListener('DOMContentLoaded', init, { once: true });
        } else {
          init();
        }

        doc[ENHANCER_KEY] = {
          init,
          ensure: ensureToolbarTitleWithRetry,
        };
        doc.__appToolbarTitleInit = doc[ENHANCER_KEY];
      })();
    </script>
    """,
        unsafe_allow_html=True,
    )


def _reset_app() -> None:
    reset_state()
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


 
 


 


def main() -> None:
    if threading.current_thread() is threading.main_thread():
        _inject_css_from_file()
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
