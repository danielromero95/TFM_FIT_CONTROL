from __future__ import annotations

from pathlib import Path
import json
import threading

import streamlit as st

APP_TITLE = "Exercise Performance Analyzer"

_SESSION_FLAG = "__theme_injected_once"
_CSS_MISSING_FLAG = "__theme_css_missing_warned"

_JS_TEMPLATE = """
<script>
  (() => {
    const TITLE = %%TITLE%%;
    const ENHANCER_KEY = '__appEnhancer';
    const doc = (window.parent && window.parent.document) ? window.parent.document : document;
    if (!doc) return;

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
      if (!header) return false;
      let title = header.querySelector('.app-toolbar-title');
      if (!title) {
        title = doc.createElement('div');
        title.className = 'app-toolbar-title';
        header.insertBefore(title, header.firstChild);
      }
      if (title.textContent !== TITLE) {
        title.textContent = TITLE;
      }
      if (reattach) attachHeaderObserver();
      return true;
    }

    function ensureToolbarTitleWithRetry() {
      if (ensureToolbarTitle()) return;
      const retryInterval = setInterval(() => {
        if (ensureToolbarTitle()) clearInterval(retryInterval);
      }, 150);
      setTimeout(() => clearInterval(retryInterval), 5000);
    }

    function attachHeaderObserver() {
      const header = doc.querySelector('header[data-testid="stHeader"]');
      if (!header) { headerObserver.disconnect(); return false; }
      headerObserver.disconnect();
      headerObserver.observe(header, { childList: true });
      return true;
    }

    function attachMainObserver() {
      const main = doc.querySelector('main');
      if (!main) { mainObserver.disconnect(); return false; }
      mainObserver.disconnect();
      mainObserver.observe(main, { childList: true, subtree: true });
      return true;
    }

    function scheduleEnhancements() {
      if (enhancementFrame !== null) return;
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
      wrappers.forEach((w) => w.classList.remove('app-button-wrapper'));
      const buttons = doc.querySelectorAll('div[data-testid="stButton"] button');
      buttons.forEach((button) => {
        const text = (button.textContent || '').trim().toLowerCase();
        button.classList.remove('app-button','app-button--back','app-button--continue','app-button--disabled');
        if (!text) return;
        if (text === 'back') {
          button.classList.add('app-button','app-button--back');
          const wrapper = button.closest('div[data-testid="stButton"]');
          if (wrapper) wrapper.classList.add('app-button-wrapper');
        }
        if (text === 'continue') {
          button.classList.add('app-button','app-button--continue');
          const wrapper = button.closest('div[data-testid="stButton"]');
          if (wrapper) wrapper.classList.add('app-button-wrapper');
        }
        if (button.disabled) button.classList.add('app-button--disabled');
      });
    }

    function tagDetectStep() {
      doc.querySelectorAll('.app-step-detect').forEach((n) => n.classList.remove('app-step-detect'));
      const headings = Array.from(doc.querySelectorAll('h3'));
      for (const heading of headings) {
        const label = (heading.textContent || '').trim().toLowerCase();
        if (!label.startsWith('2. detect the exercise')) continue;
        const block = heading.closest('[data-testid="stVerticalBlock"]');
        if (block) block.classList.add('app-step-detect');
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

    doc[ENHANCER_KEY] = { init, ensure: ensureToolbarTitleWithRetry };
    doc.__appToolbarTitleInit = doc[ENHANCER_KEY];
  })();
</script>
"""


def _read_styles() -> str | None:
    """Read styles.css from src/ui/styles.css."""
    css_path = Path(__file__).parent / "styles.css"
    try:
        return css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _format_js(title: str) -> str:
    return _JS_TEMPLATE.replace("%%TITLE%%", json.dumps(title))


def inject_css_js(title: str = APP_TITLE) -> None:
    """Inject CSS and JS into Streamlit in an idempotent fashion."""
    if st.session_state.get(_SESSION_FLAG):
        return

    if threading.current_thread() is not threading.main_thread():
        return

    css = _read_styles()
    if css is None:
        if not st.session_state.get(_CSS_MISSING_FLAG):
            st.error("Custom CSS file not found at ui/styles.css.")
            st.session_state[_CSS_MISSING_FLAG] = True
    else:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.markdown(_format_js(title), unsafe_allow_html=True)

    st.session_state[_SESSION_FLAG] = True
