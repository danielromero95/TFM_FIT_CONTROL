"""Utilities for injecting UI assets into the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


_ASSETS_DIR = Path(__file__).resolve().parent
_THEME_DIR = _ASSETS_DIR / "theme"
_STEPS_DIR = _ASSETS_DIR / "steps"

_CSS_FILES = [
    _THEME_DIR / "variables.css",
    _THEME_DIR / "layout-and-header.css",
    _THEME_DIR / "ui-components.css",
    _STEPS_DIR / "configure" / "configure.css",
    _STEPS_DIR / "detect" / "detect.css",
    _STEPS_DIR / "results" / "results.css",
    _STEPS_DIR / "upload" / "upload.css",
    _STEPS_DIR / "running" / "running.css",
]


@st.cache_data(show_spinner=False)
def _load_css(path: str) -> str:
    """Load the CSS file content from disk and cache it."""
    css_path = Path(path)
    return css_path.read_text(encoding="utf-8")


_APP_ENHANCER_TEMPLATE = """
<script>
  (() => {
    const TITLE = __APP_TITLE__;
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
"""


def inject_css() -> None:
    """Inject the cached CSS content and inline styles into the Streamlit app."""
    css_fragments: list[str] = []
    missing_files: list[Path] = []

    for css_path in _CSS_FILES:
        try:
            css_fragments.append(_load_css(str(css_path)))
        except FileNotFoundError:
            missing_files.append(css_path)

    if missing_files:
        missing = ", ".join(path.name for path in missing_files)
        st.warning(f"Custom CSS file(s) not found: {missing}.")

    if css_fragments:
        combined_css = "\n\n".join(fragment for fragment in css_fragments if fragment.strip())
        if combined_css:
            st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)


def inject_js(title: str, enable: bool = True) -> None:
    """Inject the app enhancer script if enabled."""
    if not enable:
        return
    # Safely JSON-encode the title to embed in JS
    title_js = json.dumps(title)
    script = _APP_ENHANCER_TEMPLATE.replace("__APP_TITLE__", title_js)
    st.markdown(script, unsafe_allow_html=True)
