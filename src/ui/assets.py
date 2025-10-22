"""Utilities for injecting UI assets into the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html


def ensure_toolbar_title(title: str) -> None:
    """Ensure the Streamlit toolbar shows the provided title."""

    html(
        f"""
    <script>
      (function() {{
        const mount = () => {{
          const header = parent.document.querySelector('header[data-testid="stHeader"]');
          if (!header) return;
          let el = header.querySelector('.app-toolbar-title');
          if (!el) {{
            el = parent.document.createElement('span');
            el.className = 'app-toolbar-title';
            header.appendChild(el);
          }}
          el.textContent = {title!r};
        }};
        mount();
        const obs = new MutationObserver(mount);
        obs.observe(parent.document.body, {{ childList: true, subtree: true }});
      })();
    </script>
    """,
        height=0,
    )


_ASSETS_DIR = Path(__file__).resolve().parent
_CSS_FILE = _ASSETS_DIR / "styles.css"


@st.cache_data(show_spinner=False)
def _load_css(path: str) -> str:
    """Load the CSS file content from disk and cache it."""
    css_path = Path(path)
    return css_path.read_text(encoding="utf-8")


_INLINE_CSS = """
<style>
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

  /* Styled navigation buttons */
  .app-button {
    border-radius: 12px !important;
    min-height: 44px;
    width: 100%;
    font-weight: 600;
    font-size: .95rem;
    letter-spacing: .01em;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 1px solid transparent;
    transition: background .15s ease, border-color .15s ease, transform .15s ease, box-shadow .15s ease;
  }

  .app-button--back {
    background: transparent !important;
    color: #ef4444 !important;
    border-color: rgba(239, 68, 68, .6) !important;
  }
  .app-button--back:hover {
    background: rgba(239, 68, 68, .10) !important;
    border-color: rgba(239, 68, 68, .9) !important;
    transform: translateY(-1px);
  }

  .app-button--continue {
    background: linear-gradient(135deg, rgba(34, 197, 94, .95), rgba(16, 185, 129, .95)) !important;
    color: #ecfdf5 !important;
    border-color: rgba(34, 197, 94, .8) !important;
    box-shadow: 0 12px 24px rgba(16, 185, 129, .35);
  }
  .app-button--continue:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 28px rgba(16, 185, 129, .45);
  }

  .app-button--disabled {
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

  .app-step-detect .form-label--inline {
    min-height: auto;
  }

  .app-step-detect [data-testid="stHorizontalBlock"] {
    margin-top: .25rem !important;
    align-items: center !important;
  }

  .app-step-detect [data-testid="column"] > div {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }

  .app-step-detect .app-nav-buttons {
    margin-top: .75rem;
  }

  .app-step-detect .app-nav-buttons [data-testid="stHorizontalBlock"] {
    margin-top: 0 !important;
    gap: .75rem !important;
    align-items: stretch !important;
  }

  .app-step-detect .app-nav-buttons [data-testid="column"] > div {
    width: 100%;
  }

  .app-step-detect .app-nav-buttons button {
    border-radius: 12px !important;
    min-height: 44px;
    width: 100%;
    font-weight: 600;
    font-size: .95rem;
    letter-spacing: .01em;
    border: 1px solid transparent;
    transition: background .15s ease, border-color .15s ease, transform .15s ease, box-shadow .15s ease;
  }

  .app-step-detect .app-nav-buttons [data-testid="column"]:first-child button {
    background: transparent !important;
    color: #ef4444 !important;
    border-color: rgba(239, 68, 68, .6) !important;
  }

  .app-step-detect .app-nav-buttons [data-testid="column"]:first-child button:hover:not(:disabled) {
    background: rgba(239, 68, 68, .10) !important;
    border-color: rgba(239, 68, 68, .9) !important;
    transform: translateY(-1px);
  }

  .app-step-detect .app-nav-buttons [data-testid="column"]:last-child button {
    background: linear-gradient(135deg, rgba(34, 197, 94, .95), rgba(16, 185, 129, .95)) !important;
    color: #ecfdf5 !important;
    border-color: rgba(34, 197, 94, .8) !important;
    box-shadow: 0 12px 24px rgba(16, 185, 129, .35);
  }

  .app-step-detect .app-nav-buttons [data-testid="column"]:last-child button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 16px 28px rgba(16, 185, 129, .45);
  }

  .app-step-detect .app-nav-buttons button:disabled {
    opacity: .55 !important;
    transform: none !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
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
"""


_APP_ENHANCER = """
<script>
  (() => {
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

    const mainObserver = new MutationObserver(() => {
      scheduleEnhancements();
    });

    let enhancementFrame = null;

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
      // The toolbar title is now managed via ensure_toolbar_title() in app.py.
      // Add future DOM tweaks for other components here.
    }

    function init() {
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
      ensure: applyEnhancements,
    };
  })();
</script>
"""


def inject_css() -> None:
    """Inject the cached CSS content and inline styles into the Streamlit app."""
    try:
        css_content = _load_css(str(_CSS_FILE))
    except FileNotFoundError:
        st.error(
            "Custom CSS file not found. Please check the path.",
        )
    else:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    st.markdown(_INLINE_CSS, unsafe_allow_html=True)


def inject_js(enable: bool = True) -> None:
    """Inject the app enhancer script if enabled."""
    if not enable:
        return
    st.markdown(_APP_ENHANCER, unsafe_allow_html=True)
