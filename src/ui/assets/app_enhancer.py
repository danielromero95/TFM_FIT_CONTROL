"""Generación del script JavaScript que mejora la experiencia de la app."""

from __future__ import annotations

import json

import streamlit as st

_APP_ENHANCER_TEMPLATE = """
<script>
  (() => {
    const TITLE = __APP_TITLE__;
    const ENHANCER_KEY = '__appEnhancer';
    const doc = (window.parent && window.parent.document) ? window.parent.document : document;
    if (!doc) { return; }

    const existingEnhancer = doc[ENHANCER_KEY];
    if (existingEnhancer && typeof existingEnhancer.init === 'function') {
      existingEnhancer.init();
      return;
    }

    const headerObserver = new MutationObserver(() => { ensureToolbarTitle(false); });
    const mainObserver = new MutationObserver(() => { scheduleEnhancements(); });

    let enhancementFrame = null;

    function ensureToolbarTitle(reattach = true) {
      const header = doc.querySelector('header[data-testid="stHeader"]');
      if (!header) { return false; }

      const toolbar = header.querySelector('[data-testid="stToolbar"]');
      const host = toolbar || header;

      let title = host.querySelector('.app-toolbar-title');
      if (!title) {
        title = doc.createElement('div');
        title.className = 'app-toolbar-title';
        host.insertBefore(title, host.firstChild);
      }

      if (title.textContent !== TITLE) { title.textContent = TITLE; }
      if (reattach) { attachHeaderObserver(); }
      return true;
    }

    function ensureToolbarTitleWithRetry() {
      if (ensureToolbarTitle()) { return; }
      const retryInterval = setInterval(() => {
        if (ensureToolbarTitle()) { clearInterval(retryInterval); }
      }, 150);
      setTimeout(() => clearInterval(retryInterval), 5000);
    }

    function attachHeaderObserver() {
      const header = doc.querySelector('header[data-testid="stHeader"]');
      if (!header) { headerObserver.disconnect(); return false; }
      headerObserver.disconnect();
      headerObserver.observe(header, { childList: true, subtree: true });
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
      if (enhancementFrame !== null) { return; }
      const requestFrame = window.requestAnimationFrame || function (cb) { return setTimeout(cb, 16); };
      enhancementFrame = requestFrame(() => { enhancementFrame = null; applyEnhancements(); });
    }

    function applyEnhancements() { ensureToolbarTitle(false); }

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


def inject_js(title: str, enable: bool = True) -> None:
    """Inserta el script de mejora visual solo cuando está habilitado."""

    if not enable:
        return
    title_js = json.dumps(title)  # Codificación segura en JSON
    script = _APP_ENHANCER_TEMPLATE.replace("__APP_TITLE__", title_js)
    st.markdown(script, unsafe_allow_html=True)
