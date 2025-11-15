"""Generación del script JavaScript que mejora la experiencia de la app."""

from __future__ import annotations

import json

import streamlit as st

_APP_ENHANCER_TEMPLATE = """
<script>
  (() => {
    const TITLE = __APP_TITLE__;
    const ENHANCER_KEY = '__appEnhancer';
    const doc = (() => {
      try {
        if (window.parent && window.parent !== window && window.parent.document) {
          return window.parent.document;
        }
      } catch (error) { /* Ignore cross-origin access errors */ }
      return document;
    })();
    if (!doc) { return; }

    const existingEnhancer = doc[ENHANCER_KEY];
    if (existingEnhancer && typeof existingEnhancer.init === 'function') {
      existingEnhancer.init();
      return;
    }

    const headerObservers = new Map();
    const mainObservers = new Map();
    let enhancementFrame = null;

    function ensureToolbarTitle(reattach = true) {
      const header = getHeader();
      if (!header) { return false; }
      let title = header.querySelector('.app-toolbar-title');
      if (!title) {
        title = doc.createElement('div');
        title.className = 'app-toolbar-title';
        header.insertBefore(title, header.firstChild);
      }
      if (reattach) { attachMainObservers(); }
      return applied;
    }

    function ensureToolbarTitleWithRetry() {
      if (ensureToolbarTitle()) { return; }
      const retryInterval = setInterval(() => {
        if (ensureToolbarTitle()) { clearInterval(retryInterval); }
      }, 150);
      setTimeout(() => clearInterval(retryInterval), 5000);
    }

    function attachHeaderObserver() {
      const header = getHeader();
      if (!header) { headerObserver.disconnect(); return false; }
      headerObserver.disconnect();
      headerObserver.observe(header, { childList: true, subtree: true });
      return true;
    }

    function attachMainObservers() {
      for (const doc of candidateDocs) {
        const main = doc.querySelector('main');
        let observer = mainObservers.get(doc);
        if (!main) {
          if (observer) { observer.disconnect(); }
          continue;
        }
        if (!observer) {
          observer = new MutationObserver(() => { scheduleEnhancements(); });
          mainObservers.set(doc, observer);
        }
        observer.disconnect();
        observer.observe(main, { childList: true, subtree: true });
      }
    }

    function scheduleEnhancements() {
      if (enhancementFrame !== null) { return; }
      const requestFrame = window.requestAnimationFrame || function (cb) { return setTimeout(cb, 16); };
      enhancementFrame = requestFrame(() => { enhancementFrame = null; applyEnhancements(); });
    }

    function applyEnhancements() { ensureToolbarTitle(false); }

    function getHeader() {
      return doc.querySelector('header[data-testid="stHeader"], div[data-testid="stHeader"]');
    }

    function init() {
      ensureToolbarTitleWithRetry();
      ensureToolbarTitle();
      applyEnhancements();
    }

    let started = false;
    function start() {
      if (started) { return; }
      started = true;
      init();
    }

    for (const doc of Array.from(new Set([stateDoc, scriptDoc]))) {
      if (!doc) { continue; }
      if (doc.readyState === 'loading') {
        doc.addEventListener('DOMContentLoaded', start, { once: true });
      } else {
        start();
      }
    }

    stateDoc[ENHANCER_KEY] = { init, ensure: ensureToolbarTitleWithRetry };
    stateDoc.__appToolbarTitleInit = stateDoc[ENHANCER_KEY];
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
