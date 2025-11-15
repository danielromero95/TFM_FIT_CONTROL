"""Generación del script JavaScript que mejora la experiencia de la app."""

from __future__ import annotations

import json

import streamlit.components.v1 as components

_APP_ENHANCER_TEMPLATE = """
<script>
  (() => {
    const TITLE = __APP_TITLE__;
    const ENHANCER_KEY = '__appEnhancer';
    const HEADER_SELECTORS = [
      'header[data-testid="stHeader"]',
      'div[data-testid="stHeader"]',
      'header[data-testid="stAppHeader"]',
      'div[data-testid="stAppHeader"]',
    ];
    const TOOLBAR_SELECTORS = [
      '[data-testid="stToolbar"]',
      '[data-testid="stHeaderToolbar"]',
      '[data-testid="stToolbarActions"]',
    ];
    const scriptDoc = document;
    const candidateDocs = (() => {
      const docs = [];
      try {
        if (window.parent && window.parent !== window && window.parent.document) {
          docs.push(window.parent.document);
        }
      } catch (error) { /* Ignore cross-origin access errors */ }
      docs.push(scriptDoc);
      return Array.from(new Set(docs.filter(Boolean)));
    })();
    if (!candidateDocs.length) { return; }

    const stateDoc = candidateDocs[0];
    const existingEnhancer = stateDoc[ENHANCER_KEY];
    if (existingEnhancer && typeof existingEnhancer.init === 'function') {
      existingEnhancer.init();
      return;
    }

    const headerObservers = new Map();
    const toolbarObservers = new Map();
    const mainObservers = new Map();
    const containerObservers = new Map();
    let enhancementFrame = null;

    function ensureToolbarTitle(reattach = true) {
      let applied = false;
      for (const doc of candidateDocs) {
        const headers = getHeaders(doc);
        if (!headers.length) {
          detachHeaderObserver(doc);
          detachToolbarObserver(doc);
          continue;
        }
        for (const header of headers) {
          const ownerDoc = header.ownerDocument || doc;
          const target = getToolbarContainer(header);
          if (!target) { continue; }
          let title = target.querySelector('.app-toolbar-title');
          if (!title) {
            title = ownerDoc.createElement('div');
            title.className = 'app-toolbar-title';
            if (target.firstChild) {
              target.insertBefore(title, target.firstChild);
            } else {
              target.appendChild(title);
            }
          }
          if (title.textContent !== TITLE) {
            title.textContent = TITLE;
          }
          title.setAttribute('title', TITLE);
          applied = true;
          if (reattach) {
            attachHeaderObserver(doc, header);
            attachToolbarObserver(doc, target);
            attachContainerObserver(header);
          }
        }
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

    function attachHeaderObserver(doc, header) {
      if (!header) { return detachHeaderObserver(doc); }
      let observer = headerObservers.get(doc);
      if (!observer) {
        observer = new MutationObserver(() => { ensureToolbarTitle(false); });
        headerObservers.set(doc, observer);
      }
      observer.disconnect();
      observer.observe(header, { childList: true, subtree: true });
      return true;
    }

    function detachHeaderObserver(doc) {
      const observer = headerObservers.get(doc);
      if (observer) { observer.disconnect(); }
      return false;
    }

    function attachToolbarObserver(doc, toolbar) {
      if (!toolbar) { return detachToolbarObserver(doc); }
      let observer = toolbarObservers.get(doc);
      if (!observer) {
        observer = new MutationObserver(() => { ensureToolbarTitle(false); });
        toolbarObservers.set(doc, observer);
      }
      observer.disconnect();
      observer.observe(toolbar, { childList: true, subtree: true });
      return true;
    }

    function detachToolbarObserver(doc) {
      const observer = toolbarObservers.get(doc);
      if (observer) { observer.disconnect(); }
      return false;
    }

    function attachContainerObserver(header) {
      const container = getHeaderContainer(header);
      if (!container) { return false; }
      let observer = containerObservers.get(container);
      if (!observer) {
        observer = new MutationObserver(() => { scheduleEnhancements(); });
        containerObservers.set(container, observer);
      }
      observer.disconnect();
      observer.observe(container, { childList: true });
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

    function getHeaders(doc) {
      const roots = collectSearchRoots(doc);
      const found = new Set();
      for (const root of roots) {
        if (!root || typeof root.querySelectorAll !== 'function') { continue; }
        for (const selector of HEADER_SELECTORS) {
          const matches = root.querySelectorAll(selector);
          if (matches && matches.length) {
            for (const header of matches) {
              if (header) { found.add(header); }
            }
          }
        }
      }
      return Array.from(found);
    }

    function collectSearchRoots(doc) {
      if (!doc) { return []; }
      const roots = [];
      const seen = new Set();
      const queue = [];
      queue.push(doc);
      while (queue.length) {
        const root = queue.shift();
        if (!root || seen.has(root)) { continue; }
        seen.add(root);
        roots.push(root);
        if (typeof root.querySelectorAll !== 'function') { continue; }
        const elements = root.querySelectorAll('*');
        for (const element of elements) {
          if (element && element.shadowRoot) {
            queue.push(element.shadowRoot);
          }
        }
      }
      return roots;
    }

    function getToolbarContainer(header) {
      if (!header) { return null; }
      for (const selector of TOOLBAR_SELECTORS) {
        const node = header.querySelector(selector);
        if (node) { return node; }
      }
      return header;
    }

    function getHeaderContainer(header) {
      if (!header || typeof header.closest !== 'function') { return null; }
      const selectors = [
        '[data-testid="stAppViewContainer"]',
        '[data-testid="stApp"]',
        '#root',
        'body',
      ];
      for (const selector of selectors) {
        const node = header.closest(selector);
        if (node) { return node; }
      }
      return header.parentElement || null;
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
    components.html(script, height=0, width=0)
