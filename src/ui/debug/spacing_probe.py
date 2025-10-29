"""Utilities to inspect spacing between steps during development."""

from __future__ import annotations

import os
from textwrap import dedent
from typing import Iterable

import streamlit as st


_DEBUG_QUERY_VALUE = "spacing"


def _query_params() -> dict[str, list[str]]:
    """Return the query parameters regardless of the Streamlit API version."""
    try:  # Streamlit >= 1.27
        params = st.query_params  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - fallback for older versions
        params = st.experimental_get_query_params()
    if isinstance(params, dict):
        # The runtime returns MutableMapping[str, list[str]] – normalize values.
        return {str(key): [str(v) for v in value] for key, value in params.items()}
    # Safety net – Streamlit shouldn't hit this branch, but normalize anyway.
    return {}


def _has_debug_flag(values: Iterable[str]) -> bool:
    return any(value.lower() == _DEBUG_QUERY_VALUE for value in values)


def spacing_debug_enabled() -> bool:
    """Return True when the spacing probe should be injected."""
    if os.getenv("SPACING_DEBUG") == "1":
        return True
    params = _query_params()
    debug_values = params.get("debug", [])
    if _has_debug_flag(debug_values):
        return True
    # Support repeated query params such as ?debug=foo&debug=spacing
    for key, values in params.items():
        if key.lower() == "debug" and _has_debug_flag(values):
            return True
    return False


def inject_spacing_probe() -> None:
    """Inject a DOM probe that reports spacing metrics for each step."""
    if not spacing_debug_enabled():
        return

    script = dedent(
        """
        <script>
        (() => {
          const doc = (window.parent && window.parent.document) ? window.parent.document : document;
          if (!doc) { return; }

          function ensureReportContainer() {
            let pre = doc.getElementById('spacing-report');
            if (!pre) {
              pre = doc.createElement('pre');
              pre.id = 'spacing-report';
              pre.style.position = 'fixed';
              pre.style.bottom = '12px';
              pre.style.right = '12px';
              pre.style.maxWidth = '480px';
              pre.style.maxHeight = '60vh';
              pre.style.overflow = 'auto';
              pre.style.zIndex = '9999';
              pre.style.background = 'rgba(15, 23, 42, 0.92)';
              pre.style.color = '#f8fafc';
              pre.style.fontSize = '12px';
              pre.style.lineHeight = '1.4';
              pre.style.padding = '12px';
              pre.style.border = '1px solid rgba(148, 163, 184, 0.4)';
              pre.style.borderRadius = '8px';
              pre.textContent = 'Collecting spacing metrics...';
              doc.body.appendChild(pre);
            }
            return pre;
          }

          function findClosest(element, selector) {
            let current = element;
            while (current) {
              if (current.matches && current.matches(selector)) {
                return current;
              }
              current = current.parentElement;
            }
            return null;
          }

          function describeColumn(element) {
            if (!element) { return 'n/a'; }
            const siblings = Array.from(element.parentElement ? element.parentElement.children : []);
            const index = siblings.indexOf(element);
            return index >= 0 ? `column[index=${index}]` : 'column';
          }

          function collectStyleOrder() {
            const styleTags = Array.from(doc.querySelectorAll('style'));
            const lastFive = styleTags.slice(-5);
            return lastFive.map((style, idx) => {
              const raw = style.textContent || '';
              const snippet = raw.replace(/\s+/g, ' ').trim().slice(0, 120);
              return `#${styleTags.length - lastFive.length + idx + 1}: ${snippet}`;
            });
          }

          function gatherMetrics() {
            const pre = ensureReportContainer();
            const view = doc.defaultView || window;
            const steps = Array.from(doc.querySelectorAll('.step'));
            if (!steps.length) {
              pre.textContent = 'No .step elements found.';
              return;
            }
            const lines = [];
            steps.forEach((step, idx) => {
              const metrics = view.getComputedStyle(step);
              const block = findClosest(step, '[data-testid="stVerticalBlock"]');
              const column = findClosest(step, '[data-testid="column"]');
              let blockMetrics = 'n/a';
              if (block) {
                const blockStyles = view.getComputedStyle(block);
                blockMetrics = `gap=${blockStyles.gap}; row-gap=${blockStyles.rowGap}; column-gap=${blockStyles.columnGap}`;
              }
              lines.push(
                `Step ${idx + 1}: margin-top=${metrics.marginTop}; margin-bottom=${metrics.marginBottom}; block[${blockMetrics}]; parent=${describeColumn(column)}`
              );
            });
            lines.push('\nLast <style> tags:');
            const styles = collectStyleOrder();
            if (styles.length) {
              styles.forEach((snippet) => lines.push(`  ${snippet}`));
            } else {
              lines.push('  (No <style> tags found)');
            }
            pre.textContent = lines.join('\n');
          }

          let pendingFrame = null;

          const observer = new MutationObserver(() => {
            if (pendingFrame) { return; }
            const requestFrame = window.requestAnimationFrame || function (cb) { return setTimeout(cb, 16); };
            pendingFrame = requestFrame(() => {
              pendingFrame = null;
              gatherMetrics();
            });
          });

          function init() {
            gatherMetrics();
            const root = doc.querySelector('main') || doc.body;
            if (root) {
              observer.observe(root, { childList: true, subtree: true });
            }
          }

          if (doc.readyState === 'loading') {
            doc.addEventListener('DOMContentLoaded', init, { once: true });
          } else {
            init();
          }
        })();
        </script>
        """
    )

    st.markdown(script, unsafe_allow_html=True)
