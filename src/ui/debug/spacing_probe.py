"""Utilities to inspect spacing between steps during development."""

from __future__ import annotations

import os
from textwrap import dedent

import streamlit as st


_DEBUG_QUERY_VALUE = "spacing"


def spacing_debug_enabled() -> bool:
    """Return True when the spacing probe should be injected."""
    if os.getenv("SPACING_DEBUG") == "1":
        return True

    params: dict[str, list[str]] | dict[str, str] | None
    try:  # Streamlit >= 1.27
        params = st.experimental_get_query_params()
    except Exception:  # pragma: no cover - fallback for newer APIs
        try:
            params = st.query_params  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - Streamlit < 1.27
            return False

    if not isinstance(params, dict):
        return False

    flattened = [
        str(value).lower()
        for values in params.values()
        for value in (values if isinstance(values, list) else [values])
    ]
    if _DEBUG_QUERY_VALUE in flattened:
        return True

    debug_values = params.get("debug", [])
    if not isinstance(debug_values, list):
        debug_values = [debug_values]
    return any(str(value).lower() == _DEBUG_QUERY_VALUE for value in debug_values)


def inject_spacing_probe() -> None:
    """Inject a DOM probe that reports spacing metrics for each step."""
    if not spacing_debug_enabled():
        return

    script = dedent(
        """
        <script>
        (() => {
          const doc = document;
          if (!doc) { return; }

          const BANNER_ID = 'spacing-debug-banner';
          const PANEL_ID = 'spacing-panel';
          const TEXTAREA_ID = 'spacing-report-text';
          const BUTTON_ID = 'spacing-download-btn';

          function ensureBanner() {
            let banner = doc.getElementById(BANNER_ID);
            if (!banner) {
              banner = doc.createElement('div');
              banner.id = BANNER_ID;
              banner.textContent = '[SPACING DEBUG] probe enabled – scroll down to see results';
              Object.assign(banner.style, {
                position: 'fixed',
                top: '0',
                left: '0',
                right: '0',
                padding: '10px 16px',
                background: '#b91c1c',
                color: '#fff',
                fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
                fontSize: '14px',
                fontWeight: '600',
                textAlign: 'center',
                zIndex: '9999',
                boxShadow: '0 2px 6px rgba(0, 0, 0, 0.35)',
                pointerEvents: 'none',
              });
              doc.body.appendChild(banner);
            }
            return banner;
          }

          function ensurePanel() {
            let panel = doc.getElementById(PANEL_ID);
            if (!panel) {
              panel = doc.createElement('div');
              panel.id = PANEL_ID;
              Object.assign(panel.style, {
                margin: '80px auto 32px',
                maxWidth: '960px',
                padding: '16px',
                background: '#111827',
                color: '#f9fafb',
                borderRadius: '12px',
                border: '1px solid rgba(148, 163, 184, 0.35)',
                boxShadow: '0 16px 40px rgba(15, 23, 42, 0.35)',
                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                position: 'relative',
                zIndex: '999',
              });

              const heading = doc.createElement('div');
              heading.textContent = 'Spacing report';
              Object.assign(heading.style, {
                fontSize: '15px',
                fontWeight: '700',
                marginBottom: '8px',
              });

              const textarea = doc.createElement('textarea');
              textarea.id = TEXTAREA_ID;
              textarea.readOnly = true;
              Object.assign(textarea.style, {
                width: '100%',
                height: '280px',
                boxSizing: 'border-box',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid rgba(148, 163, 184, 0.4)',
                background: '#0f172a',
                color: '#f8fafc',
                resize: 'vertical',
                fontSize: '13px',
                lineHeight: '1.5',
              });

              const button = doc.createElement('button');
              button.id = BUTTON_ID;
              button.type = 'button';
              button.textContent = 'Download spacing.txt';
              Object.assign(button.style, {
                marginTop: '12px',
                padding: '10px 16px',
                borderRadius: '8px',
                border: 'none',
                background: '#f59e0b',
                color: '#111827',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
              });

              button.addEventListener('click', () => {
                const textareaEl = doc.getElementById(TEXTAREA_ID);
                const text = textareaEl ? textareaEl.value : '';
                const blob = new Blob([text], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const link = doc.createElement('a');
                link.href = url;
                link.download = 'spacing.txt';
                doc.body.appendChild(link);
                link.click();
                setTimeout(() => {
                  doc.body.removeChild(link);
                  URL.revokeObjectURL(url);
                }, 0);
              });

              panel.appendChild(heading);
              panel.appendChild(textarea);
              panel.appendChild(button);
              doc.body.appendChild(panel);
            }
            return panel;
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

          function describeColumn(columnEl) {
            if (!columnEl) { return 'n/a'; }
            const parent = columnEl.parentElement;
            if (!parent) { return 'column'; }
            const columns = Array.from(parent.children).filter((child) => {
              return child.getAttribute && child.getAttribute('data-testid') === 'column';
            });
            const index = columns.indexOf(columnEl);
            return index >= 0 ? String(index) : 'column';
          }

          function describeBlock(blockEl) {
            if (!blockEl) { return 'n/a'; }
            const view = doc.defaultView || window;
            const styles = view.getComputedStyle(blockEl);
            return `gap=${styles.gap}; row-gap=${styles.rowGap}; column-gap=${styles.columnGap}`;
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

          function buildReport() {
            const view = doc.defaultView || window;
            const steps = Array.from(doc.querySelectorAll('.step'));
            if (!steps.length) {
              return 'No .step elements found.';
            }

            const lines = [];
            steps.forEach((step, idx) => {
              const metrics = view.getComputedStyle(step);
              const verticalBlock = findClosest(step, '[data-testid="stVerticalBlock"]');
              const horizontalBlock = findClosest(step, '[data-testid="stHorizontalBlock"]');
              const column = findClosest(step, '[data-testid="column"]');

              lines.push(`Step ${idx + 1} (${step.className || '(no class)'})`);
              lines.push(`  margin-top: ${metrics.marginTop}`);
              lines.push(`  margin-bottom: ${metrics.marginBottom}`);
              lines.push(`  verticalBlock: ${describeBlock(verticalBlock)}`);
              if (horizontalBlock && horizontalBlock !== verticalBlock) {
                lines.push(`  horizontalBlock: ${describeBlock(horizontalBlock)}`);
              }
              lines.push(`  parentColumn: ${describeColumn(column)}`);
              lines.push('');
            });

            lines.push('Last <style> tags:');
            const styles = collectStyleOrder();
            if (styles.length) {
              styles.forEach((snippet) => lines.push(`  ${snippet}`));
            } else {
              lines.push('  (No <style> tags found)');
            }

            return lines.join('\n');
          }

          let scheduled = false;

          function updateReport() {
            ensureBanner();
            ensurePanel();
            const textarea = doc.getElementById(TEXTAREA_ID);
            if (textarea) {
              textarea.value = buildReport();
            }
            scheduled = false;
          }

          function scheduleUpdate() {
            if (scheduled) { return; }
            scheduled = true;
            const requestFrame = window.requestAnimationFrame || function (cb) { return setTimeout(cb, 16); };
            requestFrame(updateReport);
          }

          function init() {
            ensureBanner();
            ensurePanel();
            updateReport();
            const root = doc.querySelector('main') || doc.body;
            if (root && window.MutationObserver) {
              const observer = new MutationObserver(() => scheduleUpdate());
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
