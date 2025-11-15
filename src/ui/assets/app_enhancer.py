"""Expose runtime UI tokens consumed by the CSS theme."""

from __future__ import annotations

import json

import streamlit as st


def _toolbar_token_css(title: str) -> str:
    """Genera la definición CSS que expone el nombre de la app al tema."""

    # ``json.dumps`` garantiza un string escapado compatible con CSS.
    return f":root {{ --app-toolbar-title: {json.dumps(title)}; }}"


def inject_js(title: str, enable: bool = True) -> None:
    """Publica el nombre de la app como variable CSS para la barra."""

    if not enable or not title:
        return

    st.markdown(f"<style>{_toolbar_token_css(title)}</style>", unsafe_allow_html=True)
