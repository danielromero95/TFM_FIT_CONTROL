"""Pequeños helpers de maquetación para los pasos del asistente."""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


@contextmanager
def step_container(name: str):
    """Crea un ``div`` con clases BEM para agrupar el contenido del paso."""

    st.markdown(f"<div class='step step--{name}'>", unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)
