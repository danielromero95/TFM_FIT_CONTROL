"""Carga y registro de estilos CSS reutilizados en la interfaz."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Directorios base relativos al paquete ``src.ui`` para localizar los estilos.
_BASE_UI_DIR = Path(__file__).resolve().parent.parent
_THEME_DIR = _BASE_UI_DIR / "theme"
_STEPS_DIR = _BASE_UI_DIR / "steps"

# Listado explícito de hojas de estilo para mantener el orden de inyección.
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


def _load_css(path: str) -> str:
    """Lee el contenido de un archivo CSS.

    El CSS forma parte de la experiencia de desarrollo y se modifica con
    frecuencia mientras se ajusta el diseño.  Usar ``st.cache_data`` impide
    que los cambios se reflejen en caliente, porque Streamlit conservaría en
    memoria la versión antigua incluso tras un rerun del script.  Leer el
    archivo directamente en cada render garantiza que cualquier edición se
    aplique de inmediato.
    """

    css_path = Path(path)
    return css_path.read_text(encoding="utf-8")


def inject_css() -> None:
    """Inyecta todas las hojas de estilo disponibles dentro de Streamlit."""

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
