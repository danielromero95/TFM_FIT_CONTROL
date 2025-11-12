"""Utilidades para publicar recursos estáticos de la interfaz."""

from .app_enhancer import inject_js
from .css import inject_css

__all__ = ["inject_js", "inject_css"]
