"""Utilidades para cargar configuraciones por defecto o desde archivos YAML.

El objetivo es aclarar cómo se inicializan los parámetros cuando se ejecuta la
aplicación y cómo se combinan con configuraciones externas."""
from pathlib import Path
from .models import Config, _update_dataclass

try:  # Optional dependency – only needed when loading from YAML files.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML is optional at runtime
    yaml = None  # type: ignore


def load_default() -> Config:
    """Obtener la configuración por defecto empleada por la aplicación Streamlit."""
    return Config()


def from_yaml(path: str | Path) -> Config:
    """Cargar una configuración desde un YAML y mezclarla con los valores base."""
    if yaml is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("PyYAML is not available. Install it to load YAML files.")

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = load_default()
    _update_dataclass(cfg, data)
    return cfg
