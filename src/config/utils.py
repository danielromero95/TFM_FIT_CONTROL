"""Herramientas para cargar configuración por defecto o desde archivos YAML."""
from pathlib import Path
from .models import Config, _update_dataclass

try:  # Dependencia opcional: solo necesaria al leer archivos YAML.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML es opcional en tiempo de ejecución
    yaml = None  # type: ignore


def load_default() -> Config:
    """Devuelve la configuración predeterminada usada por la aplicación Streamlit."""
    return Config()


def from_yaml(path: str | Path) -> Config:
    """Carga una configuración desde YAML y la fusiona con los valores por defecto."""
    if yaml is None:  # pragma: no cover - protección para dependencia opcional
        raise RuntimeError("PyYAML no está disponible. Instálalo para leer archivos YAML.")

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = load_default()
    _update_dataclass(cfg, data)
    return cfg
