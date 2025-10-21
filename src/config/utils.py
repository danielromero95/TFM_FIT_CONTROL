"""
Utilities for loading configuration from defaults or files (e.g., YAML).
"""
from pathlib import Path
from .models import Config, _update_dataclass

try:  # Optional dependency – only needed when loading from YAML files.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML is optional at runtime
    yaml = None  # type: ignore


def load_default() -> Config:
    """Return the default configuration used by the Streamlit application."""
    return Config()


def from_yaml(path: str | Path) -> Config:
    """Load a configuration from a YAML file and merge it with defaults."""
    if yaml is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("PyYAML is not available. Install it to load YAML files.")

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = load_default()
    _update_dataclass(cfg, data)
    return cfg
