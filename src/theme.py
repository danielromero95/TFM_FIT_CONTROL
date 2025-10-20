"""Helpers to share theme colours between UI front-ends."""

from pathlib import Path
from typing import Dict, NamedTuple

try:  # pragma: no cover - optional dependency on Python < 3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python versions
    try:  # type: ignore[no-redef]
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - final fallback
        tomllib = None  # type: ignore


DEFAULT_PRIMARY_COLOR = "#22c55e"
DEFAULT_SECONDARY_COLOR = "#ef4444"
DEFAULT_PRIMARY_SHADOW = "rgba(16,185,129,.35)"
DEFAULT_PRIMARY_SHADOW_HOVER = "rgba(16,185,129,.45)"
DEFAULT_SECONDARY_BORDER = "rgba(239,68,68,.6)"
DEFAULT_SECONDARY_BORDER_HOVER = "rgba(239,68,68,.9)"
DEFAULT_SECONDARY_BG_HOVER = "rgba(239,68,68,.10)"


class ThemeColors(NamedTuple):
    primary: str
    secondary: str


def _normalise_hex_color(value: object, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    stripped = value.strip()
    if not stripped:
        return fallback
    if stripped.startswith("#"):
        stripped = stripped[1:]
    if len(stripped) not in (3, 6):
        return fallback
    try:
        int(stripped, 16)
    except ValueError:
        return fallback
    if len(stripped) == 3:
        stripped = "".join(ch * 2 for ch in stripped)
    return f"#{stripped.lower()}"


def rgba_from_hex(hex_color: str, alpha: float, fallback: str) -> str:
    colour = _normalise_hex_color(hex_color, fallback)
    if not colour.startswith("#"):
        return fallback
    hex_value = colour[1:]
    try:
        red = int(hex_value[0:2], 16)
        green = int(hex_value[2:4], 16)
        blue = int(hex_value[4:6], 16)
    except ValueError:
        return fallback
    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    alpha_str = f"{alpha_clamped:.2f}".rstrip("0").rstrip(".")
    return f"rgba({red},{green},{blue},{alpha_str})"


def load_theme_colors(project_root: str | Path) -> ThemeColors:
    primary = DEFAULT_PRIMARY_COLOR
    secondary = DEFAULT_SECONDARY_COLOR
    config_path = Path(project_root) / ".streamlit" / "config.toml"
    if config_path.exists():
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        data: Dict[str, object] | None = None
        if tomllib is not None and text:
            try:
                data = tomllib.loads(text)
            except Exception:  # pragma: no cover - invalid TOML falls back to manual parsing
                data = None
        if data and isinstance(data, dict):
            theme_section = data.get("theme")
            if isinstance(theme_section, dict):
                primary = str(theme_section.get("primaryColor", primary))
                secondary = str(
                    theme_section.get(
                        "secondaryColor",
                        theme_section.get("secondary_color", secondary),
                    )
                )
        elif text:
            for raw_line in text.splitlines():
                if "=" not in raw_line:
                    continue
                key, raw_value = raw_line.split("=", 1)
                cleaned_key = key.strip()
                cleaned_value = raw_value.strip().strip('"').strip("'")
                if cleaned_key == "primaryColor" and cleaned_value:
                    primary = cleaned_value
                elif cleaned_key == "secondaryColor" and cleaned_value:
                    secondary = cleaned_value
        primary = _normalise_hex_color(primary, DEFAULT_PRIMARY_COLOR)
        secondary = _normalise_hex_color(secondary, DEFAULT_SECONDARY_COLOR)
    return ThemeColors(primary=primary, secondary=secondary)
