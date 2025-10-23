from __future__ import annotations

import inspect
import json
import uuid
from math import ceil
from pathlib import Path
from string import Template

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

from src.ui.video import get_video_source


# ---------- Helpers de datos (vectorizados) ----------

def _series_list(s: pd.Series) -> list[float | None]:
    """Convierte una serie a lista float, usando None para NaN (JSON-safe)."""
    return pd.to_numeric(s, errors="coerce").where(pd.notna, None).tolist()


def _build_payload(
    df: pd.DataFrame,
    selected: list[str],
    fps: float | int,
    *,
    max_points: int = 3000,
) -> dict:
    fps = float(fps) if isinstance(fps, (int, float)) and fps > 0 else 1.0

    if "frame_idx" in df.columns:
        frame_idx = pd.to_numeric(df["frame_idx"], errors="coerce")
        # FutureWarning: usar ffill/bfill en vez de fillna(method=...)
        frame_idx = frame_idx.ffill().bfill()
        times_series = frame_idx.astype(float) / fps
        x_mode = "time"
    else:
        times_series = pd.Series(range(len(df)), dtype=float)
        x_mode = "frame"

    total_points = len(times_series)
    stride = 1 if not (max_points and total_points > max_points) else max(1, ceil(total_points / max_points))
    times = times_series.iloc[::stride].tolist()

    # Downsampling antes de convertir → más eficiente
    df_down = df.iloc[::stride].reset_index(drop=True)

    series: dict[str, list[float | None]] = {}
    for col in selected:
        if col in df_down.columns:
            series[col] = _series_list(df_down[col])

    return {"times": times, "series": series, "fps": fps, "x_mode": x_mode}


# ---------- Carga de assets (CSS/JS) ----------

def _read_text_asset(filename: str) -> str:
    """Lee assets junto al módulo, con fallback seguro si se empaqueta."""
    try:
        from importlib.resources import files

        return (files(__package__) / filename).read_text(encoding="utf-8")
    except Exception:
        return (Path(__file__).parent / filename).read_text(encoding="utf-8")


# ---------- Render principal ----------

def render_video_with_metrics_sync(
    *,
    video_path: str,
    metrics_df: pd.DataFrame,
    selected_metrics: list[str],
    fps: float | int,
    rep_intervals: list[tuple[int, int]] | None = None,
    start_at_s: float | None = None,
    scroll_zoom: bool = True,
    key: str = "video_metrics_sync",
    max_width_px: int = 720,
    bottom_margin_rem: float = 1.0,
) -> None:
    """Visor combinado: vídeo + gráfica Plotly con cursor sincronizado (tema claro)."""
    try:
        source_uri, mime = get_video_source(video_path, format=None)
    except FileNotFoundError:
        st.error(f"Video file not found: {video_path}")
        return

    if metrics_df is None or metrics_df.empty:
        st.info("No metrics available to display.")
        return
    if not selected_metrics:
        st.info("Select at least one metric to plot.")
        return

    payload = _build_payload(metrics_df, selected_metrics, fps=fps)
    payload["rep"] = rep_intervals or []
    payload["startAt"] = float(start_at_s) if start_at_s is not None else None
    data_json = json.dumps(payload, separators=(",", ":"))
    plot_config_json = json.dumps(
        {"displayModeBar": False, "responsive": True, "scrollZoom": bool(scroll_zoom)},
        separators=(",", ":"),
    )

    video_id = f"vmx-video-{uuid.uuid4().hex}"
    plot_id = f"vmx-plot-{uuid.uuid4().hex}"
    wrapper_dom_id = f"vmx-wrap-{uuid.uuid4().hex}"

    css_str = _read_text_asset("metrics_sync.css")
    js_template_str = _read_text_asset("metrics_sync.js")

    css_vars = f"--vmx-max-width: {max_width_px}px; --vmx-bottom-margin: {bottom_margin_rem:.2f}rem;"

    js_filled = Template(js_template_str).substitute(
        data_json=data_json,
        plot_config_json=plot_config_json,
        video_id=video_id,
        plot_id=plot_id,
    )

    html_template = Template(
        """
<style>$css</style>

<div id="$wrapper_dom_id" class="vmx-wrapper" style="$css_vars">
  <div class="vmx-video-box">
    <video id="$video_id" controls preload="metadata" playsinline webkit-playsinline aria-label="Analysis video">
      <source src="$source_uri" type="$mime">
    </video>
  </div>
  <div id="$plot_id" class="vmx-plot-box"></div>
</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
$js
</script>
"""
    )

    html_markup = html_template.substitute(
        css=css_str,
        wrapper_dom_id=wrapper_dom_id,
        css_vars=css_vars,
        video_id=video_id,
        source_uri=source_uri,
        mime=mime,
        plot_id=plot_id,
        js=js_filled,
    )

    try:
        signature = inspect.signature(html)
    except Exception:
        signature = None
    html_kwargs: dict[str, object] = {"height": 560, "scrolling": False}
    if signature and "key" in signature.parameters:
        html_kwargs["key"] = key

    html(html_markup, **html_kwargs)
