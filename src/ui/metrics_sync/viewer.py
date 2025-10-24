from __future__ import annotations

import inspect
import json
import uuid
from math import ceil
from string import Template
from typing import Iterable

import pandas as pd
import streamlit as st
from importlib.resources import files
from streamlit.components.v1 import html

from src.ui.video import get_video_source


def _series_list(s: Iterable[float]) -> list[float | None]:
    out: list[float | None] = []
    for v in s:
        if pd.isna(v):
            out.append(None)
        else:
            try:
                out.append(float(v))
            except Exception:
                out.append(None)
    return out


def _build_payload(
    df: pd.DataFrame,
    selected: list[str],
    fps: float | int,
    *,
    max_points: int = 3000,
) -> dict:
    fps = float(fps) if float(fps) > 0 else 1.0
    if "frame_idx" in df.columns:
        times_raw = [float(i) / fps for i in df["frame_idx"].tolist()]
        x_mode = "time"
    else:
        times_raw = list(range(len(df)))
        x_mode = "frame"

    stride = 1
    if max_points and len(times_raw) > max_points:
        stride = max(1, ceil(len(times_raw) / max_points))
    times = times_raw[::stride]

    series: dict[str, list[float | None]] = {}
    for col in selected:
        if col in df.columns:
            vals = _series_list(df[col].tolist())
            series[col] = vals[::stride]

    return {"times": times, "series": series, "fps": fps, "x_mode": x_mode}


@st.cache_data(show_spinner=False)
def _load_asset_text(name: str) -> str:
    path = files("src.ui.metrics_sync").joinpath("assets", name)
    return path.read_text(encoding="utf-8")


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
    wrapper_id = f"vmx-wrap-{uuid.uuid4().hex}"

    css = _load_asset_text("metrics_sync.css")
    js_tpl = Template(_load_asset_text("metrics_sync.js"))
    js = js_tpl.substitute(
        VIDEO_ID=video_id,
        PLOT_ID=plot_id,
        WRAPPER_ID=wrapper_id,
        SOURCE_URI=source_uri,
        MIME=mime,
        DATA_JSON=data_json,
        PLOT_CONFIG_JSON=plot_config_json,
    )

    try:
        signature = inspect.signature(html)
    except Exception:
        signature = None
    html_kwargs: dict[str, object] = {"height": 560, "scrolling": False}
    if signature and "key" in signature.parameters:
        html_kwargs["key"] = key

    wrapper_style = (
        f"width:100%;max-width:{max_width_px}px;margin:0 auto {bottom_margin_rem:.2f}rem auto;"
    )

    html(
        f"""
<style>{css}</style>

<div id="{wrapper_id}" style="{wrapper_style}">
  <div class="vmx-video-box">
    <video id="{video_id}" controls preload="metadata" playsinline webkit-playsinline aria-label="Analysis video">
      <source src="{source_uri}" type="{mime}">
    </video>
  </div>
  <div id="{plot_id}"></div>
</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>{js}</script>
        """,
        **html_kwargs,
    )
