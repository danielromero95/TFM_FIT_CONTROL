from __future__ import annotations

import inspect
import json
import uuid
from math import ceil
from string import Template
from typing import Iterable

import numpy as np
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
    # Normalize FPS
    fps = float(fps) if float(fps) > 0 else 1.0

    n = int(len(df))
    if n <= 0:
        return {"times": [], "series": {}, "fps": fps, "x_mode": "frame"}

    # Compute stride once based on total points
    stride = 1
    if max_points and n > int(max_points):
        stride = max(1, int(ceil(n / float(max_points))))

    # Build a single index array for all downsampling
    idx = np.arange(0, n, stride, dtype=int)

    # X axis: time from frame_idx (if present), otherwise frame index
    if "frame_idx" in df.columns:
        # to_numpy avoids per-row Python overhead; coerce to float64
        frames = pd.to_numeric(df["frame_idx"], errors="coerce").to_numpy(
            dtype="float64", copy=False
        )
        # Fallback if all-NaN: use 0..n-1
        if not np.isfinite(frames).any():
            frames = np.arange(n, dtype="float64")
        times_arr = frames[idx] / fps
        x_mode = "time"
    else:
        times_arr = idx.astype("float64", copy=False)
        x_mode = "frame"

    # Convert times to Python scalars
    times = times_arr.tolist()

    # Prepare selected series that exist in the frame
    present = [c for c in selected if c in df.columns]
    series: dict[str, list[float | None]] = {}
    if present:
        # Single materialization: (n, k)
        data = df[present].to_numpy(dtype="float64", copy=False)
        # Downsample rows
        data_ds = data[idx, :]
        # Vectorized NaN -> None conversion for JSON (object dtype)
        mask = np.isfinite(data_ds)
        obj = data_ds.astype(object)
        obj[~mask] = None
        # Export one list per series
        for j, name in enumerate(present):
            series[name] = obj[:, j].tolist()

    return {"times": times, "series": series, "fps": fps, "x_mode": x_mode}


@st.cache_data(show_spinner=False)
def _load_asset_text(name: str) -> str:
    path = files("src.ui.metrics_sync").joinpath("assets", name)
    return path.read_text(encoding="utf-8")


def render_video_with_metrics_sync(
    *,
    video_path: str | None = None,
    metrics_df: pd.DataFrame,
    selected_metrics: list[str],
    fps: float | int,
    rep_intervals: list[tuple[int, int]] | None = None,
    thresholds: list[float] | tuple[float, float] | None = None,
    start_at_s: float | None = None,
    scroll_zoom: bool = True,
    key: str = "video_metrics_sync",
    max_width_px: int = 720,
    show_video: bool = False,
    sync_channel: str | None = None,
) -> None:
    if metrics_df is None or metrics_df.empty:
        st.info("No metrics available to display.")
        return
    if not selected_metrics:
        st.info("Select at least one metric to plot.")
        return

    payload = _build_payload(metrics_df, selected_metrics, fps=fps)
    payload["rep"] = rep_intervals or []
    payload["startAt"] = float(start_at_s) if start_at_s is not None else None
    thr_values: list[float] = []
    if thresholds is not None:
        if isinstance(thresholds, (int, float)):
            candidates = [thresholds]
        else:
            try:
                candidates = list(thresholds)
            except TypeError:
                candidates = [thresholds]
        for value in candidates:
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                thr_values.append(fv)
    if thr_values:
        payload["thr"] = thr_values

    data_json = json.dumps(payload, separators=(",", ":"))
    plot_config_json = json.dumps(
        {"displayModeBar": False, "responsive": True, "scrollZoom": bool(scroll_zoom)},
        separators=(",", ":"),
    )

    video_id = f"vmx-video-{uuid.uuid4().hex}"
    plot_id = f"vmx-plot-{uuid.uuid4().hex}"
    wrapper_id = f"vmx-wrap-{uuid.uuid4().hex}"

    show_video_final = bool(show_video and video_path)
    source_uri: str | None = None
    mime: str | None = None
    if show_video_final:
        try:
            source_uri, mime = get_video_source(video_path, format=None)
        except FileNotFoundError:
            st.error(f"Video file not found: {video_path}")
            show_video_final = False

    css = _load_asset_text("metrics_sync.css")
    js_tpl = Template(_load_asset_text("metrics_sync.js"))
    js = js_tpl.substitute(
        VIDEO_ID=video_id,
        PLOT_ID=plot_id,
        WRAPPER_ID=wrapper_id,
        HAS_VIDEO=str(show_video_final).lower(),
        DATA_JSON=data_json,
        PLOT_CONFIG_JSON=plot_config_json,
        SYNC_CHANNEL=json.dumps(sync_channel),
    )

    try:
        signature = inspect.signature(html)
    except Exception:
        signature = None
    html_kwargs: dict[str, object] = {"height": 560, "scrolling": False}
    if signature and "key" in signature.parameters:
        html_kwargs["key"] = key

    wrapper_style = f"width:100%;max-width:{max_width_px}px;margin:0 auto;"
    wrapper_class = "vmx-wrapper vmx-wrapper--plot-only" if not show_video_final else "vmx-wrapper"

    video_block = ""
    if show_video_final and source_uri and mime:
        video_block = (
            f"  <div class=\"vmx-video-box\">\n"
            f"    <video id=\"{video_id}\" controls preload=\"metadata\" playsinline "
            f"webkit-playsinline aria-label=\"Analysis video\">\n"
            f"      <source src=\"{source_uri}\" type=\"{mime}\">\n"
            f"    </video>\n"
            f"  </div>\n"
        )
    elif show_video and not video_path:
        st.warning("No video was provided to display alongside the metrics.")

    html(
        f"""
<style>{css}</style>

<div id="{wrapper_id}" class="{wrapper_class}" style="{wrapper_style}">
{video_block}  <div id="{plot_id}" class="vmx-plot"></div>
</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>{js}</script>
        """,
        **html_kwargs,
    )
