from __future__ import annotations

import inspect
import json
from math import ceil
from string import Template
from typing import Iterable

import pandas as pd
import streamlit as st
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
    bottom_margin_rem: float = 0.0,
) -> None:
    """
    Visor combinado: vídeo + gráfica Plotly con cursor sincronizado.
    - Cursor sigue el currentTime (redondeado a frame).
    - Click en gráfica -> seek exacto a frame.
    - Downsampling para fluidez.
    - Bandas por repetición + scrollZoom + double-click reset.
    """
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

    import uuid
    video_id = f"vmx-video-{uuid.uuid4().hex}"
    plot_id = f"vmx-plot-{uuid.uuid4().hex}"
    wrapper_id = f"vmx-wrap-{uuid.uuid4().hex}"

    bottom_margin_value = max(float(bottom_margin_rem), 0.0)

    template = Template(
        """
<style>
  #$wrapper_id {
    width: 100%;
    max-width: ${max_width}px;
    margin: 0 auto $bottom_margin auto;
  }
  .vmx-video-box {
    position: relative;
    width: 100%;
    padding-top: 56.25%;
    background: #000;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 18px 36px rgba(15,23,42,.35);
    margin: 0 auto .75rem auto;
  }
  .vmx-video-box > video {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: contain;
    background: #000;
  }
</style>

<div id="$wrapper_id">
  <div class="vmx-video-box">
    <video id="$video_id" controls preload="metadata" playsinline webkit-playsinline aria-label="Analysis video">
      <source src="$source_uri" type="$mime">
    </video>
  </div>
  <div id="$plot_id"></div>
</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
(function() {
  const DATA = $data_json;
  const CFG  = $plot_config_json;

  const video = document.getElementById("$video_id");
  const plot  = document.getElementById("$plot_id");

  const x = DATA.times.slice();
  const names = Object.keys(DATA.series || {});
  if (!names.length) {
    plot.innerHTML = "<div style='color:#9ca3af'>No series to render.</div>";
    return;
  }

  const traces = names.map((name) => ({
    x: x,
    y: DATA.series[name],
    mode: "lines",
    name,
    hovertemplate: "%{y:.2f}<extra>%{x:.2f}</extra>"
  }));

  const fps = DATA.fps > 0 ? DATA.fps : 1.0;

  const cursor = { type: "line", x0: 0, x1: 0, y0: 0, y1: 1, xref: "x", yref: "paper", line: { width: 2, dash: "dot" } };
  const bands = (DATA.rep || []).map(([f0, f1]) => ({
    type: "rect", xref: "x", yref: "paper",
    x0: (DATA.x_mode === "time") ? (f0 / fps) : f0,
    x1: (DATA.x_mode === "time") ? (f1 / fps) : f1,
    y0: 0, y1: 1, fillcolor: "rgba(160,160,160,0.15)", line: {width: 0}
  }));

  const layout = {
    margin: {l: 40, r: 20, t: 10, b: 40},
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    showlegend: true,
    hovermode: "x unified",
    xaxis: { title: (DATA.x_mode === "time") ? "Time (s)" : "Frame", zeroline: false },
    yaxis: { zeroline: false },
    shapes: [cursor, ...bands]
  };

  Plotly.newPlot(plot, traces, layout, CFG);

  let lastT = -1;
  function updateCursorFromVideo() {
    const rawT = video.currentTime || 0;
    const frame = Math.max(0, Math.round(rawT * fps));
    const t = frame / fps;
    if (Math.abs(t - lastT) < 0.01) return;
    lastT = t;
    const xVal = (DATA.x_mode === "time") ? t : frame;
    Plotly.relayout(plot, {"shapes[0].x0": xVal, "shapes[0].x1": xVal});
  }

  ["timeupdate","seeked"].forEach((ev) => video.addEventListener(ev, updateCursorFromVideo));
  video.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(DATA.startAt)) {
      try { video.currentTime = Math.max(0, DATA.startAt); } catch (e) {}
    }
    updateCursorFromVideo();
  });
  updateCursorFromVideo();

  let rafId = null;
  function tick() {
    updateCursorFromVideo();
    if (!video.paused && !video.ended) { rafId = requestAnimationFrame(tick); }
  }
  video.addEventListener("play",  () => { cancelAnimationFrame(rafId); tick(); });
  video.addEventListener("pause", () => { cancelAnimationFrame(rafId); });
  video.addEventListener("ended", () => { cancelAnimationFrame(rafId); });

  plot.on("plotly_click", (ev) => {
    if (!ev || !ev.points || !ev.points.length) return;
    const xClicked = ev.points[0].x;
    const targetFrame = (DATA.x_mode === "time")
      ? Math.max(0, Math.round(xClicked * fps))
      : Math.max(0, Math.round(xClicked));
    const newTime = targetFrame / fps;
    try { video.currentTime = Math.max(0, newTime); video.pause(); updateCursorFromVideo(); }
    catch (err) { console.warn("Seek error:", err); }
  });

  plot.on("plotly_doubleclick", () => Plotly.relayout(plot, {"xaxis.autorange": true}));
})();
</script>
        """
    )

    # Compat: pasar 'key' a html sólo si la versión lo soporta (Streamlit 1.50.0 no lo soporta)
    try:
        signature = inspect.signature(html)
    except Exception:
        signature = None
    html_kwargs: dict[str, object] = {"height": 560, "scrolling": False}
    if signature and "key" in signature.parameters:
        html_kwargs["key"] = key

    html(
        template.substitute(
            wrapper_id=wrapper_id,
            video_id=video_id,
            plot_id=plot_id,
            source_uri=source_uri,
            mime=mime,
            data_json=data_json,
            plot_config_json=plot_config_json,
            max_width=max_width_px,
            bottom_margin=f"{bottom_margin_value:.2f}rem",
        ),
        **html_kwargs,
    )
