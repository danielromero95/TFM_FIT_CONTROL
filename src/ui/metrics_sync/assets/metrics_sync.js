(function() {
  const DATA = $DATA_JSON;
  const CFG  = $PLOT_CONFIG_JSON;

  const video   = document.getElementById("$VIDEO_ID");
  const plot    = document.getElementById("$PLOT_ID");
  const wrapper = document.getElementById("$WRAPPER_ID");

  const x = DATA.times.slice();
  const names = Object.keys(DATA.series || {});
  if (!names.length) {
    plot.innerHTML = "<div style='color:#9ca3af'>No series to render.</div>";
    return;
  }

  const isTime = (DATA.x_mode === "time");

  function secondsToLabel(sec) {
    if (sec < 60) return sec.toFixed(2);
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  }

  const formattedHoverX = x.map((value) => (isTime ? secondsToLabel(value) : Math.round(value)));
  const hoverTemplate = isTime
    ? "%{y:.2f}<extra>Time: %{customdata}</extra>"
    : "%{y:.2f}<extra>Frame: %{customdata}</extra>";

  const traces = names.map((name) => ({
    x: x,
    y: DATA.series[name],
    mode: "lines",
    name,
    hovertemplate: hoverTemplate,
    customdata: formattedHoverX.slice()
  }));

  const fps = DATA.fps > 0 ? DATA.fps : 1.0;

  const cursor = { type: "line", x0: 0, x1: 0, y0: 0, y1: 1, xref: "x", yref: "paper", line: { width: 2, dash: "dot", color: "#ef4444" } };
  const bands = (DATA.rep || []).map(([f0, f1]) => ({
    type: "rect", xref: "x", yref: "paper",
    x0: isTime ? (f0 / fps) : f0,
    x1: isTime ? (f1 / fps) : f1,
    y0: 0, y1: 1, fillcolor: "rgba(160,160,160,0.18)", line: {width: 0}
  }));

  const xTitle = isTime ? "Time (s)" : "Frame #";

  const layout = {
    uirevision: "keep",
    margin: { l: 56, r: 24, t: 16, b: 88 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: { color: "#111827", size: 12 },
    showlegend: true,
    hovermode: "x unified",
    hoverlabel: { bgcolor: "#ffffff", bordercolor: "#94a3b8", font: { color: "#111827" } },
    xaxis: {
      title: xTitle,
      zeroline: false,
      showgrid: true,
      gridcolor: "rgba(0,0,0,0.06)",
      linecolor: "#94a3b8",
      ticks: "outside",
      tickcolor: "#94a3b8",
      tickformat: isTime ? ".2f" : null
    },
    yaxis: {
      title: "Angle (°)",
      zeroline: false,
      showgrid: true,
      gridcolor: "rgba(0,0,0,0.06)",
      linecolor: "#94a3b8",
      ticks: "outside",
      tickcolor: "#94a3b8"
    },
    legend: {
      orientation: "h",
      x: 0.5,
      xanchor: "center",
      y: -0.25,
      yanchor: "top"
    },
    shapes: [cursor, ...bands]
  };

  let fitScheduled = false;
  function fit() {
    if (fitScheduled) return;
    fitScheduled = true;
    (window.requestAnimationFrame || function(cb){ return setTimeout(cb, 16); })(() => {
      fitScheduled = false;
      if (window.frameElement && wrapper) {
        window.frameElement.style.height = (wrapper.scrollHeight + 24) + "px";
      }
    });
  }

  Plotly.newPlot(plot, traces, layout, CFG).then(() => {
    fit();
    plot.on("plotly_relayout", fit);
    plot.on("plotly_afterplot", fit);
    plot.on("plotly_restyle", fit);
    plot.on("plotly_update", fit);
    plot.on("plotly_legendclick", () => { setTimeout(fit, 0); });
    plot.on("plotly_legenddoubleclick", () => { setTimeout(fit, 0); });
  });

  let lastT = -1;
  function updateCursorFromVideo() {
    const rawT = video.currentTime || 0;
    const frame = Math.max(0, Math.round(rawT * fps));
    const t = frame / fps;
    if (Math.abs(t - lastT) < 0.01) return;
    lastT = t;
    const xVal = (DATA.x_mode === "time") ? t : frame;
    Plotly.relayout(plot, {"shapes[0].x0": xVal, "shapes[0].x1": xVal});
    fit();
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

  if (typeof ResizeObserver !== "undefined" && wrapper) {
    new ResizeObserver(fit).observe(wrapper);
  }
  window.addEventListener("load", fit, { once: true });
  setTimeout(fit, 50);
})();
