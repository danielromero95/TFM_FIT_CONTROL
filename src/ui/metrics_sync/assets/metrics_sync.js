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

  const traces = names.map((name) => ({
    x: x,
    y: DATA.series[name],
    mode: "lines",
    name,
    hovertemplate: "%{y:.2f}<extra>%{x:.2f}</extra>"
  }));

  const fps = DATA.fps > 0 ? DATA.fps : 1.0;

  const cursor = { type: "line", x0: 0, x1: 0, y0: 0, y1: 1, xref: "x", yref: "paper", line: { width: 2, dash: "dot", color: "#ef4444" } };
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

  if (window.frameElement && wrapper) {
    const fit = () => { window.frameElement.style.height = (wrapper.scrollHeight + 24) + "px"; };
    const ro = (typeof ResizeObserver !== "undefined") ? new ResizeObserver(fit) : null;
    if (ro) ro.observe(wrapper);
    window.addEventListener("load", fit, { once: true });
    setTimeout(fit, 50);
  }
})();
