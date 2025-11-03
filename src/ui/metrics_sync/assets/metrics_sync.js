(function() {
  const DATA = $DATA_JSON;
  const CFG  = $PLOT_CONFIG_JSON;

  const HAS_VIDEO = $HAS_VIDEO;
  const videoId = "$VIDEO_ID";
  const plotId = "$PLOT_ID";
  const wrapperId = "$WRAPPER_ID";
  const SYNC_CHANNEL = $SYNC_CHANNEL; // string o null

  const video   = HAS_VIDEO ? document.getElementById(videoId) : null;
  const plot    = document.getElementById(plotId);
  const wrapper = document.getElementById(wrapperId);
  const bc = (SYNC_CHANNEL && typeof BroadcastChannel !== "undefined")
    ? new BroadcastChannel(SYNC_CHANNEL) : null;

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

  const cursor = {
    type: "line",
    x0: 0,
    x1: 0,
    y0: 0,
    y1: 1,
    xref: "x",
    yref: "paper",
    layer: "above",
    line: { width: 2, dash: "dot", color: "#ef4444" }
  };
  const CURSOR_INDEX = 0;
  const bands = (DATA.rep || []).map(([f0, f1]) => ({
    type: "rect", xref: "x", yref: "paper",
    x0: (DATA.x_mode === "time") ? (f0 / fps) : f0,
    x1: (DATA.x_mode === "time") ? (f1 / fps) : f1,
    y0: 0, y1: 1, fillcolor: "rgba(160,160,160,0.15)", line: {width: 0}
  }));

  const layout = {
    // más espacio abajo para la leyenda horizontal
    margin: { l: 40, r: 20, t: 10, b: 64 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    hovermode: "x unified",
    dragmode: "pan",
    showlegend: true,
    // texto en blanco (aplica a ejes, leyenda, hoverlabel por defecto)
    font: { color: "#ffffff" },
    // ejes: ticks y títulos en blanco, rejilla sutil
    xaxis: {
      title: (DATA.x_mode === "time") ? "Time (s)" : "Frame",
      zeroline: false,
      showgrid: true,
      gridcolor: "rgba(255,255,255,0.08)",
      gridwidth: 1,
      linecolor: "rgba(255,255,255,0.20)",
      ticks: "outside",
      tickfont: { color: "#ffffff" },
      titlefont: { color: "#ffffff" }
    },
    yaxis: {
      zeroline: false,
      showgrid: true,
      gridcolor: "rgba(255,255,255,0.08)",
      gridwidth: 1,
      linecolor: "rgba(255,255,255,0.20)",
      ticks: "outside",
      tickfont: { color: "#ffffff" },
      titlefont: { color: "#ffffff" }
    },
    // leyenda horizontal debajo del gráfico
    legend: {
      orientation: "h",
      x: 0, xanchor: "left",
      y: -0.2, yanchor: "top",
      bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ffffff" }
    },
    shapes: [cursor, ...bands]
  };

  Plotly.newPlot(plot, traces, layout, CFG);

  const hasTimeAxis = DATA.x_mode === "time";
  const xMin = x.length ? x[0] : 0;
  const xMax = x.length ? x[x.length - 1] : xMin;
  const frameMaxFromData = hasTimeAxis ? Math.round(xMax * fps) : Math.round(xMax);
  const EPS = (xMax - xMin) * 1e-3 || 1e-6;
  const videoFrameLimit = (video && Number.isFinite(video.duration))
    ? Math.round(video.duration * fps)
    : frameMaxFromData;
  const frameUpperBound = Number.isFinite(videoFrameLimit)
    ? Math.max(0, videoFrameLimit)
    : frameMaxFromData;

  let lastT = -1;

  function clampFrame(frame) {
    if (!Number.isFinite(frame)) return 0;
    if (frameUpperBound > 0) {
      return Math.min(frameUpperBound, Math.max(0, Math.round(frame)));
    }
    return Math.max(0, Math.round(frame));
  }

  function frameFromX(xVal) {
    const rawFrame = hasTimeAxis ? Math.round(xVal * fps) : Math.round(xVal);
    return clampFrame(rawFrame);
  }

  function xFromFrame(frame) {
    const clamped = clampFrame(frame);
    return hasTimeAxis ? clamped / fps : clamped;
  }

  function setCursorForFrame(frame) {
    const clamped = clampFrame(frame);
    const time = clamped / fps;
    if (Math.abs(time - lastT) < 1e-6) {
      return { frame: clamped, time, x: xFromFrame(clamped) };
    }
    lastT = time;
    const axisValue = xFromFrame(clamped);
    const nextCursor = { ...cursor, x0: axisValue, x1: axisValue };
    const shapes = plot.layout && Array.isArray(plot.layout.shapes) ? plot.layout.shapes.slice() : [];
    shapes[CURSOR_INDEX] = nextCursor;
    Plotly.relayout(plot, { shapes });
    return { frame: clamped, time, x: axisValue };
  }

  function seekVideoToFrame(frame, { pause = true } = {}) {
    if (!video) return;
    const { time } = setCursorForFrame(frame);
    if (!Number.isFinite(time)) return;
    try {
      if (pause && !video.paused) {
        video.pause();
      }
      if (Math.abs((video.currentTime || 0) - time) > 1 / fps) {
        video.currentTime = Math.max(0, time);
      }
    } catch (err) {
      console.warn("Seek error:", err);
    }
  }

  function seekVideoToAxis(xVal, opts) {
    const clampedX = Math.min(xMax, Math.max(xMin, xVal));
    const frame = frameFromX(clampedX);
    const state = setCursorForFrame(frame);
    if (video) seekVideoToFrame(frame, opts);
    if (bc) bc.postMessage({ type: "seek", t: state.time, origin: "metrics" });
  }

  function updateCursorFromVideo() {
    if (!video) return;
    const rawT = video.currentTime || 0;
    const frame = Math.max(0, Math.round(rawT * fps));
    const t = frame / fps;
    if (Math.abs(t - lastT) < 0.01) return;
    setCursorForFrame(frame);
    if (bc) bc.postMessage({ type: "time", t });
  }

  if (video) {
    ["timeupdate","seeked"].forEach((ev) => video.addEventListener(ev, updateCursorFromVideo));
    video.addEventListener("loadedmetadata", () => {
      if (Number.isFinite(DATA.startAt)) {
        try { video.currentTime = Math.max(0, DATA.startAt); } catch (e) {}
      }
      updateCursorFromVideo();
    });
    updateCursorFromVideo();
  } else {
    // Evita el borde exacto del eje para que sea visible
    const initialX = x.length ? Math.min(xMax - EPS, Math.max(xMin + EPS, x[0])) : 0;
    setCursorForFrame(frameFromX(initialX));
  }

  let rafId = null;
  function tick() {
    if (!video) return;
    updateCursorFromVideo();
    if (!video.paused && !video.ended) { rafId = requestAnimationFrame(tick); }
  }
  if (video) {
    video.addEventListener("play",  () => { cancelAnimationFrame(rafId); tick(); });
    video.addEventListener("pause", () => { cancelAnimationFrame(rafId); });
    video.addEventListener("ended", () => { cancelAnimationFrame(rafId); });
  }

  let scrubbing = false;
  let lastHoverSync = 0;

  function finishScrub() {
    scrubbing = false;
  }

  if (video) {
    plot.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0) return;
      if (ev.target && ev.target.closest && ev.target.closest(".modebar")) return;
      scrubbing = true;
      try { video.pause(); } catch (err) {}
    });
    window.addEventListener("pointerup", finishScrub);
    plot.addEventListener("pointerleave", (ev) => {
      if (!ev.buttons) {
        finishScrub();
      }
    });

    plot.on("plotly_hover", (ev) => {
      if (!ev || !ev.points || !ev.points.length) return;
      const isActiveDrag = scrubbing || (ev.event && ev.event.buttons === 1);
      if (!isActiveDrag) return;
      const xHover = ev.points[0].x;
      const now = Date.now();
      if (now - lastHoverSync < 40) return;
      lastHoverSync = now;
      seekVideoToAxis(xHover, { pause: true });
    });

    plot.on("plotly_click", (ev) => {
      if (!ev || !ev.points || !ev.points.length) return;
      const xClicked = ev.points[0].x;
      seekVideoToAxis(xClicked, { pause: true });
      finishScrub();
    });
  }

  if (!video) {
    plot.on("plotly_hover", (ev) => {
      if (!ev || !ev.points || !ev.points.length) return;
      const isActiveDrag = (ev.event && ev.event.buttons === 1);
      if (!isActiveDrag) return;
      const xHover = ev.points[0].x;
      const now = Date.now();
      if (now - lastHoverSync < 40) return;
      lastHoverSync = now;
      seekVideoToAxis(xHover, { pause: true });
    });

    plot.on("plotly_click", (ev) => {
      if (!ev || !ev.points || !ev.points.length) return;
      const xClicked = ev.points[0].x;
      seekVideoToAxis(xClicked, { pause: true });
      finishScrub();
    });
  }

  // Canal de sincronía: recibir tiempo/seek externo
  if (bc) {
    bc.onmessage = (ev) => {
      const msg = ev && ev.data ? ev.data : null;
      if (!msg || typeof msg !== "object") return;
      if (msg.type === "time" && Number.isFinite(msg.t)) {
        const frame = Math.max(0, Math.round(msg.t * fps));
        setCursorForFrame(frame);
      } else if (msg.type === "seek" && Number.isFinite(msg.t)) {
        const target = Math.max(0, msg.t);
        if (video) {
          try { if (!video.paused) video.pause(); video.currentTime = target; } catch (e) {}
        } else {
          const frame = Math.max(0, Math.round(target * fps));
          setCursorForFrame(frame);
        }
      }
    };
  }

  plot.on("plotly_doubleclick", () => Plotly.relayout(plot, {"xaxis.autorange": true}));

  if (window.frameElement && wrapper) {
    const fit = () => { window.frameElement.style.height = (wrapper.scrollHeight + 24) + "px"; };
    const ro = (typeof ResizeObserver !== "undefined") ? new ResizeObserver(fit) : null;
    if (ro) ro.observe(wrapper);
    window.addEventListener("load", fit, { once: true });
    setTimeout(fit, 50);
  }
})();
