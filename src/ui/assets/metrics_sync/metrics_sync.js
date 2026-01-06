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

  const timeOffset = Number.isFinite(DATA.time_offset_s)
    ? Number(DATA.time_offset_s)
    : 0;

  const axisRef = typeof DATA.axis_ref === "string" ? DATA.axis_ref : "analysis";

  const x = DATA.times.slice();
  const frames = Array.isArray(DATA.frames) ? DATA.frames.slice() : x.map((_, idx) => idx);
  const axisTimes = Array.isArray(DATA.axis_times) && DATA.axis_times.length ? DATA.axis_times.slice() : x;
  const axisFrames = Array.isArray(DATA.axis_frames) && DATA.axis_frames.length ? DATA.axis_frames.slice() : frames;
  const names = Object.keys(DATA.series || {});
  const thr = Array.isArray(DATA.thr)
    ? DATA.thr.filter((v) => Number.isFinite(v))
    : [];
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
  const thrShapes = thr.map((y) => ({
    type: "line",
    xref: "paper",
    yref: "y",
    x0: 0,
    x1: 1,
    y0: y,
    y1: y,
    layer: "above",
    line: { width: 1, dash: "dot", color: "rgba(255,255,255,0.7)" }
  }));
  const bands = [];

  const layout = {
    margin: { l: 40, r: 10, t: 10, b: 90 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    showlegend: true,
    hovermode: "x unified",
    dragmode: "pan",
    xaxis: {
      title: (DATA.x_mode === "time") ? "Time (s)" : "Frame",
      zeroline: false,
      showgrid: true,
      gridcolor: "rgba(255,255,255,0.10)",
      gridwidth: 1,
      linecolor: "rgba(255,255,255,0.25)",
      tickfont: { color: "#ffffff" },
      titlefont: { color: "#ffffff" }
    },
    yaxis: {
      zeroline: false,
      showgrid: true,
      gridcolor: "rgba(255,255,255,0.10)",
      gridwidth: 1,
      linecolor: "rgba(255,255,255,0.25)",
      tickfont: { color: "#ffffff" }
    },
    hoverlabel: {
      bgcolor: "rgba(17,24,39,0.85)",
      bordercolor: "rgba(255,255,255,0.15)",
      font: { color: "#ffffff" }
    },
    legend: {
      font: { color: "#ffffff" },
      orientation: "h",
      x: 0.5,
      xanchor: "center",
      y: -0.15,
      yanchor: "top"
    },
    shapes: [cursor, ...thrShapes, ...bands]
  };

  Plotly.newPlot(plot, traces, layout, CFG);

  const hasTimeAxis = DATA.x_mode === "time";
  const xBounds = hasTimeAxis && axisTimes.length ? axisTimes : x;
  const xMin = xBounds.length ? xBounds[0] : 0;
  const xMax = xBounds.length ? xBounds[xBounds.length - 1] : xMin;
  const EPS = (xMax - xMin) * 1e-3 || 1e-6;

  let lastT = -1;

  function nearestIndex(target, arr) {
    if (!arr.length) return 0;
    let lo = 0;
    let hi = arr.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (lo > 0 && Math.abs(arr[lo - 1] - target) <= Math.abs(arr[lo] - target)) {
      return lo - 1;
    }
    return lo;
  }

  function stateFromAxis(axisValue) {
    const axisLookup = hasTimeAxis ? axisTimes : x;
    const plotLookup = x;
    if (!axisLookup.length && !plotLookup.length) {
      return { frame: 0, time: 0, axisTime: 0, x: axisValue, axisIdx: 0, plotIdx: 0, idx: 0 };
    }
    const axisIdx = nearestIndex(axisValue, axisLookup.length ? axisLookup : plotLookup);
    const plotIdx = nearestIndex(axisValue, plotLookup.length ? plotLookup : axisLookup);
    const frameVal = Number.isFinite(axisFrames[axisIdx]) ? axisFrames[axisIdx] : axisIdx;
    const axisTime = hasTimeAxis
      ? axisLookup[axisIdx]
      : (Number.isFinite(frameVal) && fps > 0 ? frameVal / fps : 0);
    const timeVal = axisTime + timeOffset;
    const xVal = plotLookup.length ? plotLookup[plotIdx] : axisValue;
    return { frame: frameVal, time: timeVal, axisTime, x: xVal, axisIdx, plotIdx, idx: plotIdx };
  }

  function timeForFrame(frameVal) {
    if (!hasTimeAxis || !axisTimes.length || !axisFrames.length) return null;
    const idx = nearestIndex(frameVal, axisFrames);
    const t = axisTimes[idx];
    return Number.isFinite(t) ? t : null;
  }

  if (Array.isArray(DATA.rep)) {
    DATA.rep.forEach(([f0, f1]) => {
      const start = hasTimeAxis ? (timeForFrame(f0) ?? f0 / fps) : f0;
      const end = hasTimeAxis ? (timeForFrame(f1) ?? f1 / fps) : f1;
      bands.push({
        type: "rect", xref: "x", yref: "paper",
        x0: start, x1: end,
        y0: 0, y1: 1, fillcolor: "rgba(160,160,160,0.15)", line: { width: 0 }
      });
    });
    if (bands.length) {
      const shapes = [cursor, ...thrShapes, ...bands];
      Plotly.relayout(plot, { shapes });
    }
  }

  function setCursorForAxis(axisValue) {
    const state = stateFromAxis(axisValue);
    if (Math.abs(state.time - lastT) < 1e-6) {
      return state;
    }
    lastT = state.time;
    const nextCursor = { ...cursor, x0: state.x, x1: state.x };
    const shapes = plot.layout && Array.isArray(plot.layout.shapes) ? plot.layout.shapes.slice() : [];
    shapes[CURSOR_INDEX] = nextCursor;
    Plotly.relayout(plot, { shapes });
    return state;
  }

  function seekVideoToAxis(xVal, opts = {}) {
    const clampedX = Math.min(xMax, Math.max(xMin, xVal));
    const state = setCursorForAxis(clampedX);
    if (video) {
      try {
        if (opts.pause && !video.paused) {
          video.pause();
        }
        if (Math.abs((video.currentTime || 0) - state.time) > 1 / fps) {
          video.currentTime = Math.max(0, state.time);
        }
      } catch (err) {
        console.warn("Seek error:", err);
      }
    }
    if (bc) bc.postMessage({ type: "seek", t: state.time, origin: "metrics" });
  }

  function axisValueFromSeconds(seconds) {
    const t = Number.isFinite(seconds) ? seconds : 0;
    return (hasTimeAxis ? t : t * fps) - timeOffset;
  }

  function updateCursorFromVideo(seconds) {
    if (!video) return;
    const rawT = Number.isFinite(seconds) ? seconds : video.currentTime || 0;
    const axisValue = axisValueFromSeconds(rawT);
    const state = setCursorForAxis(axisValue);
    if (bc) bc.postMessage({ type: "time", t: state.time });
  }

  const supportsVFC = video && typeof video.requestVideoFrameCallback === "function";

  if (video) {
    ["timeupdate","seeked"].forEach((ev) => video.addEventListener(ev, () => updateCursorFromVideo()));
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
    setCursorForAxis(initialX);
  }

  let rafId = null;
  let vfcHandle = null;

  function cancelVideoFrameCallback() {
    if (video && typeof video.cancelVideoFrameCallback === "function" && vfcHandle !== null) {
      try { video.cancelVideoFrameCallback(vfcHandle); } catch (e) {}
    }
    vfcHandle = null;
  }

  function tickVFC() {
    if (!video || !supportsVFC) return;
    const cb = (_now, metadata) => {
      const mediaTime = Number.isFinite(metadata?.mediaTime) ? metadata.mediaTime : video.currentTime || 0;
      updateCursorFromVideo(mediaTime);
      if (!video.paused && !video.ended) {
        vfcHandle = video.requestVideoFrameCallback(cb);
      }
    };
    vfcHandle = video.requestVideoFrameCallback(cb);
  }

  function tick() {
    if (!video) return;
    updateCursorFromVideo();
    if (!video.paused && !video.ended) { rafId = requestAnimationFrame(tick); }
  }
  if (video) {
    if (supportsVFC) {
      video.addEventListener("play", () => { cancelAnimationFrame(rafId); cancelVideoFrameCallback(); tickVFC(); });
      video.addEventListener("pause", () => { cancelVideoFrameCallback(); });
      video.addEventListener("ended", () => { cancelVideoFrameCallback(); });
      tickVFC();
    } else {
      video.addEventListener("play",  () => { cancelAnimationFrame(rafId); tick(); });
      video.addEventListener("pause", () => { cancelAnimationFrame(rafId); });
      video.addEventListener("ended", () => { cancelAnimationFrame(rafId); });
    }
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
        const axisValue = (hasTimeAxis ? msg.t : msg.t * fps) - timeOffset;
        setCursorForAxis(axisValue);
      } else if (msg.type === "seek" && Number.isFinite(msg.t)) {
        const target = Math.max(0, msg.t);
        if (video) {
          try { if (!video.paused) video.pause(); video.currentTime = target; } catch (e) {}
        } else {
          const axisValue = (hasTimeAxis ? target : target * fps) - timeOffset;
          setCursorForAxis(axisValue);
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
