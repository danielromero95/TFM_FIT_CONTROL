"""Vistas responsables de mostrar y exportar los resultados del análisis."""

from __future__ import annotations

import datetime
import html
import io
import json
import math
import zipfile
from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None  # pragma: no cover

from src.C_analysis.repetition_counter import count_repetitions_with_config
from src.pipeline_data import Report, RunStats
from src.ui.metrics_catalog import human_metric_name, metric_base_description
from src.ui.metrics_sync import render_video_with_metrics_sync
from src.ui.state import get_state
from ..utils import step_container


_METRIC_HELP_CSS_EMITTED_KEY = "_metric_help_css_emitted"


def _counting_relation_text(
    metric: str,
    exercise: str,
    primary_metric: str | None,
    *,
    is_primary: bool,
    ) -> str:
        primary_label = human_metric_name(primary_metric) if primary_metric else "the auto-selected primary angle"
        primary_candidates = {
            "squat": {"left_knee", "right_knee"},
            "bench_press": {"left_elbow", "right_elbow"},
            "deadlift": {"left_hip", "right_hip"},
        }
        if is_primary:
            return (
                "Reps are counted when this angle dips below the lower threshold and then rises past the upper threshold; "
                "threshold filtering only applies when Enable is turned on."
            )
        if primary_metric:
            if metric in primary_candidates.get(exercise, set()):
                return (
                    f"Not used for counting in this run; reps are detected from the {primary_label} crossing the configured "
                    "thresholds (filtering is applied only when Enable is turned on)."
                )
            return (
                f"Not used for counting; reps are detected from the {primary_label} crossing the configured thresholds (the upper "
                "threshold only filters reps when Enable is turned on)."
            )
        return "Not used for counting; the system will pick a primary angle automatically when enough data is available."


def _build_metric_help(
    *,
    exercise: str,
    primary_metric: str | None,
    metric_options: List[str],
) -> Dict[str, Dict[str, str]]:
    help_map: Dict[str, Dict[str, str]] = {}
    exercise = exercise or ""

    for metric in metric_options:
        base_desc: str | None
        relation: str
        prefix = "PRIMARY — " if primary_metric and metric == primary_metric else ""

        if metric.startswith("ang_vel_"):
            source_metric = metric[8:]
            human_label = human_metric_name(source_metric)
            base_desc = f"Angular velocity of the {human_label} in degrees per second."
            relation = _counting_relation_text(source_metric, exercise, primary_metric, is_primary=False)
        else:
            base_desc = metric_base_description(metric, exercise)
            if base_desc is None:
                continue
            relation = _counting_relation_text(metric, exercise, primary_metric, is_primary=(metric == primary_metric))

        body = f"{prefix}{base_desc} {relation}".strip()
        title = " ".join(body.split())
        help_map[metric] = {"title": title, "body": body}

    return help_map


def _format_run_duration(duration_ms: float | None) -> str | None:
    """Convert milliseconds into a brief human-friendly duration string."""

    if duration_ms is None or duration_ms <= 0:
        return None

    seconds_total = duration_ms / 1000.0
    minutes, seconds = divmod(seconds_total, 60)
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:0.1f}s"
    return f"{seconds_total:0.1f}s"


def _serialize_stat_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _serialize_stats(stats: RunStats) -> Dict[str, object]:
    serialized: Dict[str, object] = {}
    for field in fields(stats):
        serialized[field.name] = _serialize_stat_value(getattr(stats, field.name))
    return serialized


def _session_name_from_paths(*candidates: object | None) -> str | None:
    """Return the session directory name from the first valid candidate path."""

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parent = Path(candidate).expanduser().parent
        except Exception:
            continue
        if parent and parent.name:
            return parent.name
    return None


def _build_debug_report_bundle(
    *,
    report: Report,
    stats_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None,
    metrics_csv: str | None,
    effective_config_bytes: bytes | None,
    video_name: str | None,
) -> bytes:
    """Package run data, config, and metrics into a portable debug report."""

    bundle = io.BytesIO()
    generated_at = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    effective_config = effective_config_bytes
    if effective_config is None:
        effective_config = json.dumps(
            report.config_used.to_serializable_dict(), indent=2, ensure_ascii=False
        ).encode("utf-8")

    payload: Dict[str, object] = {
        "generated_at_utc": generated_at,
        "video_name": video_name,
        "stats": _serialize_stats(report.stats),
        "warnings": list(report.stats.warnings),
        "skip_reason": report.stats.skip_reason,
        "debug_notes": getattr(report.stats, "debug_notes", []),
        "config_sha1": report.stats.config_sha1,
        "config_used": report.config_used.to_serializable_dict(),
    }

    if metrics_df is not None:
        payload["metrics_preview"] = metrics_df.head(20).to_dict(orient="records")

    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("report.json", json.dumps(payload, indent=2, ensure_ascii=False))
        zf.writestr("run_stats.csv", stats_df.to_csv(index=False))
        if metrics_csv:
            zf.writestr("metrics.csv", metrics_csv)
        if effective_config:
            zf.writestr("effective_config.json", effective_config)

    bundle.seek(0)
    return bundle.read()


def _run_parameters(stats: RunStats) -> List[Tuple[str, str, str | None]]:
    """Build a compact list of run parameters to highlight in the UI."""
    params: List[Tuple[str, str, str | None]] = []

    duration = _format_run_duration(getattr(stats, "t_total_ms", None))
    if duration:
        params.append(("Analysis time", duration, None))

    frames = getattr(stats, "frames", 0) or 0
    if frames > 0:
        params.append(("Frames analyzed", f"{frames:,}", None))

    confidence = getattr(stats, "detection_confidence", 0.0) or 0.0
    if confidence > 0:
        params.append(
            (
                "Detection confidence",
                f"{confidence * 100:.0f}%",
                (
                    "How confident the system was when automatically detecting the exercise "
                    "type and camera view for this video.\n\n"
                    "1) The exercise classifier compares squat, deadlift, and bench scores "
                    "from the detected poses, converts them into probabilities, and keeps the "
                    "probability of the winning label.\n"
                    "2) The camera-view classifier estimates how reliably the clip looks front "
                    "or side based on shoulder width/tilt and frame reliability; its score is "
                    "reduced if landmarks are unstable or few frames are trustworthy.\n\n"
                    "The detection confidence shown here is the lower of those two scores, so "
                    "it drops whenever either the exercise label or the inferred view is "
                    "uncertain."
                ),
            )
        )

    return params


def _render_run_parameters(params: List[Tuple[str, str, str | None]]) -> None:
    st.markdown(
        """
        <style>
        .run-param-card {
            text-align: center;
            padding: 6px 0;
        }

        .run-param-label {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 600;
            margin-bottom: 2px;
        }

        .run-param-value {
            font-size: 22px;
            font-weight: 700;
            color: rgba(255, 255, 255, 0.9);
            line-height: 1.2;
        }

        .run-param-help {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-left: 6px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 1px solid rgba(255, 255, 255, 0.5);
            font-size: 11px;
            font-weight: 700;
            line-height: 1;
            color: rgba(255, 255, 255, 0.85);
            background: rgba(255, 255, 255, 0.08);
            cursor: help;
        }

        @media (prefers-color-scheme: light) {
            .run-param-help {
                color: rgba(30, 30, 30, 0.8);
                background: rgba(255, 255, 255, 0.95);
                border-color: rgba(120, 120, 120, 0.5);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(min(3, len(params)))
    for idx, (label, value, help_text) in enumerate(params):
        help_icon = ""
        if help_text:
            escaped_help = html.escape(help_text, quote=True).replace("\n", "&#10;")
            help_icon = f'<span class="run-param-help" title="{escaped_help}">?</span>'
        label_html = f"{html.escape(label)}{help_icon}"
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="run-param-card">
                    <div class="run-param-label">{label_html}</div>
                    <div class="run-param-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _emit_metric_help_assets() -> None:
    if st.session_state.get(_METRIC_HELP_CSS_EMITTED_KEY):
        return
    st.session_state[_METRIC_HELP_CSS_EMITTED_KEY] = True
    st.markdown(
        """
<style>
ul[role="listbox"] li[role="option"].metric-help-option {
    position: relative;
    padding-right: 34px;
}

.metric-help-icon {
    position: absolute;
    top: 50%;
    right: 12px;
    transform: translateY(-50%);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    border: 1px solid rgba(120, 120, 120, 0.7);
    background: rgba(255, 255, 255, 0.95);
    color: rgba(34, 34, 34, 0.9);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
    z-index: 1;
}

.metric-help-icon:hover,
.metric-help-icon.metric-help-icon--open {
    background: rgba(240, 240, 240, 0.95);
    border-color: rgba(34, 34, 34, 0.8);
    color: rgba(34, 34, 34, 1);
}

.metric-help-popover {
    position: absolute;
    z-index: 99999;
    max-width: 280px;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.98);
    color: rgba(20, 20, 20, 0.94);
    border: 1px solid rgba(0, 0, 0, 0.08);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    font-size: 13px;
    line-height: 1.45;
    white-space: pre-wrap;
}

@media (prefers-color-scheme: dark) {
    .metric-help-icon {
        background: rgba(40, 40, 40, 0.95);
        color: rgba(230, 230, 230, 0.9);
        border-color: rgba(160, 160, 160, 0.7);
    }

    .metric-help-icon:hover,
    .metric-help-icon.metric-help-icon--open {
        background: rgba(70, 70, 70, 0.95);
        border-color: rgba(230, 230, 230, 0.8);
        color: rgba(255, 255, 255, 0.95);
    }

    .metric-help-popover {
        background: rgba(32, 32, 32, 0.98);
        color: rgba(245, 245, 245, 0.92);
        border-color: rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _emit_metric_help_script(widget_key: str, metric_help: Dict[str, Dict[str, str]]) -> None:
    if not metric_help:
        return
    payload = json.dumps(metric_help, ensure_ascii=False)
    st.markdown(
        f"""
<script>
(function() {{
    const widgetKey = {json.dumps(widget_key)};
    const helpMap = {payload};
    window.__metricHelpStore = window.__metricHelpStore || {{ maps: {{}}, init: false }};
    const store = window.__metricHelpStore;
    store.maps[widgetKey] = helpMap;

    function normaliseOptionText(text) {{
        if (!text) return "";
        return text.replace(/\s+/g, " ").trim();
    }}

    function ensureInit() {{
        if (store.init) return;
        store.init = true;
        store.activePopover = null;
        store.activeIcon = null;
        store.optionMap = {{}};

        store.closePopover = function() {{
            if (store.activePopover && store.activePopover.parentNode) {{
                store.activePopover.parentNode.removeChild(store.activePopover);
            }}
            store.activePopover = null;
            if (store.activeIcon) {{
                store.activeIcon.classList.remove('metric-help-icon--open');
            }}
            store.activeIcon = null;
        }};

        document.addEventListener('click', function(evt) {{
            if (!store.activePopover) return;
            const clickedIcon = evt.target.closest('.metric-help-icon');
            if (clickedIcon && clickedIcon === store.activeIcon) {{
                return;
            }}
            if (!evt.target.closest('.metric-help-popover')) {{
                store.closePopover();
            }}
        }}, true);

        document.addEventListener('keydown', function(evt) {{
            if (evt.key === 'Escape') {{
                store.closePopover();
            }}
        }});

        store.showPopover = function(icon, entry) {{
            const body = entry.body || entry.title || '';
            if (!body) return;
            if (store.activeIcon === icon) {{
                store.closePopover();
                return;
            }}
            store.closePopover();
            const pop = document.createElement('div');
            pop.className = 'metric-help-popover';
            pop.textContent = body;
            pop.style.left = '0px';
            pop.style.top = '0px';
            document.body.appendChild(pop);
            const iconRect = icon.getBoundingClientRect();
            const popRect = pop.getBoundingClientRect();
            const top = iconRect.bottom + window.scrollY + 8;
            let left = iconRect.left + window.scrollX;
            if (left + popRect.width > window.scrollX + window.innerWidth - 12) {{
                left = window.scrollX + window.innerWidth - popRect.width - 12;
            }}
            if (left < window.scrollX + 8) {{
                left = window.scrollX + 8;
            }}
            pop.style.top = `${{top}}px`;
            pop.style.left = `${{left}}px`;
            store.activePopover = pop;
            store.activeIcon = icon;
            icon.classList.add('metric-help-icon--open');
        }};

        store.decorateOptions = function() {{
            const lists = document.querySelectorAll('ul[role="listbox"]');
            lists.forEach(list => {{
                const options = list.querySelectorAll('li[role="option"]');
                options.forEach(optionEl => {{
                    if (optionEl.dataset.metricHelpDecorated === '1') return;
                    const optionText = normaliseOptionText(optionEl.textContent);
                    if (!optionText) return;
                    const entry = store.optionMap[optionText];
                    if (!entry) return;
                    optionEl.dataset.metricHelpDecorated = '1';
                    optionEl.classList.add('metric-help-option');
                    const icon = document.createElement('span');
                    icon.className = 'metric-help-icon';
                    icon.textContent = '?';
                    icon.setAttribute('title', entry.title || entry.body || '');
                    icon.addEventListener('click', function(ev) {{
                        ev.preventDefault();
                        ev.stopPropagation();
                        store.showPopover(icon, entry);
                    }});
                    optionEl.appendChild(icon);
                }});
            }});
        }};

        store.apply = function() {{
            store.optionMap = {{}};
            Object.values(store.maps).forEach(map => {{
                Object.keys(map).forEach(label => {{
                    const entry = map[label];
                    if (!entry) return;
                    const normalised = normaliseOptionText(label);
                    if (!normalised) return;
                    store.optionMap[normalised] = entry;
                }});
            }});
            store.decorateOptions();
        }};

        const observer = new MutationObserver(function() {{
            store.apply();
        }});
        observer.observe(document.body, {{ childList: true, subtree: true }});
    }}

    ensureInit();
    store.apply();
}})();
</script>
        """,
        unsafe_allow_html=True,
    )


def _compute_rep_intervals(
    metrics_df: pd.DataFrame,
    report: Report,
    stats: RunStats,
    numeric_columns: List[str],
) -> Tuple[List[Tuple[int, int]], List[int], pd.Series, str | None]:
    if metrics_df.empty:
        return [], [], pd.Series(dtype=float), None

    fps = float(getattr(stats, "fps_effective", 0.0) or 0.0)
    fps = fps if fps > 0 else 1.0

    df = metrics_df.reset_index(drop=True)
    if "frame_idx" in df.columns:
        frame_values = pd.to_numeric(df["frame_idx"], errors="coerce")
        frame_values = frame_values.where(~frame_values.isna(), pd.Series(range(len(df))))
    else:
        frame_values = pd.Series(range(len(df)))

    if frame_values.empty:
        return [], [], frame_values, None

    first_frame = int(frame_values.iloc[0])
    last_frame = int(frame_values.iloc[-1]) if len(frame_values) > 1 else first_frame + 1

    candidate = getattr(stats, "primary_angle", None)
    if not candidate or candidate not in df.columns:
        for fb in ("left_knee", "right_knee"):
            if fb in df.columns:
                candidate = fb
                break
    if not candidate or candidate not in df.columns:
        candidate = numeric_columns[0] if numeric_columns else None
    if not candidate or candidate not in df.columns:
        return [], [], frame_values, None

    valley_indices: List[int] = []
    config_used = getattr(report, "config_used", None)
    counting_cfg = getattr(config_used, "counting", None)
    faults_cfg = getattr(config_used, "faults", None)
    if counting_cfg is not None:
        try:
            _, debug = count_repetitions_with_config(
                df, counting_cfg, fps, faults_cfg=faults_cfg
            )
            valley_indices = list(getattr(debug, "valley_indices", []))
        except Exception:
            valley_indices = []

    if not valley_indices and find_peaks is not None:
        series = pd.to_numeric(df[candidate], errors="coerce").ffill().bfill().to_numpy()
        if series.size:
            prominence = float(getattr(stats, "min_prominence", 0.0) or 0.0)
            prominence_param = None if prominence <= 0 else prominence
            distance_sec = float(getattr(stats, "min_distance_sec", 0.0) or 0.0)
            distance_frames = max(1, int(round(distance_sec * fps)))
            valleys, _ = find_peaks(-series, prominence=prominence_param, distance=distance_frames)
            valley_indices = [int(i) for i in valleys]
            refractory_sec = float(getattr(stats, "refractory_sec", 0.0) or 0.0)
            if refractory_sec > 0 and valley_indices:
                refractory_frames = max(1, int(round(refractory_sec * fps)))
                filtered: List[int] = []
                for idx in valley_indices:
                    if filtered and idx - filtered[-1] < refractory_frames:
                        continue
                    filtered.append(idx)
                valley_indices = filtered

    if not valley_indices:
        return [], [], frame_values, candidate

    frame_count = len(frame_values)
    valley_frames = []
    for idx in valley_indices:
        pos = min(max(idx, 0), frame_count - 1)
        valley_frames.append(int(frame_values.iloc[pos]))

    valley_frames = sorted(set(valley_frames))
    if not valley_frames:
        return [], [], frame_values, candidate

    total_frames = max(last_frame, first_frame)
    if total_frames <= first_frame:
        total_frames = first_frame + max(len(df) - 1, 1)

    intervals: List[Tuple[int, int]] = []
    for i, frame in enumerate(valley_frames):
        start_frame = first_frame if i == 0 else int(round((valley_frames[i - 1] + frame) / 2))
        end_frame = total_frames if i == len(valley_frames) - 1 else int(round((frame + valley_frames[i + 1]) / 2))
        start_frame = max(first_frame, start_frame)
        end_frame = max(start_frame, end_frame)
        intervals.append((start_frame, end_frame))
    return intervals, valley_frames, frame_values, candidate


def _compute_rep_speeds(
    rep_intervals: List[Tuple[int, int]],
    stats: RunStats,
    *,
    valley_frames: List[int],
    frame_values: pd.Series,
    metrics_df: pd.DataFrame | None,
    primary_metric: str | None,
) -> pd.DataFrame:
    """Estimate per-rep cadence, duration, and phase speeds from frame intervals."""

    if not rep_intervals:
        return pd.DataFrame()

    fps = float(getattr(stats, "fps_effective", 0.0) or 0.0)
    fps = fps if fps > 0 else 1.0

    if metrics_df is None or metrics_df.empty or primary_metric not in metrics_df.columns:
        metrics_series = None
    else:
        metrics_series = pd.to_numeric(metrics_df[primary_metric], errors="coerce")

    def _metric_at_frame(frame: int) -> float:
        if metrics_series is None or frame_values.empty:
            return math.nan
        try:
            nearest_idx = (frame_values - frame).abs().idxmin()
        except Exception:
            return math.nan
        try:
            return float(metrics_series.iloc[int(nearest_idx)])
        except Exception:
            return math.nan

    rows: List[Dict[str, float]] = []
    for i, (start_frame, end_frame) in enumerate(rep_intervals, start=1):
        span_frames = max(1, end_frame - start_frame)
        duration_s = span_frames / fps
        cadence_rps = (1.0 / duration_s) if duration_s > 0 else 0.0

        bottom_frame = valley_frames[i - 1] if i - 1 < len(valley_frames) else int(round((start_frame + end_frame) / 2))
        bottom_frame = min(max(bottom_frame, start_frame), end_frame)

        start_val = _metric_at_frame(start_frame)
        bottom_val = _metric_at_frame(bottom_frame)
        end_val = _metric_at_frame(end_frame)

        first_phase_frames = max(1, bottom_frame - start_frame)
        second_phase_frames = max(1, end_frame - bottom_frame)

        if any(math.isnan(v) for v in (start_val, bottom_val, end_val)):
            start_delta = 0.0
            end_delta = 0.0
        else:
            start_delta = start_val - bottom_val
            end_delta = end_val - bottom_val

        if abs(start_delta) >= abs(end_delta):
            down_duration = first_phase_frames / fps
            up_duration = second_phase_frames / fps
            down_change = start_delta
            up_change = end_delta
        else:
            down_duration = second_phase_frames / fps
            up_duration = first_phase_frames / fps
            down_change = end_delta
            up_change = start_delta

        down_speed = (abs(down_change) / down_duration) if down_duration > 0 else 0.0
        up_speed = (abs(up_change) / up_duration) if up_duration > 0 else 0.0

        rows.append(
            {
                "Repetition": i,
                "Start frame": float(start_frame),
                "End frame": float(end_frame),
                "Duration (s)": duration_s,
                "Cadence (reps/min)": cadence_rps * 60.0,
                "Bottom frame": float(bottom_frame),
                "Down duration (s)": down_duration,
                "Up duration (s)": up_duration,
                "Down speed (units/s)": down_speed,
                "Up speed (units/s)": up_speed,
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["speed_unit"] = speed_unit
    return df


def _results_panel() -> Dict[str, bool]:
    with step_container("results"):
        st.markdown("### 5. Results")
        st.markdown('<div class="results-panel">', unsafe_allow_html=True)

        # Maintain a stable action payload shape even without rendering the buttons
        actions: Dict[str, bool] = {"adjust": False, "reset": False}

        state = get_state()

        if state.pipeline_error:
            st.error("An error occurred during the analysis")
            st.code(str(state.pipeline_error))
        elif state.report is not None:
            report: Report = state.report
            stats: RunStats = report.stats
            repetitions = report.repetitions
            metrics_df = report.metrics
            if metrics_df is not None:
                metrics_df = metrics_df.reset_index(drop=True)

            numeric_columns: List[str] = []
            if metrics_df is not None:
                numeric_candidates = [
                    c
                    for c in metrics_df.columns
                    if metrics_df[c].dtype.kind in "fi" and c != "frame_idx"
                ]
                numeric_columns = numeric_candidates
            metric_options = numeric_columns
            rep_intervals: List[Tuple[int, int]] = []
            valley_frames: List[int] = []
            frame_values = pd.Series(dtype=float)
            primary_metric: str | None = None

            st.markdown(f"**Detected repetitions:** {repetitions}")

            if stats.counting_accuracy_warning:
                st.warning(stats.counting_accuracy_warning)

            if state.preview_enabled:
                st.info("El vídeo con landmarks se visualizó durante el análisis.")

            debug_video_enabled = bool((state.configure_values or {}).get("debug_video", True))
            overlay_stream_path = getattr(report, "overlay_video_stream_path", None)
            overlay_raw_path = getattr(report, "overlay_video_path", None)
            debug_video_path = (
                overlay_stream_path or overlay_raw_path or report.debug_video_path
            )

            if metrics_df is not None:
                st.markdown('<div class="results-metrics-block">', unsafe_allow_html=True)
                rep_intervals, valley_frames, frame_values, primary_metric = _compute_rep_intervals(
                    metrics_df=metrics_df,
                    report=report,
                    stats=stats,
                    numeric_columns=(numeric_columns if numeric_columns else metric_options),
                )
                if metric_options:
                    # Prefer exercise-specific metrics for defaults
                    ex = getattr(stats, "exercise_detected", "")
                    ex_str = getattr(ex, "value", str(ex))
                    preferred_by_ex = {
                        "squat": ["left_knee", "right_knee", "trunk_inclination_deg"],
                        "bench_press": ["left_elbow", "right_elbow", "shoulder_width"],
                        "deadlift": ["left_hip", "right_hip", "trunk_inclination_deg"],
                    }
                    preferred = [m for m in preferred_by_ex.get(ex_str, []) if m in numeric_columns]
                    chosen = getattr(stats, "primary_angle", None)
                    metric_help = _build_metric_help(
                        exercise=ex_str,
                        primary_metric=chosen,
                        metric_options=metric_options,
                    )
                    if chosen and chosen in numeric_columns and chosen not in preferred:
                        preferred = [chosen] + preferred
                    default_selection = preferred[:3] or (
                        numeric_columns[:3] if numeric_columns else metric_options[:3]
                    )

                    run_sig = getattr(stats, "config_sha1", "") or ""
                    frames_val = getattr(stats, "frames", None)
                    if frames_val is not None:
                        run_sig = f"{run_sig}-{frames_val}"
                    default_key = f"metrics_default_{run_sig}"

                    if default_key not in st.session_state:
                        st.session_state[default_key] = default_selection

                    widget_key = f"metrics_multiselect_{run_sig}"

                    selected_metrics = st.multiselect(
                        "View metrics",
                        options=metric_options,
                        default=st.session_state[default_key],
                        key=widget_key,
                    )
                    if metric_help:
                        _emit_metric_help_assets()
                        _emit_metric_help_script(widget_key, metric_help)
                    if selected_metrics:
                        cfg_vals = state.configure_values or {}
                        thresholds: List[float] = []
                        for key in ("low", "high"):
                            if key not in cfg_vals:
                                continue
                            try:
                                value = float(cfg_vals[key])
                            except (TypeError, ValueError):
                                continue
                            if not math.isfinite(value):
                                continue
                            thresholds.append(value)
                        sync_channel = None
                        if getattr(stats, "config_sha1", None):
                            frames_val = getattr(stats, "frames", None)
                            if frames_val is not None:
                                sync_channel = f"vmx-sync-{stats.config_sha1}-{frames_val}"

                        render_video_with_metrics_sync(
                            video_path=None,
                            metrics_df=metrics_df,
                            selected_metrics=selected_metrics,
                            fps=stats.fps_effective,
                            rep_intervals=rep_intervals,
                            thresholds=thresholds,
                            start_at_s=None,
                            scroll_zoom=True,
                            key="results_video_metrics_sync",
                            max_width_px=720,
                            show_video=False,
                            sync_channel=sync_channel,
                        )
                    else:
                        st.info("Select at least one metric to visualize.")
                else:
                    st.info("No numeric metrics available for charting.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No metrics were generated for this run.")

            rep_speeds_df = _compute_rep_speeds(
                rep_intervals,
                stats,
                valley_frames=valley_frames,
                frame_values=frame_values,
                metrics_df=metrics_df,
                primary_metric=primary_metric,
            )
            if not rep_speeds_df.empty:
                speed_unit = rep_speeds_df.attrs.get("speed_unit", "units/s")
                down_speed_col = f"Down speed ({speed_unit})"
                up_speed_col = f"Up speed ({speed_unit})"

                st.markdown("#### Repetition speed")
                st.caption(
                    "Down = top to bottom of the rep. Up = bottom back to the top. "
                    "Speeds are computed from the primary angle so you can compare lowering and lifting tempos."
                )

                rep_chart_df = rep_speeds_df.melt(
                    id_vars=["Repetition"],
                    value_vars=["Down speed (units/s)", "Up speed (units/s)"],
                    var_name="Phase",
                    value_name="Speed",
                )
                chart = (
                    alt.Chart(rep_chart_df)
                    .mark_bar(size=28)
                    .encode(
                        x=alt.X("Repetition:O", title="Repetition"),
                        y=alt.Y("Speed:Q", title="Speed (units/s)"),
                        color=alt.Color(
                            "Phase:N",
                            scale=alt.Scale(
                                domain=["Down speed (units/s)", "Up speed (units/s)"],
                                range=["#e4572e", "#2e86de"],
                            ),
                            title="Phase",
                        ),
                        tooltip=[
                            alt.Tooltip("Repetition:O"),
                            alt.Tooltip("Phase:N", title="Phase"),
                            alt.Tooltip("Speed:Q", title="Speed (units/s)", format=".2f"),
                            alt.Tooltip("Down duration (s):Q", title="Down duration (s)", format=".2f"),
                            alt.Tooltip("Up duration (s):Q", title="Up duration (s)", format=".2f"),
                            alt.Tooltip("Cadence (reps/min):Q", title="Cadence", format=".1f"),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)

            stats_rows = [
                {"Field": "CONFIG_SHA1", "Value": stats.config_sha1},
                {"Field": "fps_original", "Value": f"{stats.fps_original:.2f}"},
                {"Field": "fps_effective", "Value": f"{stats.fps_effective:.2f}"},
                {"Field": "frames", "Value": stats.frames},
                {
                    "Field": "exercise_selected",
                    "Value": stats.exercise_selected or "N/A",
                },
                {"Field": "exercise_detected", "Value": stats.exercise_detected},
                {"Field": "view_detected", "Value": stats.view_detected},
                {
                    "Field": "detection_confidence",
                    "Value": f"{stats.detection_confidence:.0%}",
                },
                {"Field": "primary_angle", "Value": stats.primary_angle or "N/A"},
                {"Field": "angle_range_deg", "Value": f"{stats.angle_range_deg:.2f}"},
                {"Field": "min_prominence", "Value": f"{stats.min_prominence:.2f}"},
                {"Field": "min_distance_sec", "Value": f"{stats.min_distance_sec:.2f}"},
                {"Field": "refractory_sec", "Value": f"{stats.refractory_sec:.2f}"},
            ]
            stats_df = pd.DataFrame(stats_rows, columns=["Field", "Value"]).astype({"Value": "string"})

            if metrics_df is not None:
                with st.expander("Calculated metrics", expanded=False):
                    st.dataframe(metrics_df, width="stretch")

            if stats.warnings:
                st.warning("\n".join(f"• {msg}" for msg in stats.warnings))

            if stats.skip_reason:
                st.error(f"Repetition counting skipped: {stats.skip_reason}")

            eff_path = getattr(report, "effective_config_path", None)
            eff_bytes = None
            if eff_path:
                try:
                    eff_bytes = Path(eff_path).read_bytes()
                except Exception:
                    st.warning("Could not read effective config for the debug report.")

            with st.expander("Run statistics", expanded=False):
                try:
                    st.dataframe(stats_df, width="stretch")
                except Exception:
                    st.json({row["Field"]: row["Value"] for row in stats_rows})

            if debug_video_enabled and debug_video_path:
                debug_path = Path(debug_video_path)
                has_overlay = bool(overlay_stream_path or overlay_raw_path)
                download_label = (
                    "📹 Download landmark overlay video"
                    if has_overlay
                    else "Download debug video"
                )
                try:
                    debug_bytes = debug_path.read_bytes()
                except FileNotFoundError:
                    st.error("The debug video for download was not found.")
                except OSError as exc:
                    st.error(f"Could not read debug video: {exc}")
                else:
                    st.download_button(
                        download_label,
                        data=debug_bytes,
                        file_name=debug_path.name,
                        mime="video/mp4",
                        width="stretch",
                    )

            metrics_data: str | None = None
            if state.metrics_path is not None:
                try:
                    metrics_data = Path(state.metrics_path).read_text(
                        encoding="utf-8"
                    )
                except FileNotFoundError:
                    st.error("The metrics file for download was not found.")
                except OSError as exc:
                    st.error(f"Could not read metrics: {exc}")
                
            try:
                report_bundle = _build_debug_report_bundle(
                    report=report,
                    stats_df=stats_df,
                    metrics_df=metrics_df,
                    metrics_csv=metrics_data,
                    effective_config_bytes=eff_bytes,
                    video_name=Path(state.video_path).name if state.video_path else None,
                )
            except Exception as exc:
                st.error(f"Could not assemble debug report: {exc}")
            else:
                report_name = _session_name_from_paths(
                    state.metrics_path,
                    getattr(report.stats, "config_path", None),
                    getattr(report, "metrics_path", None),
                )
                if not report_name:
                    report_name = Path(state.video_path).stem if state.video_path else None
                report_name = f"{report_name}.zip" if report_name else "analysis_report.zip"
                st.download_button(
                    "📑 Download report",
                    data=report_bundle,
                    file_name=report_name,
                    mime="application/zip",
                    help=(
                        "Includes effective config, run statistics, warnings, and a"
                        " metrics snapshot to help debug issues."
                    ),
                    width="stretch",
                )

        else:
            st.info("No results found to display.")

        st.markdown("</div>", unsafe_allow_html=True)

        return actions


def _results_summary() -> None:
    with step_container("results"):
        st.markdown("### 4. Run the analysis")
        state = get_state()
        if state.last_run_success:
            st.success("Analysis complete ✅")

            report: Report | None = getattr(state, "report", None)
            stats: RunStats | None = getattr(report, "stats", None) if report else None
            if stats:
                params = _run_parameters(stats)
                if params:
                    _render_run_parameters(params)
        else:
            st.info("No results found to display.")
