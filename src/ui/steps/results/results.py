"""Vistas responsables de mostrar y exportar los resultados del análisis."""

from __future__ import annotations

import csv
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
import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None  # pragma: no cover

from src.A_preprocessing.video_metadata import get_video_metadata
from src.C_analysis.repetition_counter import count_repetitions_with_config
from src.pipeline_data import Report, RunStats
from src.ui.metrics_catalog import human_metric_name, metric_base_description
from src.ui.metrics_sync import render_video_with_metrics_sync
from src.ui.metrics_sync.run_tokens import (
    metrics_chart_key as _metrics_chart_key,
    metrics_run_token as _metrics_run_token,
    sync_channel_for_run,
)
from src.ui.state import AppState, get_state
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


def _metadata_to_csv(metadata: Dict[str, object]) -> str:
    """Serializa un diccionario de metadatos a un CSV clave-valor.

    Todos los valores se convierten a ``str`` para máxima compatibilidad; la
    versión JSON conserva los tipos nativos.
    """

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["key", "value"])
    for key in sorted(metadata.keys()):
        value = metadata[key]
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        if value is None:
            serialized = ""
        else:
            serialized = str(value)
        writer.writerow([key, serialized])
    return buffer.getvalue()


def _metadata_to_json_bytes(metadata: Dict[str, object]) -> bytes:
    """Convierte un diccionario en JSON UTF-8 con sangría, preservando tipos."""

    return json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")


def _preferred_video_base_name(
    video_original_name: str | None,
    video_path: str | Path | None,
    *,
    fallback_token: str | None = None,
) -> str | None:
    """Chooses the best available base name for artifacts.

    Preference order: original upload name (stem) -> video_path stem -> fallback token.
    """

    if video_original_name:
        try:
            stem = Path(video_original_name).stem
        except Exception:
            stem = video_original_name
        if stem:
            return stem

    if video_path:
        try:
            stem = Path(video_path).stem
        except Exception:
            stem = None
        if stem:
            return stem

    return fallback_token


def _report_bundle_file_name(
    base_name: str | None, timestamp: datetime.datetime | None = None
) -> str:
    """Builds a consistent report bundle filename using the provided base."""

    ts = timestamp or datetime.datetime.now()
    ts_str = ts.strftime("%Y_%m_%d-%H_%M")
    base = base_name or "analysis_report"
    return f"{base}-{ts_str}.zip"

def _build_debug_report_bundle(
    *,
    report: Report,
    stats_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None,
    metrics_csv: str | None,
    effective_config_bytes: bytes | None,
    video_name: str | None,
    video_path: str | Path | None = None,
    rep_intervals: List[Tuple[int, int]] | None = None,
    valley_frames: List[int] | None = None,
    rep_speeds_df: pd.DataFrame | None = None,
    rep_chart_df: pd.DataFrame | None = None,
    exercise_key: str | None = None,
    primary_metric: str | None = None,
    phase_order: Tuple[str, str] | None = None,
    interval_strategy: str | None = None,
    thresholds_used: Tuple[float, float] | None = None,
) -> bytes:
    """Package run data, config, metrics, and rep-speed artifacts into a report."""

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

    try:
        video_metadata: Dict[str, object] = (
            get_video_metadata(Path(video_path), original_name=video_name)
            if video_path
            else {}
        )
    except Exception as exc:
        video_metadata = {
            "error": str(exc),
            "timestamp_extracted_utc": datetime.datetime.utcnow().isoformat(
                timespec="seconds"
            )
            + "Z",
        }

    if not video_metadata:
        video_metadata = {
            "timestamp_extracted_utc": datetime.datetime.utcnow().isoformat(
                timespec="seconds"
            )
            + "Z",
            "video_name": video_name,
        }
    elif video_name and "video_name" not in video_metadata:
        video_metadata["video_name"] = video_name

    rotation_applied = getattr(report.stats, "rotation_applied", None)
    if rotation_applied is None or rotation_applied == 0:
        orientation_handling = "none"
    else:
        orientation_handling = f"rotate={rotation_applied} applied"

    preprocessing_decisions = {
        "orientation_handling": orientation_handling,
        "fps_effective_used": getattr(report.stats, "fps_effective", None),
        "frames_analyzed": getattr(report.stats, "frames", None),
        "sampling_strategy": getattr(report.stats, "sampling_strategy", None),
        "sample_rate": getattr(report.stats, "sample_rate", None),
        "warnings_related_to_video": list(getattr(report.stats, "warnings", [])),
    }
    if getattr(report.stats, "fps_original", None) is not None:
        preprocessing_decisions["fps_original"] = getattr(
            report.stats, "fps_original", None
        )

    video_metadata["preprocessing_decisions"] = preprocessing_decisions

    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("report.json", json.dumps(payload, indent=2, ensure_ascii=False))
        zf.writestr("run_stats.csv", stats_df.to_csv(index=False))
        zf.writestr("video_data.csv", _metadata_to_csv(video_metadata))
        zf.writestr("video_data.json", _metadata_to_json_bytes(video_metadata))
        if metrics_csv:
            zf.writestr("metrics.csv", metrics_csv)
        if effective_config:
            zf.writestr("effective_config.json", effective_config)

        rep_intervals = rep_intervals or []
        valley_frames = valley_frames or []
        rep_speeds_df = rep_speeds_df if rep_speeds_df is not None else pd.DataFrame()
        rep_chart_df = rep_chart_df if rep_chart_df is not None else pd.DataFrame()
        phase_order = phase_order or phase_order_for_exercise(exercise_key)

        def _bottom_for_interval(start: int, end: int) -> int | None:
            if not valley_frames:
                return None
            candidates = [v for v in valley_frames if start <= v <= end]
            if not candidates:
                return None
            mid = (start + end) / 2.0
            return min(candidates, key=lambda v: abs(v - mid))

        rep_interval_rows: List[Dict[str, int | None]] = []
        for i, (start_frame, end_frame) in enumerate(rep_intervals, start=1):
            rep_interval_rows.append(
                {
                    "rep": i,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "bottom_frame": _bottom_for_interval(start_frame, end_frame),
                }
            )

        intervals_df = pd.DataFrame(
            rep_interval_rows,
            columns=["rep", "start_frame", "end_frame", "bottom_frame"],
        )

        zf.writestr("rep_intervals.csv", intervals_df.to_csv(index=False))
        zf.writestr("rep_speeds.csv", rep_speeds_df.to_csv(index=False))
        zf.writestr("rep_speed_long.csv", rep_chart_df.to_csv(index=False))

        speed_meta = {
            "exercise_key": exercise_key,
            "primary_metric": primary_metric,
            "phase_order": list(phase_order or ()),
            "expected_reps": getattr(report, "repetitions", None),
            "interval_count": len(rep_intervals),
            "valley_frames": valley_frames,
            "interval_strategy": interval_strategy,
            "thresholds_used":
                {"lower": thresholds_used[0], "upper": thresholds_used[1]}
                if thresholds_used
                else None,
        }

        zf.writestr(
            "rep_speed_meta.json", json.dumps(speed_meta, indent=2, ensure_ascii=False)
        )

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
        return text.replace(/\\s+/g, " ").trim();
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
    *,
    exercise_key: str | None,
) -> Tuple[
    List[Tuple[int, int]],
    List[int],
    pd.Series,
    str | None,
    str | None,
    Tuple[float, float] | None,
    list[dict],
]:
    if metrics_df.empty:
        return [], [], pd.Series(dtype=float), None, None, None, []

    # Preserve frame_idx for time mapping but operate on a stable positional index
    try:
        metrics_df = metrics_df.reset_index(drop=True)
    except Exception:
        metrics_df = metrics_df.copy()

    fps = float(getattr(stats, "fps_effective", 0.0) or 0.0)
    fps = fps if fps > 0 else 1.0

    if "frame_idx" in metrics_df.columns:
        frame_values = pd.to_numeric(metrics_df["frame_idx"], errors="coerce")
    else:
        frame_values = pd.Series(metrics_df.index.to_numpy(), index=metrics_df.index)
    frame_values = frame_values.where(~frame_values.isna(), pd.Series(range(len(metrics_df)), index=metrics_df.index))

    if frame_values.empty:
        return [], [], frame_values, None, None, None, []

    first_frame = int(frame_values.iloc[0])
    last_frame = int(frame_values.iloc[-1]) if len(frame_values) > 1 else first_frame + 1

    candidate = getattr(stats, "primary_angle", None)
    if not candidate or candidate not in metrics_df.columns:
        for fb in ("left_knee", "right_knee"):
            if fb in metrics_df.columns:
                candidate = fb
                break
    if not candidate or candidate not in metrics_df.columns:
        candidate = numeric_columns[0] if numeric_columns else None
    if not candidate or candidate not in metrics_df.columns:
        return [], [], frame_values, None, None, None, []

    valley_indices: List[int] = []
    rep_intervals_by_index: List[Tuple[int, int]] = []
    rep_candidates_data: list[dict] = []
    config_used = getattr(report, "config_used", None)
    counting_cfg = getattr(config_used, "counting", None)
    faults_cfg = getattr(config_used, "faults", None)
    if counting_cfg is not None:
        try:
            _, debug = count_repetitions_with_config(
                metrics_df, counting_cfg, fps, faults_cfg=faults_cfg
            )
            valley_indices = list(getattr(debug, "valley_indices", []))
            rep_intervals_by_index = list(getattr(debug, "rep_intervals", []) or [])
            rep_candidates_data = list(getattr(debug, "rep_candidates", []) or [])
            if not rep_intervals_by_index:
                rep_starts = getattr(debug, "rep_start_frames", None)
                rep_ends = getattr(debug, "rep_end_frames", None)
                if rep_starts is not None and rep_ends is not None:
                    rep_intervals_by_index = list(zip(rep_starts, rep_ends))
        except Exception:
            valley_indices = []
            rep_intervals_by_index = []

    if not valley_indices and find_peaks is not None:
        series = pd.to_numeric(metrics_df[candidate], errors="coerce")
        series_interp = series.interpolate(limit_direction="both")
        if series_interp.size:
            prominence = float(getattr(stats, "min_prominence", 0.0) or 0.0)
            prominence_param = None if prominence <= 0 else prominence
            distance_sec = float(getattr(stats, "min_distance_sec", 0.0) or 0.0)
            distance_frames = max(1, int(round(distance_sec * fps)))
            valleys, _ = find_peaks(
                -series_interp.to_numpy(), prominence=prominence_param, distance=distance_frames
            )
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

    frame_count = len(frame_values)
    frame_set = set(
        pd.to_numeric(frame_values, errors="coerce").dropna().astype(int).tolist()
    )
    try:
        valley_indices = [int(idx) for idx in valley_indices]
    except Exception:
        valley_indices = []
    valley_indices = sorted(dict.fromkeys(valley_indices))

    series_interp_full = pd.to_numeric(metrics_df[candidate], errors="coerce").interpolate(
        limit_direction="both"
    )

    exercise = (exercise_key or "").lower()

    if not valley_indices and not rep_intervals_by_index and exercise != "deadlift":
        return [], [], frame_values, candidate, None, None, []

    def _frame_at_index(pos: int) -> int:
        pos = min(max(pos, 0), frame_count - 1)
        try:
            return int(frame_values.iloc[pos])
        except Exception:
            return int(frame_values.to_numpy()[pos])

    def _to_frame(v: int) -> int:
        v = int(v)
        if v in frame_set:
            return v
        if 0 <= v < frame_count:
            return _frame_at_index(v)
        return v

    valley_frames = [_to_frame(idx) for idx in valley_indices]
    valley_frames = sorted(dict.fromkeys(valley_frames))

    accepted_candidates = [
        c
        for c in rep_candidates_data
        if c.get("accepted") is True and c.get("end_frame") is not None
    ]
    accepted_candidates = sorted(
        accepted_candidates, key=lambda c: c.get("start_frame") if c.get("start_frame") is not None else 0
    )

    expected_reps_raw = getattr(report, "repetitions", None)
    expected_reps = (
        int(expected_reps_raw)
        if isinstance(expected_reps_raw, (int, float)) and expected_reps_raw > 0
        else None
    )
    raw_expected = getattr(stats, "reps_detected_raw", None)
    target_reps = expected_reps if expected_reps is not None else None
    if target_reps is None:
        if accepted_candidates:
            target_reps = len(accepted_candidates)
        elif rep_candidates_data:
            target_reps = len(rep_candidates_data)
        elif raw_expected:
            target_reps = raw_expected

    if target_reps is not None and accepted_candidates and len(accepted_candidates) > target_reps:
        accepted_candidates = accepted_candidates[:target_reps]

    intervals: List[Tuple[int, int]] = []
    interval_strategy: str | None = None
    thresholds_used: Tuple[float, float] | None = None

    def _midpoint_intervals_from_valleys() -> List[Tuple[int, int]]:
        intervals_from_valleys: List[Tuple[int, int]] = []
        if not valley_frames:
            return intervals_from_valleys

        total_frames = last_frame
        if total_frames <= first_frame:
            total_frames = first_frame + 1

        def _midpoint(a: int, b: int) -> int:
            return int((a + b) / 2)

        for i, valley_frame in enumerate(valley_frames):
            if i == 0:
                start_frame = first_frame
            else:
                start_frame = _midpoint(valley_frames[i - 1], valley_frame)

            if i == len(valley_frames) - 1:
                end_frame = total_frames
            else:
                end_frame = _midpoint(valley_frame, valley_frames[i + 1])

            end_frame = max(start_frame + 1, end_frame)
            intervals_from_valleys.append((start_frame, end_frame))

        return intervals_from_valleys

    if exercise == "deadlift":
        if rep_intervals_by_index:
            for start_idx, end_idx in rep_intervals_by_index:
                start_frame = _to_frame(start_idx)
                end_frame = _to_frame(end_idx)
                end_frame = max(start_frame + 1, end_frame)
                intervals.append((start_frame, end_frame))
            interval_strategy = "debug_intervals"
        else:
            lower_threshold = getattr(stats, "lower_threshold", None)
            upper_threshold = getattr(stats, "upper_threshold", None)
            if lower_threshold is None and counting_cfg is not None:
                lower_threshold = getattr(counting_cfg, "lower_threshold", None)
            if upper_threshold is None and counting_cfg is not None:
                upper_threshold = getattr(counting_cfg, "upper_threshold", None)
            if lower_threshold is None and faults_cfg is not None:
                lower_threshold = getattr(faults_cfg, "low_thresh", None)
            if upper_threshold is None and faults_cfg is not None:
                upper_threshold = getattr(faults_cfg, "high_thresh", None)
            if lower_threshold is None and counting_cfg is not None:
                lower_threshold = getattr(counting_cfg, "lower_thresh", None)
            if upper_threshold is None and counting_cfg is not None:
                upper_threshold = getattr(counting_cfg, "upper_thresh", None)
            if lower_threshold is None and faults_cfg is not None:
                lower_threshold = getattr(faults_cfg, "lower_thresh", None)
            if upper_threshold is None and faults_cfg is not None:
                upper_threshold = getattr(faults_cfg, "upper_thresh", None)

            def _threshold_intervals() -> List[Tuple[int, int]]:
                nonlocal thresholds_used
                if lower_threshold is None or upper_threshold is None:
                    return []
                try:
                    lower_val = float(lower_threshold)
                    upper_val = float(upper_threshold)
                except Exception:
                    return []
                if not math.isfinite(lower_val) or not math.isfinite(upper_val):
                    return []
                thresholds_used = (lower_val, upper_val)
                intervals_from_thresholds: List[Tuple[int, int]] = []
                start_idx: int | None = None
                above_seen = False
                for idx, val in series_interp_full.items():
                    if not math.isfinite(val):
                        continue
                    if val <= lower_val:
                        if start_idx is None:
                            start_idx = idx
                        if above_seen and start_idx is not None:
                            end_idx = idx
                            start_frame = _to_frame(start_idx)
                            end_frame = _to_frame(end_idx)
                            end_frame = max(start_frame + 1, end_frame)
                            intervals_from_thresholds.append((start_frame, end_frame))
                            start_idx = idx
                            above_seen = False
                    elif start_idx is not None and val >= upper_val:
                        above_seen = True
                return intervals_from_thresholds

            intervals = _threshold_intervals()

            if intervals:
                interval_strategy = "thresholds"

            if not intervals and valley_indices:
                first_valley_idx = valley_indices[0]
                if first_valley_idx > 0:
                    prefix = series_interp_full.iloc[: first_valley_idx + 1]
                    if not prefix.empty and not prefix.isna().all():
                        inferred_idx = int(prefix.idxmin())
                        if inferred_idx < first_valley_idx:
                            valley_indices = [inferred_idx] + valley_indices
                            valley_frames = [_to_frame(inferred_idx)] + valley_frames
                last_valley_idx = valley_indices[-1]
                if last_valley_idx < frame_count - 1:
                    suffix = series_interp_full.iloc[last_valley_idx:]
                    if not suffix.empty and not suffix.isna().all():
                        inferred_end_idx = int(suffix.idxmin())
                        if inferred_end_idx > last_valley_idx:
                            valley_indices = valley_indices + [inferred_end_idx]
                            valley_frames = valley_frames + [_to_frame(inferred_end_idx)]

                valley_indices = sorted(dict.fromkeys(valley_indices))
                valley_frames = sorted(dict.fromkeys(valley_frames))

                for i in range(len(valley_indices) - 1):
                    start_idx = valley_indices[i]
                    end_idx = valley_indices[i + 1]
                    start_frame = _to_frame(start_idx)
                    end_frame = _to_frame(end_idx)
                    end_frame = max(start_frame + 1, end_frame)
                    intervals.append((start_frame, end_frame))

                if intervals:
                    interval_strategy = "valley_pairs"
    else:
        midpoint_intervals = _midpoint_intervals_from_valleys()
        debug_intervals: List[Tuple[int, int]] = []

        if rep_intervals_by_index:
            for start_idx, end_idx in rep_intervals_by_index:
                start_frame = _to_frame(start_idx)
                end_frame = _to_frame(end_idx)
                end_frame = max(start_frame + 1, end_frame)
                debug_intervals.append((start_frame, end_frame))

        if rep_intervals_by_index and expected_reps is not None:
            if len(debug_intervals) < expected_reps and len(midpoint_intervals) == expected_reps:
                intervals = midpoint_intervals
                interval_strategy = "midpoint_valleys"
            else:
                intervals = debug_intervals
                interval_strategy = "debug_intervals"
        elif rep_intervals_by_index:
            intervals = debug_intervals
            interval_strategy = "debug_intervals"
        else:
            intervals = midpoint_intervals
            interval_strategy = "midpoint_valleys"

    if target_reps is not None:
        if len(intervals) > target_reps:
            intervals = intervals[:target_reps]
        elif len(intervals) < target_reps:
            warning_msg = (
                f"Repetition speed uses {len(intervals)} intervals because only {len(valley_indices)} bottoms were detected; "
                "some reps may be missing."
            )
            try:
                if not getattr(stats, "counting_accuracy_warning", None):
                    stats.counting_accuracy_warning = warning_msg
            except Exception:
                pass

    return (
        intervals,
        valley_frames,
        frame_values,
        candidate,
        interval_strategy,
        thresholds_used,
        accepted_candidates,
    )


def _metric_extreme(metric_name: str | None) -> str:
    """Return the expected "bottom" extreme for the metric ("min" or "max")."""

    if not metric_name:
        return "min"

    name = metric_name.lower()
    if "trunk_inclination" in name or name.startswith("trunk_"):
        return "max"
    return "min"


def normalize_exercise_key(exercise_key: str | None) -> str:
    """Return a lowercase exercise key, tolerating missing or non-string inputs."""

    try:
        return str(exercise_key or "").lower()
    except Exception:
        return ""


def _compute_rep_speeds(
    rep_intervals: List[Tuple[int, int]],
    stats: RunStats,
    *,
    exercise_key: str | None,
    valley_frames: List[int],
    frame_values: pd.Series,
    metrics_df: pd.DataFrame | None,
    primary_metric: str | None,
    rep_candidates: list[dict] | None = None,
) -> pd.DataFrame:
    """Estimate per-rep cadence, duration, and phase speeds from frame intervals."""

    exercise_normalized = normalize_exercise_key(exercise_key)

    if not rep_intervals and not rep_candidates:
        return pd.DataFrame()

    try:
        metrics_df = metrics_df.reset_index(drop=True) if metrics_df is not None else None
    except Exception:
        metrics_df = metrics_df.copy() if metrics_df is not None else None

    fps = float(getattr(stats, "fps_effective", 0.0) or 0.0)
    fps = fps if fps > 0 else 1.0

    if metrics_df is None or metrics_df.empty or primary_metric not in metrics_df.columns:
        metrics_series = None
        metrics_interp = None
    else:
        metrics_series = pd.to_numeric(metrics_df[primary_metric], errors="coerce")
        metrics_interp = metrics_series.interpolate(limit_direction="both")

    pose_ok = None
    if metrics_df is not None and "pose_ok" in metrics_df.columns:
        try:
            pose_ok = pd.to_numeric(metrics_df["pose_ok"], errors="coerce").to_numpy(
                copy=False
            )
        except Exception:
            pose_ok = None

    frame_arr = frame_values.to_numpy(copy=False)
    raw_values = metrics_series.to_numpy(copy=False) if metrics_series is not None else None

    def _frame_from_index(idx: int | None) -> int | None:
        if idx is None:
            return None
        try:
            return int(frame_values.iloc[int(idx)])
        except Exception:
            try:
                pos = min(max(int(idx), 0), len(frame_values) - 1)
                return int(frame_values.to_numpy()[pos])
            except Exception:
                return int(idx)

    def _metric_at_frame(frame: int) -> float:
        if frame_values.empty:
            return math.nan
        if metrics_interp is None and metrics_series is None:
            return math.nan
        try:
            nearest_pos = int(np.nanargmin(np.abs(frame_arr - frame)))
        except Exception:
            return math.nan
        series_for_lookup = metrics_interp if metrics_interp is not None else metrics_series
        try:
            return float(series_for_lookup.iloc[nearest_pos])
        except Exception:
            return math.nan

    def _nearest_valid_frame(frame: int, start: int, end: int) -> int:
        if raw_values is None or not len(raw_values) or frame_values.empty:
            return frame
        try:
            interval_mask = (frame_arr >= start) & (frame_arr <= end)
        except Exception:
            return frame
        if not interval_mask.any():
            return frame
        valid_mask = interval_mask & np.isfinite(raw_values)
        if pose_ok is not None:
            try:
                valid_mask = valid_mask & (pose_ok >= 0.5)
            except Exception:
                pass
        if not valid_mask.any():
            return frame
        candidate_positions = np.nonzero(valid_mask)[0]
        if candidate_positions.size == 0:
            return frame
        candidate_frames = frame_arr[candidate_positions]
        finite_mask = np.isfinite(candidate_frames)
        if not finite_mask.any():
            return frame
        candidate_positions = candidate_positions[finite_mask]
        candidate_frames = candidate_frames[finite_mask]
        try:
            pos = int(candidate_positions[np.nanargmin(np.abs(candidate_frames - frame))])
            return int(frame_arr[pos])
        except Exception:
            return frame

    first_phase_label, second_phase_label = (
        ("Up", "Down") if exercise_normalized == "deadlift" else ("Down", "Up")
    )
    bottom_extreme = _metric_extreme(primary_metric)

    rows: List[Dict[str, float]] = []

    candidates_source: list[dict] = []
    if rep_candidates:
        candidates_source = [
            c
            for c in rep_candidates
            if c.get("accepted") is True and c.get("end_frame") is not None
        ]

    if candidates_source:
        iterable = sorted(
            candidates_source,
            key=lambda c: (
                c.get("start_frame")
                if c.get("start_frame") is not None
                else c.get("rep_index", 0)
            ),
        )
    else:
        iterable = [
            {
                "rep_index": i - 1,
                "start_frame": start,
                "end_frame": end,
                "turning_frame": None,
                "accepted": True,
                "rejection_reason": "NONE",
            }
            for i, (start, end) in enumerate(rep_intervals, start=1)
        ]

    for i, entry in enumerate(iterable, start=1):
        start_frame_idx = entry.get("start_frame")
        end_frame_idx = entry.get("end_frame")
        turning_frame_idx = entry.get("turning_frame")

        start_frame = _frame_from_index(start_frame_idx)
        end_frame = _frame_from_index(end_frame_idx)
        turning_frame = _frame_from_index(turning_frame_idx)

        accepted = bool(entry.get("accepted", True))
        rejection_reason = str(entry.get("rejection_reason", "NONE"))
        rep_number = i

        if start_frame is None or end_frame is None or end_frame <= start_frame:
            duration_s = math.nan
            cadence_rps = 0.0
            start_frame = start_frame if start_frame is not None else 0
            end_frame = end_frame if end_frame is not None else start_frame
            missing_data = True
            first_duration = second_duration = math.nan
            turning_frame = turning_frame if turning_frame is not None else start_frame
            first_speed = second_speed = math.nan
        else:
            span_frames = max(1, end_frame - start_frame)
            duration_s = span_frames / fps
            cadence_rps = (1.0 / duration_s) if duration_s > 0 else 0.0

            def _turning_point(
                start: int, end: int, *, use_max: bool
            ) -> Tuple[int, float]:
                if metrics_interp is None or frame_values.empty:
                    return start, math.nan
                mask = (frame_values >= start) & (frame_values <= end)
                try:
                    masked = metrics_interp[mask]
                except Exception:
                    masked = metrics_interp.iloc[0:0]
                if masked.empty or masked.isna().all():
                    return start, math.nan
                idx = masked.idxmax() if use_max else masked.idxmin()
                try:
                    frame_at_turn = int(frame_values.loc[idx])
                except Exception:
                    try:
                        pos = metrics_interp.index.get_loc(idx)
                        frame_at_turn = int(frame_values.iloc[pos])
                    except Exception:
                        frame_at_turn = start
                frame_at_turn = min(max(frame_at_turn, start), end)
                return frame_at_turn, float(masked.loc[idx])

            use_max_turning = (
                bottom_extreme == "min"
                if exercise_normalized == "deadlift"
                else bottom_extreme == "max"
            )
            turning_frame, turning_val = _turning_point(
                start_frame, end_frame, use_max=use_max_turning
            )

            start_sample_frame = _nearest_valid_frame(start_frame, start_frame, turning_frame)
            end_sample_frame = _nearest_valid_frame(end_frame, turning_frame, end_frame)

            start_val = _metric_at_frame(start_sample_frame)
            end_val = _metric_at_frame(end_sample_frame)

            first_phase_frames = max(1, turning_frame - start_frame)
            second_phase_frames = max(1, end_frame - turning_frame)

            first_duration = first_phase_frames / fps
            second_duration = second_phase_frames / fps

            if any(math.isnan(v) for v in (start_val, turning_val)):
                first_speed = math.nan
            else:
                first_speed = (
                    abs(start_val - turning_val) / first_duration if first_duration > 0 else math.nan
                )

            if any(math.isnan(v) for v in (end_val, turning_val)):
                second_speed = math.nan
            else:
                second_speed = (
                    abs(end_val - turning_val) / second_duration if second_duration > 0 else math.nan
                )

            missing_data = math.isnan(first_speed) or math.isnan(second_speed)

        if first_phase_label == "Up":
            up_duration = first_duration
            up_speed = first_speed
            down_duration = second_duration
            down_speed = second_speed
        else:
            down_duration = first_duration
            down_speed = first_speed
            up_duration = second_duration
            up_speed = second_speed

        missing_data = missing_data or not math.isfinite(duration_s)
        if missing_data:
            down_speed = 0.0 if not math.isfinite(down_speed) else down_speed
            up_speed = 0.0 if not math.isfinite(up_speed) else up_speed
            down_duration = 0.0 if not math.isfinite(down_duration) else down_duration
            up_duration = 0.0 if not math.isfinite(up_duration) else up_duration

        rows.append(
            {
                "Repetition": rep_number,
                "Start frame": float(start_frame),
                "End frame": float(end_frame),
                "Duration (s)": duration_s,
                "Cadence (reps/min)": cadence_rps * 60.0,
                "Bottom frame": float(turning_frame),
                "Down duration (s)": down_duration,
                "Up duration (s)": up_duration,
                "Down speed (deg/s)": down_speed,
                "Up speed (deg/s)": up_speed,
                "Missing data": missing_data,
                "Accepted": accepted,
                "Rejection reason": rejection_reason,
            }
        )

    return pd.DataFrame(rows)


def _exercise_key_from_report(report: Report | None) -> str | None:
    """Best-effort extraction of the exercise key for the current run."""

    if report is None:
        return None

    try:
        counting_cfg = getattr(report.config_used, "counting", None)
        exercise_from_config = getattr(counting_cfg, "exercise", None)
        if exercise_from_config:
            return str(exercise_from_config)
    except Exception:
        pass

    stats = getattr(report, "stats", None)
    exercise_detected = getattr(stats, "exercise_detected", None)
    if exercise_detected:
        return str(getattr(exercise_detected, "value", exercise_detected))

    return None


def phase_order_for_exercise(exercise_key: str | None) -> Tuple[str, str]:
    """Return the label order for the two phases of a repetition."""

    if str(exercise_key).lower() == "deadlift":
        return ("Up", "Down")
    return ("Down", "Up")


def _build_rep_speed_chart_df(
    rep_speeds_df: pd.DataFrame, phase_order: Tuple[str, str]
) -> pd.DataFrame:
    """Reshape per-repetition speeds for charting with the requested phase order."""

    if rep_speeds_df.empty:
        return pd.DataFrame(columns=["Repetition", "Speed", "Phase duration (s)", "Phase"])

    def _phase_cols(label: str) -> Tuple[str, str]:
        if label == "Up":
            return ("Up speed (deg/s)", "Up duration (s)")
        return ("Down speed (deg/s)", "Down duration (s)")

    phase_frames = []
    base_cols = ["Repetition", "Cadence (reps/min)"]
    if "Accepted" in rep_speeds_df.columns:
        base_cols.append("Accepted")
    if "Rejection reason" in rep_speeds_df.columns:
        base_cols.append("Rejection reason")
    if "Missing data" in rep_speeds_df.columns:
        base_cols.append("Missing data")

    for label in phase_order:
        speed_col, duration_col = _phase_cols(label)
        use_cols = base_cols + [speed_col, duration_col]
        phase_frames.append(
            rep_speeds_df[use_cols]
            .rename(columns={speed_col: "Speed", duration_col: "Phase duration (s)"})
            .assign(Phase=label)
        )

    return pd.concat(phase_frames, ignore_index=True)


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
            exercise_key = _exercise_key_from_report(report)
            repetitions = report.repetitions
            metrics_df_raw = report.metrics
            try:
                metrics_df = metrics_df_raw.reset_index(drop=True) if metrics_df_raw is not None else None
            except Exception:
                metrics_df = metrics_df_raw

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
            interval_strategy: str | None = None
            thresholds_used: Tuple[float, float] | None = None
            rep_candidates_data: list[dict] = []
            phase_order = phase_order_for_exercise(exercise_key)
            rep_chart_df = pd.DataFrame()

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
            run_token = _metrics_run_token(state, stats)
            sync_channel = sync_channel_for_run(run_token)

            if metrics_df is not None:
                st.markdown('<div class="results-metrics-block">', unsafe_allow_html=True)
                (
                    rep_intervals,
                    valley_frames,
                    frame_values,
                    primary_metric,
                    interval_strategy,
                    thresholds_used,
                    rep_candidates_data,
                ) = _compute_rep_intervals(
                    metrics_df=metrics_df,
                    report=report,
                    stats=stats,
                    numeric_columns=(numeric_columns if numeric_columns else metric_options),
                    exercise_key=exercise_key,
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

                        fps_for_sync = getattr(stats, "fps_original", None) or getattr(
                            stats, "fps_effective", 0.0
                        )
                        render_video_with_metrics_sync(
                            video_path=None,
                            metrics_df=metrics_df,
                            selected_metrics=selected_metrics,
                            fps=fps_for_sync,
                            rep_intervals=rep_intervals,
                            thresholds=thresholds,
                            start_at_s=None,
                            scroll_zoom=True,
                            key=_metrics_chart_key(run_token),
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
                exercise_key=exercise_key,
                valley_frames=valley_frames,
                frame_values=frame_values,
                metrics_df=metrics_df,
                primary_metric=primary_metric,
                rep_candidates=rep_candidates_data,
            )
            rep_chart_df = _build_rep_speed_chart_df(rep_speeds_df, phase_order)

            if not rep_speeds_df.empty:
                st.markdown("#### Repetition speed")
                if phase_order[0] == "Up":
                    caption = (
                        "Up = bottom back to the top of the rep. Down = top to bottom. "
                        "Angular speeds (deg/s) are derived from the primary angle, so linear m/s values "
                        "aren't available with the normalized pose coordinates."
                    )
                else:
                    caption = (
                        "Down = top to bottom of the rep. Up = bottom back to the top. "
                        "Angular speeds (deg/s) are derived from the primary angle, so linear m/s values "
                        "aren't available with the normalized pose coordinates."
                    )
                st.caption(caption)

                rep_chart_df = _build_rep_speed_chart_df(rep_speeds_df, phase_order)
                chart = (
                    alt.Chart(rep_chart_df)
                    .mark_bar(size=28)
                    .encode(
                        x=alt.X("Repetition:O", title="Repetition"),
                        y=alt.Y("Speed:Q", title="Angular speed (deg/s)"),
                        color=alt.Color(
                            "Phase:N",
                            scale=alt.Scale(
                                domain=list(phase_order),
                                range=[
                                    "#2e86de" if phase_order[0] == "Up" else "#e4572e",
                                    "#e4572e" if phase_order[1] == "Down" else "#2e86de",
                                ],
                            ),
                            title="Phase",
                        ),
                        tooltip=[
                            alt.Tooltip("Repetition:O"),
                            alt.Tooltip("Phase:N", title="Phase"),
                            alt.Tooltip("Speed:Q", title="Speed (deg/s)", format=".2f"),
                            alt.Tooltip(
                                "Phase duration (s):Q", title="Phase duration (s)", format=".2f"
                            ),
                            alt.Tooltip("Cadence (reps/min):Q", title="Cadence", format=".1f"),
                            alt.Tooltip("Accepted:N", title="Accepted"),
                            alt.Tooltip("Rejection reason:N", title="Rejection"),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, width='stretch')

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
                
            video_original_name = getattr(state, "video_original_name", None)
            video_name_for_report = video_original_name
            if not video_name_for_report and state.video_path:
                try:
                    video_name_for_report = Path(state.video_path).name
                except Exception:
                    video_name_for_report = None

            try:
                report_bundle = _build_debug_report_bundle(
                    report=report,
                    stats_df=stats_df,
                    metrics_df=metrics_df,
                    metrics_csv=metrics_data,
                    effective_config_bytes=eff_bytes,
                    video_name=video_name_for_report,
                    video_path=Path(state.video_path) if state.video_path else None,
                    rep_intervals=rep_intervals,
                    valley_frames=valley_frames,
                    rep_speeds_df=rep_speeds_df,
                    rep_chart_df=rep_chart_df,
                    exercise_key=exercise_key,
                    primary_metric=primary_metric,
                    phase_order=phase_order,
                    interval_strategy=interval_strategy,
                    thresholds_used=thresholds_used,
                )
            except Exception as exc:
                st.error(f"Could not assemble debug report: {exc}")
            else:
                base_name = _preferred_video_base_name(
                    video_original_name,
                    state.video_path,
                    fallback_token=run_token,
                )
                report_name = _report_bundle_file_name(base_name)
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
