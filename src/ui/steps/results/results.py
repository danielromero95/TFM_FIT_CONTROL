"""Vistas responsables de mostrar y exportar los resultados del análisis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None  # pragma: no cover

from src.C_analysis.repetition_counter import count_repetitions_with_config
from src.pipeline_data import Report, RunStats
from src.ui.metrics_sync import render_video_with_metrics_sync
from src.ui.state import get_state
from ..utils import step_container


_METRIC_HELP_CSS_EMITTED_KEY = "_metric_help_css_emitted"


def _humanize_metric_name(metric: str) -> str:
    base_names = {
        "left_knee": "left knee angle",
        "right_knee": "right knee angle",
        "left_elbow": "left elbow angle",
        "right_elbow": "right elbow angle",
        "left_hip": "left hip angle",
        "right_hip": "right hip angle",
        "trunk_inclination_deg": "trunk inclination",
        "shoulder_width": "shoulder width",
        "foot_separation": "foot separation",
        "knee_symmetry": "knee symmetry",
        "elbow_symmetry": "elbow symmetry",
    }
    if metric in base_names:
        return base_names[metric]
    if metric.startswith("raw_"):
        return f"raw {metric[4:].replace('_', ' ')}"
    if metric.startswith("ang_vel_"):
        return f"angular velocity of {metric[8:].replace('_', ' ')}"
    return metric.replace("_", " ")


def _metric_base_description(metric: str, exercise: str) -> str | None:
    exercise = exercise or ""
    descriptions = {
        "left_knee": {
            "squat": "Left knee flexion angle that captures squat depth.",
            "bench_press": "Left knee flexion angle recorded to show leg drive stability during bench press.",
            "deadlift": "Left knee flexion angle illustrating the pull setup and lockout.",
            "default": "Left knee flexion angle measured at the hip–knee–ankle joints.",
        },
        "right_knee": {
            "squat": "Right knee flexion angle mirroring squat depth on the working side.",
            "bench_press": "Right knee flexion angle captured for leg drive monitoring during bench press.",
            "deadlift": "Right knee flexion angle that reflects the pull stance and lockout.",
            "default": "Right knee flexion angle measured at the hip–knee–ankle joints.",
        },
        "left_elbow": {
            "squat": "Left elbow flexion angle, useful for spotting arm movement during squats.",
            "bench_press": "Left elbow flexion angle tracing press depth and lockout.",
            "deadlift": "Left elbow flexion angle, helpful for verifying straight arms during the pull.",
            "default": "Left elbow flexion angle measured at the shoulder–elbow–wrist joints.",
        },
        "right_elbow": {
            "squat": "Right elbow flexion angle for checking upper-body posture during squats.",
            "bench_press": "Right elbow flexion angle monitoring press depth and lockout symmetry.",
            "deadlift": "Right elbow flexion angle confirming arm extension during the pull.",
            "default": "Right elbow flexion angle measured at the shoulder–elbow–wrist joints.",
        },
        "left_hip": {
            "squat": "Left hip hinge angle complementing the view of squat depth.",
            "bench_press": "Left hip hinge angle that shows lower-body tension on the bench.",
            "deadlift": "Left hip hinge angle measuring the deadlift setup and lockout.",
            "default": "Left hip hinge angle measured at the shoulder–hip–knee joints.",
        },
        "right_hip": {
            "squat": "Right hip hinge angle mirroring the squat descent and ascent.",
            "bench_press": "Right hip hinge angle indicating leg drive while benching.",
            "deadlift": "Right hip hinge angle capturing hip extension through the deadlift.",
            "default": "Right hip hinge angle measured at the shoulder–hip–knee joints.",
        },
        "trunk_inclination_deg": {
            "squat": "Torso inclination relative to the hips, showing forward lean in the squat.",
            "bench_press": "Torso inclination relative to the hips, highlighting arch control on the bench.",
            "deadlift": "Torso inclination relative to the hips, indicating back angle in the pull.",
            "default": "Torso inclination relative to the hips expressed in degrees.",
        },
        "shoulder_width": {
            "squat": "Horizontal distance between shoulders (normalized), reflecting upper-body stance.",
            "bench_press": "Horizontal distance between shoulders (normalized), tracking shoulder width on the bench.",
            "deadlift": "Horizontal distance between shoulders (normalized), confirming back tightness in the pull setup.",
            "default": "Horizontal distance between shoulders in normalized screen units.",
        },
        "foot_separation": {
            "squat": "Horizontal distance between ankles (normalized), showing squat stance width.",
            "bench_press": "Horizontal distance between ankles (normalized), showing bench foot placement.",
            "deadlift": "Horizontal distance between ankles (normalized), showing deadlift stance width.",
            "default": "Horizontal distance between ankles in normalized screen units.",
        },
        "knee_symmetry": {
            "squat": "Symmetry score between left and right knee angles (1 means perfectly matched).",
            "bench_press": "Symmetry score between left and right knee angles to monitor lower-body balance on the bench.",
            "deadlift": "Symmetry score between left and right knee angles to verify even pull mechanics.",
            "default": "Symmetry score between knee angles where 1 indicates identical motion.",
        },
        "elbow_symmetry": {
            "squat": "Symmetry score between left and right elbow angles (1 means perfectly matched).",
            "bench_press": "Symmetry score between left and right elbow angles to track pressing balance.",
            "deadlift": "Symmetry score between left and right elbow angles to confirm straight-arm symmetry.",
            "default": "Symmetry score between elbow angles where 1 indicates identical motion.",
        },
    }
    base = descriptions.get(metric, {})
    if exercise in base:
        return base[exercise]
    return base.get("default", None)


def _counting_relation_text(
    metric: str,
    exercise: str,
    primary_metric: str | None,
    *,
    is_primary: bool,
) -> str:
    primary_label = _humanize_metric_name(primary_metric) if primary_metric else "the auto-selected primary angle"
    primary_candidates = {
        "squat": {"left_knee", "right_knee"},
        "bench_press": {"left_elbow", "right_elbow"},
        "deadlift": {"left_hip", "right_hip"},
    }
    if is_primary:
        return "Reps are counted when this angle dips below the lower threshold and then rises past the upper threshold."
    if primary_metric:
        if metric in primary_candidates.get(exercise, set()):
            return f"Not used for counting in this run; reps are detected from the {primary_label} crossing the configured thresholds."
        return f"Not used for counting; reps are detected from the {primary_label} crossing the configured thresholds."
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

        if metric.startswith("raw_"):
            source_metric = metric[4:]
            human_label = _humanize_metric_name(source_metric)
            base_desc = f"Raw (unfiltered) {human_label} values straight from pose detection."
            relation = _counting_relation_text(source_metric, exercise, primary_metric, is_primary=False)
        elif metric.startswith("ang_vel_"):
            source_metric = metric[8:]
            human_label = _humanize_metric_name(source_metric)
            base_desc = f"Angular velocity of the {human_label} in degrees per second."
            relation = _counting_relation_text(source_metric, exercise, primary_metric, is_primary=False)
        else:
            base_desc = _metric_base_description(metric, exercise)
            if base_desc is None:
                continue
            relation = _counting_relation_text(metric, exercise, primary_metric, is_primary=(metric == primary_metric))

        body = f"{prefix}{base_desc} {relation}".strip()
        title = " ".join(body.split())
        help_map[metric] = {"title": title, "body": body}

    return help_map


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
) -> List[Tuple[int, int]]:
    if metrics_df.empty:
        return []

    fps = float(getattr(stats, "fps_effective", 0.0) or 0.0)
    fps = fps if fps > 0 else 1.0

    df = metrics_df.reset_index(drop=True)
    if "frame_idx" in df.columns:
        frame_values = pd.to_numeric(df["frame_idx"], errors="coerce")
        frame_values = frame_values.where(~frame_values.isna(), pd.Series(range(len(df))))
    else:
        frame_values = pd.Series(range(len(df)))

    if frame_values.empty:
        return []

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
        return []

    valley_indices: List[int] = []
    counting_cfg = getattr(getattr(report, "config_used", None), "counting", None)
    if counting_cfg is not None:
        try:
            _, debug = count_repetitions_with_config(df, counting_cfg, fps)
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
        return []

    frame_count = len(frame_values)
    valley_frames = []
    for idx in valley_indices:
        pos = min(max(idx, 0), frame_count - 1)
        valley_frames.append(int(frame_values.iloc[pos]))

    valley_frames = sorted(set(valley_frames))
    if not valley_frames:
        return []

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
    return intervals


def _results_panel() -> Dict[str, bool]:
    with step_container("results"):
        st.markdown("### 5. Results")
        st.markdown('<div class="results-panel">', unsafe_allow_html=True)

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
            raw_numeric_columns: List[str] = []
            if metrics_df is not None:
                numeric_candidates = [
                    c
                    for c in metrics_df.columns
                    if metrics_df[c].dtype.kind in "fi" and c != "frame_idx"
                ]
                raw_numeric_columns = [c for c in numeric_candidates if c.startswith("raw_")]
                numeric_columns = [c for c in numeric_candidates if not c.startswith("raw_")]
            metric_options = numeric_columns + raw_numeric_columns

            st.markdown(f"**Detected repetitions:** {repetitions}")

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
                        rep_intervals = _compute_rep_intervals(
                            metrics_df=metrics_df,
                            report=report,
                            stats=stats,
                            numeric_columns=(numeric_columns if numeric_columns else metric_options),
                        )
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
                st.markdown("#### Calculated metrics")
                st.dataframe(metrics_df, width="stretch")

            if stats.warnings:
                st.warning("\n".join(f"• {msg}" for msg in stats.warnings))

            if stats.skip_reason:
                st.error(f"Repetition counting skipped: {stats.skip_reason}")

            eff_path = getattr(report, "effective_config_path", None)
            if eff_path:
                try:
                    eff_bytes = Path(eff_path).read_bytes()
                except Exception:
                    pass
                else:
                    st.download_button(
                        "Download effective config",
                        data=eff_bytes,
                        file_name=Path(eff_path).name,
                        mime="application/json",
                    )

            with st.expander("Run statistics (optional)", expanded=False):
                try:
                    st.dataframe(stats_df, width="stretch")
                except Exception:
                    st.json({row["Field"]: row["Value"] for row in stats_rows})

            if debug_video_enabled and debug_video_path:
                debug_path = Path(debug_video_path)
                has_overlay = bool(overlay_stream_path or overlay_raw_path)
                download_label = (
                    "Download landmark overlay video"
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
                    )

            if state.metrics_path is not None:
                metrics_data = None
                try:
                    metrics_data = Path(state.metrics_path).read_text(
                        encoding="utf-8"
                    )
                except FileNotFoundError:
                    st.error("The metrics file for download was not found.")
                except OSError as exc:
                    st.error(f"Could not read metrics: {exc}")
                else:
                    st.download_button(
                        "Download metrics",
                        data=metrics_data,
                        file_name=f"{Path(state.video_path).stem}_metrics.csv",
                        mime="text/csv",
                    )

        else:
            st.info("No results found to display.")

        adjust_col, reset_col = st.columns(2)
        with adjust_col:
            st.markdown('<div class="btn--continue">', unsafe_allow_html=True)
            adjust_clicked = st.button("Adjust configuration and re-run", key="results_adjust")
            st.markdown('</div>', unsafe_allow_html=True)
            if adjust_clicked:
                actions["adjust"] = True
        with reset_col:
            st.markdown('<div class="btn--back">', unsafe_allow_html=True)
            reset_clicked = st.button("Back to start", key="results_reset")
            st.markdown('</div>', unsafe_allow_html=True)
            if reset_clicked:
                actions["reset"] = True

        st.markdown("</div>", unsafe_allow_html=True)

        return actions


def _results_summary() -> None:
    with step_container("results"):
        st.markdown("### 4. Run the analysis")
        state = get_state()
        if state.last_run_success:
            st.success("Analysis complete ✅")
