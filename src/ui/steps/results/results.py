from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as components_html

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None  # pragma: no cover

from src.C_repetition_analysis.reps.api import count_repetitions_with_config
from src.pipeline_data import Report, RunStats
from src.ui.metrics_sync.viewer import render_video_with_metrics_sync
from src.ui.state import get_state
from ..utils import step_container


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
            overlay_video_path = getattr(report, "overlay_video_path", None)
            debug_video_path = overlay_video_path or report.debug_video_path

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
                    primary_metric = chosen if chosen in metric_options else None
                    metric_descriptions = _build_metric_descriptions(
                        exercise=ex_str,
                        options=metric_options,
                        primary_metric=primary_metric,
                    )

                    selected_metrics = st.multiselect(
                        "View metrics",
                        options=metric_options,
                        default=st.session_state[default_key],
                        key=widget_key,
                    )
                    if metric_descriptions:
                        anchor_id = f"{widget_key}-anchor"
                        st.markdown(
                            f"<div data-metric-anchor=\"{anchor_id}\"></div>",
                            unsafe_allow_html=True,
                        )
                        _render_metric_help(
                            widget_key=widget_key,
                            anchor_id=anchor_id,
                            descriptions=metric_descriptions,
                        )
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
                download_label = (
                    "Download landmark overlay video"
                    if overlay_video_path
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


def _build_metric_descriptions(
    exercise: str, options: List[str], primary_metric: Optional[str]
) -> Dict[str, str]:
    exercise_key = (exercise or "").lower().strip()
    descriptions: Dict[str, str] = {}
    for option in options:
        desc = _describe_metric(exercise_key, option, primary_metric)
        if desc:
            descriptions[option] = desc
    return descriptions


def _describe_metric(exercise: str, metric: str, primary_metric: Optional[str]) -> Optional[str]:
    is_primary = primary_metric == metric
    base_metric = metric
    is_raw = False
    if metric.startswith("raw_"):
        base_metric = metric[4:]
        is_raw = True
    info = _get_metric_info(exercise, base_metric)
    if info is None:
        info = _get_fallback_metric_info(exercise, base_metric)
    if info is None:
        return None

    body = info.get("body", "").strip()
    primary_sentence = info.get("primary", "").strip()
    secondary_sentence = info.get("secondary", "").strip()

    if not body:
        return None

    if is_raw:
        # Raw metrics are informational regardless of the processed primary metric.
        combined = [body, "Raw signal before smoothing; does not affect rep counting."]
        text = " ".join(sentence for sentence in combined if sentence)
        return text

    counting_sentence = primary_sentence if is_primary else secondary_sentence
    combined = [body, counting_sentence]
    text = " ".join(sentence for sentence in combined if sentence)
    if not text:
        return None
    if is_primary:
        text = f"PRIMARY — {text}"
    return text


def _get_metric_info(exercise: str, metric: str) -> Optional[Dict[str, str]]:
    exercise_map = _EXERCISE_METRIC_INFO.get(exercise, {})
    if metric in exercise_map:
        return exercise_map[metric]
    return _COMMON_METRIC_INFO.get(metric)


def _get_fallback_metric_info(exercise: str, metric: str) -> Optional[Dict[str, str]]:
    exercise_label = exercise.replace("_", " ") if exercise else "exercise"
    if metric.startswith("ang_vel_"):
        base_metric = metric[8:]
        joint_label = _humanize_metric(base_metric)
        return {
            "body": f"Instantaneous angular velocity of the {joint_label} during the {exercise_label}.",
            "primary": "Reps would only be counted from the underlying angle signal, not the velocity values.",
            "secondary": "Highlights speed changes; does not affect rep counting.",
        }
    if metric.endswith("_symmetry"):
        base_metric = metric[: -len("_symmetry")]
        joint_label = _humanize_metric(base_metric)
        return {
            "body": f"Difference between left and right {joint_label} measurements during the {exercise_label}.",
            "primary": "Reps are still determined by the primary angle thresholds, not by this symmetry score.",
            "secondary": "Shows side-to-side balance; does not affect rep counting.",
        }
    if metric.endswith("_width") or metric.endswith("_separation"):
        joint_label = _humanize_metric(metric)
        return {
            "body": f"Distance measurement ({joint_label}) captured throughout the {exercise_label} setup.",
            "primary": "Reps continue to rely on the primary angle crossing its thresholds.",
            "secondary": "Monitors positioning; does not affect rep counting.",
        }
    if metric in {"left_knee", "right_knee", "left_elbow", "right_elbow", "left_hip", "right_hip"}:
        joint_label = _humanize_metric(metric)
        return {
            "body": f"Tracks the {joint_label} angle during the {exercise_label}.",
            "primary": "Reps counted when the angle drops below the lower threshold and rises above the upper threshold.",
            "secondary": "Provides additional context but does not change rep counting.",
        }
    if metric:
        joint_label = _humanize_metric(metric)
        return {
            "body": f"Tracks the {joint_label} measurement during the {exercise_label}.",
            "primary": "Reps are detected when this measurement crosses the configured thresholds.",
            "secondary": "Reference only; does not affect rep counting.",
        }
    return None


def _humanize_metric(metric: str) -> str:
    metric = metric.replace("raw_", "")
    parts = metric.split("_")
    return " ".join(part for part in parts if part)


def _render_metric_help(widget_key: str, anchor_id: str, descriptions: Dict[str, str]) -> None:
    if not descriptions:
        return
    _ensure_metric_help_styles()
    payload = {
        "widgetKey": widget_key,
        "anchorId": anchor_id,
        "descriptions": descriptions,
    }
    script = _METRIC_HELP_SCRIPT_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    components_html(script, height=0)


def _ensure_metric_help_styles() -> None:
    if st.session_state.get("_metric_help_styles_applied"):
        return
    st.session_state["_metric_help_styles_applied"] = True
    st.markdown(
        """
<style>
.metric-help-button {
    border: none;
    background: transparent;
    color: inherit;
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 600;
    width: 1.5rem;
    height: 1.5rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0;
}
.metric-help-button:hover,
.metric-help-button:focus-visible {
    background-color: rgba(0, 0, 0, 0.08);
    outline: none;
}
.metric-help-popover {
    position: absolute;
    z-index: 9999;
    max-width: 260px;
    padding: 0.75rem 0.9rem;
    background: var(--background-color, #ffffff);
    border: 1px solid rgba(0, 0, 0, 0.2);
    border-radius: 0.5rem;
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.15);
    font-size: 0.85rem;
    line-height: 1.35;
}
.metric-help-option {
    position: relative;
    padding-right: 2.4rem !important;
}
.metric-help-option .metric-help-button {
    position: absolute;
    right: 0.4rem;
    top: 50%;
    transform: translateY(-50%);
}
</style>
        """,
        unsafe_allow_html=True,
    )


_METRIC_HELP_SCRIPT_TEMPLATE = """
<script>
(function() {
  const payload = __PAYLOAD__;
  const parentWindow = window.parent;
  if (!parentWindow || !parentWindow.document) {
    return;
  }
  const doc = parentWindow.document;
  const globalState = parentWindow.__metricHelpGlobal = parentWindow.__metricHelpGlobal || {};
  globalState.registry = globalState.registry || {};
  const registry = globalState.registry;
  registry[payload.widgetKey] = registry[payload.widgetKey] || {};
  const state = registry[payload.widgetKey];
  state.descriptions = payload.descriptions || {};
  state.anchorId = payload.anchorId;

  if (!globalState.closeAllPopovers) {
    globalState.closeAllPopovers = function () {
      if (globalState.activePopover && globalState.activePopover.element) {
        globalState.activePopover.element.remove();
      }
      globalState.activePopover = null;
    };
  }

  if (!globalState.ensureGlobalHandlers) {
    globalState.ensureGlobalHandlers = function () {
      if (globalState.handlersAttached) {
        return;
      }
      doc.addEventListener('click', function (event) {
        if (event.target.closest('.metric-help-button') || event.target.closest('.metric-help-popover')) {
          return;
        }
        globalState.closeAllPopovers();
      });
      doc.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
          globalState.closeAllPopovers();
        }
      });
      globalState.handlersAttached = true;
    };
  }

  globalState.ensureGlobalHandlers();

  function ensureWidget() {
    const anchor = doc.querySelector('[data-metric-anchor="' + state.anchorId + '"]');
    if (!anchor) {
      window.setTimeout(ensureWidget, 100);
      return;
    }
    const multiSelect = findMultiSelect(anchor);
    if (!multiSelect) {
      window.setTimeout(ensureWidget, 100);
      return;
    }
    const input = multiSelect.querySelector('input[role="combobox"]');
    if (!input) {
      window.setTimeout(ensureWidget, 100);
      return;
    }
    state.multiSelect = multiSelect;
    state.input = input;
    state.listboxId = input.getAttribute('aria-controls');
    bindHandlers(state);
    applyToMenu(state);
  }

  function findMultiSelect(anchor) {
    let container = anchor.parentElement;
    const limit = 10;
    let depth = 0;
    while (container && depth < limit) {
      const selects = container.querySelectorAll('[data-testid="stMultiSelect"]');
      if (selects.length) {
        return selects[selects.length - 1];
      }
      container = container.parentElement;
      depth += 1;
    }
    return null;
  }

  function bindHandlers(state) {
    if (!state.input) {
      return;
    }
    if (!state.boundInputHandler) {
      const handler = function () {
        window.setTimeout(function () {
          applyToMenu(state);
        }, 0);
      };
      state.boundInputHandler = handler;
      state.input.addEventListener('click', handler);
      state.input.addEventListener('focus', handler);
      state.input.addEventListener('input', handler);
      const toggleButton = state.multiSelect.querySelector('svg')?.parentElement;
      if (toggleButton) {
        toggleButton.addEventListener('click', handler);
      }
    }
    if (state.menuObserver) {
      state.menuObserver.disconnect();
    }
    const observer = new MutationObserver(function () {
      applyToMenu(state);
    });
    observer.observe(doc.body, { childList: true, subtree: true });
    state.menuObserver = observer;
  }

  function applyToMenu(state) {
    if (!state.listboxId) {
      return;
    }
    const menu = doc.getElementById(state.listboxId);
    if (!menu) {
      return;
    }
    if (!menu.dataset.metricHelpBound) {
      menu.addEventListener('click', function () {
        globalState.closeAllPopovers();
      });
      menu.dataset.metricHelpBound = '1';
    }
    const options = menu.querySelectorAll('[role="option"]');
    options.forEach(function (option) {
      decorateOption(option, state);
    });
  }

  function decorateOption(option, state) {
    const descriptions = state.descriptions || {};
    let label = option.getAttribute('data-metric-help-name');
    if (!label) {
      label = (option.textContent || '').replace(/\u2713/g, '').trim();
      const newlineIndex = label.indexOf('\n');
      if (newlineIndex !== -1) {
        label = label.slice(0, newlineIndex).trim();
      }
      option.setAttribute('data-metric-help-name', label);
    }
    const description = descriptions[label];
    const existingButton = option.querySelector('.metric-help-button');
    if (!description) {
      if (existingButton) {
        existingButton.remove();
      }
      option.classList.remove('metric-help-option');
      return;
    }
    option.classList.add('metric-help-option');
    let button = existingButton;
    if (!button) {
      button = doc.createElement('button');
      button.type = 'button';
      button.className = 'metric-help-button';
      button.textContent = '?';
      button.addEventListener('click', function (event) {
        event.preventDefault();
        event.stopPropagation();
        togglePopover(label, description, button);
      });
      option.appendChild(button);
    }
    button.title = description;
    button.setAttribute('aria-label', 'Show description for ' + label);
    button.dataset.metricDescription = description;
  }

  function togglePopover(label, description, button) {
    if (globalState.activePopover && globalState.activePopover.button === button) {
      globalState.closeAllPopovers();
      return;
    }
    globalState.closeAllPopovers();
    const popover = doc.createElement('div');
    popover.className = 'metric-help-popover';
    popover.setAttribute('data-metric', label);
    const lines = String(description || '').split('\n');
    popover.innerHTML = '';
    lines.forEach(function (line, index) {
      const lineEl = doc.createElement('div');
      lineEl.textContent = line;
      popover.appendChild(lineEl);
      if (index < lines.length - 1 && lines[index + 1] === '') {
        const spacer = doc.createElement('div');
        spacer.style.height = '0.5rem';
        popover.appendChild(spacer);
      }
    });
    doc.body.appendChild(popover);
    const rect = button.getBoundingClientRect();
    popover.style.left = (rect.left + parentWindow.scrollX) + 'px';
    popover.style.top = (rect.bottom + parentWindow.scrollY + 6) + 'px';
    globalState.activePopover = { element: popover, button: button };
  }

  ensureWidget();
})();
</script>
"""


_EXERCISE_METRIC_INFO: Dict[str, Dict[str, Dict[str, str]]] = {
    "squat": {
        "left_knee": {
            "body": "Tracks the flexion angle at the left knee throughout the squat motion.",
            "primary": "Reps counted when the angle dips below the lower threshold and then rises beyond the upper threshold.",
            "secondary": "Helps compare sides; does not directly trigger rep counting.",
        },
        "right_knee": {
            "body": "Tracks the flexion angle at the right knee during the squat.",
            "primary": "Reps counted when the angle dips below the lower threshold and then rises beyond the upper threshold.",
            "secondary": "Supports side comparison; does not directly trigger rep counting.",
        },
        "trunk_inclination_deg": {
            "body": "Measures torso lean relative to vertical while you squat.",
            "primary": "Reps counted when the torso angle dips below the lower threshold and then rises beyond the upper threshold.",
            "secondary": "Shows chest position control; does not affect rep counting.",
        },
        "foot_separation": {
            "body": "Measures stance width by tracking ankle separation during the squat setup.",
            "primary": "Reps are still based on the primary knee angle thresholds.",
            "secondary": "Keeps stance width in check; does not affect rep counting.",
        },
        "knee_symmetry": {
            "body": "Reports the difference between left and right knee flexion in the squat.",
            "primary": "Reps remain governed by the primary knee angle crossing its thresholds.",
            "secondary": "Highlights imbalances; does not affect rep counting.",
        },
        "ang_vel_left_knee": {
            "body": "Instantaneous angular velocity for the left knee flexion during the squat.",
            "primary": "Reps are determined from the knee angle thresholds rather than velocity.",
            "secondary": "Shows speed through the motion; does not affect rep counting.",
        },
        "ang_vel_right_knee": {
            "body": "Instantaneous angular velocity for the right knee during the squat.",
            "primary": "Reps come from the knee angle crossing its thresholds, not velocity.",
            "secondary": "Captures tempo; does not affect rep counting.",
        },
    },
    "bench_press": {
        "left_elbow": {
            "body": "Tracks the flexion angle at the left elbow throughout the press.",
            "primary": "Reps counted when the elbow angle drops below the lower threshold and then opens past the upper threshold.",
            "secondary": "Helps compare both arms; does not directly change rep counting.",
        },
        "right_elbow": {
            "body": "Tracks the flexion angle at the right elbow during the press.",
            "primary": "Reps counted when the elbow angle drops below the lower threshold and then opens past the upper threshold.",
            "secondary": "Assists with asymmetry checks; does not directly change rep counting.",
        },
        "shoulder_width": {
            "body": "Measures shoulder-to-shoulder distance to monitor bench setup width.",
            "primary": "Reps are still driven by the chosen elbow angle thresholds.",
            "secondary": "Tracks setup consistency; does not affect rep counting.",
        },
        "elbow_symmetry": {
            "body": "Reports the difference between left and right elbow angles during the press.",
            "primary": "Counting remains based on the primary elbow angle crossing thresholds.",
            "secondary": "Shows side-to-side control; does not affect rep counting.",
        },
        "ang_vel_left_elbow": {
            "body": "Instantaneous angular velocity for the left elbow during the press.",
            "primary": "Reps rely on angle thresholds, not velocity.",
            "secondary": "Highlights speed changes; does not affect rep counting.",
        },
        "ang_vel_right_elbow": {
            "body": "Instantaneous angular velocity for the right elbow during the press.",
            "primary": "Reps rely on angle thresholds, not velocity.",
            "secondary": "Highlights speed changes; does not affect rep counting.",
        },
    },
    "deadlift": {
        "left_hip": {
            "body": "Tracks the hip hinge angle on the left side during the deadlift pull.",
            "primary": "Reps counted when the hip angle crosses the configured thresholds.",
            "secondary": "Assists with hip tracking; does not directly change rep counting.",
        },
        "right_hip": {
            "body": "Tracks the hip hinge angle on the right side during the deadlift.",
            "primary": "Reps counted when the hip angle crosses the configured thresholds.",
            "secondary": "Monitors hip symmetry; does not directly change rep counting.",
        },
        "trunk_inclination_deg": {
            "body": "Measures torso pitch relative to vertical throughout the deadlift.",
            "primary": "Reps counted when the torso angle drops past the lower threshold and then extends beyond the upper threshold.",
            "secondary": "Highlights back control; does not affect rep counting.",
        },
        "foot_separation": {
            "body": "Measures stance width by tracking ankle separation during the deadlift setup.",
            "primary": "Reps remain tied to the primary hip angle thresholds.",
            "secondary": "Checks stance consistency; does not affect rep counting.",
        },
        "ang_vel_left_hip": {
            "body": "Instantaneous angular velocity for the left hip hinge during the deadlift.",
            "primary": "Counting still uses the hip angle thresholds rather than velocity.",
            "secondary": "Shows pull tempo; does not affect rep counting.",
        },
        "ang_vel_right_hip": {
            "body": "Instantaneous angular velocity for the right hip hinge during the deadlift.",
            "primary": "Counting still uses the hip angle thresholds rather than velocity.",
            "secondary": "Shows pull tempo; does not affect rep counting.",
        },
    },
}


_COMMON_METRIC_INFO: Dict[str, Dict[str, str]] = {}


def _results_summary() -> None:
    with step_container("results"):
        st.markdown("### 4. Run the analysis")
        state = get_state()
        if state.last_run_success:
            st.success("Analysis complete ✅")
