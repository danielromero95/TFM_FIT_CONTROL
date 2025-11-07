from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None  # pragma: no cover

from src.C_repetition_analysis.reps.api import count_repetitions_with_config
from src.pipeline_data import Report, RunStats
from src.ui.metrics_sync.viewer import render_video_with_metrics_sync
from src.ui.state import get_state
from ..utils import step_container


def _metric_descriptions_for(
    ex_str: str,
    low_thr: float | int | None,
    available: list[str],
) -> dict[str, str]:
    low_txt = f"{float(low_thr):.0f}°" if low_thr is not None else "the lower threshold"
    d = {
        "trunk_inclination_deg": "Torso tilt (deg)\nAngle between shoulder–hip line and vertical.",
        "shoulder_width": "Shoulder width (normalized)\nHorizontal distance between shoulders.",
        "foot_separation": "Foot separation (normalized)\nHorizontal distance between ankles/feet.",
    }

    if ex_str == "squat":
        d["left_knee"] = (
            f"Left knee angle (thigh–shank)\nReps counted when the primary knee dips below {low_txt}."
        )
        d["right_knee"] = (
            f"Right knee angle (thigh–shank)\nReps counted when the primary knee dips below {low_txt}."
        )
    elif ex_str == "bench_press":
        d["left_elbow"] = (
            f"Left elbow angle (upper arm–forearm)\nValleys near {low_txt} mark each rep."
        )
        d["right_elbow"] = (
            f"Right elbow angle (upper arm–forearm)\nValleys near {low_txt} mark each rep."
        )
        d["shoulder_width"] += "\nUseful to monitor grip consistency."
    elif ex_str == "deadlift":
        d["left_hip"] = (
            f"Hip angle (torso–thigh, left)\nReps valid when hip angle rises above {low_txt}."
        )
        d["right_hip"] = (
            f"Hip angle (torso–thigh, right)\nReps valid when hip angle rises above {low_txt}."
        )
        d["trunk_inclination_deg"] += "\nUseful to track hip hinge."

    for name in list(available):
        if name.startswith("ang_vel_"):
            base = name.removeprefix("ang_vel_")
            d[name] = f"Angular velocity of {base.replace('_', ' ')} (deg/s)."
        if name.startswith("raw_"):
            base = name.removeprefix("raw_")
            d[name] = (
                f"Raw (uninterpolated) samples of '{base}'.\nMay include gaps from low-visibility frames."
            )

    return {k: v for k, v in d.items() if k in available}


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

                    ex = getattr(stats, "exercise_detected", "")
                    ex_str = getattr(ex, "value", str(ex))
                    low_thr = None
                    try:
                        low_thr = float((state.configure_values or {}).get("low"))
                    except Exception:
                        pass
                    metric_desc = _metric_descriptions_for(ex_str, low_thr, metric_options)
                    primary = getattr(stats, "primary_angle", None)
                    if primary and primary in metric_desc:
                        metric_desc[primary] = f"PRIMARY — {metric_desc[primary]}"

                    selected_metrics = st.multiselect(
                        "View metrics",
                        options=metric_options,
                        default=st.session_state[default_key],
                        key=widget_key,
                        help="Hover the legend to read metric definitions.",
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
                            metric_descriptions=metric_desc,
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


def _results_summary() -> None:
    with step_container("results"):
        st.markdown("### 4. Run the analysis")
        state = get_state()
        if state.last_run_success:
            st.success("Analysis complete ✅")
