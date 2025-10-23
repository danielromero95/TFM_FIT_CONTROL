from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None  # pragma: no cover

from src.D_modeling.count_reps import count_repetitions_with_config
from src.pipeline_data import Report, RunStats
from src.ui.state import get_state
from src.ui.video import render_uniform_video
from src.ui.metrics import render_video_with_metrics_sync


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
    st.markdown('<div class="results-panel">', unsafe_allow_html=True)
    st.markdown("### 5. Results")

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

        numeric_columns: List[str] = [
            c
            for c in metrics_df.columns
            if metrics_df[c].dtype.kind in "fi" and c != "frame_idx"
        ] if metrics_df is not None else []

        st.markdown(f"**Detected repetitions:** {repetitions}")

        debug_video_enabled = bool((state.configure_values or {}).get("debug_video", True))
        debug_video_path = str(report.debug_video_path) if report.debug_video_path else None
        preferred_video_path = str(state.video_path) if state.video_path else None
        if debug_video_enabled and debug_video_path:
            preferred_video_path = debug_video_path

        rendered_sync_viewer = False
        should_render_fallback_video = metrics_df is None

        if metrics_df is not None:
            st.markdown('<div class="results-metrics-block">', unsafe_allow_html=True)
            if numeric_columns:
                selected_metrics = st.multiselect(
                    "View metrics",
                    options=numeric_columns,
                    default=numeric_columns[:3],
                    key="metrics_multiselect",
                )
                if selected_metrics:
                    fps_value = float(getattr(stats, "fps_effective", 0.0) or 0.0)
                    rep_intervals: List[Tuple[int, int]] = []
                    start_at_s: Optional[float] = None
                    if preferred_video_path:
                        rep_intervals = _compute_rep_intervals(
                            metrics_df, report, stats, numeric_columns
                        )
                        if rep_intervals:
                            rep_options = list(range(1, len(rep_intervals) + 1))
                            rep_idx = st.select_slider(
                                "Go to rep",
                                options=rep_options,
                                value=rep_options[0],
                                key="results_rep_slider",
                            )
                            fps_for_slider = fps_value if fps_value > 0 else 1.0
                            start_at_s = rep_intervals[rep_idx - 1][0] / fps_for_slider

                    if preferred_video_path:
                        render_video_with_metrics_sync(
                            video_path=preferred_video_path,
                            metrics_df=metrics_df,
                            selected_metrics=selected_metrics,
                            fps=fps_value,
                            rep_intervals=rep_intervals,
                            start_at_s=start_at_s,
                            key="results_video_metrics_sync",
                            bottom_margin_rem=0.18,
                        )
                        rendered_sync_viewer = True
                    else:
                        st.info("Video not available for sync; showing simple chart.")
                        st.line_chart(metrics_df[selected_metrics])
                else:
                    st.info(
                        "Select at least one metric to visualize. Showing the video below as a fallback."
                    )
                    should_render_fallback_video = True
            else:
                st.info("No numeric metrics available for charting.")
                should_render_fallback_video = True
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            should_render_fallback_video = True

        if should_render_fallback_video and not rendered_sync_viewer and preferred_video_path is not None:
            render_uniform_video(
                preferred_video_path,
                key="results_video_fallback",
                bottom_margin=0.18,
            )

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

        with st.expander("Run statistics (optional)", expanded=False):
            try:
                st.dataframe(stats_df, width="stretch")
            except Exception:
                st.json({row["Field"]: row["Value"] for row in stats_rows})

            if state.video_path:
                st.markdown("### Original video")
                render_uniform_video(
                    str(state.video_path),
                    key="results_original_video",
                    bottom_margin=0.25,
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

        if state.count_path is not None:
            count_data = None
            try:
                count_data = Path(state.count_path).read_text(
                    encoding="utf-8"
                )
            except FileNotFoundError:
                st.error("The repetition file for download was not found.")
            except OSError as exc:
                st.error(f"Could not read the repetition file: {exc}")
            else:
                st.download_button(
                    "Download repetition count",
                    data=count_data,
                    file_name=f"{Path(state.video_path).stem}_count.txt",
                    mime="text/plain",
                )

    else:
        st.info("No results found to display.")

    adjust_col, reset_col = st.columns(2)
    with adjust_col:
        if st.button("Adjust configuration and re-run", key="results_adjust"):
            actions["adjust"] = True
    with reset_col:
        if st.button("Back to start", key="results_reset"):
            actions["reset"] = True

    st.markdown("</div>", unsafe_allow_html=True)

    return actions


def _results_summary() -> None:
    st.markdown("### 4. Run the analysis")
    state = get_state()
    if state.last_run_success:
        st.success("Analysis complete ✅")
