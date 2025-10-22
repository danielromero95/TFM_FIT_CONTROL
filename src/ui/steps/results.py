from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import Report
from src.ui.state import get_state
from src.ui.video import render_uniform_video


@dataclass
class ResultsActions:
    adjust: bool = False
    reset: bool = False


def _results_panel() -> ResultsActions:
    st.markdown('<div class="results-panel">', unsafe_allow_html=True)
    st.markdown("### 5. Results")

    actions = ResultsActions()

    state = get_state()

    if state.pipeline_error:
        st.error("An error occurred during the analysis")
        st.code(str(state.pipeline_error))
    elif state.report is not None:
        report: Report = state.report
        stats = report.stats
        repetitions = report.repetitions
        metrics_df = report.metrics
        numeric_columns: list[str] = []
        if metrics_df is not None:
            numeric_columns = [
                col
                for col in metrics_df.columns
                if metrics_df[col].dtype.kind in "fi"
            ]

        st.markdown(f"**Detected repetitions:** {repetitions}")

        if report.debug_video_path and bool(
            (state.configure_values or {}).get("debug_video", True)
        ):
            render_uniform_video(
                str(report.debug_video_path),
                key="results_debug_video",
                bottom_margin=0.18,
            )

        if metrics_df is not None:
            st.markdown(
                '<div class="results-metrics-block">',
                unsafe_allow_html=True,
            )
            if numeric_columns:
                default_selection = numeric_columns[:3]
                selected_metrics = st.multiselect(
                    "View metrics",
                    options=numeric_columns,
                    default=default_selection,
                )
                if selected_metrics:
                    st.line_chart(metrics_df[selected_metrics])
            else:
                st.info("No numeric metrics available for charting.")
            st.markdown("</div>", unsafe_allow_html=True)

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
            actions.adjust = True
    with reset_col:
        if st.button("Back to start", key="results_reset"):
            actions.reset = True

    st.markdown("</div>", unsafe_allow_html=True)

    return actions


def _results_summary() -> None:
    st.markdown("### 4. Run the analysis")
    state = get_state()
    if state.last_run_success:
        st.success("Analysis complete ✅")
