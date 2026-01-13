"""Pantalla que muestra el progreso en tiempo real del pipeline de análisis."""

from __future__ import annotations

import time
from pathlib import Path
from queue import Empty
from typing import Optional

from concurrent.futures import CancelledError

import cv2
import streamlit as st

from src.ui.controllers import (
    RunHandle,
    cancel_run,
    get_progress_queue,
    phase_for,
    poll_progress,
    start_run,
)
from src.ui.state import Step, get_state, go_to
from ..utils import ensure_video_path, prepare_pipeline_inputs, step_container


def _running_step() -> None:
    """Gestiona la ejecución en curso y refleja su progreso en la UI."""

    with step_container("running"):
        st.markdown("### 4. Running the analysis")
        progress_queue = get_progress_queue()

        # Live preview (debug) apagado por defecto.
        SHOW_LIVE_PREVIEW = False
        preview_placeholder = st.empty() if SHOW_LIVE_PREVIEW else None
        preview_stats_placeholder = st.empty() if SHOW_LIVE_PREVIEW else None

        def _drain_progress_queue() -> None:
            while True:
                try:
                    progress_queue.get_nowait()
                except Empty:
                    break

        def _make_handle(run_id: Optional[str], future) -> Optional[RunHandle]:
            if run_id and future:
                return RunHandle(run_id=run_id, future=future)
            return None

        state = get_state()
        if state.analysis_future and state.analysis_future.done():
            state.analysis_future = None
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
            return

        cancel_disabled = not state.analysis_future or state.analysis_future.done()
        if st.button("Cancel analysis", disabled=cancel_disabled):
            handle = _make_handle(state.run_id, state.analysis_future)
            if handle:
                cancel_run(handle)
            state.run_id = None
            state.last_run_success = False
            state.pipeline_error = "Analysis canceled by the user."
            state.preview_enabled = False
            state.overlay_video_stream_path = None
            state.overlay_video_download_path = None

        debug_enabled = bool((state.configure_values or {}).get("debug_video", True))

        state.last_run_success = False

        ensure_video_path()
        state = get_state()
        if not state.video_path:
            state.pipeline_error = "The video to process was not found."
            state.preview_enabled = False
            state.overlay_video_stream_path = None
            state.overlay_video_download_path = None
            go_to(Step.RESULTS)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
            return

        if state.analysis_future is None:
            try:
                video_path, cfg, prefetched_detection = prepare_pipeline_inputs(state)
            except ValueError as exc:
                state.pipeline_error = str(exc)
                state.preview_enabled = False
                state.overlay_video_stream_path = None
                state.overlay_video_download_path = None
                go_to(Step.RESULTS)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
                return

            state.pipeline_error = None
            state.report = None
            state.metrics_path = None
            state.arm_debug_timeseries_path = None
            state.cfg_fingerprint = None
            state.overlay_video_stream_path = None
            state.overlay_video_download_path = None
            state.progress_value_from_cb = 0
            state.phase_text_from_cb = phase_for(0, debug_enabled=debug_enabled)
            # Asegura que el estado de preview arranca deshabilitado
            state.preview_enabled = False
            state.preview_frame_count = 0
            state.preview_last_ts_ms = 0.0

            _drain_progress_queue()

            # Define callback solo si queremos mostrar preview en vivo
            def _preview_callback(frame_bgr, frame_idx: int, ts_ms: float) -> None:
                if not SHOW_LIVE_PREVIEW:
                    return
                try:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    frame_rgb = frame_bgr
                try:

                    if preview_placeholder is not None:
                        preview_placeholder.image(
                            frame_rgb, width="stretch", channels="RGB"
                        )

                    seconds = ts_ms / 1000.0 if ts_ms else 0.0
                    fps_display = (frame_idx + 1) / seconds if seconds > 0 else 0.0
                    if preview_stats_placeholder is not None:
                        preview_stats_placeholder.caption(
                            f"Frame {frame_idx + 1} • {fps_display:.1f} FPS"
                        )
                    state_ref = get_state()
                    state_ref.preview_frame_count = frame_idx + 1
                    state_ref.preview_last_ts_ms = float(ts_ms)
                except Exception:
                    pass

            preview_fps_value = getattr(getattr(cfg, "debug", object()), "preview_fps", state.preview_fps)
            handle = start_run(
                video_path=video_path,
                cfg=cfg,
                prefetched_detection=prefetched_detection,
                exercise_selected=state.exercise_selected,
                view_selected=state.view_selected,
                debug_enabled=debug_enabled,
                # No enviar callback si el preview está desactivado
                preview_callback=(_preview_callback if SHOW_LIVE_PREVIEW else None),
                preview_fps=(preview_fps_value if SHOW_LIVE_PREVIEW else None),
            )
            state.preview_enabled = bool(SHOW_LIVE_PREVIEW)
            if SHOW_LIVE_PREVIEW:
                state.preview_fps = float(preview_fps_value or state.preview_fps)
            state.run_id = handle.run_id
            state.analysis_future = handle.future
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
            return

        future = state.analysis_future
        with st.status("Analyzing video...", expanded=True) as status:
            latest_progress = getattr(state, "progress_value_from_cb", 0)
            latest_phase = getattr(
                state,
                "phase_text_from_cb",
                phase_for(latest_progress, debug_enabled=debug_enabled),
            )
            bar = st.progress(latest_progress)
            progress_message = status.empty()

            status.update(
                label=f"Analyzing video... {latest_progress}%",
                state="running",
                expanded=True,
            )
            bar.progress(latest_progress)
            progress_message.markdown(
                f"**Current phase:** {latest_phase} ({latest_progress}%)"
            )

            while future and not future.done():
                state = get_state()
                if (
                    SHOW_LIVE_PREVIEW
                    and state.preview_enabled
                    and state.preview_frame_count
                    and state.preview_last_ts_ms
                ):
                    seconds = state.preview_last_ts_ms / 1000.0 if state.preview_last_ts_ms else 0.0
                    fps_display = state.preview_frame_count / seconds if seconds > 0 else 0.0
                    if preview_stats_placeholder is not None:
                        preview_stats_placeholder.caption(
                            f"Frame {state.preview_frame_count} • {fps_display:.1f} FPS"
                        )
                if state.run_id is None:
                    state.report = None
                    state.metrics_path = None
                    state.arm_debug_timeseries_path = None
                    state.cfg_fingerprint = None
                    state.overlay_video_stream_path = None
                    state.overlay_video_download_path = None
                    state.last_run_success = False
                    state.progress_value_from_cb = latest_progress
                    state.phase_text_from_cb = latest_phase
                    state.analysis_future = None
                    state.pipeline_error = "Analysis canceled by the user."
                    state.preview_enabled = False
                    go_to(Step.RESULTS)
                    _drain_progress_queue()
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                    return

                current_future = state.analysis_future
                if current_future is None:
                    break

                handle = _make_handle(state.run_id, current_future)
                if handle:
                    latest_progress, latest_phase = poll_progress(
                        handle,
                        latest_progress,
                        debug_enabled=debug_enabled,
                    )
                    state.progress_value_from_cb = latest_progress
                    state.phase_text_from_cb = latest_phase

                status.update(
                    label=f"Analyzing video... {latest_progress}%",
                    state="running",
                    expanded=True,
                )
                bar.progress(latest_progress)
                progress_message.markdown(
                    f"**Current phase:** {latest_phase} ({latest_progress}%)"
                )
                time.sleep(0.2)
                future = state.analysis_future

            state = get_state()
            current_future = state.analysis_future
            if not current_future:
                latest_phase = "Canceled"
                state.pipeline_error = "Analysis canceled by the user."
                state.report = None
                state.metrics_path = None
                state.arm_debug_timeseries_path = None
                state.cfg_fingerprint = None
                state.overlay_video_stream_path = None
                state.overlay_video_download_path = None
                state.last_run_success = False
                state.progress_value_from_cb = latest_progress
                state.phase_text_from_cb = latest_phase
                state.preview_enabled = False
                status.update(label="Analysis canceled", state="error", expanded=True)
                bar.progress(latest_progress)
                progress_message.markdown(
                    f"**Current phase:** {latest_phase} ({latest_progress}%)"
                )
                _drain_progress_queue()
                state.run_id = None
                go_to(Step.RESULTS)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
                return

        handle = _make_handle(state.run_id, current_future)
        try:
            if handle:
                latest_progress, latest_phase = poll_progress(
                    handle,
                    latest_progress,
                    debug_enabled=debug_enabled,
                )
                state.progress_value_from_cb = latest_progress
                state.phase_text_from_cb = latest_phase

            try:
                completed_run_id, report = current_future.result()
                _drain_progress_queue()
            except CancelledError:
                latest_phase = "Canceled"
                state.pipeline_error = "Analysis canceled by the user."
                state.report = None
                state.metrics_path = None
                state.arm_debug_timeseries_path = None
                state.cfg_fingerprint = None
                state.overlay_video_stream_path = None
                state.overlay_video_download_path = None
                state.last_run_success = False
                state.progress_value_from_cb = latest_progress
                state.phase_text_from_cb = latest_phase
                state.run_id = None
                state.preview_enabled = False
                status.update(label="Analysis canceled", state="error", expanded=True)
                bar.progress(latest_progress)
                progress_message.markdown(
                    f"**Current phase:** {latest_phase} ({latest_progress}%)"
                )
                _drain_progress_queue()
                state.analysis_future = None
                go_to(Step.RESULTS)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
                return

            if state.run_id != completed_run_id:
                latest_phase = "Canceled"
                state.pipeline_error = "Analysis canceled by the user."
                state.report = None
                state.metrics_path = None
                state.cfg_fingerprint = None
                state.overlay_video_stream_path = None
                state.overlay_video_download_path = None
                state.last_run_success = False
                state.progress_value_from_cb = latest_progress
                state.phase_text_from_cb = latest_phase
                state.run_id = None
                state.preview_enabled = False
                status.update(
                    label="Analysis canceled",
                    state="error",
                    expanded=False,
                )
                bar.progress(latest_progress)
                progress_message.markdown(
                    f"**Current phase:** {latest_phase} ({latest_progress}%)"
                )
                _drain_progress_queue()
                state.analysis_future = None
                go_to(Step.RESULTS)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
                return

            state.pipeline_error = None
            state.report = report
            state.cfg_fingerprint = report.stats.config_sha1

            def _extract_candidate(container: object, keys: tuple[str, ...]) -> object | None:
                if isinstance(container, dict):
                    for key in keys:
                        value = container.get(key)
                        if value:
                            return value
                else:
                    for key in keys:
                        if hasattr(container, key):
                            value = getattr(container, key)
                            if value:
                                return value
                return None

            overlay_stream_candidate = _extract_candidate(
                report,
                ("overlay_video_stream_path", "overlay_video_stream"),
            )
            overlay_download_candidate = _extract_candidate(
                report,
                (
                    "overlay_video_path",
                    "overlay_video",
                    "debug_video_path",
                    "debug_video",
                ),
            )
            if overlay_stream_candidate is None:
                overlay_stream_candidate = overlay_download_candidate

            def _to_existing_path(candidate: object | None) -> Path | None:
                if not candidate:
                    return None
                try:
                    path = Path(str(candidate)).expanduser()
                except Exception:
                    return None
                if path.exists() and path.is_file():
                    return path
                return None

            stream_path = _to_existing_path(overlay_stream_candidate)
            download_path = _to_existing_path(overlay_download_candidate)
            debug_requested = bool((state.configure_values or {}).get("debug_video", True))

            if debug_requested and stream_path is not None:
                state.overlay_video_stream_path = str(stream_path)
            else:
                state.overlay_video_stream_path = None

            if debug_requested:
                chosen_download = download_path or stream_path
                state.overlay_video_download_path = (
                    str(chosen_download) if chosen_download is not None else None
                )
            else:
                state.overlay_video_download_path = None

            file_errors: list[str] = []
            video_path = state.video_path
            metrics_path = getattr(report, "metrics_path", None)
            if metrics_path and Path(metrics_path).exists():
                state.metrics_path = str(metrics_path)
            else:
                state.metrics_path = None
            arm_debug_path = getattr(report, "arm_debug_timeseries_path", None)
            if arm_debug_path and Path(arm_debug_path).exists():
                state.arm_debug_timeseries_path = str(arm_debug_path)
            else:
                state.arm_debug_timeseries_path = None

            latest_progress = 100
            latest_phase = phase_for(100, debug_enabled=debug_enabled)
            state.progress_value_from_cb = latest_progress
            state.phase_text_from_cb = latest_phase
            bar.progress(latest_progress)

            if file_errors:
                state.pipeline_error = "\n".join(file_errors)
                state.last_run_success = False
                status.update(
                    label="Analysis completed with errors",
                    state="error",
                    expanded=True,
                )
            else:
                state.last_run_success = True
                status.update(
                    label="Analysis completed!",
                    state="complete",
                    expanded=False,
                )

            progress_message.markdown(
                f"**Current phase:** {latest_phase} ({latest_progress}%)"
            )
        except Exception as exc:
            state.pipeline_error = f"Error in analysis thread: {exc}"
            state.report = None
            state.metrics_path = None
            state.arm_debug_timeseries_path = None
            state.cfg_fingerprint = None
            state.overlay_video_stream_path = None
            state.overlay_video_download_path = None
            state.progress_value_from_cb = latest_progress
            state.phase_text_from_cb = latest_phase
            state.last_run_success = False
            state.preview_enabled = False
            status.update(label="Analysis error", state="error", expanded=True)
        finally:
            _drain_progress_queue()
            state = get_state()
            state.run_id = None
            if state.analysis_future is current_future:
                state.analysis_future = None
                go_to(Step.RESULTS)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
            return
