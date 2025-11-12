"""Funciones para orquestar la ejecución del pipeline desde la UI."""

from __future__ import annotations

import atexit
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Callable, Optional, Tuple

import numpy as np
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from src.C_analysis import run_pipeline

from .progress import make_progress_callback, phase_for


@st.cache_resource
def get_executor() -> ThreadPoolExecutor:
    """Crea un ``ThreadPoolExecutor`` único por sesión de Streamlit."""

    ex = ThreadPoolExecutor(max_workers=1)
    atexit.register(ex.shutdown, wait=False, cancel_futures=True)
    return ex


@st.cache_resource
def get_progress_queue() -> SimpleQueue:
    """Devuelve la cola compartida para publicar estados de progreso."""

    return SimpleQueue()


@dataclass(frozen=True)
class RunHandle:
    """Pequeño contenedor para vincular un ``run_id`` con su ``Future``."""

    run_id: str
    future: Future


def start_run(
    *,
    video_path: str,
    cfg,
    prefetched_detection: Optional[Tuple[str, str, float]],
    debug_enabled: bool,
    preview_callback: Optional[Callable[[np.ndarray, int, float], None]] = None,
    preview_fps: Optional[float] = None,
) -> RunHandle:
    """Inicia ``run_pipeline`` en segundo plano y retorna su identificador."""

    from uuid import uuid4

    run_id = uuid4().hex
    queue = get_progress_queue()
    cb = make_progress_callback(queue, run_id, debug_enabled)

    ctx = get_script_run_ctx()

    def _job():
        """Encapsula la ejecución del pipeline dentro del hilo de trabajo."""

        if ctx is not None:
            try:
                add_script_run_ctx(threading.current_thread(), ctx=ctx)
            except Exception:
                pass
        report = run_pipeline(
            str(video_path),
            cfg,
            progress_callback=cb,
            prefetched_detection=prefetched_detection,
            preview_callback=preview_callback,
            preview_fps=preview_fps,
        )
        return run_id, report

    fut = get_executor().submit(_job)
    ctx = get_script_run_ctx()
    if ctx is not None:
        add_script_run_ctx(fut, ctx=ctx)
    return RunHandle(run_id=run_id, future=fut)


def cancel_run(handle: RunHandle) -> None:
    """Intenta cancelar un análisis en curso de forma preventiva."""

    try:
        handle.future.cancel()
    except Exception:
        pass  # best-effort


def poll_progress(handle: RunHandle, last_progress: int, *, debug_enabled: bool) -> tuple[int, str]:
    """Agrupa los mensajes de la cola de progreso y devuelve el estado visible."""

    queue = get_progress_queue()
    latest = int(max(0, min(100, last_progress)))
    phase = phase_for(latest, debug_enabled=debug_enabled)

    while True:
        try:
            msg_run_id, p, ph = queue.get_nowait()
        except Empty:
            break
        if msg_run_id != handle.run_id:
            continue
        p = int(max(latest, p))
        phase = ph if p == latest else phase_for(p, debug_enabled=debug_enabled)
        latest = p

    return latest, phase
