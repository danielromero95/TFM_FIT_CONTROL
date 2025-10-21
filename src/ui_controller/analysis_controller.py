from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import SimpleQueue, Empty
from typing import Optional, Tuple

import streamlit as st

from src.services.analysis_service import run_pipeline
from src.ui_controller.progress import phase_for, make_progress_callback


# --- Recursos cacheados: un único executor y una única cola por sesión ----------
@st.cache_resource
def get_executor() -> ThreadPoolExecutor:
    ex = ThreadPoolExecutor(max_workers=1)
    atexit.register(ex.shutdown, wait=False, cancel_futures=True)
    return ex


@st.cache_resource
def get_progress_queue() -> SimpleQueue:
    return SimpleQueue()


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    future: Future


def start_run(
    *,
    video_path: str,
    cfg,
    prefetched_detection: Optional[Tuple[str, str, float]],
    debug_enabled: bool,
) -> RunHandle:
    """
    Lanza run_pipeline en background con callback de progreso
    y devuelve un RunHandle con (run_id, future). La cola de progreso
    global es la de get_progress_queue().
    """
    from uuid import uuid4

    run_id = uuid4().hex
    queue = get_progress_queue()
    cb = make_progress_callback(queue, run_id, debug_enabled)

    def _job():
        report = run_pipeline(
            str(video_path),
            cfg,
            progress_callback=cb,
            prefetched_detection=prefetched_detection,
        )
        return run_id, report

    fut = get_executor().submit(_job)
    return RunHandle(run_id=run_id, future=fut)


def cancel_run(handle: RunHandle) -> None:
    try:
        handle.future.cancel()
    except Exception:
        pass  # best-effort


def poll_progress(handle: RunHandle, last_progress: int, *, debug_enabled: bool) -> tuple[int, str]:
    """
    Lee la cola global y consolida el progreso para el run_id del handle.
    Devuelve (latest_progress, latest_phase). Si no hay novedades, devuelve
    el último valor recibido.
    """
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
