from __future__ import annotations
from queue import SimpleQueue


def phase_for(p: int, *, debug_enabled: bool) -> str:
    value = int(max(0, min(100, p)))
    if value < 10:
        return "Preparing…"
    if value < 25:
        return "Extracting frames…"
    if value < 50:
        return "Estimating pose…"
    if value < 65:
        return "Filtering and interpolating…"
    if debug_enabled and value < 75:
        return "Rendering debug video…"
    if debug_enabled:
        if value < 90:
            return "Computing metrics…"
    else:
        if value < 85:
            return "Computing metrics…"
    if value < 100:
        return "Counting repetitions…"
    return "Finishing up…"


def make_progress_callback(queue: SimpleQueue, run_id: str, debug_enabled: bool):
    def _cb(p: int) -> None:
        try:
            value = max(0, min(100, int(p)))
            queue.put((run_id, value, phase_for(value, debug_enabled=debug_enabled)))
        except Exception:
            # defensivo: no romper el hilo de análisis por errores de UI/cola
            pass

    return _cb
