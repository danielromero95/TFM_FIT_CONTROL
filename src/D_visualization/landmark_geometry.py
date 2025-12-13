"""Utilidades geométricas para normalizar coordenadas de marcadores.
Agrupa cálculos matemáticos y estimaciones de encuadre sin depender de OpenCV para
que puedan probarse de forma aislada y reutilizarse en distintos renderizadores."""

from __future__ import annotations

import math
from typing import Optional, Sequence

# Declaramos la API pública del módulo.
__all__ = ["_normalize_points_for_frame", "_estimate_subject_bbox"]


def _normalize_points_for_frame(
    frame_landmarks,
    crop_box: Optional[Sequence[float]],
    orig_w: int,
    orig_h: int,
    proc_w: int,
    proc_h: int,
) -> dict[int, tuple[int, int]]:
    """Convierte puntos normalizados del modelo en coordenadas de píxel absolutas.
    Lo necesitamos para dibujar sobre frames reales y medir distancias en píxeles sin
    repetir el mismo cálculo de escalado en cada lugar del código."""

    pts: dict[int, tuple[int, int]] = {}
    if frame_landmarks is None:
        return pts
    try:
        if all((math.isnan(lm["x"]) for lm in frame_landmarks)):
            return pts
    except Exception:
        # Si los datos no son numéricos, simplemente continuamos.
        pass
    crop_vals: Optional[tuple[float, float, float, float]] = None
    if crop_box is not None:
        try:
            crop_vals = tuple(map(float, crop_box))  # type: ignore[arg-type]
        except Exception:
            crop_vals = None

    coords_are_pixels = False
    treat_as_global = crop_vals is None
    if crop_vals is not None and max(crop_vals) > 1.0:
        # Algunos modelos devuelven coordenadas absolutas cuando el recorte ya está en píxeles.
        spans_frame = (
            math.isclose(crop_vals[0], 0.0, abs_tol=1e-3)
            and math.isclose(crop_vals[1], 0.0, abs_tol=1e-3)
            and math.isclose(crop_vals[2], float(proc_w), rel_tol=0.01)
            and math.isclose(crop_vals[3], float(proc_h), rel_tol=0.01)
        )
        treat_as_global = spans_frame
        for lm in frame_landmarks:
            try:
                x = float(lm["x"])
                y = float(lm["y"])
            except Exception:
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if x > 1.0 or y > 1.0:
                coords_are_pixels = True
                break

    sx, sy = (orig_w / float(proc_w)), (orig_h / float(proc_h))
    for idx, lm in enumerate(frame_landmarks):
        try:
            x, y = float(lm["x"]), float(lm["y"])
            if math.isnan(x) or math.isnan(y):
                continue
        except Exception:
            continue
        if coords_are_pixels:
            final_x = int(round(x * sx))
            final_y = int(round(y * sy))
        elif not treat_as_global and crop_vals is not None:
            x1_p, y1_p, x2_p, y2_p = crop_vals
            # Convertimos el punto relativo al recorte en coordenadas absolutas del frame procesado.
            abs_x_p = x1_p + x * (x2_p - x1_p)
            abs_y_p = y1_p + y * (y2_p - y1_p)
            final_x = int(round(abs_x_p * sx))
            final_y = int(round(abs_y_p * sy))
        else:
            final_x = int(round(x * orig_w))
            final_y = int(round(y * orig_h))
        final_x = max(0, min(orig_w - 1, final_x))
        final_y = max(0, min(orig_h - 1, final_y))
        pts[idx] = (final_x, final_y)
    return pts


def _estimate_subject_bbox(
    pts: dict[int, tuple[int, int]],
    frame_w: int,
    frame_h: int,
    *,
    margin: float = 0.12,
) -> tuple[int, int, int, int] | None:
    """Calcula una caja delimitadora expandida que abarque los puntos detectados.
    Añadimos un margen configurable para no cortar extremidades al generar recortes o
    reenfocar la cámara virtual alrededor de la persona analizada."""

    if not pts:
        return None

    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    if not xs or not ys:
        return None

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    base = min(frame_w, frame_h)
    margin_px = float(max(0.0, margin)) * float(base)

    x1 = int(math.floor(min_x - margin_px))
    y1 = int(math.floor(min_y - margin_px))
    x2 = int(math.ceil(max_x + margin_px + 1.0))
    y2 = int(math.ceil(max_y + margin_px + 1.0))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, max(x2, x1 + 1))
    y2 = min(frame_h, max(y2, y1 + 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2
