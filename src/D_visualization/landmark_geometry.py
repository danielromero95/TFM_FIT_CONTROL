"""Utilidades geométricas para normalizar coordenadas de marcadores.
Agrupa cálculos matemáticos y estimaciones de encuadre sin depender de OpenCV para
que puedan probarse de forma aislada y reutilizarse en distintos renderizadores."""

from __future__ import annotations

import math
from typing import Optional, Sequence

# Declaramos la API pública del módulo.
__all__ = ["_normalize_points_for_frame", "_estimate_subject_bbox"]


def _landmarks_in_unit_square(frame_landmarks, *, tolerance: float = 0.025) -> bool:
    """Comprueba si los puntos finitos están normalizados en ``[0, 1]`` con holgura.

    Admitimos un margen configurable (por defecto ±2.5 %) para absorber errores de
    interpolación o redondeos que podrían llevar coordenadas a valores como 1.01.
    Si todos los puntos finitos caen en ``[-tol, 1+tol]`` consideramos que la
    serie ya está global y no requiere reproyección adicional.
    """

    any_finite = False
    tol = max(0.0, float(tolerance))
    lower, upper = -tol, 1.0 + tol

    for lm in frame_landmarks or []:
        try:
            x = float(lm["x"])
            y = float(lm["y"])
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        any_finite = True
        if not (lower <= x <= upper and lower <= y <= upper):
            return False

    return any_finite


def _landmarks_look_like_pixels(
    frame_landmarks,
    *,
    width: float,
    height: float,
    tolerance: float = 0.05,
) -> bool:
    """Heurística para detectar coordenadas ya expresadas en píxeles.

    Consideramos "píxel" cuando:

    * al menos un valor supera 1.0 (para diferenciar de normalización), y
    * todos los valores finitos caen dentro del rango esperado del frame con un
      margen configurable (por defecto ±5 %).
    """

    any_finite = False
    any_over_unit = False
    max_abs_val = 0.0
    max_x = max(1.0, float(width))
    max_y = max(1.0, float(height))
    tol = max(0.0, float(tolerance))
    lower_x, upper_x = -tol * max_x, (1.0 + tol) * max_x
    lower_y, upper_y = -tol * max_y, (1.0 + tol) * max_y

    for lm in frame_landmarks or []:
        try:
            x = float(lm["x"])
            y = float(lm["y"])
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        any_finite = True
        any_over_unit = any_over_unit or x > 1.0 or y > 1.0
        max_abs_val = max(max_abs_val, abs(x), abs(y))
        if not (lower_x <= x <= upper_x and lower_y <= y <= upper_y):
            return False

    min_dim = min(max_x, max_y)
    scaled_enough = max_abs_val >= 0.1 * min_dim
    return any_finite and any_over_unit and scaled_enough


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

    landmarks_are_normalized = _landmarks_in_unit_square(frame_landmarks)
    landmarks_look_like_pixels = _landmarks_look_like_pixels(
        frame_landmarks, width=proc_w or orig_w, height=proc_h or orig_h
    )
    crop_vals: Optional[tuple[float, float, float, float]] = None
    if crop_box is not None:
        try:
            crop_vals = tuple(map(float, crop_box))  # type: ignore[arg-type]
        except Exception:
            crop_vals = None

    treat_as_global = False
    coord_mode = "normalized" if landmarks_are_normalized else (
        "pixel" if landmarks_look_like_pixels else "normalized"
    )
    if coord_mode == "pixel":
        # Las coordenadas ya vienen en espacio de píxel; ignoramos recortes.
        treat_as_global = True
    elif crop_vals is None:
        # Sin recorte no hay transformación adicional.
        treat_as_global = True
    else:
        crop_is_normalized = max(crop_vals) <= 1.0
        if crop_is_normalized and landmarks_are_normalized:
            # Tanto recorte como puntos están en [0,1]; ya son globales.
            treat_as_global = True

    proc_w = max(1, int(proc_w)) if math.isfinite(proc_w) else orig_w
    proc_h = max(1, int(proc_h)) if math.isfinite(proc_h) else orig_h
    sx, sy = (orig_w / float(proc_w)), (orig_h / float(proc_h))
    for idx, lm in enumerate(frame_landmarks):
        try:
            x, y = float(lm["x"]), float(lm["y"])
            if math.isnan(x) or math.isnan(y):
                continue
        except Exception:
            continue
        if coord_mode == "pixel":
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
