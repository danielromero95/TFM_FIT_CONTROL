"""Rutinas de dibujo para superponer poses sobre frames de vídeo.
Reúne la lógica de anotación para reutilizarla en cualquier flujo de renderizado y
mantener un estilo consistente sin reimplementar primitivas en cada módulo."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import cv2

from src.config import video_landmarks_visualization as vlv

from .landmark_overlay_styles import OverlayStyle

# Exponemos las utilidades de dibujo más relevantes.
__all__ = ["draw_pose_on_frame", "_adaptive_style_for_region"]


def draw_pose_on_frame(
    frame,
    points_xy: Mapping[int, tuple[int, int]],
    *,
    connections: Sequence[Tuple[int, int]] = tuple(vlv.POSE_CONNECTIONS),
    style: OverlayStyle = OverlayStyle(),
) -> None:
    """Dibuja conexiones y puntos sobre ``frame`` usando los valores dados.
    Centralizar el trazado evita inconsistencias visuales cuando distintos flujos
    desean inspeccionar los mismos datos de pose."""

    for a, b in connections:
        if a in points_xy and b in points_xy:
            # Unimos cada par de puntos con líneas coloreadas para sugerir el esqueleto.
            cv2.line(frame, points_xy[a], points_xy[b], style.connection_bgr, style.connection_thickness)
    for p in points_xy.values():
        # Representamos los puntos clave con círculos rellenos para que la pose sea legible.
        cv2.circle(frame, p, style.landmark_radius, style.landmark_bgr, -1)


def _adaptive_style_for_region(width: int, height: int) -> OverlayStyle:
    """Calcula un estilo proporcional al tamaño del recorte trabajado.
    Ajustamos grosor y radio para que el sujeto se vea legible sin importar el zoom,
    evitando líneas gigantes cuando el recorte es pequeño o viceversa."""

    base = max(1, min(int(width), int(height)))
    thickness = max(2, round(base / 320 * 2))
    radius = max(3, round(base / 320 * 4))
    return OverlayStyle(connection_thickness=thickness, landmark_radius=radius)
