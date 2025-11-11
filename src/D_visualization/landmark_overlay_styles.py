"""Estilos de superposición para la visualización de marcadores corporales.
Define dataclasses que documentan cómo trazamos huesos y puntos para compartir un
lenguaje común entre los distintos componentes de depuración."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from src.config import video_landmarks_visualization as vlv

# Exportamos explícitamente los elementos principales del módulo.
__all__ = ["OverlayStyle", "RenderStats"]


@dataclass(frozen=True)
class OverlayStyle:
    """Modelo sencillo con los parámetros visuales usados al dibujar la pose.
    Mantenerlos agrupados facilita probar estilos alternativos sin tocar lógica."""

    # Grosor de las líneas que unen los puntos esqueléticos.
    connection_thickness: int = vlv.THICKNESS_DEFAULT
    # Radio de los círculos que representan cada punto clave.
    landmark_radius: int = vlv.RADIUS_DEFAULT
    # Color BGR utilizado para las conexiones entre puntos.
    connection_bgr: Tuple[int, int, int] = tuple(vlv.CONNECTION_COLOR)
    # Color BGR utilizado para los puntos individuales.
    landmark_bgr: Tuple[int, int, int] = tuple(vlv.LANDMARK_COLOR)


@dataclass(frozen=True)
class RenderStats:
    """Resumen con métricas relevantes del proceso de renderizado.
    Estas métricas ayudan a depurar cuellos de botella y evaluar calidad."""

    # Número total de frames procesados de la fuente.
    frames_in: int
    # Número total de frames que se escribieron en el vídeo destino.
    frames_written: int
    # Cantidad de frames saltados por no contener información útil.
    skipped_empty: int
    # Duración total del proceso medida en milisegundos.
    duration_ms: float
    # Codec efectivo que utilizó OpenCV para crear el vídeo resultante.
    used_fourcc: str
