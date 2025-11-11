"""Paquete de utilidades para visualización de marcadores.
Reexporta la API tradicional para no romper integraciones existentes mientras los
módulos internos se organizan por responsabilidad específica."""

from .landmark_drawing import draw_pose_on_frame
from .landmark_overlay_styles import OverlayStyle, RenderStats
from .landmark_renderers import render_landmarks_video, render_landmarks_video_streaming
from .landmark_video_io import transcode_video

__all__ = [
    "OverlayStyle",
    "RenderStats",
    "draw_pose_on_frame",
    "render_landmarks_video",
    "render_landmarks_video_streaming",
    "transcode_video",
]
