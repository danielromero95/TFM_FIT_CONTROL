"""Paquete de extracción de fotogramas que expone las APIs públicas de streaming."""

from .core import extract_frames_stream, extract_processed_frames_stream
from .preprocess import extract_and_preprocess_frames
from .state import FrameInfo

__all__ = [
    "FrameInfo",
    "extract_frames_stream",
    "extract_processed_frames_stream",
    "extract_and_preprocess_frames",
]
