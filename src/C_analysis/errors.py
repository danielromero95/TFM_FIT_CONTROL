"""Excepciones específicas del dominio utilizadas en los servicios de análisis."""


class PipelineError(Exception):
    """Excepción base para fallos fatales en la *pipeline*."""


class VideoOpenError(PipelineError):
    """Se lanza cuando no es posible abrir un archivo de vídeo para lectura."""


class NoFramesExtracted(PipelineError):
    """Se lanza cuando la extracción de fotogramas no produce ningún cuadro."""
