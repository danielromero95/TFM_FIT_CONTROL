# src/A_preprocessing/video_metadata.py (Versión con depuración mejorada)

import logging

logger = logging.getLogger(__name__)

def get_video_rotation(video_path: str) -> int:
    """
    Usa moviepy para leer los metadatos de un vídeo y detectar su rotación.
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError as e:
        # --- CAMBIO CLAVE: Ahora mostramos el error real ---
        logger.warning(f"No se pudo importar MoviePy. Puede que falte una dependencia como FFMPEG.")
        logger.warning(f"Error original: {e}")
        return 0

    try:
        with VideoFileClip(video_path) as clip:
            rotation = clip.rotation
            if rotation in [90, 180, 270]:
                logger.info(f"Rotación detectada en los metadatos del vídeo: {rotation} grados.")
                return rotation
            else:
                logger.info("No se detectó rotación en los metadatos (o es 0).")
                return 0
    except Exception as e:
        logger.error(f"Error al leer los metadatos del vídeo con MoviePy: {e}")
        return 0