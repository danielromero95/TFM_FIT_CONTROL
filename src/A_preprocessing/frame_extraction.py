# src/A_preprocessing/frame_extraction.py (Versión con auto-rotación)

import cv2
import os
import logging
from src import config
from .video_metadata import get_video_rotation

logger = logging.getLogger(__name__)

def extract_and_preprocess_frames(
        video_path,
        sample_rate=1,
        # --- CAMBIO: 'rotate' ahora es opcional. Si es None, se auto-detecta ---
        rotate: int | None = None,
        progress_callback=None
    ):
    """
    Extrae fotogramas, detecta y aplica la rotación automáticamente,
    y los devuelve como una lista de imágenes en memoria A TAMAÑO COMPLETO.
    """
    logger.info(f"Iniciando extracción para: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in config.VIDEO_EXTENSIONS:
        raise ValueError(f"Extensión de vídeo no soportada: '{ext}'.")

    # --- CAMBIO: Detectamos la rotación si no se ha especificado una manualmente ---
    if rotate is None:
        rotate = get_video_rotation(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Propiedades del vídeo: {frame_count} frames, {fps:.2f} FPS")

    original_frames = []
    idx = 0
    last_percent_done = -1

    while True:
        ret, frame = cap.read()
        if not ret: break

        if progress_callback and frame_count > 0:
            percent_done = int((idx / frame_count) * 100)
            if percent_done > last_percent_done:
                progress_callback(percent_done)
                last_percent_done = percent_done

        if idx % sample_rate == 0:
            if rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            original_frames.append(frame)
        
        idx += 1

    cap.release()
    logger.info(f"Proceso completado. Se han extraído {len(original_frames)} fotogramas en memoria.")
    return original_frames, fps