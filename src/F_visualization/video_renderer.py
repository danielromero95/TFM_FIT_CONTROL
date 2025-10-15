# src/F_visualization/video_renderer.py (Versión Definitiva)

import cv2
import numpy as np
import logging
from src import config

logger = logging.getLogger(__name__)

def render_landmarks_on_video_hq(
    original_frames: list,
    landmarks_sequence: np.ndarray,
    crop_boxes: np.ndarray,
    output_path: str,
    fps: float
):
    """
    Dibuja landmarks (transformando coordenadas correctamente) sobre los
    fotogramas originales de alta calidad y guarda el vídeo.
    """
    logger.info(f"Iniciando renderizado de vídeo HQ en: {output_path}")
    if not original_frames:
        logger.warning("No hay fotogramas para renderizar.")
        return

    orig_h, orig_w, _ = original_frames[0].shape
    proc_w, proc_h = config.DEFAULT_TARGET_WIDTH, config.DEFAULT_TARGET_HEIGHT
    
    # Factores de escala para pasar del mundo de 256x256 al mundo de alta resolución
    scale_x = orig_w / proc_w
    scale_y = orig_h / proc_h

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (orig_w, orig_h))
    if not writer.isOpened():
        logger.error(f"No se pudo abrir VideoWriter para la ruta: {output_path}")
        return

    for i, frame in enumerate(original_frames):
        annotated_frame = frame.copy()
        
        if i < len(landmarks_sequence) and landmarks_sequence[i] is not None:
            frame_landmarks = landmarks_sequence[i]
            if all(np.isnan(lm['x']) for lm in frame_landmarks):
                writer.write(annotated_frame)
                continue
            
            points_to_draw = {}
            crop_box = crop_boxes[i] if crop_boxes is not None and i < len(crop_boxes) and not np.isnan(crop_boxes[i]).all() else None

            for lm_idx, lm in enumerate(frame_landmarks):
                if not np.isnan(lm['x']):
                    
                    # --- LÓGICA DE TRANSFORMACIÓN CORREGIDA ---
                    
                    if crop_box is not None:
                        # --- Caso CON CROP ---
                        # 1. Convertir landmark de [0,1] (relativo al crop) a píxeles en la imagen procesada (256x256)
                        x1_p, y1_p, x2_p, y2_p = crop_box
                        crop_w_p = x2_p - x1_p
                        crop_h_p = y2_p - y1_p
                        
                        abs_x_p = x1_p + lm['x'] * crop_w_p
                        abs_y_p = y1_p + lm['y'] * crop_h_p
                        
                        # 2. Escalar el punto de la imagen procesada a la imagen original (alta resolución)
                        final_x = int(abs_x_p * scale_x)
                        final_y = int(abs_y_p * scale_y)
                        
                    else:
                        # --- Caso SIN CROP ---
                        # Los landmarks son relativos a la imagen procesada (256x256). Solo necesitamos escalarlos.
                        final_x = int(lm['x'] * orig_w)
                        final_y = int(lm['y'] * orig_h)
                        
                    points_to_draw[lm_idx] = (final_x, final_y)

            # Dibujar el esqueleto con los puntos ya transformados
            for p1_idx, p2_idx in config.POSE_CONNECTIONS:
                if p1_idx in points_to_draw and p2_idx in points_to_draw:
                    cv2.line(annotated_frame, points_to_draw[p1_idx], points_to_draw[p2_idx], config.CONNECTION_COLOR, 2)
            for point in points_to_draw.values():
                cv2.circle(annotated_frame, point, 4, config.LANDMARK_COLOR, -1)
        
        writer.write(annotated_frame)

    writer.release()
    logger.info("Vídeo de depuración HQ renderizado con éxito.")