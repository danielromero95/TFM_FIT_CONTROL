# src/pipeline.py

import logging
import pandas as pd
import os
import cv2
from src import config
from src.A_preprocessing.frame_extraction import extract_and_preprocess_frames
from src.B_pose_estimation.processing import extract_landmarks_from_frames, filter_and_interpolate_landmarks, calculate_metrics_from_sequence
from src.D_modeling.count_reps import count_repetitions_from_df
from src.F_visualization.video_renderer import render_landmarks_on_video_hq

logger = logging.getLogger(__name__)

def run_full_pipeline_in_memory(video_path: str, settings: dict, progress_callback=None):
    """
    Ejecuta el pipeline completo de análisis en memoria, desde la extracción
    hasta el conteo de repeticiones y la generación de salidas de depuración.
    """
    def notify(value, message):
        """Función helper para notificar progreso y loguear en un paso."""
        logger.info(message)
        if progress_callback:
            progress_callback(value)

    # --- Preparación ---
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = settings.get('output_dir', '.')
    session_dir = os.path.join(output_dir, base_name)
    os.makedirs(session_dir, exist_ok=True)
    
    # --- FASE 1: Extracción y Auto-Rotación ---
    notify(5, "FASE 1: Extrayendo y rotando fotogramas...")
    original_frames, fps = extract_and_preprocess_frames(
        video_path=video_path,
        rotate=settings.get('rotate', None), 
        sample_rate=settings.get('sample_rate', 1)
    )
    if not original_frames:
        raise ValueError("No se pudieron extraer fotogramas del vídeo.")
    
    # Preprocesamiento (Redimensionado) para el modelo
    target_size = (settings.get('target_width'), settings.get('target_height'))
    processed_frames = [cv2.resize(f, target_size) for f in original_frames]
    
    # --- FASE 2: Estimación de Pose ---
    notify(25, "FASE 2: Estimando pose en los fotogramas...")
    df_raw_landmarks = extract_landmarks_from_frames(
        frames=processed_frames, 
        use_crop=settings.get('use_crop', True),
        visibility_threshold=config.MIN_DETECTION_CONFIDENCE
    )

    # --- FASE 3: Filtrado e Interpolación ---
    notify(50, "FASE 3: Filtrando e interpolando landmarks...")
    filtered_sequence, crop_boxes = filter_and_interpolate_landmarks(df_raw_landmarks)
    
    # --- FASE EXTRA: Renderizado de vídeo (si se solicita) ---
    output_video_path = None # Inicializamos como None
    if settings.get('generate_debug_video', False):
        notify(65, "FASE EXTRA: Renderizando vídeo de depuración HQ...")
        output_video_path = os.path.join(session_dir, f"{base_name}_debug_HQ.mp4")
        render_landmarks_on_video_hq(original_frames, filtered_sequence, crop_boxes, output_video_path, fps)

    # --- FASE 4: Cálculo de Métricas ---
    notify(75, "FASE 4: Calculando métricas biomecánicas...")
    df_metrics = calculate_metrics_from_sequence(filtered_sequence, fps)
    
    # --- FASE 5: Conteo de Repeticiones ---
    notify(90, "FASE 5: Contando repeticiones...")
    n_reps = count_repetitions_from_df(
        df_metrics,
        low_thresh=settings.get('low_thresh', config.SQUAT_LOW_THRESH)
    )
    
    # Guardado de datos de depuración (si se solicita)
    if settings.get('debug_mode', False):
        logger.info("MODO DEPURACIÓN: Guardando datos intermedios...")
        df_raw_landmarks.to_csv(os.path.join(session_dir, f"{base_name}_1_raw_landmarks.csv"), index=False)
        df_metrics.to_csv(os.path.join(session_dir, f"{base_name}_2_metrics.csv"), index=False)
        logger.info(f"Archivos de depuración guardados en: {session_dir}")

    notify(100, "PIPELINE COMPLETADO")
    
    # --- CAMBIO CLAVE: Devolvemos la ruta del vídeo junto al resto de resultados ---
    return {
        "repeticiones_contadas": n_reps,
        "dataframe_metricas": df_metrics,
        "debug_video_path": output_video_path 
    }