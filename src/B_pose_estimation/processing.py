# src/B_pose_estimation/processing.py

import numpy as np
import pandas as pd
import logging
from .estimators import PoseEstimator, CroppedPoseEstimator
from .metrics import normalize_landmarks, extract_joint_angles, calculate_distances, calculate_angular_velocity, calculate_symmetry

logger = logging.getLogger(__name__)

def extract_landmarks_from_frames(frames: list, use_crop: bool = False, visibility_threshold: float = 0.5) -> pd.DataFrame:
    """Toma imágenes, extrae landmarks y devuelve un DataFrame raw."""
    logger.info(f"Extrayendo landmarks de {len(frames)} frames. Usando crop: {use_crop}")
    estimator = CroppedPoseEstimator(min_detection_confidence=visibility_threshold) if use_crop else PoseEstimator(min_detection_confidence=visibility_threshold)
    
    rows = []
    for i, img in enumerate(frames):
        crop_box = None
        if use_crop:
            landmarks, _, crop_box = estimator.estimate(img)
        else:
            landmarks, _ = estimator.estimate(img)
            # --- ¡CAMBIO CLAVE! ---
            # Si no usamos crop, el "crop box" es la imagen entera.
            # Esto asegura que siempre tengamos esta información para el renderizador.
            if landmarks:
                h, w, _ = img.shape
                crop_box = [0, 0, w, h]
        
        row = {"frame_idx": i}
        if landmarks:
            for lm_idx, pt in enumerate(landmarks):
                row.update({f"x{lm_idx}": pt['x'], f"y{lm_idx}": pt['y'], f"z{lm_idx}": pt['z'], f"v{lm_idx}": pt['visibility']})
            
            # Ahora este bloque funcionará en ambos casos
            if crop_box:
                row.update({'crop_x1': crop_box[0], 'crop_y1': crop_box[1], 'crop_x2': crop_box[2], 'crop_y2': crop_box[3]})
        else:
            for lm_idx in range(33):
                row.update({f"x{lm_idx}": np.nan, f"y{lm_idx}": np.nan, f"z{lm_idx}": np.nan, f"v{lm_idx}": np.nan})
        rows.append(row)
        
    estimator.close()
    return pd.DataFrame(rows)

def filter_and_interpolate_landmarks(df_raw: pd.DataFrame, min_confidence: float = 0.5) -> tuple[np.ndarray, np.ndarray | None]:
    """Filtra e interpola landmarks, devuelve la secuencia y las cajas de recorte."""
    logger.info(f"Filtrando e interpolando {len(df_raw)} frames de landmarks.")
    n_frames, n_points = len(df_raw), 33
    arr = np.full((n_frames, n_points, 4), np.nan, dtype=float)
    
    for t, (_, row) in enumerate(df_raw.iterrows()):
        for i in range(n_points):
            visibility = row.get(f"v{i}", np.nan)
            if pd.notna(visibility) and visibility >= min_confidence:
                arr[t, i, 0], arr[t, i, 1], arr[t, i, 2], arr[t, i, 3] = row.get(f"x{i}"), row.get(f"y{i}"), row.get(f"z{i}"), visibility

    for i in range(n_points):
        valid_mask = ~np.isnan(arr[:, i, 0])
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 1:
            interp_indices = np.arange(n_frames)
            for j in range(3):
                arr[:, i, j] = np.interp(interp_indices, valid_indices, arr[valid_indices, i, j])

    filtered_sequence = []
    for t in range(n_frames):
        frame_landmarks = [{'x': arr[t, i, 0], 'y': arr[t, i, 1], 'z': arr[t, i, 2], 'visibility': arr[t, i, 3] if pd.notna(arr[t, i, 3]) else 0.0} for i in range(n_points)]
        filtered_sequence.append(frame_landmarks)
    
    crop_coords = df_raw[['crop_x1', 'crop_y1', 'crop_x2', 'crop_y2']].to_numpy() if 'crop_x1' in df_raw.columns else None
    return np.array(filtered_sequence, dtype=object), crop_coords

def calculate_metrics_from_sequence(sequence: np.ndarray, fps: float) -> pd.DataFrame:
    """Calcula métricas biomecánicas desde una secuencia de landmarks."""
    logger.info(f"Calculando métricas para una secuencia de {len(sequence)} frames.")
    all_metrics = []
    for idx, frame_landmarks in enumerate(sequence):
        row = {"frame_idx": idx}
        if frame_landmarks is None or any(np.isnan(lm['x']) for lm in frame_landmarks):
            row.update({'rodilla_izq': np.nan, 'rodilla_der': np.nan, 'codo_izq': np.nan, 'codo_der': np.nan, 'anchura_hombros': np.nan, 'separacion_pies': np.nan})
        else:
            norm_lm = normalize_landmarks(frame_landmarks)
            angles = extract_joint_angles(norm_lm)
            dists = calculate_distances(norm_lm)
            row.update(angles); row.update(dists)
        all_metrics.append(row)
    
    dfm = pd.DataFrame(all_metrics)
    if dfm.empty: return dfm
    
    # Corregir advertencia de fillna
    dfm_filled = dfm.ffill().bfill()
    for col in ['rodilla_izq', 'rodilla_der', 'codo_izq', 'codo_der']:
        dfm[f"vel_ang_{col}"] = calculate_angular_velocity(dfm_filled[col].tolist(), fps)
        
    dfm["sim_rodilla"] = dfm.apply(lambda r: calculate_symmetry(r['rodilla_izq'], r['rodilla_der']), axis=1)
    dfm["sim_codo"] = dfm.apply(lambda r: calculate_symmetry(r['codo_izq'], r['codo_der']), axis=1)
    return dfm