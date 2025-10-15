# src/D_modeling/count_reps.py
"""
Contiene la lógica para el conteo de repeticiones a partir de series
temporales de métricas (ej. ángulos).
"""
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import logging

# Importamos la configuración centralizada
from src import config

logger = logging.getLogger(__name__)

def count_reps_by_valleys(
        angle_sequence: list, 
        peak_height_thresh: float, 
        prominence: float = 10, 
        distance: int = 15
    ) -> int:
    """
    Cuenta repeticiones detectando los valles (puntos más bajos) en una
    secuencia de ángulos. Este método es más robusto ante el ruido.

    Args:
        angle_sequence (list): Serie temporal de ángulos (en grados).
        peak_height_thresh (float): El umbral que el ángulo debe cruzar (hacia abajo)
                                    para ser considerado un valle válido. Corresponde
                                    al 'low_thresh' de una sentadilla.
        prominence (float): La prominencia mínima de un pico. Filtra valles poco profundos.
        distance (int): La distancia mínima (en número de frames) entre valles.
                        Evita contar dobles fondos en una misma repetición.

    Returns:
        int: Número de repeticiones detectadas.
    """
    # Invertimos la señal para que los valles se conviertan en picos
    inverted_angles = -np.array(angle_sequence)
    
    # El umbral de altura también se invierte. Buscamos picos por encima de este valor.
    inverted_threshold = -peak_height_thresh
    
    # Usamos find_peaks de SciPy para encontrar los valles
    valleys, properties = find_peaks(
        inverted_angles, 
        height=inverted_threshold, 
        prominence=prominence, 
        distance=distance
    )
    
    logger.info(f"Detección de picos encontró {len(valleys)} valles válidos.")
    
    return len(valleys)


def count_repetitions_from_df(
        df_metrics: pd.DataFrame, 
        angle_column: str = 'rodilla_izq', 
        low_thresh: float = config.SQUAT_LOW_THRESH
    ) -> int:
    """
    Toma un DataFrame de métricas, extrae la columna de ángulo indicada,
    y devuelve el número de repeticiones contadas usando el método de detección de valles.
    """
    if df_metrics.empty or angle_column not in df_metrics.columns:
        logger.warning(f"La columna '{angle_column}' no se encontró o el DataFrame está vacío. Se devuelven 0 repeticiones.")
        return 0

    # Rellenamos valores nulos para que no interrumpan el análisis
    # Usamos forward-fill y luego backward-fill para asegurar que no queden NaNs
    angles = df_metrics[angle_column].ffill().bfill().tolist()

    if not angles:
        return 0

    # Llamamos a nuestro nuevo y más robusto algoritmo de conteo
    n_reps = count_reps_by_valleys(
        angle_sequence=angles,
        peak_height_thresh=low_thresh
    )
    
    return n_reps