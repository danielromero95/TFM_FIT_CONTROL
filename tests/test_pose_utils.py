# tests/test_pose_utils.py

import os
import numpy as np
import pytest

# Ruta relativa al proyecto (ajústala si tu estructura cambia)
FILTERED_PATH = "data/processed/poses/1-Squat_Own_filtered.npy"

def test_filtered_npy_exists():
    """
    Comprueba que el archivo .npy generado por filter_interp
    existe en la ruta esperada.
    """
    assert os.path.exists(FILTERED_PATH), f"El fichero {FILTERED_PATH} no existe."

def test_load_filtered_npy():
    """
    Intenta cargar el .npy y verifica que:
    - Es un numpy.ndarray
    - Su dtype es object
    - Su tamaño coincide con el número de fotogramas (por ejemplo, >0)
    """
    arr = np.load(FILTERED_PATH, allow_pickle=True)
    assert isinstance(arr, np.ndarray), "np.load no devolvió un ndarray."
    assert arr.dtype == object, f"Se esperaba dtype=object, pero vino {arr.dtype}."
    # Opcional: comprueba que al menos haya un elemento (no vacío)
    assert arr.size > 0, "El array cargado está vacío."
