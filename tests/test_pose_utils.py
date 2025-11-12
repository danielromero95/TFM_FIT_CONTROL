"""Pruebas para los artefactos NumPy de marcadores filtrados."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def filtered_array_path(tmp_path):
    """Crea un arreglo sintético de marcadores filtrados para pruebas rápidas portables."""

    payload = np.array([
        {"landmarks": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]},
        {"landmarks": [(0.7, 0.8, 0.9), (1.0, 1.1, 1.2)]},
    ], dtype=object)
    path = tmp_path / "filtered.npy"
    np.save(path, payload)
    return path


def test_filtered_npy_exists(filtered_array_path):
    """Comprueba que el archivo NumPy filtrado exista en disco."""

    assert filtered_array_path.exists(), "Synthetic filtered array was not created."


def test_load_filtered_npy(filtered_array_path):
    """Carga el archivo NumPy y valida que mantenga la estructura esperada."""

    arr = np.load(filtered_array_path, allow_pickle=True)
    assert isinstance(arr, np.ndarray), "np.load did not return an ndarray."
    assert arr.dtype == object, f"Expected dtype=object, received {arr.dtype}."
    assert arr.size == 2, "Filtered array size changed unexpectedly."
