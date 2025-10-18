"""Tests covering the filtered landmark numpy artefacts."""

import os

import numpy as np
import pytest

FILTERED_PATH = "data/processed/poses/1-Squat_Own_filtered.npy"


def test_filtered_npy_exists():
    """Ensure the filtered numpy file exists on disk."""
    assert os.path.exists(FILTERED_PATH), f"File {FILTERED_PATH} does not exist."


def test_load_filtered_npy():
    """Load the numpy file and validate its structure."""
    arr = np.load(FILTERED_PATH, allow_pickle=True)
    assert isinstance(arr, np.ndarray), "np.load did not return an ndarray."
    assert arr.dtype == object, f"Expected dtype=object, received {arr.dtype}."
    assert arr.size > 0, "Loaded array is empty."
