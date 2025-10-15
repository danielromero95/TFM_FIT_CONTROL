# tests/test_video_utils.py

import numpy as np
import cv2
import pytest

from src.A_preprocessing.video_utils import validate_video

def create_dummy_video(path, width=64, height=48, num_frames=30, fps=15):
    """
    Crea un vídeo muy sencillo en 'path' usando OpenCV (frames en negro),
    para usar en los tests de validación.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("No se pudo abrir VideoWriter para crear vídeo de prueba")
    # Generar y escribir frames negros
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(num_frames):
        writer.write(frame)
    writer.release()

def test_validate_video_ok(tmp_path):
    """
    Test: el vídeo existe y es válido.
    - Comprueba que validate_video devuelve un dict con keys 'fps', 'frame_count' y 'duration'.
    - FPS aproximado a 15, frame_count igual a 30, duration ≈ 2.0 s.
    """
    dummy_video = tmp_path / "test_ok.mp4"
    create_dummy_video(str(dummy_video), width=64, height=48, num_frames=30, fps=15)

    info = validate_video(str(dummy_video))
    assert isinstance(info, dict)
    assert abs(info["fps"] - 15) < 1e-3
    assert info["frame_count"] == 30
    assert abs(info["duration"] - 2.0) < 1e-3

def test_validate_video_no_exist():
    """
    Test: si la ruta no existe, validate_video debe levantar IOError.
    """
    with pytest.raises(IOError):
        validate_video("ruta/que/no/existe/video.avi")

def test_validate_video_corrupt(tmp_path, monkeypatch):
    """
    Test: simular un vídeo abierto pero con FPS=0 (video corrupto).
    - Se fuerza un VideoCapture falso que devuelve FPS=0.
    - Debe levantar ValueError.
    """
    dummy_video = tmp_path / "corrupt.mp4"
    create_dummy_video(str(dummy_video), width=64, height=48, num_frames=10, fps=10)

    # FakeCap simula un VideoCapture con fps=0 para forzar el error
    class FakeCap:
        def __init__(self, path):
            self.opened = True
        def isOpened(self):
            return True
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 0  # forzar FPS=0
            elif prop == cv2.CAP_PROP_FRAME_COUNT:
                return 10
            return 0
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda x: FakeCap(x))

    with pytest.raises(ValueError):
        validate_video(str(dummy_video))
