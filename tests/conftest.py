# tests/conftest.py
"""Utilidades de configuración comunes para la batería de pruebas."""
from __future__ import annotations

import types
from pathlib import Path
import sys
from typing import Any


def _install_cv2_stub() -> None:
    """Proporciona un *stub* ligero de cv2 para evitar depender de OpenCV en las pruebas."""

    if "cv2" in sys.modules:
        return

    try:
        import numpy as _np
    except Exception:  # pragma: no cover - NumPy está disponible en la CI
        _np = None

    class _StubVideoCapture:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self._opened = True
            self._props: dict[int, float] = {}

        def isOpened(self) -> bool:  # pragma: no cover - respaldo para pruebas de humo
            return self._opened

        def read(self) -> tuple[bool, Any]:  # pragma: no cover - respaldo para pruebas de humo
            if _np is not None:
                frame = _np.zeros((1, 1, 3), dtype=_np.uint8)
            else:
                frame = None
            return False, frame

        def get(self, prop: int) -> float:
            return self._props.get(prop, 0.0)

        def set(self, prop: int, value: float) -> bool:  # pragma: no cover - respaldo para pruebas de humo
            self._props[prop] = float(value)
            return True

        def release(self) -> None:
            self._opened = False

    class _StubVideoWriter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self._opened = True
            self._frames: list[Any] = []

        def isOpened(self) -> bool:
            return self._opened

        def write(self, frame: Any) -> None:  # pragma: no cover - respaldo para pruebas de humo
            self._frames.append(frame)

        def release(self) -> None:
            self._opened = False

    def _identity(frame: Any, *_args: Any, **_kwargs: Any) -> Any:
        return frame

    def _video_writer_fourcc(*_codes: str) -> int:  # pragma: no cover - respaldo para pruebas de humo
        return 0

    cv2_stub = types.SimpleNamespace(
        COLOR_BGR2RGB=0,
        COLOR_BGR2GRAY=1,
        CAP_PROP_FPS=0,
        CAP_PROP_FRAME_COUNT=1,
        CAP_PROP_POS_FRAMES=2,
        CAP_PROP_POS_MSEC=3,
        CAP_PROP_FRAME_WIDTH=4,
        CAP_PROP_FRAME_HEIGHT=5,
        CAP_PROP_FOURCC=6,
        ROTATE_90_CLOCKWISE=90,
        ROTATE_180=180,
        ROTATE_90_COUNTERCLOCKWISE=270,
        INTER_AREA=1,
        INTER_LINEAR=2,
        VideoCapture=_StubVideoCapture,
        VideoWriter=_StubVideoWriter,
        VideoWriter_fourcc=_video_writer_fourcc,
        cvtColor=_identity,
        rotate=_identity,
        resize=_identity,
        line=lambda *_args, **_kwargs: None,
        circle=lambda *_args, **_kwargs: None,
        putText=lambda *_args, **_kwargs: None,
    )

    sys.modules["cv2"] = cv2_stub


_install_cv2_stub()


# Repo root = parent de 'tests'
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Asegura que los paquetes en src/ son importables
src_str = str(SRC)
if src_str not in sys.path:
    sys.path.insert(0, src_str)

# (Opcional) si quieres mantener también la raíz, ponla detrás
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.append(root_str)
