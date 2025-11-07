"""Thread-safe pool of Mediapipe pose graph instances."""

from __future__ import annotations

import atexit
import threading
from typing import Dict, List, Tuple


class PoseGraphPool:
    """Pool of Mediapipe ``Pose`` graphs keyed by configuration."""

    _lock = threading.Lock()
    _free: Dict[Tuple[bool, int, float], List[object]] = {}
    _all: List[object] = []
    _imported = False
    mp_pose = None
    mp_drawing = None

    @classmethod
    def _ensure_imports(cls) -> None:
        if not cls._imported:
            from mediapipe.python.solutions import drawing_utils as mp_drawing
            from mediapipe.python.solutions import pose as mp_pose

            cls.mp_pose = mp_pose
            cls.mp_drawing = mp_drawing
            cls._imported = True

    @classmethod
    def acquire(
        cls,
        *,
        static_image_mode: bool,
        model_complexity: int,
        min_detection_confidence: float,
    ) -> tuple[object, Tuple[bool, int, float]]:
        cls._ensure_imports()
        key = (
            bool(static_image_mode),
            int(model_complexity),
            float(min_detection_confidence),
        )
        with cls._lock:
            bucket = cls._free.get(key)
            if bucket:
                try:
                    return bucket.pop(), key
                except IndexError:
                    pass
        inst = cls.mp_pose.Pose(  # type: ignore[call-arg]
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )
        with cls._lock:
            cls._all.append(inst)
        return inst, key

    @classmethod
    def release(cls, inst: object, key: Tuple[bool, int, float]) -> None:
        with cls._lock:
            cls._free.setdefault(key, []).append(inst)

    @classmethod
    def close_all(cls) -> None:
        with cls._lock:
            for inst in cls._all:
                try:
                    inst.close()
                except Exception:
                    pass
            cls._all.clear()
            cls._free.clear()


atexit.register(PoseGraphPool.close_all)

__all__ = ["PoseGraphPool"]
