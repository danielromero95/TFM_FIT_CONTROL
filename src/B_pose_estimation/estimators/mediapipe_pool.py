"""Pool de grafos de pose de Mediapipe seguro para múltiples hilos."""

from __future__ import annotations

import atexit
import threading
from typing import Dict, List, Tuple

from src.config.settings import build_pose_kwargs


class PoseGraphPool:
    """Agrupa instancias de ``Pose`` de Mediapipe indexadas por configuración."""

    _lock = threading.Lock()
    _free: Dict[Tuple[bool, int, float, float, bool, bool], List[object]] = {}
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
        min_tracking_confidence: float,
        smooth_landmarks: bool | None = None,
        enable_segmentation: bool | None = None,
    ) -> tuple[object, Tuple[bool, int, float, float, bool, bool]]:
        cls._ensure_imports()
        pose_kwargs = build_pose_kwargs(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
        )
        key = (
            bool(pose_kwargs["static_image_mode"]),
            int(pose_kwargs["model_complexity"]),
            float(pose_kwargs["min_detection_confidence"]),
            float(pose_kwargs["min_tracking_confidence"]),
            bool(pose_kwargs.get("smooth_landmarks", True)),
            bool(pose_kwargs.get("enable_segmentation", False)),
        )
        with cls._lock:
            bucket = cls._free.get(key)
            if bucket:
                try:
                    return bucket.pop(), key
                except IndexError:
                    pass
        inst = cls.mp_pose.Pose(**pose_kwargs)  # type: ignore[arg-type]
        with cls._lock:
            cls._all.append(inst)
        return inst, key

    @classmethod
    def release(cls, inst: object, key: Tuple[bool, int, float, float, bool, bool]) -> None:
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
