"""Pool de grafos de pose de Mediapipe seguro para múltiples hilos."""

from __future__ import annotations

import atexit
import logging
import threading
from typing import Dict, List, Tuple

from src.config.settings import build_pose_kwargs


class PoseGraphPool:
    """Agrupa instancias de ``Pose`` de Mediapipe indexadas por configuración."""

    _lock = threading.Lock()
    _free: Dict[Tuple[str, bool, int, float, float, bool, bool], List[object]] = {}
    _all: Dict[str, List[object]] = {}
    _imported = False
    mp_pose = None
    mp_drawing = None
    _logger = logging.getLogger(__name__)

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
        run_id: str,
        static_image_mode: bool,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        smooth_landmarks: bool | None = None,
        enable_segmentation: bool | None = None,
    ) -> tuple[object, Tuple[str, bool, int, float, float, bool, bool]]:
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
            str(run_id),
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
                    inst = bucket.pop()
                    cls._logger.debug(
                        "Reusing MediaPipe Pose instance (run_id=%s, model_complexity=%s).",
                        run_id,
                        pose_kwargs.get("model_complexity"),
                    )
                    return inst, key
                except IndexError:
                    pass
        cls._logger.info(
            "Creating MediaPipe Pose instance (run_id=%s, model_complexity=%s, static_image_mode=%s).",
            run_id,
            pose_kwargs.get("model_complexity"),
            pose_kwargs.get("static_image_mode"),
        )
        inst = cls.mp_pose.Pose(**pose_kwargs)  # type: ignore[arg-type]
        with cls._lock:
            cls._all.setdefault(str(run_id), []).append(inst)
        return inst, key

    @classmethod
    def release(cls, inst: object, key: Tuple[str, bool, int, float, float, bool, bool]) -> None:
        with cls._lock:
            cls._free.setdefault(key, []).append(inst)

    @classmethod
    def close_all(cls) -> None:
        with cls._lock:
            for instances in cls._all.values():
                for inst in instances:
                    try:
                        inst.close()
                    except Exception:
                        pass
            cls._all.clear()
            cls._free.clear()

    @classmethod
    def close_run(cls, run_id: str) -> None:
        """Cerrar instancias asociadas a un run sin afectar otras sesiones."""
        run_key = str(run_id)
        with cls._lock:
            instances = cls._all.pop(run_key, [])
            for inst in instances:
                try:
                    inst.close()
                except Exception:
                    pass
            for key in list(cls._free.keys()):
                if key[0] == run_key:
                    cls._free.pop(key, None)

    @classmethod
    def reset_for_run(cls) -> None:
        """Reiniciar el pool completo manualmente (no usar por defecto)."""
        cls.close_all()


atexit.register(PoseGraphPool.close_all)

__all__ = ["PoseGraphPool"]
