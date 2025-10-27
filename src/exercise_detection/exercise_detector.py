# src/exercise_detection/exercise_detector.py
"""Heuristic exercise detection based on MediaPipe pose landmarks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.core.types import ExerciseType, ViewType, as_exercise, as_view

logger = logging.getLogger(__name__)

# --- Tunable thresholds -------------------------------------------------------

MIN_VALID_FRAMES = 20
SQUAT_KNEE_ROM_THRESHOLD_DEG = 40.0
SQUAT_HIP_ROM_THRESHOLD_DEG = 30.0
SQUAT_PELVIS_DISPLACEMENT_THRESHOLD = 0.20  # normalized by torso length
SQUAT_KNEE_MIN_ANGLE_DEG = 130.0
SQUAT_HIP_MIN_ANGLE_DEG = 150.0
SQUAT_SIDE_SYMMETRY_TOLERANCE_DEG = 18.0

BENCH_ELBOW_ROM_THRESHOLD_DEG = 55.0
BENCH_WRIST_HORIZONTAL_THRESHOLD = 0.08
BENCH_TORSO_TILT_THRESHOLD_DEG = 40.0  # degrees away from vertical
BENCH_TORSO_MIN_DEG = 55.0
BENCH_PELVIS_DROP_NORM_MAX = 0.12
BENCH_LOWER_ROM_MAX_DEG = 25.0

DEADLIFT_HIP_ROM_THRESHOLD_DEG = 40.0
DEADLIFT_WRIST_VERTICAL_THRESHOLD = 0.12
DEADLIFT_TORSO_UPRIGHT_TARGET_DEG = 35.0

VIEW_FRONT_WIDTH_THRESHOLD = 0.55      # normalized shoulder width / torso length
VIEW_WIDTH_STD_THRESHOLD = 0.12
YAW_FRONT_MAX_DEG = 20.0
YAW_SIDE_MIN_DEG = 25.0
Z_DELTA_FRONT_MAX = 0.08               # normalized MediaPipe units (smaller is more front)
SIDE_WIDTH_MAX = 0.50                  # if width_norm mean ≤ this, likely side
ANKLE_FRONT_WIDTH_THRESHOLD = 0.50
ANKLE_SIDE_WIDTH_MAX = 0.40
ANKLE_WIDTH_STD_THRESHOLD = 0.12

VIEW_SCORE_PER_EVIDENCE_THRESHOLD = 0.22
VIEW_MARGIN_PER_EVIDENCE_THRESHOLD = 0.10
VIEW_FRONT_FALLBACK_YAW_DEG = 24.0
VIEW_SIDE_FALLBACK_YAW_DEG = 27.0

CLASSIFICATION_MARGIN = 0.12
MIN_CONFIDENCE_SCORE = 0.45

DEFAULT_SAMPLING_RATE = 30.0

FEATURE_NAMES: Tuple[str, ...] = (
    "knee_angle_left",
    "knee_angle_right",
    "hip_angle_left",
    "hip_angle_right",
    "elbow_angle_left",
    "elbow_angle_right",
    "shoulder_angle_left",
    "shoulder_angle_right",
    "pelvis_y",
    "torso_length",
    "wrist_left_x",
    "wrist_left_y",
    "wrist_right_x",
    "wrist_right_y",
    "shoulder_width_norm",
    "shoulder_yaw_deg",
    "shoulder_z_delta_abs",
    "torso_tilt_deg",
    "ankle_width_norm",
)


@dataclass(frozen=True)
class DetectionResult:
    label: ExerciseType
    view: ViewType
    confidence: float


def make_detection_result(label: str, view: str, confidence: float) -> DetectionResult:
    """Normalize legacy string outputs into the enum-based ``DetectionResult``."""

    return DetectionResult(as_exercise(label), as_view(view), float(confidence))


@dataclass
class FeatureSeries:
    """Container for the extracted time-series and metadata."""

    data: Dict[str, np.ndarray]
    sampling_rate: float
    valid_frames: int
    total_frames: int


class _FeatureBuffer:
    """Lightweight dynamic array used by the incremental extractor."""

    __slots__ = ("array", "size", "dtype")

    def __init__(self, initial_capacity: int = 512, *, dtype: Any = float) -> None:
        dtype_obj = np.dtype(dtype)
        self.array = np.empty(int(initial_capacity), dtype=dtype_obj)
        self.size = 0
        self.dtype = dtype_obj

    def append(self, value: float) -> None:
        if self.size >= self.array.size:
            self._grow()
        self.array[self.size] = value
        self.size += 1

    def to_array(self) -> np.ndarray:
        if self.size == 0:
            return np.empty(0, dtype=self.dtype)
        return np.asarray(self.array[: self.size], dtype=float)

    def _grow(self) -> None:
        new_capacity = max(self.array.size * 2, 1024)
        new_array = np.empty(new_capacity, dtype=self.dtype)
        new_array[: self.size] = self.array[: self.size]
        self.array = new_array


class IncrementalExerciseFeatureExtractor:
    """Incrementally subsample frames and classify the exercise in one pass."""

    def __init__(
        self,
        *,
        target_fps: float,
        source_fps: float,
        max_frames: int = 300,
    ) -> None:
        self.target_fps = max(float(target_fps or 0.0), 0.1)
        self.source_fps = float(source_fps or 0.0)
        self.max_frames = max(1, int(max_frames))
        self._stride = self._compute_stride()
        self._pose = None
        self._pose_landmark = None
        self._feature_buffers: Dict[str, _FeatureBuffer] = {
            name: _FeatureBuffer() for name in FEATURE_NAMES
        }
        self._samples = 0
        self._valid_samples = 0
        self._initialised = False
        self._done = False

    def _compute_stride(self) -> int:
        if self.source_fps <= 0.0:
            approx_source = max(self.target_fps, DEFAULT_SAMPLING_RATE)
        else:
            approx_source = self.source_fps
        stride = int(round(approx_source / self.target_fps)) if self.target_fps > 0 else 1
        return max(1, stride)

    def _ensure_pose(self) -> None:
        if self._initialised:
            return
        try:
            from mediapipe.python.solutions import pose as mp_pose_module
        except ImportError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("MediaPipe is not available for exercise detection") from exc

        pose_kwargs = dict(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = mp_pose_module.Pose(**pose_kwargs)
        self._pose_landmark = mp_pose_module.PoseLandmark
        self._initialised = True

    def add_frame(self, frame_idx: int, frame: np.ndarray, ts_ms: float) -> None:
        """Optionally sample ``frame`` and update the incremental feature buffers."""

        if self._done or frame_idx % self._stride != 0:
            return
        if self._samples >= self.max_frames:
            self._done = True
            return

        self._ensure_pose()
        assert self._pose is not None and self._pose_landmark is not None  # for type checkers

        feature_lists: Dict[str, _FeatureBuffer] = self._feature_buffers
        valid = _process_frame(frame, self._pose, self._pose_landmark, feature_lists)
        self._samples += 1
        if valid:
            self._valid_samples += 1
        if not valid:
            # ``_append_nan`` was already called by ``_process_frame``.
            pass

    def finalize(self) -> DetectionResult:
        """Close resources and classify the aggregated feature series."""

        try:
            if not self._initialised or self._samples == 0:
                return DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)

            data = {name: buf.to_array() for name, buf in self._feature_buffers.items()}
            sampling_rate = self._effective_sampling_rate()
            features = FeatureSeries(
                data=data,
                sampling_rate=float(sampling_rate),
                valid_frames=int(self._valid_samples),
                total_frames=int(self._samples),
            )
            try:
                label, view, confidence = classify_features(features)
            except Exception:  # pragma: no cover - classification fallback
                logger.exception("Failed to classify incremental detection features")
                return DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)
            return make_detection_result(label, view, confidence)
        finally:
            if self._initialised and self._pose is not None:
                try:
                    self._pose.close()
                except Exception:  # pragma: no cover - best effort
                    logger.debug("Failed to close MediaPipe Pose instance", exc_info=True)
            self._pose = None
            self._pose_landmark = None
            self._initialised = False

    def _effective_sampling_rate(self) -> float:
        if self.source_fps > 0 and self._stride > 0:
            return float(self.source_fps / self._stride)
        if self.target_fps > 0:
            return float(self.target_fps)
        return DEFAULT_SAMPLING_RATE

# --- Public API ----------------------------------------------------------------


def detect_exercise(video_path: str, max_frames: int = 300) -> Tuple[str, str, float]:
    """Detect the exercise label, view and confidence for ``video_path``.

    Any failure during extraction or classification returns ``("unknown", "unknown", 0.0)``.
    """

    try:
        features = extract_features(video_path, max_frames=max_frames)
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("Failed to extract features for exercise detection")
        return "unknown", "unknown", 0.0

    if features.total_frames == 0 or features.valid_frames < MIN_VALID_FRAMES:
        logger.info(
            "Inconclusive detection: valid frames %d of %d",
            features.valid_frames,
            features.total_frames,
        )
        return "unknown", "unknown", 0.0

    try:
        label, view, confidence = classify_features(features)
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("Failed to classify the extracted features")
        return "unknown", "unknown", 0.0

    logger.info(
        "Ejercicio detectado: label=%s view=%s confidence=%.2f (frames=%d/%d)",
        label,
        view,
        confidence,
        features.valid_frames,
        features.total_frames,
    )
    return label, view, confidence


def detect_exercise_from_frames(
    frames: Iterable[np.ndarray], *, fps: float, max_frames: int = 300
) -> Tuple[str, str, float]:
    """
    Detect exercise using an iterable of preprocessed frames (already rotated/resized).
    Consumes up to ``max_frames`` frames from the iterable. Returns (label, view, confidence).
    """

    try:
        features = extract_features_from_frames(
            frames, fps=fps, max_frames=max_frames
        )
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("Failed to extract features for exercise detection (streaming)")
        return "unknown", "unknown", 0.0

    if features.total_frames == 0 or features.valid_frames < MIN_VALID_FRAMES:
        logger.info(
            "Inconclusive detection (streaming): valid frames %d of %d",
            features.valid_frames,
            features.total_frames,
        )
        return "unknown", "unknown", 0.0

    try:
        label, view, confidence = classify_features(features)
    except Exception:  # pragma: no cover
        logger.exception("Failed to classify features (streaming)")
        return "unknown", "unknown", 0.0

    return label, view, confidence


def detect_exercise_result(video_path: str, max_frames: int = 300) -> DetectionResult:
    """Return the structured :class:`DetectionResult` for the given ``video_path``."""

    label, view, confidence = detect_exercise(video_path, max_frames)
    return make_detection_result(label, view, confidence)


# --- Feature extraction --------------------------------------------------------


def extract_features(video_path: str, max_frames: int = 300) -> FeatureSeries:
    """Extract pose-derived time series required for exercise classification."""

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise IOError(f"Could not open the video for detection: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    max_frames = max(1, int(max_frames))

    if frame_count > 0:
        sample_count = min(frame_count, max_frames)
        sample_indices = np.linspace(0, frame_count - 1, num=sample_count, dtype=int)
    else:
        sample_indices = None
        sample_count = max_frames

    try:
        from mediapipe.python.solutions import pose as mp_pose_module
    except ImportError as exc:  # pragma: no cover - environment safeguard
        capture.release()
        raise RuntimeError("MediaPipe is not available in the runtime environment") from exc

    pose_landmark = mp_pose_module.PoseLandmark

    pose_kwargs = dict(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    feature_lists: Dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}

    total_processed = 0
    valid_frames = 0

    with mp_pose_module.Pose(**pose_kwargs) as pose:
        if sample_indices is not None:
            for idx in sample_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                success, frame = capture.read()
                total_processed += 1
                if not success:
                    _append_nan(feature_lists)
                    continue
                valid = _process_frame(frame, pose, pose_landmark, feature_lists)
                if valid:
                    valid_frames += 1
        else:
            while total_processed < sample_count:
                success, frame = capture.read()
                if not success:
                    break
                total_processed += 1
                valid = _process_frame(frame, pose, pose_landmark, feature_lists)
                if valid:
                    valid_frames += 1
            # Fill the remainder with NaNs if the video was shorter than expected
            while total_processed < sample_count:
                _append_nan(feature_lists)
                total_processed += 1

    capture.release()

    data = {key: np.asarray(values, dtype=float) for key, values in feature_lists.items()}

    sampling_rate = _estimate_sampling_rate(fps, frame_count, total_processed)

    percent_valid = (valid_frames / total_processed * 100.0) if total_processed else 0.0
    logger.info(
        "Exercise detection extraction: frames=%d valid=%d (%.1f%%) sample_rate=%.2f",
        total_processed,
        valid_frames,
        percent_valid,
        sampling_rate,
    )

    return FeatureSeries(
        data=data,
        sampling_rate=float(sampling_rate),
        valid_frames=int(valid_frames),
        total_frames=int(total_processed),
    )


def extract_features_from_frames(
    frames: Iterable[np.ndarray], *, fps: float, max_frames: int = 300
) -> FeatureSeries:
    """
    Build the same feature series as ``extract_features(video_path)`` but reading from an
    iterable of frames. Uses the same MediaPipe Pose setup and ``_process_frame`` per-frame
    logic.
    """

    try:
        from mediapipe.python.solutions import pose as mp_pose_module
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MediaPipe is not available in the runtime environment") from exc

    pose_landmark = mp_pose_module.PoseLandmark
    pose_kwargs = dict(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    feature_lists: Dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}

    total_processed = 0
    valid_frames = 0

    with mp_pose_module.Pose(**pose_kwargs) as pose:
        for frame in frames:
            if total_processed >= max_frames:
                break
            total_processed += 1
            valid = _process_frame(frame, pose, pose_landmark, feature_lists)
            if valid:
                valid_frames += 1

    data = {key: np.asarray(values, dtype=float) for key, values in feature_lists.items()}

    sampling_rate = float(fps) if (fps and fps > 0) else DEFAULT_SAMPLING_RATE

    logger.info(
        "Exercise detection (streaming) extraction: frames=%d valid=%d (%.1f%%) sample_rate=%.2f",
        total_processed,
        valid_frames,
        (valid_frames / total_processed * 100.0) if total_processed else 0.0,
        sampling_rate,
    )

    return FeatureSeries(
        data=data,
        sampling_rate=sampling_rate,
        valid_frames=int(valid_frames),
        total_frames=int(total_processed),
    )


def _process_frame(
    frame: np.ndarray,
    pose: Any,
    pose_landmark: Any,
    feature_lists: Dict[str, list[float]],
) -> bool:
    """Run pose detection on a frame and append measurements to ``feature_lists``."""

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        _append_nan(feature_lists)
        return False

    landmarks = results.pose_landmarks.landmark

    def pt(index: int) -> np.ndarray:
        landmark = landmarks[index]
        return np.array([landmark.x, landmark.y, landmark.z], dtype=float)

    left_hip = pt(pose_landmark.LEFT_HIP.value)
    right_hip = pt(pose_landmark.RIGHT_HIP.value)
    left_knee = pt(pose_landmark.LEFT_KNEE.value)
    right_knee = pt(pose_landmark.RIGHT_KNEE.value)
    left_ankle = pt(pose_landmark.LEFT_ANKLE.value)
    right_ankle = pt(pose_landmark.RIGHT_ANKLE.value)
    left_shoulder = pt(pose_landmark.LEFT_SHOULDER.value)
    right_shoulder = pt(pose_landmark.RIGHT_SHOULDER.value)
    left_elbow = pt(pose_landmark.LEFT_ELBOW.value)
    right_elbow = pt(pose_landmark.RIGHT_ELBOW.value)
    left_wrist = pt(pose_landmark.LEFT_WRIST.value)
    right_wrist = pt(pose_landmark.RIGHT_WRIST.value)

    hip_mid = (left_hip + right_hip) / 2.0
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0

    feature_lists["knee_angle_left"].append(
        _angle_degrees(left_hip, left_knee, left_ankle)
    )
    feature_lists["knee_angle_right"].append(
        _angle_degrees(right_hip, right_knee, right_ankle)
    )
    feature_lists["hip_angle_left"].append(
        _angle_degrees(left_shoulder, left_hip, left_knee)
    )
    feature_lists["hip_angle_right"].append(
        _angle_degrees(right_shoulder, right_hip, right_knee)
    )
    feature_lists["elbow_angle_left"].append(
        _angle_degrees(left_shoulder, left_elbow, left_wrist)
    )
    feature_lists["elbow_angle_right"].append(
        _angle_degrees(right_shoulder, right_elbow, right_wrist)
    )
    feature_lists["shoulder_angle_left"].append(
        _angle_degrees(left_hip, left_shoulder, left_elbow)
    )
    feature_lists["shoulder_angle_right"].append(
        _angle_degrees(right_hip, right_shoulder, right_elbow)
    )

    pelvis_y = float((left_hip[1] + right_hip[1]) / 2.0)
    feature_lists["pelvis_y"].append(pelvis_y)

    feature_lists["wrist_left_x"].append(float(left_wrist[0]))
    feature_lists["wrist_left_y"].append(float(left_wrist[1]))
    feature_lists["wrist_right_x"].append(float(right_wrist[0]))
    feature_lists["wrist_right_y"].append(float(right_wrist[1]))

    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    torso_left = float(np.linalg.norm(left_shoulder[:2] - left_hip[:2]))
    torso_right = float(np.linalg.norm(right_shoulder[:2] - right_hip[:2]))
    torso_length = (torso_left + torso_right) / 2.0
    feature_lists["torso_length"].append(float(torso_length))
    norm_width = shoulder_width / (torso_length + 1e-6)
    feature_lists["shoulder_width_norm"].append(norm_width)

    ankle_width = float(np.linalg.norm(left_ankle[:2] - right_ankle[:2]))
    ankle_width_norm = ankle_width / (torso_length + 1e-6)
    feature_lists["ankle_width_norm"].append(ankle_width_norm)

    dx = abs(float(left_shoulder[0] - right_shoulder[0]))
    dz = abs(float(left_shoulder[2] - right_shoulder[2]))
    yaw_deg = math.degrees(math.atan2(dz, dx + 1e-6))
    feature_lists["shoulder_yaw_deg"].append(float(yaw_deg))
    feature_lists["shoulder_z_delta_abs"].append(float(dz))

    torso_vec = shoulder_mid - hip_mid
    tilt = math.degrees(math.atan2(abs(torso_vec[0]), abs(torso_vec[1]) + 1e-6))
    feature_lists["torso_tilt_deg"].append(float(tilt))

    return True


def _append_nan(feature_lists: Dict[str, list[float]]) -> None:
    for values in feature_lists.values():
        values.append(float("nan"))


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the planar angle ABC in degrees using the XY plane."""

    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return float("nan")
    cos_angle = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(math.degrees(math.acos(cos_angle)))


def _estimate_sampling_rate(fps: float, frame_count: int, samples: int) -> float:
    if samples <= 0:
        return DEFAULT_SAMPLING_RATE
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
        if duration > 0:
            return float(samples / duration)
    if fps > 0:
        return float(fps)
    return DEFAULT_SAMPLING_RATE


# --- Classification ------------------------------------------------------------


def classify_features(features: FeatureSeries) -> Tuple[str, str, float]:
    """Classify the exercise based on the extracted feature series."""

    if features.valid_frames < MIN_VALID_FRAMES:
        return "unknown", "unknown", 0.0

    data = features.data

    sr = max(1.0, float(features.sampling_rate or DEFAULT_SAMPLING_RATE))

    smooth_keys = {
        "knee_angle_left",
        "knee_angle_right",
        "hip_angle_left",
        "hip_angle_right",
        "elbow_angle_left",
        "elbow_angle_right",
        "shoulder_width_norm",
        "shoulder_yaw_deg",
        "shoulder_z_delta_abs",
        "pelvis_y",
        "torso_tilt_deg",
        "ankle_width_norm",
        "torso_length",
    }

    smoothed: Dict[str, np.ndarray] = {}
    for key in smooth_keys:
        series = data.get(key)
        if series is None:
            smoothed[key] = np.array([])
        else:
            smoothed[key] = _smooth(np.asarray(series), sr=sr)

    def _series(name: str) -> np.ndarray:
        series = smoothed.get(name)
        if series is not None and series.size:
            return series
        raw = data.get(name)
        if raw is None:
            return np.array([])
        return np.asarray(raw)

    knee_left = _series("knee_angle_left")
    knee_right = _series("knee_angle_right")
    hip_left = _series("hip_angle_left")
    hip_right = _series("hip_angle_right")

    elbow_left = _series("elbow_angle_left")
    elbow_right = _series("elbow_angle_right")
    pelvis_series = _series("pelvis_y")
    torso_tilt_series = _series("torso_tilt_deg")
    shoulder_width_series = _series("shoulder_width_norm")
    shoulder_yaw_series = _series("shoulder_yaw_deg")
    shoulder_z_delta_series = _series("shoulder_z_delta_abs")
    ankle_width_series = _series("ankle_width_norm")
    torso_length_series = _series("torso_length")
    rep_slices = _segment_reps(features)

    if not rep_slices:
        total_len = int(
            max((np.asarray(arr).size for arr in data.values() if isinstance(arr, np.ndarray)), default=0)
        )
        if total_len > 0:
            rep_slices = [slice(0, total_len)]

    rep_data = {
        "knee_angle_left": knee_left,
        "knee_angle_right": knee_right,
        "hip_angle_left": hip_left,
        "hip_angle_right": hip_right,
        "elbow_angle_left": elbow_left,
        "elbow_angle_right": elbow_right,
        "pelvis_y": pelvis_series,
        "torso_tilt_deg": torso_tilt_series,
    }

    rep_stats = [
        _rep_features(rep_data, rep, sr)
        for rep in rep_slices
    ]

    def _metric(values, reducer):
        finite_values = _finite(np.asarray(values, dtype=float))
        if finite_values.size == 0:
            return 0.0
        return float(reducer(finite_values))

    duration_med = _metric([rep.duration_s for rep in rep_stats], np.median)
    cadence_hz = (1.0 / duration_med) if duration_med > 0 else 0.0

    if cadence_hz > 0:
        updated = False
        for key in smooth_keys:
            raw = data.get(key)
            if raw is None:
                continue
            refined = _smooth(np.asarray(raw), sr=sr, cadence_hz=cadence_hz)
            if refined.size:
                smoothed[key] = refined
                updated = True
        if updated:
            knee_left = _series("knee_angle_left")
            knee_right = _series("knee_angle_right")
            hip_left = _series("hip_angle_left")
            hip_right = _series("hip_angle_right")
            elbow_left = _series("elbow_angle_left")
            elbow_right = _series("elbow_angle_right")
            pelvis_series = _series("pelvis_y")
            torso_tilt_series = _series("torso_tilt_deg")
            shoulder_width_series = _series("shoulder_width_norm")
            shoulder_yaw_series = _series("shoulder_yaw_deg")
            shoulder_z_delta_series = _series("shoulder_z_delta_abs")
            ankle_width_series = _series("ankle_width_norm")
            torso_length_series = _series("torso_length")
            rep_data = {
                "knee_angle_left": knee_left,
                "knee_angle_right": knee_right,
                "hip_angle_left": hip_left,
                "hip_angle_right": hip_right,
                "elbow_angle_left": elbow_left,
                "elbow_angle_right": elbow_right,
                "pelvis_y": pelvis_series,
                "torso_tilt_deg": torso_tilt_series,
            }
            rep_stats = [
                _rep_features(rep_data, rep, sr)
                for rep in rep_slices
            ]

    knee_rom_med = _metric([rep.knee_rom for rep in rep_stats], np.median)
    hip_rom_med = _metric([rep.hip_rom for rep in rep_stats], np.median)
    elbow_rom_med = _metric([rep.elbow_rom for rep in rep_stats], np.median)
    pelvis_drop_med = _metric([rep.pelvis_drop for rep in rep_stats], np.median)
    tilt_med = _metric([rep.torso_tilt_mean for rep in rep_stats], np.median)

    knee_rom = knee_rom_med
    hip_rom = hip_rom_med
    elbow_rom = elbow_rom_med
    torso_tilt_mean = tilt_med

    torso_med = _finite_stat(torso_length_series, np.median)
    if not np.isfinite(torso_med) or torso_med < 1e-3:
        pelvis_drop_norm = 0.0
    else:
        pelvis_drop_norm = pelvis_drop_med / (torso_med + 1e-6)
    pelvis_drop_norm = float(np.clip(pelvis_drop_norm, 0.0, 1.5))

    shoulder_rom = _max_range(data.get("shoulder_angle_left"), data.get("shoulder_angle_right"))
    wrist_vertical = _max_range(
        data.get("wrist_left_y"), data.get("wrist_right_y")
    )
    wrist_horizontal = _max_range(
        data.get("wrist_left_x"), data.get("wrist_right_x")
    )

    view = _classify_view(
        shoulder_width_series=shoulder_width_series,
        shoulder_yaw_series=shoulder_yaw_series,
        shoulder_z_delta_series=shoulder_z_delta_series,
        ankle_width_series=ankle_width_series,
    )

    width_mean = _finite_stat(shoulder_width_series, np.mean)
    width_std = _finite_stat(shoulder_width_series, np.std)
    yaw_med = _finite_stat(shoulder_yaw_series, np.median)
    z_med = _finite_stat(shoulder_z_delta_series, np.median)
    ankle_width_mean = _finite_stat(ankle_width_series, np.mean)
    ankle_width_std = _finite_stat(ankle_width_series, np.std)
    torso_med_value = float(torso_med) if np.isfinite(torso_med) else 0.0
    rep_count = len(rep_stats)

    details = {
        "rep_count": rep_count,
        "cadence_hz": float(cadence_hz),
        "knee_rom_med": float(knee_rom_med),
        "hip_rom_med": float(hip_rom_med),
        "elbow_rom_med": float(elbow_rom_med),
        "pelvis_drop_med": float(pelvis_drop_med),
        "pelvis_drop_norm": float(pelvis_drop_norm),
        "tilt_med": float(tilt_med),
        "torso_med": torso_med_value,
    }
    logger.info(
        "DET DEBUG — knee=%.1f hip=%.1f elbow=%.1f pelvisDrop=%.3f "
        "wV=%.3f wH=%.3f tilt=%.1f widthMean=%.2f widthStd=%.2f yawMed=%.1f zMed=%.3f "
        "ankleMean=%.2f ankleStd=%.2f repCount=%d cadence=%.2f details=%s",
        knee_rom,
        hip_rom,
        elbow_rom,
        pelvis_drop_med,
        wrist_vertical,
        wrist_horizontal,
        torso_tilt_mean,
        width_mean,
        width_std,
        yaw_med,
        z_med,
        ankle_width_mean,
        ankle_width_std,
        rep_count,
        cadence_hz,
        details,
    )

    knee_min = _min_finite_percentile(knee_left, knee_right, 15)
    hip_min = _min_finite_percentile(hip_left, hip_right, 15)
    symmetry = _symmetry_score(knee_left, knee_right, SQUAT_SIDE_SYMMETRY_TOLERANCE_DEG)

    def cadence_adjust(cadence: float, bonus_range: tuple[float, float], penalty_threshold: float) -> float:
        if cadence <= 0:
            return 0.0
        bonus = 0.0
        low, high = bonus_range
        if low <= cadence <= high:
            bonus += 0.35
        if cadence > penalty_threshold:
            bonus -= 0.35 * min(2.0, (cadence - penalty_threshold) / penalty_threshold + 1.0)
        return bonus

    squat_score = (
        _score(knee_rom, SQUAT_KNEE_ROM_THRESHOLD_DEG)
        + 0.9 * _score(pelvis_drop_norm, SQUAT_PELVIS_DISPLACEMENT_THRESHOLD)
        + 0.4 * _score(hip_rom, SQUAT_HIP_ROM_THRESHOLD_DEG)
        + 0.6 * _score_inverse(knee_min, SQUAT_KNEE_MIN_ANGLE_DEG, scale=1.8)
        + 0.4 * _score_inverse(hip_min, SQUAT_HIP_MIN_ANGLE_DEG, scale=1.6)
        + 0.5 * symmetry
        + cadence_adjust(cadence_hz, (0.2, 0.8), 1.2)
    )

    bench_score = (
        _score(elbow_rom, BENCH_ELBOW_ROM_THRESHOLD_DEG)
        + 0.7 * _score(wrist_horizontal, BENCH_WRIST_HORIZONTAL_THRESHOLD)
        + 0.6 * _score(torso_tilt_mean, BENCH_TORSO_TILT_THRESHOLD_DEG, scale=1.0)
        + 0.25 * cadence_adjust(cadence_hz, (0.4, 1.2), 1.5)
    )

    bench_gate = min(
        _score(torso_tilt_mean, BENCH_TORSO_MIN_DEG, scale=1.0),
        _score_inverse(max(pelvis_drop_norm, 1e-6), BENCH_PELVIS_DROP_NORM_MAX, scale=1.0),
        _score_inverse(max(knee_rom, 1e-6), BENCH_LOWER_ROM_MAX_DEG, scale=1.0),
        _score_inverse(max(hip_rom, 1e-6), BENCH_LOWER_ROM_MAX_DEG, scale=1.0),
    )
    bench_gate = max(0.0, min(1.0, bench_gate))

    bench_score = max(0.0, bench_score * (0.25 + 0.75 * bench_gate))
    bench_score -= 0.3 * _score(knee_rom, BENCH_LOWER_ROM_MAX_DEG * 1.5)
    bench_score -= 0.3 * _score(hip_rom, BENCH_LOWER_ROM_MAX_DEG * 1.5)
    bench_score = max(0.0, bench_score)

    deadlift_upright_bonus = max(0.0, DEADLIFT_TORSO_UPRIGHT_TARGET_DEG - torso_tilt_mean)
    deadlift_score = (
        _score(hip_rom, DEADLIFT_HIP_ROM_THRESHOLD_DEG)
        + 0.7 * _score(wrist_vertical, DEADLIFT_WRIST_VERTICAL_THRESHOLD)
        + 0.5 * _score(deadlift_upright_bonus, DEADLIFT_TORSO_UPRIGHT_TARGET_DEG, scale=1.0)
        + 0.2 * _score(shoulder_rom, SQUAT_HIP_ROM_THRESHOLD_DEG)
        + cadence_adjust(cadence_hz, (0.2, 0.8), 1.2)
    )

    scores = {
        "squat": float(squat_score),
        "bench_press": float(bench_score),
        "deadlift": float(deadlift_score),
    }

    label, confidence = _pick_label(scores)
    if label == "unknown":
        return "unknown", view, confidence

    return label, view, confidence


@dataclass
class RepStats:
    knee_rom: float
    hip_rom: float
    elbow_rom: float
    torso_tilt_mean: float
    pelvis_drop: float
    duration_s: float


def _segment_reps(features: FeatureSeries) -> list[slice]:
    sr = max(1.0, float(features.sampling_rate or DEFAULT_SAMPLING_RATE))
    activity_series = []
    derivative_length = None

    for name in (
        "knee_angle_left",
        "knee_angle_right",
        "hip_angle_left",
        "hip_angle_right",
        "elbow_angle_left",
        "elbow_angle_right",
    ):
        series = features.data.get(name)
        if series is None or np.asarray(series).size == 0:
            continue
        smoothed = _smooth(np.asarray(series), sr=sr)
        if smoothed.size < 3:
            continue
        z = _zscore(smoothed)
        dz = np.diff(z)
        if dz.size == 0:
            continue
        activity_series.append(np.abs(dz))
        derivative_length = dz.size if derivative_length is None else min(derivative_length, dz.size)

    if not activity_series or derivative_length is None or derivative_length <= 0:
        return []

    trimmed = [series[:derivative_length] for series in activity_series]
    activity = np.nan_to_num(np.vstack(trimmed), nan=0.0).mean(axis=0)
    if activity.size == 0:
        return []

    activity = np.where(np.isfinite(activity), activity, 0.0)
    percentile_65 = float(np.percentile(activity, 65)) if activity.size else 0.0
    mean_val = float(np.mean(activity)) if activity.size else 0.0
    std_val = float(np.std(activity)) if activity.size else 0.0
    adaptive_height = mean_val + std_val * 0.25
    height = max(percentile_65, adaptive_height)
    if not np.isfinite(height) or height <= 0.0:
        return []  # flat/noisy signal → skip rep segmentation

    distance_frames = max(1, int(round(max(0.35 * sr, 0.25 * sr))))

    peaks, _ = find_peaks(activity, height=height, distance=distance_frames)
    if peaks.size == 0:
        return []

    activity_len = activity.size
    series_len = derivative_length + 1

    midpoints = (((peaks[:-1] + peaks[1:]) // 2) + 1).astype(int) if peaks.size > 1 else np.array([], dtype=int)
    boundaries = np.concatenate(([0], midpoints, [activity_len]))

    rep_slices: list[slice] = []
    for idx, peak in enumerate(peaks):
        start_frame = int(boundaries[idx])
        end_frame = int(boundaries[idx + 1])
        start_frame = max(0, min(start_frame, activity_len - 1))
        end_frame = max(start_frame + 1, min(end_frame, activity_len))

        start_idx = max(0, min(start_frame, series_len - 2))
        end_idx = max(start_idx + 2, min(end_frame + 1, series_len))
        if end_idx - start_idx <= 1:
            continue
        duration_s = (end_idx - start_idx) / sr if sr > 0 else 0.0
        if duration_s < 0.25 or duration_s > 4.0:
            continue
        rep_slices.append(slice(start_idx, end_idx))

    return rep_slices


def _rep_features(data: Dict[str, np.ndarray], rep: slice, sr: float) -> RepStats:
    start = int(rep.start or 0)
    stop = int(rep.stop or start)
    if stop <= start:
        return RepStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _slice(name: str) -> np.ndarray:
        series = data.get(name)
        if series is None or np.asarray(series).size == 0:
            return np.array([])
        arr = np.asarray(series)
        start_idx = max(0, min(start, arr.size))
        stop_idx = max(start_idx, min(stop, arr.size))
        return arr[start_idx:stop_idx]

    knee_left = _slice("knee_angle_left")
    knee_right = _slice("knee_angle_right")
    hip_left = _slice("hip_angle_left")
    hip_right = _slice("hip_angle_right")
    elbow_left = _slice("elbow_angle_left")
    elbow_right = _slice("elbow_angle_right")
    pelvis = _slice("pelvis_y")
    tilt = _slice("torso_tilt_deg")

    knee_rom = _max_range(knee_left, knee_right)
    hip_rom = _max_range(hip_left, hip_right)
    elbow_rom = _max_range(elbow_left, elbow_right)
    pelvis_drop = _range_of(pelvis)
    torso_tilt_mean = float(np.nanmean(tilt)) if tilt.size else 0.0
    duration_s = (stop - start) / sr if sr > 0 else 0.0

    return RepStats(
        knee_rom=knee_rom,
        hip_rom=hip_rom,
        elbow_rom=elbow_rom,
        torso_tilt_mean=torso_tilt_mean,
        pelvis_drop=pelvis_drop,
        duration_s=duration_s,
    )


def _max_range(*series: np.ndarray | None) -> float:
    ranges = [_range_of(s) for s in series if s is not None]
    return max(ranges) if ranges else 0.0


def _range_of(series: np.ndarray | None) -> float:
    if series is None or series.size == 0:
        return 0.0
    valid = series[~np.isnan(series)]
    if valid.size == 0:
        return 0.0
    return float(valid.max() - valid.min())


def _min_finite_percentile(
    series_a: np.ndarray | None,
    series_b: np.ndarray | None,
    percentile: float,
) -> float:
    values = []
    if series_a is not None:
        values.append(_finite_percentile(series_a, percentile))
    if series_b is not None:
        values.append(_finite_percentile(series_b, percentile))
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(min(finite))


def _finite_stat(series: np.ndarray | None, reducer) -> float:
    if series is None or series.size == 0:
        return float("nan")
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return float("nan")
    return float(reducer(finite))


def _finite_percentile(series: np.ndarray | None, percentile: float) -> float:
    if series is None or series.size == 0:
        return float("nan")
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return float("nan")
    percentile = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(finite, percentile))


def _score(value: float, threshold: float, scale: float = 1.5) -> float:
    if threshold <= 0:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    if value <= 0:
        return 0.0
    ratio = value / threshold
    ratio = max(0.0, min(ratio, scale * 2.0))
    return ratio


def _score_inverse(value: float, threshold: float, scale: float = 1.5) -> float:
    if threshold <= 0:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    if value <= 0:
        return 0.0
    headroom = threshold - value
    if headroom <= 0:
        return 0.0
    ratio = headroom / threshold
    ratio = max(0.0, min(ratio, scale * 2.0))
    return ratio


def _symmetry_score(
    series_a: np.ndarray | None,
    series_b: np.ndarray | None,
    tolerance: float,
) -> float:
    if tolerance <= 0:
        return 0.0
    if series_a is None or series_b is None:
        return 0.0
    a_med = _finite_percentile(series_a, 15)
    b_med = _finite_percentile(series_b, 15)
    if not np.isfinite(a_med) or not np.isfinite(b_med):
        return 0.0
    diff = abs(a_med - b_med)
    return _score_inverse(diff, tolerance, scale=2.0)


def _pick_label(scores: Dict[str, float]) -> Tuple[str, float]:
    values = np.array(list(scores.values()), dtype=float)
    labels = list(scores.keys())
    max_index = int(np.argmax(values))
    max_value = float(values[max_index])
    sorted_indices = np.argsort(values)
    second_best = float(values[sorted_indices[-2]]) if values.size > 1 else 0.0

    if max_value < MIN_CONFIDENCE_SCORE or (max_value - second_best) < CLASSIFICATION_MARGIN:
        return "unknown", 0.0

    exp_scores = np.exp(values - np.max(values))
    probs = exp_scores / exp_scores.sum()
    confidence = float(probs[max_index])
    return labels[max_index], confidence


def _pick_view_label(scores: Dict[str, float], evidence: int) -> str:
    if not scores:
        return "unknown"

    labels = list(scores.keys())
    values = np.array(list(scores.values()), dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return "unknown"

    values = values[finite_mask]
    labels = [label for label, mask in zip(labels, finite_mask) if mask]

    max_index = int(np.argmax(values))
    max_value = float(values[max_index])
    if max_value <= 0:
        return "unknown"

    if values.size > 1:
        sorted_values = np.sort(values)
        second_best = float(sorted_values[-2])
    else:
        second_best = 0.0

    evidence = max(1, int(evidence))
    min_required = VIEW_SCORE_PER_EVIDENCE_THRESHOLD * evidence
    margin_required = VIEW_MARGIN_PER_EVIDENCE_THRESHOLD * evidence

    if max_value < min_required:
        return "unknown"

    if (max_value - second_best) < margin_required:
        return "unknown"

    return labels[max_index]


def _classify_view(
    shoulder_width_series: np.ndarray | None,
    shoulder_yaw_series: np.ndarray | None = None,
    shoulder_z_delta_series: np.ndarray | None = None,
    ankle_width_series: np.ndarray | None = None,
) -> str:
    yaw_med = _finite_stat(shoulder_yaw_series, np.median)
    yaw_p75 = _finite_percentile(shoulder_yaw_series, 75)
    z_med = _finite_stat(shoulder_z_delta_series, np.median)
    width_mean = _finite_stat(shoulder_width_series, np.mean)
    width_std = _finite_stat(shoulder_width_series, np.std)
    width_p10 = _finite_percentile(shoulder_width_series, 10)
    ankle_mean = _finite_stat(ankle_width_series, np.mean)
    ankle_std = _finite_stat(ankle_width_series, np.std)
    ankle_p10 = _finite_percentile(ankle_width_series, 10)

    front_score = 0.0
    side_score = 0.0
    evidence = 0

    front_votes = 0
    side_votes = 0

    if np.isfinite(yaw_med):
        front_score += _score_inverse(yaw_med, YAW_FRONT_MAX_DEG, scale=2.0)
        if yaw_med <= YAW_FRONT_MAX_DEG * 1.05:
            front_votes += 1
        if yaw_med >= YAW_SIDE_MIN_DEG * 0.9:
            side_votes += 1
        evidence += 1
    if np.isfinite(yaw_p75):
        side_score += _score(yaw_p75, YAW_SIDE_MIN_DEG, scale=2.0)
        if yaw_p75 >= YAW_SIDE_MIN_DEG:
            side_votes += 1

    if np.isfinite(z_med):
        front_score += _score_inverse(z_med, Z_DELTA_FRONT_MAX, scale=2.0)
        side_score += _score(z_med, Z_DELTA_FRONT_MAX * 1.6, scale=2.0)
        if z_med <= Z_DELTA_FRONT_MAX * 1.05:
            front_votes += 1
        if z_med >= Z_DELTA_FRONT_MAX * 1.6:
            side_votes += 1
        evidence += 1

    if np.isfinite(width_mean):
        front_score += _score(width_mean, VIEW_FRONT_WIDTH_THRESHOLD, scale=2.0)
        side_score += _score_inverse(width_mean, SIDE_WIDTH_MAX, scale=2.0)
        if width_mean >= VIEW_FRONT_WIDTH_THRESHOLD * 0.96:
            front_votes += 1
        if width_mean <= SIDE_WIDTH_MAX * 1.04:
            side_votes += 1
        evidence += 1

    if np.isfinite(width_std):
        front_score += _score_inverse(width_std, VIEW_WIDTH_STD_THRESHOLD, scale=2.0)
        side_score += _score(width_std, VIEW_WIDTH_STD_THRESHOLD * 0.9, scale=2.0)
        if width_std <= VIEW_WIDTH_STD_THRESHOLD * 0.9:
            front_votes += 1
        if width_std >= VIEW_WIDTH_STD_THRESHOLD * 1.05:
            side_votes += 1
        evidence += 1

    if np.isfinite(width_p10):
        front_score += _score(width_p10, VIEW_FRONT_WIDTH_THRESHOLD * 0.9, scale=1.8)
        side_score += _score_inverse(width_p10, SIDE_WIDTH_MAX * 1.1, scale=1.8)
        if width_p10 >= VIEW_FRONT_WIDTH_THRESHOLD * 0.85:
            front_votes += 1
        if width_p10 <= SIDE_WIDTH_MAX * 1.1:
            side_votes += 1

    if np.isfinite(ankle_mean):
        front_score += 0.8 * _score(ankle_mean, ANKLE_FRONT_WIDTH_THRESHOLD * 0.95, scale=2.0)
        side_score += 0.8 * _score_inverse(ankle_mean, ANKLE_SIDE_WIDTH_MAX * 1.05, scale=2.0)
        if ankle_mean >= ANKLE_FRONT_WIDTH_THRESHOLD * 0.9:
            front_votes += 1
        if ankle_mean <= ANKLE_SIDE_WIDTH_MAX * 1.05:
            side_votes += 1
        evidence += 1

    if np.isfinite(ankle_std):
        front_score += 0.6 * _score_inverse(ankle_std, ANKLE_WIDTH_STD_THRESHOLD * 1.1, scale=2.0)
        side_score += 0.6 * _score(ankle_std, ANKLE_WIDTH_STD_THRESHOLD, scale=2.0)
        if ankle_std <= ANKLE_WIDTH_STD_THRESHOLD:
            front_votes += 1
        if ankle_std >= ANKLE_WIDTH_STD_THRESHOLD * 1.05:
            side_votes += 1
        evidence += 1

    if np.isfinite(ankle_p10):
        front_score += 0.6 * _score(ankle_p10, ANKLE_FRONT_WIDTH_THRESHOLD * 0.85, scale=1.8)
        side_score += 0.6 * _score_inverse(ankle_p10, ANKLE_SIDE_WIDTH_MAX * 1.15, scale=1.8)
        if ankle_p10 >= ANKLE_FRONT_WIDTH_THRESHOLD * 0.82:
            front_votes += 1
        if ankle_p10 <= ANKLE_SIDE_WIDTH_MAX * 1.1:
            side_votes += 1

    scores = {"front": float(front_score), "side": float(side_score)}
    view = _pick_view_label(scores, evidence)

    logger.info(
        "VIEW DEBUG — scores=%s evidence=%d yawMed=%.1f yawP75=%.1f zMed=%.3f widthMean=%.3f "
        "widthStd=%.3f widthP10=%.3f ankleMean=%.3f ankleStd=%.3f ankleP10=%.3f "
        "frontVotes=%d sideVotes=%d",
        scores,
        evidence,
        float(yaw_med) if np.isfinite(yaw_med) else float("nan"),
        float(yaw_p75) if np.isfinite(yaw_p75) else float("nan"),
        float(z_med) if np.isfinite(z_med) else float("nan"),
        float(width_mean) if np.isfinite(width_mean) else float("nan"),
        float(width_std) if np.isfinite(width_std) else float("nan"),
        float(width_p10) if np.isfinite(width_p10) else float("nan"),
        float(ankle_mean) if np.isfinite(ankle_mean) else float("nan"),
        float(ankle_std) if np.isfinite(ankle_std) else float("nan"),
        float(ankle_p10) if np.isfinite(ankle_p10) else float("nan"),
        front_votes,
        side_votes,
    )

    if view == "unknown":
        if np.isfinite(yaw_med):
            if yaw_med <= VIEW_FRONT_FALLBACK_YAW_DEG:
                return "front"
            if yaw_med >= VIEW_SIDE_FALLBACK_YAW_DEG:
                return "side"
        if np.isfinite(z_med):
            if z_med <= Z_DELTA_FRONT_MAX:
                return "front"
            if z_med >= Z_DELTA_FRONT_MAX * 1.8:
                return "side"
        if np.isfinite(ankle_mean):
            if ankle_mean >= ANKLE_FRONT_WIDTH_THRESHOLD * 0.90:
                return "front"
            if ankle_mean <= ANKLE_SIDE_WIDTH_MAX * 1.10:
                return "side"
        if np.isfinite(width_mean):
            return "front" if width_mean >= VIEW_FRONT_WIDTH_THRESHOLD else "side"

    return view


def _smooth(
    series: np.ndarray,
    sr: float | None = None,
    cadence_hz: float | None = None,
    min_ms: float = 180.0,
    max_ms: float = 300.0,
    poly: int = 2,
) -> np.ndarray:
    """Return a smoothed copy of ``series`` using an adaptive Savitzky–Golay window."""

    if series is None:
        return np.array([])

    x = np.asarray(series, dtype=float)
    if x.size == 0:
        return np.array([])

    mask = np.isfinite(x)
    finite_count = int(mask.sum())
    if finite_count < 5:
        return x

    if sr is None or sr <= 0:
        window = 11
    else:
        if cadence_hz is not None and cadence_hz > 0:
            period_s = 1.0 / cadence_hz
            target_ms = float(np.clip(0.20 * period_s * 1000.0, min_ms, max_ms))
        else:
            target_ms = float(np.clip(250.0, min_ms, max_ms))
        window = int(round(sr * target_ms / 1000.0))
        window = max(5, min(window, 51))

    if window % 2 == 0:
        window += 1

    if window > x.size:
        window = x.size if x.size % 2 == 1 else x.size - 1

    if window < 5:
        return x

    if not mask.all():
        finite_idx = np.flatnonzero(mask)
        if finite_idx.size == 0:
            return x
        x[~mask] = np.interp(np.flatnonzero(~mask), finite_idx, x[mask])

    if finite_count < window:
        window = finite_count if finite_count % 2 == 1 else finite_count - 1
        if window < 5:
            return x

    polyorder = int(min(poly, max(1, window - 1)))

    try:
        return savgol_filter(x, window_length=int(window), polyorder=polyorder)
    except Exception:
        return x


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / (s + 1e-6)


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]
