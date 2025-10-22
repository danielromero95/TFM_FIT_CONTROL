# src/exercise_detection/exercise_detector.py
"""Heuristic exercise detection based on MediaPipe pose landmarks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from src.core.types import ExerciseType, ViewType, as_exercise, as_view

logger = logging.getLogger(__name__)

# --- Tunable thresholds -------------------------------------------------------

MIN_VALID_FRAMES = 20
SQUAT_KNEE_ROM_THRESHOLD_DEG = 40.0
SQUAT_HIP_ROM_THRESHOLD_DEG = 30.0
SQUAT_PELVIS_DISPLACEMENT_THRESHOLD = 0.06
SQUAT_KNEE_MIN_ANGLE_DEG = 135.0
SQUAT_HIP_MIN_ANGLE_DEG = 150.0
SQUAT_SIDE_SYMMETRY_TOLERANCE_DEG = 18.0

BENCH_ELBOW_ROM_THRESHOLD_DEG = 55.0
BENCH_WRIST_HORIZONTAL_THRESHOLD = 0.08
BENCH_TORSO_TILT_THRESHOLD_DEG = 40.0  # degrees away from vertical

DEADLIFT_HIP_ROM_THRESHOLD_DEG = 35.0
DEADLIFT_WRIST_VERTICAL_THRESHOLD = 0.12
DEADLIFT_TORSO_UPRIGHT_TARGET_DEG = 35.0

VIEW_FRONT_WIDTH_THRESHOLD = 0.55      # normalized shoulder width / torso length
VIEW_WIDTH_STD_THRESHOLD = 0.12
YAW_FRONT_MAX_DEG = 20.0
YAW_SIDE_MIN_DEG = 25.0
Z_DELTA_FRONT_MAX = 0.08               # normalized MediaPipe units (smaller is more front)
SIDE_WIDTH_MAX = 0.50                  # if width_norm mean ≤ this, likely side

VIEW_SCORE_PER_EVIDENCE_THRESHOLD = 0.22
VIEW_MARGIN_PER_EVIDENCE_THRESHOLD = 0.10
VIEW_FRONT_FALLBACK_YAW_DEG = 24.0
VIEW_SIDE_FALLBACK_YAW_DEG = 27.0

CLASSIFICATION_MARGIN = 0.15
MIN_CONFIDENCE_SCORE = 0.50

DEFAULT_SAMPLING_RATE = 30.0


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

    feature_lists: Dict[str, list[float]] = {
        "knee_angle_left": [],
        "knee_angle_right": [],
        "hip_angle_left": [],
        "hip_angle_right": [],
        "elbow_angle_left": [],
        "elbow_angle_right": [],
        "shoulder_angle_left": [],
        "shoulder_angle_right": [],
        "pelvis_y": [],
        "wrist_left_x": [],
        "wrist_left_y": [],
        "wrist_right_x": [],
        "wrist_right_y": [],
        "shoulder_width_norm": [],
        "shoulder_yaw_deg": [],
        "shoulder_z_delta_abs": [],
        "torso_tilt_deg": [],
    }

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
    norm_width = shoulder_width / (torso_length + 1e-6)
    feature_lists["shoulder_width_norm"].append(norm_width)

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

    knee_left = data.get("knee_angle_left")
    knee_right = data.get("knee_angle_right")
    hip_left = data.get("hip_angle_left")
    hip_right = data.get("hip_angle_right")

    knee_rom = _max_range(knee_left, knee_right)
    hip_rom = _max_range(hip_left, hip_right)
    elbow_rom = _max_range(data.get("elbow_angle_left"), data.get("elbow_angle_right"))
    shoulder_rom = _max_range(
        data.get("shoulder_angle_left"), data.get("shoulder_angle_right")
    )
    pelvis_disp = _range_of(data.get("pelvis_y"))
    wrist_vertical = _max_range(
        data.get("wrist_left_y"), data.get("wrist_right_y")
    )
    wrist_horizontal = _max_range(
        data.get("wrist_left_x"), data.get("wrist_right_x")
    )

    torso_tilt_series = data.get("torso_tilt_deg", np.empty(0))
    torso_tilt_mean = float(np.nanmean(torso_tilt_series)) if torso_tilt_series.size else 0.0

    shoulder_width_series = data.get("shoulder_width_norm", np.empty(0))
    shoulder_yaw_series = data.get("shoulder_yaw_deg", np.empty(0))
    shoulder_z_delta_series = data.get("shoulder_z_delta_abs", np.empty(0))

    view = _classify_view(
        shoulder_width_series=shoulder_width_series,
        shoulder_yaw_series=shoulder_yaw_series,
        shoulder_z_delta_series=shoulder_z_delta_series,
    )

    width_mean = _finite_stat(shoulder_width_series, np.mean)
    width_std = _finite_stat(shoulder_width_series, np.std)
    yaw_med = _finite_stat(shoulder_yaw_series, np.median)
    z_med = _finite_stat(shoulder_z_delta_series, np.median)
    logger.info(
        "DET DEBUG — knee=%.1f hip=%.1f elbow=%.1f pelvis=%.3f "
        "wV=%.3f wH=%.3f tilt=%.1f widthMean=%.2f widthStd=%.2f yawMed=%.1f zMed=%.3f",
        knee_rom,
        hip_rom,
        elbow_rom,
        pelvis_disp,
        wrist_vertical,
        wrist_horizontal,
        torso_tilt_mean,
        width_mean,
        width_std,
        yaw_med,
        z_med,
    )

    knee_min = _min_finite_percentile(knee_left, knee_right, 15)
    hip_min = _min_finite_percentile(hip_left, hip_right, 15)
    symmetry = _symmetry_score(knee_left, knee_right, SQUAT_SIDE_SYMMETRY_TOLERANCE_DEG)

    squat_score = (
        _score(knee_rom, SQUAT_KNEE_ROM_THRESHOLD_DEG)
        + 0.8 * _score(pelvis_disp, SQUAT_PELVIS_DISPLACEMENT_THRESHOLD)
        + 0.4 * _score(hip_rom, SQUAT_HIP_ROM_THRESHOLD_DEG)
        + 0.6 * _score_inverse(knee_min, SQUAT_KNEE_MIN_ANGLE_DEG, scale=1.8)
        + 0.4 * _score_inverse(hip_min, SQUAT_HIP_MIN_ANGLE_DEG, scale=1.6)
        + 0.5 * symmetry
    )

    bench_score = (
        _score(elbow_rom, BENCH_ELBOW_ROM_THRESHOLD_DEG)
        + 0.7 * _score(wrist_horizontal, BENCH_WRIST_HORIZONTAL_THRESHOLD)
        + 0.6 * _score(torso_tilt_mean, BENCH_TORSO_TILT_THRESHOLD_DEG, scale=1.0)
    )

    deadlift_upright_bonus = max(0.0, DEADLIFT_TORSO_UPRIGHT_TARGET_DEG - torso_tilt_mean)
    deadlift_score = (
        _score(hip_rom, DEADLIFT_HIP_ROM_THRESHOLD_DEG)
        + 0.7 * _score(wrist_vertical, DEADLIFT_WRIST_VERTICAL_THRESHOLD)
        + 0.5 * _score(deadlift_upright_bonus, DEADLIFT_TORSO_UPRIGHT_TARGET_DEG, scale=1.0)
        + 0.2 * _score(shoulder_rom, SQUAT_HIP_ROM_THRESHOLD_DEG)
    )

    scores = {
        "squat": float(squat_score),
        "bench": float(bench_score),
        "deadlift": float(deadlift_score),
    }

    label, confidence = _pick_label(scores)
    if label == "unknown":
        return "unknown", view, confidence

    return label, view, confidence


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
) -> str:
    yaw_med = _finite_stat(shoulder_yaw_series, np.median)
    yaw_p75 = _finite_percentile(shoulder_yaw_series, 75)
    z_med = _finite_stat(shoulder_z_delta_series, np.median)
    width_mean = _finite_stat(shoulder_width_series, np.mean)
    width_std = _finite_stat(shoulder_width_series, np.std)
    width_p10 = _finite_percentile(shoulder_width_series, 10)

    front_score = 0.0
    side_score = 0.0
    evidence = 0

    if np.isfinite(yaw_med):
        front_score += _score_inverse(yaw_med, YAW_FRONT_MAX_DEG, scale=2.0)
        evidence += 1
    if np.isfinite(yaw_p75):
        side_score += _score(yaw_p75, YAW_SIDE_MIN_DEG, scale=2.0)

    if np.isfinite(z_med):
        front_score += _score_inverse(z_med, Z_DELTA_FRONT_MAX, scale=2.0)
        side_score += _score(z_med, Z_DELTA_FRONT_MAX * 1.6, scale=2.0)
        evidence += 1

    if np.isfinite(width_mean):
        front_score += _score(width_mean, VIEW_FRONT_WIDTH_THRESHOLD, scale=2.0)
        side_score += _score_inverse(width_mean, SIDE_WIDTH_MAX, scale=2.0)
        evidence += 1

    if np.isfinite(width_std):
        front_score += _score_inverse(width_std, VIEW_WIDTH_STD_THRESHOLD, scale=2.0)
        side_score += _score(width_std, VIEW_WIDTH_STD_THRESHOLD * 0.9, scale=2.0)
        evidence += 1

    if np.isfinite(width_p10):
        front_score += _score(width_p10, VIEW_FRONT_WIDTH_THRESHOLD * 0.9, scale=1.8)
        side_score += _score_inverse(width_p10, SIDE_WIDTH_MAX * 1.1, scale=1.8)

    scores = {"front": float(front_score), "side": float(side_score)}
    view = _pick_view_label(scores, evidence)

    logger.info(
        "VIEW DEBUG — scores=%s evidence=%d yawMed=%.1f yawP75=%.1f zMed=%.3f widthMean=%.3f "
        "widthStd=%.3f widthP10=%.3f",
        scores,
        evidence,
        float(yaw_med) if np.isfinite(yaw_med) else float("nan"),
        float(yaw_p75) if np.isfinite(yaw_p75) else float("nan"),
        float(z_med) if np.isfinite(z_med) else float("nan"),
        float(width_mean) if np.isfinite(width_mean) else float("nan"),
        float(width_std) if np.isfinite(width_std) else float("nan"),
        float(width_p10) if np.isfinite(width_p10) else float("nan"),
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
        if np.isfinite(width_mean):
            return "front" if width_mean >= VIEW_FRONT_WIDTH_THRESHOLD else "side"

    return view
