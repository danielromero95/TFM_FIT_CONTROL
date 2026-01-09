"""Shared catalog of metric names and descriptions for the UI."""

from __future__ import annotations

HUMAN_METRIC_NAMES: dict[str, str] = {
    "left_knee": "left knee angle",
    "right_knee": "right knee angle",
    "left_elbow": "left elbow angle",
    "right_elbow": "right elbow angle",
    "left_hip": "left hip angle",
    "right_hip": "right hip angle",
    "trunk_inclination_deg": "trunk inclination",
    "shoulder_width": "shoulder width",
    "foot_separation": "foot separation",
    "knee_symmetry": "knee symmetry",
    "elbow_symmetry": "elbow symmetry",
}


def _with_side_descriptions(side: str, templates: dict[str, str]) -> dict[str, str]:
    return {
        key: value.format(
            side=side,
            side_cap=side.capitalize(),
            side_leg=f"{side} leg",
            side_arm=f"{side} arm",
            side_hip=f"{side} hip",
        )
        for key, value in templates.items()
    }


KNEE_BASE_DESCRIPTIONS = {
    "squat": "{side_cap} knee flexion angle tracking squat depth.",
    "bench_press": "{side_cap} knee flexion angle recorded to show leg drive stability during bench press.",
    "deadlift": "{side_cap} knee flexion angle illustrating the pull setup and lockout.",
    "default": "Angle formed by the femur and tibia of the {side_leg} at the hip–knee–ankle joints.",
}

ELBOW_BASE_DESCRIPTIONS = {
    "squat": "{side_cap} elbow flexion angle, useful for spotting arm movement during squats.",
    "bench_press": "{side_cap} elbow flexion angle tracing press depth and lockout.",
    "deadlift": "{side_cap} elbow flexion angle, helpful for verifying straight arms during the pull.",
    "default": "Angle formed by the humerus and forearm of the {side_arm} at the shoulder–elbow–wrist joints.",
}

HIP_BASE_DESCRIPTIONS = {
    "squat": "{side_cap} hip hinge angle complementing the view of squat depth.",
    "bench_press": "{side_cap} hip hinge angle that shows lower-body tension on the bench.",
    "deadlift": "{side_cap} hip hinge angle measuring the deadlift setup and lockout.",
    "default": "Angle at the shoulder–hip–knee joints describing the hinge of the {side_hip}.",
}

METRIC_BASE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "left_knee": _with_side_descriptions("left", KNEE_BASE_DESCRIPTIONS),
    "right_knee": _with_side_descriptions("right", KNEE_BASE_DESCRIPTIONS),
    "left_elbow": _with_side_descriptions("left", ELBOW_BASE_DESCRIPTIONS),
    "right_elbow": _with_side_descriptions("right", ELBOW_BASE_DESCRIPTIONS),
    "left_hip": _with_side_descriptions("left", HIP_BASE_DESCRIPTIONS),
    "right_hip": _with_side_descriptions("right", HIP_BASE_DESCRIPTIONS),
    "trunk_inclination_deg": {
        "squat": "Torso inclination relative to the hips, showing forward lean in the squat.",
        "bench_press": "Torso inclination relative to the hips, highlighting arch control on the bench.",
        "deadlift": "Torso inclination relative to the hips, indicating back angle in the pull.",
        "default": "Torso inclination relative to the hips expressed in degrees.",
    },
    "shoulder_width": {
        "squat": "Horizontal distance between shoulders (normalized), reflecting upper-body stance.",
        "bench_press": "Horizontal distance between shoulders (normalized), tracking shoulder width on the bench.",
        "deadlift": "Horizontal distance between shoulders (normalized), confirming back tightness in the pull setup.",
        "default": "Horizontal distance between shoulders in normalized screen units.",
    },
    "foot_separation": {
        "squat": "Horizontal distance between ankles (normalized), showing squat stance width.",
        "bench_press": "Horizontal distance between ankles (normalized), showing bench foot placement.",
        "deadlift": "Horizontal distance between ankles (normalized), showing deadlift stance width.",
        "default": "Horizontal distance between ankles in normalized screen units.",
    },
    "knee_symmetry": {
        "squat": "Symmetry score between left and right knee angles (1 means perfectly matched).",
        "bench_press": "Symmetry score between left and right knee angles to monitor lower-body balance on the bench.",
        "deadlift": "Symmetry score between left and right knee angles to verify even pull mechanics.",
        "default": "Symmetry score between knee angles where 1 indicates identical motion.",
    },
    "elbow_symmetry": {
        "squat": "Symmetry score between left and right elbow angles (1 means perfectly matched).",
        "bench_press": "Symmetry score between left and right elbow angles to track pressing balance.",
        "deadlift": "Symmetry score between left and right elbow angles to confirm straight-arm symmetry.",
        "default": "Symmetry score between elbow angles where 1 indicates identical motion.",
    },
}


def human_metric_name(metric: str) -> str:
    """Return a human-readable name for a metric key."""

    if metric in HUMAN_METRIC_NAMES:
        return HUMAN_METRIC_NAMES[metric]
    if metric.startswith("ang_vel_"):
        return f"angular velocity of {metric[8:].replace('_', ' ')}"
    return metric.replace("_", " ")


def metric_base_description(metric: str, exercise: str) -> str | None:
    """Return the base description of a metric, falling back to the default entry."""

    exercise = exercise or ""
    base = METRIC_BASE_DESCRIPTIONS.get(metric, {})
    if exercise in base:
        return base[exercise]
    return base.get("default", None)


def is_user_facing_metric(name: str) -> bool:
    """Return True when a metric should be shown in the UI selector."""

    if not name:
        return False
    excluded = {
        "analysis_frame_idx",
        "frame_idx",
        "source_frame_idx",
        "time_s",
        "source_time_s",
        "pose_ok",
    }
    if name in excluded:
        return False
    if name.endswith("_idx"):
        return False
    if "time" in name and name.endswith("_s"):
        return False
    return True
