"""Versioned runtime feature contract for the legacy 90-feature angle model."""

from __future__ import annotations

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
BASE_INPUT_COLS = [f"{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("x", "y", "z")]
DERIVED_FEATURES = [
    "left_elbow_angle_3d", "right_elbow_angle_3d",
    "left_shoulder_angle_3d", "right_shoulder_angle_3d",
    "left_hip_angle_3d", "right_hip_angle_3d",
    "left_knee_angle_3d", "right_knee_angle_3d",
    "trunk_vs_left_thigh_angle_3d", "trunk_vs_right_thigh_angle_3d",
    "neck_vs_trunk_angle_3d", "shoulder_line_vs_hip_line_angle_3d",
]
for plane in ("xy", "xz", "yz"):
    DERIVED_FEATURES.extend([
        f"left_elbow_angle_{plane}", f"right_elbow_angle_{plane}",
        f"left_shoulder_angle_{plane}", f"right_shoulder_angle_{plane}",
        f"left_hip_angle_{plane}", f"right_hip_angle_{plane}",
        f"left_knee_angle_{plane}", f"right_knee_angle_{plane}",
        f"shoulder_line_vs_hip_line_angle_{plane}",
    ])

FEATURE_NAMES = BASE_INPUT_COLS + DERIVED_FEATURES
FEATURE_SCHEMA_VERSION = "legacy-yolo17-v1"

FEATURE_SEMANTICS = {
    "x": "YOLO pixel x coordinate",
    "y": "YOLO pixel y coordinate",
    "z": "YOLO keypoint confidence, NOT physical depth",
    "derived": "angles computed by Utils.Functions.add_pose_angles",
}


def validate_feature_frame(frame):
    actual = list(frame.columns)
    if actual != FEATURE_NAMES:
        mismatch = next((i for i, pair in enumerate(zip(actual, FEATURE_NAMES)) if pair[0] != pair[1]), None)
        raise ValueError(f"Feature schema mismatch at index {mismatch}: expected {FEATURE_SCHEMA_VERSION}")
    if frame.shape[1] != 90:
        raise ValueError(f"Expected 90 features, received {frame.shape[1]}")
    return True
