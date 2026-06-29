"""MediaPipe 33-landmark pose backend and REBA-visible angle adapter."""

from __future__ import annotations

import cv2
import sys

# MediaPipe checks TensorFlow only for optional documentation helpers. A
# partially installed TensorFlow on Windows can crash that optional import;
# this production backend does not use TensorFlow at all.
sys.modules["tensorflow"] = None
import mediapipe as mp
import numpy as np


MP_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)
_POSE = None


def _detector():
    global _POSE
    if _POSE is None:
        _POSE = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.45, min_tracking_confidence=0.45,
        )
    return _POSE


def detect_pose33(frame_bgr):
    """Return 33 pixel landmarks [x,y,visibility] and relative 3D landmarks."""
    h, w = frame_bgr.shape[:2]
    result = _detector().process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return None, None
    image = [[lm.x * w, lm.y * h, lm.visibility] for lm in result.pose_landmarks.landmark]
    world = None
    if result.pose_world_landmarks:
        world = [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_world_landmarks.landmark]
    return image, world


def _p(kp, i, conf):
    return np.asarray(kp[i][:2], float) if kp and kp[i][2] >= conf else None


def _mid(a, b):
    return None if a is None or b is None else (a + b) / 2


def _angle(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-8: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1))))


def _joint(a, b, c):
    return None if a is None or b is None or c is None else _angle(a-b, c-b)


def _worst(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def compute_reba_pose33(kp, confidence=0.35):
    """Compute visible REBA components including wrist and foot landmarks."""
    p = lambda i: _p(kp, i, confidence)
    # MediaPipe: shoulders 11/12, elbows 13/14, wrists 15/16,
    # hand direction 19/20, hips 23/24, knees 25/26, ankles 27/28,
    # heels 29/30 and foot indices 31/32.
    shoulders, hips = _mid(p(11), p(12)), _mid(p(23), p(24))
    ears = _mid(p(7), p(8))
    if shoulders is None or hips is None: return None
    up, down = np.array([0., -1.]), np.array([0., 1.])
    trunk = _angle(shoulders-hips, up)
    neck = _angle(ears-shoulders, shoulders-hips) if ears is not None else None
    upper = _worst([_angle(p(13)-p(11), down) if p(13) is not None and p(11) is not None else None,
                    _angle(p(14)-p(12), down) if p(14) is not None and p(12) is not None else None])
    elbows = [_joint(p(11),p(13),p(15)), _joint(p(12),p(14),p(16))]
    elbow_dev = _worst([abs(v-80) for v in elbows if v is not None])
    elbow_angle = None if not elbows or all(v is None for v in elbows) else elbows[np.argmax([abs((v or 80)-80) for v in elbows])]
    knees = [_joint(p(23),p(25),p(27)), _joint(p(24),p(26),p(28))]
    knee_flex = _worst([180-v for v in knees if v is not None])
    wrist_angles = [_joint(p(13),p(15),p(19)), _joint(p(14),p(16),p(20))]
    wrist_dev = _worst([abs(180-v) for v in wrist_angles if v is not None])
    foot_angles = [_joint(p(25),p(27),p(31)), _joint(p(26),p(28),p(32))]

    neck_s = 1 if neck is None or neck <= 20 else 2
    trunk_s = 1 if trunk <= 5 else 2 if trunk <= 20 else 3 if trunk <= 60 else 4
    legs_s = 1 if knee_flex is None or knee_flex < 30 else 2 if knee_flex <= 60 else 3
    upper_s = 1 if upper is None or upper <= 20 else 2 if upper <= 45 else 3 if upper <= 90 else 4
    lower_s = 1 if elbow_angle is not None and 60 <= elbow_angle <= 100 else 2
    wrist_s = 1 if wrist_dev is None or wrist_dev <= 15 else 2
    return {
        "steps": [neck_s,trunk_s,legs_s,upper_s,lower_s,wrist_s],
        "angles": {"neck_deg":neck,"trunk_deg":trunk,"upper_arm_deg":upper,
                   "elbow_deg":elbow_angle,"knee_flex_deg":knee_flex,
                   "wrist_deviation_deg":wrist_dev,
                   "left_ankle_deg":foot_angles[0],"right_ankle_deg":foot_angles[1]},
    }
