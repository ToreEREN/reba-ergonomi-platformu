"""Explainable 2D pose measurements for responsive REBA screening.

These measurements use only coordinates that YOLO Pose actually observes.
They are suitable for live feedback, while hidden rotations still require a
validated 3D model or depth sensor. Camera is assumed upright and side-on or
oblique enough for joint flexion to be visible.
"""

from __future__ import annotations

import math
import numpy as np


NOSE, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16


def _point(kp, idx, confidence):
    if idx >= len(kp) or kp[idx][2] < confidence:
        return None
    return np.asarray(kp[idx][:2], dtype=float)


def _mid(a, b):
    return None if a is None or b is None else (a + b) / 2.0


def _angle(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-8:
        return None
    cosine = np.clip(np.dot(v1, v2) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _joint(a, b, c):
    return None if a is None or b is None or c is None else _angle(a - b, c - b)


def _max_valid(values):
    values = [x for x in values if x is not None]
    return max(values) if values else None


def compute_observed_reba(keypoints, confidence=0.30):
    p = lambda idx: _point(keypoints, idx, confidence)
    nose = p(NOSE)
    ls, rs, lh, rh = p(LS), p(RS), p(LH), p(RH)
    shoulders, hips = _mid(ls, rs), _mid(lh, rh)
    if shoulders is None or hips is None:
        return None

    up = np.array([0.0, -1.0])
    down = -up
    trunk_angle = _angle(shoulders - hips, up)
    neck_angle = _angle(nose - shoulders, shoulders - hips) if nose is not None else None
    elbow_l = _joint(ls, p(LE), p(LW))
    elbow_r = _joint(rs, p(RE), p(RW))
    knee_l = _joint(lh, p(LK), p(LA))
    knee_r = _joint(rh, p(RK), p(RA))
    upper_l = _angle(p(LE) - ls, down) if p(LE) is not None and ls is not None else None
    upper_r = _angle(p(RE) - rs, down) if p(RE) is not None and rs is not None else None

    if trunk_angle is None:
        return None
    neck_score = 1 if neck_angle is None or neck_angle <= 20 else 2
    if trunk_angle <= 5:
        trunk_score = 1
    elif trunk_angle <= 20:
        trunk_score = 2
    elif trunk_angle <= 60:
        trunk_score = 3
    else:
        trunk_score = 4

    knee_flex = _max_valid([180 - x for x in (knee_l, knee_r) if x is not None])
    legs_score = 1 if knee_flex is None or knee_flex < 30 else (2 if knee_flex <= 60 else 3)
    upper_elevation = _max_valid([upper_l, upper_r])
    if upper_elevation is None or upper_elevation <= 20:
        upper_score = 1
    elif upper_elevation <= 45:
        upper_score = 2
    elif upper_elevation <= 90:
        upper_score = 3
    else:
        upper_score = 4
    elbow = _max_valid([elbow_l, elbow_r])
    lower_score = 1 if elbow is not None and 60 <= elbow <= 100 else 2

    return {
        "steps": [neck_score, trunk_score, legs_score, upper_score, lower_score, 1],
        "angles": {
            "neck_deg": neck_angle,
            "trunk_deg": trunk_angle,
            "knee_flex_deg": knee_flex,
            "upper_arm_deg": upper_elevation,
            "elbow_deg": elbow,
            "wrist_deviation_deg": None,
        },
        "missing_angles": ["wrist_deviation_deg"],
        "measurement_space": "image_2d",
        "assumptions": "upright camera; wrist neutral; no hidden-plane rotation",
    }
