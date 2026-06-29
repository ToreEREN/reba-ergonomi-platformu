"""Central, explainable REBA scoring engine.

The engine is independent from pose detection and UI. Observed angles may
come from MediaPipe, a depth sensor or a learned estimator; task properties
that cannot be inferred reliably from one image are explicit modifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


TABLE_A = [
    [[1,2,3,4],[1,2,3,4],[3,3,5,6]],
    [[2,3,4,5],[3,4,5,6],[4,5,6,7]],
    [[2,4,5,6],[4,5,6,7],[5,6,7,8]],
    [[3,5,6,7],[5,6,7,8],[6,7,8,9]],
    [[4,6,7,8],[6,7,8,9],[7,8,9,9]],
]
TABLE_B = [
    [[1,2,2],[1,2,3]], [[1,2,3],[2,3,4]],
    [[3,4,5],[4,5,5]], [[4,5,5],[5,6,7]],
    [[6,7,8],[7,8,8]], [[7,8,8],[8,9,9]],
]
TABLE_C = [
    [1,1,1,2,3,3,4,5,6,7,7,7], [1,2,2,3,4,4,5,6,6,7,7,8],
    [2,3,3,3,4,5,6,7,7,8,8,8], [3,4,4,4,5,6,7,8,8,9,9,9],
    [4,4,4,5,6,7,8,8,9,9,9,9], [6,6,6,7,8,8,9,9,10,10,10,10],
    [7,7,7,8,9,9,9,10,10,11,11,11], [8,8,8,9,10,10,10,10,10,11,11,11],
    [9,9,9,10,10,10,11,11,11,12,12,12], [10,10,10,11,11,11,11,12,12,12,12,12],
    [11,11,11,11,12,12,12,12,12,12,12,12], [12]*12,
]


@dataclass
class RebaModifiers:
    neck_twist_or_side: bool = False
    neck_extension: bool = False
    trunk_twist_or_side: bool = False
    trunk_extension: bool = False
    shoulder_raised: bool = False
    arm_abducted: bool = False
    arm_supported: bool = False
    wrist_twist_or_deviation: bool = False
    bilateral_support: bool = True
    load_score: int = 0
    coupling_score: int = 0
    activity_score: int = 0


@dataclass
class RebaResult:
    neck: int; trunk: int; legs: int
    upper_arm: int; lower_arm: int; wrist: int
    table_a: int; table_b: int; score_a: int; score_b: int
    score_c: int; final: int; risk: str; action: str
    explanations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _risk(score):
    if score == 1: return "Negligible risk", "No action necessary"
    if score <= 3: return "Low risk", "Change may be needed"
    if score <= 7: return "Medium risk", "Investigate and implement change"
    if score <= 10: return "High risk", "Investigate and implement change soon"
    return "Very high risk", "Implement change now"


def score_reba(angles: dict, modifiers: Optional[RebaModifiers] = None) -> RebaResult:
    m = modifiers or RebaModifiers()
    exp, warnings = [], []
    def value(name, neutral=0.0):
        v = angles.get(name)
        if v is None:
            warnings.append(f"{name} ölçülemedi; nötr varsayıldı")
            return neutral
        return float(v)

    neck_a, trunk_a = value("neck_deg"), value("trunk_deg")
    knee, upper = value("knee_flex_deg"), value("upper_arm_deg")
    elbow, wrist_a = value("elbow_deg", 80.0), value("wrist_deviation_deg")

    neck = 1 if neck_a <= 20 and not m.neck_extension else 2
    if m.neck_twist_or_side: neck += 1
    neck = min(neck, 3)
    exp.append({"segment":"Neck","angle":neck_a,"score":neck,"reason":"0-20° neutral range; modifier applied if selected"})

    trunk = 2 if m.trunk_extension else (1 if trunk_a <= 5 else 2 if trunk_a <= 20 else 3 if trunk_a <= 60 else 4)
    if m.trunk_twist_or_side: trunk += 1
    trunk = min(trunk, 5)
    exp.append({"segment":"Trunk","angle":trunk_a,"score":trunk,"reason":"Vertical deviation and twist/side modifier"})

    legs = 1 if m.bilateral_support else 2
    legs += 1 if 30 <= knee < 60 else 2 if knee >= 60 else 0
    legs = min(legs, 4)
    exp.append({"segment":"Legs","angle":knee,"score":legs,"reason":"Knee flexion and weight support"})

    upper_arm = 1 if upper <= 20 else 2 if upper <= 45 else 3 if upper <= 90 else 4
    upper_arm += int(m.shoulder_raised) + int(m.arm_abducted) - int(m.arm_supported)
    upper_arm = min(6, max(1, upper_arm))
    exp.append({"segment":"Upper arm","angle":upper,"score":upper_arm,"reason":"Elevation, shoulder, abduction and support"})

    lower_arm = 1 if 60 <= elbow <= 100 else 2
    wrist = 1 if wrist_a <= 15 else 2
    wrist = min(3, wrist + int(m.wrist_twist_or_deviation))
    exp.append({"segment":"Lower arm","angle":elbow,"score":lower_arm,"reason":"60-100° preferred range"})
    exp.append({"segment":"Wrist","angle":wrist_a,"score":wrist,"reason":"Flexion and twist/deviation"})

    table_a = TABLE_A[trunk-1][min(neck,3)-1][legs-1]
    table_b = TABLE_B[upper_arm-1][lower_arm-1][wrist-1]
    score_a = min(12, table_a + max(0, min(3, int(m.load_score))))
    score_b = min(12, table_b + max(0, min(3, int(m.coupling_score))))
    score_c = TABLE_C[score_a-1][score_b-1]
    final = min(15, score_c + max(0, min(3, int(m.activity_score))))
    risk, action = _risk(final)
    return RebaResult(neck,trunk,legs,upper_arm,lower_arm,wrist,table_a,table_b,
                      score_a,score_b,score_c,final,risk,action,exp,warnings)
