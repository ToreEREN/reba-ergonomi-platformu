"""Pure REBA scoring utilities.

This module deliberately depends only on NumPy and pandas. Importing scoring
must never initialize TensorFlow, OpenCV, Blender or notebook-only packages.
"""

import numpy as np
import pandas as pd

rules = {
    # xz: önden bakış
    # yz: yandan bakış
    #
    # NOT: Model tahminleri "ham açı" verir.
    # Pipeline tarafında "deviation" (sapma) kolonları türetilir:
    #   - _from180_abs: abs(180 - x) - düz durumdan sapma
    #   - _deviation:   model çıktısını REBA convention'a çeviren türetilmiş kolon
    #
    # REBA standart convention:
    #   Trunk: 0° = düz duruş, + = öne eğilme
    #   Neck:  0° = nötr, + = öne eğilme
    #   Legs:  0° = düz, + = bükülme
    #   Upper Arm: 0° = kollar yanda, + = kaldırma
    #   Elbow: 0° = tam bükük (60-100° arası nötr), sapma = skor
    #   Wrist: 0° = nötr, + = bükülme

    # =========================================================
    # STEP 1: HEAD / NECK POSITION
    # =========================================================
    "head_vs_neck_yz_deg": [
        ((0, 20), 1),
        ((20, None), 2),
    ],

    "head_vs_neck_xz_deg": [
        ((5, None), 1),
    ],

    "head_quaternion_rule": {
        "cols": ["head_q_x", "head_q_y", "head_q_z"],
        "type": "abs_max",
        "threshold": 1.0,
        "score": 1,
    },

    "head_quaternion_negative_rule": {
        "cols": ["head_q_x", "head_q_y", "head_q_z"],
        "type": "any_less_than",
        "threshold": -0.3,
        "score": 2,
    },

    # =========================================================
    # STEP 2: TRUNK - spine eğilme açısı
    # Kullanılan kolon: trunk_deviation (pipeline'da hesaplanır)
    # trunk_deviation = spine1_vs_spine2_yz_deg + thigh sapması (kalibre)
    # =========================================================
    "trunk_deviation": [
        ((0, 5), 1),
        ((5, 20), 2),
        ((20, 60), 3),
        ((60, None), 4),
    ],

    "thigh_quaternion_rule": {
        "cols": [
            "thighL_q_x", "thighL_q_y", "thighL_q_z",
            "thighR_q_x", "thighR_q_y", "thighR_q_z",
            "spine1_q_x", "spine1_q_y", "spine1_q_z",
            "spine2_q_x", "spine2_q_y", "spine2_q_z",
            "spine3_q_x", "spine3_q_y", "spine3_q_z",
            "spine4_q_x", "spine4_q_y", "spine4_q_z",
        ],
        "type": "abs_max",
        "threshold": 1.5,
        "score": 1,
    },

    "spine2_quaternion_negative_rule": {
        "cols": ["spine2_q_x", "spine2_q_y", "spine2_q_z"],
        "type": "any_less_than",
        "threshold": -0.5,
        "score": 1,
    },

    # =========================================================
    # STEP 3: LEGS / KNEE
    # Kullanılan kolon: knee_deviation (pipeline'da hesaplanır)
    # knee_deviation = kalibre edilmiş diz bükülme açısı
    # =========================================================
    "kneeR_deviation": [
        ((0, 5), 1),
        ((5, 30), 1),
        ((30, 60), 1),
        ((60, None), 2),
    ],

    "kneeL_deviation": [
        ((0, 5), 1),
        ((5, 30), 1),
        ((30, 60), 1),
        ((60, None), 2),
    ],

    "leg_quaternion_rule": {
        "cols": [
            "shinL_q_x", "shinL_q_y", "shinL_q_z",
            "shinR_q_x", "shinR_q_y", "shinR_q_z",
            "thighL_q_x", "thighL_q_y", "thighL_q_z",
            "thighR_q_x", "thighR_q_y", "thighR_q_z",
        ],
        "type": "abs_max",
        "threshold": 1.5,
        "score": 1,
    },

    # =========================================================
    # STEP 4: UPPER ARM
    # Kullanılan kolon: upperarm_elevation (pipeline'da hesaplanır)
    # Kol yanda = 0°, kol kalkık = büyük açı
    # =========================================================
    "upperarmL_elevation": [
        ((0, 20), 1),
        ((20, 45), 2),
        ((45, 90), 3),
        ((90, None), 4),
    ],

    "upperarmR_elevation": [
        ((0, 20), 1),
        ((20, 45), 2),
        ((45, 90), 3),
        ((90, None), 4),
    ],

    "upperarm_quaternion_rule": {
        "cols": [
            "upperarmL_q_x", "upperarmL_q_y", "upperarmL_q_z",
            "upperarmR_q_x", "upperarmR_q_y", "upperarmR_q_z",
        ],
        "type": "abs_max",
        "threshold": 1.5,
        "score": 1,
    },

    # =========================================================
    # STEP 5: ELBOW
    # Kullanılan kolon: elbow_deviation (pipeline'da hesaplanır)
    # REBA: 60-100° = skor 1, dışı = skor 2
    # =========================================================
    "elbowL_deviation": [
        ((0, 60), 1),
        ((60, 100), 2),
        ((100, None), 2),
    ],

    "elbowR_deviation": [
        ((0, 60), 1),
        ((60, 100), 2),
        ((100, None), 2),
    ],

    # =========================================================
    # STEP 6: WRIST
    # Kullanılan kolon: wrist_deviation (pipeline'da hesaplanır)
    # wrist_deviation = bilek bükülme açısı (0=düz)
    # =========================================================
    "wristL_deviation": [
        ((0, 15), 1),
        ((15, None), 2),
    ],

    "wristR_deviation": [
        ((0, 15), 1),
        ((15, None), 2),
    ],

    "hand_quaternion_rule": {
        "cols": [
            "handL_q_x", "handL_q_y", "handL_q_z",
            "handR_q_x", "handR_q_y", "handR_q_z",
        ],
        "type": "abs_max",
        "threshold": 1.5,
        "score": 1,
    },
}


# Table A: [Trunk-1..5][Neck-1..3][Legs-1..4]
reba_table_A = {
    1: {1: [1,2,3,4],
        2: [1,2,3,4],
        3: [3,4,5,6]},
    
    2: {1: [2,3,4,5],
        2: [3,4,5,6],
        3: [4,5,6,7]},
    
    3: {1: [3,4,5,6],
        2: [4,5,6,7],
        3: [5,6,7,8]},
    
    4: {1: [4,5,6,7],
        2: [5,6,7,8],
        3: [6,7,8,9]},
    
    5: {1: [5,6,7,8],
        2: [6,7,8,9],
        3: [7,8,9,9]}
}
reba_table_B = {
    1: {  # lower arm = 1
        1: [1,2,2],
        2: [1,2,3],
        3: [3,4,5],
        4: [4,5,5],
        5: [6,7,8],
        6: [7,8,8],
    },
    2: {  # lower arm = 2
        1: [1,2,3],
        2: [2,3,4],
        3: [4,5,5],
        4: [5,6,7],
        5: [7,8,8],
        6: [8,9,9],
    }
}
reba_table_C = {
    1:  [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    2:  [1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 8, 8],
    3:  [2, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8],
    4:  [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    5:  [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    6:  [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    7:  [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    8:  [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    9:  [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    10: [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    11: [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    12: [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
}
def get_reba_tableA_score(neck, trunk, legs):
    try:
        return reba_table_A[trunk][neck][legs-1]
    except:
        return np.nan
def get_reba_tableB_score(upper_arm, lower_arm, wrist):
    try:
        return reba_table_B[lower_arm][upper_arm][wrist-1]
    except:
        return np.nan    
def get_reba_tableC_score(tableA, tableB):
    try:
        return reba_table_C[tableA][tableB - 1]
    except:
        return np.nan    
    
def get_force_load_score(load_value, shock_or_rapid=False):
    """
    REBA force/load:
    < 5 kg   -> 0
    5-10 kg  -> 1
    > 10 kg  -> 2

    Eğer shock / rapid build-up varsa +1
    """
    if pd.isna(load_value):
        base_score = 0
    elif load_value < 5:
        base_score = 0
    elif load_value <= 10:
        base_score = 1
    else:
        base_score = 2

    if shock_or_rapid:
        base_score += 1

    return base_score
def get_coupling_score(coupling_value):
    """
    REBA coupling:
    good       -> 0
    fair       -> 1
    poor       -> 2
    unacceptable -> 3

    coupling_value hem string hem sayısal olabilir.
    """
    if pd.isna(coupling_value):
        return 0

    # sayısal geldiyse direkt kullan
    if isinstance(coupling_value, (int, float, np.integer, np.floating)):
        v = int(coupling_value)
        return max(0, min(v, 3))

    # string geldiyse map et
    s = str(coupling_value).strip().lower()

    mapping = {
        "good": 0,
        "well-fit": 0,
        "well fit": 0,
        "ideal": 0,

        "fair": 1,
        "acceptable": 1,

        "poor": 2,
        "bad": 2,

        "unacceptable": 3,
        "none": 3
    }

    return mapping.get(s, 0)


# =========================================================
# Yardımcı fonksiyonlar
# =========================================================
def check_range(x, low, high):
    """
    low-high aralığını kontrol eder.
    Kurallar:
    - low ve high varsa: low <= x <= high
    - sadece low varsa: x > low
    - sadece high varsa: x <= high
    """
    if pd.isna(x):
        return False

    if low is not None and high is not None:
        return low <= x <= high
    elif low is not None and high is None:
        return x > low
    elif low is None and high is not None:
        return x <= high

    return False


def apply_range_rule(x, rule_list, default_score=0):
    """
    Liste tipindeki kuralları uygular.
    Örn:
    [
        ((0, 20), 1),
        ((20, None), 2),
    ]
    """
    if pd.isna(x):
        return default_score

    for (low, high), score in rule_list:
        if check_range(x, low, high):
            return score

    return default_score


def apply_abs_max_rule(df, rule):
    """
    cols içindeki mutlak maksimum threshold'dan büyükse score verir.
    """
    cols = rule["cols"]
    threshold = rule["threshold"]
    score = rule["score"]

    vals = df[cols].abs().max(axis=1)
    return np.where(vals > threshold, score, 0)


def apply_any_less_than_rule(df, rule):
    """
    cols içindeki herhangi bir değer threshold'dan küçükse score verir.
    """
    cols = rule["cols"]
    threshold = rule["threshold"]
    score = rule["score"]

    cond = (df[cols] < threshold).any(axis=1)
    return np.where(cond, score, 0)


def apply_any_in_ranges_rule(df, rule, default_score=0):
    """
    cols içindeki herhangi bir kolon, verilen ranges içindeki bir aralığa düşerse
    ilgili score döner. Birden fazla eşleşme varsa en büyük score alınır.

    rule örneği:
    {
        "cols": [...],
        "type": "any_in_ranges",
        "ranges": [
            ((0, 0), 1),
            ((0, 20), 2),
            ((20, 60), 3),
            ((60, None), 4),
        ],
    }
    """
    cols = rule["cols"]
    ranges = rule["ranges"]

    out = []

    for _, row in df[cols].iterrows():
        matched_scores = []

        for x in row.values:
            if pd.isna(x):
                continue

            for (low, high), score in ranges:
                if check_range(x, low, high):
                    matched_scores.append(score)

        if len(matched_scores) == 0:
            out.append(default_score)
        else:
            out.append(max(matched_scores))

    return np.array(out)


def apply_rule_by_type(df, key, rule):
    """
    rule tipine göre ilgili score serisini döndürür.
    """
    # -----------------------------------------------------
    # 1) Liste tipi kural
    # -----------------------------------------------------
    if isinstance(rule, list):
        return df[key].apply(lambda x: apply_range_rule(x, rule, default_score=0))

    # -----------------------------------------------------
    # 2) Dict tipi özel kural
    # -----------------------------------------------------
    elif isinstance(rule, dict):
        rule_type = rule.get("type")

        if rule_type == "abs_max":
            return pd.Series(apply_abs_max_rule(df, rule), index=df.index)

        elif rule_type == "any_less_than":
            return pd.Series(apply_any_less_than_rule(df, rule), index=df.index)

        elif rule_type == "any_in_ranges":
            return pd.Series(apply_any_in_ranges_rule(df, rule, default_score=0), index=df.index)

        else:
            raise ValueError(f"Bilinmeyen rule type: {rule_type}")

    else:
        raise ValueError(f"{key} için rule tipi desteklenmiyor.")


# =========================================================
# Ana skor üretim fonksiyonu
# =========================================================
def compute_reba_scores(df_reba, rules):
    df_out = df_reba.copy()

    for key, rule in rules.items():
        # New production models may intentionally predict only the 19 angles
        # used by REBA. Optional quaternion rules must not make inference fail.
        required = [key] if isinstance(rule, list) else rule.get("cols", [])
        if any(col not in df_out.columns for col in required):
            continue
        score_col = f"{key}_score"
        df_out[score_col] = apply_rule_by_type(df_out, key, rule)

    # Tüm score kolonları
    score_cols = [c for c in df_out.columns if c.endswith("_score")]

    # -----------------------------------------------------
    # Step bazlı skorlar
    # -----------------------------------------------------
    step1_cols = [
        "head_vs_neck_yz_deg_score",
        "head_vs_neck_xz_deg_score",
        "head_quaternion_rule_score",
        "head_quaternion_negative_rule_score",
    ]

    step2_cols = [
        "trunk_deviation_score",
        "thigh_quaternion_rule_score",
        "spine2_quaternion_negative_rule_score",
    ]

    step3_cols = [
        "kneeR_deviation_score",
        "kneeL_deviation_score",
        "leg_quaternion_rule_score",
    ]

    step4_cols = [
        "upperarmL_elevation_score",
        "upperarmR_elevation_score",
        "upperarm_quaternion_rule_score",
    ]

    step5_cols = [
        "elbowL_deviation_score",
        "elbowR_deviation_score",
    ]

    step6_cols = [
        "wristL_deviation_score",
        "wristR_deviation_score",
        "hand_quaternion_rule_score",
    ]

    # Sadece mevcut kolonları topla
    step1_cols = [c for c in step1_cols if c in df_out.columns]
    step2_cols = [c for c in step2_cols if c in df_out.columns]
    step3_cols = [c for c in step3_cols if c in df_out.columns]
    step4_cols = [c for c in step4_cols if c in df_out.columns]
    step5_cols = [c for c in step5_cols if c in df_out.columns]
    step6_cols = [c for c in step6_cols if c in df_out.columns]

    # REBA standardı: her step için sol/sağ max alınır, quaternion bonus eklenir
    # Step 1: Neck - tek değer + quaternion bonusları
    df_out["Score_Step1"] = df_out[step1_cols].fillna(0).max(axis=1)

    # Step 2: Trunk - tek trunk_deviation + quaternion bonus
    df_out["Score_Step2"] = df_out[step2_cols].fillna(0).max(axis=1)

    # Step 3: Legs - sol/sağ max + quaternion bonus
    df_out["Score_Step3"] = df_out[step3_cols].fillna(0).max(axis=1)

    # Step 4: Upper Arm - sol/sağ max + quaternion bonus
    df_out["Score_Step4"] = df_out[step4_cols].fillna(0).max(axis=1)

    # Step 5: Elbow - sol/sağ max
    df_out["Score_Step5"] = df_out[step5_cols].fillna(0).max(axis=1)

    # Step 6: Wrist - sol/sağ max + quaternion bonus
    df_out["Score_Step6"] = df_out[step6_cols].fillna(0).max(axis=1)


    return df_out

def clamp(val, low, high):
    return max(low, min(high, int(val)))
def get_activity_score(static_posture=False,
                       repetitive_small_range=False,
                       rapid_large_change_or_unstable=False):
    score = 0

    if bool(static_posture):
        score += 1

    if bool(repetitive_small_range):
        score += 1

    if bool(rapid_large_change_or_unstable):
        score += 1

    return score
def classify_reba_risk(score):
    if pd.isna(score):
        return None
    elif score == 1:
        return "Negligible risk"
    elif score in [2, 3]:
        return "Low risk"
    elif score in [4, 5, 6, 7]:
        return "Medium risk"
    elif score in [8, 9, 10]:
        return "High risk"
    else:
        return "Very high risk"
