"""
REBA Ergonomik Risk Analiz Platformu - Web Uygulaması
=====================================================
TÜBİTAK Projesi - Sanayi İçin Derin Öğrenme Tabanlı Ergonomik Karar Destek Sistemi

Streamlit tabanlı web arayüzü:
- Video yükleme ve analiz
- Webcam ile canlı analiz
- Görsel (resim) analizi
- Sonuç dashboard'u (grafikler, tablolar)
- CSV/JSON dışa aktarma
- Geçmiş analizleri görüntüleme

Çalıştırma:
  streamlit run app.py
"""

import os
import sys
import json
import time
import tempfile
import joblib
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import streamlit as st

from ultralytics import YOLO

# Proje modülleri
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Utils.Functions import add_pose_angles
from Utils.Reba import (
    rules, compute_reba_scores,
    get_reba_tableA_score, get_reba_tableB_score, get_reba_tableC_score,
    get_force_load_score, get_coupling_score, get_activity_score,
    classify_reba_risk, clamp
)
from Utils.RebaCalibration import compute_reba_deviations
from Utils.PoseReba2D import compute_observed_reba
from Utils.MediaPipePose import MP_CONNECTIONS, detect_pose33, compute_reba_pose33
from Utils.RebaEngine import RebaModifiers, score_reba
from Utils.FeatureSchema import validate_feature_frame
from Utils.ModelReliability import predict_with_tree_uncertainty, uncertainty_summary

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="REBA Ergonomik Risk Analizi",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SABİTLER
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n-pose.pt")
MODEL_PATH = os.path.join(BASE_DIR, "BestModel", "best_ft_transformer_multioutput_compatible.h5")
BENCHMARK_MODEL_PATH = os.path.join(BASE_DIR, "ModelExperiments", "extra_trees.joblib")
BENCHMARK_MANIFEST_PATH = os.path.join(BASE_DIR, "ModelExperiments", "manifest.json")
SCALER_PATH = os.path.join(BASE_DIR, "BestModel", "ft_transformer_scaler.pkl")
META_PATH = os.path.join(BASE_DIR, "BestModel", "advanced_ft_transformer_meta.json")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

BASE_INPUT_COLS = []
for name in KEYPOINT_NAMES:
    BASE_INPUT_COLS.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

TARGET_COLS = [
    'elbowL_yz_deg', 'elbowR_yz_deg',
    'head_vs_neck_xz_deg', 'head_vs_neck_yz_deg',
    'kneeL_xz_deg', 'kneeL_yz_deg', 'kneeR_xz_deg', 'kneeR_yz_deg',
    'shoulderL_vs_upperarmL_xz_deg', 'shoulderL_vs_upperarmL_yz_deg',
    'shoulderR_vs_upperarmR_xz_deg', 'shoulderR_vs_upperarmR_yz_deg',
    'spine1_vs_spine2_yz_deg', 'spine2_vs_spine3_yz_deg', 'spine3_vs_spine4_yz_deg',
    'thighL_vs_spine1_yz_deg', 'thighR_vs_spine1_yz_deg',
    'wristL_yz_deg', 'wristR_yz_deg',
    'head_q_x', 'head_q_y', 'head_q_z',
    'upperarmL_q_x', 'upperarmL_q_y', 'upperarmL_q_z',
    'upperarmR_q_x', 'upperarmR_q_y', 'upperarmR_q_z',
    'handL_q_x', 'handL_q_y', 'handL_q_z',
    'handR_q_x', 'handR_q_y', 'handR_q_z',
    'thighL_q_x', 'thighL_q_y', 'thighL_q_z',
    'thighR_q_x', 'thighR_q_y', 'thighR_q_z',
    'shinL_q_x', 'shinL_q_y', 'shinL_q_z',
    'shinR_q_x', 'shinR_q_y', 'shinR_q_z',
    'spine1_q_x', 'spine1_q_y', 'spine1_q_z',
    'spine2_q_x', 'spine2_q_y', 'spine2_q_z',
    'spine3_q_x', 'spine3_q_y', 'spine3_q_z',
    'spine4_q_x', 'spine4_q_y', 'spine4_q_z'
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

RISK_COLORS_BGR = {
    "Negligible risk": (0, 200, 0),
    "Low risk": (0, 255, 100),
    "Medium risk": (0, 200, 255),
    "High risk": (0, 100, 255),
    "Very high risk": (0, 0, 255),
}


# ============================================================
# MODEL YÜKLEME (Cache ile)
# ============================================================
@st.cache_resource
def load_all_models():
    yolo = YOLO(YOLO_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Prefer the independently benchmarked angle-only model. It is faster on
    # CPU and avoids poorly identified quaternion targets.
    if os.path.exists(BENCHMARK_MODEL_PATH) and os.path.exists(BENCHMARK_MANIFEST_PATH):
        with open(BENCHMARK_MANIFEST_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["runtime"] = "sklearn_angle_model"
        meta["target_names"] = meta["targets"]
        return yolo, joblib.load(BENCHMARK_MODEL_PATH), scaler, meta

    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    meta["runtime"] = "tensorflow_ft_transformer"

    # TensorFlow is a legacy fallback only. The production benchmark model
    # above can run even when the local TensorFlow native runtime is broken.
    from Utils.Models import build_advanced_ft_transformer_regression

    ft_model = build_advanced_ft_transformer_regression(
        n_features=meta['input_dim'],
        n_targets=meta['n_targets'],
        d_token=96, n_blocks=6, n_heads=8,
        dropout=0.20, ff_mult=4,
        shared_mlp_units=(256, 128),
        per_target_hidden=64,
        use_uncertainty=meta['use_uncertainty']
    )
    _dummy = np.zeros((1, meta['input_dim']), dtype='float32')
    _ = ft_model(_dummy)
    ft_model.load_weights(MODEL_PATH)

    return yolo, ft_model, scaler, meta


# ============================================================
# ANALİZ FONKSİYONLARI
# ============================================================
def _pose33_only_result(pose33, world33, force_load=0, coupling=0,
                        shock_or_rapid=False, static_posture=False,
                        repetitive=False, unstable=False, manual_modifiers=None):
    observed = compute_reba_pose33(pose33)
    if observed is None:
        return None
    mods = manual_modifiers or RebaModifiers(
        load_score=get_force_load_score(force_load, shock_or_rapid),
        coupling_score=get_coupling_score(coupling),
        activity_score=get_activity_score(static_posture, repetitive, unstable),
    )
    scored = score_reba(observed["angles"], mods)
    return {
        "step1_neck":scored.neck,"step2_trunk":scored.trunk,"step3_legs":scored.legs,
        "step4_upper_arm":scored.upper_arm,"step5_elbow":scored.lower_arm,"step6_wrist":scored.wrist,
        "score_a":scored.score_a,"score_b":scored.score_b,
        "final_reba":scored.final,"risk_level":scored.risk,
        "predicted_angles":{},"observed_angles":observed["angles"],
        "scoring_source":"mediapipe_pose33","skeleton_type":"mediapipe33",
        "world_landmarks_available":world33 is not None,
        "explanations":scored.explanations,"warnings":scored.warnings,
        "recommended_action":scored.action,"table_a":scored.table_a,
        "table_b":scored.table_b,"table_c":scored.score_c,
    }


def analyze_frame(frame, yolo_model, ft_model, scaler, meta, conf=0.25,
                  force_load=0, coupling=0, shock_or_rapid=False,
                  static_posture=False, repetitive=False, unstable=False,
                  manual_modifiers=None):
    """Tek kareyi analiz eder."""
    pose33, world33 = detect_pose33(frame)
    results = yolo_model.predict(source=frame, conf=conf, save=False, verbose=False)

    if not results or results[0].keypoints is None:
        return (_pose33_only_result(pose33, world33, force_load, coupling,
                shock_or_rapid, static_posture, repetitive, unstable, manual_modifiers), pose33) if pose33 else (None, None)

    kp_data = results[0].keypoints.data.cpu().numpy()
    if kp_data.shape[0] == 0:
        return (_pose33_only_result(pose33, world33, force_load, coupling,
                shock_or_rapid, static_posture, repetitive, unstable, manual_modifiers), pose33) if pose33 else (None, None)

    pts = kp_data[0]
    keypoints = [[float(kp[0]), float(kp[1]), float(kp[2])] for kp in pts]

    input_row = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y, z = keypoints[i] if i < len(keypoints) else (0.0, 0.0, 0.0)
        input_row[f"{name}_x"] = x
        input_row[f"{name}_y"] = y
        input_row[f"{name}_z"] = z

    df_kp = pd.DataFrame([input_row])
    df_kp.columns = BASE_INPUT_COLS
    df_input = add_pose_angles(df_kp, add_planar_angles=True)
    validate_feature_frame(df_input)

    X_input = df_input.values.astype('float32')
    X_scaled = scaler.transform(X_input)
    model_uncertainty = None
    if meta.get("runtime") == "sklearn_angle_model":
        y_pred, y_std = predict_with_tree_uncertainty(ft_model, X_scaled)
        df_pred = pd.DataFrame(y_pred, columns=meta["target_names"])
        model_uncertainty = uncertainty_summary(y_pred, y_std, meta["target_names"])
    else:
        from Utils.Models import predict_advanced_multioutput
        y_pred = predict_advanced_multioutput(ft_model, X_scaled, use_uncertainty=False, batch_size=1)
        df_pred = pd.DataFrame(y_pred, columns=TARGET_COLS)

    df_reba = compute_reba_deviations(df_pred)

    df_sc = compute_reba_scores(df_reba, rules)
    df_sc = df_sc[["Score_Step1", "Score_Step2", "Score_Step3",
                   "Score_Step4", "Score_Step5", "Score_Step6"]].copy()

    s1 = int(df_sc["Score_Step1"].iloc[0])
    s2 = int(df_sc["Score_Step2"].iloc[0])
    s3 = int(df_sc["Score_Step3"].iloc[0])
    s4 = int(df_sc["Score_Step4"].iloc[0])
    s5 = int(df_sc["Score_Step5"].iloc[0])
    s6 = int(df_sc["Score_Step6"].iloc[0])

    # Live scoring must primarily follow geometry actually visible in the
    # frame. The learned model remains useful for hidden-angle estimates, but
    # its third YOLO input is confidence rather than true depth.
    observed = (compute_reba_pose33(pose33, confidence=max(0.25, conf))
                if pose33 is not None else
                compute_observed_reba(keypoints, confidence=max(0.20, conf)))
    scoring_source = "learned_angle_fallback"
    if observed is not None:
        mods = manual_modifiers or RebaModifiers(
            load_score=get_force_load_score(force_load, shock_or_rapid),
            coupling_score=get_coupling_score(coupling),
            activity_score=get_activity_score(static_posture, repetitive, unstable),
        )
        scored = score_reba(observed["angles"], mods)
        return {
            "step1_neck":scored.neck,"step2_trunk":scored.trunk,"step3_legs":scored.legs,
            "step4_upper_arm":scored.upper_arm,"step5_elbow":scored.lower_arm,"step6_wrist":scored.wrist,
            "score_a":scored.score_a,"score_b":scored.score_b,
            "final_reba":scored.final,"risk_level":scored.risk,
            "predicted_angles":{col:float(df_pred[col].iloc[0]) for col in TARGET_COLS[:19]},
            "observed_angles":observed["angles"],"scoring_source":"mediapipe_pose33" if pose33 is not None else "observed_2d_pose",
            "skeleton_type":"mediapipe33" if pose33 is not None else "coco17",
            "world_landmarks_available":world33 is not None,
            "explanations":scored.explanations,"warnings":scored.warnings,
            "recommended_action":scored.action,"table_a":scored.table_a,
            "table_b":scored.table_b,"table_c":scored.score_c,
            "model_uncertainty":model_uncertainty,
        }, pose33 if pose33 is not None else keypoints

    tA = get_reba_tableA_score(neck=clamp(s1, 1, 3), trunk=clamp(s2, 1, 5), legs=clamp(s3, 1, 4))
    scoreA = tA + get_force_load_score(force_load, shock_or_rapid)
    tB = get_reba_tableB_score(upper_arm=clamp(s4, 1, 6), lower_arm=clamp(s5, 1, 2), wrist=clamp(s6, 1, 3))
    scoreB = tB + get_coupling_score(coupling)
    tC = get_reba_tableC_score(clamp(scoreA, 1, 12), clamp(scoreB, 1, 12))
    final_reba = tC + get_activity_score(static_posture, repetitive, unstable)
    risk_level = classify_reba_risk(final_reba)

    result = {
        "step1_neck": s1, "step2_trunk": s2, "step3_legs": s3,
        "step4_upper_arm": s4, "step5_elbow": s5, "step6_wrist": s6,
        "score_a": int(scoreA), "score_b": int(scoreB),
        "final_reba": int(final_reba), "risk_level": risk_level,
        "predicted_angles": {col: float(df_pred[col].iloc[0]) for col in TARGET_COLS[:19]},
        "observed_angles": observed["angles"] if observed else {},
        "scoring_source": scoring_source,
        "skeleton_type": "mediapipe33" if pose33 is not None else "coco17",
        "world_landmarks_available": world33 is not None,
        "model_uncertainty": model_uncertainty,
    }

    return result, pose33 if pose33 is not None else keypoints


def draw_annotated_frame(frame, result, keypoints, conf_threshold=0.3):
    """İskelet ve REBA overlay çizer."""
    annotated = frame.copy()

    if keypoints:
        connections = MP_CONNECTIONS if result and result.get("skeleton_type") == "mediapipe33" else SKELETON_CONNECTIONS
        for (i, j) in connections:
            if i >= len(keypoints) or j >= len(keypoints):
                continue
            x1, y1, c1 = keypoints[i]
            x2, y2, c2 = keypoints[j]
            if c1 < conf_threshold or c2 < conf_threshold:
                continue
            cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_AA)

        for x, y, c in keypoints:
            if c < conf_threshold:
                continue
            cv2.circle(annotated, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)

    if result:
        risk_color = RISK_COLORS_BGR.get(result["risk_level"], (255, 255, 255))
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (330, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        y = 35
        cv2.putText(annotated, "REBA ANALIZI", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40
        cv2.putText(annotated, f"SKOR: {result['final_reba']}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, risk_color, 3)
        y += 30
        cv2.putText(annotated, f"Risk: {result['risk_level']}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, risk_color, 2)
        y += 28
        cv2.putText(annotated, f"Boyun:{result['step1_neck']} Govde:{result['step2_trunk']} Bacak:{result['step3_legs']}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 20
        cv2.putText(annotated, f"UstKol:{result['step4_upper_arm']} Dirsek:{result['step5_elbow']} Bilek:{result['step6_wrist']}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    else:
        cv2.putText(annotated, "Kisi tespit edilemedi", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return annotated


def build_current_modifiers(force_load=None, coupling=None):
    """Build task modifiers that cannot be inferred reliably from one frame."""
    load = st.session_state.get("global_load_kg", 0) if force_load is None else force_load
    coupling_value = st.session_state.get("global_coupling", 0) if coupling is None else coupling
    return RebaModifiers(
        neck_twist_or_side=st.session_state.get("mod_neck", False),
        neck_extension=st.session_state.get("mod_neck_ext", False),
        trunk_twist_or_side=st.session_state.get("mod_trunk", False),
        trunk_extension=st.session_state.get("mod_trunk_ext", False),
        shoulder_raised=st.session_state.get("mod_shoulder", False),
        arm_abducted=st.session_state.get("mod_abducted", False),
        arm_supported=st.session_state.get("mod_supported", False),
        wrist_twist_or_deviation=st.session_state.get("mod_wrist", False),
        bilateral_support=st.session_state.get("mod_bilateral", True),
        load_score=get_force_load_score(load, st.session_state.get("mod_shock", False)),
        coupling_score=get_coupling_score(coupling_value),
        activity_score=get_activity_score(
            st.session_state.get("mod_static", False),
            st.session_state.get("mod_repetitive", False),
            st.session_state.get("mod_unstable", False),
        ),
    )


# ============================================================
# SAYFA: ANA PANEL
# ============================================================
def page_home():
    st.title("🏭 REBA Ergonomik Risk Analiz Platformu")
    st.markdown("**TUBİTAK Projesi** - Sanayi İçin Derin Öğrenme Tabanlı Ergonomik Karar Destek Sistemi")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📷 Görsel Analiz")
        st.write("Tek bir fotoğraftan REBA analizi yapın.")
        if st.button("Görsel Analiz", key="btn_img", use_container_width=True):
            st.session_state.page = "image"
            st.rerun()

    with col2:
        st.markdown("### 🎬 Video Analiz")
        st.write("Video dosyası yükleyerek kare kare analiz yapın.")
        if st.button("Video Analiz", key="btn_vid", use_container_width=True):
            st.session_state.page = "video"
            st.rerun()

    with col3:
        st.markdown("### 📹 Canlı Webcam")
        st.write("Webcam ile gerçek zamanlı analiz yapın.")
        if st.button("Webcam Analiz", key="btn_cam", use_container_width=True):
            st.session_state.page = "webcam"
            st.rerun()

    st.markdown("---")

    st.markdown("### Sistem Mimarisi")
    st.code("""
    Görüntü/Video/Webcam
         │
         ▼
    MediaPipe Pose (33 Keypoint + göreli 3B)
         │
         ▼
    Görünür REBA Açıları (boyun/gövde/kol/bilek/diz/ayak)
         │
         ├──────── YOLO11 + Extra Trees (19 gizli açı, ikincil)
         │
         ▼
    Merkezî Açıklanabilir REBA Motoru
    (Tablo A + B + C + görev modifikatörleri)
         │
         ▼
    Risk + Aksiyon + "Neden Bu Skor?" (1-15 → 5 seviye)
    """, language="text")

    st.markdown("### Risk Seviyeleri")
    risk_df = pd.DataFrame({
        "Skor": ["1", "2-3", "4-7", "8-10", "11-15"],
        "Seviye": ["Negligible", "Low", "Medium", "High", "Very High"],
        "Aksiyon": ["Gerekli değil", "Gerekebilir", "Gerekli", "Kısa sürede gerekli", "Acil müdahale"],
    })
    st.table(risk_df)


# ============================================================
# SAYFA: GÖRSEL ANALİZ
# ============================================================
def page_image_analysis():
    st.title("📷 Görsel Analiz")
    st.write("Bir fotoğraf yükleyin, REBA ergonomik risk analizi yapılsın.")

    yolo_model, ft_model, scaler, meta = load_all_models()

    uploaded_file = st.file_uploader("Görsel Seçin", type=["jpg", "jpeg", "png", "bmp"])

    col_conf, col_force, col_coupling = st.columns(3)
    with col_conf:
        conf = st.slider("YOLO Güven Eşiği", 0.1, 0.9, 0.25, 0.05)
    with col_force:
        force_load = st.number_input("Yük (kg)", 0, 50, 0)
    with col_coupling:
        coupling = st.selectbox("Kavrama Kalitesi", ["Good (0)", "Fair (1)", "Poor (2)", "Unacceptable (3)"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if st.button("🔍 Analiz Et", use_container_width=True, type="primary"):
            with st.spinner("Analiz yapılıyor..."):
                coupling_score = int(coupling.split("(")[1].split(")")[0])
                result, keypoints = analyze_frame(
                    frame, yolo_model, ft_model, scaler, meta, conf,
                    force_load=force_load, coupling=coupling_score,
                    manual_modifiers=build_current_modifiers(force_load, coupling_score),
                )

            if result is None:
                st.error("Görüntüde kişi tespit edilemedi. Farklı bir görsel deneyin veya güven eşiğini düşürün.")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Yüklenen Görsel")
                return

            annotated = draw_annotated_frame(frame, result, keypoints)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption="REBA Analiz Sonucu", use_column_width=True)

            with col2:
                reba_score = result["final_reba"]
                risk = result["risk_level"]

                if reba_score <= 3:
                    color = "green"
                elif reba_score <= 7:
                    color = "orange"
                else:
                    color = "red"

                st.markdown(f"### REBA Skoru")
                st.markdown(f"<h1 style='color:{color}; text-align:center;'>{reba_score}</h1>",
                            unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center; color:{color};'><b>{risk}</b></p>",
                            unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**Detay Skorları**")

                metrics_data = {
                    "Boyun (Step 1)": result["step1_neck"],
                    "Gövde (Step 2)": result["step2_trunk"],
                    "Bacaklar (Step 3)": result["step3_legs"],
                    "Üst Kol (Step 4)": result["step4_upper_arm"],
                    "Dirsek (Step 5)": result["step5_elbow"],
                    "Bilek (Step 6)": result["step6_wrist"],
                }

                for label, val in metrics_data.items():
                    st.metric(label, val)

                st.markdown("---")
                st.metric("Score A", result["score_a"])
                st.metric("Score B", result["score_b"])

            # Tahmin edilen açılar
            with st.expander("Tahmin Edilen Açılar (Detay)"):
                angles_df = pd.DataFrame([result["predicted_angles"]]).T
                angles_df.columns = ["Derece"]
                st.dataframe(angles_df, use_container_width=True)

            if result.get("observed_angles"):
                with st.expander("Kareden Doğrudan Ölçülen 2B Açılar"):
                    observed_df = pd.DataFrame([result["observed_angles"]]).T
                    observed_df.columns = ["Derece"]
                    st.caption("Canlı REBA alt skorları öncelikle bu görünür geometriden hesaplanır.")
                    st.dataframe(observed_df, use_container_width=True)

            if result.get("explanations"):
                with st.expander("Neden bu skor çıktı?", expanded=True):
                    explanation_df = pd.DataFrame(result["explanations"])
                    st.dataframe(explanation_df, use_container_width=True)
                    st.info(f"Önerilen aksiyon: {result.get('recommended_action', '—')}")
            for warning in result.get("warnings", []):
                st.warning(warning)

            uncertainty = result.get("model_uncertainty")
            if uncertainty:
                with st.expander("Model belirsizliği ve eşik uyarıları"):
                    st.metric("Ortalama ağaçlar arası std", f"{uncertainty['mean_tree_std_deg']:.1f}°")
                    st.metric("En yüksek ağaçlar arası std", f"{uncertainty['max_tree_std_deg']:.1f}°")
                    if uncertainty["flagged_targets"]:
                        st.warning(
                            "REBA eşiğine yakın veya belirsiz hedefler: "
                            + ", ".join(uncertainty["flagged_targets"])
                        )
                    st.caption("Bu değer kalibre edilmiş güven aralığı değil; Extra Trees üyeleri arasındaki yayılımdır.")

            # İndir
            result_json = json.dumps(result, ensure_ascii=False, indent=2)
            st.download_button("📥 JSON İndir", result_json,
                               file_name=f"reba_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                               mime="application/json")
        else:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Yüklenen Görsel", use_column_width=True)


# ============================================================
# SAYFA: VİDEO ANALİZ
# ============================================================
def page_video_analysis():
    st.title("🎬 Video Analiz")
    st.write("Video dosyası yükleyin, kare kare REBA analizi yapılsın.")

    yolo_model, ft_model, scaler, meta = load_all_models()

    uploaded_video = st.file_uploader("Video Seçin", type=["mp4", "avi", "mov", "mkv"])

    col1, col2, col3 = st.columns(3)
    with col1:
        conf = st.slider("YOLO Güven Eşiği", 0.1, 0.9, 0.25, 0.05, key="vid_conf")
    with col2:
        skip_frames = st.number_input("Kare Atlama", 0, 30, 2, help="0=her kare, 2=her 3. kare")
    with col3:
        max_frames = st.number_input("Maks Kare (0=tümü)", 0, 10000, 0)

    if uploaded_video is not None:
        # Geçici dosyaya yaz
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.info(f"Video: {w}x{h} @ {fps:.1f} FPS | Toplam: {total} kare ({total/fps:.1f} sn)")

        if st.button("▶️ Analizi Başlat", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_display = st.empty()

            csv_rows = []
            frame_idx = 0
            processed = 0
            last_result = None
            start_time = time.time()
            actual_max = max_frames if max_frames > 0 else total

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames > 0 and frame_idx > max_frames:
                    break

                should_analyze = (skip_frames == 0) or (frame_idx % (skip_frames + 1) == 0)

                if should_analyze:
                    result, keypoints = analyze_frame(
                        frame, yolo_model, ft_model, scaler, meta, conf,
                        manual_modifiers=build_current_modifiers(),
                    )
                    last_result = result
                    processed += 1

                    row = {"frame": frame_idx, "timestamp_sec": round(frame_idx / fps, 3)}
                    if result:
                        row.update({
                            "final_reba": result["final_reba"],
                            "risk_level": result["risk_level"],
                            "step1_neck": result["step1_neck"],
                            "step2_trunk": result["step2_trunk"],
                            "step3_legs": result["step3_legs"],
                            "step4_upper_arm": result["step4_upper_arm"],
                            "step5_elbow": result["step5_elbow"],
                            "step6_wrist": result["step6_wrist"],
                            "score_a": result["score_a"],
                            "score_b": result["score_b"],
                        })
                        row.update(result.get("predicted_angles", {}))
                    else:
                        row["final_reba"] = None
                        row["risk_level"] = "Tespit edilemedi"
                    csv_rows.append(row)
                else:
                    result = last_result
                    keypoints = None

                # Her 10 karede görsel güncelle
                if frame_idx % 10 == 0:
                    progress = min(frame_idx / actual_max, 1.0)
                    progress_bar.progress(progress)
                    elapsed = time.time() - start_time
                    proc_fps = processed / elapsed if elapsed > 0 else 0
                    reba_val = result['final_reba'] if result else '-'
                    status_text.text(f"Kare {frame_idx}/{actual_max} | REBA: {reba_val} | {proc_fps:.1f} FPS")

                    if result and keypoints:
                        annotated = draw_annotated_frame(frame, result, keypoints)
                        frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                            use_column_width=True)

            cap.release()
            os.unlink(tfile.name)
            progress_bar.progress(1.0)

            total_time = time.time() - start_time
            status_text.text(f"Tamamlandı! {processed} kare / {total_time:.1f} sn / {processed/total_time:.1f} FPS")

            # Sonuçları kaydet ve göster
            if csv_rows:
                df_results = pd.DataFrame(csv_rows)
                st.session_state["video_results"] = df_results
                st.session_state["video_source_name"] = uploaded_video.name

                st.success(f"Analiz tamamlandı! {processed} kare işlendi.")
                show_video_results(df_results, uploaded_video.name, fps)

        cap.release()


def show_video_results(df, source_name, fps):
    """Video analiz sonuçlarını gösterir."""
    df_valid = df[df["final_reba"].notna()].copy()

    if df_valid.empty:
        st.warning("Geçerli analiz sonucu bulunamadı.")
        return

    reba_scores = df_valid["final_reba"].values

    # Özet metrikler
    st.markdown("### Özet")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Ortalama REBA", f"{np.mean(reba_scores):.1f}")
    col2.metric("Medyan", f"{np.median(reba_scores):.0f}")
    col3.metric("Min", f"{int(np.min(reba_scores))}")
    col4.metric("Max", f"{int(np.max(reba_scores))}")
    col5.metric("Yüksek Risk %", f"{(reba_scores >= 8).sum() / len(reba_scores) * 100:.1f}%")

    # Grafikler
    st.markdown("### Grafikler")
    tab1, tab2, tab3, tab4 = st.tabs(["Zaman Serisi", "Risk Dağılımı", "Step Skorları", "Açılar"])

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 4))
        t = df_valid["timestamp_sec"].values
        ax.plot(t, reba_scores, 'b-', lw=1, alpha=0.7)
        ax.axhspan(0, 1, alpha=0.1, color='green')
        ax.axhspan(1, 3, alpha=0.1, color='limegreen')
        ax.axhspan(3, 7, alpha=0.1, color='orange')
        ax.axhspan(7, 10, alpha=0.1, color='orangered')
        ax.axhspan(10, 15, alpha=0.1, color='red')
        if len(reba_scores) > 10:
            w = min(30, len(reba_scores) // 3)
            ax.plot(t, pd.Series(reba_scores).rolling(w, min_periods=1).mean(), 'r-', lw=2, label=f'Hareketli Ort.({w})')
            ax.legend()
        ax.set_xlabel("Zaman (sn)")
        ax.set_ylabel("REBA Skoru")
        ax.set_title("REBA Skoru Zaman Serisi")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(6, 6))
        risk_counts = df_valid["risk_level"].value_counts()
        colors_map = {'Negligible risk': '#2ecc71', 'Low risk': '#a3d977',
                      'Medium risk': '#f39c12', 'High risk': '#e67e22', 'Very high risk': '#e74c3c'}
        labels, sizes, cols = [], [], []
        for r in ['Negligible risk', 'Low risk', 'Medium risk', 'High risk', 'Very high risk']:
            if r in risk_counts.index:
                labels.append(r.replace(' risk', ''))
                sizes.append(risk_counts[r])
                cols.append(colors_map[r])
        if sizes:
            ax.pie(sizes, labels=labels, colors=cols, autopct='%1.1f%%', startangle=90)
        ax.set_title("Risk Dağılımı")
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 5))
        step_cols = ['step1_neck', 'step2_trunk', 'step3_legs', 'step4_upper_arm', 'step5_elbow', 'step6_wrist']
        step_labels = ['Boyun', 'Gövde', 'Bacak', 'Üst Kol', 'Dirsek', 'Bilek']
        data = [df_valid[c].dropna().values for c in step_cols if c in df_valid.columns]
        if data:
            bp = ax.boxplot(data, labels=step_labels[:len(data)], patch_artist=True)
            bpc = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
            for patch, color in zip(bp['boxes'], bpc[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_ylabel("Skor")
        ax.set_title("Step Skorları Dağılımı")
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()

    with tab4:
        angle_cols = [c for c in df_valid.columns if c.endswith('_deg') and c in TARGET_COLS[:19]]
        if angle_cols and len(df_valid) > 2:
            fig, ax = plt.subplots(figsize=(12, 5))
            t = df_valid["timestamp_sec"].values
            for col in angle_cols[:8]:
                ax.plot(t, df_valid[col].values, lw=1, label=col.replace('_deg', ''), alpha=0.8)
            ax.set_xlabel("Zaman (sn)")
            ax.set_ylabel("Derece")
            ax.set_title("Tahmin Edilen Açılar")
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

    # Tablo
    st.markdown("### Veri Tablosu")
    display_cols = ['frame', 'timestamp_sec', 'final_reba', 'risk_level',
                    'step1_neck', 'step2_trunk', 'step3_legs',
                    'step4_upper_arm', 'step5_elbow', 'step6_wrist']
    existing_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[existing_cols], use_container_width=True, height=300)

    # İndirme
    st.markdown("### Dışa Aktar")
    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 CSV İndir", csv_data,
                           file_name=f"reba_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", use_container_width=True)
    with col2:
        summary = {
            "source": source_name,
            "date": datetime.now().isoformat(),
            "frames": len(df_valid),
            "mean_reba": float(np.mean(reba_scores)),
            "max_reba": int(np.max(reba_scores)),
            "high_risk_pct": f"{(reba_scores >= 8).sum() / len(reba_scores) * 100:.1f}%",
            "risk_dist": df_valid["risk_level"].value_counts().to_dict(),
        }
        st.download_button("📥 JSON Özet İndir", json.dumps(summary, ensure_ascii=False, indent=2),
                           file_name=f"reba_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                           mime="application/json", use_container_width=True)


# ============================================================
# SAYFA: WEBCAM
# ============================================================
def page_webcam():
    st.title("📹 Webcam Canlı Analiz")
    st.write("Webcam ile gerçek zamanlı REBA analizi.")

    yolo_model, ft_model, scaler, meta = load_all_models()

    col_conf, col_skip, col_dur = st.columns(3)
    with col_conf:
        conf = st.slider("YOLO Güven Eşiği", 0.1, 0.9, 0.3, 0.05, key="cam_conf")
    with col_skip:
        analyze_every = st.slider("Her N karede analiz", 2, 15, 5, key="cam_skip",
                                  help="Yüksek = daha akıcı ama daha az analiz")
    with col_dur:
        max_seconds = st.number_input("Süre (sn, 0=sınırsız)", 0, 300, 30, key="cam_dur")

    st.markdown("---")

    if "webcam_results" not in st.session_state:
        st.session_state.webcam_results = []

    if st.button("▶️ Webcam Analizi Başlat", use_container_width=True, type="primary"):
        st.session_state.webcam_results = []

        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()
        progress_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam açılamadı! Kameranın bağlı ve başka uygulama tarafından kullanılmadığından emin olun.")
            return

        # Webcam çözünürlüğünü düşür (performans)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        start_time = time.time()
        last_result = None
        last_keypoints = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_count += 1
                elapsed = time.time() - start_time

                # Süre limiti
                if max_seconds > 0 and elapsed > max_seconds:
                    break

                # Analiz (sadece her N karede)
                if frame_count % analyze_every == 0:
                    # Küçült → analiz et → performans artışı
                    small_frame = cv2.resize(frame, (320, 240))
                    result, keypoints = analyze_frame(
                        small_frame, yolo_model, ft_model, scaler, meta, conf,
                        manual_modifiers=build_current_modifiers(),
                    )

                    if result:
                        # Keypoint'leri orijinal boyuta ölçekle
                        scale_x = frame.shape[1] / 320.0
                        scale_y = frame.shape[0] / 240.0
                        keypoints = [[kp[0]*scale_x, kp[1]*scale_y, kp[2]] for kp in keypoints]
                        last_result = result
                        last_keypoints = keypoints

                        st.session_state.webcam_results.append({
                            "frame": frame_count,
                            "timestamp_sec": round(elapsed, 2),
                            "final_reba": result["final_reba"],
                            "risk_level": result["risk_level"],
                            "step1_neck": result["step1_neck"],
                            "step2_trunk": result["step2_trunk"],
                            "step3_legs": result["step3_legs"],
                            "step4_upper_arm": result["step4_upper_arm"],
                            "step5_elbow": result["step5_elbow"],
                            "step6_wrist": result["step6_wrist"],
                            "score_a": result["score_a"],
                            "score_b": result["score_b"],
                        })
                    else:
                        last_result = None
                        last_keypoints = None

                # Her 3 karede ekranı güncelle (donmayı önler)
                if frame_count % 3 == 0:
                    annotated = draw_annotated_frame(frame, last_result, last_keypoints)
                    frame_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        use_column_width=True
                    )

                    fps_val = frame_count / elapsed if elapsed > 0 else 0
                    reba_text = (f"**REBA: {last_result['final_reba']}** | Risk: {last_result['risk_level']} "
                                 f"| Kaynak: {last_result.get('scoring_source', 'bilinmiyor')}") if last_result else "Kişi tespit edilemedi"
                    metrics_placeholder.markdown(f"{reba_text} | FPS: {fps_val:.0f} | {elapsed:.0f}s")

                    if max_seconds > 0:
                        progress_placeholder.progress(min(elapsed / max_seconds, 1.0))

        except Exception as e:
            st.error(f"Hata: {e}")
        finally:
            cap.release()

        # Sonuçları göster
        if st.session_state.webcam_results:
            st.success(f"Analiz tamamlandı! {len(st.session_state.webcam_results)} kare analiz edildi.")
            df_cam = pd.DataFrame(st.session_state.webcam_results)
            show_video_results(df_cam, "webcam", 30.0)
        else:
            st.warning("Hiçbir kare analiz edilemedi.")


# ============================================================
# SAYFA: GEÇMİŞ ANALİZLER
# ============================================================
def page_history():
    st.title("📂 Geçmiş Analizler")

    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')] if os.path.exists(RESULTS_DIR) else []

    if not result_files:
        st.info("Henüz kayıtlı analiz bulunmuyor. Video veya görsel analiz yaparak sonuçları kaydedin.")
        return

    selected = st.selectbox("Analiz Seçin", sorted(result_files, reverse=True))

    if selected:
        df = pd.read_csv(os.path.join(RESULTS_DIR, selected))
        st.write(f"Dosya: {selected} | Satır: {len(df)}")
        show_video_results(df, selected.replace('.csv', ''), 30.0)


# ============================================================
# SIDEBAR & NAVIGATION
# ============================================================
def main():
    # CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/T%C3%BCbitak_logo.png/220px-T%C3%BCbitak_logo.png", width=150)
        st.markdown("## REBA Platform")
        st.markdown("---")

        page = st.radio("Sayfa Seçin", [
            "🏠 Ana Panel",
            "📷 Görsel Analiz",
            "🎬 Video Analiz",
            "📹 Webcam",
            "📂 Geçmiş Analizler",
        ], index=0)

        with st.expander("REBA görev ve duruş modifikatörleri"):
            st.caption("Tek kameradan güvenilir ölçülemeyen koşulları işaretleyin.")
            st.number_input("Taşınan yük (kg)", 0, 100, 0, key="global_load_kg")
            coupling_label = st.selectbox(
                "Kavrama kalitesi", ["Good (0)", "Fair (1)", "Poor (2)", "Unacceptable (3)"],
                key="global_coupling_label",
            )
            st.session_state.global_coupling = int(coupling_label.split("(")[1].split(")")[0])
            st.checkbox("Şok / ani kuvvet artışı", key="mod_shock")
            st.checkbox("Boyun dönüşü veya yana eğilme", key="mod_neck")
            st.checkbox("Boyun geriye bükülmüş", key="mod_neck_ext")
            st.checkbox("Gövde dönüşü veya yana eğilme", key="mod_trunk")
            st.checkbox("Gövde geriye bükülmüş", key="mod_trunk_ext")
            st.checkbox("Omuz yükselmiş", key="mod_shoulder")
            st.checkbox("Kol yana açılmış (abdüksiyon)", key="mod_abducted")
            st.checkbox("Kol destekli / yerçekimi yardımlı", key="mod_supported")
            st.checkbox("Bilek dönmüş veya yana sapmış", key="mod_wrist")
            st.checkbox("İki ayağa dengeli yük veriliyor", value=True, key="mod_bilateral")
            st.checkbox("Statik duruş > 1 dakika", key="mod_static")
            st.checkbox("Tekrarlı hareket > 4/dakika", key="mod_repetitive")
            st.checkbox("Ani büyük değişim / dengesiz taban", key="mod_unstable")

        st.markdown("---")
        st.markdown("**Hakkında**")
        st.markdown(
            "TÜBİTAK Projesi\n\n"
            "Sanayi ortamında çalışanların\n"
            "ergonomik risk analizini\n"
            "yapay zeka ile gerçekleştiren\n"
            "karar destek sistemi."
        )

        st.markdown("---")
        st.markdown("v1.0 | 2026")

    # Page routing
    if "Ana Panel" in page:
        page_home()
    elif "Görsel Analiz" in page:
        page_image_analysis()
    elif "Video Analiz" in page:
        page_video_analysis()
    elif "Webcam" in page:
        page_webcam()
    elif "Geçmiş" in page:
        page_history()


if __name__ == "__main__":
    main()
