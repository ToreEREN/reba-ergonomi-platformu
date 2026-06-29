"""
REBA Video Pipeline - CSV + Annotated Video + Grafik
=====================================================
TUBİTAK Projesi - Sanayi İçin Derin Öğrenme Tabanlı Ergonomik Karar Destek Sistemi

Hem video dosyası hem webcam desteği ile:
  1. YOLO Pose ile kare kare keypoint çıkarımı
  2. FT-Transformer ile gizli açı tahmini
  3. REBA skor hesaplaması
  4. Annotated video çıktısı (iskelet + skor overlay)
  5. CSV çıktısı (kare kare tüm veriler)
  6. Zaman serisi grafikleri

Kullanım:
  python REBA_Video_Pipeline.py --source video.mp4
  python REBA_Video_Pipeline.py --source 0          (webcam)
  python REBA_Video_Pipeline.py --source video.mp4 --output_dir Results --no-display
"""

import os
import sys
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Utils.Functions import add_pose_angles
from Utils.Models import (
    FeatureTokenizer, FeaturePositionalEmbedding,
    AttentionPooling, ResidualMLPBlock,
    build_advanced_ft_transformer_regression,
    predict_advanced_multioutput
)
from Utils.Reba import (
    rules, compute_reba_scores,
    get_reba_tableA_score, get_reba_tableB_score, get_reba_tableC_score,
    get_force_load_score, get_coupling_score, get_activity_score,
    classify_reba_risk, clamp
)
from Utils.RebaCalibration import compute_reba_deviations

# ============================================================
# KONFIGÜRASYON
# ============================================================
YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11n-pose.pt")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BestModel", "best_ft_transformer_multioutput_compatible.h5")
SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BestModel", "ft_transformer_scaler.pkl")
META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BestModel", "advanced_ft_transformer_meta.json")

YOLO_CONF = 0.25

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

# COCO iskelet bağlantıları (YOLO 17 keypoint)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # yüz
    (5, 6),                                 # omuzlar arası
    (5, 7), (7, 9),                         # sol kol
    (6, 8), (8, 10),                        # sağ kol
    (5, 11), (6, 12),                       # gövde
    (11, 12),                               # kalça
    (11, 13), (13, 15),                     # sol bacak
    (12, 14), (14, 16),                     # sağ bacak
]

RISK_COLORS = {
    "Negligible risk": (0, 200, 0),
    "Low risk": (0, 255, 100),
    "Medium risk": (0, 200, 255),
    "High risk": (0, 100, 255),
    "Very high risk": (0, 0, 255),
}


# ============================================================
# MODEL YÜKLEME
# ============================================================
def load_models():
    """YOLO ve FT-Transformer modellerini yükler."""
    print("[INFO] Modeller yukleniyor...")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"  YOLO yuklendi: {YOLO_MODEL_PATH}")

    scaler = joblib.load(SCALER_PATH)
    print(f"  Scaler yuklendi: {SCALER_PATH}")

    with open(META_PATH, 'r') as f:
        meta_info = json.load(f)

    ft_model = build_advanced_ft_transformer_regression(
        n_features=meta_info['input_dim'],
        n_targets=meta_info['n_targets'],
        d_token=96,
        n_blocks=6,
        n_heads=8,
        dropout=0.20,
        ff_mult=4,
        shared_mlp_units=(256, 128),
        per_target_hidden=64,
        use_uncertainty=meta_info['use_uncertainty']
    )

    _dummy = np.zeros((1, meta_info['input_dim']), dtype='float32')
    _ = ft_model(_dummy)
    ft_model.load_weights(MODEL_PATH)
    print(f"  FT-Transformer yuklendi: {MODEL_PATH}")
    print(f"  Input dim: {meta_info['input_dim']}, Output dim: {meta_info['n_targets']}")

    return yolo_model, ft_model, scaler, meta_info


# ============================================================
# KARE BAZLI ANALİZ
# ============================================================
def process_frame(frame, yolo_model, ft_model, scaler):
    """
    Tek bir kareyi analiz eder.

    Returns:
        result_dict: Analiz sonuçları (None ise kişi tespit edilemedi)
        keypoints: Tespit edilen keypoint'ler (çizim için)
    """
    results = yolo_model.predict(source=frame, conf=YOLO_CONF, save=False, verbose=False)

    if not results or results[0].keypoints is None:
        return None, None

    res = results[0]
    kp_data = res.keypoints.data.cpu().numpy()

    if kp_data.shape[0] == 0:
        return None, None

    # İlk kişiyi al
    pts = kp_data[0]
    keypoints = [[float(kp[0]), float(kp[1]), float(kp[2])] for kp in pts]

    # DataFrame oluştur
    input_row = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y, z = keypoints[i] if i < len(keypoints) else (0.0, 0.0, 0.0)
        input_row[f"{name}_x"] = x
        input_row[f"{name}_y"] = y
        input_row[f"{name}_z"] = z

    df_kp = pd.DataFrame([input_row])
    df_kp.columns = BASE_INPUT_COLS

    # Açı hesaplama
    df_input = add_pose_angles(df_kp, add_planar_angles=True)
    input_cols = df_input.columns.tolist()

    # FT-Transformer tahmin
    X_input = df_input.values.astype("float32")
    X_scaled = scaler.transform(X_input)
    y_pred = predict_advanced_multioutput(ft_model, X_scaled, use_uncertainty=False, batch_size=1)

    df_pred = pd.DataFrame(y_pred, columns=TARGET_COLS)

    # REBA hesaplama - kalibre edilmiş deviation'lar
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

    tA = get_reba_tableA_score(neck=clamp(s1, 1, 3), trunk=clamp(s2, 1, 5), legs=clamp(s3, 1, 4))
    scoreA = tA + get_force_load_score(0, False)

    tB = get_reba_tableB_score(upper_arm=clamp(s4, 1, 6), lower_arm=clamp(s5, 1, 2), wrist=clamp(s6, 1, 3))
    scoreB = tB + get_coupling_score(0)

    tC = get_reba_tableC_score(clamp(scoreA, 1, 12), clamp(scoreB, 1, 12))
    final_reba = tC + get_activity_score(False, False, False)
    risk_level = classify_reba_risk(final_reba)

    result_dict = {
        "step1_neck": s1,
        "step2_trunk": s2,
        "step3_legs": s3,
        "step4_upper_arm": s4,
        "step5_elbow": s5,
        "step6_wrist": s6,
        "score_a": int(scoreA),
        "score_b": int(scoreB),
        "table_c": int(tC),
        "final_reba": int(final_reba),
        "risk_level": risk_level,
        "predicted_angles": {col: float(df_pred[col].iloc[0]) for col in TARGET_COLS[:19]},
    }

    return result_dict, keypoints


# ============================================================
# GÖRSEL ÇİZİM FONKSİYONLARI
# ============================================================
def draw_skeleton(frame, keypoints, conf_threshold=0.3):
    """Keypoint'leri ve iskelet bağlantılarını çizer."""
    h, w = frame.shape[:2]

    for (i, j) in SKELETON_CONNECTIONS:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]

        if c1 < conf_threshold or c2 < conf_threshold:
            continue

        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    for i, (x, y, c) in enumerate(keypoints):
        if c < conf_threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)

    return frame


def draw_reba_overlay(frame, result_dict):
    """REBA skor bilgilerini frame üzerine overlay olarak çizer."""
    if result_dict is None:
        cv2.putText(frame, "Kisi tespit edilemedi", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    h, w = frame.shape[:2]
    final_reba = result_dict["final_reba"]
    risk_level = result_dict["risk_level"]
    risk_color = RISK_COLORS.get(risk_level, (255, 255, 255))

    # Arka plan kutusu
    overlay = frame.copy()
    box_h = 220
    box_w = 320
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Başlık
    y_offset = 35
    cv2.putText(frame, "REBA ANALIZI", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Final REBA skoru (büyük)
    y_offset += 40
    cv2.putText(frame, f"SKOR: {final_reba}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, risk_color, 3)

    # Risk seviyesi
    y_offset += 30
    cv2.putText(frame, f"Risk: {risk_level}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)

    # Step skorları
    y_offset += 30
    cv2.putText(frame, f"Boyun:{result_dict['step1_neck']} Govde:{result_dict['step2_trunk']} Bacak:{result_dict['step3_legs']}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    y_offset += 22
    cv2.putText(frame, f"UstKol:{result_dict['step4_upper_arm']} Dirsek:{result_dict['step5_elbow']} Bilek:{result_dict['step6_wrist']}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    y_offset += 22
    cv2.putText(frame, f"Score A:{result_dict['score_a']}  Score B:{result_dict['score_b']}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Risk seviyesi renk barı (alt kısım)
    bar_y = 10 + box_h - 25
    bar_x = 20
    bar_width = box_w - 20
    segment_w = bar_width // 5

    risk_segments = [
        ("1", (0, 200, 0)),
        ("2-3", (0, 255, 100)),
        ("4-7", (0, 200, 255)),
        ("8-10", (0, 100, 255)),
        ("11+", (0, 0, 255)),
    ]

    for idx, (label, color) in enumerate(risk_segments):
        x1 = bar_x + idx * segment_w
        x2 = x1 + segment_w
        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + 15), color, -1)
        cv2.putText(frame, label, (x1 + 2, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    # Skor göstergesi (üçgen)
    marker_x = bar_x + int((min(final_reba, 15) / 15.0) * bar_width)
    pts_tri = np.array([
        [marker_x, bar_y - 2],
        [marker_x - 5, bar_y - 10],
        [marker_x + 5, bar_y - 10]
    ], np.int32)
    cv2.fillPoly(frame, [pts_tri], (255, 255, 255))

    return frame


# ============================================================
# ANA PIPELINE
# ============================================================
def run_pipeline(source, output_dir="Output_Video", display=True, skip_frames=0,
                 max_frames=None, save_video=True):
    """
    Ana video analiz pipeline'ı.

    Args:
        source: Video dosya yolu veya kamera indeksi (0, 1, ...)
        output_dir: Çıktı klasörü
        display: Gerçek zamanlı görüntüleme
        skip_frames: Her N karede bir analiz yap (0=her kare)
        max_frames: Maksimum işlenecek kare sayısı (None=tümü)
        save_video: Annotated video kaydet
    """
    os.makedirs(output_dir, exist_ok=True)

    # Kaynak belirleme
    if str(source).isdigit():
        source = int(source)
        source_name = f"webcam_{source}"
        is_webcam = True
    else:
        source_name = os.path.splitext(os.path.basename(source))[0]
        is_webcam = False

    print(f"\n{'='*60}")
    print(f"  REBA VIDEO PIPELINE")
    print(f"  Kaynak: {'Webcam ' + str(source) if is_webcam else source}")
    print(f"  Cikti:  {output_dir}")
    print(f"{'='*60}\n")

    # Modelleri yükle
    yolo_model, ft_model, scaler, meta_info = load_models()

    # Video aç
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[HATA] Video acilamadi: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1

    print(f"[INFO] Video: {frame_w}x{frame_h} @ {fps:.1f} FPS")
    if total_frames > 0:
        print(f"[INFO] Toplam kare: {total_frames}")

    # Video writer
    video_writer = None
    if save_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_out_path = os.path.join(output_dir, f"{source_name}_reba_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_w, frame_h))
        print(f"[INFO] Video cikti: {video_out_path}")

    # CSV verileri
    csv_rows = []
    frame_idx = 0
    processed_count = 0
    start_time = time.time()
    last_result = None

    print(f"\n[INFO] Analiz basliyor... (Cikmak icin 'q' tuslayın)")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    continue
                break

            frame_idx += 1

            if max_frames and frame_idx > max_frames:
                break

            # Frame atlama (performans için)
            should_analyze = (skip_frames == 0) or (frame_idx % (skip_frames + 1) == 0)

            if should_analyze:
                result_dict, keypoints = process_frame(frame, yolo_model, ft_model, scaler)
                last_result = result_dict
                processed_count += 1

                # CSV satırı
                row = {
                    "frame": frame_idx,
                    "timestamp_sec": frame_idx / fps,
                }

                if result_dict:
                    row.update({
                        "final_reba": result_dict["final_reba"],
                        "risk_level": result_dict["risk_level"],
                        "step1_neck": result_dict["step1_neck"],
                        "step2_trunk": result_dict["step2_trunk"],
                        "step3_legs": result_dict["step3_legs"],
                        "step4_upper_arm": result_dict["step4_upper_arm"],
                        "step5_elbow": result_dict["step5_elbow"],
                        "step6_wrist": result_dict["step6_wrist"],
                        "score_a": result_dict["score_a"],
                        "score_b": result_dict["score_b"],
                    })
                    row.update(result_dict.get("predicted_angles", {}))
                else:
                    row["final_reba"] = None
                    row["risk_level"] = "Tespit edilemedi"

                csv_rows.append(row)

                # İlerleme göster
                if processed_count % 30 == 0:
                    elapsed = time.time() - start_time
                    proc_fps = processed_count / elapsed if elapsed > 0 else 0
                    if total_frames > 0:
                        progress = (frame_idx / total_frames) * 100
                        print(f"  Kare {frame_idx}/{total_frames} ({progress:.1f}%) | "
                              f"REBA: {result_dict['final_reba'] if result_dict else '-'} | "
                              f"{proc_fps:.1f} FPS")
                    else:
                        print(f"  Kare {frame_idx} | "
                              f"REBA: {result_dict['final_reba'] if result_dict else '-'} | "
                              f"{proc_fps:.1f} FPS")
            else:
                result_dict = last_result
                keypoints = None

            # Annotated frame oluştur
            annotated = frame.copy()
            if keypoints:
                annotated = draw_skeleton(annotated, keypoints)
            annotated = draw_reba_overlay(annotated, result_dict)

            # FPS göster
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0
            cv2.putText(annotated, f"FPS: {current_fps:.1f}", (frame_w - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Kare: {frame_idx}", (frame_w - 130, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Video kaydet
            if video_writer:
                video_writer.write(annotated)

            # Ekranda göster
            if display:
                cv2.imshow("REBA Video Analizi", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[INFO] Kullanici tarafindan durduruldu.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Klavye ile durduruldu.")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ANALIZ TAMAMLANDI")
    print(f"  Toplam sure: {total_time:.1f} sn")
    print(f"  Islenen kare: {processed_count}")
    print(f"  Ortalama FPS: {processed_count/total_time:.1f}")
    print(f"{'='*60}")

    # CSV kaydet
    if csv_rows:
        df_csv = pd.DataFrame(csv_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"{source_name}_reba_results_{timestamp}.csv")
        df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n[CIKTI] CSV kaydedildi: {csv_path}")

        # Grafikleri oluştur
        generate_charts(df_csv, output_dir, source_name, timestamp)

        # Özet JSON
        save_summary(df_csv, output_dir, source_name, timestamp, total_time, fps)

    return csv_rows


# ============================================================
# GRAFİK OLUŞTURMA
# ============================================================
def generate_charts(df, output_dir, source_name, timestamp):
    """Analiz sonuçlarından grafikler oluşturur."""
    print("\n[INFO] Grafikler olusturuluyor...")

    df_valid = df[df["final_reba"].notna()].copy()
    if df_valid.empty:
        print("[UYARI] Gecerli veri yok, grafik olusturulamadi.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"REBA Ergonomik Risk Analizi - {source_name}", fontsize=14, fontweight='bold')

    # 1. REBA Skoru Zaman Serisi
    ax = axes[0, 0]
    time_sec = df_valid["timestamp_sec"].values
    reba_scores = df_valid["final_reba"].values

    ax.plot(time_sec, reba_scores, 'b-', linewidth=1, alpha=0.7)
    ax.axhspan(0, 1, alpha=0.1, color='green', label='Negligible (1)')
    ax.axhspan(1, 3, alpha=0.1, color='limegreen', label='Low (2-3)')
    ax.axhspan(3, 7, alpha=0.1, color='orange', label='Medium (4-7)')
    ax.axhspan(7, 10, alpha=0.1, color='orangered', label='High (8-10)')
    ax.axhspan(10, 15, alpha=0.1, color='red', label='Very High (11+)')

    # Hareketli ortalama
    if len(reba_scores) > 10:
        window = min(30, len(reba_scores) // 3)
        rolling_avg = pd.Series(reba_scores).rolling(window=window, min_periods=1).mean()
        ax.plot(time_sec, rolling_avg, 'r-', linewidth=2, label=f'Hareketli Ort. ({window} kare)')

    ax.set_xlabel("Zaman (sn)")
    ax.set_ylabel("REBA Skoru")
    ax.set_title("REBA Skoru - Zaman Serisi")
    ax.set_ylim(0, max(15, reba_scores.max() + 1))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Risk Dağılımı (Pasta Grafik)
    ax = axes[0, 1]
    risk_counts = df_valid["risk_level"].value_counts()
    risk_order = ["Negligible risk", "Low risk", "Medium risk", "High risk", "Very high risk"]
    risk_colors_plt = ['#2ecc71', '#a3d977', '#f39c12', '#e67e22', '#e74c3c']

    ordered_counts = []
    ordered_labels = []
    ordered_colors = []
    for r, c in zip(risk_order, risk_colors_plt):
        if r in risk_counts.index:
            ordered_counts.append(risk_counts[r])
            ordered_labels.append(r.replace(" risk", ""))
            ordered_colors.append(c)

    if ordered_counts:
        wedges, texts, autotexts = ax.pie(
            ordered_counts, labels=ordered_labels, colors=ordered_colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9}
        )
        ax.set_title("Risk Seviyesi Dagilimi")

    # 3. Step Skorları Box Plot
    ax = axes[1, 0]
    step_cols = ["step1_neck", "step2_trunk", "step3_legs",
                 "step4_upper_arm", "step5_elbow", "step6_wrist"]
    step_labels = ["Boyun", "Govde", "Bacak", "Ust Kol", "Dirsek", "Bilek"]

    step_data = []
    valid_labels = []
    for col, label in zip(step_cols, step_labels):
        if col in df_valid.columns:
            data = df_valid[col].dropna().values
            if len(data) > 0:
                step_data.append(data)
                valid_labels.append(label)

    if step_data:
        bp = ax.boxplot(step_data, labels=valid_labels, patch_artist=True)
        colors_bp = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        for patch, color in zip(bp['boxes'], colors_bp[:len(step_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_ylabel("Skor")
    ax.set_title("Step Skorlari Dagilimi")
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Skor A ve Skor B Zaman Serisi
    ax = axes[1, 1]
    if "score_a" in df_valid.columns and "score_b" in df_valid.columns:
        ax.plot(time_sec, df_valid["score_a"].values, 'b-', linewidth=1.5, label='Score A (Govde)', alpha=0.8)
        ax.plot(time_sec, df_valid["score_b"].values, 'r-', linewidth=1.5, label='Score B (Kol)', alpha=0.8)
        ax.set_xlabel("Zaman (sn)")
        ax.set_ylabel("Skor")
        ax.set_title("Score A & Score B - Zaman Serisi")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, f"{source_name}_reba_charts_{timestamp}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[CIKTI] Grafik kaydedildi: {chart_path}")

    # Ek grafik: Açı değişimleri
    generate_angle_chart(df_valid, output_dir, source_name, timestamp)


def generate_angle_chart(df, output_dir, source_name, timestamp):
    """Tahmin edilen açıların zaman serisi grafiği."""
    angle_cols = [c for c in df.columns if c.endswith("_deg") and c in TARGET_COLS[:19]]

    if not angle_cols or len(df) < 2:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Tahmin Edilen Vucut Acilari - Zaman Serisi", fontsize=12, fontweight='bold')
    time_sec = df["timestamp_sec"].values

    # Üst gövde açıları
    upper_cols = [c for c in angle_cols if "spine" in c or "neck" in c or "head" in c]
    ax = axes[0]
    for col in upper_cols:
        if col in df.columns:
            ax.plot(time_sec, df[col].values, linewidth=1, label=col.replace("_deg", ""), alpha=0.8)
    ax.set_ylabel("Derece")
    ax.set_title("Ust Govde / Boyun Acilari")
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # Kol açıları
    arm_cols = [c for c in angle_cols if "elbow" in c or "wrist" in c or "shoulder" in c or "upperarm" in c]
    ax = axes[1]
    for col in arm_cols:
        if col in df.columns:
            ax.plot(time_sec, df[col].values, linewidth=1, label=col.replace("_deg", ""), alpha=0.8)
    ax.set_ylabel("Derece")
    ax.set_title("Kol / Dirsek / Bilek Acilari")
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bacak açıları
    leg_cols = [c for c in angle_cols if "knee" in c or "thigh" in c]
    ax = axes[2]
    for col in leg_cols:
        if col in df.columns:
            ax.plot(time_sec, df[col].values, linewidth=1, label=col.replace("_deg", ""), alpha=0.8)
    ax.set_xlabel("Zaman (sn)")
    ax.set_ylabel("Derece")
    ax.set_title("Bacak / Diz Acilari")
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    angle_chart_path = os.path.join(output_dir, f"{source_name}_angle_charts_{timestamp}.png")
    plt.savefig(angle_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[CIKTI] Aci grafigi kaydedildi: {angle_chart_path}")


# ============================================================
# ÖZET KAYDETME
# ============================================================
def save_summary(df, output_dir, source_name, timestamp, total_time, fps):
    """Analiz özet bilgilerini JSON olarak kaydeder."""
    df_valid = df[df["final_reba"].notna()]

    if df_valid.empty:
        return

    reba_scores = df_valid["final_reba"].values

    summary = {
        "source": source_name,
        "analysis_date": datetime.now().isoformat(),
        "total_frames_analyzed": len(df_valid),
        "total_duration_sec": float(total_time),
        "video_fps": float(fps),
        "reba_statistics": {
            "mean": float(np.mean(reba_scores)),
            "median": float(np.median(reba_scores)),
            "std": float(np.std(reba_scores)),
            "min": int(np.min(reba_scores)),
            "max": int(np.max(reba_scores)),
            "percentile_25": float(np.percentile(reba_scores, 25)),
            "percentile_75": float(np.percentile(reba_scores, 75)),
        },
        "risk_distribution": df_valid["risk_level"].value_counts().to_dict(),
        "risk_percentage": {
            k: f"{v/len(df_valid)*100:.1f}%"
            for k, v in df_valid["risk_level"].value_counts().items()
        },
        "step_averages": {
            "neck": float(df_valid["step1_neck"].mean()) if "step1_neck" in df_valid.columns else None,
            "trunk": float(df_valid["step2_trunk"].mean()) if "step2_trunk" in df_valid.columns else None,
            "legs": float(df_valid["step3_legs"].mean()) if "step3_legs" in df_valid.columns else None,
            "upper_arm": float(df_valid["step4_upper_arm"].mean()) if "step4_upper_arm" in df_valid.columns else None,
            "elbow": float(df_valid["step5_elbow"].mean()) if "step5_elbow" in df_valid.columns else None,
            "wrist": float(df_valid["step6_wrist"].mean()) if "step6_wrist" in df_valid.columns else None,
        },
        "high_risk_frames": int((reba_scores >= 8).sum()),
        "high_risk_percentage": f"{(reba_scores >= 8).sum() / len(reba_scores) * 100:.1f}%",
    }

    summary_path = os.path.join(output_dir, f"{source_name}_summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[CIKTI] Ozet JSON kaydedildi: {summary_path}")

    # Konsol özeti
    print(f"\n{'='*60}")
    print("  ANALIZ OZETI")
    print(f"{'='*60}")
    print(f"  Ortalama REBA Skoru: {summary['reba_statistics']['mean']:.1f}")
    print(f"  Medyan REBA Skoru:   {summary['reba_statistics']['median']:.1f}")
    print(f"  Min / Max:           {summary['reba_statistics']['min']} / {summary['reba_statistics']['max']}")
    print(f"  Yuksek Risk Orani:   {summary['high_risk_percentage']}")
    print(f"\n  Risk Dagilimi:")
    for level, count in sorted(summary["risk_distribution"].items()):
        pct = summary["risk_percentage"][level]
        print(f"    {level}: {count} kare ({pct})")
    print(f"{'='*60}\n")


# ============================================================
# KOMUT SATIRI ARAYÜZÜ
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="REBA Video Pipeline - Ergonomik Risk Analizi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python REBA_Video_Pipeline.py --source video.mp4
  python REBA_Video_Pipeline.py --source 0                    (webcam)
  python REBA_Video_Pipeline.py --source video.mp4 --skip 2   (her 3 karede 1)
  python REBA_Video_Pipeline.py --source 0 --no-display       (headless)
  python REBA_Video_Pipeline.py --source video.mp4 --max-frames 500
        """
    )

    parser.add_argument("--source", type=str, default="0",
                        help="Video dosya yolu veya kamera indeksi (varsayilan: 0 = webcam)")
    parser.add_argument("--output-dir", type=str, default="Output_Video",
                        help="Cikti klasoru (varsayilan: Output_Video)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Kare atlama sayisi (0=her kare, 2=her 3. kare)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maksimum islenecek kare sayisi")
    parser.add_argument("--no-display", action="store_true",
                        help="Gercek zamanli gosterimi kapat")
    parser.add_argument("--no-video", action="store_true",
                        help="Video kaydetmeyi kapat")

    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        output_dir=args.output_dir,
        display=not args.no_display,
        skip_frames=args.skip,
        max_frames=args.max_frames,
        save_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
