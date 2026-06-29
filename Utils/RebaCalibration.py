"""
REBA Kalibrasyon Modülü
========================
FT-Transformer model tahminlerini REBA convention'a çevirir.

Problem: Model "ham açı" tahmin eder (örn: shoulderL_vs_upperarmL_yz_deg = 70°).
Ancak REBA "sapma" (deviation from neutral) bekler.

Bu modül:
1. Model tahminlerinden "kalibre edilmiş sapma" (deviation) değerlerini üretir.
2. Modelin regression-to-mean bias'ını düzeltir.
3. REBA convention'a uygun türetilmiş kolonları oluşturur.

Kalibrasyon değerleri: BestModel/advanced_ft_transformer_predictions_all.csv
üzerindeki analiz sonuçlarına dayanır.
"""

import numpy as np
import pandas as pd


# ============================================================
# KALİBRASYON SABİTLERİ
# ============================================================
# Model'in "düz duruş" (standing) pozisyonu için tipik tahmin değerleri.
# Bunlar, modelin standing pose'larda (thighR_true > 170) ortalama tahmin ettiği değerlerdir.
# standing_neutral[angle_name] = model bu açı için standing'de bu değeri tahmin eder
STANDING_NEUTRAL = {
    "thighL_vs_spine1_yz_deg": 155.0,
    "thighR_vs_spine1_yz_deg": 156.0,
    "kneeL_yz_deg": 161.0,
    "kneeR_yz_deg": 163.0,
    "shoulderL_vs_upperarmL_yz_deg": 64.0,   # Kollar yanda sarkıkken model ~64° verir
    "shoulderR_vs_upperarmR_yz_deg": 76.0,   # Kollar yanda sarkıkken model ~76° verir
    "elbowL_yz_deg": 160.0,
    "elbowR_yz_deg": 161.0,
    "wristL_yz_deg": 161.0,
    "wristR_yz_deg": 157.0,
    "spine1_vs_spine2_yz_deg": 10.5,
}


def compute_reba_deviations(df_pred):
    """
    FT-Transformer tahminlerinden REBA kurallarına uygun 'deviation' kolonları üretir.

    Args:
        df_pred: TARGET_COLS kolonlarını içeren DataFrame (model tahmin çıktısı)

    Returns:
        df_reba: Tüm orijinal kolonlar + türetilmiş deviation kolonları
    """
    df = df_pred.copy()

    # =========================================================
    # TRUNK DEVIATION (Gövde eğilme)
    # Mantık: spine1_vs_spine2 açısı doğrudan gövde eğilmesini gösterir.
    # Ayrıca thigh_vs_spine sapması eklenir (ama kalibre edilmiş).
    # Standing'de spine1_vs_spine2 ~10.5° → bunu baseline olarak çıkar.
    # =========================================================
    spine_deviation = (df["spine1_vs_spine2_yz_deg"] - STANDING_NEUTRAL["spine1_vs_spine2_yz_deg"]).clip(lower=0)

    # Thigh-spine sapması: model'in standing tahmini referans alınır
    thighR_dev = (STANDING_NEUTRAL["thighR_vs_spine1_yz_deg"] - df["thighR_vs_spine1_yz_deg"]).clip(lower=0)
    thighL_dev = (STANDING_NEUTRAL["thighL_vs_spine1_yz_deg"] - df["thighL_vs_spine1_yz_deg"]).clip(lower=0)
    thigh_dev = (thighR_dev + thighL_dev) / 2.0

    # Trunk deviation = spine eğilmesi + thigh katkısı (ağırlıklı)
    df["trunk_deviation"] = spine_deviation + thigh_dev * 0.5

    # =========================================================
    # KNEE DEVIATION (Diz bükülmesi)
    # Standing'de model ~161-163° tahmin eder.
    # Deviation = standing_neutral - predicted (ne kadar büküldü)
    # =========================================================
    df["kneeR_deviation"] = (STANDING_NEUTRAL["kneeR_yz_deg"] - df["kneeR_yz_deg"]).clip(lower=0)
    df["kneeL_deviation"] = (STANDING_NEUTRAL["kneeL_yz_deg"] - df["kneeL_yz_deg"]).clip(lower=0)

    # =========================================================
    # UPPER ARM ELEVATION (Kol kaldırma)
    # Kollar yanda sarkıkken model shoulder_yz ~64-76° verir.
    # Kol kaldırıldıkça bu değer ARTAR (omuz-kol açısı büyür).
    # Elevation = predicted - standing_neutral
    # =========================================================
    df["upperarmL_elevation"] = (df["shoulderL_vs_upperarmL_yz_deg"] - STANDING_NEUTRAL["shoulderL_vs_upperarmL_yz_deg"]).clip(lower=0)
    df["upperarmR_elevation"] = (df["shoulderR_vs_upperarmR_yz_deg"] - STANDING_NEUTRAL["shoulderR_vs_upperarmR_yz_deg"]).clip(lower=0)

    # =========================================================
    # ELBOW DEVIATION (Dirsek bükülmesi)
    # Model standing (düz kol) için ~160° verir.
    # REBA'da dirsek: 60-100° flexion = skor 1, dışı = skor 2
    # Burada deviation = standing'den sapma (düz koldan ne kadar büküldü)
    # =========================================================
    df["elbowL_deviation"] = (STANDING_NEUTRAL["elbowL_yz_deg"] - df["elbowL_yz_deg"]).clip(lower=0)
    df["elbowR_deviation"] = (STANDING_NEUTRAL["elbowR_yz_deg"] - df["elbowR_yz_deg"]).clip(lower=0)

    # =========================================================
    # WRIST DEVIATION (Bilek bükülmesi)
    # Model standing (düz bilek) için ~157-161° verir.
    # Deviation = standing'den sapma
    # =========================================================
    df["wristL_deviation"] = (STANDING_NEUTRAL["wristL_yz_deg"] - df["wristL_yz_deg"]).clip(lower=0)
    df["wristR_deviation"] = (STANDING_NEUTRAL["wristR_yz_deg"] - df["wristR_yz_deg"]).clip(lower=0)

    # =========================================================
    # Eski convention uyumluluğu (bazı kurallar hala bunları kullanabilir)
    # =========================================================
    df["thighR_vs_spine1_yz_from180_abs"] = (180 - df["thighR_vs_spine1_yz_deg"]).abs()
    df["thighL_vs_spine1_yz_from180_abs"] = (180 - df["thighL_vs_spine1_yz_deg"]).abs()
    df["kneeR_yz_deg180_abs"] = (180 - df["kneeR_yz_deg"]).abs()
    df["kneeL_yz_deg180_abs"] = (180 - df["kneeL_yz_deg"]).abs()
    df["kneeR_xz_deg180_abs"] = (180 - df["kneeR_xz_deg"]).abs() if "kneeR_xz_deg" in df.columns else 0
    df["kneeL_xz_deg180_abs"] = (180 - df["kneeL_xz_deg"]).abs() if "kneeL_xz_deg" in df.columns else 0
    df["upperarmL_vs_shoulderL_yz_from180_abs"] = (180 - df["shoulderL_vs_upperarmL_yz_deg"]).abs()
    df["upperarmR_vs_shoulderR_yz_from180_abs"] = (180 - df["shoulderR_vs_upperarmR_yz_deg"]).abs()

    return df
