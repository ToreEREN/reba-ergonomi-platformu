"""Fast local verification for model artifacts and the REBA core."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def ok(label):
    print(f"[OK] {label}")


def main():
    required = [
        ROOT / "ModelExperiments/extra_trees.joblib",
        ROOT / "ModelExperiments/manifest.json",
        ROOT / "ModelExperiments/leaderboard.csv",
        ROOT / "BestModel/ft_transformer_scaler.pkl",
        ROOT / "yolo11n-pose.pt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Eksik dosyalar:\n" + "\n".join(missing))
    ok("Gerekli model dosyaları mevcut")

    manifest = json.loads(required[1].read_text(encoding="utf-8"))
    model = joblib.load(required[0])
    pred = model.predict(np.zeros((1, 90), dtype=np.float32))
    if pred.shape != (1, len(manifest["targets"])) or not np.isfinite(pred).all():
        raise RuntimeError(f"Geçersiz model çıktısı: {pred.shape}")
    ok(f"Yeni model yüklendi ve {pred.shape[1]} açı üretti")

    board = pd.read_csv(required[2])
    winner = board.sort_values("macro_r2", ascending=False).iloc[0]
    ok(f"Lider: {winner.model}, test macro R2={winner.macro_r2:.3f}")

    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT, check=False,
    )
    if result.returncode:
        raise RuntimeError("REBA testleri başarısız")
    ok("REBA çekirdek testleri geçti")
    print("\nDOĞRULAMA BAŞARILI")


if __name__ == "__main__":
    main()
