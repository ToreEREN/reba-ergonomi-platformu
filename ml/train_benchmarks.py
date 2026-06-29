"""Leakage-aware benchmark training for the REBA angle estimator.

The archived prediction CSV contains the scaled 90-dimensional model input,
ground-truth targets and old model predictions.  This script reconstructs a
clean train/test experiment, removes every held-out row from the training
pool, applies conservative train-only augmentation and compares regressors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ANGLE_TARGETS = [
    "elbowL_yz_deg", "elbowR_yz_deg", "head_vs_neck_xz_deg",
    "head_vs_neck_yz_deg", "kneeL_xz_deg", "kneeL_yz_deg",
    "kneeR_xz_deg", "kneeR_yz_deg",
    "shoulderL_vs_upperarmL_xz_deg", "shoulderL_vs_upperarmL_yz_deg",
    "shoulderR_vs_upperarmR_xz_deg", "shoulderR_vs_upperarmR_yz_deg",
    "spine1_vs_spine2_yz_deg", "spine2_vs_spine3_yz_deg",
    "spine3_vs_spine4_yz_deg", "thighL_vs_spine1_yz_deg",
    "thighR_vs_spine1_yz_deg", "wristL_yz_deg", "wristR_yz_deg",
]


def load_archives(all_path: Path, test_path: Path):
    all_df = pd.read_csv(all_path)
    test_df = pd.read_csv(test_path)
    feature_cols = [str(i) for i in range(90)]
    target_cols = [f"{name}_true" for name in ANGLE_TARGETS]
    missing = set(feature_cols + target_cols) - set(all_df.columns)
    if missing:
        raise ValueError(f"Archive is missing {len(missing)} required columns")

    # Exact feature hashes are stable because test rows were exported from the
    # same scaled matrix. Remove all matches, including augmented duplicates.
    test_keys = pd.util.hash_pandas_object(test_df[feature_cols], index=False)
    all_keys = pd.util.hash_pandas_object(all_df[feature_cols], index=False)
    train_df = all_df.loc[~all_keys.isin(set(test_keys))].copy()
    if train_df.empty:
        raise ValueError("Leakage removal left no training rows")
    return train_df, test_df, feature_cols, target_cols


def augment(X, y, copies: int, noise: float, seed: int):
    """Small Gaussian perturbations in standardized feature space, train only."""
    if copies <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    xs, ys = [X], [y]
    for _ in range(copies):
        xs.append(X + rng.normal(0.0, noise, X.shape).astype(np.float32))
        ys.append(y.copy())
    return np.concatenate(xs), np.concatenate(ys)


def metrics(y_true, y_pred):
    rows = []
    for idx, name in enumerate(ANGLE_TARGETS):
        rows.append({
            "target": name,
            "mae": mean_absolute_error(y_true[:, idx], y_pred[:, idx]),
            "rmse": mean_squared_error(y_true[:, idx], y_pred[:, idx]) ** 0.5,
            "r2": r2_score(y_true[:, idx], y_pred[:, idx]),
        })
    frame = pd.DataFrame(rows)
    summary = {
        "macro_mae": float(frame.mae.mean()),
        "macro_rmse": float(frame.rmse.mean()),
        "macro_r2": float(frame.r2.mean()),
        "median_r2": float(frame.r2.median()),
        "positive_r2_targets": int((frame.r2 > 0).sum()),
    }
    return frame, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", type=Path, default=Path("BestModel/advanced_ft_transformer_predictions_all.csv"))
    parser.add_argument("--test", type=Path, default=Path("BestModel/advanced_ft_transformer_predictions_test.csv"))
    parser.add_argument("--output", type=Path, default=Path("ModelExperiments"))
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train, test, features, targets = load_archives(args.all, args.test)
    X_train = train[features].to_numpy(np.float32)
    y_train = train[targets].to_numpy(np.float32)
    X_test = test[features].to_numpy(np.float32)
    y_test = test[targets].to_numpy(np.float32)
    X_aug, y_aug = augment(X_train, y_train, args.augment_copies, args.noise, args.seed)

    models = {
        "ridge": Ridge(alpha=10.0),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=args.trees, min_samples_leaf=2, max_features=0.8,
            n_jobs=-1, random_state=args.seed,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=max(100, args.trees // 2), min_samples_leaf=2,
            max_features=0.8, n_jobs=-1, random_state=args.seed,
        ),
    }

    leaderboard = []
    best = None
    for name, model in models.items():
        model.fit(X_aug, y_aug)
        pred = model.predict(X_test)
        detail, summary = metrics(y_test, pred)
        detail.to_csv(args.output / f"{name}_metrics.csv", index=False)
        joblib.dump(model, args.output / f"{name}.joblib", compress=3)
        leaderboard.append({"model": name, **summary})
        if best is None or summary["macro_r2"] > best[0]:
            best = (summary["macro_r2"], name)

    board = pd.DataFrame(leaderboard).sort_values("macro_r2", ascending=False)
    board.to_csv(args.output / "leaderboard.csv", index=False)
    manifest = {
        "best_model": best[1],
        "feature_space": "existing_standardized_90_features",
        "targets": ANGLE_TARGETS,
        "train_rows_after_leakage_removal": len(train),
        "test_rows": len(test),
        "augmented_train_rows": len(X_aug),
        "augmentation": {"copies": args.augment_copies, "gaussian_noise_std": args.noise},
        "warning": "Production inference must use the original feature builder and scaler.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()
