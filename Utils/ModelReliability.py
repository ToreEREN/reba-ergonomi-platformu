"""Model reliability, uncertainty and artifact integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors


REBA_THRESHOLDS_DEG = np.asarray([15, 20, 30, 45, 60, 90, 100], dtype=float)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_with_tree_uncertainty(model, X):
    """Return ensemble mean and epistemic spread across Extra Trees members."""
    X = np.asarray(X, dtype=np.float32)
    per_tree = np.stack([tree.predict(X) for tree in model.estimators_], axis=0)
    return per_tree.mean(axis=0), per_tree.std(axis=0, ddof=1)


def uncertainty_summary(prediction, std, target_names, high_std_deg=15.0):
    prediction = np.asarray(prediction)[0]
    std = np.asarray(std)[0]
    distance = np.min(np.abs(prediction[:, None] - REBA_THRESHOLDS_DEG[None, :]), axis=1)
    near_threshold = distance <= np.maximum(3.0, std)
    high_uncertainty = std >= high_std_deg
    flagged = near_threshold | high_uncertainty
    return {
        "mean_tree_std_deg": float(std.mean()),
        "max_tree_std_deg": float(std.max()),
        "flagged_targets": [str(target_names[i]) for i in np.flatnonzero(flagged)],
        "near_threshold_targets": [str(target_names[i]) for i in np.flatnonzero(near_threshold)],
        "high_uncertainty_targets": [str(target_names[i]) for i in np.flatnonzero(high_uncertainty)],
        "method": "uncalibrated standard deviation across Extra Trees members",
    }


def nearest_neighbor_leakage_audit(X_train, X_test, threshold=0.05):
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(X_train)
    distances, _ = nn.kneighbors(X_test)
    distances = distances.ravel()
    return {
        "threshold": float(threshold),
        "near_duplicate_ratio": float((distances < threshold).mean()),
        "min_distance": float(distances.min()),
        "median_distance": float(np.median(distances)),
        "p05_distance": float(np.percentile(distances, 5)),
    }


def paired_model_comparison(y_true, y_new, y_old, n_boot=2000, seed=42):
    """Paired Wilcoxon and row-bootstrap CIs for model improvement."""
    y_true, y_new, y_old = map(np.asarray, (y_true, y_new, y_old))
    err_new = np.abs(y_true - y_new).mean(axis=1)
    err_old = np.abs(y_true - y_old).mean(axis=1)
    stat, p_value = wilcoxon(err_old, err_new, alternative="greater")
    rng = np.random.default_rng(seed)
    r2_new, r2_gain, mae_new, mae_gain = [], [], [], []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y_true), len(y_true))
        truth, new, old = y_true[idx], y_new[idx], y_old[idx]
        rn = r2_score(truth, new, multioutput="uniform_average")
        ro = r2_score(truth, old, multioutput="uniform_average")
        mn, mo = mean_absolute_error(truth, new), mean_absolute_error(truth, old)
        r2_new.append(rn); r2_gain.append(rn - ro)
        mae_new.append(mn); mae_gain.append(mo - mn)
    ci = lambda values: [float(x) for x in np.percentile(values, [2.5, 50, 97.5])]
    return {
        "wilcoxon_statistic": float(stat),
        "wilcoxon_one_sided_p": float(p_value),
        "r2_new_ci95": ci(r2_new),
        "r2_gain_ci95": ci(r2_gain),
        "mae_new_ci95_deg": ci(mae_new),
        "mae_improvement_ci95_deg": ci(mae_gain),
    }


def target_bias_table(y_true, y_pred, target_names):
    residual = np.asarray(y_true) - np.asarray(y_pred)
    return pd.DataFrame({
        "target": target_names,
        "mean_signed_error_deg": residual.mean(axis=0),
        "median_signed_error_deg": np.median(residual, axis=0),
        "underestimation_ratio": (residual > 0).mean(axis=0),
    })


def validate_angle_range(values, low=0.0, high=180.0):
    values = np.asarray(values)
    return {
        "valid": bool(np.isfinite(values).all() and (values >= low).all() and (values <= high).all()),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "expected_range": [float(low), float(high)],
    }
