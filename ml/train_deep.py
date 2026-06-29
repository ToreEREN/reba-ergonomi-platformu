"""Train a regularized residual MLP on the leakage-free archived split."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_benchmarks import ANGLE_TARGETS, load_archives


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class AngleMLP(nn.Module):
    def __init__(self, n_in=90, n_out=19, width=256, blocks=4, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, width), nn.GELU(),
            *[ResidualBlock(width, dropout) for _ in range(blocks)],
            nn.LayerNorm(width), nn.Linear(width, n_out),
        )

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", type=Path, default=Path("BestModel/advanced_ft_transformer_predictions_all.csv"))
    p.add_argument("--test", type=Path, default=Path("BestModel/advanced_ft_transformer_predictions_test.csv"))
    p.add_argument("--output", type=Path, default=Path("ModelExperiments/deep_residual.pt"))
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    train, test, features, targets = load_archives(args.all, args.test)
    X = train[features].to_numpy(np.float32)
    y = train[targets].to_numpy(np.float32)
    X_test = test[features].to_numpy(np.float32)
    y_test = test[targets].to_numpy(np.float32)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.18, random_state=args.seed)
    y_mean, y_std = y_tr.mean(0), np.maximum(y_tr.std(0), 1e-6)
    y_tr_n, y_val_n = (y_tr-y_mean)/y_std, (y_val-y_mean)/y_std

    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_n)),
                        batch_size=128, shuffle=True)
    model = AngleMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=8)
    loss_fn = nn.SmoothL1Loss(beta=.5)
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val_n)
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            # Feature-space augmentation is deliberately train-only.
            xb = xb + torch.randn_like(xb) * 0.01
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        model.eval()
        with torch.no_grad(): val_loss = loss_fn(model(Xv), yv).item()
        schedule.step(val_loss)
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, stale = val_loss, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): pred = model(torch.from_numpy(X_test)).numpy() * y_std + y_mean
    per_r2 = [r2_score(y_test[:, i], pred[:, i]) for i in range(len(ANGLE_TARGETS))]
    report = {
        "model": "residual_mlp", "epochs_completed": epoch + 1,
        "macro_r2": float(np.mean(per_r2)), "median_r2": float(np.median(per_r2)),
        "macro_mae": float(mean_absolute_error(y_test, pred)),
        "per_target_r2": dict(zip(ANGLE_TARGETS, map(float, per_r2))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "y_mean": y_mean, "y_std": y_std,
                "targets": ANGLE_TARGETS, "report": report}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
