"""
Phase 9a: is there a USDJPY target where NONLINEAR models clearly beat a strong linear baseline?

Phase 8c: on log RV of the next 20 bars a scratch Transformer only matches a tuned linear
volatility-memory model, so pretraining has nothing to improve there. Before training anything
again, screen candidate targets model-free: a target is worth a Transformer / pretraining study only
if some nonlinear model beats the strong linear baseline by a margin that survives a paired
day-block bootstrap.

Input: the 60 log returns r_{t-59..t} (no time-of-day -- a sequence model would not see it either).
Common window set: rows t-60 .. t+240 gap-free, so every target is defined on the same windows.

Targets (all strictly after t):
  rv20    log RMS(r_{t+1..t+20})                       (Phase 8 reference)
  rv60    log RMS(r_{t+1..t+60})
  rv240   log RMS(r_{t+1..t+240})
  dvol20  log RMS(next 20) - log RMS(last 20)          (volatility change)
  jump20  1[max |r_{t+1..t+20}| > K_JUMP * RMS(last 60)]   (binary)

Features (shared): log RMS over the last 1/2/5/10/20/40/60 returns, the 60 log|r| values, signed
return sums over 5/20/60 -- i.e. the Phase 8c strong linear baseline's inputs. Models:
  linear  : ridge (regression) / L2 logistic (jump), standardized features
  hgb     : HistGradientBoosting (nonlinear interactions on the same features)
  mlp     : 2-layer MLP on the same features + the 60 z-scored raw returns
  knn     : k=100 on the standardized multi-scale RMS + signed sums (low-dimensional state)

Split: whole days assigned 80/20 train/test (seeded), windows within PURGE of a boundary dropped.
Scores on a fixed test subsample shared by all models. Regression: MSE and R^2; jump: log loss and
AUC. For each nonlinear model: paired day-block bootstrap (B=500) 95% CI of (linear - model) loss.

Usage: python nonlinear_headroom_screen.py   (writes nonlinear_headroom_screen.json)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_representations import _artifact_store, _dataset_artifact_path

OBS = 60
MAX_H = 240
PURGE = OBS + MAX_H
K_JUMP = 4.0
N_TRAIN = 300_000
N_TEST = 100_000
N_TRAIN_KNN = 100_000
B = 500
SEED = 0
SCALES = (1, 2, 5, 10, 20, 40, 60)


def _rms(a: np.ndarray) -> np.ndarray:
    return np.sqrt((a ** 2).mean(axis=1))


def build(df: pd.DataFrame):
    close = df["close"].to_numpy(dtype=np.float64)
    r = np.diff(np.log(close))
    gap = df.index.to_series().diff().dt.total_seconds().ne(60).to_numpy()[1:]
    bad = np.concatenate([[0], np.cumsum(gap)])
    t = np.arange(OBS - 1, len(r) - MAX_H)
    t = t[(bad[t + MAX_H + 1] - bad[t - OBS + 1]) == 0]
    win = np.lib.stride_tricks.sliding_window_view(r, OBS)
    fut = np.lib.stride_tricks.sliding_window_view(r, MAX_H)
    X = win[t - OBS + 1]                     # [n, 60] input returns
    F = fut[t + 1]                           # [n, 240] future returns
    eps = 1e-7
    past20, past60 = _rms(X[:, -20:]) + eps, _rms(X) + eps
    targets = {
        "rv20": np.log(_rms(F[:, :20]) + eps),
        "rv60": np.log(_rms(F[:, :60]) + eps),
        "rv240": np.log(_rms(F) + eps),
        "dvol20": np.log(_rms(F[:, :20]) + eps) - np.log(past20),
        "jump20": (np.abs(F[:, :20]).max(axis=1) > K_JUMP * past60).astype(np.int8),
    }
    feats = np.c_[
        np.log(np.c_[[_rms(X[:, -s:]) for s in SCALES]].T + eps),
        np.log(np.abs(X) + 1e-6),
        X[:, -5:].sum(1), X[:, -20:].sum(1), X.sum(1),
    ]
    day = np.asarray(df.index[t + 1].normalize().asi8)
    return X, feats, targets, day


def blocked_split(day: np.ndarray):
    days = np.unique(day)
    rng = np.random.default_rng(SEED)
    test_days = rng.choice(days, size=int(len(days) * 0.2), replace=False)
    is_test = np.isin(day, test_days)
    change = np.flatnonzero(is_test[1:] != is_test[:-1])
    near = np.zeros(len(day) + 1, dtype=np.int64)
    np.add.at(near, np.maximum(change - PURGE + 1, 0), 1)
    np.add.at(near, np.minimum(change + PURGE + 1, len(day)), -1)
    near = np.cumsum(near)[: len(day)] > 0
    return np.flatnonzero(~is_test & ~near), np.flatnonzero(is_test & ~near)


def _bootstrap(loss_lin: np.ndarray, loss_m: np.ndarray, day: np.ndarray) -> tuple[float, float]:
    """95% CI of mean(loss_lin - loss_m) resampling whole test days."""
    d = loss_lin - loss_m
    days, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=d)
    c = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(SEED + 1)
    idx = rng.integers(0, len(days), size=(B, len(days)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_regression(name, y, Ftr, Fte, Rtr, Rte, Ktr, Kte, ytr_idx, day_te, out, persist_te):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor

    ytr, yte = y[ytr_idx["all"]], y[ytr_idx["test"]]
    var = yte.var()
    preds = {}
    # unfitted: log RMS of the last 20 returns (for dvol20 the "no change" forecast, 0)
    preds["persistence"] = persist_te if name != "dvol20" else np.zeros(len(yte))
    t0 = time.time()
    preds["linear"] = RidgeCV(alphas=[0.1, 1, 10, 100]).fit(Ftr, ytr).predict(Fte)
    preds["hgb"] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED).fit(Ftr, ytr).predict(Fte)
    mlp_in_tr, mlp_in_te = np.c_[Ftr, Rtr], np.c_[Fte, Rte]
    preds["mlp"] = MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=100,
                                random_state=SEED).fit(mlp_in_tr, ytr).predict(mlp_in_te)
    ktr_sub = ytr_idx["knn"]
    preds["knn"] = KNeighborsRegressor(n_neighbors=100).fit(Ktr[ktr_sub], ytr[ktr_sub]).predict(Kte)
    res = {"test_var": float(var)}
    lin_loss = (yte - preds["linear"]) ** 2
    for m, p in preds.items():
        loss = (yte - p) ** 2
        entry = {"mse": float(loss.mean()), "r2": float(1 - loss.mean() / var)}
        if m not in ("linear", "persistence"):
            lo, hi = _bootstrap(lin_loss, loss, day_te)
            entry.update({"gain_vs_linear": float(lin_loss.mean() - loss.mean()),
                          "gain_rel": float((lin_loss.mean() - loss.mean()) / lin_loss.mean()),
                          "ci95": [lo, hi]})
        res[m] = entry
    out[name] = res
    line = "  ".join(f"{m} {res[m]['r2']:.3f}" for m in preds)
    gains = "  ".join(f"{m} {res[m]['gain_rel']*100:+.1f}% [{res[m]['ci95'][0]:+.4f},{res[m]['ci95'][1]:+.4f}]"
                      for m in ("hgb", "mlp", "knn"))
    print(f"{name:<7} R2: {line}\n        gain vs linear (rel MSE, 95% CI of abs): {gains}   ({time.time()-t0:.0f}s)", flush=True)


def run_jump(y, Ftr, Fte, Rtr, Rte, Ktr, Kte, idx, day_te, out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    ytr, yte = y[idx["all"]], y[idx["test"]]
    t0 = time.time()
    probs = {
        "base_rate": np.full(len(yte), ytr.mean()),
        "linear": LogisticRegression(C=1.0, max_iter=500).fit(Ftr, ytr).predict_proba(Fte)[:, 1],
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(Ftr, ytr).predict_proba(Fte)[:, 1],
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=100,
                             random_state=SEED).fit(np.c_[Ftr, Rtr], ytr).predict_proba(np.c_[Fte, Rte])[:, 1],
    }
    probs["knn"] = KNeighborsClassifier(n_neighbors=100).fit(Ktr[idx["knn"]], ytr[idx["knn"]]).predict_proba(Kte)[:, 1]
    ll = lambda p: -(yte * np.log(np.clip(p, 1e-6, 1)) + (1 - yte) * np.log(np.clip(1 - p, 1e-6, 1)))
    res = {"test_rate": float(yte.mean()), "train_rate": float(ytr.mean())}
    lin_loss = ll(probs["linear"])
    for m, p in probs.items():
        loss = ll(p)
        entry = {"logloss": float(loss.mean()),
                 "auc": float(roc_auc_score(yte, p)) if m != "base_rate" else 0.5}
        if m not in ("linear", "base_rate"):
            lo, hi = _bootstrap(lin_loss, loss, day_te)
            entry.update({"gain_vs_linear": float(lin_loss.mean() - loss.mean()),
                          "gain_rel": float((lin_loss.mean() - loss.mean()) / lin_loss.mean()),
                          "ci95": [lo, hi]})
        res[m] = entry
    out["jump20"] = res
    print(f"jump20  rate {res['test_rate']:.3f}  logloss/AUC: " +
          "  ".join(f"{m} {res[m]['logloss']:.4f}/{res[m]['auc']:.3f}" for m in probs), flush=True)
    print("        gain vs linear: " + "  ".join(
        f"{m} {res[m]['gain_rel']*100:+.1f}% [{res[m]['ci95'][0]:+.4f},{res[m]['ci95'][1]:+.4f}]"
        for m in ("hgb", "mlp", "knn")) + f"   ({time.time()-t0:.0f}s)", flush=True)


async def main() -> None:
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(_artifact_store() / await _dataset_artifact_path(29), columns=["close"]).iloc[-1_000_000:]
    X, feats, targets, day = build(df)
    tr, te = blocked_split(day)
    rng = np.random.default_rng(SEED)
    tr = np.sort(rng.choice(tr, size=min(N_TRAIN, len(tr)), replace=False))
    te = np.sort(rng.choice(te, size=min(N_TEST, len(te)), replace=False))
    print(f"windows={len(day)}  train sample={len(tr)}  test sample={len(te)}  "
          f"test days={len(np.unique(day[te]))}", flush=True)

    sc = StandardScaler().fit(feats[tr])
    Ftr, Fte = sc.transform(feats[tr]), sc.transform(feats[te])
    sigma = X[tr].std()
    Rtr, Rte = X[tr] / sigma, X[te] / sigma
    kcols = list(range(len(SCALES))) + [feats.shape[1] - 3, feats.shape[1] - 2, feats.shape[1] - 1]
    Ktr, Kte = Ftr[:, kcols], Fte[:, kcols]
    idx = {"all": tr, "test": te,
           "knn": np.sort(rng.choice(len(tr), size=min(N_TRAIN_KNN, len(tr)), replace=False))}
    day_te = day[te]

    out = {"n_windows": int(len(day)), "n_train": int(len(tr)), "n_test": int(len(te)), "k_jump": K_JUMP}
    for name in ("rv20", "rv60", "rv240", "dvol20"):
        run_regression(name, targets[name], Ftr, Fte, Rtr, Rte, Ktr, Kte, idx, day_te, out,
                       persist_te=feats[te][:, SCALES.index(20)])
    run_jump(targets["jump20"], Ftr, Fte, Rtr, Rte, Ktr, Kte, idx, day_te, out)

    path = Path(__file__).resolve().parent / "nonlinear_headroom_screen.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    asyncio.run(main())
