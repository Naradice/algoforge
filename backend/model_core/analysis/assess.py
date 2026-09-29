"""assess_target: is a target worth training a sequence model on? (docs/model-layer.md point 6)

Before any training run, score cheap predictors on the exact windows a model would see and say
which of four situations the target is in:

  trivial       the strong linear baseline (or persistence) already explains almost everything
                (R^2 >= trivial_r2, or AUC >= trivial_auc) -- the target very likely overlaps the
                input window; a model "win" on it would not measure the intended ability
  unpredictable even the strong linear baseline finds no signal (R^2 < min_signal_r2 / AUC < 0.55)
  headroom      a smooth nonlinear model (MLP or k-NN) beats the strong linear baseline by
                >= headroom_threshold (relative loss) with the paired day-block bootstrap CI
                excluding zero -- a target where a neural sequence model could plausibly learn more
  tree_only_headroom
                only gradient-boosted trees clear that bar while the MLP that was run does not --
                the gain is threshold/interaction structure trees find but a smooth net did not
                (on USDJPY volatility this is what Phase 9a found); weak evidence for training a
                Transformer on it
  no_headroom   predictable, but nonlinear models add nothing reliable over the linear baseline

The linear baseline is deliberately strong (multi-scale RMS HAR, every log|r|, signed sums, their
squares/interactions and -- with with_time -- time of day, weekday and hour x RMS), because a weak
baseline makes any model look good (Phase 8c). Models are fitted on train days only and scored once
on held-out days; nothing is selected on the test set.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from model_core.analysis.bootstrap import paired_block_bootstrap_ci
from model_core.analysis.features import SCALES, har_features, linear_extras, time_features, vol_memory_features
from model_core.analysis.splits import blocked_split
from model_core.analysis.targets import CLASSIFICATION, SIGNED, make_target, persistence
from model_core.analysis.windows import build_return_windows

NONLINEAR = ("hgb", "mlp", "knn")


def _subsample(idx: np.ndarray, n: int, rng) -> np.ndarray:
    return idx if len(idx) <= n else np.sort(rng.choice(idx, size=n, replace=False))


def _std(train: np.ndarray, *arrays: np.ndarray):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return [(a - mu) / sd for a in (train, *arrays)]


def assess_target(
    close: pd.Series,
    target: str = "future_log_rv",
    horizon: int = 20,
    obs: int = 60,
    with_time: bool = True,
    models: tuple[str, ...] = NONLINEAR,
    n_train: int = 300_000,
    n_test: int = 100_000,
    test_frac: float = 0.2,
    seed: int = 0,
    n_boot: int = 500,
    headroom_threshold: float = 0.02,
    trivial_r2: float = 0.95,
    trivial_auc: float = 0.99,
    min_signal_r2: float = 0.05,
    target_kwargs: dict | None = None,
) -> dict:
    """Score persistence, the strong linear baseline and nonlinear `models` on `target` built from
    `close` (a DatetimeIndex-ed price series), and return metrics plus a verdict (module docstring)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    unknown = set(models) - set(NONLINEAR)
    if unknown:
        raise ValueError(f"unknown models {sorted(unknown)}; choose from {NONLINEAR}")
    t0 = time.time()
    w = build_return_windows(close, obs=obs, horizon=horizon)
    if len(w) < 1000:
        raise ValueError(f"only {len(w)} gap-free windows -- not enough to assess")
    y = make_target(target, w.X, w.F, horizon, **(target_kwargs or {}))
    defined = np.isfinite(y)
    if not defined.all():                     # e.g. direction over a flat future
        w, y = w.take(np.flatnonzero(defined)), y[defined]
    classification = target in CLASSIFICATION
    signed = target in SIGNED

    rng = np.random.default_rng(seed)
    tr, te = blocked_split(w.day, test_frac=test_frac, purge=obs + horizon, seed=seed)
    tr, te = _subsample(tr, n_train, rng), _subsample(te, n_test, rng)
    if classification and len(np.unique(y[tr])) < 2:
        raise ValueError("the training split contains a single class -- nothing to assess")

    base = vol_memory_features(w.X)
    ts = w.anchor_ts if with_time else None
    clock = time_features(w.anchor_ts) if with_time else np.empty((len(w), 0))
    lags = w.X if signed else np.empty((len(w), 0))    # AR(obs) term for sign-dependent targets
    F_lin = np.c_[base, linear_extras(w.X, ts), clock, lags]
    F_tree = np.c_[base, clock, lags]
    knn_cols = np.c_[har_features(w.X), base[:, -3:], clock]
    Lin_tr, Lin_te = _std(F_lin[tr], F_lin[te])
    Mlp_tr, Mlp_te = _std(np.c_[F_tree[tr], w.X[tr]], np.c_[F_tree[te], w.X[te]])
    Knn_tr, Knn_te = _std(knn_cols[tr], knn_cols[te])
    knn_sub = _subsample(np.arange(len(tr)), 100_000, rng)
    ytr, yte, groups = y[tr], y[te], w.day[te]

    preds: dict[str, np.ndarray] = {}
    if classification:
        preds["base_rate"] = np.full(len(te), ytr.mean())
        preds["linear"] = LogisticRegression(C=1.0, max_iter=1000).fit(Lin_tr, ytr).predict_proba(Lin_te)[:, 1]
        if "hgb" in models:
            preds["hgb"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed) \
                .fit(F_tree[tr], ytr).predict_proba(F_tree[te])[:, 1]
        if "mlp" in models:
            preds["mlp"] = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=200,
                                         random_state=seed).fit(Mlp_tr, ytr).predict_proba(Mlp_te)[:, 1]
        if "knn" in models:
            preds["knn"] = KNeighborsClassifier(n_neighbors=100).fit(Knn_tr[knn_sub], ytr[knn_sub]).predict_proba(Knn_te)[:, 1]
        loss = {m: -(yte * np.log(np.clip(p, 1e-6, 1)) + (1 - yte) * np.log(np.clip(1 - p, 1e-6, 1)))
                for m, p in preds.items()}
        metrics = {m: {"logloss": float(loss[m].mean()),
                       "auc": float(roc_auc_score(yte, p)) if m != "base_rate" else 0.5}
                   for m, p in preds.items()}
    else:
        preds["persistence"] = persistence(target, w.X[te], horizon)
        preds["linear"] = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Lin_tr, ytr).predict(Lin_te)
        if "hgb" in models:
            preds["hgb"] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed) \
                .fit(F_tree[tr], ytr).predict(F_tree[te])
        if "mlp" in models:
            preds["mlp"] = MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=200,
                                        random_state=seed).fit(Mlp_tr, ytr).predict(Mlp_te)
        if "knn" in models:
            preds["knn"] = KNeighborsRegressor(n_neighbors=100).fit(Knn_tr[knn_sub], ytr[knn_sub]).predict(Knn_te)
        var = float(yte.var())
        loss = {m: (yte - p) ** 2 for m, p in preds.items()}
        metrics = {m: {"mse": float(loss[m].mean()), "r2": float(1 - loss[m].mean() / var)} for m in preds}

    ref = "base_rate" if classification else "persistence"
    lo, hi = paired_block_bootstrap_ci(loss[ref], loss["linear"], groups, n_boot=n_boot, seed=seed + 2)
    gain = float(loss[ref].mean() - loss["linear"].mean())
    metrics["linear"].update({"reference": ref, "gain_vs_reference": gain,
                              "gain_rel_vs_reference": gain / float(loss[ref].mean()), "ci95_vs_reference": [lo, hi]})

    for m in models:
        if m in loss:
            lo, hi = paired_block_bootstrap_ci(loss["linear"], loss[m], groups, n_boot=n_boot, seed=seed + 1)
            gain = float(loss["linear"].mean() - loss[m].mean())
            metrics[m].update({"gain_vs_linear": gain, "gain_rel": gain / float(loss["linear"].mean()),
                               "ci95": [lo, hi]})

    verdict, reason = _verdict(metrics, classification, models, headroom_threshold, trivial_r2,
                               trivial_auc, min_signal_r2)
    return {
        "target": target, "horizon": horizon, "obs": obs, "with_time": with_time,
        "task": "classification" if classification else "regression",
        "n_windows": int(len(w)), "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_test_days": int(len(np.unique(groups))),
        "positive_rate": float(yte.mean()) if classification else None,
        "metrics": metrics, "verdict": verdict, "reason": reason,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def _verdict(metrics, classification, models, threshold, trivial_r2, trivial_auc, min_signal_r2):
    if classification:
        lin_auc = metrics["linear"]["auc"]
        if lin_auc >= trivial_auc:
            return "trivial", f"linear AUC {lin_auc:.3f} >= {trivial_auc}: target likely computable from the input"
        if lin_auc < 0.55:
            return "unpredictable", f"linear AUC {lin_auc:.3f} < 0.55: no usable signal in the window"
    else:
        best_simple = max(metrics["linear"]["r2"], metrics["persistence"]["r2"])
        if best_simple >= trivial_r2:
            return "trivial", (f"a simple predictor reaches R2 {best_simple:.3f} >= {trivial_r2}: the target "
                               "very likely overlaps the input window")
        if metrics["linear"]["r2"] < min_signal_r2:
            return "unpredictable", f"linear R2 {metrics['linear']['r2']:.3f} < {min_signal_r2}: no usable signal"
    gains = {m: metrics[m] for m in models if m in metrics}
    passing = {m: g for m, g in gains.items() if g["gain_rel"] >= threshold and g["ci95"][0] > 0}
    if passing:
        m = max(passing, key=lambda k: passing[k]["gain_rel"])
        detail = (f"{m} beats the strong linear baseline by {passing[m]['gain_rel']*100:.1f}% "
                  f"(95% CI of loss gain {passing[m]['ci95'][0]:+.4f}..{passing[m]['ci95'][1]:+.4f})")
        if set(passing) == {"hgb"} and "mlp" in gains:
            return "tree_only_headroom", (f"{detail}, but the MLP changes loss by "
                                          f"{gains['mlp']['gain_rel']*100:+.1f}% -- only trees find it")
        return "headroom", detail
    best = max(gains, key=lambda k: gains[k]["gain_rel"]) if gains else None
    if best is None:
        return "no_headroom", "no nonlinear model was run"
    g = gains[best]
    return "no_headroom", (f"best nonlinear ({best}) changes loss by {g['gain_rel']*100:+.1f}% vs linear "
                           f"(CI {g['ci95'][0]:+.4f}..{g['ci95'][1]:+.4f}); threshold {threshold*100:.0f}% with CI > 0")
