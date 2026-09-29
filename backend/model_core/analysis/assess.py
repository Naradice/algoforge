"""assess_target: is a target worth training a sequence model on? (docs/model-layer.md point 6)

Before any training run, score cheap predictors on the exact windows a model would see and say
which of four situations the target is in:

  trivial       the strong linear baseline (or persistence) already explains almost everything
                (R^2 >= trivial_r2, or AUC >= trivial_auc) -- the target very likely overlaps the
                input window; a model "win" on it would not measure the intended ability
  unpredictable no model finds signal (best R^2 < min_signal_r2 / best AUC < 0.55) -- checked over
                every model, not only the linear one: a target can be a nonlinear function of the
                input that only the nonlinear models compute (Phase 10A, trend_change)
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
from model_core.analysis.features import (
    aux_features, exog_features, future_aux_mean, har_features, past_aux_mean, linear_extras, market_features, time_features, trend_features,
    vol_memory_features,
)
from model_core.analysis.splits import blocked_split, chronological_split, holdout_split
from model_core.analysis.targets import CLASSIFICATION, REGIME, SIGNED, make_target, persistence
from model_core.analysis.windows import build_return_windows, concat_windows

NONLINEAR = ("hgb", "mlp", "knn")


def _subsample(idx: np.ndarray, n: int, rng) -> np.ndarray:
    return idx if len(idx) <= n else np.sort(rng.choice(idx, size=n, replace=False))


def _std(train: np.ndarray, *arrays: np.ndarray):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return [(a - mu) / sd for a in (train, *arrays)]


def assess_target(
    close: pd.Series | dict[str, pd.Series],
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
    exog: pd.DataFrame | None = None,
    split: str = "blocked",
    bootstrap_group: str = "day",
    max_gap: pd.Timedelta | str | None = None,
    market: bool = False,
    test_period: tuple | None = None,
    aux: pd.DataFrame | None = None,
) -> dict:
    """Score persistence, the strong linear baseline and nonlinear `models` on `target` built from
    `close` (a DatetimeIndex-ed price series), and return metrics plus a verdict (module docstring).

    `exog`: closes of other instruments on `close`'s index (windows.align_closes). Every model then
    also gets their multi-scale signed sums and log RMS up to the anchor (features.exog_features);
    `linear_own` -- the same linear baseline without them -- is scored too, and
    metrics.linear.exog_gain_rel / exog_ci95 say whether the other instruments add anything linearly.
    The verdict's headroom is measured over the linear baseline *with* the other instruments.

    Panel / long horizons: `close` may be {name: series} -- windows and targets are built per
    instrument (a target's thresholds, e.g. extreme's quantile, are per instrument) and pooled into
    one model. `split="chronological"` tests on the last test_frac of dates after an
    obs + horizon-day purge (splits.chronological_split) instead of random days; `bootstrap_group=
    "month"` resamples months (use it when overlapping targets span days); `max_gap` lets weekends /
    holidays inside daily bars count as contiguous; `market=True` (panel only) adds cross-instrument
    means of past returns / volatility per day (features.market_features), scored like `exog`.
    `test_period=(start, end)` holds out that date block instead (splits.holdout_split; purged on
    both sides, the rest trains) -- to check a chronological result on other periods.

    `aux`: row-level non-price series on `close`'s index (e.g. log volume, a signed-flow proxy),
    already transformed. Models get each one's mean over the last 1/5/20/60 bars, scored like
    `exog` (`linear_own` without them). target="future_aux" forecasts aux column 0's mean over the
    next `horizon` bars (persistence: its mean over the last `horizon` bars)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    unknown = set(models) - set(NONLINEAR)
    if unknown:
        raise ValueError(f"unknown models {sorted(unknown)}; choose from {NONLINEAR}")
    t0 = time.time()
    if split not in ("blocked", "chronological") or bootstrap_group not in ("day", "month"):
        raise ValueError("split must be blocked|chronological and bootstrap_group day|month")
    panel = isinstance(close, dict)
    if panel and exog is not None:
        raise ValueError("exog is for a single series; a panel can use market=True")
    if aux is not None and panel:
        raise ValueError("aux is for a single series")
    if target == "future_aux" and aux is None:
        raise ValueError("target future_aux needs aux")
    if market and not panel:
        raise ValueError("market=True needs a panel ({name: close})")
    series = close if panel else {"": close}
    parts, ys, names = [], [], []
    for name, s in series.items():
        wi = build_return_windows(s, obs=obs, horizon=horizon, exog=exog, max_gap=max_gap, aux=aux)
        if len(wi):
            parts.append(wi)
            ys.append(future_aux_mean(wi, horizon) if target == "future_aux"
                      else make_target(target, wi.X, wi.F, horizon, **(target_kwargs or {})))
            names.append(name)
    if not parts:
        raise ValueError("only 0 gap-free windows -- not enough to assess")
    w = parts[0] if len(parts) == 1 else concat_windows(parts)
    y = np.concatenate(ys)
    if len(w) < 1000:
        raise ValueError(f"only {len(w)} gap-free windows -- not enough to assess")
    defined = np.isfinite(y)
    if not defined.all():                     # e.g. direction over a flat future
        w, y = w.take(np.flatnonzero(defined)), y[defined]
    classification = target in CLASSIFICATION
    signed = target in SIGNED
    M = market_features(w) if market else None     # per-day cross-section: needs all windows

    rng = np.random.default_rng(seed)
    if test_period is not None:
        tr, te = holdout_split(w.day, *test_period, purge_days=obs + horizon)
    elif split == "chronological":
        tr, te = chronological_split(w.day, test_frac=test_frac, purge_days=obs + horizon)
    else:
        tr, te = blocked_split(w.day, test_frac=test_frac, purge=obs + horizon, seed=seed)
    tr, te = _subsample(tr, n_train, rng), _subsample(te, n_test, rng)
    # Features only for the sampled windows: on ~2M windows the full feature matrix is several GB.
    n_windows = len(w)
    sel = np.r_[tr, te]
    w, y = w.take(sel), y[sel]
    M = M[sel] if M is not None else None
    tr, te = np.arange(len(tr)), np.arange(len(tr), len(sel))
    if classification and len(np.unique(y[tr])) < 2:
        raise ValueError("the training split contains a single class -- nothing to assess")

    base = vol_memory_features(w.X)
    ts = w.anchor_ts if with_time else None
    clock = time_features(w.anchor_ts) if with_time else np.empty((len(w), 0))
    lags = w.X if signed else np.empty((len(w), 0))    # AR(obs) term for sign-dependent targets
    trend = trend_features(w.X) if target in REGIME else np.empty((len(w), 0))
    E = np.c_[M if M is not None else exog_features(w), aux_features(w)]   # other instruments / market / aux
    F_own = np.c_[base, linear_extras(w.X, ts), clock, lags, trend]
    F_lin = np.c_[F_own, E]
    F_tree = np.c_[base, clock, lags, trend, E]
    knn_cols = np.c_[har_features(w.X), base[:, -3:], clock, trend]
    Lin_tr, Lin_te = _std(F_lin[tr], F_lin[te])
    Own_tr, Own_te = _std(F_own[tr], F_own[te]) if E.shape[1] else (None, None)
    del F_lin, F_own                                    # standardized copies are all that is used below
    Mlp_tr, Mlp_te = _std(np.c_[F_tree[tr], w.X[tr]], np.c_[F_tree[te], w.X[te]])
    Knn_tr, Knn_te = _std(knn_cols[tr], knn_cols[te])
    knn_sub = _subsample(np.arange(len(tr)), 100_000, rng)
    ytr, yte = y[tr], y[te]
    if bootstrap_group == "month":
        groups = np.asarray(w.anchor_ts[te].year * 12 + w.anchor_ts[te].month)
    else:
        groups = w.day[te]

    preds: dict[str, np.ndarray] = {}
    if classification:
        preds["base_rate"] = np.full(len(te), ytr.mean())
        preds["linear"] = LogisticRegression(C=1.0, max_iter=1000).fit(Lin_tr, ytr).predict_proba(Lin_te)[:, 1]
        if Own_tr is not None:
            preds["linear_own"] = LogisticRegression(C=1.0, max_iter=1000).fit(Own_tr, ytr).predict_proba(Own_te)[:, 1]
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
        preds["persistence"] = (past_aux_mean(w.take(te), horizon) if target == "future_aux"
                                else persistence(target, w.X[te], horizon))
        preds["linear"] = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Lin_tr, ytr).predict(Lin_te)
        if Own_tr is not None:
            preds["linear_own"] = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Own_tr, ytr).predict(Own_te)
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

    if "linear_own" in loss:
        lo, hi = paired_block_bootstrap_ci(loss["linear_own"], loss["linear"], groups, n_boot=n_boot, seed=seed + 3)
        gain = float(loss["linear_own"].mean() - loss["linear"].mean())
        metrics["linear"].update({"exog_gain": gain, "exog_gain_rel": gain / float(loss["linear_own"].mean()),
                                  "exog_ci95": [lo, hi]})

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
        "exog": list(w.exog_names) + list(w.aux_names) + (["market"] if market else []),
        "instruments": names if panel else None, "split": split, "bootstrap_group": bootstrap_group,
        "test_period": [str(w.anchor_ts[te].min()), str(w.anchor_ts[te].max())],
        "task": "classification" if classification else "regression",
        "n_windows": int(n_windows), "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_test_days": int(len(np.unique(w.day[te]))), "n_bootstrap_groups": int(len(np.unique(groups))),
        "positive_rate": float(yte.mean()) if classification else None,
        "metrics": metrics, "verdict": verdict, "reason": reason,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def _verdict(metrics, classification, models, threshold, trivial_r2, trivial_auc, min_signal_r2):
    if classification:
        lin_auc = metrics["linear"]["auc"]
        best_auc = max(metrics[m]["auc"] for m in metrics if m != "base_rate")
        if lin_auc >= trivial_auc:
            return "trivial", f"linear AUC {lin_auc:.3f} >= {trivial_auc}: target likely computable from the input"
        if best_auc < 0.55:
            return "unpredictable", f"best AUC {best_auc:.3f} < 0.55 (linear {lin_auc:.3f}): no usable signal in the window"
    else:
        best_simple = max(metrics["linear"]["r2"], metrics["persistence"]["r2"])
        if best_simple >= trivial_r2:
            return "trivial", (f"a simple predictor reaches R2 {best_simple:.3f} >= {trivial_r2}: the target "
                               "very likely overlaps the input window")
        best_r2 = max(v["r2"] for m, v in metrics.items() if m != "persistence")
        if best_r2 < min_signal_r2:
            return "unpredictable", (f"best R2 {best_r2:.3f} < {min_signal_r2} (linear {metrics['linear']['r2']:.3f}): "
                                     "no usable signal")
        if max(v["r2"] for v in metrics.values()) >= trivial_r2:
            return "trivial", f"a nonlinear model reaches R2 >= {trivial_r2}: the target very likely overlaps the input window"
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
