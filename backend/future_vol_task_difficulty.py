"""
Phase 8a: task difficulty of a future-only volatility target, before any training.

Phase 7b showed the pipeline's target (vol_20 at the bar after the window) shares 19 of its 20
returns with the input window: a hand feature gets R^2 0.994 and a from-scratch Transformer 0.98,
so it cannot measure transfer. Proposed replacement, keeping the 60-bar input window:

    input : closes t-60 .. t        -> 60 log returns r_{t-59} .. r_t
    target: y_t = std(r_{t+1}, ..., r_{t+20})       (no return shared with the input)

Only windows whose whole span t-60 .. t+20 is gap-free are used. Predictors, weakest first:

  - mean            : unconditional train mean (R^2 = 0 by construction on the train split)
  - time_of_day     : train-mean y per minute-of-day (USDJPY only; the model never sees time)
  - persistence     : y_hat = std(r_{t-19} .. r_t), no fitting
  - persistence_fit : linear fit on that same feature
  - har             : linear regression of log y on log std over the last 5 / 20 / 60 returns
  - knn             : k=50 nearest neighbours on the z-scored 60-return window
  - mlp             : 2x64 MLP on the z-scored 60-return window

Every predictor is scored on R^2 of y and of log y, under two splits:
  - chronological : train on the first 80% of windows, test on the last 20%, with an 80-bar purge
  - random        : random 80/20 split of windows -- like regime_controlled, adjacent windows share
                    future returns across train/test
  - blocked       : 1-day blocks assigned randomly 80/20, purging PURGE windows at every block
                    boundary -- same regime mix as random, no adjacent-window sharing. So
                    (random - blocked) = split leakage, (blocked - chronological) = regime shift

Usage: python future_vol_task_difficulty.py   (writes future_vol_task_difficulty.json)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_representations import _artifact_store, _dataset_artifact_path

OBS = 60          # input returns
HORIZON = 20      # future returns in the target
PURGE = OBS + HORIZON
N_TRAIN, N_TEST = 100_000, 30_000   # subsample for knn/mlp (linear models use all windows)
SEED = 0

DATASETS_ALL = {
    # label: (dataset id, max_rows tail, has time-of-day structure)
    "USDJPY": (29, 1_000_000, True),
    "DDM": (52, None, False),
}
DATASETS = {k: v for k, v in DATASETS_ALL.items() if not sys.argv[1:] or k in sys.argv[1:]}


def _windows(df: pd.DataFrame):
    close = df["close"].to_numpy(dtype=np.float64)
    r = np.diff(np.log(close))                       # r[k] = log return from row k to k+1
    gap = df.index.to_series().diff().dt.total_seconds().ne(60).to_numpy()[1:]  # gap[k]: return k spans a gap
    n_ret = len(r)
    # window anchor t = index of r_t (last input return); input r[t-59..t], target r[t+1..t+20]
    bad = np.concatenate([[0], np.cumsum(gap)])
    t = np.arange(OBS - 1, n_ret - HORIZON)
    ok = (bad[t + HORIZON + 1] - bad[t - OBS + 1]) == 0
    t = t[ok]
    X = np.lib.stride_tricks.sliding_window_view(r, OBS)[t - OBS + 1]        # [n, 60]
    F = np.lib.stride_tricks.sliding_window_view(r, HORIZON)[t + 1]          # [n, 20]
    y = F.std(axis=1, ddof=1)
    ts = df.index[t + 1]                                                      # time of the anchor close
    keep = y > 0
    return X[keep], y[keep], ts[keep]


def _r2(y, p):
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def _score(name, y_te, p_te, out):
    p_te = np.maximum(p_te, 1e-12)
    out[name] = {"r2": _r2(y_te, p_te), "r2_log": _r2(np.log(y_te), np.log(p_te))}


def _evaluate(X, y, ts, tr, te, tod: bool) -> dict:
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor

    out = {}
    rng = np.random.default_rng(SEED)
    y_tr, y_te = y[tr], y[te]
    _score("mean", y_te, np.full(len(te), y_tr.mean()), out)

    if tod:
        mod = ts.hour * 60 + ts.minute
        table = pd.Series(y_tr).groupby(np.asarray(mod[tr])).mean()
        p = pd.Series(np.asarray(mod[te])).map(table).fillna(y_tr.mean()).to_numpy()
        _score("time_of_day", y_te, p, out)

    s20 = X[:, -20:].std(axis=1, ddof=1)
    _score("persistence", y_te, s20[te], out)
    lr = LinearRegression().fit(s20[tr, None], y_tr)
    _score("persistence_fit", y_te, lr.predict(s20[te, None]), out)

    har = np.log(np.c_[X[:, -5:].std(axis=1, ddof=1), s20, X.std(axis=1, ddof=1)] + 1e-12)
    lr = LinearRegression().fit(har[tr], np.log(y_tr))
    _score("har", y_te, np.exp(lr.predict(har[te])), out)

    sub_tr = rng.choice(tr, size=min(N_TRAIN, len(tr)), replace=False)
    sub_te = rng.choice(np.arange(len(te)), size=min(N_TEST, len(te)), replace=False)
    sigma = X[sub_tr].std()
    Z_tr, Z_te = X[sub_tr] / sigma, X[te][sub_te] / sigma
    # target on log scale for the learned models (volatility is heavy-tailed)
    knn = KNeighborsRegressor(n_neighbors=50).fit(Z_tr, np.log(y[sub_tr]))
    _score("knn", y_te[sub_te], np.exp(knn.predict(Z_te)), out)
    # |r| windows: raw-return distances are dominated by sign patterns irrelevant to volatility
    knn_abs = KNeighborsRegressor(n_neighbors=50).fit(np.abs(Z_tr), np.log(y[sub_tr]))
    _score("knn_abs", y_te[sub_te], np.exp(knn_abs.predict(np.abs(Z_te))), out)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=200,
                       random_state=SEED).fit(Z_tr, np.log(y[sub_tr]))
    _score("mlp", y_te[sub_te], np.exp(mlp.predict(Z_te)), out)
    return out


def _blocked_split(ts: pd.DatetimeIndex, n: int):
    day = np.asarray(ts.normalize().asi8)
    days = np.unique(day)
    test_days = np.random.default_rng(SEED).choice(days, size=int(len(days) * 0.2), replace=False)
    is_test = np.isin(day, test_days)
    # purge: drop any window within PURGE positions of a train/test boundary
    boundary = np.flatnonzero(np.diff(is_test.astype(np.int8)) != 0)
    near = np.zeros(n, dtype=bool)
    for b in boundary:
        near[max(0, b - PURGE + 1): b + PURGE + 1] = True
    return np.flatnonzero(~is_test & ~near), np.flatnonzero(is_test & ~near)


async def main() -> None:
    results = {}
    for label, (dataset_id, tail, tod) in DATASETS.items():
        df = pd.read_parquet(_artifact_store() / await _dataset_artifact_path(dataset_id), columns=["close"])
        if tail:
            df = df.iloc[-tail:]
        X, y, ts = _windows(df)
        n = len(y)
        cut = int(n * 0.8)
        splits = {
            "chronological": (np.arange(0, cut - PURGE), np.arange(cut, n)),
            "random": tuple(np.sort(a) for a in np.split(np.random.default_rng(SEED).permutation(n), [cut])),
            "blocked": _blocked_split(ts, n),
        }
        results[label] = {"n_windows": n}
        print(f"\n=== {label}: {n} windows ===")
        print(f"{'predictor':<17}" + "".join(f"{s + ' R2':>18}{s + ' R2(log)':>22}" for s in splits))
        per_split = {s: _evaluate(X, y, ts, tr, te, tod) for s, (tr, te) in splits.items()}
        for name in per_split["chronological"]:
            print(f"{name:<17}" + "".join(
                f"{per_split[s][name]['r2']:>18.3f}{per_split[s][name]['r2_log']:>22.3f}" for s in splits), flush=True)
        results[label].update(per_split)

    out = Path(__file__).resolve().parent / "future_vol_task_difficulty.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
