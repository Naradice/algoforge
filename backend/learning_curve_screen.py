"""
Phase 9b: learning curves on USDJPY volatility targets -- is there a small-data regime where
pretraining could matter?

Phase 9a found no target where a smooth nonlinear model beats a strong linear baseline with the
clock given. Option 3 (pretraining as SAMPLE EFFICIENCY) only has practical value if, with little
USDJPY data, learned models fall well short of what they reach with all of it -- and if the strong
linear baseline itself does too. If the linear model is already near its ceiling at 5-10% of the
data, pretraining could at best help a neural net catch up to a model that is cheap to fit.

Same windows, features and blocked day split as nonlinear_headroom_screen.py (Phase 9a). Training
subsets are random WHOLE training days (fraction of days), 3 draws per fraction; the full pool is
capped at N_TRAIN windows. Fixed test sample. Models, each with time-of-day/weekday features:
  linear : ridge on the screen features + squares/interactions of the 7 RMS scales + ToD
           + hour x RMS(5/20/60)
  hgb    : HistGradientBoosting on the screen features + ToD
  mlp    : 2x(128, 64) MLP on standardized screen features + ToD + z-scored raw returns

Usage: python learning_curve_screen.py   (writes learning_curve_screen.json)
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

from nonlinear_headroom_screen import MAX_H, N_TEST, N_TRAIN, OBS, SCALES, SEED, blocked_split, build
from probe_representations import _artifact_store, _dataset_artifact_path

FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.25, 1.0)
DRAWS = 3
TARGETS = ("rv20", "rv60")


def _time_features(df: pd.DataFrame, n_windows_expected: int):
    close = df["close"].to_numpy()
    r = np.diff(np.log(close))
    gap = df.index.to_series().diff().dt.total_seconds().ne(60).to_numpy()[1:]
    bad = np.concatenate([[0], np.cumsum(gap)])
    t = np.arange(OBS - 1, len(r) - MAX_H)
    t = t[(bad[t + MAX_H + 1] - bad[t - OBS + 1]) == 0]
    assert len(t) == n_windows_expected
    ts = df.index[t + 1]
    mod = (ts.hour * 60 + ts.minute).to_numpy()
    tod = np.c_[np.c_[[f(2 * np.pi * k * mod / 1440) for k in (1, 2, 3, 4) for f in (np.sin, np.cos)]].T,
                np.eye(7)[ts.dayofweek.to_numpy()][:, :5]]
    return tod, np.eye(24)[ts.hour.to_numpy()]


def _fit_all(y, tr, te, F_lin, F_tree, F_mlp):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    yte = y[te]
    var = yte.var()
    out = {}
    sc = StandardScaler().fit(F_lin[tr])
    p = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(sc.transform(F_lin[tr]), y[tr]).predict(sc.transform(F_lin[te]))
    out["linear"] = float(1 - ((yte - p) ** 2).mean() / var)
    p = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED).fit(F_tree[tr], y[tr]).predict(F_tree[te])
    out["hgb"] = float(1 - ((yte - p) ** 2).mean() / var)
    sc = StandardScaler().fit(F_mlp[tr])
    p = MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=200,
                     random_state=SEED).fit(sc.transform(F_mlp[tr]), y[tr]).predict(sc.transform(F_mlp[te]))
    out["mlp"] = float(1 - ((yte - p) ** 2).mean() / var)
    return out


async def main() -> None:
    df = pd.read_parquet(_artifact_store() / await _dataset_artifact_path(29), columns=["close"]).iloc[-1_000_000:]
    X, feats, targets, day = build(df)
    tod, hour = _time_features(df, len(day))
    rms = feats[:, :len(SCALES)]
    inter = np.column_stack([rms[:, i] * rms[:, j] for i in range(7) for j in range(i + 1, 7)])
    hx = np.column_stack([hour[:, h] * rms[:, SCALES.index(s)] for h in range(24) for s in (5, 20, 60)])
    F_lin = np.c_[feats, rms ** 2, inter, tod, hour, hx]
    F_tree = np.c_[feats, tod]
    F_mlp = np.c_[feats, tod, X / X.std()]

    tr_all, te_all = blocked_split(day)
    rng = np.random.default_rng(SEED)
    te = np.sort(rng.choice(te_all, size=min(N_TEST, len(te_all)), replace=False))
    train_days = np.unique(day[tr_all])
    print(f"train pool: {len(tr_all)} windows / {len(train_days)} days; test sample {len(te)}", flush=True)

    results = {t: {} for t in TARGETS}
    for frac in FRACTIONS:
        for draw in range(DRAWS if frac < 1.0 else 1):
            drng = np.random.default_rng(SEED + 100 * draw + int(frac * 1000))
            n_days = max(1, int(round(len(train_days) * frac)))
            chosen = drng.choice(train_days, size=n_days, replace=False)
            tr = tr_all[np.isin(day[tr_all], chosen)]
            if len(tr) > N_TRAIN:
                tr = np.sort(drng.choice(tr, size=N_TRAIN, replace=False))
            for target in TARGETS:
                t0 = time.time()
                r = _fit_all(targets[target], tr, te, F_lin, F_tree, F_mlp)
                results[target].setdefault(str(frac), []).append({"n_days": int(n_days), "n_windows": int(len(tr)), **r})
                print(f"{target} frac={frac:<5} draw={draw} days={n_days:<4} windows={len(tr):<7} "
                      f"linear {r['linear']:.3f}  hgb {r['hgb']:.3f}  mlp {r['mlp']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\nSummary (mean over draws; R^2 on the fixed test sample)")
    for target in TARGETS:
        full = results[target]["1.0"][0]
        print(f"--- {target}  (full data: linear {full['linear']:.3f}  hgb {full['hgb']:.3f}  mlp {full['mlp']:.3f})")
        for frac in FRACTIONS:
            rows = results[target][str(frac)]
            m = {k: np.mean([row[k] for row in rows]) for k in ("linear", "hgb", "mlp")}
            s = {k: np.std([row[k] for row in rows]) for k in ("linear", "hgb", "mlp")}
            print(f"  {frac*100:>5.0f}%  days={rows[0]['n_days']:<4} " + "  ".join(
                f"{k} {m[k]:.3f}±{s[k]:.3f} (gap {m[k]-full[k]:+.3f})" for k in ("linear", "hgb", "mlp")))

    path = Path(__file__).resolve().parent / "learning_curve_screen.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    asyncio.run(main())
