"""POST HOC supplement to the A1 SA/SB reproducibility gate (written after the gate result was seen; it does NOT change the gate or its outcome).
SA and SB are the same process with different seeds, so for every point the 10 run-level M vectors are exchangeable. Random 5/5 re-splits give the distribution of the
agreement fraction that the pre-committed criterion (|SA-SB| <= 2*sqrt(SE_SA^2+SE_SB^2), all M components) produces when nothing but seed noise differs.
Output: results/a1_gate_posthoc.json"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402
from dr_a1_analysis import run_means  # noqa: E402

N_SPLITS = 2000
SEED = 20261126
MULTS = (1.0, 2.0, 3.0)


def agree_fraction(a, b, mult):
    """a, b: [points, 5 runs, components] -> (fraction of points agreeing in all components, per-component fractions)."""
    se = np.sqrt(a.var(axis=1, ddof=1) / a.shape[1] + b.var(axis=1, ddof=1) / b.shape[1])
    ok = np.abs(a.mean(axis=1) - b.mean(axis=1)) <= mult * se
    return float(ok.all(axis=1).mean()), ok.mean(axis=0)


def main():
    code = E.code_state()
    dec = json.load(open(os.path.join(E.RESULTS, "a1_decisions.json"), encoding="utf-8"))
    comps = tuple(dec["M_components"])
    PT = pd.read_csv(os.path.join(E.RESULTS, "a1_points.csv"))
    R = pickle.load(open(os.path.join(E.OUT, "a1_stats.pkl"), "rb"))["runs"]
    pids = PT.loc[PT["all10"], "pid"].tolist()
    X = np.array([np.vstack([run_means(R, s, p, E.TPC_MAIN, comps, dec["exclude_first_window"]) for s in ("SA", "SB")]) for p in pids])      # [P, 10, C]
    rng = np.random.default_rng(SEED)
    out = {"_code": code, "note": "post hoc, descriptive; the pre-committed gate result stands", "n_points": len(pids), "components": list(comps), "n_random_splits": N_SPLITS, "by_multiplier": {}}
    perms = [np.argsort(rng.random((len(pids), 10)), axis=1) for _ in range(N_SPLITS)]
    rows = np.arange(len(pids))[:, None]
    for m in MULTS:
        obs, obs_c = agree_fraction(X[:, :5], X[:, 5:], m)
        null, null_c = [], []
        for p in perms:
            Y = X[rows, p]
            f, fc = agree_fraction(Y[:, :5], Y[:, 5:], m)
            null.append(f)
            null_c.append(fc)
        null = np.array(null)
        out["by_multiplier"][str(m)] = {"observed": obs, "observed_by_component": dict(zip(comps, map(float, obs_c))), "null_mean": float(null.mean()),
                                        "null_mean_by_component": dict(zip(comps, map(float, np.mean(null_c, axis=0)))),
                                        "null_p5": float(np.percentile(null, 5)), "null_p95": float(np.percentile(null, 95)),
                                        "null_fraction_ge_0.80": float((null >= 0.8).mean()), "null_fraction_le_observed": float((null <= obs + 1e-12).mean())}
    json.dump(out, open(os.path.join(E.RESULTS, "a1_gate_posthoc.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out["by_multiplier"]), flush=True)


if __name__ == "__main__":
    main()
