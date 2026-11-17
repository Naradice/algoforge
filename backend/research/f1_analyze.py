"""F1 (#889 v0.3): model selection on the validation period and the final-period analysis.

  python research/f1_analyze.py select <workdir>      -> selection.json (validation only)
  python research/f1_analyze.py final  <workdir>      -> final_result.json (reads selection.json)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import f1_common as fc  # noqa: E402
import f1_guard as fg  # noqa: E402

SELECT_POOL = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7N", "B7T"]
TIE_ORDER = ["B1", "B2", "B3", "B4", "B7N", "B5", "B6", "B7T", "B0"]     # fewer free parameters first
TIE_TOL = 0.005
MARGIN = 0.02
N_BOOT = 10_000
DDM_SEEDS = ("DDM1", "DDM2", "DDM3")


def load_scores(workdir, period):
    z = np.load(os.path.join(workdir, f"scores_{period}.npz"))
    return {k: z[k] for k in z.files}


PIT_REPEATS = 100


def pit_randomisation_spread(lo, w, seed=0):
    """Repeat the tie randomisation PIT = lo + U*w PIT_REPEATS times; return mean/sd of the KS distance to uniform and the
    sd of each of the 20 histogram bins. With no ties the spread is exactly 0."""
    rng = np.random.default_rng(seed)
    ks, hists = [], []
    for _ in range(PIT_REPEATS):
        pit = lo + rng.random(len(lo)) * w
        ks.append(stats.kstest(pit, "uniform").statistic)
        hists.append(np.histogram(pit, bins=20, range=(0, 1))[0] / len(pit))
    return {"ks_mean": float(np.mean(ks)), "ks_sd": float(np.std(ks)), "hist_bin_sd_max": float(np.std(hists, axis=0).max()),
            "tie_share": float(np.mean(w > 0))}


def assert_same_origins(sc, models, h):
    ref = sc[f"{models[0]}|h{h}|j"]
    for m in models[1:]:
        if not np.array_equal(ref, sc[f"{m}|h{h}|j"]):
            raise RuntimeError(f"origin sets differ between {models[0]} and {m} at h={h}")


def mean_crps(sc, model, h):
    return float(sc[f"{model}|h{h}|crps"].mean())


def stage_select(workdir):
    sc = load_scores(workdir, "validation")
    for h in fc.HORIZONS:
        assert_same_origins(sc, SELECT_POOL + ["B8", "B4s"], h)
    sel = {}
    table = {}
    for h in fc.HORIZONS:
        vals = {m: mean_crps(sc, m, h) for m in SELECT_POOL}
        table[h] = vals
        best = min(vals.values())
        near = [m for m in TIE_ORDER if vals[m] <= best * (1 + TIE_TOL)]
        sel[h] = near[0]
    json.dump({"selected": {str(h): m for h, m in sel.items()}, "validation_crps": {str(h): v for h, v in table.items()}},
              open(os.path.join(workdir, "selection.json"), "w"), indent=1)
    print("validation mean fair CRPS (x1e4):")
    for m in SELECT_POOL + ["B8", "B4s"]:
        print(f"  {m:4s}", " ".join(f"{mean_crps(sc, m, h) * 1e4:8.4f}" for h in fc.HORIZONS))
    print("selected per horizon:", sel)


def ddm_crps_per_origin(sc, h):
    arr = np.stack([sc[f"{d}|h{h}|crps"] for d in DDM_SEEDS])
    return arr.mean(axis=0)


def newey_west_dm(d, lag):
    n = len(d)
    dm = d - d.mean()
    gamma0 = np.dot(dm, dm) / n
    var = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * np.dot(dm[k:], dm[:-k]) / n
    se = np.sqrt(var / n)
    t = d.mean() / se
    return float(t), float(2 * (1 - stats.norm.cdf(abs(t))))


def stage_final(workdir):
    sel = json.load(open(os.path.join(workdir, "selection.json")))["selected"]
    sc = load_scores(workdir, "final")
    ctx_idx = {h: sc[f"B1|h{h}|j"] for h in fc.HORIZONS}
    for h in fc.HORIZONS:                       # every compared model must be scored on exactly the same origins
        assert_same_origins(sc, ["B1", "B0", "B2", "B3", "B4", "B5", "B6", "B7N", "B7T", "B8", "B4s", *DDM_SEEDS], h)
    s = fc.load_series()
    out = {"selected": sel, "per_horizon": {}, "models_final_crps": {}}
    # week id per origin and the common week universe
    week = {}
    for h in fc.HORIZONS:
        ts = s.t_time[ctx_idx[h]]
        iso = ts.isocalendar()
        week[h] = (iso.year.values.astype(int) * 100 + iso.week.values.astype(int))
    all_weeks = np.unique(np.concatenate(list(week.values())))
    rng = np.random.default_rng(20261117)
    # per-horizon per-week sums for fast joint bootstrap
    wsum = {}
    for h in fc.HORIZONS:
        stat = sc[f"{sel[str(h)]}|h{h}|crps"]
        ddm = ddm_crps_per_origin(sc, h)
        wk = week[h]
        sum_d = pd.Series(ddm).groupby(wk).sum().reindex(all_weeks, fill_value=0.0).values
        sum_s = pd.Series(stat).groupby(wk).sum().reindex(all_weeks, fill_value=0.0).values
        cnt = pd.Series(np.ones(len(wk))).groupby(wk).sum().reindex(all_weeks, fill_value=0.0).values
        wsum[h] = (sum_d, sum_s, cnt)
    point = {h: wsum[h][0].sum() / wsum[h][1].sum() - 1 for h in fc.HORIZONS}
    draws = np.empty((N_BOOT, len(fc.HORIZONS)))
    for b in range(N_BOOT):
        pick = rng.integers(0, len(all_weeks), len(all_weeks))
        for i, h in enumerate(fc.HORIZONS):
            sd, ss, _ = wsum[h]
            draws[b, i] = sd[pick].sum() / ss[pick].sum() - 1
    se = draws.std(axis=0)
    pt = np.array([point[h] for h in fc.HORIZONS])
    crit = float(np.quantile(np.max(np.abs(draws - pt) / se, axis=1), 0.95))
    marg = [np.quantile(draws[:, i], [0.025, 0.975]) for i in range(len(fc.HORIZONS))]
    pvals = []
    for i, h in enumerate(fc.HORIZONS):
        d = ddm_crps_per_origin(sc, h) - sc[f"{sel[str(h)]}|h{h}|crps"]
        t, p = newey_west_dm(d, h + 23)
        pvals.append(p)
    order = np.argsort(pvals)
    holm = np.empty(len(pvals))
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[idx])
        holm[idx] = min(1.0, run)
    for i, h in enumerate(fc.HORIZONS):
        lo, hi = pt[i] - crit * se[i], pt[i] + crit * se[i]
        if hi < -MARGIN:
            label = "DDM better"
        elif lo > MARGIN:
            label = "statistical model better"
        elif lo >= -MARGIN and hi <= MARGIN:
            label = "equivalent"
        else:
            label = "inconclusive"
        seeds = [float(sc[f"{d}|h{h}|crps"].mean()) for d in DDM_SEEDS]
        out["per_horizon"][str(h)] = {
            "stat_model": sel[str(h)], "n_origins": int(len(ctx_idx[h])),
            "crps_ddm_mean_of_seeds": float(np.mean(seeds)), "crps_ddm_seeds": seeds,
            "ddm_seed_sd_rel": float(np.std(seeds) / np.mean(seeds)),
            "crps_stat": float(sc[f"{sel[str(h)]}|h{h}|crps"].mean()),
            "delta": float(pt[i]), "se": float(se[i]), "simultaneous_band": [float(lo), float(hi)],
            "marginal_ci95": [float(marg[i][0]), float(marg[i][1])], "dm_p": float(pvals[i]), "dm_p_holm": float(holm[i]),
            "label": label,
        }
    out["simultaneous_critical_value"] = crit
    # all models: final CRPS, calibration, tails, exploratory comparison with DDM
    models = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7N", "B7T", "B8", "B4s"]
    for m in models:
        row = {}
        for h in fc.HORIZONS:
            x = sc[f"{m}|h{h}|crps"]
            pit = sc[f"{m}|h{h}|pit"]
            hist = np.histogram(pit, bins=20, range=(0, 1))[0] / len(pit)
            ks = float(stats.kstest(pit, "uniform").statistic)
            row[str(h)] = {"crps": float(x.mean()), "crps_plain": float(sc[f"{m}|h{h}|crps_plain"].mean()),
                           "delta_ddm_vs_model": float(ddm_crps_per_origin(sc, h).mean() / x.mean() - 1),
                           "hit50": float(sc[f"{m}|h{h}|hit50"].mean()), "hit80": float(sc[f"{m}|h{h}|hit80"].mean()),
                           "hit90": float(sc[f"{m}|h{h}|hit90"].mean()), "hit95": float(sc[f"{m}|h{h}|hit95"].mean()),
                           "ks_pit": ks, "pit_hist": hist.tolist(),
                           "pit_mc": pit_randomisation_spread(sc[f"{m}|h{h}|pit_lo"], sc[f"{m}|h{h}|pit_w"]),
                           "ql": {q: float(sc[f"{m}|h{h}|ql{q}"].mean()) for q in fc.TAIL_LEVELS}}
        out["models_final_crps"][m] = row
    ddm_row = {}
    for h in fc.HORIZONS:
        c = ddm_crps_per_origin(sc, h)
        pits = np.concatenate([sc[f"{d}|h{h}|pit"] for d in DDM_SEEDS])
        ddm_row[str(h)] = {"crps": float(c.mean()), "hit50": float(np.mean([sc[f'{d}|h{h}|hit50'].mean() for d in DDM_SEEDS])),
                           "hit80": float(np.mean([sc[f'{d}|h{h}|hit80'].mean() for d in DDM_SEEDS])),
                           "hit90": float(np.mean([sc[f'{d}|h{h}|hit90'].mean() for d in DDM_SEEDS])),
                           "hit95": float(np.mean([sc[f'{d}|h{h}|hit95'].mean() for d in DDM_SEEDS])),
                           "ks_pit": float(stats.kstest(pits, "uniform").statistic),
                           "pit_hist": (np.histogram(pits, bins=20, range=(0, 1))[0] / len(pits)).tolist(),
                           "pit_mc": pit_randomisation_spread(np.concatenate([sc[f"{d}|h{h}|pit_lo"] for d in DDM_SEEDS]),
                                                              np.concatenate([sc[f"{d}|h{h}|pit_w"] for d in DDM_SEEDS])),
                           "ql": {q: float(np.mean([sc[f'{d}|h{h}|ql{q}'].mean() for d in DDM_SEEDS])) for q in fc.TAIL_LEVELS}}
    out["models_final_crps"]["DDM"] = ddm_row
    json.dump(out, open(os.path.join(workdir, "final_result.json"), "w"), indent=1)
    print("final period (partially known holdout): Delta_h = (CRPS_DDM - CRPS_Stat*) / CRPS_Stat*   (negative = DDM better)")
    for h in fc.HORIZONS:
        r = out["per_horizon"][str(h)]
        print(f"h={h:2d} stat={r['stat_model']:4s} n={r['n_origins']} CRPS ddm={r['crps_ddm_mean_of_seeds']*1e4:.4f} stat={r['crps_stat']*1e4:.4f} "
              f"Delta={r['delta']*100:+.2f}% simultaneous band [{r['simultaneous_band'][0]*100:+.2f}, {r['simultaneous_band'][1]*100:+.2f}]% "
              f"marginal [{r['marginal_ci95'][0]*100:+.2f}, {r['marginal_ci95'][1]*100:+.2f}] DM p={r['dm_p']:.3g} holm={r['dm_p_holm']:.3g} "
              f"ddm-seed-sd={r['ddm_seed_sd_rel']*100:.2f}% -> {r['label']}")
    print("critical value (simultaneous 95%):", round(crit, 3))


if __name__ == "__main__":
    fg.start_run(sys.argv[2], f"analyze_{sys.argv[1]}",
                 inputs=[os.path.join(sys.argv[2], f) for f in ("scores_validation.npz", "scores_final.npz", "selection.json")])
    {"select": stage_select, "final": stage_final}[sys.argv[1]](sys.argv[2])
