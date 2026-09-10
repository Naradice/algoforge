"""Predictive-usefulness evaluation: does Step 5's calibration improvement translate into an
actually-better probabilistic forecast of real USDJPY's future, versus the baseline DDM?

Three-way comparison (Baseline DDM / Step 5 DDM tau=300 / Real out-of-sample), same forward
evaluation as before plus a proper scoring rule:

  A. Future volatility correlation  -- Spearman(quantile, mean windowed_vol) for each DDM variant
     vs. the real out-of-sample value (the real value is a fixed property of the delta predictor
     + real data; only the DDM columns move).
  B. Cumulative-return dispersion   -- pooled std of cum_return, DDM/Real ratio per horizon.
  C. Interval coverage              -- pooled empirical coverage at 50/80/90/95% nominal.
  D. Tail probability               -- pooled predicted vs realized P(|r| > x) at real-derived
                                       thresholds.
  E. CRPS (the new bit)             -- continuous ranked probability score of each DDM ensemble's
     predictive cum-return distribution against the realized real cum-returns, per quantile and
     pooled. CRPS rewards calibration AND sharpness, so it answers "is this a better forecast",
     not just "is the distribution shape right". Also scored: a regime-AGNOSTIC reference (the
     Step 5 ensemble pooled across all quantiles) -- if regime-conditioning doesn't beat this,
     the quantile signal isn't buying predictive power regardless of calibration.

Cheap: no phase1 rerun. Loads the two existing horizon-stats caches and computes the real side
once (single model load).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ddm_horizon_analysis import HORIZONS_WINDOWS, _unflatten_ddm_stats, get_real_horizon_stats

RESULTS = {}

BASELINE_CACHE = "ddm_horizon_stats.npz"
STEP5_CACHE = "ddm_horizon_stats_shockP0.002_S0.3_tau300.0_floor0.0.npz"
N_QUANTILES = 20
NOMINAL_LEVELS = [50, 80, 90, 95]


def crps_empirical(samples: np.ndarray, obs: np.ndarray) -> float:
    """Mean CRPS of the empirical predictive distribution `samples` against each value in `obs`.
    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|. The second term is estimated once (it doesn't depend
    on y) via the sorted-sample identity E|X-X'| = (2/m^2) sum_i (2i - m - 1) x_(i)."""
    s = np.sort(samples)
    m = len(s)
    i = np.arange(1, m + 1)
    e_xx = (2.0 / m**2) * np.sum((2 * i - m - 1) * s)
    e_xy = np.mean(np.abs(samples[:, None] - obs[None, :]))
    return float(e_xy - 0.5 * e_xx)


def main():
    baseline = _unflatten_ddm_stats(np.load(BASELINE_CACHE))
    step5 = _unflatten_ddm_stats(np.load(STEP5_CACHE))
    print(f"Loaded baseline={BASELINE_CACHE}, step5={STEP5_CACHE}")

    print("\nComputing real out-of-sample horizon stats (single model load)...")
    real = get_real_horizon_stats()

    variants = {"baseline": baseline, "step5_tau300": step5}

    # ---- A. Future volatility correlation ----
    print("\n=== A. Future volatility correlation: Spearman(quantile, mean windowed_vol) ===")
    print(f"{'h':>3} {'baseline DDM':>14} {'step5 DDM':>12} {'REAL':>10}")
    qs = list(range(N_QUANTILES))
    for h in HORIZONS_WINDOWS:
        row = {}
        for name, dd in variants.items():
            vals = [np.nanmean(dd[q][h]["windowed_vol"]) for q in qs]
            row[name], _ = spearmanr(qs, vals)
        real_vals = [np.nanmean(real[q][h]["windowed_vol"]) for q in qs]
        r_real, _ = spearmanr(qs, real_vals)
        print(f"{h:>3} {row['baseline']:>14.4f} {row['step5_tau300']:>12.4f} {r_real:>10.4f}")

    # ---- B. Cumulative-return dispersion ----
    print("\n=== B. Cumulative-return dispersion: pooled std, and DDM/Real ratio ===")
    print(f"{'h':>3} {'base std':>10} {'base/real':>10} {'step5 std':>10} {'step5/real':>11} {'real std':>10}")
    for h in HORIZONS_WINDOWS:
        real_all = np.concatenate([real[q][h]["cum_return"] for q in qs])
        real_std = np.nanstd(real_all)
        out = []
        for name, dd in variants.items():
            ddm_all = np.concatenate([dd[q][h]["cum_return"] for q in qs])
            out.append((np.nanstd(ddm_all), np.nanstd(ddm_all) / real_std))
        print(f"{h:>3} {out[0][0]:>10.6f} {out[0][1]:>10.3f} {out[1][0]:>10.6f} {out[1][1]:>11.3f} {real_std:>10.6f}")

    # ---- C. Interval coverage (pooled) ----
    print("\n=== C. Interval coverage (pooled across quantiles), empirical % vs nominal ===")
    for h in HORIZONS_WINDOWS:
        print(f"  h={h}:")
        print(f"    {'nominal':>8} {'baseline':>10} {'step5':>10}")
        for lvl in NOMINAL_LEVELS:
            lo_p, hi_p = (100 - lvl) / 2, 100 - (100 - lvl) / 2
            cov = {}
            for name, dd in variants.items():
                hits = []
                for q in qs:
                    d, r = dd[q][h]["cum_return"], real[q][h]["cum_return"]
                    if len(d) == 0 or len(r) == 0:
                        continue
                    lo, hi = np.percentile(d, [lo_p, hi_p])
                    hits.extend(((r >= lo) & (r <= hi)).tolist())
                cov[name] = 100 * np.mean(hits)
            print(f"    {lvl:>8} {cov['baseline']:>10.1f} {cov['step5_tau300']:>10.1f}")

    # ---- D. Tail probability (pooled) ----
    print("\n=== D. Tail probability P(|cum_return| > x), pooled: predicted vs realized ===")
    for h in HORIZONS_WINDOWS:
        real_all = np.concatenate([real[q][h]["cum_return"] for q in qs])
        thr = np.percentile(np.abs(real_all), [50, 70, 85, 95])
        realized = [100 * np.mean(np.abs(real_all) > x) for x in thr]
        pred = {}
        for name, dd in variants.items():
            ddm_all = np.concatenate([dd[q][h]["cum_return"] for q in qs])
            pred[name] = [100 * np.mean(np.abs(ddm_all) > x) for x in thr]
        print(f"  h={h}: thresholds={[f'{x:.5f}' for x in thr]}")
        print(f"    {'pctile':>8} {'realized':>10} {'baseline':>10} {'step5':>10}")
        for i, p in enumerate([50, 70, 85, 95]):
            print(f"    {p:>8} {realized[i]:>10.1f} {pred['baseline'][i]:>10.1f} {pred['step5_tau300'][i]:>10.1f}")

    # ---- E. CRPS ----
    print("\n=== E. CRPS (lower = better forecast). regime_agnostic = step5 ensemble pooled over all q ===")
    for h in HORIZONS_WINDOWS:
        step5_pooled = np.concatenate([step5[q][h]["cum_return"] for q in qs])
        tot = {"baseline": [], "step5_tau300": [], "regime_agnostic": []}
        weights = []
        per_q = []
        for q in qs:
            obs = real[q][h]["cum_return"]
            if len(obs) == 0:
                continue
            c_base = crps_empirical(baseline[q][h]["cum_return"], obs)
            c_s5 = crps_empirical(step5[q][h]["cum_return"], obs)
            c_agn = crps_empirical(step5_pooled, obs)
            tot["baseline"].append(c_base * len(obs))
            tot["step5_tau300"].append(c_s5 * len(obs))
            tot["regime_agnostic"].append(c_agn * len(obs))
            weights.append(len(obs))
            per_q.append((q, len(obs), c_base, c_s5, c_agn))
        W = sum(weights)
        cb, cs, ca = (sum(tot[k]) / W for k in ("baseline", "step5_tau300", "regime_agnostic"))
        print(f"  h={h:>3}: pooled CRPS  baseline={cb:.6f}  step5={cs:.6f}  regime_agnostic={ca:.6f}  "
              f"| step5/baseline={cs/cb:.3f}  step5 skill vs agnostic={1 - cs/ca:+.3f}")
        RESULTS.setdefault("crps_pooled", {})[str(h)] = {
            "baseline": cb, "step5_tau300": cs, "regime_agnostic": ca,
            "step5_over_baseline": cs / cb, "step5_skill_vs_agnostic": 1 - cs / ca,
        }

    # ---- F. Per-quantile, h=1 and h=24 ----
    print("\n=== F. Per-quantile CRPS + 90% coverage + dispersion ratio (h=1 / h=24) ===")
    for h in [1, 24]:
        step5_pooled = np.concatenate([step5[q][h]["cum_return"] for q in qs])
        real_all = np.concatenate([real[q][h]["cum_return"] for q in qs])
        real_std = np.nanstd(real_all)
        print(f"\n  h={h}:")
        print(f"  {'q':>3} {'n':>5} {'CRPS base':>10} {'CRPS step5':>11} {'s5/base':>8} "
              f"{'cov90 base':>11} {'cov90 s5':>9} {'disp base':>10} {'disp s5':>9}")
        for q in qs:
            obs = real[q][h]["cum_return"]
            if len(obs) == 0:
                continue
            cb = crps_empirical(baseline[q][h]["cum_return"], obs)
            cs = crps_empirical(step5[q][h]["cum_return"], obs)
            covs = {}
            for name, dd in (("b", baseline), ("s", step5)):
                lo, hi = np.percentile(dd[q][h]["cum_return"], [5, 95])
                covs[name] = 100 * np.mean((obs >= lo) & (obs <= hi))
            db = np.nanstd(baseline[q][h]["cum_return"]) / real_std
            ds = np.nanstd(step5[q][h]["cum_return"]) / real_std
            print(f"  {q:>3} {len(obs):>5} {cb:>10.6f} {cs:>11.6f} {cs/cb:>8.2f} "
                  f"{covs['b']:>11.1f} {covs['s']:>9.1f} {db:>10.2f} {ds:>9.2f}")
            RESULTS.setdefault("per_quantile", {}).setdefault(str(h), []).append({
                "q": q, "n": len(obs), "crps_baseline": cb, "crps_step5": cs, "s5_over_base": cs / cb,
                "cov90_baseline": covs["b"], "cov90_step5": covs["s"],
                "disp_baseline": db, "disp_step5": ds,
            })


if __name__ == "__main__":
    main()
    out = Path(__file__).resolve().parent / "results" / "ddm_step5_predictive_eval.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nWrote {out}")
