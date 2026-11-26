"""V3b (plan 6): conditional residual of volclust_50 with volclust_50 removed from the matching, using the frozen rules of v3b_rules.json and the definitions of the
previous study (g = f - f0 >= 0.05 in the point estimate and in the 90% lower bounds of the seed layer (percentile and t4) and of the real layer), SA->SB and SB->SA,
with and without the matching eligibility. For the adjusted statistic the seasonality is re-estimated in every replicate. Output: results/v3b.json"""
from __future__ import annotations

import numpy as np

import vc_env as E
import vc_core as C
import vc_data as D
import dr_match as MT
from dr_b_rules import split_est
from vc_v3a import VARIANTS, estimation_table

G_THRESHOLD = 0.05


def matched(WA, Mn, fr, P, mset, eligibility):
    c = np.array(fr["c"])
    M, SE = P[[f"{mset}_{m}" for m in Mn]].values, P[[f"{mset}_{m}_se" for m in Mn]].values
    elig = (SE <= c / 2).all(axis=1) if eligibility else np.ones(len(M), bool)
    mask = np.zeros((len(WA), len(M)), bool)
    mask[:, elig] = MT.box_mask(WA[list(Mn)].values, M[elig], c)
    return mask, elig


def main():
    code = E.begin("vc_v3b")
    identified = E.load("v0a.json")["time_basis"]["identified_as_fixed_EST"]
    rules = E.load("v3b_rules.json")["variants"]
    T, W, se = estimation_table(identified)
    evals = ["volclust_raw"] + (["volclust_adj"] if identified else [])
    WA, WB = split_est(T)
    odd = (T["week"].values % 100) % 2 == 1
    base, _, ok = D.ddm_points()
    vals = D.ddm_run_values(base, ok)
    PT = D.a1_points()
    P = PT.set_index("pid").loc[base].reset_index()
    rng = np.random.default_rng(E.SEEDS["v3"])
    combos, out = [], {"_code": code, "g_threshold": G_THRESHOLD, "results": []}
    for vname, Mn in VARIANTS.items():
        for e in evals:
            fr = rules[vname][e]
            for mset, eset in (("SA", "SB"), ("SB", "SA")):
                for elig in (True, False):
                    mask, el = matched(WA, Mn, fr, P, mset, elig)
                    reach = mask.any(axis=1)
                    nw = int(reach.sum())
                    rec = {"variant": vname, "M": list(Mn), "E": e, "direction": f"{mset}->{eset}", "eligibility": elig, "n_points_matched_eligible": int(el.sum()),
                           "n_reachable": nw, "fraction_reachable": float(reach.mean()), "n_weeks_reachable": int(WA.loc[reach, "week"].nunique()), "noise_floor_computable": fr["noise_floor_computable"]}
                    cond = reach.mean() >= 0.05 and nw >= 50 and rec["n_weeks_reachable"] >= 20 and fr["noise_floor_computable"]
                    rec["residual_conditions_met"] = bool(cond)
                    out["results"].append(rec)
                    if cond:
                        runs = np.array(vals[eset])                                  # [P, 5]
                        combos.append({"rec": rec, "vname": vname, "e": e, "fr": fr, "mask": mask, "runs": runs})
    sig = lambda c: np.array(c["fr"]["sigma"])
    # point estimates and seed layer (tau, f0 fixed)
    for c in combos:
        EA = WA[[c["e"]]].values
        used = np.flatnonzero(c["mask"].any(axis=0))
        Rm, runs = c["mask"][:, used], c["runs"][used]
        Eth = runs.mean(axis=1)[:, None]
        ed, _ = MT.ddm_reference(Rm, Eth)
        wA = np.ones(len(EA))
        d, Dm, f, _ = MT.residual_stats(ed, EA, wA, sig(c), np.array(c["fr"]["tau"]))
        g = float(f[0] - c["fr"]["f0"][0])
        c.update(Rm=Rm, Eth=Eth, used=used)
        gp, gt = np.empty(E.B), np.empty(E.B)
        sem = runs.std(axis=1, ddof=1)[:, None] / np.sqrt(runs.shape[1])
        rows = np.arange(len(used))[:, None]
        for b in range(E.B):
            for arr, Es in ((gp, runs[rows, rng.integers(0, runs.shape[1], runs.shape)].mean(axis=1)[:, None]), (gt, Eth + sem * rng.standard_t(4, Eth.shape))):
                e2, _ = MT.ddm_reference(Rm, Es)
                arr[b] = MT.residual_stats(e2, EA, wA, sig(c), np.array(c["fr"]["tau"]))[2][0] - c["fr"]["f0"][0]
        c["rec"].update(D=float(Dm[0]), D_median=float(np.median(d[:, 0])), f=float(f[0]), f0=float(c["fr"]["f0"][0]), g=g, tau=float(c["fr"]["tau"][0]),
                        seed_layer_pct_g_90=C.interval(gp), seed_layer_t_g_90=C.interval(gt), mean_E_ddm=float(np.nanmean(ed)),
                        mean_E_real_reachable=float(EA[c["mask"].any(axis=1), 0].mean()), n_distinct_points_used=int(len(used)),
                        top1_point_share=float(Rm.sum(axis=0).max() / Rm.sum()))
        c["g_real"] = np.empty(E.B)
    # real layer: resample R-A weeks and R-B weeks; adjusted statistic re-estimated from the resampled weeks
    unit = None
    if se is not None:
        units = se.units
    else:
        units = np.unique(T["week"].values)
    unit = np.searchsorted(units, T["week"].values)
    uA = np.flatnonzero((units % 100) % 2 == 1)
    uB = np.flatnonzero((units % 100) % 2 == 0)
    masks_BA = {(c["vname"], c["e"]): MT.box_mask(WB[list(VARIANTS[c["vname"]])].values, WA[list(VARIANTS[c["vname"]])].values, np.array(c["fr"]["c"])) for c in combos}
    raw_all = T["volclust_raw"].values
    need_adj = any(c["e"] == "volclust_adj" for c in combos)
    for b in range(E.B if combos else 0):
        m = np.zeros(len(units))
        for u in (uA, uB):
            m[u] = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        mw = m[unit]
        mA, mB = mw[odd], mw[~odd]
        values = {"volclust_raw": raw_all}
        if need_adj:
            sel = mw > 0
            v = np.zeros(len(T))                                     # windows of weeks not drawn carry weight 0
            v[sel] = C.volclust(se.adjusted(W["R"], m, sel))
            values["volclust_adj"] = v
        cache = {}
        for c in combos:
            key = (c["vname"], c["e"])
            EA, EB = values[c["e"]][odd][:, None], values[c["e"]][~odd][:, None]
            if key not in cache:
                d, okb = MT.dn_real_real(EB, EA, masks_BA[key], mA, sig(c))
                tau = MT.tau_from(d, okb, mB)
                f0, _ = MT.crossfit_f0(d, okb, mB, WB["week"].values, MT.K_INNER, rng)
                cache[key] = (tau, f0)
            tau, f0 = cache[key]
            ed, _ = MT.ddm_reference(c["Rm"], c["Eth"])
            c["g_real"][b] = MT.residual_stats(ed, EA, mA, sig(c), tau)[2][0] - f0[0]
    for c in combos:
        r = c["rec"]
        r["real_layer_g_90"] = C.interval(c["g_real"])
        lows = (r["seed_layer_pct_g_90"][0], r["seed_layer_t_g_90"][0], r["real_layer_g_90"][0])
        r["exceeds_noise_floor"] = bool(r["g"] >= G_THRESHOLD and all(x > G_THRESHOLD for x in lows))
    agree = {}
    for r in out["results"]:
        agree.setdefault(f"{r['variant']}|{r['E']}|eligibility={r['eligibility']}", {})[r["direction"]] = r.get("exceeds_noise_floor")
    out["direction_agreement"] = {k: {**v, "usable_for_conclusion": bool(len(set(v.values())) == 1 and None not in v.values())} for k, v in agree.items()}
    E.save("v3b.json", out)
    for r in out["results"]:
        print(r["variant"], r["E"], r["direction"], r["eligibility"], r["n_reachable"], round(r.get("g", float("nan")), 3), r.get("exceeds_noise_floor"), flush=True)
    E.done()


if __name__ == "__main__":
    main()
