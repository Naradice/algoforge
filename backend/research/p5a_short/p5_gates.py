"""P5-A pre-analysis gates (plan v0.2 section 5). Stops (exit 1) when a gate fails. Output: OUT/gates.json"""
from __future__ import annotations

import inspect
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5_env as E  # noqa: E402
import p5_lib as K  # noqa: E402
import p4_lib as P  # noqa: E402
import p4_core as C4  # noqa: E402
import p1_lib as P1  # noqa: E402

RES = {}
INFO = {}


def gate(name, ok, detail=None):
    RES[name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name, flush=True)


class GuardCells(dict):
    """only ESTIMATION cells are readable; any other access raises (gate 4)."""

    def __getitem__(self, key):
        if key[0] != "estimation":
            raise KeyError(f"non-estimation access: {key}")
        return dict.__getitem__(self, key)


class FakeData:
    def __init__(self, cells):
        self.cells = cells


def main():
    os.makedirs(E.OUT, exist_ok=True)
    code = E.code_state()
    data = P.V.Data()
    # ---- gate 1: input identity ------------------------------------------------------------------------------------------------------
    import p2_env as E2
    sha = P1.sha256_file(E2.PARQUET)
    ok_sha = sha == "c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828"
    p3 = json.load(open(os.path.join(E.P3_RESULTS, "p3_report.json"), encoding="utf-8"))["metrics"]
    ddm = C4.DdmEval(data, "C0s")
    nbad, worst = 0, 0.0
    for si, s in enumerate(K.SEEDS):
        for per in ("validation", "final"):
            for h in K.HZ:
                c = data.cells[(per, h)]
                m = P.metrics(ddm.o[(per, h)][si], c["y"])
                ref = p3[f"C0s|{s}"][f"{per}|h{h}"]
                d = max(abs(m["crps"] - ref["crps"]), abs(m["cov90"] - ref["cov90"]))
                worst = max(worst, d)
                nbad += int(m["n"] != ref["n"] or d > 1e-12)
    p4 = json.load(open(os.path.join(E.P4_RESULTS, "analysis_B.json"), encoding="utf-8"))["cells"]
    worst4, nbad4 = 0.0, 0
    for per in ("validation", "final"):
        for h in K.HZ:
            y_fit, b_fit = C4.fit_set(data, h, "B-1")
            pred = P.Pred(C4.fit_E1("E1a", y_fit, b_fit))
            c = data.cells[(per, h)]
            m = P.metrics(pred.per_origin(c["b"], c["y"]), c["y"])
            ref = p4[f"B-1|E1a|{per}|h{h}"]["baseline"]
            d = max(abs(m["crps"] - ref["crps"]), abs(m["cov90"] - ref["cov90"]))
            worst4 = max(worst4, d)
            nbad4 += int(m["n"] != ref["n"] or d > 1e-12)
    gate("1 inputs: raw parquet hash; DDM(C0s) metrics reproduce P3 stored values; E1a (B-1) metrics reproduce P4 stored values (tol 1e-12)",
         ok_sha and nbad == 0 and nbad4 == 0, {"sha256": sha, "ddm_max_abs_diff": worst, "ddm_bad": nbad, "e1a_max_abs_diff": worst4, "e1a_bad": nbad4})

    # ---- gate 2a: exact implementation checks ------------------------------------------------------------------------------------------
    p = K.QA
    z = stats.norm.ppf(p)
    r = K.compare(2.0 * z, z)
    ok_r = np.nanmax(np.abs(r["r"][np.arange(len(p)) != K.I50] - 2.0)) < 1e-12 and np.nanmax(np.abs(r["shape"][np.isfinite(r["shape"])] - 1.0)) < 1e-12
    r2 = K.compare(z + 0.5, z)
    ok_loc = abs(r2["loc"] - 0.5 / (z[K.I75] - z[K.I25])) < 1e-12
    # asymmetric (two-piece) distribution: upper scale 1.5, lower scale 0.5 relative to the reference
    qd = np.where(p > 0.5, 1.5 * z, 0.5 * z)
    r3 = K.compare(qd, z)
    up, lo = p > 0.5, p < 0.5
    ok_asym = np.allclose(r3["r"][up], 1.5) and np.allclose(r3["r"][lo], 0.5) and np.allclose(r3["shape"][p > 0.75], 1.0) and np.allclose(r3["shape"][p < 0.25], 1.0)
    # undefined handling: constant reference -> everything NaN (never 0 or 1); tie at one quantile -> only that ratio undefined
    qe = z.copy()
    qe[np.searchsorted(p, 0.90)] = qe[K.I50]
    r4 = K.compare(2.0 * z, qe)
    k90 = int(np.searchsorted(p, 0.90))
    ok_undef = bool(np.isnan(r4["r"][k90]) and np.isfinite(r4["r"][k90 - 1]) and np.isnan(K.compare(z, np.zeros_like(z))["r"]).all()
                    and np.isnan(r4["shape"][k90]))
    rng = np.random.default_rng(3)
    ens = np.sort(rng.standard_normal((20, 1000)), axis=1)
    sc1 = K.scale_about_median(ens, 1.0)
    c = data.cells[("final", 4)]
    y_ = c["y"][:2000]
    b_ = c["b"][:2000]
    crps_a = P.fair_crps_list([ens[k] for k in range(20)], b_, y_)
    crps_b = P.fair_crps_list([sc1[k] for k in range(20)], b_, y_)
    ok_t1 = bool(np.array_equal(sc1, ens) and np.abs(crps_a - crps_b).max() < 1e-12)
    sc2 = K.scale_about_median(ens, 2.0)
    med = np.median(ens, axis=1, keepdims=True)
    ok_t1b = bool(np.allclose(np.median(sc2, axis=1, keepdims=True), med, atol=1e-12) and np.all(np.diff(sc2, axis=1) >= 0))
    gate("2a quantile-ratio / shape / location implementation on known distributions (symmetric, asymmetric, undefined cases kept as NaN); T1 with s=1 returns "
         "the original ensemble exactly (CRPS equal to 1e-12); T1 scaling fixes the median and keeps the order", bool(ok_r and ok_loc and ok_asym and ok_undef and ok_t1 and ok_t1b),
         {"scale_ratio": bool(ok_r), "location": bool(ok_loc), "asymmetric": bool(ok_asym), "undefined": ok_undef, "T1_s1_exact": ok_t1, "T1_scaling": ok_t1b})

    # ---- gate 2b: identity PIT transform; same-representation CRPS --------------------------------------------------------------------
    ens_s = {s: P.V.load_ens("C0s", s) for s in K.SEEDS}
    e0 = ens_s[1]
    ident = lambda u: u  # noqa: E731
    same = all(np.array_equal(K.recal_grid(e0[k, hi], ident), K.recal_grid(e0[k, hi], None)) for k in range(20) for hi in range(4))
    mono = all(np.all(np.diff(K.recal_grid(e0[k, hi], None)) >= 0) for k in range(20) for hi in range(4))
    # the grid is a faithful quantile representation: CDF of the identity-transformed grid at the grid points is p_i (to the grid resolution)
    dev = 0.0
    for k in (0, 7, 19):
        for hi in range(4):
            g = K.recal_grid(e0[k, hi], None)
            dev = max(dev, float(np.abs(np.interp(g, e0[k, hi], (np.arange(1000) + 0.5) / 1000) - K.grid_p()).max()))
    # CRPS of the grid representation equals a brute-force fair CRPS computed on the same grid (same numerical representation)
    brute = []
    gk = [K.recal_grid(e0[k, 1], None) for k in range(20)]
    cc = data.cells[("final", 4)]
    idx = np.flatnonzero(cc["b"] == 5)[:20]
    fast = P.fair_crps_list(gk, cc["b"][idx], cc["y"][idx])
    brute = np.array([P.fair_crps_brute(gk[5], cc["y"][i])[0] for i in idx])
    ok_same_repr = float(np.abs(fast - brute).max()) < 1e-12
    gate("2b identity PIT transform (G(u)=u) reproduces the T0g quantile grid exactly; grid is monotone; CRPS equals brute force on the same representation (1e-12)",
         bool(same and mono and ok_same_repr), {"identity_exact": bool(same), "grid_monotone": bool(mono), "same_repr_crps_max_diff": float(np.abs(fast - brute).max()),
                                                "cdf_deviation_at_grid_points_max": dev})

    # ---- gate 2c: approximation error of the grid representation (recorded; exceeding the tolerance is not a failure) --------------------------
    ev0 = K.EnsEval(data, ens_s)
    ev0g = K.EnsEval(data, K.apply_transform("T0g", ens_s, None))
    errs = {}
    for per in K.PERIODS:
        for h in K.HZ:
            a = P.avg_metrics([P.metrics(o, data.cells[(per, h)]["y"]) for o in ev0.o[(per, h)]])["crps"]
            b = P.avg_metrics([P.metrics(o, data.cells[(per, h)]["y"]) for o in ev0g.o[(per, h)]])["crps"]
            errs[f"{per}|h{h}"] = abs(b - a) / a
    over = {k: v for k, v in errs.items() if v > K.APPROX_TOL}
    gate("2c grid (T0g) vs sample (T0) CRPS relative error recorded against the fixed tolerance 1e-3 (exceedances are reported, not failures)", True,
         {"tolerance": K.APPROX_TOL, "max_rel_error": max(errs.values()), "n_cells": len(errs), "n_over_tolerance": len(over), "over": over, "all": errs})

    # ---- gate 3: candidate recovery -------------------------------------------------------------------------------------------------------
    mp = K.load_mapping()
    import p3_sim as S3
    import f1_ddm as F1
    src_ok = ("cands[attempts % len(cands)]" in inspect.getsource(S3)) and ("cands[attempts % len(cands)]" in inspect.getsource(F1))
    cands_ok = all(len(b["candidates"]) == K.N_CAND for b in mp["buckets"])
    bad, fr_min, fr_max, first_ok = 0, 1.0, 0.0, True
    for s in K.SEEDS:
        ens, att = K.load_raw_ens("C0s", s)
        info = json.load(open(os.path.join(E.ENS_DIR, f"C0s_seed{s}.json")))["info"]
        for b in range(20):
            a = att[b]
            bad += int(not (np.all(np.diff(a) > 0) and a.min() >= 0 and a.max() < int(info[str(b)]["attempts"])))
            fr = np.bincount(a % K.N_CAND, minlength=K.N_CAND) / len(a)
            fr_min, fr_max = min(fr_min, fr.min()), max(fr_max, fr.max())
            if info[str(b)]["attempts"] == info[str(b)]["successes"]:
                first_ok &= bool(np.array_equal(a, np.arange(len(a))))
            rec = [tuple(mp["buckets"][b]["candidates"][k % K.N_CAND]) for k in a[:150]]                 # recovered (delta, wma) by the F1 rule
            exp = [tuple(mp["buckets"][b]["candidates"][i % K.N_CAND]) for i in range(150)] if info[str(b)]["attempts"] == info[str(b)]["successes"] else rec
            first_ok &= rec == exp
    gate("3 candidate recovery: the generation rule 'cands[attempt % len(cands)]' appears in F1 and P3 code; 8 candidates per bucket; attempt ids increasing within "
         "the recorded attempts; the first 150 recovered candidates equal the rule; per-candidate path share within [0.10, 0.15]", bool(src_ok and cands_ok and bad == 0 and first_ok and fr_min > 0.10 and fr_max < 0.15),
         {"rule_in_sources": src_ok, "8_candidates_per_bucket": cands_ok, "n_bad_attempt_arrays": bad, "first150_match": bool(first_ok),
          "share_min": float(fr_min), "share_max": float(fr_max)})

    # ---- gate 4: estimation-only fitting ----------------------------------------------------------------------------------------------------
    cells_g = GuardCells({k: v for k, v in data.cells.items()})
    f_guard = K.fit_all(FakeData(cells_g), ens_s)                       # raises if any non-estimation cell is read
    rnd = np.random.default_rng(11)
    cells_r = {}
    for k, v in data.cells.items():
        if k[0] == "estimation":
            cells_r[k] = v
        else:
            cells_r[k] = {**v, "y": rnd.standard_normal(len(v["y"])), "b": rnd.integers(0, 20, len(v["b"]))}
    f_rand = K.fit_all(FakeData(cells_r), ens_s)
    f_real = K.fit_all(data, ens_s)
    same_fit = all(f_guard["s1"][h] == f_real["s1"][h] == f_rand["s1"][h] and np.array_equal(f_guard["s2"][h], f_real["s2"][h]) and np.array_equal(f_rand["s2"][h], f_real["s2"][h])
                   and np.array_equal(f_guard["pit_pooled"][h], f_real["pit_pooled"][h]) and np.array_equal(f_rand["pit_pooled"][h], f_real["pit_pooled"][h])
                   and all(np.array_equal(a, b) for a, b in zip(f_guard["pit_bucket"][h], f_real["pit_bucket"][h])) for h in K.HZ)
    # candidate selection uses the same estimation-only rule
    raw = {s: K.load_raw_ens("C0s", s)[0] for s in K.SEEDS}
    att = {s: K.load_raw_ens("C0s", s)[1] for s in K.SEEDS}
    sel_g = K.select_subsets(FakeData(cells_g), raw, att, K.SEEDS, 1)
    sel_r = K.select_subsets(FakeData(cells_r), raw, att, K.SEEDS, 1)
    sel_real = K.select_subsets(data, raw, att, K.SEEDS, 1)
    same_sel = all(a["single"] == b["single"] == c_["single"] and a["pair"] == b["pair"] == c_["pair"] for a, b, c_ in zip(sel_g, sel_r, sel_real))
    gate("4 estimation-only fitting: T1-T4 coefficients and the candidate selection do not read non-estimation cells (guard raises) and are unchanged when validation/final "
         "data are replaced by random numbers", bool(same_fit and same_sel), {"fits_same": bool(same_fit), "selection_same": bool(same_sel),
                                                                              "s1": f_real["s1"], "s2_range": {h: [float(f_real["s2"][h].min()), float(f_real["s2"][h].max())] for h in K.HZ}})
    # ---- gate 5: monotone CDF after T3/T4 ----------------------------------------------------------------------------------------------------
    ok5, n5 = True, 0
    for kind in ("T3", "T4"):
        tr = K.apply_transform(kind, ens_s, f_real)
        for s in K.SEEDS:
            ok5 &= bool(np.all(np.diff(tr[s], axis=2) >= 0))
            n5 += tr[s].shape[0] * tr[s].shape[1]
    ginv_ok = all(np.all(np.diff(K.make_ginv(f_real["pit_pooled"][h])(K.grid_p())) >= 0) for h in K.HZ)
    gate("5 T3/T4: transformed quantile functions (and G^-1) are monotone, i.e. the transformed CDFs are monotone", bool(ok5 and ginv_ok), {"n_rows_checked": n5})
    json.dump({"_code": code, "all_pass": all(v["pass"] for v in RES.values()), "gates": RES}, open(os.path.join(E.OUT, "gates.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("ALL PASS" if all(v["pass"] for v in RES.values()) else "GATE FAILURE", flush=True)
    sys.exit(0 if all(v["pass"] for v in RES.values()) else 1)


if __name__ == "__main__":
    main()
