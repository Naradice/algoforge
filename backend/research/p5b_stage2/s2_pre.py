"""P5-B stage 2 pre-regeneration gates (plan v0.2 section 4, gates 1 and 4). Stops (exit 1) when a gate fails. Output: OUT/pre_gates.json"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2_env as E  # noqa: E402
import s2_lib as S2  # noqa: E402
import p5b_lib as L5  # noqa: E402
import p5_lib as K  # noqa: E402
import p4_lib as P  # noqa: E402
import p1_lib as P1  # noqa: E402
from data.characteristics import _acf  # noqa: E402

RES = {}


def gate(name, ok, detail=None):
    RES[name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def git_show(commit, rel):
    return subprocess.check_output(["git", "-C", E.REPO, "show", f"{commit}:{rel}"]).replace(b"\r\n", b"\n")


def main():
    os.makedirs(E.OUT, exist_ok=True)
    code = E.code_state()
    mp = S2.load_mapping()
    # ---- gate 1: input and code identity --------------------------------------------------------------------------------------------------
    import p2_env as E2
    sha = P1.sha256_file(E2.PARQUET)
    ok_sha = sha == "c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828"
    s1 = json.load(open(os.path.join(E.STAGE1_RESULTS, "analysis.json"), encoding="utf-8"))
    worst = 0.0
    for cond in E.CONDS:
        raw, att = L5.load_cond(cond)
        for q in E.BUCKETS:
            for h in L5.HZ:
                st = L5.cand_stats(raw, att, q, h)
                ref = s1["C"][str(q)][f"h{h}"][cond]
                worst = max(worst, abs(st["mix_sd"] - ref["mix_sd"]), abs(st["mix_w90"] - ref["mix_w90"]))
    gate("1a raw parquet hash; stage-1 stored values (mixture sd and 90% width of C0s and C3 for buckets 0, 8, 15, 19) reproduce to 1e-12", ok_sha and worst < 1e-12,
         {"sha256": sha, "max_abs_diff": worst})
    files = {"backend/research/p3_burnin/p3_sim.py": E.P3_COMMIT, "backend/data/collectors/ddm_simulator.py": E.F1_COMMIT, "backend/research/f1_ddm.py": E.F1_COMMIT}
    ident = {}
    for rel, commit in files.items():
        cur = open(os.path.join(E.REPO, rel), "rb").read().replace(b"\r\n", b"\n")
        ident[rel] = bool(cur == git_show(commit, rel))
    gate("1b generation code (p3_sim.py, ddm_simulator.py, f1_ddm.py) in the working tree equals the commits used for the stored ensembles (line endings ignored)", all(ident.values()), ident)
    cells_ok, detail = True, {}
    for cond in E.CONDS:
        info = json.load(open(os.path.join(E.ENS_DIR, f"{cond}_seed{E.SEED_ID}.json")))["info"]
        att = np.load(os.path.join(E.ENS_DIR, f"{cond}_seed{E.SEED_ID}.npz"), allow_pickle=True)["attempt_ids"]
        for q in E.BUCKETS:
            ok = info[str(q)]["attempts"] == info[str(q)]["successes"] == E.N_PATHS and bool(np.array_equal(att[q], np.arange(E.N_PATHS)))
            detail[f"{cond}|q{q}"] = ok
            cells_ok &= ok
    gate("1c the 8 target cells have attempts = successes = 1000 (no failed attempts) and attempt ids 0..999", cells_ok, detail)

    # ---- gate 4: statistics definitions ----------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.standard_normal(1001) * 1e-4 * (1 + 0.5 * np.abs(np.sin(np.arange(1001) / 40)))) + 4.6
    closes = np.exp(x)
    r = np.diff(np.log(closes))
    atlas = S2.full_characterize(pd.DataFrame({"close": closes}))
    ref = {"std": float(r.std()), "skewness": float(sps.skew(r)), "kurtosis": float(sps.kurtosis(r, fisher=False))}
    acf = _acf(r, nlags=50, fft=True)
    ref.update(acf_1=float(acf[1]), acf_5=float(acf[5]), acf_20=float(acf[20]))
    ref["volclust_50"] = float(_acf(np.abs(r), nlags=100, fft=True)[50])
    d_atlas = max(abs(atlas[k] - ref[k]) for k in ref)
    prof = S2.acf_profile(r, 50)
    d_prof = float(np.max(np.abs(prof - acf[1:51])))
    d_vr = max(abs(S2.vr_win(r, k) - S2.vr_win_brute(r, k)) for k in S2.KS)
    ps = S2.path_stats(np.exp(np.concatenate([[4.6], 4.6 + np.cumsum(rng.standard_normal(1440) * 1e-4)])))
    gate("4a statistics definitions: full_characterize equals numpy/scipy/statsmodels references on a synthetic series (1e-9); the ACF profile equals statsmodels acf for "
         "lags 1-50 (1e-9); VR_win equals a brute-force implementation (1e-12)", bool(d_atlas < 1e-9 and d_prof < 1e-9 and d_vr < 1e-12), {"atlas_vs_ref": d_atlas, "acf_profile_vs_statsmodels": d_prof, "vr_win_vs_brute": d_vr})
    data = P.V.Data()
    raw = data.raw
    win_ = L5.Windows(raw.close)
    worst_real, nwin = 0.0, 0
    for q in E.BUCKETS:
        bl = np.array(mp["buckets"][q]["used_blocks"])
        stats_ = [S2.full_characterize(pd.DataFrame({"close": raw.close[b * 60:b * 60 + 1001]})) for b in bl]
        rs = mp["buckets"][q]["real_stats"]
        for k in rs:
            v = float(np.nanmean([s[k] for s in stats_]))
            worst_real = max(worst_real, abs(v - rs[k]) / max(abs(rs[k]), 1e-12))
        nwin += len(bl)
        # ACF profile at lags 1, 5, 20 equals the acf_1, acf_5, acf_20 of full_characterize on the same real windows
        for b, s in list(zip(bl, stats_))[:20]:
            rr = np.diff(np.log(raw.close[b * 60:b * 60 + 1001]))
            pr = S2.acf_profile(rr, 20)
            worst_real = max(worst_real, abs(pr[0] - s["acf_1"]), abs(pr[4] - s["acf_5"]), abs(pr[19] - s["acf_20"]))
    gate("4b the statistics reproduce the stored real_stats of the mapping on the 300 used windows of buckets 0, 8, 15, 19 (relative 1e-9), and the ACF profile equals the "
         "mapping's acf_1/5/20", worst_real < 1e-9, {"max_rel_diff": worst_real, "n_windows": nwin})
    json.dump({"_code": code, "all_pass": all(v["pass"] for v in RES.values()), "gates": RES}, open(os.path.join(E.OUT, "pre_gates.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("ALL PASS" if all(v["pass"] for v in RES.values()) else "GATE FAILURE", flush=True)
    sys.exit(0 if all(v["pass"] for v in RES.values()) else 1)


if __name__ == "__main__":
    main()
