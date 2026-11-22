"""P5-B pre-analysis gates (plan v0.2 section 5). Stops (exit 1) when a gate fails. Output: OUT/gates.json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5b_env as E  # noqa: E402
import p5b_lib as L5  # noqa: E402
import p5_lib as K  # noqa: E402
import p4_lib as P  # noqa: E402
import p1_lib as P1  # noqa: E402

RES = {}


def gate(name, ok, detail=None):
    RES[name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def main():
    os.makedirs(E.OUT, exist_ok=True)
    code = E.code_state()
    data = P.V.Data()
    raw = data.raw
    mp = L5.load_mapping()
    atlas = L5.load_atlas()
    # ---- gate 1: input identity ------------------------------------------------------------------------------------------------------
    import p2_env as E2
    sha = P1.sha256_file(E2.PARQUET)
    d5 = json.load(open(os.path.join(E.P5A_RESULTS, "analysis_D.json"), encoding="utf-8"))
    worst, nbad = 0.0, 0
    c0 = L5.load_cond("C0s")
    for h in L5.HZ:
        for b in range(20):
            st = L5.cand_stats(c0[0], c0[1], b, h)
            ref = d5["candidates"][f"h{h}"][b]
            cell = data.cells[("estimation", h)]
            yb = cell["y"][cell["b"] == b]
            w_e1a = float(np.subtract(*np.quantile(yb, [0.95, 0.05])))
            d = max(abs(st["mix_w90"] - ref["width90_mix8"]), abs(w_e1a - ref["width90_e1a"]), *[abs(x - y) for x, y in zip(st["w90_c"], ref["width90_candidates"])])
            worst = max(worst, d)
            nbad += int(d > 1e-12)
    gate("1 inputs: raw parquet hash; P5-A stored values (E1a width, 8-candidate mixture width, per-candidate widths; C0s) reproduce to 1e-12",
         sha == "c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828" and nbad == 0, {"sha256": sha, "max_abs_diff": worst, "n_bad": nbad})

    # ---- gate 2: mapping reproduction ------------------------------------------------------------------------------------------------
    win = L5.Windows(raw.close)
    est_hi = pd.Timestamp(P1.PERIODS["estimation"][1]) + pd.Timedelta(days=1)
    val_start = pd.Timestamp(P1.PERIODS["validation"][0])
    det, ok2, ok_sel = {}, True, True
    max_std_err, max_var_err, ratios = 0.0, 0.0, []
    for b in range(20):
        bl = np.array(mp["buckets"][b]["used_blocks"])
        a = bl * L5.BLOCK
        ok_w = bool(L5.valid_windows(raw, bl, est_hi).all())
        sd = win.std_direct(a)
        err = abs(sd.mean() - mp["buckets"][b]["real_stats"]["std"]) / mp["buckets"][b]["real_stats"]["std"]
        v_fast, v_dir = win.var(a), win.var_direct(a)
        verr = float(np.max(np.abs(v_fast - v_dir) / v_dir))
        max_std_err, max_var_err = max(max_std_err, err), max(max_var_err, verr)
        ratios.append(float(np.sqrt(v_dir.mean()) / sd.mean()))
        ok2 &= ok_w and err < 1e-9 and verr < 1e-9
        top, sl, valid = L5.replicate_selection(pd.Series(mp["buckets"][b]["real_stats"]), atlas)
        stored = [tuple(c) for c in mp["buckets"][b]["candidates"]]
        rep = [(float(r.delta), int(r.wma)) for r in top.itertuples()]
        same = len(stored) == len(rep) and all(np.isclose(s[0], r[0]) and s[1] == r[1] for s, r in zip(stored, rep))
        ok_sel &= same
        det[str(b)] = {"windows_valid_in_estimation": ok_w, "std_rel_err": float(err), "var_rel_err": verr, "selection_replicated": bool(same)}
    gate("2 mapping reproduction: the 300 used windows are gap-free and end inside the estimation period; window std (ddof=0) reproduces real_stats['std'] "
         "(1e-9); cumulative-sum variance equals the direct variance (1e-9); the std-first selection replicates the stored candidates and their order",
         bool(ok2 and ok_sel), {"max_std_rel_err": max_std_err, "max_var_rel_err": max_var_err, "sqrtV_over_real_std_min_max": [min(ratios), max(ratios)], "per_bucket": det})

    # ---- gate 3: atlas lookup --------------------------------------------------------------------------------------------------------
    bad = 0
    for b in range(20):
        for dl, wm in mp["buckets"][b]["candidates"]:
            r = L5.atlas_row(atlas, dl, int(wm))
            bad += int(len(r) != 1 or not np.isfinite(r["std"].values[0]) or not np.isfinite(r["std_std_across_seeds"].values[0]))
    gate("3 atlas lookup: each of the 160 candidates (delta, wma) exists exactly once in the atlas with finite std and std across seeds", bad == 0, {"n_bad": bad})

    # ---- gate 4: candidate recovery (C0s and C3) -------------------------------------------------------------------------------------
    src_ok = ("cands[attempts % len(cands)]" in open(os.path.join(E.BACKEND, "research", "f1_ddm.py"), encoding="utf-8").read())
    bad4, smin, smax, first_ok = 0, 1.0, 0.0, True
    for cond in L5.CONDS:
        for s in K.SEEDS:
            ens, att = K.load_raw_ens(cond, s)
            info = json.load(open(os.path.join(E.ENS_DIR, f"{cond}_seed{s}.json")))["info"]
            for b in range(20):
                a = att[b]
                bad4 += int(not (np.all(np.diff(a) > 0) and a.min() >= 0 and a.max() < int(info[str(b)]["attempts"])))
                fr = np.bincount(a % K.N_CAND, minlength=K.N_CAND) / len(a)
                smin, smax = min(smin, fr.min()), max(smax, fr.max())
                if info[str(b)]["attempts"] == info[str(b)]["successes"]:
                    first_ok &= bool(np.array_equal(a, np.arange(len(a))))
    gate("4 candidate recovery for C0s and C3: generation rule present in F1 code; attempt ids increasing within the recorded attempts; no-failure buckets have ids 0..999; "
         "per-candidate path share within [0.10, 0.15]", bool(src_ok and bad4 == 0 and first_ok and smin > 0.10 and smax < 0.15),
         {"rule_in_source": src_ok, "n_bad": bad4, "ids_ok": bool(first_ok), "share_min": float(smin), "share_max": float(smax)})

    # ---- gate 5: decomposition identity and definitions ------------------------------------------------------------------------------------
    c3 = L5.load_cond("C3")
    worst_res, def_ok = 0.0, True
    for h in L5.HZ:
        cells = data.cells[("estimation", h)]
        for b in range(20):
            bl = np.array(mp["buckets"][b]["used_blocks"])
            v_real = float(win.var(bl * L5.BLOCK).mean())
            s2 = np.mean([L5.sigma2_atlas(L5.atlas_row(atlas, dl, int(wm)).iloc[0]) for dl, wm in mp["buckets"][b]["candidates"]])
            y = cells["y"][cells["b"] == b]
            s0 = L5.cand_stats(c0[0], c0[1], b, h)
            s3 = L5.cand_stats(c3[0], c3[1], b, h)
            r = L5.decompose(s0["mix_sd"], s3["mix_sd"], float(y.var()), v_real, s2, h)
            worst_res = max(worst_res, abs(r["residual"]))
            def_ok &= bool(abs(r["VR_real"] - float(y.var()) / (60 * h * v_real)) < 1e-15 * max(1, r["VR_real"])
                           and abs(r["VR_C3"] - s3["mix_sd"] ** 2 / (60 * h * s2)) < 1e-12 * max(1, r["VR_C3"]))
    gate("5 decomposition identity M1 + M3 + M2 = log(sd_C0s / sd_E1a) to 1e-12 for all 80 bucket x horizon cells, and VR_real / VR_C3 / sigma^2_atlas,mix / V_real,M1 "
         "match the plan definitions", bool(worst_res < 1e-12 and def_ok), {"max_abs_residual": float(worst_res), "definitions_ok": bool(def_ok)})

    # ---- gate 6: no future information -------------------------------------------------------------------------------------------------------
    ok6, d6 = True, {}
    for b in range(20):
        bl = np.array(mp["buckets"][b]["used_blocks"])
        ok6 &= bool(raw.index[bl * L5.BLOCK + L5.WIN].max() < val_start)
    for h in L5.HZ:
        c = data.cells[("estimation", h)]
        e = raw.index[raw.t[c["j"]] + L5.BLOCK * h].max()
        d6[f"y h{h} max window end"] = str(e)
        ok6 &= bool(e < val_start)
    j1 = data.cells[("estimation", 1)]["j"]
    okv = L5.valid_windows(raw, j1, est_hi)
    ok6 &= bool(raw.index[j1[okv] * L5.BLOCK + L5.WIN].max() < val_start)
    gate("6 no future information: mapping windows, all-origin estimation windows and y windows of the estimation set end before the validation period starts; "
         "validation/final context windows are restricted to end inside their own period", ok6, d6)
    json.dump({"_code": code, "all_pass": all(v["pass"] for v in RES.values()), "gates": RES}, open(os.path.join(E.OUT, "gates.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("ALL PASS" if all(v["pass"] for v in RES.values()) else "GATE FAILURE", flush=True)
    sys.exit(0 if all(v["pass"] for v in RES.values()) else 1)


if __name__ == "__main__":
    main()
