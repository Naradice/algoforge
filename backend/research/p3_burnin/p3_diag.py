"""P3 section 8: internal-state relaxation diagnostic (no realised data). Plan v0.3 @ c65cede.
50 paths per bucket (diagnostic seed_id = 9), configurations C3 (no shock) and C0s (shock on, clock from 0), snapshots after candle k
(= after 20(k+1) trades) for k in SNAP. Criterion (fixed in the plan): late reference = mean of the path-mean over k in LATE;
stationarity: max-min over LATE <= 0.10|mu_ref| + a_S; a_S = 2*SE_ref (SE_ref = SD over the 20 bucket means of the late average / sqrt(20));
state quantity satisfies the criterion at B if |m_B - mu_ref| <= max(0.05|mu_ref|, a_S)."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_env as E  # noqa: E402
import p3_sim as S  # noqa: E402

SNAP = [0, 1, 2, 5, 10, 15, 30, 60, 120, 240, 480, 720, 960, 1200, 1440]
LATE = [480, 720, 960, 1200, 1440]
STATS = ["pos_imbalance", "agent_price_sd", "cross_gap", "wma_abs", "abs_ret5", "shock_prob"]
N_DIAG = 50
SEED_ID = 9


def state_stats(m, closes):
    pos = m.position
    longp = m.agent_prices[pos == 1]
    shortp = m.agent_prices[pos == -1]
    gap = float(longp.max() - (shortp.min() + m.spread)) if len(longp) and len(shortp) else float("nan")
    r = np.abs(np.diff(np.log(np.asarray(closes[-6:]))))
    return [abs(float(pos.sum())) / len(pos), float(np.std(m.agent_prices)), gap, abs(float(m._wma_diff())),
            float(r.mean()) if len(r) >= 1 else float("nan"),
            float(m._current_shock_probability()) if m.exogenous_shock_probability > 0 else 0.0]


def diag_job(args):
    q, cands, cfg = args
    shock_on = cfg == "C0s"
    out = []
    attempts = 0
    while len(out) < N_DIAG and attempts < S.MAX_ATTEMPTS:
        delta, wma = cands[attempts % len(cands)]
        seed = 8_000_000 + SEED_ID * 1_000_000 + q * 20_000 + attempts
        attempts += 1
        m = S.build_model(delta, wma, seed, shock_on, True)
        try:
            allp = []
            rec = []
            done = 0
            ok = True
            for k in SNAP:
                n_tr = S.TPC * (k + 1) - done
                p = m.simulate(n_trades=n_tr)["price"].values
                done += n_tr
                if not np.all(np.isfinite(p)) or np.any(p <= 0):
                    ok = False
                    break
                allp.extend(p.tolist())
                closes = allp[S.TPC - 1::S.TPC]                  # candle closes so far (k + 1 of them)
                rec.append(state_stats(m, closes))
            if ok and len(rec) == len(SNAP):
                out.append(rec)
        except (ValueError, RuntimeError):
            continue
    return q, np.array(out)          # [paths, snaps, stats]


def main():
    code = E.code_state()
    mp = json.load(open(S.MAPPING))
    res = {"_code": code, "snap": SNAP, "late": LATE, "stats": STATS, "configs": {}}
    t0 = time.time()
    for cfg in ("C3", "C0s"):
        jobs = [(b["q"], [tuple(c) for c in b["candidates"]], cfg) for b in mp["buckets"]]
        arr = {}
        with ProcessPoolExecutor(max_workers=18) as ex:
            for q, a in ex.map(diag_job, jobs):
                arr[q] = a
                print(cfg, "bucket", q, a.shape, f"{time.time() - t0:.0f}s", flush=True)
        # bucket means [20, snaps, stats]
        bm = np.stack([np.nanmean(arr[q], axis=0) for q in range(20)])
        pm = bm.mean(axis=0)                                         # pooled mean over buckets [snaps, stats]
        sd_b = bm.std(axis=0, ddof=1)
        cfg_res = {"n_paths_per_bucket": {str(q): int(arr[q].shape[0]) for q in range(20)}, "by_stat": {}}
        li = [SNAP.index(k) for k in LATE]
        for si, nm in enumerate(STATS):
            late_means = pm[li, si]
            mu_ref = float(np.mean(late_means))
            late_bucket = bm[:, li, si].mean(axis=1)                 # [20] bucket late average
            se_ref = float(np.std(late_bucket, ddof=1) / np.sqrt(20))
            a_s = 2 * se_ref
            stationary = bool(late_means.max() - late_means.min() <= 0.10 * abs(mu_ref) + a_s)
            entry = {"mu_ref": mu_ref, "a_S": a_s, "late_means": late_means.tolist(), "stationary": stationary,
                     "mean_by_snap": pm[:, si].tolist(), "between_bucket_sd_by_snap": sd_b[:, si].tolist(), "B": {}}
            for B in (15, 60, 240):
                bi = SNAP.index(B)
                tol = max(0.05 * abs(mu_ref), a_s)
                pooled_ok = bool(abs(pm[bi, si] - mu_ref) <= tol)
                bucket_ok = np.abs(bm[:, bi, si] - mu_ref) <= tol
                entry["B"][str(B)] = {"m_B": float(pm[bi, si]), "tol": float(tol), "pooled_ok": pooled_ok,
                                      "bucket_pass_fraction": float(bucket_ok.mean()), "bucket_values": bm[:, bi, si].tolist()}
            cfg_res["by_stat"][nm] = entry
        verdict = {}
        for B in (15, 60, 240):
            allok = all(cfg_res["by_stat"][nm]["stationary"] and cfg_res["by_stat"][nm]["B"][str(B)]["pooled_ok"] for nm in STATS)
            minfrac = min(cfg_res["by_stat"][nm]["B"][str(B)]["bucket_pass_fraction"] for nm in STATS)
            verdict[str(B)] = {"all_observed_stats_meet_criterion": bool(allok), "min_bucket_pass_fraction": float(minfrac),
                               "bucket_scope_ok(>=0.8)": bool(minfrac >= 0.8)}
        cfg_res["verdict"] = verdict
        res["configs"][cfg] = cfg_res
        print(cfg, json.dumps(verdict), flush=True)
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open(os.path.join(E.OUT, "diag_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
