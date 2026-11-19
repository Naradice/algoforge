"""P3 section 9B: C0s consistency gate against the F1 C0 (tolerances fixed in the plan v0.3 @ c65cede before any run).
32 comparisons = {validation, final} x h {1,4,12,24} x {relative CRPS, cov90, k90, P(y>m)}: difference of the 3-seed means.
T = max(R, f): R = range (max-min) of the 3 F1-C0 seeds in that cell/metric (CRPS: relative to the C0 mean); floors f: CRPS 0.2% (relative),
coverage 0.5 pp, k90 0.03, P(y>m) 0.01. Pass: >= 29 of 32 within T and all within 2T. Exit code 0 = pass."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_env as E  # noqa: E402
import p3_eval as V  # noqa: E402

FLOOR = {"crps_rel": 0.002, "cov90": 0.005, "k90": 0.03, "P_up": 0.01}


def main():
    code = E.code_state()
    data = V.Data()
    res = {"_code": code, "comparisons": [], "floors": FLOOR}
    M0 = {s: V.all_metrics(data, "C0", s, keep_crps=False)[0] for s in (1, 2, 3)}
    MS = {s: V.all_metrics(data, "C0s", s, keep_crps=False)[0] for s in (1, 2, 3)}
    n_ok = n_2t = n_tot = 0
    for per in ("validation", "final"):
        for h in V.HZ:
            cell = f"{per}|h{h}"
            for name in ("crps_rel", "cov90", "k90", "P_up"):
                if name == "crps_rel":
                    c0 = np.array([M0[s][cell]["crps"] for s in (1, 2, 3)])
                    cs = np.array([MS[s][cell]["crps"] for s in (1, 2, 3)])
                    base = c0.mean()
                    r = (c0.max() - c0.min()) / base
                    diff = (cs.mean() - base) / base
                else:
                    c0 = np.array([M0[s][cell][name] for s in (1, 2, 3)])
                    cs = np.array([MS[s][cell][name] for s in (1, 2, 3)])
                    r = c0.max() - c0.min()
                    diff = cs.mean() - c0.mean()
                T = max(r, FLOOR[name])
                within, within2 = abs(diff) <= T, abs(diff) <= 2 * T
                n_tot += 1
                n_ok += within
                n_2t += within2
                res["comparisons"].append({"cell": cell, "metric": name, "diff": float(diff), "R_C0_seed_range": float(r), "T": float(T),
                                           "within_T": bool(within), "within_2T": bool(within2),
                                           "C0_seed_values": c0.tolist(), "C0s_seed_values": cs.tolist()})
    res.update(n_total=n_tot, n_within_T=int(n_ok), n_within_2T=int(n_2t), passed=bool(n_tot == 32 and n_ok >= 29 and n_2t == n_tot))
    json.dump(res, open(os.path.join(E.OUT, "c0s_gate.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("C0s gate:", res["n_within_T"], "/ 32 within T,", res["n_within_2T"], "within 2T ->", "PASS" if res["passed"] else "FAIL", flush=True)
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
