"""P3 phase 2: allocation guide (plan v0.3 section 7; NOT a scientific pass/fail criterion). A condition (seed 1) is flagged when, in any
{validation, final} x h cell, versus C0s (seed 1): |d cov90| >= 0.02, or |relative CRPS change| >= 0.005, or |d k90| >= 0.10, or |d P(y>m)| >= 0.03."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_env as E  # noqa: E402
import p3_eval as V  # noqa: E402

if __name__ == "__main__":
    code = E.code_state()
    data = V.Data()
    ref = V.all_metrics(data, "C0s", 1, keep_crps=False)[0]
    out = {"_code": code, "conditions": {}, "flagged": []}
    for cond in ("C1_15", "C1_240", "C2_15", "C2_240", "C5_15", "C5_240"):
        m = V.all_metrics(data, cond, 1, keep_crps=False)[0]
        hits = []
        for per in ("validation", "final"):
            for h in V.HZ:
                c = f"{per}|h{h}"
                d = {"cov90": m[c]["cov90"] - ref[c]["cov90"], "crps_rel": m[c]["crps"] / ref[c]["crps"] - 1,
                     "k90": m[c]["k90"] - ref[c]["k90"], "P_up": m[c]["P_up"] - ref[c]["P_up"]}
                if abs(d["cov90"]) >= 0.02 or abs(d["crps_rel"]) >= 0.005 or abs(d["k90"]) >= 0.10 or abs(d["P_up"]) >= 0.03:
                    hits.append({"cell": c, **d})
        out["conditions"][cond] = {"n_cells_meeting_guide": len(hits), "cells": hits}
        if hits:
            out["flagged"].append(cond)
    json.dump(out, open(os.path.join(E.OUT, "phase2_flags.json"), "w"), indent=1)
    print("flagged:", out["flagged"])
