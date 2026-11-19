"""Diagnostic for a gate-4 case-1 Type-I excess (plan section 6, rule (a)/(b)): is the excess an implementation defect or a property of the
test design (finite ensembles M=1000 shared by ~150 origins per bucket, whose sampling error is not captured by the week-block bootstrap)?
Variants of case 1 (everything else identical): V0 = plan design (random ensemble, M=1000); V1 = noise-free ensemble (deterministic
predictive quantile grid, M=1000); V2 = random ensemble with M=20000. Does NOT change the plan criteria; reporting only."""
import json, os, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p2_lib as L, p2_env as E, p2_synth as S

def run(variant, trials):
    rej = {k: 0 for k in ("FA90", "FB", "FC", "FE", "FA50")}
    for t in range(trials):
        d = S.gen_case(1, t)
        rng = np.random.default_rng([S.SEED, 11, t])
        if variant == "V1":
            q = stats.t.ppf((np.arange(S.M) + 0.5) / S.M, 5) * S.S5
            ens = np.tile(q, (S.N_BUCKET, 1))
        elif variant == "V2":
            ens = np.sort(stats.t.rvs(5, size=(S.N_BUCKET, 20000), random_state=rng) * S.S5, axis=1)
        else:
            ens = d["ens"]
        o = L.origin_stats(ens, d["bucket"], d["y"])
        wb = L.WeekBoot(d["week"], S.B_TEST, seed=[S.SEED, 1, t, 1])
        ft = L.family_tests(o, d["y"], wb)
        for k in rej:
            rej[k] += int(ft[k]["p"] is not None and ft[k]["p"] < 0.05)
    return {k: v / trials for k, v in rej.items()}

if __name__ == "__main__":
    code = E.code_state()
    out = {"_code": code, "trials": 1000}
    for v in ("V0", "V1", "V2"):
        out[v] = run(v, 1000)
        print(v, out[v], flush=True)
    json.dump(out, open(os.path.join(E.OUT, "synth_case1_diag.json"), "w"), indent=1)
