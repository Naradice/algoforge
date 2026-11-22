"""P5-B stage 2: regeneration of the stored paths (plan v0.2 @ 6d6af4c). Usage: python s2_regen.py <C3|C0s>
Calls p3_sim.gen_closes unchanged with the stored rule: seed = 8_000_000 + seed_id*1_000_000 + q*20_000 + attempt, candidate = cands[attempt % 8].
Gate 2: all 4000 regenerated success paths of the condition must reproduce the stored cumulative returns bit for bit (np.array_equal).
Gate 3: 8 attempts per bucket regenerated sequentially (1 worker) must equal the pooled (18 workers) result exactly. Stops (exit 1) on any mismatch;
C0s is not generated unless the C3 gate has passed."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2_env as E  # noqa: E402
import s2_lib as S2  # noqa: E402
import p3_sim as S  # noqa: E402

CANDS = {}


def _init():
    mp = S2.load_mapping()
    CANDS.update({b["q"]: [tuple(c) for c in b["candidates"]] for b in mp["buckets"]})


def regen_one(args):
    cond, q, a = args
    delta, wma = CANDS[q][a % E.N_CAND]
    seed = 8_000_000 + E.SEED_ID * 1_000_000 + q * 20_000 + a
    win, _ = S.gen_closes(delta, wma, seed, cond, sep=True)
    return None if win is None else np.asarray(win, dtype=np.float64)


def main(cond):
    os.makedirs(E.OUT, exist_ok=True)
    code = E.code_state()
    if cond == "C0s":
        g = json.load(open(os.path.join(E.OUT, "regen_C3.json"), encoding="utf-8"))
        if not g.get("pass"):
            raise SystemExit("C3 regeneration gate has not passed: C0s must not be generated")
    ens = np.load(os.path.join(E.ENS_DIR, f"{cond}_seed{E.SEED_ID}.npz"), allow_pickle=True)
    stored, att = ens["ens"], ens["attempt_ids"]
    info = json.load(open(os.path.join(E.ENS_DIR, f"{cond}_seed{E.SEED_ID}.json")))["info"]
    for q in E.BUCKETS:
        assert info[str(q)]["attempts"] == info[str(q)]["successes"] == E.N_PATHS and np.array_equal(att[q], np.arange(E.N_PATHS)), (cond, q)
    jobs = [(cond, q, a) for q in E.BUCKETS for a in range(E.N_PATHS)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=E.WORKERS, initializer=_init) as ex:
        res = list(ex.map(regen_one, jobs, chunksize=4))
    elapsed = time.time() - t0
    n_none = sum(r is None for r in res)
    mism, closes = [], {q: np.full((E.N_PATHS, 1441), np.nan) for q in E.BUCKETS}
    for (c, q, a), win in zip(jobs, res):
        if win is None:
            mism.append({"q": q, "attempt": a, "reason": "regenerated attempt failed (None)"})
            continue
        closes[q][a] = win
        if not np.array_equal(np.array(S.cum_returns(win)), stored[q, :, a]):
            mism.append({"q": q, "attempt": a, "reason": "cumulative returns differ", "regen": S.cum_returns(win), "stored": stored[q, :, a].tolist()})
    # gate 3: parallel independence (sequential regeneration in this process)
    _init()
    par_bad = []
    for q in E.BUCKETS:
        for a in E.PAR_ATTEMPTS:
            w1 = regen_one((cond, q, a))
            if w1 is None or not np.array_equal(w1, closes[q][a]):
                par_bad.append({"q": q, "attempt": a})
    ok = len(mism) == 0 and len(par_bad) == 0
    manifest = {}
    if ok:
        for q in E.BUCKETS:
            np.save(S2.closes_path(cond, q), closes[q])
            manifest[os.path.basename(S2.closes_path(cond, q))] = S2.sha256_file(S2.closes_path(cond, q))
    out = {"_code": code, "cond": cond, "pass": bool(ok), "n_paths": len(jobs), "n_regenerated_calls_main": len(jobs), "n_parallel_check_calls": len(E.BUCKETS) * len(E.PAR_ATTEMPTS),
           "n_none": n_none, "n_mismatch": len(mism), "mismatches_first20": mism[:20], "parallel_independence_bad": par_bad, "elapsed_s": elapsed, "sha256": manifest}
    json.dump(out, open(os.path.join(E.OUT, f"regen_{cond}.json"), "w", encoding="utf-8"), indent=1)
    print(f"{cond}: paths {len(jobs)}, mismatches {len(mism)}, parallel-check failures {len(par_bad)}, {elapsed:.0f}s -> {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
