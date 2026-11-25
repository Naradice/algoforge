"""Stage A1 generation (plan v0.6 section 4.3 and 4.5): 176 grid points x 10 seeds (SA 5 + SB 5) x 100,001 trades, num_agent=300, no shock.
Stores the trade-level price series as int32 ticks (price / 0.001). Resumable (existing run files are skipped). Gate: 8 runs re-generated sequentially equal the pooled ones.
Output: OUT/a1/ticks_*.npy, OUT/a1/meta.jsonl, results/a1_gen.json"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402
from data.collectors.ddm_simulator import DDMv3  # noqa: E402


def gen_ticks(delta, wma, seed):
    random.seed(seed)
    np.random.seed(seed)
    model = DDMv3(num_agent=E.NUM_AGENT, max_volatility=0.02, min_volatility=0.01, wma=wma, dealer_sensitive_min=-3.5 + delta, dealer_sensitive_max=-1.5 + delta)
    prices = model.simulate(n_trades=E.N_TRADES)["price"].values
    return prices


def run_one(args):
    set_name, pid, delta, wma, k = args
    seed = E.seed_of(set_name, pid, k)
    t0 = time.time()
    meta = {"set": set_name, "pid": pid, "delta": delta, "wma": wma, "k": k, "seed": seed}
    try:
        prices = gen_ticks(delta, wma, seed)
        ok = len(prices) == E.N_TRADES and bool(np.all(np.isfinite(prices))) and bool(np.all(prices > 0))
        if ok:
            ticks = np.rint(prices / E.TICK)
            meta["max_abs_tick_rounding_error"] = float(np.max(np.abs(prices / E.TICK - ticks)))
            ok = meta["max_abs_tick_rounding_error"] < 1e-6
            if ok:
                np.save(E.run_file(set_name, pid, k), ticks.astype(np.int32))
        meta["success"] = bool(ok)
        meta["reason"] = None if ok else "non-finite, non-positive, short or non-tick price series"
    except (ValueError, RuntimeError) as e:
        meta["success"] = False
        meta["reason"] = f"{type(e).__name__}: {e}"[:200]
    meta["elapsed_s"] = time.time() - t0
    return meta


def main():
    code = E.code_state()
    os.makedirs(E.A1_DIR, exist_ok=True)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    meta_path = os.path.join(E.A1_DIR, "meta.jsonl")
    done = set()
    if os.path.exists(meta_path):
        for line in open(meta_path, encoding="utf-8"):
            m = json.loads(line)
            done.add((m["set"], m["pid"], m["k"]))
    tasks = [(s, pid, d, w, k) for s in ("SA", "SB") for (pid, d, w) in E.grid() for k in range(E.SEEDS_PER_SET) if (s, pid, k) not in done]
    print(f"tasks to run: {len(tasks)} (already done: {len(done)})", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=E.WORKERS) as ex, open(meta_path, "a", encoding="utf-8") as f:
        futs = [ex.submit(run_one, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            m = fu.result()
            f.write(json.dumps(m) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0
    metas = [json.loads(line) for line in open(meta_path, encoding="utf-8")]
    n_ok = sum(m["success"] for m in metas)
    # gate: sequential regeneration of 8 successful SA runs equals the pooled ones
    chosen = [m for m in metas if m["set"] == "SA" and m["k"] == 0 and m["success"]]
    chosen = chosen[:: max(1, len(chosen) // 8)][:8]
    bad = []
    for m in chosen:
        p = gen_ticks(m["delta"], m["wma"], m["seed"])
        if not np.array_equal(np.rint(p / E.TICK).astype(np.int32), np.load(E.run_file(m["set"], m["pid"], m["k"]))):
            bad.append((m["pid"], m["k"]))
    out = {"_code": code, "n_runs": len(metas), "n_success": int(n_ok), "elapsed_s_this_invocation": elapsed, "sum_run_elapsed_s": float(sum(m["elapsed_s"] for m in metas)),
           "reproducibility_gate": {"n_checked": len(chosen), "n_mismatch": len(bad), "pass": len(bad) == 0, "checked": [(m["pid"], m["k"]) for m in chosen]}}
    json.dump(out, open(os.path.join(E.RESULTS, "a1_gen.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out)[:600], flush=True)
    sys.exit(0 if out["reproducibility_gate"]["pass"] else 1)


if __name__ == "__main__":
    main()
