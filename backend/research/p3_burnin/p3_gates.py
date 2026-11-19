"""P3 phase-0 gates (plan v0.3 @ c65cede, section 9). Any failure -> stop (exit code 1). Output: C:/tmp/p3_out/gates.json."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import p3_env as E  # noqa: E402
import p3_sim as S  # noqa: E402

RES = {}
FAILS = []


def check(name, ok, detail=None):
    RES[name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name, detail if (detail is not None and not ok) else "", flush=True)
    if not ok:
        FAILS.append(name)


class CountingRandom(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.n = 0

    def random(self):
        self.n += 1
        return super().random()


@contextlib.contextmanager
def count_global():
    names = ["random", "choices", "uniform", "shuffle", "choice", "randint"]
    orig = {n: getattr(random, n) for n in names}
    cnt = {n: 0 for n in names}

    def mk(n):
        def f(*a, **k):
            cnt[n] += 1
            return orig[n](*a, **k)
        return f
    for n in names:
        setattr(random, n, mk(n))
    try:
        yield cnt
    finally:
        for n in names:
            setattr(random, n, orig[n])


def step_n(model, n):
    """n raw simulation steps (same body as simulate_stream; price_history trimmed as there)."""
    keep = max(model.wma, 0) + 2
    for _ in range(n):
        model._add_time_for_ticks()
        price, _t = model._common_step()
        if price is not None and len(model.price_history) > keep:
            del model.price_history[0]


def fp(model):
    h = hashlib.sha256()
    h.update(model.agent_prices.tobytes())
    h.update(model.position.tobytes())
    h.update(repr(tuple(model.price_history)).encode())
    return h.hexdigest()


def rs():
    return hashlib.sha256(repr(random.getstate()).encode()).hexdigest()


FORCED = dict(exogenous_shock_probability=1.0, exogenous_shock_decay_tau=None)


def main():
    code = E.code_state()
    RES["_code"] = code
    t0 = time.time()
    # --- gate 7 (inputs) and simulator identity
    ident = E.simulator_identity()
    check("simulator and f1_ddm.py identical to F1 commit", all(v["identical"] for v in ident.values()), ident)
    sys.path.insert(0, os.path.join(E.BACKEND, "research"))
    import f1_ddm  # production module (read-only use: reference implementation for the window test)
    # P2 gate-1 style input hashes
    sys.path.insert(0, os.path.join(E.REPO, "backend", "research", "p2_diagnostics"))
    import p2_env as P2E
    sys.path.insert(0, P2E.P1_DIR)
    import p1_lib as P1
    listed = {}
    for line in open(P2E.SHA_LIST):
        h, f = line.strip().split(" *")
        listed[f] = h
    actual = P1.listdir_hashes(E.RUN_DIR)
    bad = [f for f in listed if actual.get(f) != listed[f]]
    pq = P1.sha256_file(P2E.PARQUET)
    check("input hashes (58 files + raw parquet)", not bad and len(listed) == 58 and pq == P2E.PARQUET_SHA, {"bad": bad, "parquet": pq})

    # --- gate 1: C0 byte-for-byte reproduction (legacy mode), buckets 0, 10, 19 of seed 1
    stored = np.load(os.path.join(E.RUN_DIR, "ddm_ens_seed1.npy"))
    sinfo = json.load(open(os.path.join(E.RUN_DIR, "ddm_ens_seed1.json")))
    ens, att, info = S.make_ensemble("C0", 1, workers=18, buckets=[0, 10, 19], sep=False)
    same = {q: bool(np.array_equal(ens[q], stored[q])) for q in (0, 10, 19)}
    info_ok = {q: (info[q]["successes"] == sinfo[str(q)]["successes"] and info[q]["attempts"] == sinfo[str(q)]["attempts"]) for q in (0, 10, 19)}
    check("gate1 C0 byte-identical reproduction (buckets 0,10,19, seed 1)", all(same.values()) and all(info_ok.values()), {"same": same, "info_ok": info_ok})

    # --- gate 2: window tests
    ok_a = True
    for (d, w, sd) in ((0.0, 7, 101), (0.3, 15, 202), (-0.2, 7, 303)):
        ref = f1_ddm.simulate_long_path(d, w, sd)
        win, _ = S.gen_closes(d, w, sd, "C0", sep=False)
        ok_a &= ref is not None and win is not None and np.array_equal(ref, win)
    check("gate2a B=0 window equals f1_ddm.simulate_long_path", ok_a)
    ok_b = True
    detail = {}
    for B, cond in ((0, "C0s"), (15, "C1_15"), (60, "C1_60"), (240, "C1_240")):
        d, w, sd = 0.0, 7, 4242
        win, info_ = S.gen_closes(d, w, sd, cond, sep=True)
        m = S.build_model(d, w, sd, True, True)
        prices = m.simulate(n_trades=S.TPC * (B + S.MAX_H * S.BLOCK + 1))["price"].values
        base_idx = S.TPC * (B + 1) - 1
        manual = [float(np.log(prices[base_idx + S.TPC * S.BLOCK * h] / prices[base_idx])) for h in S.HORIZONS]
        got = S.cum_returns(win)
        ok_b &= len(win) == 1441 and win[0] == prices[base_idx] and np.allclose(manual, got, rtol=0, atol=0)
        detail[cond] = {"len": len(win), "manual": manual, "got": got}
    check("gate2b window length 1441, base = trade 20(B+1), y_h = log C[B+60h] - log C[B]", ok_b, detail)
    d, w, sd = 0.0, 7, 777
    w0, i0 = S.gen_closes(d, w, sd, "C0s")
    c1, i1 = S.gen_closes(d, w, sd, "C1_60")
    c2, i2 = S.gen_closes(d, w, sd, "C2_60")
    c5, i5 = S.gen_closes(d, w, sd, "C5_60")
    c3, j3 = S.gen_closes(d, w, sd, "C3")
    c4, j4 = S.gen_closes(d, w, sd, "C4_60")
    pre = 61
    ok_c = (np.array_equal(i1["closes_full"][:1441], w0) and np.array_equal(i2["closes_full"][:pre], w0[:pre])
            and np.array_equal(i5["closes_full"][:pre], w0[:pre]) and np.array_equal(j4["closes_full"][:1441], c3))
    check("gate2c C1/C2/C5 prefixes equal C0s; C4 prefix equals C3", ok_c)
    m = S.build_model(d, w, sd, True, True)
    pa = np.concatenate([m.simulate(n_trades=20 * 61)["price"].values, m.simulate(n_trades=20 * 1440)["price"].values])
    m = S.build_model(d, w, sd, True, True)
    pb = m.simulate(n_trades=20 * 1501)["price"].values
    check("gate2d two-stage simulate equals single-run (no intervention)", np.array_equal(pa, pb))

    # --- gate 3: forced-shock unit tests (test-only configuration)
    m = S.build_model(0.0, 7, 55, True, True, shock_override=FORCED)
    step_n(m, 300)
    check("gate3a p=1: a shock at every step", len(m._shock_events) == 300 and m._step_count == 300, len(m._shock_events))
    # b: branch at first shock
    def run_branch(turn_on_at, n, cps):
        mm = S.build_model(0.0, 7, 66, False, True, shock_override=FORCED)       # starts with p = 0 (disabled)
        snaps = {}
        for k in range(n):
            if k == turn_on_at:
                mm.exogenous_shock_probability = 1.0
            step_n(mm, 1)
            if k + 1 in cps:
                snaps[k + 1] = (fp(mm), rs())
        return snaps, len(mm._shock_events)
    K, N = 100, 2000
    CPS = (K - 1, K, K + 1, K + 50, N)
    sa, ea = run_branch(K, N, CPS)
    sb, eb = run_branch(10 ** 9, N, CPS)            # never turned on
    before = (sa[K - 1][0] == sb[K - 1][0])
    after = all(sa[k][0] != sb[k][0] for k in (K + 1, K + 50, N))
    rng_equal = all(sa[k][1] == sb[k][1] for k in sa)
    check("gate3b C0s-like vs C3-like: identical before the first shock, states diverge after", before and after and ea > 0 and eb == 0, {"before": before, "after": after, "events": (ea, eb)})
    check("gate4b' random state equal at every checkpoint around a forced shock (before/after)", rng_equal, {k: sa[k][1] == sb[k][1] for k in sa})
    # c: clock reset
    m = S.build_model(0.0, 7, 77, True, True, shock_override=dict(exogenous_shock_probability=1.0, exogenous_shock_decay_tau=300.0))
    step_n(m, 600)
    p_before = m._current_shock_probability()
    m._step_count = 0
    p_after = m._current_shock_probability()
    check("gate3c reset: probability back to P0 after _step_count = 0", abs(p_before - np.exp(-2.0)) < 0.01 and abs(p_after - 1.0) < 1e-12, (p_before, p_after))
    # d: C5 disable
    m = S.build_model(0.0, 7, 88, True, True, shock_override=FORCED)
    step_n(m, 200)
    n_before = len(m._shock_events)
    m.exogenous_shock_probability = 0.0
    step_n(m, 800)
    check("gate3d C5: no shock events after disabling", len(m._shock_events) == n_before and n_before == 200, (n_before, len(m._shock_events)))
    # e: C3/C4 no events (production configuration), f: production-config reset probability
    ev = 0
    for cond in ("C3", "C4_60"):
        win, inf = S.gen_closes(0.0, 7, 99, cond)
        ev += len(inf["model"]._shock_events)
    check("gate3e C3/C4 produce no shock events", ev == 0, ev)
    m = S.build_model(0.0, 7, 111, True, True)
    m.simulate(n_trades=20 * 61)
    m._step_count = 0
    check("gate3f production config: P(step=0) = 0.002 after reset", abs(m._current_shock_probability() - 0.002) < 1e-15)

    # --- gate 4: common random numbers
    scen = {"p0_sep": dict(burn=False, ov=None), "p1_sep": dict(burn=True, ov=FORCED), "natural_sep": dict(burn=True, ov=None)}
    cnt_ok = True
    cnt_detail = {}
    for nm, sc in scen.items():
        m = S.build_model(0.0, 7, 121, sc["burn"], True, shock_override=sc["ov"])
        m._shock_rng = CountingRandom(121 + S.SHOCK_SEED_OFFSET)
        n = 3000
        with count_global() as cnt:
            step_n(m, n)
        trades = len(m.price_history)  # not used
        ok = cnt["choices"] == n and cnt["random"] == 0 and cnt["uniform"] == 0 and cnt["shuffle"] == 0 and m._shock_rng.n == 2 * n
        cnt_detail[nm] = {"choices": cnt["choices"], "global random": cnt["random"], "shock_rng draws": m._shock_rng.n, "steps": n, "events": len(m._shock_events)}
        cnt_ok &= ok
    check("gate4a normal draws exactly 1/step (choices), shock_rng exactly 2/step, in p=0 / p=1 / natural", cnt_ok, cnt_detail)
    nat_with_shock = 0
    nat_ok = True
    for sd in range(300, 320):
        ma = S.build_model(0.0, 7, sd, True, True)
        step_n(ma, 4000)
        ra = rs()
        ev_a = len(ma._shock_events)
        mb = S.build_model(0.0, 7, sd, False, True)
        step_n(mb, 4000)
        rb = rs()
        nat_ok &= ra == rb
        if ev_a > 0:
            nat_with_shock += 1
            nat_ok &= fp(ma) != fp(mb)
    check("gate4c natural shocks: global random state equal after 4000 steps in 20 seeds (incl. paths with shocks)", nat_ok and nat_with_shock >= 1, {"paths_with_shock": nat_with_shock})

    # --- gate 5: attempt ids saved
    e5, a5, i5_ = S.make_ensemble("C0s", 1, workers=4, buckets=[3], n_paths=4, sep=True)
    check("gate5 attempt IDs saved and consistent", a5.shape == (20, 4) and np.all(a5[3] >= 0) and np.all(np.diff(a5[3]) > 0) and np.isfinite(e5[3]).all(), a5[3].tolist())

    # --- gate 6: no future information
    src = open(os.path.join(HERE, "p3_sim.py"), encoding="utf-8").read()
    banned = [t for t in ("read_parquet", "usdjpy", "artifact") if t in src.lower()]
    sys.path.insert(0, os.path.join(E.REPO, "backend", "research", "p2_diagnostics"))
    import p2_data as D2
    raw = D2.load_raw()
    bucket = D2.buckets_for(raw, D2.load_mapping())
    bh = hashlib.sha256(np.asarray(bucket, dtype=np.int64).tobytes()).hexdigest()
    check("gate6 p3_sim reads no market data (static); bucket array hash recorded", not banned, {"banned": banned, "bucket_hash": bh})
    RES["bucket_hash"] = bh
    RES["elapsed_s"] = time.time() - t0
    RES["all_pass"] = not FAILS
    os.makedirs(E.OUT, exist_ok=True)
    json.dump(RES, open(os.path.join(E.OUT, "gates.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
