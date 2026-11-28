"""P0 checks of the DP2 code on short synthetic inputs (not an experiment).

    python dp2_test.py <empty scratch directory>

The experimental simulations are replaced by a random walk (cfg["fake"]); the simulator is used only for the
equivalence check with non-experimental seeds.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import dp2_run as D
import dp_lib as L
import dp_run as R1
import numpy as np


def small(**kw):
    cfg = copy.deepcopy(D.CFG)
    cfg.update(burn=500, k=20_000)
    cfg.update(kw)
    return cfg


def test_bars():
    cfg = small()
    rng = np.random.default_rng(1)
    n = cfg["burn"] + cfg["k"]
    prices = 101 + np.cumsum(rng.normal(0, 0.002, n))
    t = np.exp(rng.normal(0.0, 2.0, n))
    r, count, q = D.bars(prices, t, cfg)
    times = np.cumsum(t)
    start = np.ceil(times[cfg["burn"] - 1] / 60) * 60
    lp = np.log(prices)
    for i in list(range(0, len(r), 37)) + [len(r) - 1]:
        m = np.flatnonzero((times >= start + 60 * i) & (times < start + 60 * (i + 1)))
        d = lp[m] - lp[m - 1]
        assert count[i] == len(m) and np.isclose(q[i], np.sum(d * d)) and np.isclose(r[i], np.sum(d), atol=1e-12)
    assert np.array_equal(count, D.bar_counts(t, cfg)) and np.any(count == 0)
    # a trade exactly on a boundary goes to the later bar
    t = np.full(n, 20.0)
    _, count, _ = D.bars(prices, t, cfg)
    assert np.all(count == 3)
    nb, i0, _, bins = D.grid(t, cfg)
    assert np.isclose(np.cumsum(t)[i0] % 60, 0) and bins[0] == 0 and bins[3] == 1
    st = D.series_stats(prices, np.exp(rng.normal(1.0, 1.0, n)), cfg)
    assert len(st["vr_by_n_quartile"]) == 4 and all(np.isfinite(st["vr_by_n_quartile"]))
    print("ok  fixed-time bars")


def test_rw():
    cfg = small()
    rng = np.random.default_rng(2)
    prices = 101 + np.cumsum(rng.normal(0, 0.002, cfg["burn"] + cfg["k"]))
    b = cfg["burn"]
    rw = D.make_rw(prices, 9, cfg)
    assert np.array_equal(rw[:b], prices[:b]) and np.isclose(rw[-1], prices[-1], rtol=1e-10)
    a, c = np.diff(np.log(prices[b - 1:])), np.diff(np.log(rw[b - 1:]))
    assert np.allclose(np.sort(a), np.sort(c), atol=1e-12) and not np.allclose(a, c)
    assert np.array_equal(rw, D.make_rw(prices, 9, cfg))
    print("ok  shuffled-increment series")


def test_mechanism_of_the_benchmark():
    """Independent price steps + persistent trade counts must give persistence in |bar return|; shuffled must not."""
    cfg = small(k=200_000)
    rng = np.random.default_rng(3)
    n = cfg["burn"] + cfg["k"]
    h = np.empty(n)
    h[0] = 0.0
    e = rng.normal(0, 0.05, n)
    for i in range(1, n):
        h[i] = 0.998 * h[i - 1] + e[i]
    t = 3.0 * np.exp(h)
    prices = 101 * np.exp(np.cumsum(rng.normal(0, 2e-5, n)))
    ordered = D.f_stat(D.bars(prices, t, cfg)[0], cfg)
    tt = t.copy()
    tt[cfg["burn"]:] = rng.permutation(t[cfg["burn"]:])
    shuffled = D.f_stat(D.bars(prices, tt, cfg)[0], cfg)
    assert ordered > 0.05 and abs(shuffled) < 0.02, (ordered, shuffled)
    print("ok  benchmark responds to a persistent clock")


def test_judgement():
    cfg = small()
    pairs = ((0.51, 0.9), (0.3, 0.9), (0.2, 0.9), (-0.5, 0.19), (0.1, 0.2), (0.21, 0.5), (0.5, 0.9))
    assert [D.judge_share(a, b, cfg) for a, b in pairs] == \
        ["主な原因", "寄与あり", "判定保留", "小さい", "判定保留", "寄与あり", "寄与あり"]
    rng = np.random.default_rng(4)

    def series(effect):
        s = rng.normal(0, 0.003, 20)
        return {"ord": effect + s + rng.normal(0, 0.003, 20), "shuf": s}

    ev = D.evaluate({"RW": series(0.10), "U": series(0.03), "C": series(0.01)}, cfg)
    sh = ev["shares"]
    assert ev["precondition"]["pass"] and np.isclose(sh["L_A"]["value"] + sh["L_B"]["value"] + sh["R"]["value"], 1.0)
    assert sh["L_A"]["verdict"] == "主な原因" and sh["L_B"]["verdict"] in ("寄与あり", "判定保留", "小さい")
    assert sh["L_A"]["lo"] < sh["L_A"]["value"] < sh["L_A"]["hi"]
    ev = D.evaluate({"RW": series(0.01), "U": series(0.0), "C": series(0.0)}, cfg)
    assert not ev["precondition"]["pass"] and "shares" not in ev
    json.dumps(ev, ensure_ascii=False)
    print("ok  shares and judgement")


def test_pipeline(scratch):
    os.makedirs(scratch, exist_ok=True)
    lf = np.random.default_rng(5).normal(size=4000)
    lf -= lf.mean()
    pool = os.path.join(scratch, "pool.npy")
    np.save(pool, lf)
    ref = os.path.join(scratch, "ref.json")
    cal = {"n": 50, "lam": 0.03, "kappa": 2.0, "m0": 1.5}

    def write_ref(sha):
        with open(ref, "w") as fh:
            json.dump({"calib": {"K-long": cal}, "checks": {"2_clock": {"pool": {"sha256": sha}}}}, fh)

    write_ref(R1.sha256(pool))
    base = small(fake=True, k=60_000, burn=2000, main_seeds=list(range(1, 7)), equiv_seeds=[11], equiv_trades=500,
                 min_pairs=5, n_boot=500, neg_band=0.05, half_tol=5.0, op_min=-1.0, pre_min=-9.0, workers=4,
                 dp1_s0=ref, pool_path=pool)
    state = {"head": "test", "time": "test"}
    work, res = os.path.join(scratch, "work"), os.path.join(scratch, "results")
    D.stage_s0(base, work, res, state)
    s0 = R1.load(res, "s0.json")
    assert s0["proceed"], s0["stopped"]
    try:
        D.stage_s1(base, work, res, state, changed="x.py | 1 +")
        raise AssertionError("changed code was accepted")
    except SystemExit:
        pass
    D.stage_s1(base, work, res, state)
    s1 = R1.load(res, "s1.json")
    assert s1["stopped"] is None and len(s1["valid_seeds"]) == 6 and "shares" in s1["evaluation"]
    assert set(s1["summary"]) == {"RW", "U", "C"}
    for fn in (D.stage_s0, D.stage_s1):
        try:
            fn(base, work, res, state)
            raise AssertionError("second run was allowed")
        except SystemExit:
            pass
    print("ok  pipeline s0 -> s1, once only")
    for key, value, text in (("op_min", 9.0, "check 3"), ("half_tol", -1.0, "check 2"), ("neg_band", 0.0, "check 4"),
                             ("cpu_cap_s", 0.0, "CPU time cap")):
        cfg = dict(base, **{key: value})
        w, r = os.path.join(scratch, f"w_{key}"), os.path.join(scratch, f"r_{key}")
        D.stage_s0(cfg, w, r, state)
        out = R1.load(r, "s0.json")
        assert not out["proceed"] and text in out["stopped"], (key, out["stopped"])
    write_ref("0" * 64)
    w, r = os.path.join(scratch, "w_sha"), os.path.join(scratch, "r_sha")
    D.stage_s0(base, w, r, state)
    assert "check 1" in R1.load(r, "s0.json")["stopped"]
    bad = dict(base, fake=False, price_lo=1000.0, k=200, burn=100)
    out = D.run_job((bad, "U", 31, None, scratch))
    assert not out["ok"] and "out of range" in out["error"]
    print("ok  stop paths")


if __name__ == "__main__":
    scratch = sys.argv[1]
    os.makedirs(scratch, exist_ok=True)
    assert not os.listdir(scratch), "use an empty directory"
    test_bars()
    test_rw()
    test_mechanism_of_the_benchmark()
    test_judgement()
    test_pipeline(scratch)
    print("all tests passed")
