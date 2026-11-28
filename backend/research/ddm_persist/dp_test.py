"""P0 checks of the DP1 code on short synthetic inputs (not an experiment).

    python dp_test.py <empty scratch directory>

No statistic of any experimental condition is computed on the simulator: the end-to-end part replaces the
experimental runs by a random walk (cfg["fake"]), and the simulator is only used for equivalence and
window-selection checks with non-experimental seeds.
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys

import dp_lib as L
import dp_run as R
import numpy as np


def test_equivalence():
    for s in (11, 12):
        row = L.equivalence(L.CFG, s, 800)
        assert row["equal"] and row["price_changes"] > 100, row
    print("ok  equivalence (unit intervals, eta=5 == DDMv3 equal weights)")


def test_window_selection():
    rng = np.random.default_rng(5)
    n, eta = 600, 15.0
    iv = np.exp(rng.normal(0.5, 1.4, n))
    random.seed(13)
    np.random.seed(13)
    model = L.DDMWindow(iv, eta, **L.CFG["model"])
    prices = [model.market_price]
    times = np.concatenate([[0.0], np.cumsum(iv)])
    seen_one = seen_many = False
    for k, (p, _) in enumerate(model.simulate_stream(n_trades=n), start=1):
        prices.append(p)
        js = [j for j in range(1, k + 1) if model._times[k] - model._times[j] < eta]
        expect = np.mean([prices[j] - prices[j - 1] for j in js]) if model._times[k] >= eta else 0.0
        assert abs(model._wma_diff() - expect) < 1e-12, (k, model._wma_diff(), expect)
        assert abs(model._times[k] - times[k]) < 1e-6
        seen_one |= len(js) == 1
        seen_many |= len(js) > 5
    assert seen_one and seen_many
    cfg = dict(L.CFG, burn=0, eta=eta)
    brute = [sum(1 for j in range(k + 1) if times[k + 1] - times[j + 1] < eta) for k in range(n)]
    assert np.array_equal(L.window_counts(iv, cfg), brute)
    print("ok  window selection matches brute force")


def test_metrics():
    cfg = dict(L.CFG, burn=100, k=40_000)
    rng = np.random.default_rng(1)
    x = rng.normal(size=100_000)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = 0.6 * y[i - 1] + x[i]
    assert abs(L.acf(y, 1) - 0.6) < 0.02 and abs(L.acf(y, 3) - 0.216) < 0.02
    assert abs(L.p_stat(np.abs(x), cfg)) < 0.005
    prices = 101 + np.cumsum(rng.normal(0, 0.002, cfg["burn"] + cfg["k"]))
    r = L.bar_returns(prices, cfg)
    assert len(r) == 2000
    assert np.isclose(r[0], np.log(prices[cfg["burn"] + 19] / prices[cfg["burn"] - 1]))
    assert np.isclose(r.sum(), np.log(prices[-1] / prices[cfg["burn"] - 1]))
    pos = L.positive_control(prices, 3, cfg)
    assert L.p_of_run(pos, cfg) - L.p_of_run(prices, cfg) > 0.05
    assert abs(L.permuted_p(pos, 3, cfg)) < 0.05
    assert [L.judge(*a, 0.03) for a in ((0.031, 0.05), (-0.05, -0.031), (-0.02, 0.02), (0.0, 0.05), (0.03, 0.04))] == \
        ["伝播を支持", "逆向き", "実質的な伝播なし", "判定保留", "判定保留"]
    v = rng.normal(1.0, 0.1, 20)
    ci = L.boot_ci(v, 0.975, cfg)
    assert ci["lo"] < ci["mean"] < ci["hi"] and ci["n"] == 20 and ci == L.boot_ci(v, 0.975, cfg)
    t = np.exp(rng.normal(0.0, 2.0, cfg["burn"] + cfg["k"]))          # long gaps give empty bars
    ft = L.fixed_time_returns(prices, t, cfg)
    times = np.cumsum(t)
    start = np.ceil(times[cfg["burn"] - 1] / 60) * 60
    nb = int(round((np.floor(times[-1] / 60) * 60 - start) / 60))
    before = prices[np.sum(times < start) - 1]
    close = [prices[np.sum(times < start + 60 * (i + 1)) - 1] for i in range(0, nb, 97)]
    full = before * np.exp(np.cumsum(ft))
    assert len(ft) == nb and np.allclose(full[::97], close, rtol=1e-12) and np.any(ft == 0)
    f, yy = L.har_windows(r)
    assert abs(f.mean()) < 1e-12 and abs(yy.mean()) < 1e-12
    print("ok  metrics, controls, judgement")


def test_clock():
    cfg = dict(L.CFG, burn=200, k=2000)
    lf = np.random.default_rng(2).normal(size=5000)
    lf -= lf.mean()
    cal = {"n": 16, "lam": 0.1, "m0": 1.5}
    a, b = L.make_clock(lf, cal, 7, "ord", cfg), L.make_clock(lf, cal, 7, "shuf", cfg)
    assert np.array_equal(a[:200], b[:200]) and np.array_equal(np.sort(a[200:]), np.sort(b[200:]))
    assert not np.array_equal(a[200:], b[200:]) and np.all(a > 0)
    print("ok  clock ordering and shuffle")


def mini_cfg():
    cfg = copy.deepcopy(L.CFG)
    cfg.update(fake=True, burn=1000, k=20_000, pool_trades=1500, cal_seeds=[8001, 8002], cal_k=20_000,
               diag_chains=50, diag_tol=1.0, half_tol=5.0, op_min=-1.0, main_seeds=list(range(1, 7)),
               pilot_seeds=[31, 32], equiv_seeds=[11], equiv_trades=500, min_pairs=5, n_boot=500,
               neg_band=0.05, pos_min=0.0, har_train_n=4, workers=4)
    return cfg


def test_pipeline(scratch):
    state = {"head": "test", "time": "test"}
    cfg = mini_cfg()
    work, res = os.path.join(scratch, "work"), os.path.join(scratch, "results")
    R.stage_s0(cfg, work, res, state)
    s0 = R.load(res, "s0.json")
    assert s0["proceed"], s0["stopped"]
    assert s0["run_clocks"] == ["K-short", "K-long"] and len(s0["checks"]["4_pilot"]["runs"]) == 10
    assert not any(f.startswith("S-ord_") for f in os.listdir(work)), "pilot runs must not be stored"
    R.stage_s1(cfg, work, res, state)
    s1 = R.load(res, "s1.json")
    assert s1["stopped"] is None and set(s1["comparisons"]) == {"K-short", "K-long"}
    assert s1["comparisons"]["K-short"]["pairs"] == 6 and s1["summary"]["L-ord"]["har"] is not None
    for fn in (R.stage_s0, R.stage_s1):
        try:
            fn(cfg, work, res, state)
            raise AssertionError("second run was allowed")
        except SystemExit:
            pass
    print("ok  pipeline s0 -> s1, once only")

    # a clock failing check 3 is run but not judged
    cfg = mini_cfg()
    cfg["op_min"] = 9.0
    work, res = os.path.join(scratch, "work2"), os.path.join(scratch, "results2")
    R.stage_s0(cfg, work, res, state)
    R.stage_s1(cfg, work, res, state)
    s1 = R.load(res, "s1.json")
    assert all(v["verdict"].startswith("判定しない") for v in s1["comparisons"].values())
    # clocks excluded by check 2 -> stop
    cfg = mini_cfg()
    cfg["half_tol"] = -1.0
    work, res = os.path.join(scratch, "work3"), os.path.join(scratch, "results3")
    R.stage_s0(cfg, work, res, state)
    s0 = R.load(res, "s0.json")
    assert not s0["proceed"] and s0["kept_clocks"] == [] and "no clock" in s0["stopped"]
    # CPU cap -> stop
    cfg = mini_cfg()
    cfg["cpu_cap_s"] = 0.0
    work, res = os.path.join(scratch, "work4"), os.path.join(scratch, "results4")
    R.stage_s0(cfg, work, res, state)
    assert "CPU time cap" in R.load(res, "s0.json")["stopped"]
    # failing runs are recorded, not raised
    cfg = dict(mini_cfg(), fake=False, price_lo=1000.0, k=200, burn=100)
    r = L.run_job((cfg, "B0", 31, None, None))
    assert not r["ok"] and "out of range" in r["error"]
    json.dumps(r)
    print("ok  stop paths")


if __name__ == "__main__":
    scratch = sys.argv[1]
    os.makedirs(scratch, exist_ok=True)
    assert not os.listdir(scratch), "use an empty directory"
    test_equivalence()
    test_window_selection()
    test_metrics()
    test_clock()
    test_pipeline(scratch)
    print("all tests passed")
