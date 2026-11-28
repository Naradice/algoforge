"""P0 checks of the DP3 code on short synthetic inputs (not an experiment).

    python dp3_test.py <empty scratch directory>

The experimental simulations are replaced by a random walk (cfg["fake"]). The simulator is used only for the
equivalence check and a short crash check with non-experimental seeds; no statistic is computed from them.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import dp3_run as P
import dp_lib as L
import dp_run as R1
import numpy as np


def test_coefficients():
    cfg = P.CFG
    base, zero, mix = (P.coefficients(cfg, 17, c) for c in P.CONDS)
    assert base.shape == zero.shape == mix.shape == (300,)
    assert np.all((base <= -1.5) & (base >= -3.5)) and np.all(zero == 0)
    assert np.array_equal(np.abs(mix), -base) and (mix > 0).sum() == 150 and (mix < 0).sum() == 150
    assert not np.array_equal(base, P.coefficients(cfg, 18, "base"))
    assert np.array_equal(mix, P.coefficients(cfg, 17, "mix"))
    print("ok  coefficients")


def test_equivalence_and_models():
    for s in (11, 12):
        row = P.equivalence(P.CFG, s, 800)
        assert row["equal"] and row["price_changes"] > 100 and row["d_max"] < 0, row
    cfg = copy.deepcopy(P.CFG)
    cfg.update(burn=100, k=200)
    for c in P.CONDS:
        model = L.DDMv3(**P.D.uncoupled_model_kwargs(cfg), dealer_sensitive=P.coefficients(cfg, 11, c))
        assert np.array_equal(model.dealer_sensitive, P.coefficients(cfg, 11, c)) and model.wma == 5
        assert P.run_job((cfg, c, 11, None))["cond"] == c
    bad = dict(cfg, price_lo=1000.0)
    out = P.run_job((bad, "base", 11, None))
    assert not out["ok"] and "out of range" in out["error"]
    print("ok  equivalence and model construction")


def test_comparison():
    cfg = copy.deepcopy(P.CFG)
    pairs = ((0.21, 0.5), (-0.5, -0.21), (-0.19, 0.19), (0.1, 0.3), (-0.3, 0.0), (0.2, 0.4))
    assert [P.judge_delta(a, b, cfg) for a, b in pairs] == \
        ["打ち消しが大きく減る", "打ち消しが大きく増える", "大きな変化なし", "判定保留", "判定保留", "判定保留"]
    assert [P.hypothesis_from(v) for v in ("打ち消しが大きく減る", "大きな変化なし", "打ち消しが大きく増える", "判定保留",
                                           "判定しない(前提を満たさない)", "評価不能(安定性パイロットで除外)")] == \
        ["支持", "不支持", "不支持", "保留", "保留", "保留"]
    rng = np.random.default_rng(1)

    def eff(g, rw):
        return {"g": g + rng.normal(0, 0.003, 20), "rw": rw + rng.normal(0, 0.003, 20)}

    out = P.compare(eff(0.04, 0.16), eff(0.10, 0.16), cfg)
    assert out["precondition"] and out["verdict"] == "打ち消しが大きく減る"
    assert np.isclose(out["delta"]["value"], out["rho"]["treat"]["value"] - out["rho"]["base"]["value"])
    assert out["delta"]["lo"] < out["delta"]["value"] < out["delta"]["hi"]
    assert P.compare(eff(0.04, 0.16), eff(0.042, 0.16), cfg)["verdict"] == "大きな変化なし"
    out = P.compare(eff(0.04, 0.16), eff(0.001, 0.01), cfg)
    assert not out["precondition"] and "delta" not in out
    json.dumps(out, ensure_ascii=False)
    cfg.update(burn=500, k=20_000)
    prices = 101 + np.cumsum(rng.normal(0, 0.002, cfg["burn"] + cfg["k"]))
    st = P.run_stats(prices, np.exp(rng.normal(1.0, 1.0, len(prices))), cfg)
    assert 0.5 < st["vr_all"] < 1.5 and np.isfinite(st["kurtosis_60s"])
    ps = P.price_stats(prices, cfg)
    assert abs(ps["trade_ret_acf"]["1"]) < 0.05 and ps["price_min"] <= ps["price_max"]
    print("ok  comparison, judgement, statistics")


def test_pipeline(scratch):
    os.makedirs(scratch, exist_ok=True)
    lf = np.random.default_rng(5).normal(size=4000)
    lf -= lf.mean()
    pool = os.path.join(scratch, "pool.npy")
    np.save(pool, lf)
    ref = os.path.join(scratch, "ref.json")

    def write_ref(sha):
        with open(ref, "w") as fh:
            json.dump({"calib": {"K-long": {"n": 50, "lam": 0.03, "kappa": 2.0, "m0": 1.5}},
                       "checks": {"2_clock": {"pool": {"sha256": sha}}}}, fh)

    write_ref(R1.sha256(pool))
    base = copy.deepcopy(P.CFG)
    base.update(fake=True, k=60_000, burn=2000, main_seeds=list(range(1, 7)), pilot_seeds=[31, 32], equiv_seeds=[11],
                equiv_trades=500, min_pairs=5, n_boot=500, neg_band=0.05, half_tol=5.0, op_min=-1.0, pre_min=-9.0,
                workers=4, dp1_s0=ref, pool_path=pool)
    state = {"head": "test", "time": "test"}
    work, res = os.path.join(scratch, "work"), os.path.join(scratch, "results")
    P.stage_s0(base, work, res, state)
    s0 = R1.load(res, "s0.json")
    assert s0["proceed"] and s0["run_conditions"] == ["zero", "mix"], s0["stopped"]
    assert not any(f.startswith(("zero_", "mix_")) for f in os.listdir(work)), "pilot runs must not be stored"
    try:
        P.stage_s1(base, work, res, state, changed="x.py | 1 +")
        raise AssertionError("changed code was accepted")
    except SystemExit:
        pass
    P.stage_s1(base, work, res, state)
    s1 = R1.load(res, "s1.json")
    assert s1["stopped"] is None and set(s1["comparisons"]) == {"zero", "mix"} and "delta" in s1["comparisons"]["zero"]
    assert set(s1["summary"]) == set(P.CONDS) and s1["hypothesis"] in ("支持", "不支持", "保留")
    for fn in (P.stage_s0, P.stage_s1):
        try:
            fn(base, work, res, state)
            raise AssertionError("second run was allowed")
        except SystemExit:
            pass
    print("ok  pipeline s0 -> s1, once only")
    # a condition failing the pilot is dropped; its comparison cannot be evaluated
    cfg = dict(base, fake_fail=["zero"])
    work, res = os.path.join(scratch, "w_zf"), os.path.join(scratch, "r_zf")
    P.stage_s0(cfg, work, res, state)
    P.stage_s1(cfg, work, res, state)
    s1 = R1.load(res, "s1.json")
    assert s1["comparisons"]["zero"]["verdict"].startswith("評価不能") and s1["hypothesis"] == "保留"
    assert "delta" in s1["comparisons"]["mix"]
    for key, value, text in (("fake_fail", ["base"], "base is unstable"), ("fake_fail", ["zero", "mix"], "no comparison"),
                             ("op_min", 9.0, "check 3"), ("half_tol", -1.0, "check 2"), ("neg_band", 0.0, "check 5"),
                             ("cpu_cap_s", 0.0, "CPU time cap")):
        cfg = dict(base, **{key: value})
        name = f"{key}_{text[:5].strip().replace(' ', '')}"
        w, r = os.path.join(scratch, "w_" + name), os.path.join(scratch, "r_" + name)
        P.stage_s0(cfg, w, r, state)
        out = R1.load(r, "s0.json")
        assert not out["proceed"] and text in out["stopped"], (key, out["stopped"])
    write_ref("0" * 64)
    w, r = os.path.join(scratch, "w_sha"), os.path.join(scratch, "r_sha")
    P.stage_s0(base, w, r, state)
    assert "check 1" in R1.load(r, "s0.json")["stopped"]
    print("ok  stop paths")


if __name__ == "__main__":
    scratch = sys.argv[1]
    os.makedirs(scratch, exist_ok=True)
    assert not os.listdir(scratch), "use an empty directory"
    test_coefficients()
    test_equivalence_and_models()
    test_comparison()
    test_pipeline(scratch)
    print("all tests passed")
