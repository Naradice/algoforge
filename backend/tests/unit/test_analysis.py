"""model_core.analysis: windows, targets, splits, bootstrap and assess_target verdicts."""
import numpy as np
import pandas as pd
import pytest

from model_core.analysis import (
    assess_target, blocked_split, build_return_windows, make_target, paired_block_bootstrap_ci,
)


def _series(n_days=40, per_day=300, gap_every_day=True, vol_fn=None, seed=0):
    """Minute bars, one block per day (a gap between days), log-returns with optional volatility."""
    rng = np.random.default_rng(seed)
    idx, vals, price = [], [], 100.0
    for d in range(n_days):
        t = pd.date_range(pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), periods=per_day, freq="1min")
        sig = np.full(per_day, 1e-3) if vol_fn is None else vol_fn(d, per_day, rng)
        r = rng.normal(0, 1, per_day) * sig
        p = price * np.exp(np.cumsum(r))
        price = p[-1] * (1.2 if gap_every_day else 1.0)     # price jump across each gap
        idx.append(t)
        vals.append(p)
    return pd.Series(np.concatenate(vals), index=idx[0].append(idx[1:]))


def test_windows_never_span_a_gap():
    s = _series(n_days=3, per_day=100)
    w = build_return_windows(s, obs=10, horizon=5)
    assert len(w) == 3 * (99 - 10 - 5 + 1)
    assert np.abs(np.r_[w.X.ravel(), w.F.ravel()]).max() < 0.05    # the 20% cross-gap jump never appears
    t0 = s.index.get_loc(w.anchor_ts[0])
    r = np.diff(np.log(s.to_numpy()))
    np.testing.assert_allclose(w.X[0], r[t0 - 10:t0])
    np.testing.assert_allclose(w.F[0], r[t0:t0 + 5])


def test_targets_use_rms_and_future_only():
    X = np.ones((2, 60)) * 0.01
    F = np.c_[np.full((2, 20), 0.02), np.full((2, 20), 5.0)]
    np.testing.assert_allclose(make_target("future_log_rv", X, F, 20), np.log(0.02 + 1e-7))
    np.testing.assert_allclose(make_target("vol_change", X, F, 20), np.log(0.02 + 1e-7) - np.log(0.01 + 1e-7))
    assert make_target("jump", X, F, 40).tolist() == [1, 1]
    with pytest.raises(ValueError):
        make_target("future_log_rv", X, F, 50)


def test_blocked_split_is_day_disjoint_and_purged():
    day = np.repeat(np.arange(30), 50)
    tr, te = blocked_split(day, test_frac=0.2, purge=10, seed=1)
    assert not set(day[tr]) & set(day[te])
    assert np.abs(tr[:, None] - te[None, :]).min() > 10


def test_bootstrap_ci_detects_a_real_difference():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 20)
    a = rng.normal(1.0, 0.1, 1000)
    lo, hi = paired_block_bootstrap_ci(a, a - 0.05, groups)
    assert lo > 0.04 and hi < 0.06
    lo, hi = paired_block_bootstrap_ci(a, a, groups)
    assert lo == hi == 0


def test_assess_flags_a_nearly_determined_target_as_trivial():
    # volatility is constant within each day and differs across days: over a 60-return horizon the
    # future RMS is almost fully determined by the window's own RMS -> trivial
    s = _series(n_days=40, per_day=400, vol_fn=lambda d, n, rng: np.full(n, 1e-3 * 3.0 ** (d % 5)))
    res = assess_target(s, target="future_log_rv", horizon=60, obs=60, with_time=False,
                        models=("hgb",), n_boot=100)
    assert res["verdict"] == "trivial", res["reason"]


def test_assess_flags_iid_noise_as_unpredictable():
    s = _series(n_days=40, per_day=300)            # constant volatility: nothing to predict
    res = assess_target(s, target="vol_change", horizon=20, obs=60, with_time=False,
                        models=("hgb",), n_boot=100)
    assert res["verdict"] in ("unpredictable", "no_headroom"), res["reason"]
    assert res["metrics"]["linear"]["r2"] < 0.5


def _oscillating_vol_series(n_days=80, per_day=400, block=20, seed=3):
    """Each 20-bar block's volatility is an oscillating function of the previous block's realized
    RMS -- too wiggly for squares/interactions of log RMS, easy for a tree."""
    rng = np.random.default_rng(seed)
    idx, vals, price = [], [], 100.0
    for d in range(n_days):
        sig, rets = 1e-3, []
        for _ in range(per_day // block):
            r = rng.normal(0, 1, block) * sig
            rets.append(r)
            z = np.log(np.sqrt((r ** 2).mean()) / 1e-3)
            sig = 1e-3 * np.exp(1.2 * np.sin(10 * z))
        r = np.concatenate(rets)
        p = price * np.exp(np.cumsum(r))
        price = p[-1]
        idx.append(pd.date_range(pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), periods=len(r), freq="1min"))
        vals.append(p)
    return pd.Series(np.concatenate(vals), index=idx[0].append(idx[1:]))


def test_assess_finds_nonlinear_headroom():
    res = assess_target(_oscillating_vol_series(), target="future_log_rv", horizon=20, obs=60,
                        with_time=False, models=("hgb",), n_boot=100)
    assert res["verdict"] == "headroom", res["reason"]
    assert res["metrics"]["hgb"]["gain_rel"] > 0.02 and res["metrics"]["hgb"]["ci95"][0] > 0


def test_assess_classification_runs():
    res = assess_target(_series(n_days=40, per_day=300), target="jump", horizon=20, obs=60,
                        with_time=False, models=("hgb",), n_boot=100)
    assert res["task"] == "classification" and set(res["metrics"]) >= {"base_rate", "linear", "hgb"}


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        assess_target(_series(n_days=5), models=("xgboost",))


def test_verdict_distinguishes_tree_only_headroom():
    from model_core.analysis.assess import _verdict

    def g(rel, lo):
        return {"gain_rel": rel, "ci95": [lo, lo + 0.01], "r2": 0.7}
    base = {"linear": {"r2": 0.7}, "persistence": {"r2": 0.6}}
    tree_only = {**base, "hgb": g(0.05, 0.002), "mlp": g(-0.06, -0.01)}
    assert _verdict(tree_only, False, ("hgb", "mlp"), 0.02, 0.95, 0.99, 0.05)[0] == "tree_only_headroom"
    smooth = {**base, "hgb": g(0.05, 0.002), "mlp": g(0.03, 0.001)}
    assert _verdict(smooth, False, ("hgb", "mlp"), 0.02, 0.95, 0.99, 0.05)[0] == "headroom"
    weak = {"linear": {"r2": 0.037}, "persistence": {"r2": -0.7}, "hgb": g(0.003, -0.001)}
    assert _verdict(weak, False, ("hgb",), 0.02, 0.95, 0.99, 0.05)[0] == "unpredictable"

