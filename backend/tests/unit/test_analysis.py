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
    weak = {"linear": {"r2": 0.037}, "persistence": {"r2": -0.7}, "hgb": {**g(0.003, -0.001), "r2": 0.04}}
    assert _verdict(weak, False, ("hgb",), 0.02, 0.95, 0.99, 0.05)[0] == "unpredictable"
    # Only the nonlinear model has signal (Phase 10A trend_change before trend features): never "unpredictable".
    nonlin = {"linear": {"r2": 0.02}, "persistence": {"r2": 0.0}, "hgb": {**g(0.48, 0.1), "r2": 0.49}}
    assert _verdict(nonlin, False, ("hgb",), 0.02, 0.95, 0.99, 0.05)[0] == "headroom"
    overlap = {"linear": {"r2": 0.3}, "persistence": {"r2": 0.0}, "hgb": {**g(0.9, 0.1), "r2": 0.97}}
    assert _verdict(overlap, False, ("hgb",), 0.02, 0.95, 0.99, 0.05)[0] == "trivial"



def test_return_and_regime_targets():
    X = np.tile(np.r_[0.01, -0.01], 30)[None, :].repeat(2, 0)          # a pure range: ER 0
    F = np.array([np.full(20, 0.01), np.r_[np.full(10, 0.01), np.full(10, -0.01)]])
    np.testing.assert_allclose(make_target("future_return", X, F, 20), [0.2, 0.0], atol=1e-12)
    d = make_target("direction", X, F, 20)
    assert d[0] == 1 and np.isnan(d[1])                                 # flat future is undefined
    np.testing.assert_allclose(make_target("trend_er", X, F, 20), [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(make_target("trend_change", X, F, 20), [1.0, 0.0], atol=1e-12)
    assert make_target("extreme", X, F * 2, 20).tolist() == [1, 1]       # 0.02 > the 0.01 |r| quantile
    with pytest.raises(ValueError):
        make_target("trend_change", X[:, :10], F, 20)                   # needs horizon <= obs


def _ar_series(phi=-0.3, n_days=40, per_day=400, seed=0):
    rng = np.random.default_rng(seed)
    idx, vals, price = [], [], 100.0
    for d in range(n_days):
        e = rng.normal(0, 1e-3, per_day)
        r = np.empty(per_day)
        r[0] = e[0]
        for k in range(1, per_day):
            r[k] = phi * r[k - 1] + e[k]
        p = price * np.exp(np.cumsum(r))
        price = p[-1]
        idx.append(pd.date_range(pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), periods=per_day, freq="1min"))
        vals.append(p)
    return pd.Series(np.concatenate(vals), index=idx[0].append(idx[1:]))


def test_assess_finds_return_signal_through_lagged_returns():
    s = _ar_series()
    r = assess_target(s, target="future_return", horizon=1, obs=20, with_time=False, models=("hgb",), n_boot=100)
    lin = r["metrics"]["linear"]
    assert lin["r2"] > 0.05 and lin["reference"] == "persistence" and lin["ci95_vs_reference"][0] > 0
    assert r["verdict"] != "unpredictable"
    r = assess_target(s, target="direction", horizon=1, obs=20, with_time=False, models=("hgb",), n_boot=100)
    assert r["metrics"]["linear"]["auc"] > 0.55 and r["metrics"]["linear"]["ci95_vs_reference"][0] > 0


def test_assess_drops_undefined_windows():
    s = _ar_series(phi=0.0, n_days=30, per_day=300)
    s = (s * 100).round() / 100                                       # coarse ticks: many flat futures
    r = assess_target(s, target="direction", horizon=1, obs=20, with_time=False, models=("hgb",), n_boot=50)
    w = build_return_windows(s, obs=20, horizon=1)
    assert r["n_windows"] < len(w) and 0.3 < r["positive_rate"] < 0.7


def test_trend_change_baseline_sees_the_past_trend():
    """trend_change subtracts the past efficiency ratio -- a nonlinear function of the inputs. The
    baselines get it as a feature, so a random walk shows no nonlinear 'headroom'."""
    s = _ar_series(phi=0.0, n_days=30, per_day=400)
    r = assess_target(s, target="trend_change", horizon=20, obs=60, with_time=False, models=("hgb",), n_boot=100)
    assert r["metrics"]["linear"]["r2"] > 0.3                     # the mechanical -past ER part
    assert r["verdict"] == "no_headroom", r["reason"]


def _lead_lag(n_days=30, per_day=400, beta=0.5, seed=0):
    """Instrument B follows A with a one-bar lag: b_t = beta * a_{t-1} + noise."""
    rng = np.random.default_rng(seed)
    idx, pa, pb, la, lb = [], [], [], 100.0, 50.0
    for d in range(n_days):
        a = rng.normal(0, 1e-3, per_day)
        b = beta * np.r_[0.0, a[:-1]] + rng.normal(0, 1e-3, per_day)
        pa.append(la * np.exp(np.cumsum(a))); pb.append(lb * np.exp(np.cumsum(b)))
        la, lb = pa[-1][-1], pb[-1][-1]
        idx.append(pd.date_range(pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), periods=per_day, freq="1min"))
    ix = idx[0].append(idx[1:])
    return pd.Series(np.concatenate(pa), index=ix), pd.Series(np.concatenate(pb), index=ix)


def test_exog_features_read_only_the_past():
    from model_core.analysis import align_closes
    from model_core.analysis.features import exog_features
    a, b = _lead_lag(n_days=3, per_day=200)
    close, ex = align_closes(b, {"A": a})
    w = build_return_windows(close, obs=20, horizon=5, exog=ex)
    E = exog_features(w)
    ra = np.diff(np.log(ex["A"].to_numpy()))
    k = 17
    t = w.t[k]
    assert E[k, 0] == pytest.approx(ra[t])                       # 1-return sum = r_t itself
    assert E[k, 2] == pytest.approx(ra[t - 4:t + 1].sum())       # 5-return sum ends at r_t
    np.testing.assert_allclose(w.X[k], np.diff(np.log(close.to_numpy()))[t - 19:t + 1])


def test_align_closes_fills_only_forward_and_drops_rest():
    from model_core.analysis import align_closes
    ix = pd.date_range("2024-01-01", periods=6, freq="1min")
    main = pd.Series(np.arange(1.0, 7.0), index=ix)
    other = pd.Series([10.0, np.nan, np.nan, 13.0, np.nan, 15.0], index=ix).dropna()
    c, ex = align_closes(main, {"o": other}, ffill_limit=1)
    assert list(ex["o"]) == [10.0, 10.0, 13.0, 13.0, 15.0] and len(c) == 5    # row 2 (2 bars stale) dropped


def test_assess_detects_signal_from_another_instrument():
    from model_core.analysis import align_closes
    a, b = _lead_lag()
    close, ex = align_closes(b, {"A": a})
    r = assess_target(close, target="future_return", horizon=1, obs=20, with_time=False, models=("hgb",),
                      n_boot=100, exog=ex)
    lin = r["metrics"]["linear"]
    assert r["exog"] == ["A"] and r["metrics"]["linear_own"]["r2"] < 0.02
    assert lin["r2"] > 0.15 and lin["exog_ci95"][0] > 0


def test_chronological_split_purges_before_test():
    from model_core.analysis import chronological_split
    day = np.tile(np.arange(100), 3)                      # a 3-instrument panel, 100 days each
    tr, te = chronological_split(day, test_frac=0.2, purge_days=10)
    assert set(day[te]) == set(range(80, 100)) and day[tr].max() == 69
    with pytest.raises(ValueError):
        chronological_split(np.arange(20), test_frac=0.5, purge_days=15)


def _daily(n=900, seed=0, vol_regimes=True):
    """Business-day bars (weekends are 3-day steps) with slowly switching volatility."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    sig = 0.01 * np.where((np.arange(n) // 60) % 2 == 0, 1.0, 3.0) if vol_regimes else np.full(n, 0.01)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 1, n) * sig)), index=idx)


def test_daily_windows_need_max_gap():
    s = _daily(200)
    assert len(build_return_windows(s, obs=20, horizon=5)) < 20            # every weekend is a gap
    assert len(build_return_windows(s, obs=20, horizon=5, max_gap="5D")) == 199 - 25 + 1


def test_panel_assessment_with_market_features():
    closes = {f"A{i}": _daily(900, seed=i) for i in range(4)}
    r = assess_target(closes, target="future_log_rv", horizon=5, obs=20, with_time=False, models=("hgb",),
                      split="chronological", bootstrap_group="month", max_gap="5D", market=True, n_boot=100)
    assert r["instruments"] == ["A0", "A1", "A2", "A3"] and r["exog"] == ["market"]
    assert r["metrics"]["linear"]["r2"] > 0.3 and "exog_ci95" in r["metrics"]["linear"]
    assert r["test_period"][0] > "2017-06"                                # the last 20% of dates
    with pytest.raises(ValueError):
        assess_target(closes["A0"], market=True)


def test_holdout_split_purges_both_sides():
    from model_core.analysis.splits import holdout_split
    days = pd.bdate_range("2020-01-01", periods=100)
    days = days.as_unit("ns")
    day = np.tile(np.asarray(days.asi8), 2)
    tr, te = holdout_split(day, days[40], days[60], purge_days=5)
    test_days = set(day[te])
    assert test_days == set(days[40:60].asi8)
    assert set(day[tr]) == set(days[:35].asi8) | set(days[65:].asi8)


def _volume_driven(n_days=30, per_day=400, seed=0):
    """Volatility follows an observed AR(1) 'log volume' v: r_k ~ N(0, 1e-3 * exp(v_k))."""
    rng = np.random.default_rng(seed)
    idx, px, vol, price = [], [], [], 100.0
    for d in range(n_days):
        v = np.empty(per_day)
        v[0] = rng.normal()
        for k in range(1, per_day):
            v[k] = 0.97 * v[k - 1] + 0.25 * rng.normal()
        r = rng.normal(0, 1, per_day) * 1e-3 * np.exp(0.8 * v)
        p = price * np.exp(np.cumsum(r))
        price = p[-1]
        idx.append(pd.date_range(pd.Timestamp("2024-01-01") + pd.Timedelta(days=d), periods=per_day, freq="1min"))
        px.append(p)
        vol.append(v)
    ix = idx[0].append(idx[1:])
    return pd.Series(np.concatenate(px), index=ix), pd.DataFrame({"log_volume": np.concatenate(vol)}, index=ix)


def test_aux_features_and_target_align_with_the_anchor():
    from model_core.analysis.features import aux_features, future_aux_mean, past_aux_mean
    close, aux = _volume_driven(n_days=2, per_day=200)
    w = build_return_windows(close, obs=20, horizon=5, aux=aux)
    v = aux["log_volume"].to_numpy()
    k = 11
    row = close.index.get_loc(w.anchor_ts[k])                     # the anchor bar
    assert aux_features(w)[k, 0] == pytest.approx(v[row])            # 1-bar mean = the anchor bar
    assert past_aux_mean(w, 5)[k] == pytest.approx(v[row - 4:row + 1].mean())
    assert future_aux_mean(w, 5)[k] == pytest.approx(v[row + 1:row + 6].mean())


def test_assess_uses_aux_inputs_and_forecasts_aux():
    close, aux = _volume_driven()
    r = assess_target(close, target="future_log_rv", horizon=20, obs=20, with_time=False, models=("hgb",),
                      n_boot=100, aux=aux)
    assert r["exog"] == ["log_volume"] and r["metrics"]["linear"]["exog_ci95"][0] > 0
    r = assess_target(close, target="future_aux", horizon=20, obs=20, with_time=False, models=("hgb",),
                      n_boot=100, aux=aux)
    assert r["metrics"]["linear"]["r2"] > 0.5 and r["metrics"]["persistence"]["r2"] > 0.3
    with pytest.raises(ValueError):
        assess_target(close, target="future_aux")


def test_aux_hour_interactions_shape():
    from model_core.analysis.features import aux_hour_interactions
    close, aux = _volume_driven(n_days=2, per_day=200)
    w = build_return_windows(close, obs=20, horizon=5, aux=aux)
    H = aux_hour_interactions(w)
    assert H.shape == (len(w), 24) and np.count_nonzero(H[0]) <= 1
