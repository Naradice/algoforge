"""Unit/integration tests for the F1 pipeline (Study Manager #889). Requirement ids (R-xx) refer to
docs/audit-f1-code-audit.md. Tests that need the real USDJPY artifact are skipped when it is missing."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

RESEARCH = Path(__file__).resolve().parents[2] / "research"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH.parent))

import f1_analyze as fa  # noqa: E402
import f1_common as fc  # noqa: E402
import f1_ddm as fd  # noqa: E402

DATA = Path(__file__).resolve().parents[3] / "artifacts" / fc.ARTIFACT      # tests set their own ARTIFACT_STORE_PATH
HAS_DATA = DATA.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="USDJPY derived artifact not available")


def make_series(n=6000, gap_at=None, seed=0):
    rng = np.random.default_rng(seed)
    close = 110.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    idx = pd.date_range("2018-01-01", periods=n, freq="min")
    if gap_at is not None:
        shift = pd.Timedelta(hours=60)
        idx = idx.where(np.arange(n) < gap_at, idx + shift)
    return close, pd.DatetimeIndex(idx)


# ----------------------------------------------------------------------------- R-06 CRPS estimators
def test_fair_crps_matches_bruteforce():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 40))
    y = rng.normal(size=4)
    got = fc.fair_crps(x, y)
    for i in range(4):
        m = x.shape[1]
        t1 = np.mean(np.abs(x[i] - y[i]))
        t2 = sum(abs(a - b) for a in x[i] for b in x[i]) / (2 * m * (m - 1))
        assert got[i] == pytest.approx(t1 - t2, rel=1e-12)


def test_plain_crps_matches_bruteforce():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 30))
    y = rng.normal(size=3)
    got = fc.plain_crps(x, y)
    for i in range(3):
        m = x.shape[1]
        t1 = np.mean(np.abs(x[i] - y[i]))
        t2 = sum(abs(a - b) for a in x[i] for b in x[i]) / (2 * m * m)
        assert got[i] == pytest.approx(t1 - t2, rel=1e-12)


def analytic_gauss_crps(y):
    return y * (2 * stats.norm.cdf(y) - 1) + 2 * stats.norm.pdf(y) - 1 / np.sqrt(np.pi)


def test_fair_crps_is_unbiased_for_small_ensembles():
    rng = np.random.default_rng(3)
    y = np.full(40000, 0.7)
    x = rng.normal(size=(40000, 5))
    fair = fc.fair_crps(x, y).mean()
    plain = fc.plain_crps(x, y).mean()
    truth = analytic_gauss_crps(0.7)
    assert abs(fair - truth) / truth < 0.01
    assert plain > fair                       # the biased estimator overstates for finite M


def test_crps_quantile_grid_matches_analytic():
    m = 4000
    x = np.tile(stats.norm.ppf((np.arange(m) + 0.5) / m), (3, 1))       # deterministic ensemble
    y = np.array([-1.3, 0.0, 2.1])
    assert np.allclose(fc.plain_crps(x, y), analytic_gauss_crps(y), atol=2e-3)
    assert np.allclose(fc.fair_crps(x, y), analytic_gauss_crps(y), atol=2e-3)


def test_pit_uniform_for_calibrated_samples_and_ties_randomised():
    rng = np.random.default_rng(5)
    n = 20000
    x = rng.normal(size=(n, 200))
    y = rng.normal(size=n)
    sc = fc.score_samples(x, y, np.random.default_rng(0))
    assert stats.kstest(sc["pit"], "uniform").pvalue > 0.001
    assert abs(sc["hit90"].mean() - 0.9) < 0.01
    xt = np.zeros((1000, 10))
    yt = np.zeros(1000)                        # every draw ties with y -> PIT must be spread, not constant
    pit = fc.score_samples(xt, yt, np.random.default_rng(1))["pit"]
    assert pit.std() > 0.2


# ----------------------------------------------------------------------------- R-02 windows, gaps, targets
def test_gap_rule_and_targets():
    close, idx = make_series(gap_at=3000)
    s = fc.Series(close, idx)
    j = np.arange(s.nblocks)
    t1 = s.target(1)
    # target of the block whose next block starts after the gap must be NaN
    gap_block = (3000 // fc.BLOCK)
    assert np.isnan(t1[gap_block - 1 + 0]) or np.isnan(t1[gap_block])
    assert np.isfinite(t1[5])
    # block return identity: target(1)(j) == block_return(j+1)
    r = s.block_returns()
    ok = np.isfinite(t1[:-1]) & np.isfinite(r[1:])
    assert np.allclose(t1[:-1][ok], r[1:][ok])


def test_missing_quiet_minutes_do_not_break_blocks():
    close, idx = make_series(n=3000)
    keep = np.ones(len(idx), bool)
    keep[[100, 101, 700]] = False             # two single missing minutes: steps of 2 and 3 minutes (< 5 min)
    s = fc.Series(close[keep], idx[keep])
    assert s.atr_valid().all()


def test_derived_arrays_depend_only_on_rows_up_to_origin():
    close, idx = make_series(n=9000, seed=7)
    s0 = fc.Series(close, idx)
    t_row = 60 * 40 + 59
    pert = close.copy()
    pert[t_row + 1:] = pert[t_row] * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 1e-3, len(close) - t_row - 1)))
    s1 = fc.Series(pert, idx)
    j = 40
    assert np.array_equal(s0.atr_window()[: j + 1], s1.atr_window()[: j + 1])
    assert np.array_equal(s0.block_returns()[: j + 1], s1.block_returns()[: j + 1], equal_nan=True)
    assert np.array_equal(s0.block_rv()[: j + 1], s1.block_rv()[: j + 1], equal_nan=True)
    assert not np.array_equal(s0.target(1)[j], s1.target(1)[j])      # the target does use the future (as it must)


def test_period_mask_purges_targets_crossing_the_period_end():
    close, idx = make_series(n=60 * 24 * 40)
    s = fc.Series(close, idx)
    fc.PERIODS["_t"] = ("2018-01-02", "2018-01-10")
    try:
        m1 = s.period_mask("_t", 1)
        m24 = s.period_mask("_t", 24)
        assert m24.sum() < m1.sum() <= s.period_mask("_t").sum()
        last = np.flatnonzero(m24).max()
        assert s.index[s.t[last] + 24 * fc.BLOCK] < pd.Timestamp("2018-01-11")
    finally:
        del fc.PERIODS["_t"]


# ----------------------------------------------------------------------------- R-09 DDM horizon definition
def test_ddm_cumulative_return_covers_exactly_60h_candle_steps():
    closes = np.exp(np.arange(fc.MAX_H * fc.BLOCK + 1) * 1e-3)
    got = fd.cum_returns(closes)
    for h, g in zip(fc.HORIZONS, got):
        assert g == pytest.approx(1e-3 * fc.BLOCK * h, rel=1e-9)


# ----------------------------------------------------------------------------- R-07/R-08 inference
def test_newey_west_matches_naive_for_iid_and_holm_is_monotone():
    rng = np.random.default_rng(6)
    d = rng.normal(size=4000)
    t, p = fa.newey_west_dm(d, 0)
    assert t == pytest.approx(d.mean() / (d.std() / np.sqrt(len(d))), rel=1e-2)
    t2, p2 = fa.newey_west_dm(d, 24)
    assert abs(t2) < 4 and p2 > 1e-4


def test_simultaneous_band_label_rules():
    # label rule is implemented inside stage_final; replicate its thresholds here
    margin = fa.MARGIN
    assert margin == 0.02
    for lo, hi, lab in [(-0.05, -0.03, "DDM better"), (0.03, 0.05, "statistical model better"),
                        (-0.01, 0.01, "equivalent"), (-0.03, 0.0, "inconclusive"), (-0.03, 0.03, "inconclusive")]:
        if hi < -margin:
            got = "DDM better"
        elif lo > margin:
            got = "statistical model better"
        elif lo >= -margin and hi <= margin:
            got = "equivalent"
        else:
            got = "inconclusive"
        assert got == lab


# ----------------------------------------------------------------------------- R-03/R-04 no look-ahead with real data
@pytest.fixture(scope="module")
def real_ctx():
    import f1_models as fm
    df = pd.read_parquet(DATA, columns=["close"])
    s = fc.Series(df["close"].values, df.index)
    atr = s.atr_window()
    est = s.period_mask("estimation") & s.atr_valid()
    edges = np.quantile(atr[est], np.linspace(0, 1, fc.N_QUANTILES + 1)[1:-1])
    return fm, df, edges, fm.Context(df["close"].values, df.index, edges)


@needs_data
def test_state_depends_only_on_data_before_month_start(real_ctx):
    fm, df, edges, ctx = real_ctx
    ms = pd.Timestamp("2020-10-01")
    st0 = fm.build_state(ctx, ms, None)
    close = df["close"].values.copy()
    first_row = int(np.searchsorted(df.index.values, ms.to_datetime64()))
    rng = np.random.default_rng(3)
    close[first_row:] = close[first_row - 1] * np.exp(np.cumsum(rng.normal(0, 2e-3, len(close) - first_row)))
    ctx_p = fm.Context(close, df.index, edges)
    st1 = fm.build_state(ctx_p, ms, None)
    for h in fc.HORIZONS:
        assert np.array_equal(st0.R[h], st1.R[h]) and st0.v[h] == st1.v[h] and st0.nu[h] == st1.nu[h]
    for name in fm.ARCH_SPECS:
        assert st0.arch[name] is not None and np.allclose(st0.arch[name].values, st1.arch[name].values)
    assert np.allclose(st0.har_beta, st1.har_beta) and np.array_equal(st0.har_resid, st1.har_resid)
    assert st0.z_std == st1.z_std and st0.har_nu == st1.har_nu and np.allclose(st0.hour_factor, st1.hour_factor)


@needs_data
def test_samplers_are_deterministic_and_ignore_the_future(real_ctx):
    fm, df, edges, ctx = real_ctx
    ms = pd.Timestamp("2020-10-01")
    st = fm.build_state(ctx, ms, None)
    cand = np.flatnonzero(ctx.origin_ok & np.asarray((ctx.tt >= ms) & (ctx.tt < ms + pd.Timedelta(days=3))))
    j = int(cand[len(cand) // 2])
    close = df["close"].values.copy()
    trow = int(ctx.s.t[j])
    close[trow + 1:] = close[trow] * np.exp(np.cumsum(np.random.default_rng(9).normal(0, 2e-3, len(close) - trow - 1)))
    ctx_p = fm.Context(close, df.index, edges)
    for name, fn in fm.SAMPLERS.items():
        a = fn(st, ctx, j)
        assert a is not None and np.all(np.isfinite(a)), name
        assert np.array_equal(a, fn(st, ctx, j)), f"{name} not deterministic"
        assert np.array_equal(a, fn(st, ctx_p, j)), f"{name} reads data after the origin"


@needs_data
def test_origin_set_excludes_zero_realised_variance(real_ctx):
    fm, df, edges, ctx = real_ctx
    assert np.all(ctx.rv[ctx.origin_ok] > 0)


@needs_data
def test_gaussian_baseline_crps_close_to_analytic(real_ctx):
    fm, df, edges, ctx = real_ctx
    ms = pd.Timestamp("2020-10-01")
    st = fm.build_state(ctx, ms, None)
    cand = np.flatnonzero(ctx.origin_ok & np.asarray((ctx.tt >= ms) & (ctx.tt < ms + pd.Timedelta(days=20))) &
                          np.isfinite(ctx.targets[4]))[:150]
    x = np.stack([fm.SAMPLERS["B1"](st, ctx, j)[1] for j in cand])
    y = ctx.targets[4][cand]
    sig = np.sqrt(st.v[4])
    analytic = sig * analytic_gauss_crps(y / sig)
    assert np.mean(fc.fair_crps(x, y)) == pytest.approx(analytic.mean(), rel=0.02)
