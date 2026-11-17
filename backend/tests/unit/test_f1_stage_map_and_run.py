"""Tests for F1 stage_map (build_mapping) and the score_month -> saved results -> analysis chain (Study Manager #889).
Requirement ids refer to docs/audit-f1-code-audit.md (R-09, R-12). All inputs are artificial."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RESEARCH = Path(__file__).resolve().parents[2] / "research"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH.parent))

import f1_analyze as fa  # noqa: E402
import f1_common as fc  # noqa: E402
import f1_ddm as fd  # noqa: E402


def make_sv_series(n, seed=0, start="2018-01-01", gap_at=None):
    """Random-walk close with persistent stochastic volatility (so that the ATR buckets are populated)."""
    rng = np.random.default_rng(seed)
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = 0.999 * lv[i - 1] + 0.03 * rng.standard_normal()
    close = 110.0 * np.exp(np.cumsum(1e-4 * np.exp(lv) * rng.standard_normal(n)))
    idx = pd.date_range(start, periods=n, freq="min")
    if gap_at is not None:
        idx = idx.where(np.arange(n) < gap_at, idx + pd.Timedelta(hours=60))
    return close, pd.DatetimeIndex(idx)


class Recorder:
    """Fake characterise/select functions that record exactly which close windows the mapping used."""

    def __init__(self):
        self.windows = []

    def characterize(self, df):
        w = df["close"].values.copy()
        self.windows.append(w)
        r = np.diff(np.log(w))
        return {"std": float(r.std()), "skewness": 0.0, "kurtosis": 0.0, "acf_1": 0.0, "acf_5": 0.0, "acf_20": 0.0,
                "volclust_50": 0.0}

    @staticmethod
    def select(real_row, atlas):
        cands = pd.DataFrame({"delta": [0.0, -0.1], "wma": [5, 7]})
        return {"candidates": cands, "best_std_ratio": 1.1, "out_of_range": False}


@pytest.fixture()
def est_period():
    old = dict(fc.PERIODS)
    fc.PERIODS["estimation"] = ("2018-01-01", "2018-02-05")         # artificial estimation period inside a 60-day series
    yield
    fc.PERIODS.clear()
    fc.PERIODS.update(old)


def build(close, idx, rec, **kw):
    return fd.build_mapping(fc.Series(close, idx), None, rec.characterize, rec.select, **kw)


# ----------------------------------------------------------------------------- R-09 stage_map
# R-09
def test_stage_map_windows_edges_and_forward_range(est_period):
    close, idx = make_sv_series(60 * 24 * 60, seed=1)
    s = fc.Series(close, idx)
    rec = Recorder()
    mp = build(close, idx, rec, max_per_bucket=25)
    atr = s.atr_window()
    est = s.period_mask("estimation") & s.atr_valid() & np.isfinite(atr)
    # (1) bucket edges are exactly the quantiles of the estimation-period block ATR
    assert len(mp["edges"]) == fc.N_QUANTILES - 1 and np.all(np.diff(mp["edges"]) >= 0)
    assert np.allclose(mp["edges"], np.quantile(atr[est], np.linspace(0, 1, fc.N_QUANTILES + 1)[1:-1]))
    est_hi = pd.Timestamp(fc.PERIODS["estimation"][1]) + pd.Timedelta(days=1)
    k = 0
    for b in mp["buckets"]:
        assert b["n_windows_used"] == min(b["n_windows_available"], 25) == len(b["used_blocks"])
        for j in b["used_blocks"]:
            a0 = j * fc.BLOCK
            # (2) origin in the estimation period and in the bucket given by the edges
            assert est[j] and int(np.digitize(atr[j], mp["edges"])) == b["q"]
            # (3) forward range = rows [60j, 60j+1000] (1001 closes): gap-free and ending inside the estimation period
            assert idx[a0 + 1000] < est_hi
            assert s.contiguous(np.array([a0]), np.array([a0 + 1000]))[0]
            # (4) the window given to the characteriser is exactly that slice, in the recorded order
            assert len(rec.windows[k]) == 1001 and np.array_equal(rec.windows[k], close[a0:a0 + 1001])
            k += 1
    assert k == len(rec.windows) > 0


# R-09
def test_stage_map_ignores_data_after_the_estimation_period_and_is_reproducible(est_period):
    close, idx = make_sv_series(60 * 24 * 60, seed=2)
    mp0 = build(close, idx, Recorder(), max_per_bucket=20)
    end_row = int(np.searchsorted(idx.values, np.datetime64("2018-02-06")))
    pert = close.copy()
    pert[end_row:] = close[end_row - 1] * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 5e-3, len(close) - end_row)))
    mp1 = build(pert, idx, Recorder(), max_per_bucket=20)
    assert mp0["edges"] == mp1["edges"]
    assert [b["used_blocks"] for b in mp0["buckets"]] == [b["used_blocks"] for b in mp1["buckets"]]
    assert [b["real_stats"] for b in mp0["buckets"]] == [b["real_stats"] for b in mp1["buckets"]]
    mp2 = build(close, idx, Recorder(), max_per_bucket=20)
    assert [b["used_blocks"] for b in mp0["buckets"]] == [b["used_blocks"] for b in mp2["buckets"]]      # same seed
    mp3 = build(close, idx, Recorder(), max_per_bucket=20, seed=1)
    assert [b["used_blocks"] for b in mp0["buckets"]] != [b["used_blocks"] for b in mp3["buckets"]]


# R-09
def test_stage_map_excludes_windows_with_gaps_and_windows_running_past_the_period_end(est_period):
    close, idx = make_sv_series(60 * 24 * 60, seed=3, gap_at=30000)           # a 60 h gap inside the estimation period
    s = fc.Series(close, idx)
    mp = build(close, idx, Recorder(), max_per_bucket=10_000)
    used = np.concatenate([np.array(b["used_blocks"], int) for b in mp["buckets"]])
    est_hi = pd.Timestamp(fc.PERIODS["estimation"][1]) + pd.Timedelta(days=1)
    # the gap is the step between rows 29999 and 30000; a window [a0, a0+1000] contains it iff a0 < 30000 <= a0+1000
    assert [j for j in used if j * fc.BLOCK < 30000 <= j * fc.BLOCK + 1000] == []
    assert 500 in used                                                                 # the window starting right after the gap is valid
    assert all(idx[j * fc.BLOCK + 1000] < est_hi for j in used)                         # none runs past the estimation end
    last_est_origin = np.flatnonzero(s.period_mask("estimation")).max()
    assert used.max() <= last_est_origin - 1000 // fc.BLOCK + 1                         # the last ~16 windows are purged
    assert len(used) > 100


# ----------------------------------------------------------------------------- R-12 score_month -> saved results -> analysis
@pytest.fixture()
def val_period():
    old = dict(fc.PERIODS)
    fc.PERIODS["estimation"] = ("2018-01-01", "2018-03-31")
    fc.PERIODS["validation"] = ("2018-04-11", "2018-04-17")
    fc.PERIODS["final"] = ("2018-04-11", "2018-04-17")
    yield
    fc.PERIODS.clear()
    fc.PERIODS.update(old)


# R-11
# R-12
def test_score_month_to_saved_results_end_to_end(val_period, tmp_path, monkeypatch):
    import f1_models as fm
    import f1_run as fr
    close, idx = make_sv_series(60 * 24 * 120, seed=4)
    s = fc.Series(close, idx)
    atr = s.atr_window()
    edges = np.quantile(atr[s.period_mask("estimation") & s.atr_valid()], np.linspace(0, 1, fc.N_QUANTILES + 1)[1:-1])
    ctx = fm.Context(close, idx, edges)
    rng = np.random.default_rng(0)
    scale = np.array([1, 2, 3.5, 5])[None, :, None]
    ens = [rng.standard_t(5, size=(fc.N_QUANTILES, 4, 1000)) * 1e-3 * scale for _ in range(3)]
    ms = pd.Timestamp("2018-04-01")
    mstr, out, fails = fr.score_month(ctx, ens, "validation", ms)
    assert fails == []
    models = fr.STAT_MODELS + ["DDM1", "DDM2", "DDM3"]
    assert {(m, h) for m in models for h in fc.HORIZONS} == set(out)
    # origin sets: identical across models and equal to an independent recomputation
    for h in fc.HORIZONS:
        in_month = np.asarray((ctx.tt >= ms) & (ctx.tt < ms + pd.offsets.MonthBegin(1)))
        expect = np.flatnonzero(ctx.origin_ok & ctx.s.period_mask("validation", h) & np.isfinite(ctx.targets[h]) & in_month)
        assert len(expect) > 50
        for m in models:
            assert np.array_equal(out[(m, h)]["j"], expect), (m, h)
            for k, v in out[(m, h)].items():
                assert np.all(np.isfinite(v)), (m, h, k)
            assert np.all(out[(m, h)]["crps"] >= 0)
            assert np.all((out[(m, h)]["pit"] >= 0) & (out[(m, h)]["pit"] <= 1))
    # wiring: recompute two scores directly from the sampler / the ensemble
    st = fm.build_state(ctx, ms, None)
    h, hi = 4, 1
    j0 = int(out[("B1", h)]["j"][0])
    y = np.array([ctx.targets[h][j0]])
    direct = fc.fair_crps(fm.SAMPLERS["B1"](st, ctx, j0)[hi][None, :], y)[0]
    assert out[("B1", h)]["crps"][0] == pytest.approx(direct, rel=1e-12)
    b0 = int(ctx.bucket[j0])
    assert out[("DDM2", h)]["crps"][0] == pytest.approx(fc.fair_crps(ens[1][b0][hi][None, :], y)[0], rel=1e-12)
    # determinism of the whole month
    _, out2, _ = fr.score_month(ctx, ens, "validation", ms)
    for key in out:
        for k in out[key]:
            assert np.array_equal(out[key][k], out2[key][k]), (key, k)
    # persistence and downstream analysis on the saved file
    save = fr.write_scores(str(tmp_path), "validation", {k: [v] for k, v in out.items()}, {mstr: fails})
    z = np.load(tmp_path / "scores_validation.npz")
    assert set(z.files) == set(save)
    assert all(np.all(np.diff(z[f"{m}|h{h}|j"]) > 0) for m in models for h in fc.HORIZONS)
    assert json.load(open(tmp_path / "failures_validation.json")) == {}
    shutil.copy(tmp_path / "scores_validation.npz", tmp_path / "scores_final.npz")
    monkeypatch.setattr(fc, "load_series", lambda: s)
    fa.stage_select(str(tmp_path))
    fa.stage_final(str(tmp_path))
    res = json.load(open(tmp_path / "final_result.json"))
    assert set(res["per_horizon"]) == {"1", "4", "12", "24"} and "DDM" in res["models_final_crps"]
    assert res["models_final_crps"]["B1"]["1"]["pit_mc"]["ks_sd"] >= 0


# ----------------------------------------------------------------------------- PIT randomisation spread
# R-14
def test_pit_randomisation_spread_is_zero_without_ties_and_positive_with_ties():
    lo = np.linspace(0, 1, 500, endpoint=False)
    assert fa.pit_randomisation_spread(lo, np.zeros(500))["ks_sd"] == 0.0
    spread = fa.pit_randomisation_spread(np.full(500, 0.5), np.full(500, 0.2))
    assert spread["ks_sd"] > 0 and spread["tie_share"] == 1.0
