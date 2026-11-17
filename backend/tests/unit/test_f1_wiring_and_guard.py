"""F1 (#889) tests that the production entry points call the tested functions with the right arguments, expectations built
independently of the implementation, the power of the leak test (a deliberately leaky sampler must be caught), and the run
guard (executed commit / dirty check). Requirement ids refer to docs/audit-f1-code-audit.md."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RESEARCH = Path(__file__).resolve().parents[2] / "research"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH.parent))

import f1_common as fc  # noqa: E402
import f1_ddm as fd  # noqa: E402
import f1_guard as fg  # noqa: E402
import f1_models as fm  # noqa: E402
import f1_run as fr  # noqa: E402


def make_sv_series(n, seed=0, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = 0.999 * lv[i - 1] + 0.03 * rng.standard_normal()
    close = 110.0 * np.exp(np.cumsum(1e-4 * np.exp(lv) * rng.standard_normal(n)))
    return close, pd.date_range(start, periods=n, freq="min")


@pytest.fixture()
def periods():
    old = dict(fc.PERIODS)
    fc.PERIODS["estimation"] = ("2018-01-01", "2018-03-31")
    fc.PERIODS["validation"] = ("2018-04-11", "2018-04-17")
    fc.PERIODS["final"] = ("2018-04-11", "2018-04-17")
    yield
    fc.PERIODS.clear()
    fc.PERIODS.update(old)


@pytest.fixture(scope="module")
def synthetic():
    """One scored synthetic month shared by several tests (about 30 s)."""
    old = dict(fc.PERIODS)
    fc.PERIODS["estimation"] = ("2018-01-01", "2018-03-31")
    fc.PERIODS["validation"] = ("2018-04-11", "2018-04-17")
    try:
        close, idx = make_sv_series(60 * 24 * 120, seed=4)
        s = fc.Series(close, idx)
        edges = np.quantile(s.atr_window()[s.period_mask("estimation") & s.atr_valid()],
                            np.linspace(0, 1, fc.N_QUANTILES + 1)[1:-1])
        ctx = fm.Context(close, idx, edges)
        rng = np.random.default_rng(0)
        ens = [rng.standard_t(5, size=(fc.N_QUANTILES, 4, 1000)) * 1e-3 * np.array([1, 2, 3.5, 5])[None, :, None] for _ in range(3)]
        ms = pd.Timestamp("2018-04-01")
        _, out, fails = fr.score_month(ctx, ens, "validation", ms)
        yield {"close": close, "idx": idx, "edges": edges, "ctx": ctx, "ens": ens, "ms": ms, "out": out, "fails": fails}
    finally:
        fc.PERIODS.clear()
        fc.PERIODS.update(old)


# ----------------------------------------------------------------------------- production wiring (R-09, R-12)
# R-09
def test_stage_map_calls_build_mapping_with_the_production_functions(tmp_path, monkeypatch):
    import ddm_ensemble_selection as des
    import ddm_regime_atlas as dra
    seen = {}
    sentinel = object()

    def fake_build(s, atlas, characterize, select, **kw):
        seen.update(series=s, atlas=atlas, characterize=characterize, select=select, kw=kw)
        return {"edges": [0.0] * 19, "buckets": [
            {"q": 0, "n_windows_used": 1, "n_windows_available": 2, "real_stats": {"std": 1e-4}, "candidates": [(0.0, 5)],
             "best_std_ratio": 1.0, "out_of_range": False}]}

    monkeypatch.setattr(fd, "build_mapping", fake_build)
    monkeypatch.setattr(fc, "load_series", lambda: sentinel)
    fd.stage_map(str(tmp_path))
    assert seen["series"] is sentinel
    assert seen["characterize"] is dra.full_characterize and seen["select"] is des.select_ensemble
    assert seen["atlas"].equals(pd.read_csv(Path(fd.HERE).parent / "ddm_regime_atlas.csv"))
    assert seen["kw"] == {}                                    # production uses the documented defaults
    assert fd.MAX_PER_BUCKET == 300 and fd.MAP_SEED == 20261117
    saved = json.load(open(tmp_path / "ddm_mapping.json"))
    assert saved["top_k"] == des.TOP_K and len(saved["edges"]) == 19


# R-12
def test_month_job_passes_the_loaded_objects_to_score_month(tmp_path, monkeypatch):
    ctx_s, ens_s, ms = object(), object(), pd.Timestamp("2020-10-01")
    calls = []
    monkeypatch.setattr(fr, "load_ctx", lambda wd: (ctx_s, None))
    monkeypatch.setattr(fr, "load_ens", lambda wd: ens_s)
    monkeypatch.setattr(fr, "score_month", lambda *a: calls.append(a) or ("m", {}, []))
    assert fr.month_job((str(tmp_path), "validation", ms)) == ("m", {}, [])
    assert calls == [(ctx_s, ens_s, "validation", ms)]


class _SerialExecutor:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, it):
        return map(fn, it)


def _canned_month(offset):
    models = fr.STAT_MODELS + ["DDM1", "DDM2", "DDM3"]
    out = {}
    for mi, m in enumerate(models):
        for h in fc.HORIZONS:
            j = np.arange(5) + offset
            out[(m, h)] = {"j": j, "crps": np.full(5, 1e-3 * (mi + 1)), "crps_plain": np.full(5, 2e-3), "pit": np.full(5, 0.5),
                           "pit_lo": np.full(5, 0.4), "pit_w": np.full(5, 0.2), "tie": np.zeros(5)}
    return out


# R-12
def test_stage_score_writes_what_write_scores_aggregates(tmp_path, monkeypatch):
    months = fr.fm.month_starts("validation")
    seen = []

    def fake_job(args):
        seen.append(args)
        k = len(seen)
        return str(args[2].date()), _canned_month(100 * k), []

    monkeypatch.setattr(fr, "ProcessPoolExecutor", _SerialExecutor)
    monkeypatch.setattr(fr, "month_job", fake_job)
    monkeypatch.setattr(fr, "require_leak_pass", lambda wd: {"pass": True})      # the precondition has its own tests below
    fr.stage_score(str(tmp_path), "validation")
    assert [a[2] for a in seen] == months and all(a[:2] == (str(tmp_path), "validation") for a in seen)
    z = np.load(tmp_path / "scores_validation.npz")
    j = z["B1|h1|j"]
    assert len(j) == 5 * len(months) and np.all(np.diff(j) > 0)                 # months concatenated, sorted by origin
    assert z["B1|h1|crps"].shape == j.shape and "DDM3|h24|pit_w" in z.files
    assert json.load(open(tmp_path / "failures_validation.json")) == {}


# ----------------------------------------------------------------------------- independent expectations
# R-02
# R-12
def test_origin_sets_equal_an_independent_brute_force_construction(synthetic, periods):
    close, idx, ms = synthetic["close"], synthetic["idx"], synthetic["ms"]
    out = synthetic["out"]
    ts = idx.values.astype("datetime64[s]").astype("int64")
    lo = pd.Timestamp("2018-04-11")
    hi = pd.Timestamp("2018-04-17") + pd.Timedelta(days=1)

    def contiguous(a, b):                                  # plain python: every step between rows a..b is <= 300 s
        return all(ts[k + 1] - ts[k] <= 300 for k in range(a, b))

    for h in fc.HORIZONS:
        expect = []
        for j in range(len(close) // 60):
            t = 60 * j + 59
            if j == 0 or t + 60 * h >= len(close):
                continue
            when = idx[t]
            if not (ms <= when < ms + pd.offsets.MonthBegin(1)) or not (lo <= when < hi):
                continue
            if idx[t + 60 * h] >= hi:
                continue
            if not contiguous(60 * j, t) or not contiguous(t - 60, t) or not contiguous(t, t + 60 * h):
                continue
            rv = sum((np.log(close[k + 1]) - np.log(close[k])) ** 2 for k in range(t - 60, t))
            if not rv > 0:
                continue
            expect.append(j)
        assert expect and np.array_equal(out[("B1", h)]["j"], np.array(expect)), h


# R-06
def test_scores_recomputed_by_brute_force(synthetic):
    out, ctx, ens = synthetic["out"], synthetic["ctx"], synthetic["ens"]
    h, hi = 4, 1
    j = int(out[("DDM1", h)]["j"][3])
    y = ctx.targets[h][j]
    x = ens[0][int(ctx.bucket[j])][hi]
    m = len(x)
    # the stored 1000-draw value against an independent evaluation (pairwise distances through numpy broadcasting)
    d = np.abs(x[:, None] - x[None, :])
    brute = np.mean(np.abs(x - y)) - d.sum() / (2 * m * (m - 1))
    assert out[("DDM1", h)]["crps"][3] == pytest.approx(brute, rel=1e-10)
    lo, hi90 = np.percentile(x, [5, 95])
    assert out[("DDM1", h)]["hit90"][3] == float(lo <= y <= hi90)
    assert out[("DDM1", h)]["pit"][3] == pytest.approx(np.mean(x < y), abs=1e-3)      # no ties: PIT = fraction of draws below y


# ----------------------------------------------------------------------------- leak test: wiring and power (R-10)
def _prepare_leak(synthetic, tmp_path, monkeypatch):
    ctx = synthetic["ctx"]
    df = pd.DataFrame({"close": synthetic["close"]}, index=synthetic["idx"])
    monkeypatch.setattr(fr, "load_ctx", lambda wd: (ctx, df))
    monkeypatch.setattr(fr, "load_ens", lambda wd: synthetic["ens"])
    monkeypatch.setattr(fr, "LEAK_ORIGINS", 3)
    json.dump({"edges": list(map(float, synthetic["edges"]))}, open(tmp_path / "ddm_mapping.json", "w"))


# R-10
def test_leak_job_passes_for_the_real_samplers(synthetic, periods, tmp_path, monkeypatch):
    _prepare_leak(synthetic, tmp_path, monkeypatch)
    period, n, bad = fr.leak_job((str(tmp_path), "validation", 11))
    assert n == 3 and bad == []


# R-10
def test_leak_job_detects_a_sampler_that_reads_the_future(synthetic, periods, tmp_path, monkeypatch):
    _prepare_leak(synthetic, tmp_path, monkeypatch)
    real_b1 = fm.SAMPLERS["B1"]

    def leaky(st, ctx, j):
        x = real_b1(st, ctx, j)
        return x + ctx.targets[1][j]                       # uses the realised next-block return: look-ahead

    monkeypatch.setitem(fm.SAMPLERS, "B1", leaky)
    period, n, bad = fr.leak_job((str(tmp_path), "validation", 11))
    assert {m for m, _ in bad} == {"B1"} and len(bad) == 3




# ----------------------------------------------------------------------------- R-10b: no score without a passing leak test
def _leak_record(tmp_path, **over):
    for f in fr.LEAK_BOUND:
        (tmp_path / f).write_bytes(f.encode())
    rec = {"pass": True, "head": "H" * 40, "bound_files": fr.leak_bound_files(str(tmp_path)), "results": []}
    rec.update(over)
    json.dump(rec, open(tmp_path / "leak_test.json", "w"))
    return rec


# R-10b
def test_stage_score_requires_a_passing_leak_test_on_this_commit_and_these_files(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "ProcessPoolExecutor", _SerialExecutor)
    monkeypatch.setattr(fr.fg, "current_head", lambda repo=None: "H" * 40)
    called = []
    monkeypatch.setattr(fr, "month_job", lambda a: called.append(a) or ("m", {}, []))
    with pytest.raises(SystemExit) as e:                                  # no leak_test.json at all
        fr.stage_score(str(tmp_path), "validation")
    assert "leak_test.json missing" in str(e.value) and called == []
    _leak_record(tmp_path, **{"pass": False})
    with pytest.raises(SystemExit) as e:
        fr.stage_score(str(tmp_path), "validation")
    assert "did not pass" in str(e.value) and called == []
    _leak_record(tmp_path, head="0" * 40)
    with pytest.raises(SystemExit) as e:
        fr.stage_score(str(tmp_path), "validation")
    assert "another commit" in str(e.value)
    _leak_record(tmp_path)
    (tmp_path / "ddm_ens_seed2.npy").write_bytes(b"changed after the leak test")
    with pytest.raises(SystemExit) as e:
        fr.stage_score(str(tmp_path), "validation")
    assert "changed or are missing" in str(e.value) and called == []
    (tmp_path / "ddm_ens_seed2.npy").unlink()                            # a missing ensemble file is also refused
    with pytest.raises(SystemExit):
        fr.require_leak_pass(str(tmp_path))
    _leak_record(tmp_path)
    assert fr.require_leak_pass(str(tmp_path))["pass"] is True


# R-10b
def test_stage_leak_records_the_commit_and_the_bound_files(synthetic, periods, tmp_path, monkeypatch):
    _prepare_leak(synthetic, tmp_path, monkeypatch)
    for i in range(1, 4):
        np.save(tmp_path / f"ddm_ens_seed{i}.npy", synthetic["ens"][i - 1])
    monkeypatch.setattr(fr, "ProcessPoolExecutor", _SerialExecutor)
    monkeypatch.setattr(fr.fg, "current_head", lambda repo=None: "H" * 40)
    fr.stage_leak(str(tmp_path))
    rec = json.load(open(tmp_path / "leak_test.json"))
    assert rec["pass"] is True and rec["head"] == "H" * 40 and set(rec["bound_files"]) == set(fr.LEAK_BOUND)
    assert fr.require_leak_pass(str(tmp_path))["pass"] is True
