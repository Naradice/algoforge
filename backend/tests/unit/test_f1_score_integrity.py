"""F1 (#889) R-12 (Stage C finding 3): scores are validated before they are saved. Any failure, missing (model, horizon),
differing origin set or non-finite score stops the run and no scores_<period>.npz is written."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH = Path(__file__).resolve().parents[2] / "research"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(RESEARCH.parent))

import f1_common as fc  # noqa: E402
import f1_run as fr  # noqa: E402

MODELS = fr.STAT_MODELS + [f"DDM{i + 1}" for i in range(fr.N_DDM_SEEDS)]


def month(offset=0, n=5):
    out = {}
    for mi, m in enumerate(MODELS):
        for h in fc.HORIZONS:
            out[(m, h)] = {"j": np.arange(n) + offset, "crps": np.full(n, 1e-3 * (mi + 1)), "crps_plain": np.full(n, 2e-3),
                           "pit": np.full(n, 0.5), "pit_lo": np.full(n, 0.4), "pit_w": np.full(n, 0.2), "tie": np.zeros(n)}
    return out


def results_of(*months):
    res = {}
    for mo in months:
        for k, sc in mo.items():
            res.setdefault(k, []).append(sc)
    return res


def no_scores_written(tmp_path):
    return not (tmp_path / "scores_validation.npz").exists()


# R-12
def test_clean_results_are_saved_and_the_diagnostics_say_so(tmp_path):
    save = fr.write_scores(str(tmp_path), "validation", results_of(month(0), month(100)), {"2020-10-01": [], "2020-11-01": []})
    assert "B1|h1|crps" in save and (tmp_path / "scores_validation.npz").exists()
    diag = json.load(open(tmp_path / "score_integrity_validation.json"))
    assert diag["problems"] == [] and diag["origins"]["DDM3|h24"] == 10


# R-12
def test_a_sampler_failure_stops_the_run_and_nothing_is_saved(tmp_path):
    with pytest.raises(fr.ScoreIntegrityError, match="failed or non-finite"):
        fr.write_scores(str(tmp_path), "validation", results_of(month(0)), {"2020-10-01": [("B3", 3)]})
    assert no_scores_written(tmp_path)
    assert json.load(open(tmp_path / "failures_validation.json")) == {"2020-10-01": [["B3", 3]]}
    diag = json.load(open(tmp_path / "score_integrity_validation.json"))
    assert diag["problems"] and diag["failures"]


# R-12
def test_a_model_with_a_different_origin_set_stops_the_run(tmp_path):
    mo = month(0)
    sc = mo[("DDM2", 4)]
    mo[("DDM2", 4)] = {k: v[1:] for k, v in sc.items()}                       # one origin silently missing for one model/horizon
    with pytest.raises(fr.ScoreIntegrityError, match=r"origin set of DDM2 differs .* h=4 \(1 origins differ\)"):
        fr.write_scores(str(tmp_path), "validation", results_of(mo), {"2020-10-01": []})
    assert no_scores_written(tmp_path)


# R-12
def test_a_month_missing_for_one_model_changes_the_origin_set(tmp_path):
    a, b = month(0), month(100)
    res = results_of(a, b)
    res[("B6", 12)] = [a[("B6", 12)]]                                            # B6 has no results for the second month
    with pytest.raises(fr.ScoreIntegrityError, match="origin set of B6 differs"):
        fr.write_scores(str(tmp_path), "validation", res, {})
    assert no_scores_written(tmp_path)


# R-12
def test_a_missing_model_or_horizon_stops_the_run(tmp_path):
    res = results_of(month(0))
    del res[("B5", 1)]
    with pytest.raises(fr.ScoreIntegrityError, match="missing results for B5 h=1"):
        fr.write_scores(str(tmp_path), "validation", res, {})
    assert no_scores_written(tmp_path)


# R-12
def test_non_finite_scores_and_duplicate_origins_stop_the_run(tmp_path):
    mo = month(0)
    mo[("B1", 1)]["crps"][2] = np.nan
    with pytest.raises(fr.ScoreIntegrityError, match="non-finite crps for B1 h=1"):
        fr.write_scores(str(tmp_path), "validation", results_of(mo), {})
    mo2 = month(0)
    mo2[("B2", 4)]["j"][1] = mo2[("B2", 4)]["j"][0]
    with pytest.raises(fr.ScoreIntegrityError, match="duplicated origins for B2 h=4"):
        fr.write_scores(str(tmp_path), "validation", results_of(mo2), {})
    assert no_scores_written(tmp_path)


class _Serial:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, it):
        return map(fn, it)


# R-12
def test_stage_score_stops_on_a_failed_month_and_writes_no_scores(tmp_path, monkeypatch):
    months = fr.fm.month_starts("validation")

    def job(args):
        fails = [("B4", 7)] if args[2] == months[2] else []
        return str(args[2].date()), month(100 * months.index(args[2])), fails

    monkeypatch.setattr(fr, "ProcessPoolExecutor", _Serial)
    monkeypatch.setattr(fr, "month_job", job)
    monkeypatch.setattr(fr, "require_leak_pass", lambda wd: {"pass": True})
    with pytest.raises(fr.ScoreIntegrityError):
        fr.stage_score(str(tmp_path), "validation")
    assert no_scores_written(tmp_path)


# R-12
def test_score_month_reports_a_non_converged_garch_state_as_a_failure():
    """The production path: a GARCH state that did not converge yields no sampler output, which score_month records as a
    failure (so stage_score's validation stops the run) instead of silently dropping the origin."""
    import pandas as pd
    import f1_models as fm
    from test_f1_wiring_and_guard import make_sv_series
    old = dict(fc.PERIODS)
    fc.PERIODS["estimation"] = ("2018-01-01", "2018-03-31")
    fc.PERIODS["validation"] = ("2018-04-11", "2018-04-12")
    try:
        close, idx = make_sv_series(60 * 24 * 120, seed=4)
        s = fc.Series(close, idx)
        edges = np.quantile(s.atr_window()[s.period_mask("estimation") & s.atr_valid()], np.linspace(0, 1, fc.N_QUANTILES + 1)[1:-1])
        ctx = fm.Context(close, idx, edges)
        ens = [np.random.default_rng(0).standard_normal((fc.N_QUANTILES, 4, 1000)) * 1e-3 for _ in range(3)]
        real_build = fm.build_state

        def broken(ctx_, ms, prev):
            st = real_build(ctx_, ms, prev)
            st.arch["B3"] = None
            st.arch_fail.append("B3")
            return st

        fm_build = fm.build_state
        fm.build_state = broken
        try:
            _, out, fails = fr.score_month(ctx, ens, "validation", pd.Timestamp("2018-04-01"))
        finally:
            fm.build_state = fm_build
        assert "B3" in fails and any(isinstance(f, tuple) and f[0] == "B3" for f in fails)
        problems, _ = fr.validate_scores({k: [v] for k, v in out.items()}, {"2018-04-01": fails})
        assert problems
    finally:
        fc.PERIODS.clear()
        fc.PERIODS.update(old)
