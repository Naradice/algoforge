"""F1 (#889) R-05 (Stage C finding 1): the GARCH-family forecast starts from the conditional variance that reflects information
through the forecast origin, not one block earlier.

The expectation is an independent fixed-parameter variance recursion written here in numpy (arch's documented recursions and
back-cast definition), compared with arch 8.0.0 `forecast(start=len(y)-1)` and with the production sampler `fm.sample_arch`.
In the production check the innovations are replaced by a constant +1, so the simulated path is deterministic and equals the
own recursion exactly."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

RESEARCH = Path(__file__).resolve().parents[2] / "research"
sys.path.insert(0, str(RESEARCH))

import f1_models as fm  # noqa: E402

SQ2PI = np.sqrt(2.0 / np.pi)

# name -> (arch spec, param names in arch's order, fixed parameter values)
CASES = {
    "B3": (fm.ARCH_SPECS["B3"], ["omega", "alpha[1]", "beta[1]"], [0.05, 0.08, 0.90]),
    "B4": (fm.ARCH_SPECS["B4"], ["omega", "alpha[1]", "beta[1]", "nu"], [0.05, 0.08, 0.90, 6.0]),
    "B5": (fm.ARCH_SPECS["B5"], ["omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"], [0.04, 0.03, 0.09, 0.88, 7.0]),
    "B6": (fm.ARCH_SPECS["B6"], ["omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"], [0.01, 0.12, -0.05, 0.95, 7.0]),
}


def series(n=260, seed=11):
    g = np.random.default_rng(seed)
    y = np.zeros(n)
    s = 1.0
    for i in range(n):                                    # volatility clustering, asymmetric, deterministic
        y[i] = s * g.standard_normal()
        s = np.sqrt(0.1 + 0.15 * y[i] ** 2 + (0.1 * y[i] ** 2 if y[i] < 0 else 0.0) + 0.7 * s * s)
    return y


def backcast(y):
    tau = min(75, len(y))
    w = 0.94 ** np.arange(tau)
    return float(np.sum(y[:tau] ** 2 * (w / w.sum())))


def own_variances(kind, y, par):
    """sigma2[0..n] where sigma2[t] is the variance of y[t] given y[:t] and sigma2[n] is the one-step-ahead variance."""
    n = len(y)
    bc = backcast(y)
    p = dict(par)
    s2 = np.empty(n + 1)
    if kind == "EGARCH":
        ln = np.empty(n + 1)
        ln[0] = p["omega"] + p["beta[1]"] * np.log(bc)           # no information before t = 0: |e| term equals its constant
        s2[0] = np.exp(ln[0])
        for t in range(1, n + 1):
            e = y[t - 1] / np.sqrt(s2[t - 1])
            ln[t] = p["omega"] + p["alpha[1]"] * (abs(e) - SQ2PI) + p["gamma[1]"] * e + p["beta[1]"] * ln[t - 1]
            s2[t] = np.exp(ln[t])
        return s2
    a, g_, b = p["alpha[1]"], p.get("gamma[1]", 0.0), p["beta[1]"]
    s2[0] = p["omega"] + (a + 0.5 * g_ + b) * bc
    for t in range(1, n + 1):
        s2[t] = p["omega"] + (a + g_ * (y[t - 1] < 0)) * y[t - 1] ** 2 + b * s2[t - 1]
    return s2


def own_path_with_unit_innovations(kind, y, par, steps):
    """Continue the recursion past the origin with every standardised innovation equal to +1; returns the per-step returns."""
    s2 = list(own_variances(kind, y, par))
    out, last_y = [], None
    p = dict(par)
    for _ in range(steps):
        sig = np.sqrt(s2[-1])
        yy = sig * 1.0
        out.append(yy)
        if kind == "EGARCH":
            e = 1.0
            nxt = np.exp(p["omega"] + p["alpha[1]"] * (abs(e) - SQ2PI) + p["gamma[1]"] * e + p["beta[1]"] * np.log(s2[-1]))
        else:
            nxt = p["omega"] + (p["alpha[1]"] + p.get("gamma[1]", 0.0) * (yy < 0)) * yy ** 2 + p["beta[1]"] * s2[-1]
        s2.append(nxt)
    return np.array(out)


def fixed_result(spec, names, vals, y):
    return arch_model(y, mean="Zero", rescale=False, **spec).fix(pd.Series(vals, index=names))


@pytest.mark.parametrize("name", list(CASES))
# R-05
def test_arch_conditional_variance_equals_the_independent_recursion(name):
    spec, names, vals = CASES[name]
    y = series()
    res = fixed_result(spec, names, vals, y)
    want = own_variances(spec["vol"], y, dict(zip(names, vals)))
    np.testing.assert_allclose(res.conditional_volatility ** 2, want[:-1], rtol=1e-9)


@pytest.mark.parametrize("name", list(CASES))
# R-05
def test_arch_forecast_start_is_the_variance_through_the_origin(name):
    """forecast(start=len(y)-1) step 1 must be sigma2[n], the variance that already contains y[n-1] (the origin block)."""
    spec, names, vals = CASES[name]
    y = series()
    res = fixed_result(spec, names, vals, y)
    want = own_variances(spec["vol"], y, dict(zip(names, vals)))
    if spec["vol"] == "EGARCH":
        pytest.skip("EGARCH has no analytic multi-step forecast; its first step is checked through the simulation below")
    f = res.forecast(horizon=1, start=len(y) - 1, method="analytic", reindex=False)
    got = float(f.variance.values[0, 0])
    assert got == pytest.approx(want[-1], rel=1e-9)
    assert abs(got - want[-2]) > 1e-6 * want[-2]         # and it is not the one-block-earlier variance


@pytest.mark.parametrize("name", list(CASES))
# R-05
def test_forecast_uses_only_data_up_to_the_origin(name):
    """Appending later observations must not change a forecast made from the earlier origin (truncation == long series)."""
    spec, names, vals = CASES[name]
    y = series()
    k = 200
    a = fixed_result(spec, names, vals, y[:k]).forecast(horizon=3, start=k - 1, method="simulation", simulations=50,
                                                         rng=lambda size: np.ones(size), reindex=False)
    b = fixed_result(spec, names, vals, y).forecast(horizon=3, start=k - 1, method="simulation", simulations=50,
                                                     rng=lambda size: np.ones(size), reindex=False)
    np.testing.assert_allclose(a.simulations.values[0], b.simulations.values[0], rtol=1e-9)


@pytest.mark.parametrize("name", list(CASES))
# R-05
def test_production_sampler_starts_from_the_variance_through_the_origin(name, monkeypatch):
    spec, names, vals = CASES[name]
    y_scaled = series()
    n_all, k = len(y_scaled), 200
    ctx = SimpleNamespace(valid_r=np.arange(n_all) + 7, y_all=y_scaled, hour=np.zeros(n_all + 7, int))   # rows are offset from valid positions
    st = SimpleNamespace(arch={name: pd.Series(vals, index=names)}, hour_factor=np.ones(24))
    origin_row = int(ctx.valid_r[k - 1])                  # the block ending at the origin is the k-th valid return
    monkeypatch.setattr(fm, "M", 3)
    monkeypatch.setattr(fm, "std_t", lambda g, nu, size: np.ones(size))
    monkeypatch.setattr(fm, "rng_for", lambda m, j: SimpleNamespace(standard_normal=lambda size: np.ones(size)))
    out = fm.sample_arch(name)(st, ctx, origin_row)
    steps = own_path_with_unit_innovations(spec["vol"], y_scaled[:k], dict(zip(names, vals)), 24)
    want = np.cumsum(steps) / fm.SCALE
    got = out[:, 0]
    np.testing.assert_allclose(got, [want[h - 1] for h in (1, 4, 12, 24)], rtol=1e-8)
    # one block earlier would be a different first step
    early = np.sqrt(own_variances(spec["vol"], y_scaled[:k], dict(zip(names, vals)))[-2]) / fm.SCALE
    assert abs(got[0] - early) > 1e-9
    # a later observation than the origin must not matter
    ctx2 = SimpleNamespace(valid_r=ctx.valid_r, y_all=np.concatenate([y_scaled[:k], 50 * np.ones(n_all - k)]), hour=ctx.hour)
    out2 = fm.sample_arch(name)(st, ctx2, origin_row)
    np.testing.assert_allclose(out2, out, rtol=1e-12)
