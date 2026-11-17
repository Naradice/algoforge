"""F1 (#889) R-19: the statistics that `full_characterize` computes for a real window are the same statistics, in the same units
and the same order, that the DDM atlas stores and `select_ensemble` compares. Representative fixed inputs, no new experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

import ddm_ensemble_selection as des  # noqa: E402
import ddm_regime_atlas as dra  # noqa: E402

ATLAS = BACKEND / "ddm_regime_atlas.csv"
DATA = Path(__file__).resolve().parents[3] / "artifacts" / "datasets" / "derived" / "usdjpy_1min_volatility.parquet"


def independent_stats(close):
    """Plain numpy definitions: log returns, population moments, Pearson kurtosis, unadjusted autocorrelation."""
    r = np.diff(np.log(np.asarray(close, float)))
    m = r.mean()
    d = r - m
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()

    def acf(x, k):
        x = x - x.mean()
        return float(np.dot(x[:-k], x[k:]) / np.dot(x, x))

    return {"std": float(np.sqrt(m2)), "skewness": float(m3 / m2 ** 1.5), "kurtosis": float(m4 / m2 ** 2),
            "acf_1": acf(r, 1), "acf_5": acf(r, 5), "acf_20": acf(r, 20), "volclust_50": acf(np.abs(r), 50)}


def fixed_close(n=1001, seed=3):
    rng = np.random.default_rng(seed)
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = 0.98 * lv[i - 1] + 0.1 * rng.standard_normal()
    return 110.0 * np.exp(np.cumsum(1e-4 * np.exp(lv) * rng.standard_normal(n)))


# R-19
def test_full_characterize_equals_independent_definitions_on_a_fixed_series():
    close = fixed_close()
    got = dra.full_characterize(pd.DataFrame({"close": close}))
    want = independent_stats(close)
    assert list(got) == ["std", "skewness", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50"]     # order of the keys
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-8, abs=1e-12), k


# R-19
@pytest.mark.skipif(not DATA.exists(), reason="USDJPY derived artifact not available")
def test_full_characterize_equals_independent_definitions_on_a_real_window():
    close = pd.read_parquet(DATA, columns=["close"])["close"].values[300_000:301_001]
    got = dra.full_characterize(pd.DataFrame({"close": close}))
    want = independent_stats(close)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-8, abs=1e-12), k
    assert 1e-5 < got["std"] < 1e-3                           # log-return units of one M1 candle, not price units


# R-19
def test_atlas_columns_and_selection_statistics():
    atlas = pd.read_csv(ATLAS)
    for name in dra.FULL_STAT_NAMES:
        assert name in atlas.columns and f"{name}_std_across_seeds" in atlas.columns
    assert set(des.SELECTION_STATS) <= set(atlas.columns) and set(des.SELECTION_WEIGHTS) == set(des.SELECTION_STATS)
    assert atlas["std"].dropna().between(1e-6, 0.5).all()      # same log-return units as the real windows


# R-19
def test_an_atlas_row_is_reproduced_by_the_current_characterisation():
    atlas = pd.read_csv(ATLAS)
    row = atlas[(atlas["delta"] == 0.0) & (atlas["wma"] == 15)].iloc[0]
    assert row["success_rate"] == 1.0
    stats = []
    for s in range(dra.N_SEEDS):
        df = dra.simulate_ohlc(0.0, 15, n_trades=dra.N_TRADES, seed=dra.BASE_SEED + s)
        assert len(df) == 1000                                 # 20 000 trades / 20 per candle, like a 1 001-close real window
        stats.append(dra.full_characterize(df))
    for name in dra.FULL_STAT_NAMES:
        assert float(np.nanmean([x[name] for x in stats])) == pytest.approx(row[name], rel=1e-9), name


# R-19
def test_select_ensemble_maps_the_statistics_to_the_right_atlas_columns():
    atlas = pd.read_csv(ATLAS)
    row = atlas[(atlas["delta"] == 0.0) & (atlas["wma"] == 15)].iloc[0]
    real = pd.Series({n: row[n] for n in dra.FULL_STAT_NAMES})
    sel = des.select_ensemble(real, atlas)
    best = sel["candidates"].iloc[0]
    assert (best["delta"], best["wma"]) == (0.0, 15) and sel["best_std_ratio"] == pytest.approx(1.0)
    assert not sel["out_of_range"]
    # a real window with twice the volatility must move to a candidate with a larger atlas std, not a different statistic
    real2 = real.copy()
    real2["std"] = row["std"] * 2.0
    sel2 = des.select_ensemble(real2, atlas)
    assert sel2["candidates"].iloc[0]["std"] > best["std"]
    # a real window above the atlas range is flagged out of range
    real3 = real.copy()
    real3["std"] = atlas["std"].max() * 10
    assert des.select_ensemble(real3, atlas)["out_of_range"]
