"""Unit tests for model_core/trainers/preprocessing.py's kurtosis/autocorr indicators, added for
the representation-probing analysis (probe targets: future return / volatility / distribution
statistics / local temporal structure)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_core.trainers.preprocessing import add_indicators


class TestKurtosisIndicator:
    def test_column_name_and_alignment(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, 1, 200))})
        out = add_indicators(df.copy(), [{"type": "kurtosis", "period": 20}])
        assert "kurtosis_20" in out.columns
        assert len(out) == len(df)
        # First period-1 rows of the underlying returns can't fill a full window -> NaN
        assert out["kurtosis_20"].iloc[:20].isna().all()
        assert out["kurtosis_20"].iloc[25:].notna().all()

    def test_matches_pandas_rolling_kurt_directly(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, 1, 100))})
        out = add_indicators(df.copy(), [{"type": "kurtosis", "period": 10}])
        expected = df["close"].pct_change().rolling(10).kurt()
        pd.testing.assert_series_equal(out["kurtosis_10"], expected, check_names=False)


class TestAutocorrIndicator:
    def test_column_name_and_alignment(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, 1, 200))})
        out = add_indicators(df.copy(), [{"type": "autocorr", "period": 20, "lag": 1}])
        assert "autocorr_20_lag1" in out.columns
        assert len(out) == len(df)

    def test_perfect_alternating_series_has_autocorr_near_negative_one(self):
        # A perfectly alternating return series (+r, -r, +r, -r, ...) has lag-1 autocorrelation
        # of exactly -1 -- a clean, known-answer sanity check for the rolling cov/var formula.
        n = 60
        prices = [100.0]
        for i in range(n):
            prices.append(prices[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
        df = pd.DataFrame({"close": prices})
        out = add_indicators(df.copy(), [{"type": "autocorr", "period": 20, "lag": 1}])
        assert out["autocorr_20_lag1"].iloc[-1] == pytest.approx(-1.0, abs=1e-6)
