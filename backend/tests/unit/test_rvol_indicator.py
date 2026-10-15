import numpy as np
import pandas as pd

from strategy.engine.indicators import apply_indicators


def test_rvol_matches_manual_and_percentile_in_range():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    df = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))}, index=idx)
    out = apply_indicators(df, [{"id": "rv", "type": "rvol", "params": {"window": 20, "long_window": 100}}])
    manual = np.log(df.close).diff().rolling(20, min_periods=10).std() * np.sqrt(252)
    assert np.allclose(out["rv"].dropna(), manual.dropna())
    assert out["rv_pct"].dropna().between(0, 1).all() and out["rv_ratio"].notna().sum() > 200
