import numpy as np
import pandas as pd
import pytest

from data.collectors import event_calendar as ec
from strategy.engine.event_features import event_features
from strategy.engine.indicators import apply_indicators


def _events():
    return ec._norm_frame(pd.DataFrame({
        "datetime": ["2024-01-05T13:30:00Z", "2024-02-02T13:30:00Z", "2024-02-02T15:00:00Z"],
        "currency": ["USD", "USD", "EUR"], "title": ["NFP", "NFP", "CPI"], "impact": ["High", "High", "Medium"],
        "forecast": ["170K", "180K", ""], "previous": ["150K", "216K", ""], "actual": ["216K", "353K", ""],
    }), "test")


def test_parse_number():
    assert ec.parse_number("175K") == 175000 and ec.parse_number("-0.3%") == -0.3
    assert np.isnan(ec.parse_number("")) and ec.parse_number("1.5B") == 1.5e9


def test_merge_keeps_actual_over_schedule_only():
    sched = ec._norm_frame(pd.DataFrame({"datetime": ["2024-03-08T13:30:00Z"], "currency": ["USD"], "title": ["NFP"],
                                         "impact": ["High"], "forecast": ["200K"]}), "ff")
    done = ec._norm_frame(pd.DataFrame({"datetime": ["2024-03-08T13:30:00Z"], "currency": ["USD"], "title": ["NFP"],
                                        "impact": ["High"], "forecast": ["200K"], "actual": ["275K"]}), "x")
    m = ec.merge_events(sched, done)
    assert len(m) == 1 and m["actual"].iloc[0] == 275000
    m2 = ec.merge_events(done, sched)           # order must not lose the actual
    assert m2["actual"].iloc[0] == 275000


def test_features_windows_and_surprise_no_lookahead():
    idx = pd.date_range("2024-02-02 12:00", periods=6, freq="1h")  # 12..17 UTC
    f = event_features(idx, _events(), ["USD"], pre_minutes=60, post_minutes=60)
    # release at 13:30 -> bars 13:00 (inside bar), 12:00 (<=60+60 min before? 90<120 yes), 14:00 (30 min after)
    assert f.loc["2024-02-02 13:00", "near"] == 1 and f.loc["2024-02-02 14:00", "near"] == 1
    assert f.loc["2024-02-02 17:00", "near"] == 0
    # before 13:30 only the January release is known; at 14:00 the February one is
    assert f.loc["2024-02-02 13:00", "mins_since_last"] > 24 * 60
    assert f.loc["2024-02-02 14:00", "mins_since_last"] == pytest.approx(30)
    assert f.loc["2024-02-02 12:00", "mins_to_next"] == pytest.approx(90)


def test_event_indicator_in_engine(tmp_path):
    p = tmp_path / "events.parquet"
    _events().to_parquet(p)
    df = pd.DataFrame({"close": 1.0}, index=pd.date_range("2024-02-02 12:00", periods=4, freq="1h"))
    out = apply_indicators(df, [{"id": "ev", "type": "event",
                                 "params": {"calendar": str(p), "currencies": ["USD"], "pre_minutes": 30}}])
    assert {"ev_near", "ev_mins_to_next", "ev_mins_since_last", "ev_surprise"} <= set(out.columns)
    assert out["ev_near"].iloc[1] == 1


def test_alfred_first_release_transform():
    obs = pd.DataFrame({
        "obs": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01", "2024-03-01"]),
        "rt": pd.to_datetime(["2024-02-02", "2024-03-08", "2024-03-08", "2024-04-05", "2024-04-05"]),
        "value": [100.0, 101.0, 110.0, 112.0, 125.0]})
    ev = ec.alfred_to_events(obs, "USD", "NFP", "diff", "PAYEMS")
    # first releases: Jan=100 (Feb 2), Feb=110 (Mar 8), Mar=125 (Apr 5) -> diffs 10, 15 (first has no diff)
    assert list(ev["actual"]) == [10.0, 15.0]
    assert list(ev["previous"].fillna(-1)) == [-1.0, 10.0]
