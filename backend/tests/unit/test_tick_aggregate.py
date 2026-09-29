"""tick_aggregate collector: Dukascopy ticks -> 1-minute microstructure bars with OFI."""
import numpy as np
import pandas as pd
import pytest

from data.collectors import tick_aggregate as ta

CSV = """Gmt time,Ask,Bid,AskVolume,BidVolume
02.01.2020 00:00:05.000,100.02,100.00,1.0,2.0
02.01.2020 00:00:30.000,100.03,100.01,1.5,1.0
02.01.2020 00:01:10.000,100.03,100.00,2.0,3.0
"""


def test_aggregate_bars_and_ofi(tmp_path):
    p = tmp_path / "USDJPY_Ticks_02.01.2020-02.01.2020.csv"
    p.write_text(CSV)
    t = ta.read_dukascopy_ticks(p)
    bars = ta.aggregate_ticks(t)
    assert list(bars.index) == [pd.Timestamp("2020-01-02 00:00"), pd.Timestamp("2020-01-02 00:01")]
    b0 = bars.iloc[0]
    assert b0["open"] == pytest.approx(100.01) and b0["close"] == pytest.approx(100.02)
    assert b0["tick_count"] == 2 and b0["spread"] == pytest.approx(0.02)
    assert b0["imbalance"] == pytest.approx(((2 - 1) / 3 + (1 - 1.5) / 2.5) / 2)
    # tick 2: bid up (+q^b_n = 1.0), ask up (+q^a_{n-1} = 1.0) -> e = 2.0; first tick e = 0
    assert b0["ofi"] == pytest.approx(2.0)
    # tick 3: bid down (-q^b_{n-1} = -1.0), ask same (-q^a_n + q^a_{n-1} = -2.0 + 1.5) -> -1.5
    assert bars.iloc[1]["ofi"] == pytest.approx(-1.5)


def test_collect_streams_files_in_date_order(tmp_path, monkeypatch):
    monkeypatch.setattr(ta, "ARTIFACT_STORE", tmp_path / "store")
    src = tmp_path / "tick"
    src.mkdir()
    (src / "USDJPY_Ticks_03.01.2020-03.01.2020.csv").write_text(CSV.replace("02.01.2020", "03.01.2020"))
    (src / "USDJPY_Ticks_02.01.2020-02.01.2020.csv").write_text(CSV)
    (src / "USDJPY_Ticks_01.01.2019-01.01.2019.csv").write_text(CSV.replace("02.01.2020", "01.01.2019"))
    r = ta.collect(7, {"path": str(src), "symbol": "USDJPY", "from_ts": "2020-01-01"})
    df = pd.read_parquet(tmp_path / "store" / r.artifact_path)
    assert r.row_count == 4 and df.index.is_monotonic_increasing and df.index[0].year == 2020
    # the second file's first tick continues from the first file's last tick (not reset to 0)
    assert df.loc["2020-01-03 00:00", "ofi"] != pytest.approx(2.0)
    with pytest.raises(ValueError):
        ta.collect(7, {"path": str(src), "format": "oanda"})
