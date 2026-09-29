"""Upload CSV parsing: datetime column detection and day-first (Dukascopy) dates."""
import pandas as pd

from data.service import _parse_csv_bytes, _parse_datetimes


def test_dukascopy_export_parses_day_first():
    csv = (b"Gmt time,Open,High,Low,Close,Volume\n"
           b"03.01.2016 22:00:00.000,120.188,120.204,120.183,120.204,12.75\n"
           b"13.02.2016 22:02:00.000,120.206,120.209,120.206,120.209,3\n")
    df = _parse_csv_bytes(csv, None, "USDJPY.csv")
    assert list(df.index) == [pd.Timestamp("2016-01-03 22:00"), pd.Timestamp("2016-02-13 22:02")]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"] and df["volume"].iloc[0] == 12.75


def test_gmt_offset_suffix_gives_utc():
    idx = _parse_datetimes(["13.02.2016 22:02:00.000 GMT+0100", "13.02.2016 22:03:00.000 GMT-0230"])
    assert list(idx) == [pd.Timestamp("2016-02-13 21:02"), pd.Timestamp("2016-02-14 00:33")]


def test_explicit_format_and_iso_default():
    assert _parse_datetimes(["2016/13/02"], "%Y/%d/%m")[0] == pd.Timestamp("2016-02-13")
    assert _parse_datetimes(["2016-02-13 22:02:00"])[0] == pd.Timestamp("2016-02-13 22:02")
