from unittest.mock import patch, MagicMock

import pandas as pd

from data.collectors import economic_calendar as ec


def test_fred_csv_keyless_alias_and_range():
    csv = "observation_date,IR3TIB01JPM156N\n2020-01-01,0.1\n2020-02-01,.\n2021-01-01,0.2\n"
    resp = MagicMock(text=csv); resp.raise_for_status = lambda: None
    with patch("requests.get", return_value=resp) as g:
        df = ec._fetch_fred_csv("RATE_JPY", pd.Timestamp("2020-06-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"))
    assert g.call_args.kwargs["params"]["id"] == "IR3TIB01JPM156N"
    assert list(df.value) == [0.2] and df.indicator.iloc[0] == "RATE_JPY"
