import pytest
import warnings
import pandas as pd
import numpy as np
from strategy.engine.backtest import _normalise, run_backtest
from strategy.engine.analyst_strategy import run_analyst_backtest


def test_normalise_analyst_range_reversion():
    definition = {
        "strategy_type": "analyst_range_reversion",
        "symbol": "USDJPY",
        "parameters": {
            "source": "FX168",
            "tolerance_pips": 10.0,
            "sl_pips": 15.0,
            "tp_pips": 40.0,
            "ml_filter": {
                "enabled": True,
                "threshold": 0.50,
                "model_path": "dummy.txt"
            }
        }
    }

    norm = _normalise(definition)

    # Check indicators
    assert any(i["type"] == "analyst_forecast" for i in norm["indicators"])
    # Check long conditions
    long_conds = norm["long"]["entry"]["conditions"]
    assert any(c.get("left") == "low" and c.get("right") == "buy_boundary" for c in long_conds)
    assert any(c.get("type") == "ml_rebound_filter" and c.get("direction") == "buy" for c in long_conds)

    # Check short conditions
    short_conds = norm["short"]["entry"]["conditions"]
    assert any(c.get("left") == "high" and c.get("right") == "sell_boundary" for c in short_conds)
    assert any(c.get("type") == "ml_rebound_filter" and c.get("direction") == "sell" for c in short_conds)

    # Check risk
    assert norm["risk"]["sl_pips"] == 15.0
    assert norm["risk"]["tp_pips"] == 40.0
    assert norm["risk"]["pip_size"] == 0.01


def test_normalise_analyst_breakout_momentum():
    definition = {
        "strategy_type": "analyst_breakout_momentum",
        "symbol": "USDJPY",
        "parameters": {
            "source": "SMBC",
            "sl_pips": 50.0,
            "tp_pips": 80.0,
        }
    }

    norm = _normalise(definition)
    long_conds = norm["long"]["entry"]["conditions"]
    assert any(c.get("left") == "high" and c.get("right") == "expected_high" for c in long_conds)
    short_conds = norm["short"]["entry"]["conditions"]
    assert any(c.get("left") == "low" and c.get("right") == "expected_low" for c in short_conds)


def test_deprecated_run_analyst_backtest_forwarding(tmp_path):
    # Create synthetic OHLC
    dates = pd.date_range("2024-01-01 00:00", periods=50, freq="30min", tz="UTC")
    df = pd.DataFrame({
        "open": np.full(50, 150.0),
        "high": np.full(50, 150.5),
        "low": np.full(50, 149.5),
        "close": np.full(50, 150.0),
        "volume": np.zeros(50),
    }, index=dates)

    definition = {
        "strategy_type": "analyst_range_reversion",
        "symbol": "USDJPY",
        "parameters": {
            "source": "FX168",
            "tolerance_pips": 10.0,
            "sl_pips": 15.0,
            "tp_pips": 40.0,
        }
    }

    with pytest.deprecated_call():
        trades, metrics, curve = run_analyst_backtest(definition, df)

    assert isinstance(trades, list)
    assert isinstance(metrics, dict)
    assert isinstance(curve, list)


def test_unknown_analyst_type_rejected():
    with pytest.raises(ValueError, match="Unsupported analyst strategy type"):
        _normalise({"strategy_type": "analyst_typo"})


@pytest.mark.parametrize("opposite_tp", [True, False])
def test_analyst_session_rules(monkeypatch, tmp_path, opposite_tp):
    from strategy.engine import forecast_levels

    bars = pd.date_range("2024-01-01", periods=49, freq="30min", tz="UTC")
    frame = pd.DataFrame(
        {"open": 150.0, "high": 150.05, "low": 149.99, "close": 150.0},
        index=bars,
    )
    # The second bar is actionable. The first one predates FX168 publication.
    # A take-profit on the entry bar must not enable a second daily buy.
    frame.loc[bars[2], "high"] = 151.01
    frame.loc[bars[2], "low"] = 150.5
    frame.loc[bars[2], "open"] = 150.8
    frame.loc[bars[2], "close"] = 150.8
    levels = pd.DataFrame([{
        "date": bars[0].date(), "source": "FX168",
        "expected_low": 150.0, "expected_high": 151.0,
    }])
    monkeypatch.setattr(forecast_levels, "get_forecast_records", lambda **kwargs: levels)
    artifact = tmp_path / "ohlc.parquet"
    frame.to_parquet(artifact)
    definition = {
        "strategy_type": "analyst_range_reversion",
        "symbol": "USDJPY",
        "parameters": {
            "source": "FX168", "exit_at_close": True,
            "tp_pips": "opposite" if opposite_tp else None,
        },
    }

    trades, _, _ = run_backtest(definition, str(artifact))

    buys = [trade for trade in trades if trade["direction"] == "buy"]
    assert len(buys) == 1
    assert buys[0]["opened_at"] == bars[2]
    assert buys[0]["closed_at"] == bars[2 if opposite_tp else 47]
