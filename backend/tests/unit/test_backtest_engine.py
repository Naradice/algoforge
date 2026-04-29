from __future__ import annotations

import pandas as pd

from strategy.engine import backtest


def test_run_backtest_precomputes_indicators_once(tmp_path, monkeypatch):
    """apply_indicators is called once upfront, not once per bar."""
    artifact_root = tmp_path
    dataset_path = artifact_root / "history.parquet"
    index = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
        },
        index=index,
    )
    df.to_parquet(dataset_path)
    monkeypatch.setenv("ARTIFACT_STORE_PATH", str(artifact_root))

    call_count: list[int] = []

    def fake_apply_indicators(source: pd.DataFrame, indicator_specs: list[dict]) -> pd.DataFrame:
        call_count.append(len(source))
        result = source.copy()
        result["mysig"] = 1.0  # constant signal column
        return result

    monkeypatch.setattr(backtest, "apply_indicators", fake_apply_indicators)

    definition = {
        "symbol": "TEST",
        "indicators": [{"id": "mysig", "type": "fake"}],
        "long": {
            "entry": {
                "conditions": [{"left": "mysig", "op": ">", "right": 0}],
                "logic": "and",
            },
            "exit": {},
        },
        "risk": {
            "sl_pct": 0.0,
            "tp_pct": 0.0,
            "slippage_pct": 0.0,
            "commission_pct": 0.0,
        },
    }

    trades, metrics, equity_curve = backtest.run_backtest(definition, "history.parquet")

    # Indicators pre-computed exactly once on the full dataset (not once per bar)
    assert call_count == [4]
    assert metrics["total_trades"] == 1
    assert len(equity_curve) == 4


def test_run_backtest_skips_indicator_eval_until_warmup_is_met(tmp_path, monkeypatch):
    artifact_root = tmp_path
    dataset_path = artifact_root / "short_history.parquet"
    index = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        },
        index=index,
    )
    df.to_parquet(dataset_path)
    monkeypatch.setenv("ARTIFACT_STORE_PATH", str(artifact_root))

    definition = {
        "symbol": "TEST",
        "indicators": [
            {"id": "macd", "type": "macd", "params": {"fast": 12, "slow": 26, "signal_period": 9}},
        ],
        "long": {
            "entry": {
                "conditions": [{"left": "macd_line", "op": ">", "right": "macd_signal"}],
                "logic": "and",
            },
            "exit": {},
        },
        "risk": {
            "sl_pct": 0.0,
            "tp_pct": 0.0,
            "slippage_pct": 0.0,
            "commission_pct": 0.0,
        },
    }

    trades, metrics, equity_curve = backtest.run_backtest(definition, "short_history.parquet")

    assert trades == []
    assert metrics["total_trades"] == 0
    assert len(equity_curve) == 5
