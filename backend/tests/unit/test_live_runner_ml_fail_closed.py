import math
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

def evaluate_ml_filter(
    dir_to_enter: str | None,
    ml_enabled: bool,
    ml_threshold: float,
    candles_df: pd.DataFrame | None,
    prob_return_value: any,
    raise_exception: bool = False,
):
    """
    Extracted logic equivalent to lines 468-500 in live_runner.py
    for testing fail-closed behavior.
    """
    entry_msg = "test_entry"
    strat_tag = "TEST"
    now_utc = pd.Timestamp.now(tz="UTC")
    bid = 150.0
    ask = 150.02
    exp_low = 149.0
    exp_high = 151.0
    ml_model_path = "dummy.txt"

    if dir_to_enter and ml_enabled:
        ml_passed = False
        try:
            if raise_exception:
                raise RuntimeError("Simulated inference exception")
            
            if candles_df is not None and not candles_df.empty:
                # simulate feature extraction & predict
                prob = prob_return_value
                if isinstance(prob, (int, float)) and math.isfinite(prob) and prob >= ml_threshold:
                    ml_passed = True
                    entry_msg += f" [ML: {prob:.2f}]"
            else:
                pass
        except Exception:
            pass

        if not ml_passed:
            dir_to_enter = None

    return dir_to_enter, entry_msg


def test_ml_fail_closed_empty_dataframe():
    # Empty DataFrame
    res_dir, _ = evaluate_ml_filter(
        dir_to_enter="buy",
        ml_enabled=True,
        ml_threshold=0.50,
        candles_df=pd.DataFrame(),
        prob_return_value=0.80,
    )
    assert res_dir is None, "Should fail-closed on empty DataFrame"


def test_ml_fail_closed_none_dataframe():
    # None DataFrame
    res_dir, _ = evaluate_ml_filter(
        dir_to_enter="buy",
        ml_enabled=True,
        ml_threshold=0.50,
        candles_df=None,
        prob_return_value=0.80,
    )
    assert res_dir is None, "Should fail-closed on None DataFrame"


def test_ml_fail_closed_exception():
    # Exception during inference
    df = pd.DataFrame([{"close": 150.0}])
    res_dir, _ = evaluate_ml_filter(
        dir_to_enter="buy",
        ml_enabled=True,
        ml_threshold=0.50,
        candles_df=df,
        prob_return_value=0.80,
        raise_exception=True,
    )
    assert res_dir is None, "Should fail-closed on exception"


def test_ml_fail_closed_nan():
    # NaN score
    df = pd.DataFrame([{"close": 150.0}])
    res_dir, _ = evaluate_ml_filter(
        dir_to_enter="buy",
        ml_enabled=True,
        ml_threshold=0.50,
        candles_df=df,
        prob_return_value=float("nan"),
    )
    assert res_dir is None, "Should fail-closed on NaN"


def test_ml_fail_closed_pos_inf():
    # Positive infinity
    df = pd.DataFrame([{"close": 150.0}])
    res_dir, _ = evaluate_ml_filter(
        dir_to_enter="buy",
        ml_enabled=True,
        ml_threshold=0.50,
        candles_df=df,
        prob_return_value=float("inf"),
    )
    assert res_dir is None, "Should fail-closed on +inf"


def test_ml_fail_closed_neg_inf():
    # Negative infinity
    df = pd.DataFrame([{"close": 150.0}])
    res_dir, _ = evaluate_ml_filter(
        dir_to_enter="buy",
        ml_enabled=True,
        ml_threshold=0.50,
        candles_df=df,
        prob_return_value=float("-inf"),
    )
    assert res_dir is None, "Should fail-closed on -inf"


def test_ml_fail_closed_below_threshold():
    # Below threshold (0.499 < 0.50)
    df = pd.DataFrame([{"close": 150.0}])
    res_dir, _ = evaluate_ml_filter(
        dir_to_enter="buy",
        ml_enabled=True,
        ml_threshold=0.50,
        candles_df=df,
        prob_return_value=0.499,
    )
    assert res_dir is None, "Should reject when score < threshold"


def test_ml_accept_exact_threshold():
    # Exactly on threshold (0.50 == 0.50)
    df = pd.DataFrame([{"close": 150.0}])
    res_dir, msg = evaluate_ml_filter(
        dir_to_enter="buy",
        ml_enabled=True,
        ml_threshold=0.50,
        candles_df=df,
        prob_return_value=0.50,
    )
    assert res_dir == "buy", "Should accept when score == threshold"
    assert "[ML: 0.50]" in msg


def test_ml_accept_above_threshold():
    # Above threshold (0.65 > 0.50)
    df = pd.DataFrame([{"close": 150.0}])
    res_dir, msg = evaluate_ml_filter(
        dir_to_enter="sell",
        ml_enabled=True,
        ml_threshold=0.50,
        candles_df=df,
        prob_return_value=0.65,
    )
    assert res_dir == "sell", "Should accept when score > threshold"
    assert "[ML: 0.65]" in msg


if __name__ == "__main__":
    test_ml_fail_closed_empty_dataframe()
    test_ml_fail_closed_none_dataframe()
    test_ml_fail_closed_exception()
    test_ml_fail_closed_nan()
    test_ml_fail_closed_pos_inf()
    test_ml_fail_closed_neg_inf()
    test_ml_fail_closed_below_threshold()
    test_ml_accept_exact_threshold()
    test_ml_accept_above_threshold()
    print("ALL 9 FAIL-CLOSED TESTS PASSED!")

