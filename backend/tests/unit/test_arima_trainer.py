"""Unit tests for model/trainers/arima_trainer.py."""
import numpy as np
import pytest

from model.trainers.arima_trainer import order_from_config, fit_and_evaluate_arima


class TestOrderFromConfig:
    def test_ar_defaults(self):
        assert order_from_config("ar", {}) == (2, 0, 0)

    def test_ma_defaults(self):
        assert order_from_config("ma", {}) == (0, 0, 2)

    def test_arma_defaults(self):
        assert order_from_config("arma", {}) == (2, 0, 2)

    def test_config_overrides_defaults(self):
        assert order_from_config("ar", {"p": 5, "d": 1}) == (5, 1, 0)
        assert order_from_config("ma", {"q": 3}) == (0, 0, 3)
        assert order_from_config("arma", {"p": 1, "q": 1, "d": 2}) == (1, 2, 1)

    def test_rejects_non_arima_architecture(self):
        with pytest.raises(ValueError):
            order_from_config("lstm", {})


class TestFitAndEvaluateArima:
    @pytest.fixture
    def synthetic_ar1(self):
        # Deterministic AR(1) series: x_t = 0.6 x_{t-1} + eps_t
        rng = np.random.default_rng(42)
        n = 400
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.6 * x[t - 1] + rng.normal(0, 1)
        split = 320
        return x[:split], x[split:]

    def test_returns_expected_shape(self, synthetic_ar1):
        train, val = synthetic_ar1
        result = fit_and_evaluate_arima(train, val, order=(1, 0, 0), pred_len=5)

        assert set(result.keys()) == {"results", "train_mse", "n_params", "metrics"}
        assert set(result["metrics"].keys()) == {"mae", "mse", "rmse", "directional_accuracy", "sharpe_proxy"}
        assert result["metrics"]["mse"] >= 0
        assert result["metrics"]["rmse"] == pytest.approx(result["metrics"]["mse"] ** 0.5)
        assert result["train_mse"] >= 0

    @pytest.mark.parametrize("order,expected_n_params", [
        ((2, 0, 0), 4),   # const + ar.L1 + ar.L2 + sigma2
        ((0, 0, 2), 4),   # const + ma.L1 + ma.L2 + sigma2
        ((2, 0, 2), 6),   # const + 2 ar + 2 ma + sigma2
        ((2, 1, 0), 3),   # no const when d>0: 2 ar + sigma2
    ])
    def test_n_params_matches_order(self, synthetic_ar1, order, expected_n_params):
        train, val = synthetic_ar1
        result = fit_and_evaluate_arima(train, val, order=order, pred_len=5)
        assert result["n_params"] == expected_n_params

    def test_raises_when_val_shorter_than_pred_len(self, synthetic_ar1):
        train, val = synthetic_ar1
        with pytest.raises(ValueError, match="pred_len"):
            fit_and_evaluate_arima(train, val[:3], order=(1, 0, 0), pred_len=5)

    def test_large_series_stays_fast_and_correct(self):
        # Regression test: walk-forward .append(refit=False) re-runs the Kalman filter over the
        # whole growing history each call, so without capping train length and the number of
        # walk-forward blocks, a large dataset turns "fit in one shot" into a multi-minute fit
        # (discovered via a live run against a 100k-row dataset). This must stay fast.
        import time

        rng = np.random.default_rng(7)
        n = 20_000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + rng.normal(0, 1)
        train, val = x[:16000], x[16000:]

        t0 = time.monotonic()
        result = fit_and_evaluate_arima(train, val, order=(1, 0, 0), pred_len=5)
        elapsed = time.monotonic() - t0

        assert elapsed < 30, f"took {elapsed:.1f}s — walk-forward cap regression?"
        assert result["metrics"]["mse"] >= 0
        assert np.isfinite(result["metrics"]["mse"])
