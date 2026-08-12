"""
Classical statistical baselines — AR / MA / ARMA, fit via statsmodels.

Unlike every other architecture in this package, these are not torch.nn.Module: fitting is a
single MLE optimization (no epochs), so they're dispatched by celery_worker.py through a
separate, non-torch code path rather than build_model()/get_trainer_fns(). See
celery_worker.py's _run_arima_training for the orchestration.
"""

from __future__ import annotations

import numpy as np

from .dataset import OHLCWindowDataset

ARIMA_ARCHITECTURES = ("ar", "ma", "arma")

# statsmodels' ARIMA .fit() cost grows with series length, and walk-forward .append(refit=False)
# re-runs the Kalman filter over the whole (growing) history each call — without a cap, a
# multi-tens-of-thousands-row dataset turns "fit in one shot" into a multi-minute fit. Benchmarked
# on this machine: 5000-point fit ~2s, 200 walk-forward blocks ~5s. Both caps keep the most
# recent points (same "most relevant" convention as OHLCWindowDataset._MAX_OHLC_ROWS).
MAX_ARIMA_TRAIN_POINTS = 5000
MAX_WALK_FORWARD_BLOCKS = 200


def order_from_config(architecture: str, config: dict) -> tuple[int, int, int]:
    """Map an architecture + its config dict to a statsmodels ARIMA (p, d, q) order.

    Merges with ARCHITECTURE_DEFAULTS (model/architectures/__init__.py) the same way
    build_model() does for every other architecture, so defaults live in one place.

    d defaults to 0 for all three — classical AR/MA assume an already-stationary input.
    Pick a preprocessed-dataset recipe with normalize="returns" (or raise d) if your primary
    feature column is a raw, non-stationary price series.
    """
    from model.architectures import ARCHITECTURE_DEFAULTS

    merged = {**ARCHITECTURE_DEFAULTS.get(architecture, {}), **config}
    d = int(merged.get("d", 0))
    if architecture == "ar":
        return (int(merged.get("p", 2)), d, 0)
    if architecture == "ma":
        return (0, d, int(merged.get("q", 2)))
    if architecture == "arma":
        return (int(merged.get("p", 2)), d, int(merged.get("q", 2)))
    raise ValueError(f"Not an ARIMA-family architecture: {architecture!r}")


def load_series_for_arima(
    artifact_path: str,
    feature_cols: list[str] | None,
    preprocessing: dict | None,
    normalize: str,
    val_split: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load + preprocess + normalize the primary feature column, split by time into
    (train, val) flat 1-D arrays.

    Mirrors OHLCWindowDataset.__init__'s load/normalize steps (model/trainers/dataset.py:43-64)
    but returns the flat pre-window series — ARIMA fits on the whole training slice at once
    rather than sliding windows, so there's nothing to window here. The normalize branch below
    is intentionally a copy of that logic (on a 1-D array instead of 2-D) rather than a shared
    helper, to avoid touching the well-tested neural-net loading path for this.

    No max_rows parameter here (unlike OHLCWindowDataset) -- every ARIMA-family run is subject
    to _load_preprocessed_df's default 50,000-row cap with no way to override it, on top of
    MAX_ARIMA_TRAIN_POINTS below. The returned provenance dict records both cuts so this is
    visible rather than silently assumed.
    """
    df, feature_cols, provenance = OHLCWindowDataset._load_preprocessed_df(artifact_path, feature_cols, preprocessing)
    data = df[feature_cols[0]].values.astype(np.float64)

    if normalize == "returns":
        data = np.log(data + 1e-8)
        data = np.diff(data)
    elif normalize == "minmax":
        mn, mx = data.min(), data.max()
        rng = mx - mn if mx != mn else 1.0
        data = (data - mn) / rng
    elif normalize == "zscore":
        mu, sigma = data.mean(), data.std()
        sigma = sigma if sigma != 0 else 1.0
        data = (data - mu) / sigma
    elif normalize == "robust":
        median = np.median(data)
        q25, q75 = np.percentile(data, [25, 75])
        iqr = (q75 - q25) if q75 != q25 else 1.0
        data = (data - median) / iqr

    n = len(data)
    split_idx = int(n * (1 - val_split))
    train, val = data[:split_idx], data[split_idx:]
    provenance["arima_train_points_cap"] = MAX_ARIMA_TRAIN_POINTS
    provenance["arima_train_points_before_cap"] = len(train)
    if len(train) > MAX_ARIMA_TRAIN_POINTS:
        train = train[-MAX_ARIMA_TRAIN_POINTS:]
    provenance["arima_train_points_used"] = len(train)
    return train, val, provenance


def fit_and_evaluate_arima(train: np.ndarray, val: np.ndarray, order: tuple[int, int, int], pred_len: int) -> dict:
    """Fit ARIMA(order) on `train`, then walk-forward evaluate over `val` in non-overlapping
    pred_len-sized blocks — forecast pred_len steps, compare to the actual block, extend state
    with the true values via .append(refit=False) (cheap — no re-optimization), repeat.

    Returns {"results": <fit on train only, for artifact persistence>, "train_mse": float,
    "n_params": int, "metrics": {mae, mse, rmse, directional_accuracy, sharpe_proxy}}.
    """
    from statsmodels.tsa.arima.model import ARIMA

    if len(val) < pred_len:
        raise ValueError(f"Validation split ({len(val)} points) is shorter than pred_len ({pred_len})")

    # Cap the number of walk-forward blocks, not just raw val length: keeping only the most
    # recent max_val_points would otherwise leave `state` stuck at the end of `train`, forecasting
    # into a temporal gap instead of into the truncated val window. Catch up with a single bulk
    # append of the skipped prefix (one Kalman pass) instead of paying that cost per block.
    max_val_points = MAX_WALK_FORWARD_BLOCKS * pred_len
    skipped_prefix = val[:-max_val_points] if len(val) > max_val_points else None
    if skipped_prefix is not None:
        val = val[-max_val_points:]

    model = ARIMA(train, order=order)
    results = model.fit()

    state = results
    if skipped_prefix is not None and len(skipped_prefix) > 0:
        state = state.append(skipped_prefix, refit=False)

    preds, actuals = [], []
    i = 0
    while i + pred_len <= len(val):
        block = val[i : i + pred_len]
        preds.append(state.forecast(steps=pred_len))
        actuals.append(block)
        state = state.append(block, refit=False)
        i += pred_len

    p = np.concatenate(preds)
    a = np.concatenate(actuals)

    mae = float(np.mean(np.abs(p - a)))
    mse = float(np.mean((p - a) ** 2))
    rmse = float(np.sqrt(mse))
    directional_accuracy = float(np.mean(np.sign(p) == np.sign(a)))
    sharpe_proxy = float(p.mean() / (p.std() + 1e-8))

    return {
        "results": results,
        "train_mse": float(results.mse),
        "n_params": int(len(results.params)),
        "metrics": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "directional_accuracy": directional_accuracy,
            "sharpe_proxy": sharpe_proxy,
        },
    }
