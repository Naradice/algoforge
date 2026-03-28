"""
ML model signal condition — synchronous, for use inside the backtest executor thread.

Condition spec:
    {
        "type": "ml_signal",
        "model_id": 1,
        "direction": "buy",   # "buy" | "sell" — required direction in prediction
        "step": 1,            # which prediction step to check (1-indexed, default 1)
        "min_confidence": 0.0 # optional: only fire if |close_val| >= this threshold
    }

ModelInfo (pre-loaded by executor):
    {
        "architecture": "lstm",
        "config": {...},
        "artifact_path": "models/1/training_2/best.pt",
        "hyperparams": {"obs_len": 60, "pred_len": 10, "feature_cols": ["close"], ...}
    }

The condition fires (returns True) when:
    - The deployed model predicts the required direction for the given step
    - |close_return_pred| >= min_confidence
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_ml_condition(
    df_upto: pd.DataFrame,
    cond: dict,
    model_cache: dict[int, Any],
) -> bool:
    """
    df_upto: DataFrame slice up to and including the current bar (with indicators applied).
    cond: ml_signal condition spec.
    model_cache: dict of model_id → ModelInfo dict (pre-loaded by executor).
    """
    model_id: int = cond["model_id"]
    required_direction: str = cond.get("direction", "buy")
    step: int = int(cond.get("step", 1))
    min_confidence: float = float(cond.get("min_confidence", 0.0))

    model_info = model_cache.get(model_id)
    if model_info is None:
        logger.warning(f"ml_signal: model {model_id} not in cache — skipping")
        return False

    from model.inference import predict

    hp = model_info["hyperparams"]
    feature_cols = hp.get("feature_cols", ["close"])

    # Build feature matrix from the columns available in df_upto
    available = [c for c in feature_cols if c in df_upto.columns]
    if not available:
        logger.warning(f"ml_signal: none of {feature_cols} found in DataFrame")
        return False

    features = df_upto[available].values.tolist()
    obs_len = hp.get("obs_len", 60)
    if len(features) < obs_len:
        return False  # Not enough history yet

    try:
        preds = predict(
            model_id=model_id,
            architecture=model_info["architecture"],
            model_config=model_info["config"],
            artifact_path=model_info["artifact_path"],
            hyperparams=hp,
            features=features,
            feature_names=available,
        )
    except Exception as e:
        logger.warning(f"ml_signal: inference failed for model {model_id}: {e}")
        return False

    if step < 1 or step > len(preds):
        return False

    pred = preds[step - 1]
    direction_val: int = pred.get("direction", 0)
    close_val: float = float(pred.get("close", pred.get(available[0], 0.0)))

    if abs(close_val) < min_confidence:
        return False

    if required_direction == "buy":
        return direction_val == 1
    elif required_direction == "sell":
        return direction_val == -1
    return False
