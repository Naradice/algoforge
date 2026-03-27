"""
Characteristic analysis registry.

Each analysis function takes a pandas DataFrame (OHLC) and returns a dict
of computed metrics and optional plot data.

Register new analyses:
    from .characteristics import CHARACTERISTIC_REGISTRY

    def compute_my_metric(df: pd.DataFrame) -> dict:
        return {"my_metric": value}

    CHARACTERISTIC_REGISTRY["my_metric"] = compute_my_metric

Registered analyses appear automatically in GET /api/v1/data/analyses
and the dataset detail UI.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

AnalysisFn = Callable[[pd.DataFrame], dict]

CHARACTERISTIC_REGISTRY: dict[str, AnalysisFn] = {}


def register(name: str):
    """Decorator to register an analysis function."""
    def decorator(fn: AnalysisFn) -> AnalysisFn:
        CHARACTERISTIC_REGISTRY[name] = fn
        return fn
    return decorator


# ── Built-in analyses (implemented in Phase 1) ────────────────────────────────

@register("basic_stats")
def compute_basic_stats(df: pd.DataFrame) -> dict:
    """Row count, date range, basic OHLC stats."""
    # TODO Phase 1: implement
    raise NotImplementedError


@register("acf")
def compute_acf(df: pd.DataFrame) -> dict:
    """Autocorrelation function of returns."""
    # TODO Phase 1: adapt from stocknet/scripts/compute_validation.py
    raise NotImplementedError


@register("volatility_clustering")
def compute_volatility_clustering(df: pd.DataFrame) -> dict:
    """ACF of squared returns — tests for ARCH effects."""
    # TODO Phase 1: adapt from stocknet
    raise NotImplementedError


@register("hurst")
def compute_hurst(df: pd.DataFrame) -> dict:
    """Hurst exponent via R/S analysis."""
    # TODO Phase 1: adapt from stocknet
    raise NotImplementedError


@register("fat_tails")
def compute_fat_tails(df: pd.DataFrame) -> dict:
    """Kurtosis, CCDF tail index."""
    # TODO Phase 1: adapt from stocknet
    raise NotImplementedError


@register("stationarity")
def compute_stationarity(df: pd.DataFrame) -> dict:
    """Augmented Dickey-Fuller test for stationarity."""
    # TODO Phase 1: implement
    raise NotImplementedError
