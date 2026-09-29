"""JSONB writes: NaN / Inf anywhere in a nested value become null (Postgres rejects them)."""
import math


def test_json_safe_deep_replaces_non_finite_floats_recursively():
    import celery_worker
    out = celery_worker._json_safe_deep({"hurst": float("nan"), "x": [1.0, float("inf"), {"y": -math.inf, "z": "ok"}]})
    assert out == {"hurst": None, "x": [1.0, None, {"y": None, "z": "ok"}]}
