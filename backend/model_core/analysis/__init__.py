"""Experiment-validity analysis shared by research scripts and the MCP layer.

One tested definition of the pieces every "is this target worth training on" check needs --
gap-aware return windows, future-only targets, the strong volatility-memory baseline's features,
blocked day splits, paired day-block bootstrap -- and `assess_target`, which runs
docs/model-layer.md point 6 (overlap / trivial predictors / nonlinear headroom) as code. Generic
transforms (scaling, gap detection) come from finance_client.fprocess; this package holds only
what is specific to ML experiments.
"""
from model_core.analysis.assess import assess_target
from model_core.analysis.bootstrap import paired_block_bootstrap_ci
from model_core.analysis.features import linear_extras, time_features, vol_memory_features
from model_core.analysis.splits import blocked_split, chronological_split
from model_core.analysis.targets import TARGETS, make_target
from model_core.analysis.windows import ReturnWindows, align_closes, build_return_windows

__all__ = [
    "assess_target", "paired_block_bootstrap_ci", "linear_extras", "time_features",
    "vol_memory_features", "blocked_split", "TARGETS", "make_target", "ReturnWindows",
    "build_return_windows", "align_closes", "chronological_split",
]
