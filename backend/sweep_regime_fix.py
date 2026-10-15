"""Throwaway: diagnose why ddm_regime_delta_regression training collapsed to a constant
prediction (see analyze_regime_runs.py / baseline_regime_diagnostic.py). Sweeps the two prime
suspects -- learning rate too low to escape the trivial-constant minimum, and attn_window=5
being too narrow to aggregate a whole-window statistic like realized volatility -- as short
(8-epoch) smoke runs before committing to a full 3-seed batch on whichever configuration shows
non-collapsed (nonzero prediction variance, positive R^2 direction) behavior.

ml_model id=94 ("DecoderOnly4L attn_window=60", 4 layers, attn_window=60=obs_len) is an existing
checkpoint from the original width/depth sweep with attn_window >= obs_len, i.e. effectively
unrestricted/full attention over the whole window -- reused here rather than creating a new one.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from submit_regime_training import BASE_HP, submit

CONFIGS = [
    {"label": "w5_lr001", "model_id": 99, "lr": 0.001},
    {"label": "w5_lr005", "model_id": 99, "lr": 0.005},
    {"label": "full_lr001", "model_id": 94, "lr": 0.001},
    {"label": "full_lr005", "model_id": 94, "lr": 0.005},
]


async def submit_sweep() -> list[tuple[str, int, int]]:
    submitted = []
    for cfg in CONFIGS:
        hp = {**BASE_HP, "seed": 42, "epochs": 8, "lr": cfg["lr"]}
        run_id = await submit(hp, model_id=cfg["model_id"])
        submitted.append((cfg["label"], cfg["model_id"], run_id))
    return submitted


if __name__ == "__main__":
    result = asyncio.run(submit_sweep())
    for label, model_id, run_id in result:
        print(f"{label}: model_id={model_id} run_id={run_id}")
