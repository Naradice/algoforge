"""Load aligned closes of the HistData M1 pairs (datasets 149-157) for Phase 10B."""
from __future__ import annotations

import asyncio

import pandas as pd

from probe_representations import _artifact_store, _dataset_artifact_path

HISTDATA = {149: "USDJPY", 150: "EURUSD", 151: "GBPUSD", 152: "AUDUSD", 153: "USDCHF",
            154: "EURJPY", 155: "GBPJPY", 156: "AUDJPY", 157: "CHFJPY"}


async def _paths(ids):
    return [await _dataset_artifact_path(i) for i in ids]


def load_closes(since: str = "2017-08-14", ids=HISTDATA) -> dict[str, pd.Series]:
    paths = asyncio.run(_paths(list(ids)))          # one event loop: the pooled engine is loop-bound
    out = {}
    for (i, name), p in zip(ids.items(), paths):
        s = pd.read_parquet(_artifact_store() / p, columns=["close"])["close"].sort_index()
        out[name] = s[s.index >= since]
    return out
