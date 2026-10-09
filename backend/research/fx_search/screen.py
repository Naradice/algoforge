"""Compute daily net-return series for every (family, config, pair) over full history."""
import pickle, sys, pathlib
from concurrent.futures import ProcessPoolExecutor
import pandas as pd, numpy as np
from core import *

def work(sym):
    df = load(sym); scale = vol_scale(df.close); out = {}
    for fam, grid in FAMILIES.items():
        for i, p in enumerate(grid):
            sig = FN[fam](df, **p)
            r = net_returns(df, sig, sym, scale)
            out[(fam, i)] = r.resample("1D").sum()
            r0 = net_returns(df, sig, sym, scale).copy()
    return sym, out

if __name__ == "__main__":
    with ProcessPoolExecutor(9) as ex:
        res = dict(ex.map(work, PAIRS))
    pickle.dump(res, open(DATA / "screen_daily.pkl", "wb"))
    print("done", {k: len(v) for k, v in res.items()})
