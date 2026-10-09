"""Pooled walk-forward ML (HistGB) on H1 FX features -> multi-hour forward vol-normalised return."""
import sys, pickle, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from core import *

def features(sym, closes):
    df = load(sym); c = df.close; lr = np.log(c).diff()
    vol = lr.rolling(480, min_periods=100).std()
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 6, 12, 24, 72, 240, 720):
        f[f"ret{n}"] = np.log(c).diff(n) / (vol * np.sqrt(n))
    for n in (24, 120, 480):
        f[f"emadist{n}"] = (np.log(c) - np.log(ema(c, n))) / (vol * np.sqrt(n))
    f["volratio"] = lr.rolling(24).std() / vol
    f["volratio2"] = lr.rolling(120).std() / vol
    f["range"] = (np.log(df.high / df.low)) / vol
    f["hr_s"] = np.sin(2 * np.pi * df.index.hour / 24); f["hr_c"] = np.cos(2 * np.pi * df.index.hour / 24)
    f["dow"] = df.index.dayofweek
    # cross-sectional context: mean vol-normalised 24/240-bar return across all pairs (market-wide factor), aligned on index
    f["mkt24"] = closes["ret24"].reindex(df.index).ffill(); f["mkt240"] = closes["ret240"].reindex(df.index).ffill()
    f["usdpair"] = 0.0
    return df, f, vol

def build(H):
    ctx = {}
    for s in PAIRS:
        c = load(s).close; lr = np.log(c).diff(); v = lr.rolling(480, min_periods=100).std()
        sign = -1 if s.startswith("USD") or s == "USDCHF" else 1   # crude: orient so that + = foreign ccy up vs USD (JPY crosses ignored)
        ctx[s] = {n: sign * np.log(c).diff(n) / (v * np.sqrt(n)) for n in (24, 240)}
    mk = {n: pd.concat([ctx[s][n] for s in PAIRS], axis=1).mean(axis=1) for n in (24, 240)}
    closes = pd.DataFrame({"ret24": mk[24], "ret240": mk[240]})
    rows = []
    for s in PAIRS:
        df, f, vol = features(s, closes)
        y = (np.log(df.close).shift(-H) - np.log(df.close)) / (vol * np.sqrt(H))
        f["y"] = y; f["sym"] = s; f["raw_fwd"] = np.log(df.close).shift(-H) - np.log(df.close)
        f["scale"] = vol_scale(df.close); f["close"] = df.close
        f["pair_id"] = PAIRS.index(s)
        rows.append(f)
    return pd.concat(rows).replace([np.inf, -np.inf], np.nan)

def run(H=24, first=2008, last=2022, sample_every=6):
    X = build(H); feats = [c for c in X.columns if c not in ("y", "sym", "raw_fwd", "scale", "close", "usdpair")]
    X = X.dropna(subset=feats + ["y"]); preds = []
    for yr in range(first, last + 1):
        tr = X[(X.index < pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(hours=H))]
        tr = tr[tr.index.hour % 1 == 0].iloc[::sample_every]
        te = X[(X.index >= f"{yr}-01-01") & (X.index <= f"{yr}-12-31")]
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.04, max_depth=4, min_samples_leaf=800, l2_regularization=5.0, random_state=0)
        m.fit(tr[feats], tr["y"].clip(-4, 4))
        p = te[["sym", "raw_fwd", "scale", "close", "y"]].copy(); p["pred"] = m.predict(te[feats])
        preds.append(p); ic = np.corrcoef(p.pred, p.y)[0, 1]
        print(yr, len(tr), "IC=%.4f" % ic, flush=True)
    return pd.concat(preds)

if __name__ == "__main__":
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    P = run(H)
    P.to_parquet(DATA / f"ml_preds_H{H}.parquet")
