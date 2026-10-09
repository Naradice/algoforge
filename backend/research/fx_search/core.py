"""Vectorised FX strategy screening harness (H1 bars, net of spread, walk-forward selection)."""
import numpy as np, pandas as pd, pathlib, itertools
DATA = pathlib.Path(__file__).resolve().parents[3] / "artifacts" / "research" / "fx"
PAIRS = ["USDJPY","EURUSD","GBPUSD","AUDUSD","USDCHF","EURJPY","GBPJPY","AUDJPY","CHFJPY"]
# conservative (retail, ECN-ish incl. slippage) round-trip-unit cost in pips per unit of position change
SPREAD = {"USDJPY":0.6,"EURUSD":0.7,"GBPUSD":1.0,"AUDUSD":1.0,"USDCHF":1.3,"EURJPY":1.2,"GBPJPY":2.0,"AUDJPY":1.6,"CHFJPY":2.5}
PIP = lambda s: 0.01 if s.endswith("JPY") else 0.0001
TARGET_BAR_VOL = 0.10 / np.sqrt(24 * 252)   # 10% annualised target vol per instrument
MAXLEV = 5.0

def load(sym):
    d = pd.read_parquet(DATA / f"{sym}_H1.parquet")
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    med = d.close.rolling(25, center=True, min_periods=5).median()
    bad = np.log(d.close / med).abs() > 0.03
    bad |= (np.log(d.high / d.close).abs() > 0.03) | (np.log(d.low / d.close).abs() > 0.03)
    return d[~bad]

def ema(x, n): return x.ewm(span=n, adjust=False).mean()

def vol_scale(close):
    r = np.log(close).diff()
    v = r.rolling(24 * 20, min_periods=100).std().shift(1)
    return (TARGET_BAR_VOL / v).clip(upper=MAXLEV)

# ---------- signal families: return raw direction series in {-1,0,1} (decided at bar close) ----------
def sig_donchian(df, n):
    hi = df.high.rolling(n).max().shift(1); lo = df.low.rolling(n).min().shift(1)
    s = pd.Series(np.nan, df.index); s[df.close > hi] = 1; s[df.close < lo] = -1
    return s.ffill().fillna(0)
def sig_macross(df, f, s_):
    return np.sign(ema(df.close, f) - ema(df.close, s_))
def sig_tsmom(df, L):
    return np.sign(np.log(df.close).diff(L))
def sig_rsi(df, n, th):
    d = df.close.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    s = pd.Series(np.nan, df.index); s[rsi < th] = 1; s[rsi > 100 - th] = -1
    s[(rsi - 50).abs() < 5] = 0   # exit near neutral
    return s.ffill().fillna(0)
def sig_bbrev(df, n, k):
    m = df.close.rolling(n).mean(); sd = df.close.rolling(n).std(); z = (df.close - m) / sd
    s = pd.Series(np.nan, df.index); s[z < -k] = 1; s[z > k] = -1; s[z.abs() < 0.2] = 0
    return s.ffill().fillna(0)
def sig_seasonal(df, days):
    """Hour-of-day drift: next bar's direction = sign of trailing mean return at that hour (past `days` days only)."""
    r = np.log(df.close).diff()
    key = df.index.hour
    m = r.groupby(key).transform(lambda x: x.rolling(days, min_periods=days // 2).mean())  # mean incl. current bar of same hour
    # signal for bar t+1 uses hist mean of hour(t+1) from bars strictly before t+1  -> use m shifted within hour group
    m_prev = m.groupby(key).shift(1)           # at bar b: mean over same-hour bars strictly before b
    return np.sign(m_prev.shift(-1)).fillna(0)  # known at close of t for bar t+1

FAMILIES = {
    "donchian": [dict(n=n) for n in (24, 48, 96, 192, 384, 768)],
    "macross":  [dict(f=f, s_=s) for f, s in itertools.product((12, 24, 48, 96, 240, 480, 1200), (96, 192, 384, 768, 1200, 2400, 4800)) if f * 2 <= s],
    "tsmom":    [dict(L=L) for L in (24, 72, 120, 240, 480, 960, 1920, 3840, 5760)],
    "seasonal": [dict(days=d) for d in (120, 250, 500)],
    "rsi":      [dict(n=n, th=th) for n in (6, 14, 28) for th in (15, 25, 35)],
    "bbrev":    [dict(n=n, k=k) for n in (24, 48, 120) for k in (1.5, 2.0, 2.5)],
}
FN = dict(seasonal=sig_seasonal, donchian=sig_donchian, macross=sig_macross, tsmom=sig_tsmom, rsi=sig_rsi, bbrev=sig_bbrev)

def net_returns(df, sig, sym, scale=None):
    """Position decided at bar t close is held over bar t+1. Returns per-bar net log-ret contribution."""
    scale = vol_scale(df.close) if scale is None else scale
    pos = (sig * scale).shift(1).fillna(0)
    ret = np.log(df.close).diff().fillna(0)
    cost = pos.diff().abs().fillna(0) * SPREAD[sym] * PIP(sym) / df.close
    return pos * ret - cost

def sharpe(r, ppy=24 * 252):
    r = np.asarray(r); sd = r.std()
    return 0.0 if sd == 0 else r.mean() / sd * np.sqrt(ppy)
