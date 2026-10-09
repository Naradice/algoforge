"""Resample the 9 HistData M1 datasets (ids 149-157) to H1 and cache as parquet."""
import asyncio, asyncpg, pandas as pd, pathlib, sys
OUT = pathlib.Path(__file__).resolve().parents[3] / "artifacts" / "research" / "fx"
STORE = pathlib.Path(__file__).resolve().parents[3] / "artifacts"

async def paths():
    c = await asyncpg.connect('postgresql://algoforge:algoforge@127.0.0.1:5432/algoforge')
    rows = await c.fetch("select symbol, artifact_path from datasets where id between 149 and 157")
    return {r['symbol']: r['artifact_path'] for r in rows}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sym, p in asyncio.run(paths()).items():
        df = pd.read_parquet(STORE / p)
        print(sym, df.columns.tolist(), len(df), flush=True)
        df.index = pd.to_datetime(df.index); df = df.sort_index()
        df = df[~df.index.duplicated()]
        h = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        h.to_parquet(OUT / f"{sym}_H1.parquet")
        print("  H1", len(h), h.index[0], h.index[-1])
main()
