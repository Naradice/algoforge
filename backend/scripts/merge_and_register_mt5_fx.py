"""
merge_and_register_mt5_fx.py

1. Reads existing FX data from L:\\data\\fx\\ (specifically OANDA-Japan MT5 Live and root).
2. Connects to MetaTrader 5 (MT5) to fetch missing latest data up to current date (Nov 2026).
3. Merges existing data and latest MT5 data (concat -> sort -> drop_duplicates) matching finance_client logic.
4. Updates CSV files on L:\\data\\fx\\ (with .bak backup).
5. Exports Parquet artifacts to AlgoForge's artifacts/datasets/ directory.
6. Registers or updates Dataset records in AlgoForge database.
"""

import os
import sys
import shutil
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure backend root is on sys.path
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

import MetaTrader5 as mt5

from database import async_session_factory
from sqlalchemy import select
from data.models import Dataset

FX_DATA_DIR = Path(r"L:\data\fx")
MT5_LIVE_DIR = FX_DATA_DIR / "OANDA-Japan MT5 Live"
ARTIFACT_STORE = Path(__file__).resolve().parent.parent.parent / "artifacts" / "datasets"

TIMEFRAME_MAP = {
    "d1": (mt5.TIMEFRAME_D1, "D1"),
    "h1": (mt5.TIMEFRAME_H1, "H1"),
    "h4": (mt5.TIMEFRAME_H4, "H4"),
    "min30": (mt5.TIMEFRAME_M30, "M30"),
    "min5": (mt5.TIMEFRAME_M5, "M5"),
    "min1": (mt5.TIMEFRAME_M1, "M1"),
}


def fetch_mt5_rates(symbol: str, timeframe, from_dt: datetime = None, count: int = None):
    """Fetch rates from MT5 safely."""
    if from_dt is not None:
        to_dt = datetime.now(timezone.utc)
        rates = mt5.copy_rates_range(symbol, timeframe, from_dt, to_dt)
        if rates is not None and len(rates) > 0:
            return rates

    # Fallback to chunked copy_rates_from_pos
    if count is None:
        count = 100000
    chunk_size = 20000
    start = 0
    all_chunks = []
    while start < count:
        current_chunk = min(chunk_size, count - start)
        r = mt5.copy_rates_from_pos(symbol, timeframe, start, current_chunk)
        if r is None or len(r) == 0:
            break
        all_chunks.append(r)
        start += len(r)
        if len(r) < current_chunk:
            break

    if all_chunks:
        return np.concatenate(all_chunks)
    return None


def merge_and_update_csv(csv_path: Path, symbol: str, mt5_tf, has_mt5: bool) -> pd.DataFrame:
    """Reads existing CSV, fetches MT5 if available, merges, updates CSV, and returns merged DF."""
    print(f"\n--- Processing: {csv_path.name} (Symbol: {symbol}) ---")
    df_existing = pd.read_csv(csv_path)

    # Standardize time column name
    time_col = None
    for col in ["time", "datetime", "Date", "Time", "Timestamp"]:
        if col in df_existing.columns:
            time_col = col
            break

    if time_col is None:
        time_col = df_existing.columns[0]
        print(f"Assuming column 0 ({time_col}) is time.")

    df_existing[time_col] = pd.to_datetime(df_existing[time_col], utc=True)
    df_existing = df_existing.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")

    last_time = df_existing[time_col].iloc[-1]
    first_time = df_existing[time_col].iloc[0]
    print(f"Existing data: {len(df_existing)} rows from {first_time} to {last_time}")

    if has_mt5:
        try:
            # Query from slightly before last_time to ensure continuity
            rates = fetch_mt5_rates(symbol, mt5_tf, from_dt=last_time)
            if rates is not None and len(rates) > 0:
                df_new = pd.DataFrame(rates)
                df_new["time"] = pd.to_datetime(df_new["time"], unit="s", utc=True)
                print(f"Fetched {len(df_new)} new bars from MT5 (latest: {df_new['time'].iloc[-1]})")

                # Rename to match existing column naming if needed
                if time_col != "time":
                    df_new.rename(columns={"time": time_col}, inplace=True)

                # Merge
                merged = pd.concat([df_existing, df_new], ignore_index=True)
                merged = merged.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")

                new_count = len(merged)
                added_rows = new_count - len(df_existing)
                print(f"Merged total: {new_count} rows (+{added_rows} new rows, up to {merged[time_col].iloc[-1]})")

                # Backup original CSV if not already backed up
                bak_path = csv_path.with_suffix(".csv.bak")
                if not bak_path.exists():
                    shutil.copy2(csv_path, bak_path)

                # Save updated CSV
                # Keep original column format if possible
                merged_to_save = merged.copy()
                # Format time string if original was string
                merged_to_save.to_csv(csv_path, index=False)
                print(f"Updated CSV saved: {csv_path}")
                df_existing = merged
            else:
                print(f"No newer MT5 data found for {symbol}.")
        except Exception as e:
            print(f"Warning: MT5 fetch/merge failed for {symbol}: {e}")

    return df_existing, time_col


def df_to_algoforge_format(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Standardizes DF to AlgoForge format: DatetimeIndex (UTC), lowercase open/high/low/close/volume."""
    df_clean = df.copy()
    df_clean.set_index(time_col, inplace=True)
    df_clean.index.name = "datetime"
    df_clean.index = pd.to_datetime(df_clean.index, utc=True)

    # Lowercase column mapping
    col_map = {}
    for c in df_clean.columns:
        cl = str(c).lower().strip()
        if cl in ["open", "high", "low", "close", "spread"]:
            col_map[c] = cl
        elif cl in ["tick_volume", "volume", "vol"]:
            col_map[c] = "volume"

    df_clean.rename(columns=col_map, inplace=True)

    # Remove duplicated columns (e.g. if both tick_volume and volume were present)
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

    required = ["open", "high", "low", "close"]
    for r in required:
        if r not in df_clean.columns:
            raise ValueError(f"Missing required column {r} in dataframe")

    if "volume" not in df_clean.columns:
        df_clean["volume"] = 0

    target_cols = [c for c in ["open", "high", "low", "close", "volume", "spread"] if c in df_clean.columns]
    df_clean = df_clean[target_cols]

    # Ensure numeric
    for col in target_cols:
        series = df_clean[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        df_clean[col] = pd.to_numeric(series, errors="coerce")

    df_clean = df_clean.dropna(subset=["open", "high", "low", "close"])
    return df_clean


async def register_dataset_in_db(name: str, symbol: str, timeframe: str, df: pd.DataFrame, parquet_filename: str):
    """Saves parquet artifact and registers or updates record in AlgoForge DB."""
    ARTIFACT_STORE.mkdir(parents=True, exist_ok=True)
    parquet_path = ARTIFACT_STORE / parquet_filename
    rel_artifact_path = f"datasets/{parquet_filename}"

    # Save parquet
    df.to_parquet(parquet_path)
    print(f"Parquet saved: {parquet_path} ({len(df)} rows)")

    from_ts = df.index[0].to_pydatetime()
    to_ts = df.index[-1].to_pydatetime()

    async with async_session_factory() as session:
        # Check if dataset with same name or same artifact_path exists
        stmt = select(Dataset).where(
            (Dataset.name == name) | (Dataset.artifact_path == rel_artifact_path)
        )
        res = await session.execute(stmt)
        existing_ds = res.scalar_one_or_none()

        if existing_ds:
            print(f"Updating existing Dataset ID {existing_ds.id} ({existing_ds.name})...")
            existing_ds.name = name
            existing_ds.symbol = symbol
            existing_ds.timeframe = timeframe
            existing_ds.row_count = len(df)
            existing_ds.from_ts = from_ts
            existing_ds.to_ts = to_ts
            existing_ds.artifact_path = rel_artifact_path
            existing_ds.status = "ready"
            await session.commit()
            print(f"Dataset ID {existing_ds.id} updated successfully.")
        else:
            print(f"Creating new Dataset record: {name}...")
            new_ds = Dataset(
                name=name,
                symbol=symbol,
                timeframe=timeframe,
                row_count=len(df),
                from_ts=from_ts,
                to_ts=to_ts,
                artifact_path=rel_artifact_path,
                status="ready",
            )
            session.add(new_ds)
            await session.commit()
            await session.refresh(new_ds)
            print(f"Dataset ID {new_ds.id} created successfully.")


async def main():
    mt5_initialized = mt5.initialize()
    print(f"MT5 initialized: {mt5_initialized}")
    if mt5_initialized:
        term_info = mt5.terminal_info()
        print(f"Connected to terminal: {term_info.name if term_info else 'Unknown'}")

    # Priority 1: USDJPY D1 & H1
    usdjpy_tasks = [
        ("mt5_USDJPY_d1.csv", "USDJPY", mt5.TIMEFRAME_D1, "D1", "USDJPY D1 (1971-2026 Latest Merged)", "USDJPY_D1_latest.parquet"),
        ("mt5_USDJPY_h1.csv", "USDJPY", mt5.TIMEFRAME_H1, "H1", "USDJPY H1 (2006-2026 Latest Merged)", "USDJPY_H1_latest.parquet"),
    ]

    for csv_name, symbol, mt5_tf, tf_str, ds_name, parquet_name in usdjpy_tasks:
        csv_file = MT5_LIVE_DIR / csv_name
        if csv_file.exists():
            df_merged, time_col = merge_and_update_csv(csv_file, symbol, mt5_tf, mt5_initialized)
            df_af = df_to_algoforge_format(df_merged, time_col)
            await register_dataset_in_db(ds_name, symbol, tf_str, df_af, parquet_name)

    # Priority 2: Other Major Currency Pairs and Indices from OANDA-Japan MT5 Live
    other_symbols = [
        ("mt5_EURJPY_d1.csv", "EURJPY", mt5.TIMEFRAME_D1, "D1", "EURJPY D1 Latest Merged", "EURJPY_D1_latest.parquet"),
        ("mt5_GBPJPY_d1.csv", "GBPJPY", mt5.TIMEFRAME_D1, "D1", "GBPJPY D1 Latest Merged", "GBPJPY_D1_latest.parquet"),
        ("mt5_EURUSD_d1.csv", "EURUSD", mt5.TIMEFRAME_D1, "D1", "EURUSD D1 Latest Merged", "EURUSD_D1_latest.parquet"),
        ("mt5_GBPUSD_d1.csv", "GBPUSD", mt5.TIMEFRAME_D1, "D1", "GBPUSD D1 Latest Merged", "GBPUSD_D1_latest.parquet"),
        ("mt5_AUDJPY_d1.csv", "AUDJPY", mt5.TIMEFRAME_D1, "D1", "AUDJPY D1 Latest Merged", "AUDJPY_D1_latest.parquet"),
        ("mt5_CHFJPY_d1.csv", "CHFJPY", mt5.TIMEFRAME_D1, "D1", "CHFJPY D1 Latest Merged", "CHFJPY_D1_latest.parquet"),
        ("mt5_NZDJPY_d1.csv", "NZDJPY", mt5.TIMEFRAME_D1, "D1", "NZDJPY D1 Latest Merged", "NZDJPY_D1_latest.parquet"),
        ("mt5_CADJPY_d1.csv", "CADJPY", mt5.TIMEFRAME_D1, "D1", "CADJPY D1 Latest Merged", "CADJPY_D1_latest.parquet"),
        ("mt5_JP225_d1.csv", "JP225", mt5.TIMEFRAME_D1, "D1", "JP225 D1 Latest Merged", "JP225_D1_latest.parquet"),
        ("mt5_US500_d1.csv", "US500", mt5.TIMEFRAME_D1, "D1", "US500 D1 Latest Merged", "US500_D1_latest.parquet"),
        ("mt5_XAUUSD_d1.csv", "XAUUSD", mt5.TIMEFRAME_D1, "D1", "XAUUSD D1 Latest Merged", "XAUUSD_D1_latest.parquet"),
    ]

    for csv_name, symbol, mt5_tf, tf_str, ds_name, parquet_name in other_symbols:
        csv_file = MT5_LIVE_DIR / csv_name
        if csv_file.exists():
            try:
                df_merged, time_col = merge_and_update_csv(csv_file, symbol, mt5_tf, mt5_initialized)
                df_af = df_to_algoforge_format(df_merged, time_col)
                await register_dataset_in_db(ds_name, symbol, tf_str, df_af, parquet_name)
            except Exception as e:
                print(f"Error processing {csv_name}: {e}")

    # Priority 3: HistData and other root files in L:\data\fx
    root_files = [
        ("HistData_USDJPY_30min.csv", "USDJPY", "M30", "USDJPY M30 HistData (2000-2023)", "HistData_USDJPY_30min.parquet"),
        ("USDJPY_forex_D1.csv", "USDJPY", "D1", "USDJPY D1 Forex (2000-2022)", "USDJPY_forex_D1.parquet"),
        ("USDJPY_forex_H1.csv", "USDJPY", "H1", "USDJPY H1 Forex (2000-2022)", "USDJPY_forex_H1.parquet"),
    ]

    for csv_name, symbol, tf_str, ds_name, parquet_name in root_files:
        csv_file = FX_DATA_DIR / csv_name
        if csv_file.exists():
            try:
                print(f"\n--- Processing root file: {csv_name} ---")
                df = pd.read_csv(csv_file)
                time_col = None
                for c in ["time", "datetime", "Date", "Time", "Timestamp", "DateTime"]:
                    if c in df.columns:
                        time_col = c
                        break
                if time_col is None:
                    time_col = df.columns[0]
                df[time_col] = pd.to_datetime(df[time_col], utc=True)
                df = df.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
                df_af = df_to_algoforge_format(df, time_col)
                await register_dataset_in_db(ds_name, symbol, tf_str, df_af, parquet_name)
            except Exception as e:
                print(f"Error processing {csv_name}: {e}")

    if mt5_initialized:
        mt5.shutdown()
        print("\nMT5 shutdown completed.")

    print("\n=== All datasets merged and registered successfully! ===")


if __name__ == "__main__":
    asyncio.run(main())
