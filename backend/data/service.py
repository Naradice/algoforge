"""Data Management layer — business logic."""

from __future__ import annotations

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import Datasource, DatasourceCreate, DatasourceUpdate, Dataset, CollectionJobCreate
from data.repository import data_repo

_PANDAS_OFFSET = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D",
}


def _resample_ticks(tick_series: "pd.Series", timeframe: str) -> "pd.DataFrame":
    """Resample a tick price series to OHLC candles."""
    freq = _PANDAS_OFFSET.get(timeframe, "1min")
    ohlc = tick_series.resample(freq).ohlc()
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc["volume"] = tick_series.resample(freq).count()
    return ohlc.dropna()

DATASOURCE_NOT_FOUND = "DATASOURCE_NOT_FOUND"

# ---------------------------------------------------------------------------
# Upload helpers (module-level so they can be unit-tested independently)
# ---------------------------------------------------------------------------

_ALIASES: dict[str, list[str]] = {
    "close":    ["close", "adj close", "adjusted close"],
    "open":     ["open"],
    "high":     ["high"],
    "low":      ["low"],
    "volume":   ["volume", "vol"],
    "datetime": ["datetime", "date", "time", "timestamp"],
}

# Column names recognised as a single tick price (used when no OHLC close is found)
_TICK_ALIASES = ["price", "bid", "ask", "last", "mid", "tick"]


def _parse_csv_bytes(contents: bytes, col_map: dict | None, filename: str) -> "pd.DataFrame":
    """Parse raw CSV bytes into a normalised DataFrame with a datetime index.

    Supports two modes:
    - **OHLC**: requires a close column; produces open/high/low/close/volume columns.
    - **Tick**: no close column but a price/bid/ask/last column; produces a single
      ``price`` column suitable for the DDM-compatible tick preview pipeline.

    Raises HTTPException(422) if neither mode can be resolved.
    """
    import io
    df = pd.read_csv(io.BytesIO(contents))
    col_lower = {c.lower(): c for c in df.columns}

    def _resolve(field: str) -> "str | None":
        explicit = (col_map or {}).get(field)
        if explicit:
            if explicit in df.columns:
                return explicit
            if explicit.lower() in col_lower:
                return col_lower[explicit.lower()]
        for alias in _ALIASES[field]:
            if alias in col_lower:
                return col_lower[alias]
        return None

    # ── Datetime index (shared by both modes) ────────────────────────────────
    datetime_src = _resolve("datetime")
    if datetime_src:
        df = df.rename(columns={datetime_src: "datetime"})
    if "datetime" in df.columns:
        df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)
        df.index.name = "datetime"

    # ── OHLC mode ────────────────────────────────────────────────────────────
    close_src = _resolve("close")
    if close_src is not None:
        rename: dict[str, str] = {}
        for field in ("close", "open", "high", "low", "volume"):
            src = _resolve(field)
            if src and src not in rename:
                rename[src] = field
        df = df.rename(columns=rename)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]

    # ── Tick mode ────────────────────────────────────────────────────────────
    tick_src = next((col_lower[a] for a in _TICK_ALIASES if a in col_lower), None)
    if tick_src is not None:
        df = df.rename(columns={tick_src: "price"})
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df[["price"]]

    from fastapi import HTTPException
    raise HTTPException(status_code=422, detail={
        "code": "INVALID_CSV",
        "message": f"'{filename}': could not find a close or price column. "
                   f"Available columns: {list(df.columns)}",
    })


def _extract_csvs_from_zip(contents: bytes, col_map: dict | None) -> "dict[str, pd.DataFrame]":
    """Extract and parse all CSV files from a ZIP archive.

    Returns a dict of {filename: DataFrame}, skipping files that fail to parse.
    """
    import io
    import zipfile

    result: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(contents)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
        for name in csv_names:
            try:
                csv_bytes = zf.read(name)
                df = _parse_csv_bytes(csv_bytes, col_map, name)
                result[name] = df
            except Exception:
                pass  # skip unparseable entries; caller raises if result is empty
    return result


async def _save_new_dataset(db, df: "pd.DataFrame", store: "Path", name: str, datasource_id, symbol, timeframe) -> "Dataset":
    """Write df to a new parquet artifact and insert a Dataset record."""
    import uuid
    from data.models import Dataset

    artifact_name = f"datasets/upload_{uuid.uuid4().hex}.parquet"
    df.to_parquet(store / artifact_name)

    from_ts = df.index[0].to_pydatetime() if len(df) > 0 and hasattr(df.index, '__len__') else None
    to_ts = df.index[-1].to_pydatetime() if len(df) > 0 and hasattr(df.index, '__len__') else None

    ds = Dataset(
        datasource_id=datasource_id,
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        row_count=len(df),
        artifact_path=artifact_name,
        from_ts=from_ts,
        to_ts=to_ts,
        status="ready",
    )
    db.add(ds)
    await db.flush()
    await db.refresh(ds)
    return ds
DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
DATASET_NOT_READY = "DATASET_NOT_READY"
COLLECTION_JOB_NOT_FOUND = "COLLECTION_JOB_NOT_FOUND"


def _revoke_collection_task(job_id: int) -> None:
    """Clear the dedup lock for a collection job so it can be re-enqueued.

    Celery task IDs are now UUIDs (not deterministic), so we cannot revoke the
    old task by ID.  We only delete the Redis lock key so enqueue() won't raise
    AlreadyRunningError.  The old worker process will keep running until the
    collection container is restarted, but the new run will write fresh data
    after _clear_artifact_dir() removes the stale partitions.
    Best-effort: failures are silently ignored.
    """
    import os
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        r.delete(f"algoforge:enqueued:run_collection_job:{job_id}")
    except Exception:
        pass


class DataService:
    async def list_datasources(self, db: AsyncSession, offset: int = 0, limit: int = 20) -> tuple[list[Datasource], int]:
        return await data_repo.get_datasources(db, offset=offset, limit=limit)

    async def get_datasource(self, db: AsyncSession, datasource_id: int) -> Datasource:
        obj = await data_repo.get_datasource(db, datasource_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=DATASOURCE_NOT_FOUND)
        return obj

    async def create_datasource(self, db: AsyncSession, body: DatasourceCreate) -> Datasource:
        return await data_repo.create_datasource(db, name=body.name, type=body.type, config=body.config)

    async def update_datasource(self, db: AsyncSession, datasource_id: int, body: DatasourceUpdate) -> Datasource:
        await self.get_datasource(db, datasource_id)
        updates = body.model_dump(exclude_none=True)
        return await data_repo.update_datasource(db, datasource_id, **updates)

    async def delete_datasource(self, db: AsyncSession, datasource_id: int) -> None:
        await self.get_datasource(db, datasource_id)
        await data_repo.delete_datasource(db, datasource_id)

    async def list_datasets(self, db: AsyncSession, symbol: str | None = None, timeframe: str | None = None, datasource_id: int | None = None, offset: int = 0, limit: int = 20) -> tuple[list[Dataset], int]:
        return await data_repo.get_datasets(db, symbol=symbol, timeframe=timeframe, datasource_id=datasource_id, offset=offset, limit=limit)

    async def get_dataset(self, db: AsyncSession, dataset_id: int) -> Dataset:
        obj = await data_repo.get_dataset(db, dataset_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=DATASET_NOT_FOUND)
        return obj

    async def list_collection_jobs(self, db: AsyncSession, datasource_id: int | None = None, offset: int = 0, limit: int = 20) -> tuple[list, int]:
        return await data_repo.get_collection_jobs(db, datasource_id=datasource_id, offset=offset, limit=limit)

    async def get_collection_job(self, db: AsyncSession, job_id: int):
        obj = await data_repo.get_collection_job(db, job_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=COLLECTION_JOB_NOT_FOUND)
        return obj

    async def create_collection_job(self, db: AsyncSession, body: CollectionJobCreate):
        await self.get_datasource(db, body.datasource_id)
        return await data_repo.create_collection_job(db, datasource_id=body.datasource_id, schedule_cron=body.schedule_cron)

    async def get_characteristics(self, db: AsyncSession, dataset_id: int):
        await self.get_dataset(db, dataset_id)
        return await data_repo.get_characteristics(db, dataset_id)

    async def trigger_collection(self, db: AsyncSession, job_id: int):
        """Enqueue a run_collection_job Celery task and return the job.

        If the job is already running (status=running or dedup lock held), the
        existing Celery task is revoked first so the new run starts cleanly
        without a concurrent instance writing to the same artifact directory.
        """
        job = await self.get_collection_job(db, job_id)
        _revoke_collection_task(job_id)
        from celery_app import enqueue
        await enqueue("run_collection_job", job_id)
        return job

    async def trigger_analysis(self, db: AsyncSession, dataset_id: int):
        """Enqueue a compute_characteristics Celery task and return the dataset."""
        dataset = await self.get_dataset(db, dataset_id)
        if dataset.status not in ("ready", "running"):
            raise HTTPException(status_code=422, detail=DATASET_NOT_READY)
        from celery_app import enqueue
        await enqueue("compute_characteristics", dataset_id)
        return dataset

    async def update_collection_job(self, db: AsyncSession, job_id: int, body):
        job = await self.get_collection_job(db, job_id)
        updates = body.model_dump(exclude_none=True)
        if not updates:
            return job
        return await data_repo.update_collection_job(db, job_id, **updates)

    async def delete_collection_job(self, db: AsyncSession, job_id: int):
        await self.get_collection_job(db, job_id)
        await data_repo.delete_collection_job(db, job_id)

    async def list_job_runs(self, db: AsyncSession, job_id: int) -> list:
        await self.get_collection_job(db, job_id)
        return await data_repo.get_job_runs(db, job_id)

    async def delete_dataset(self, db: AsyncSession, dataset_id: int):
        import os
        from pathlib import Path
        from strategy.models import StrategyRun
        from model.models import TrainingRun
        from sqlalchemy import select

        dataset = await self.get_dataset(db, dataset_id)
        # Check strategy run references
        ref_run = (await db.execute(select(StrategyRun).where(StrategyRun.dataset_id == dataset_id).limit(1))).scalar_one_or_none()
        if ref_run:
            raise HTTPException(status_code=409, detail={"code": "DATASET_IN_USE", "message": f"Dataset is referenced by strategy run {ref_run.id}"})
        ref_tr = (await db.execute(select(TrainingRun).where(TrainingRun.dataset_id == dataset_id).limit(1))).scalar_one_or_none()
        if ref_tr:
            raise HTTPException(status_code=409, detail={"code": "DATASET_IN_USE", "message": f"Dataset is referenced by training run {ref_tr.id}"})
        await data_repo.delete_dataset(db, dataset_id)
        # Remove artifact file from disk
        if dataset.artifact_path:
            import shutil
            store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
            artifact = store / dataset.artifact_path
            if artifact.is_dir():
                shutil.rmtree(artifact)
            elif artifact.exists():
                artifact.unlink()

    async def create_dataset_from_upload(
        self,
        db: AsyncSession,
        files: list,
        datasource_id=None,
        symbol=None,
        timeframe=None,
        col_map: dict | None = None,
        append_to: int | None = None,
        merge: bool = True,
    ) -> list["Dataset"]:
        """Parse one or more uploaded CSV/ZIP files and create/update dataset records.

        Args:
            files:        List of FastAPI UploadFile objects (one or more).
            datasource_id: Associate with this datasource (optional).
            symbol:       Symbol name stored on the dataset.
            timeframe:    Timeframe string stored on the dataset.
            col_map:      Explicit column-name remapping applied to all files.
            append_to:    Merge all file contents into this existing dataset instead
                          of creating new records. Requires merge=True.
            merge:        True — merge all files into one dataset (default).
                          False — create one dataset per file.

        Returns:
            List of Dataset ORM objects (always a list).
        """
        import os
        from pathlib import Path
        from data.models import Dataset
        from data.collectors._utils import merge_into_parquet

        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
        (store / "datasets").mkdir(parents=True, exist_ok=True)

        # ── Parse every file into DataFrames ─────────────────────────────────
        # Each file yields one or more DataFrames (ZIPs expand to multiple).
        # file_dfs: list of (name, DataFrame) tuples in upload order.
        file_dfs: list[tuple[str, "pd.DataFrame"]] = []
        for file in files:
            contents = await file.read()
            filename = file.filename or "upload"
            if filename.lower().endswith(".zip"):
                dfs = _extract_csvs_from_zip(contents, col_map)
                if not dfs:
                    raise HTTPException(status_code=422, detail={
                        "code": "INVALID_ZIP",
                        "message": f"'{filename}': no readable CSV files found inside the ZIP archive.",
                    })
                for csv_name, df in dfs.items():
                    file_dfs.append((csv_name.removesuffix(".csv"), df))
            else:
                df = _parse_csv_bytes(contents, col_map, filename)
                file_dfs.append((filename.removesuffix(".csv"), df))

        if not file_dfs:
            raise HTTPException(status_code=422, detail={
                "code": "NO_DATA", "message": "No data could be parsed from the uploaded files.",
            })

        # ── append_to: merge everything into an existing dataset ──────────────
        if append_to is not None:
            target = await self.get_dataset(db, append_to)
            if target.artifact_path is None:
                raise HTTPException(status_code=422, detail={
                    "code": "DATASET_NO_ARTIFACT",
                    "message": "Target dataset has no artifact file to append to.",
                })
            combined = pd.concat([df for _, df in file_dfs])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            total_rows = merge_into_parquet(store / target.artifact_path, combined)
            merged_df = pd.read_parquet(store / target.artifact_path)
            from sqlalchemy import update
            await db.execute(
                update(Dataset).where(Dataset.id == append_to).values(
                    row_count=total_rows,
                    from_ts=merged_df.index[0].to_pydatetime(),
                    to_ts=merged_df.index[-1].to_pydatetime(),
                )
            )
            await db.commit()
            await db.refresh(target)
            return [target]

        # ── merge=True: all DataFrames → one dataset ──────────────────────────
        if merge:
            combined = pd.concat([df for _, df in file_dfs])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            name = symbol or file_dfs[0][0]
            ds = await _save_new_dataset(
                db, combined, store, name=name,
                datasource_id=datasource_id, symbol=symbol, timeframe=timeframe,
            )
            return [ds]

        # ── merge=False: one dataset per file ─────────────────────────────────
        datasets = []
        for name, df in file_dfs:
            ds = await _save_new_dataset(
                db, df, store, name=name,
                datasource_id=datasource_id, symbol=symbol, timeframe=timeframe,
            )
            datasets.append(ds)
        return datasets

    async def trigger_datasource_collection(self, db: AsyncSession, datasource_id: int, body=None):
        """Find or create a collection job for this datasource, then enqueue it."""
        from data.models import CollectionJob
        from sqlalchemy import select
        await self.get_datasource(db, datasource_id)  # raises 404 if missing

        # Look for an existing job
        result = await db.execute(select(CollectionJob).where(CollectionJob.datasource_id == datasource_id).limit(1))
        job = result.scalar_one_or_none()

        if job is None:
            job = CollectionJob(datasource_id=datasource_id)
            db.add(job)
            await db.flush()
            await db.refresh(job)

        # Clear any stale lock (tick_scheduler sets one every 60 s for overdue jobs)
        # before enqueuing, otherwise enqueue() raises AlreadyRunningError.
        _revoke_collection_task(job.id)
        from celery_app import enqueue
        await enqueue("run_collection_job", job.id)
        return job

    async def get_dataset_preview(self, db: AsyncSession, dataset_id: int, rows: int = 100, timeframe: str | None = None) -> list[dict]:
        """Return up to `rows` OHLC rows of a dataset as a list of dicts.

        DDM tick directories: reads the most-recent batch files, resamples to
        OHLC, and returns the most-recent `rows` candles so an ongoing endless
        simulation always shows the current state.

        File-based OHLC datasets: returns the first `rows` rows.
        """
        import os
        from pathlib import Path
        import numpy as np
        import pandas as pd

        dataset = await self.get_dataset(db, dataset_id)
        if dataset.artifact_path is None or dataset.status not in ("ready", "running"):
            raise HTTPException(status_code=422, detail=DATASET_NOT_READY)

        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
        full_path = store / dataset.artifact_path

        is_tick_dir = full_path.is_dir()

        if is_tick_dir:
            # Read only the most-recent fragment files to keep load bounded.
            # 20 files × ~10 000 ticks ≈ 4-5 days of M1 data — enough for preview.
            from data.parquet_reader import load_ddm_ticks_recent
            try:
                tick_df = load_ddm_ticks_recent(full_path, n_files=20)
            except Exception:
                return []

            if tick_df.empty or "price" not in tick_df.columns:
                return []

            tf = timeframe or dataset.timeframe or "M1"
            df = _resample_ticks(tick_df["price"], tf)
            # Return the most-recent candles to reflect current simulation state.
            df = df.tail(rows)

        else:
            df = pd.read_parquet(full_path)

            if "price" in df.columns:
                # Legacy tick file (not a directory) — normalise index then resample.
                if "datetime" in df.columns:
                    df = df.set_index("datetime")
                df.index = pd.to_datetime(df.index, utc=True)
                tf = timeframe or dataset.timeframe or "M1"
                df = _resample_ticks(df["price"], tf)

            df = df.head(rows)

        # Sanitise ±inf so they don't become null in the JSON response.
        df = df.replace([np.inf, -np.inf], np.nan)
        df.index = df.index.astype(str)
        records = df.reset_index().rename(columns={df.index.name or "index": "datetime"})
        return records.to_dict(orient="records")


data_service = DataService()
