"""Data Management layer — business logic."""

from __future__ import annotations

import re

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
    "spread":   ["spread"],
    "datetime": ["datetime", "date", "time", "timestamp", "gmt time", "utc time", "local time"],
}

# dd.mm.yyyy[ HH:MM[:SS[.fff]]][ GMT+hhmm] -- Dukascopy exports. pandas would read it month-first
# (and fail from the 13th of a month), so day-first formats are tried explicitly.
_DAYFIRST = re.compile(r"^\s*\d{2}\.\d{2}\.\d{4}")
_GMT_SUFFIX = re.compile(r"\s*GMT([+-]\d{2}):?(\d{2})\s*$")
_DAYFIRST_FORMATS = ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y")


def _parse_datetimes(values, datetime_format: str | None = None) -> "pd.DatetimeIndex":
    """Parse a datetime column. An explicit `datetime_format` wins; day-first dd.mm.yyyy values
    are parsed with explicit day-first formats; anything else goes to pd.to_datetime. A trailing
    "GMT+hhmm" offset (Dukascopy local-time exports) is applied, giving naive UTC times."""
    s = pd.Series(values).astype(str)
    if datetime_format:
        return pd.DatetimeIndex(pd.to_datetime(s, format=datetime_format))
    if len(s) == 0 or not _DAYFIRST.match(s.iloc[0]):
        return pd.DatetimeIndex(pd.to_datetime(s))
    offset = s.str.extract(_GMT_SUFFIX)
    s = s.str.replace(_GMT_SUFFIX, "", regex=True)
    for fmt in _DAYFIRST_FORMATS:
        try:
            idx = pd.to_datetime(s, format=fmt)
            break
        except (ValueError, TypeError):
            continue
    else:
        idx = pd.to_datetime(s, dayfirst=True, format="mixed")
    if offset[0].notna().any():
        sign = offset[0].str[0].map({"+": 1, "-": -1}).fillna(1)
        minutes = sign * (offset[0].str[1:].astype(float).abs() * 60 + offset[1].astype(float))
        idx = idx - pd.to_timedelta(minutes.fillna(0), unit="min")
    return pd.DatetimeIndex(idx)

# Column names recognised as a single tick price (used when no OHLC close is found)
_TICK_ALIASES = ["price", "bid", "ask", "last", "mid", "tick"]


def _parse_csv_bytes(contents: bytes, col_map: dict | None, filename: str,
                     datetime_format: str | None = None) -> "pd.DataFrame":
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
        df.index = _parse_datetimes(df.index, datetime_format)
        df.index.name = "datetime"

    # ── OHLC mode ────────────────────────────────────────────────────────────
    close_src = _resolve("close")
    if close_src is not None:
        rename: dict[str, str] = {}
        for field in ("close", "open", "high", "low", "volume", "spread"):
            src = _resolve(field)
            if src and src not in rename:
                rename[src] = field
        df = df.rename(columns=rename)
        keep = [c for c in ["open", "high", "low", "close", "volume", "spread"] if c in df.columns]
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


def _extract_csvs_from_zip(contents: bytes, col_map: dict | None,
                           datetime_format: str | None = None) -> "dict[str, pd.DataFrame]":
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
                df = _parse_csv_bytes(csv_bytes, col_map, name, datetime_format)
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
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
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

    async def resolve_dataset_artifact(self, db: AsyncSession, dataset_id: int) -> tuple[Dataset, "Path"]:
        import os
        from pathlib import Path

        dataset = await self.get_dataset(db, dataset_id)
        if dataset.artifact_path is None or dataset.status not in ("ready", "running"):
            raise HTTPException(status_code=422, detail=DATASET_NOT_READY)

        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts")).resolve()
        custom_web_store = os.getenv("WEB_REPORTS_STORE_PATH")

        if dataset.artifact_path.startswith("web_reports") and custom_web_store:
            custom_root = Path(custom_web_store).resolve()
            subpath = Path(dataset.artifact_path).relative_to("web_reports")
            full_path = (custom_root / subpath).resolve()
            check_base = custom_root
        else:
            full_path = (store / dataset.artifact_path).resolve()
            check_base = store

        try:
            full_path.relative_to(check_base)
        except ValueError as exc:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "DATASET_ARTIFACT_INVALID",
                    "message": "Dataset artifact resolves outside the artifact store",
                },
            ) from exc

        if not full_path.exists():
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "DATASET_FILE_NOT_FOUND",
                    "message": "Dataset artifact not found",
                },
            )

        return dataset, full_path

    async def get_dataset_download_info(self, db: AsyncSession, dataset_id: int) -> dict:
        dataset, artifact = await self.resolve_dataset_artifact(db, dataset_id)
        is_directory = artifact.is_dir()
        return {
            "id": dataset.id,
            "name": dataset.name,
            "status": dataset.status,
            "artifact_path": dataset.artifact_path,
            "artifact_kind": "directory" if is_directory else "file",
            "download_format": "zip" if is_directory else artifact.suffix.lstrip(".") or "binary",
            "download_filename": f"dataset-{dataset.id}.zip" if is_directory else artifact.name,
        }

    async def list_collection_jobs(self, db: AsyncSession, datasource_id: int | None = None, offset: int = 0, limit: int = 20) -> tuple[list, int]:
        return await data_repo.get_collection_jobs(db, datasource_id=datasource_id, offset=offset, limit=limit)

    async def get_collection_job(self, db: AsyncSession, job_id: int):
        obj = await data_repo.get_collection_job(db, job_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=COLLECTION_JOB_NOT_FOUND)
        return obj

    async def create_collection_job(self, db: AsyncSession, body: CollectionJobCreate):
        source = await self.get_datasource(db, body.datasource_id)
        from data.scheduler import compute_next_run
        next_run_at = compute_next_run(source.type, source.config, body.schedule_cron)
        return await data_repo.create_collection_job(
            db, datasource_id=body.datasource_id, schedule_cron=body.schedule_cron, next_run_at=next_run_at
        )

    async def get_characteristics(self, db: AsyncSession, dataset_id: int):
        await self.get_dataset(db, dataset_id)
        return await data_repo.get_characteristics(db, dataset_id)

    async def trigger_collection(self, db: AsyncSession, job_id: int, overrides: dict | None = None):
        """Enqueue a run_collection_job Celery task and return the job.

        If the job is already running (status=running or dedup lock held), the
        existing Celery task is revoked first so the new run starts cleanly
        without a concurrent instance writing to the same artifact directory.
        """
        job = await self.get_collection_job(db, job_id)
        _revoke_collection_task(job_id)
        from celery_app import enqueue
        if overrides:
            await enqueue("run_collection_job", job_id, overrides)
        else:
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
        datetime_format: str | None = None,
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
                dfs = _extract_csvs_from_zip(contents, col_map, datetime_format)
                if not dfs:
                    raise HTTPException(status_code=422, detail={
                        "code": "INVALID_ZIP",
                        "message": f"'{filename}': no readable CSV files found inside the ZIP archive.",
                    })
                for csv_name, df in dfs.items():
                    file_dfs.append((csv_name.removesuffix(".csv"), df))
            else:
                df = _parse_csv_bytes(contents, col_map, filename, datetime_format)
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

    async def trigger_datasource_collection(self, db: AsyncSession, datasource_id: int, body=None, overrides: dict | None = None):
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
        if overrides:
            await enqueue("run_collection_job", job.id, overrides)
        else:
            await enqueue("run_collection_job", job.id)
        return job

    async def get_dataset_preview(self, db: AsyncSession, dataset_id: int, rows: int = 100, timeframe: str | None = None) -> list[dict]:
        """Return up to `rows` OHLC rows of a dataset as a list of dicts.

        DDM tick directories: reads the most-recent batch files, resamples to
        OHLC, and returns the most-recent `rows` candles so an ongoing endless
        simulation always shows the current state.

        File-based OHLC datasets: returns the first `rows` rows.
        """
        import numpy as np
        import pandas as pd

        dataset, full_path = await self.resolve_dataset_artifact(db, dataset_id)

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

    async def get_or_fetch_report_forecast(
        self,
        symbol: str = "USDJPY",
        target_date: date | None = None,
        source: str | None = None,
        horizon: str = "Daily",
    ) -> dict | None:
        """Fetch report forecast for a symbol & target date.

        Autonomous multi-tiered lookup:
        1. DB cache (`report_forecasts` table)
        2. Local disk scan & on-the-fly extraction
        3. On-demand collection trigger & extraction
        """
        import os
        from datetime import date as dt_date
        from pathlib import Path
        import sqlalchemy as sa
        from database import SessionLocal
        from data.models import ReportForecast
        from data.extractors.report_forecast import extract_forecast_from_pdf, save_report_forecast

        t_date = target_date or dt_date.today()
        src_filter = source or "SMBC信託"

        # Tier 1: DB Lookup
        with SessionLocal() as db:
            stmt = sa.select(ReportForecast).where(
                ReportForecast.symbol == symbol,
                ReportForecast.horizon == horizon,
                ReportForecast.report_date == t_date,
            )
            if source:
                stmt = stmt.where(ReportForecast.source == source)
            res = db.execute(stmt).scalars().first()
            if res is not None:
                return {
                    "id": res.id,
                    "source": res.source,
                    "symbol": res.symbol,
                    "horizon": res.horizon,
                    "report_date": res.report_date.isoformat(),
                    "analyst": res.analyst,
                    "direction": res.direction,
                    "expected_low": res.expected_low,
                    "expected_high": res.expected_high,
                    "title": res.title,
                }

        # Tier 2: Check Local Disk & Extract On-the-Fly
        custom_root = os.getenv("WEB_REPORTS_STORE_PATH")
        reports_root = Path(custom_root) if custom_root else (Path("artifacts") / "web_reports")

        subfolder = "SMBC" if "SMBC" in src_filter else ("みんなのFX" if "みんな" in src_filter else "SonyFinance")
        target_dir = reports_root / subfolder
        candidate_files = []
        if target_dir.exists():
            date_str = t_date.strftime("%Y%m%d")
            short_date = t_date.strftime("%y%m%d")
            candidate_files = list(target_dir.glob(f"*{date_str}*")) + list(target_dir.glob(f"*{short_date}*"))

        for cf in candidate_files:
            fc = extract_forecast_from_pdf(cf)
            if fc and fc.get("report_date") == t_date:
                with SessionLocal() as db:
                    saved = save_report_forecast(db, fc)
                    db.commit()
                    if saved:
                        return {
                            "id": saved.id,
                            "source": saved.source,
                            "symbol": saved.symbol,
                            "horizon": saved.horizon,
                            "report_date": saved.report_date.isoformat(),
                            "analyst": saved.analyst,
                            "direction": saved.direction,
                            "expected_low": saved.expected_low,
                            "expected_high": saved.expected_high,
                            "title": saved.title,
                        }

        # Tier 3: Trigger On-Demand Collection for missing report
        try:
            datasources = await self.list_datasources()
            matched_ds = next((ds for ds in datasources if src_filter in ds.name or subfolder in ds.name), None)
            if matched_ds is not None:
                logger.info("[get_or_fetch_forecast] triggering on-demand collection for datasource %s (%s)", matched_ds.id, matched_ds.name)
                # Run collection synchronously for target_date
                from data.collectors import web_report
                cfg = dict(matched_ds.config)
                cfg["backfill_days"] = 3
                web_report.collect(matched_ds.id, cfg, force=True)

                # Re-check Tier 2
                if target_dir.exists():
                    for cf in target_dir.glob(f"*{t_date.strftime('%Y%m%d')}*"):
                        fc = extract_forecast_from_pdf(cf)
                        if fc and fc.get("report_date") == t_date:
                            with SessionLocal() as db:
                                saved = save_report_forecast(db, fc, datasource_id=matched_ds.id)
                                db.commit()
                                if saved:
                                    return {
                                        "id": saved.id,
                                        "source": saved.source,
                                        "symbol": saved.symbol,
                                        "horizon": saved.horizon,
                                        "report_date": saved.report_date.isoformat(),
                                        "analyst": saved.analyst,
                                        "direction": saved.direction,
                                        "expected_low": saved.expected_low,
                                        "expected_high": saved.expected_high,
                                        "title": saved.title,
                                    }
        except Exception as exc:
            logger.warning("[get_or_fetch_forecast] on-demand fetch failed: %s", exc)

        return None

    def enrich_dataframe_with_forecasts(
        self,
        df: "pd.DataFrame",
        symbol: str = "USDJPY",
        horizon: str = "Daily",
        source: str = "SMBC信託",
    ) -> "pd.DataFrame":
        """Enrich OHLCV DataFrame with analyst expected range & breakout features."""
        import pandas as pd
        import sqlalchemy as sa
        from database import SessionLocal
        from data.models import ReportForecast

        with SessionLocal() as db:
            stmt = sa.select(ReportForecast).where(
                ReportForecast.symbol == symbol,
                ReportForecast.horizon == horizon,
                ReportForecast.source == source,
            )
            rows = db.execute(stmt).scalars().all()

        if not rows:
            return df

        fc_data = [
            {
                "forecast_date": pd.Timestamp(r.report_date),
                "fc_low": r.expected_low,
                "fc_high": r.expected_high,
                "fc_width": (r.expected_high - r.expected_low) if r.expected_high and r.expected_low else None,
                "fc_direction": r.direction,
                "fc_analyst": r.analyst,
            }
            for r in rows
            if r.expected_low is not None and r.expected_high is not None
        ]
        df_fc = pd.DataFrame(fc_data).drop_duplicates("forecast_date").set_index("forecast_date")

        res_df = df.copy()
        if not isinstance(res_df.index, pd.DatetimeIndex):
            if "date" in res_df.columns:
                res_df["dt_idx"] = pd.to_datetime(res_df["date"])
                res_df = res_df.set_index("dt_idx")
            elif "timestamp" in res_df.columns:
                res_df["dt_idx"] = pd.to_datetime(res_df["timestamp"])
                res_df = res_df.set_index("dt_idx")

        # Normalize to date for matching
        orig_index = res_df.index
        res_df["_match_date"] = res_df.index.normalize()
        res_df = res_df.merge(df_fc, left_on="_match_date", right_index=True, how="left").drop(columns=["_match_date"])
        res_df.index = orig_index

        # Add breakout and range features if High/Low are available
        if "High" in res_df.columns and "Low" in res_df.columns:
            res_df["is_in_range"] = (res_df["Low"] >= res_df["fc_low"] - 0.10) & (res_df["High"] <= res_df["fc_high"] + 0.10)
            res_df["is_high_break"] = res_df["High"] > (res_df["fc_high"] + 0.10)
            res_df["is_low_break"] = res_df["Low"] < (res_df["fc_low"] - 0.10)

        return res_df


data_service = DataService()
