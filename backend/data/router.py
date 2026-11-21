"""Data Management layer — HTTP endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from pagination import Pagination
from schemas import DataResponse, Meta
from data.models import (
    DatasourceCreate, DatasourceUpdate, DatasourceRead,
    DatasetRead, DatasetUpdate, CollectionJobCreate, CollectionJobRead,
    CollectionJobUpdate, CollectionJobRunRead,
    DataCharacteristicsRead,
)
from data.service import data_service
from data.repository import data_repo

router = APIRouter(prefix="", tags=["data"])


# ── Web-report link preview ───────────────────────────────────────────────────

class WebReportPreviewRequest(BaseModel):
    url: str
    ext: str | None = None


@router.post("/datasources/web-report/preview-links")
async def preview_web_report_links(body: WebReportPreviewRequest):
    """
    Navigate to *url* with a headless browser and return all <a> links found,
    annotated with whether they match the requested file extension.
    Runs Playwright synchronously in a thread-pool executor.
    """
    import logging
    from data.collectors.web_report import preview_links
    try:
        links = await asyncio.get_event_loop().run_in_executor(
            None, preview_links, body.url, body.ext
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("[preview_links] scan failed for %s", body.url)
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail={"code": "SCAN_FAILED", "message": str(exc)})
    matched = sum(1 for l in links if l["matches_ext"])
    return DataResponse(data=links, meta={"total": len(links), "matches": matched})


class WebReportTestRequest(BaseModel):
    url: str
    fetch_type: str = "load"
    ext: str | None = None


@router.post("/datasources/web-report/test-fetch")
async def test_web_report_fetch(body: WebReportTestRequest):
    """
    Test whether the configured fetch method can access *url*.
    Returns HTTP status, content-type, file size, and links (for HTML responses).
    Runs synchronously in a thread-pool executor.
    """
    import logging
    from data.collectors.web_report import test_fetch
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, test_fetch, body.url, body.fetch_type, body.ext
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("[test_fetch] failed for %s", body.url)
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail={"code": "TEST_FAILED", "message": str(exc)})
    return DataResponse(data=result)


# ── Web-report file browser ───────────────────────────────────────────────────

@router.get("/datasources/{datasource_id}/web-report/files")
async def list_web_report_files(datasource_id: int, db: AsyncSession = Depends(get_db)):
    """List files downloaded by a web_report datasource."""
    import os
    from datetime import datetime, timezone
    from pathlib import Path
    from fastapi import HTTPException

    item = await data_service.get_datasource(db, datasource_id)
    if item.type != "web_report":
        raise HTTPException(status_code=404, detail="Not a web_report datasource")

    subfolder = (item.config or {}).get("subfolder") or f"src_{datasource_id}"
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    report_dir = store / "web_reports" / subfolder

    if not report_dir.exists():
        return DataResponse(data=[], meta={"total": 0})

    files = []
    for f in sorted(report_dir.rglob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.is_file():
            stat = f.stat()
            rel = f.relative_to(report_dir)
            files.append({
                "name": f.name,
                "path": str(rel).replace("\\", "/"),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            })

    return DataResponse(data=files, meta={"total": len(files)})


@router.get("/datasources/{datasource_id}/web-report/files/{file_path:path}")
async def serve_web_report_file(datasource_id: int, file_path: str, db: AsyncSession = Depends(get_db)):
    """Serve a downloaded web report file."""
    import os
    from pathlib import Path
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    item = await data_service.get_datasource(db, datasource_id)
    if item.type != "web_report":
        raise HTTPException(status_code=404, detail="Not a web_report datasource")

    subfolder = (item.config or {}).get("subfolder") or f"src_{datasource_id}"
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    report_dir = store / "web_reports" / subfolder

    # Prevent path traversal
    target = (report_dir / file_path).resolve()
    if not str(target).startswith(str(report_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(target))


# ── Datasources ────────────────────────────────────────────────────────────────

@router.get("/datasources", response_model=DataResponse[list[DatasourceRead]])
async def list_datasources(
    pagination: Pagination = Depends(),
    db: AsyncSession = Depends(get_db),
):
    items, total = await data_service.list_datasources(db, offset=pagination.offset, limit=pagination.page_size)
    return DataResponse(data=items, meta=Meta(total=total, page=pagination.page, page_size=pagination.page_size))


@router.post("/datasources", response_model=DataResponse[DatasourceRead], status_code=201)
async def create_datasource(
    body: DatasourceCreate,
    auto_collect: bool = Query(False, description="Automatically trigger initial collection upon creation"),
    backfill_days: int | None = Query(None, description="Optional custom backfill days for initial collection"),
    db: AsyncSession = Depends(get_db),
):
    item = await data_service.create_datasource(db, body)
    if auto_collect:
        overrides = {"is_initial": True}
        if backfill_days is not None:
            overrides["backfill_days"] = backfill_days
        try:
            await data_service.trigger_datasource_collection(db, item.id, overrides=overrides)
        except Exception as exc:
            import logging
            logging.getLogger("data.router").warning(f"Auto-collect on datasource {item.id} failed to enqueue: {exc}")
    return DataResponse(data=item)


@router.get("/datasources/{datasource_id}", response_model=DataResponse[DatasourceRead])
async def get_datasource(datasource_id: int, db: AsyncSession = Depends(get_db)):
    item = await data_service.get_datasource(db, datasource_id)
    return DataResponse(data=item)


@router.patch("/datasources/{datasource_id}", response_model=DataResponse[DatasourceRead])
async def update_datasource(datasource_id: int, body: DatasourceUpdate, db: AsyncSession = Depends(get_db)):
    item = await data_service.update_datasource(db, datasource_id, body)
    return DataResponse(data=item)


@router.delete("/datasources/{datasource_id}", status_code=204)
async def delete_datasource(datasource_id: int, db: AsyncSession = Depends(get_db)):
    await data_service.delete_datasource(db, datasource_id)


@router.post("/datasources/{datasource_id}/collect", status_code=202)
async def trigger_collection(
    datasource_id: int,
    backfill_days: int | None = Query(None, description="Number of past business days to backfill"),
    is_initial: bool = Query(False, description="Whether to treat as initial collection (e.g. 30 days backfill)"),
    db: AsyncSession = Depends(get_db),
):
    from fastapi import HTTPException
    from celery_app import AlreadyRunningError
    overrides: dict = {}
    if backfill_days is not None:
        overrides["backfill_days"] = backfill_days
    if is_initial:
        overrides["is_initial"] = True
    try:
        job = await data_service.trigger_datasource_collection(db, datasource_id, overrides=overrides or None)
    except AlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail={"code": "ALREADY_RUNNING", "message": str(exc)})
    # job.status is still the previous run's (e.g. "error") until the worker picks the task up;
    # report what just happened so callers do not enqueue it again.
    return DataResponse(data={"job_id": job.id, "status": "queued", "previous_status": job.status})


# ── Datasource config ──────────────────────────────────────────────────────────

@router.get("/datasource-config/types/{type_name}")
async def get_datasource_type_schema(type_name: str):
    """Get configuration schema for a datasource type."""
    schemas = {
        "ohlc_download": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": {"type": "string", "enum": ["M1", "M5", "M15", "H1", "H4", "D1"]},
                "provider": {"type": "string", "enum": ["yfinance", "vantage"]},
            },
            "required": ["symbol", "timeframe"],
        },
        "manual_upload": {
            "type": "object",
            "properties": {},
        },
    }
    schema = schemas.get(type_name.lower())
    if not schema:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail={"code": "TYPE_NOT_FOUND", "message": f"Datasource type {type_name!r} not found"})
    return DataResponse(data=schema)


# ── Datasets ───────────────────────────────────────────────────────────────────

@router.get("/datasets", response_model=DataResponse[list[DatasetRead]])
async def list_datasets(
    symbol: str | None = None,
    timeframe: str | None = None,
    datasource_id: int | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(1000, ge=1, le=5000),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    items, total = await data_service.list_datasets(
        db, symbol=symbol, timeframe=timeframe, datasource_id=datasource_id, offset=offset, limit=page_size
    )
    return DataResponse(data=items, meta=Meta(total=total, page=page, page_size=page_size))


@router.get("/datasets/{dataset_id}", response_model=DataResponse[DatasetRead])
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    item = await data_service.get_dataset(db, dataset_id)
    return DataResponse(data=item)


@router.patch("/datasets/{dataset_id}", response_model=DataResponse[DatasetRead])
async def update_dataset(dataset_id: int, body: DatasetUpdate, db: AsyncSession = Depends(get_db)):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        item = await data_service.get_dataset(db, dataset_id)
    else:
        item = await data_repo.update_dataset(db, dataset_id, **updates)
        if item is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="DATASET_NOT_FOUND")
    return DataResponse(data=item)


@router.get("/datasets/{dataset_id}/characteristics", response_model=DataResponse[DataCharacteristicsRead | None])
async def get_characteristics(dataset_id: int, db: AsyncSession = Depends(get_db)):
    item = await data_service.get_characteristics(db, dataset_id)
    return DataResponse(data=item)


@router.post("/datasets/{dataset_id}/characteristics/compute", response_model=DataResponse[DatasetRead], status_code=202)
async def compute_characteristics(dataset_id: int, db: AsyncSession = Depends(get_db)):
    """Enqueue characteristic analysis for a dataset."""
    item = await data_service.trigger_analysis(db, dataset_id)
    return DataResponse(data=item)


@router.get("/datasets/{dataset_id}/live-progress")
async def dataset_live_progress(dataset_id: int, db: AsyncSession = Depends(get_db)):
    """Return live tick-count progress for a running DDM simulation dataset.

    Reads the _meta.json written by the collector after each batch flush.
    Returns null if the dataset is not a DDM simulation or hasn't written a batch yet.
    """
    from data.models import Dataset, Datasource
    from data.collectors.ddm_simulator import read_meta
    from sqlalchemy import select

    result = await db.execute(
        select(Dataset, Datasource)
        .join(Datasource, Dataset.datasource_id == Datasource.id)
        .where(Dataset.id == dataset_id)
    )
    row = result.first()
    if row is None or row.Datasource.type != "ddm_simulation":
        return DataResponse(data=None)

    meta = read_meta(row.Datasource.id)
    return DataResponse(data=meta if meta else None)


@router.get("/datasets/{dataset_id}/preview", response_model=DataResponse[list])
async def preview_dataset(dataset_id: int, rows: int = 100, timeframe: str | None = None, db: AsyncSession = Depends(get_db)):
    """Return the first N rows of the dataset as JSON.

    For tick datasets (DDM simulation) pass `timeframe` to control OHLC aggregation.
    Defaults to the dataset's stored timeframe.
    """
    items = await data_service.get_dataset_preview(db, dataset_id, rows=rows, timeframe=timeframe)
    return DataResponse(data=items)


@router.post("/datasets/upload", status_code=202)
async def upload_dataset(
    files: list[UploadFile] = File(..., description="One or more CSV or ZIP files."),
    datasource_id: int | None = Form(None),
    symbol: str | None = Form(None),
    timeframe: str | None = Form(None),
    close_col: str | None = Form(None),
    open_col: str | None = Form(None),
    high_col: str | None = Form(None),
    low_col: str | None = Form(None),
    volume_col: str | None = Form(None),
    spread_col: str | None = Form(None),
    datetime_col: str | None = Form(None),
    datetime_format: str | None = Form(None, description="strptime format of the datetime column, e.g. %d.%m.%Y %H:%M:%S.%f. "
                                                          "Default: auto (day-first dd.mm.yyyy is detected)."),
    append_to: int | None = Form(None, description="Append into this existing dataset ID instead of creating a new one."),
    merge: bool = Form(True, description="Merge all files/CSVs into one dataset (True) or create one dataset per file (False)."),
    db: AsyncSession = Depends(get_db),
):
    """Upload one or more CSV or ZIP files to create or append to a dataset.

    - **Single CSV**: creates one dataset (or appends when `append_to` is set).
    - **Single ZIP**: the ZIP is unpacked; all CSVs inside are merged into one dataset
      when `merge=True`, or each becomes its own dataset when `merge=False`.
    - **Multiple files** (CSV or ZIP, any mix): all data is merged into one dataset
      when `merge=True`, or one dataset is created per file when `merge=False`.
    - Column names are auto-detected from the first file; use `*_col` to override.
    """
    col_map = {k: v for k, v in {
        "close": close_col, "open": open_col, "high": high_col,
        "low": low_col, "volume": volume_col, "spread": spread_col, "datetime": datetime_col,
    }.items() if v}
    datasets = await data_service.create_dataset_from_upload(
        db, files,
        datasource_id=datasource_id, symbol=symbol, timeframe=timeframe,
        col_map=col_map or None, append_to=append_to, merge=merge, datetime_format=datetime_format,
    )
    data: dict = {"dataset_id": datasets[0].id, "status": datasets[0].status}
    if len(datasets) > 1:
        data["dataset_ids"] = [d.id for d in datasets]
    return DataResponse(data=data)


@router.post("/datasets/{dataset_id}/analyze", status_code=202)
async def analyze_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    item = await data_service.trigger_analysis(db, dataset_id)
    return DataResponse(data={"status": item.status})


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    await data_service.delete_dataset(db, dataset_id)


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: int, timeframe: str | None = None, db: AsyncSession = Depends(get_db)):
    import io
    import pandas as pd
    from fastapi.responses import StreamingResponse as _SR
    from data.service import _resample_ticks

    ds, artifact_path = await data_service.resolve_dataset_artifact(db, dataset_id)
    df = pd.read_parquet(artifact_path)

    if "price" in df.columns:
        df = _resample_ticks(df["price"], timeframe or ds.timeframe or "M1")

    buf = io.StringIO()
    df.to_csv(buf, index=True)
    buf.seek(0)
    return _SR(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="dataset-{dataset_id}.csv"'},
    )


@router.get("/datasets/{dataset_id}/artifact")
async def download_dataset_artifact(dataset_id: int, db: AsyncSession = Depends(get_db)):
    import shutil
    import tempfile
    from pathlib import Path
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    dataset, artifact_path = await data_service.resolve_dataset_artifact(db, dataset_id)

    if artifact_path.is_file():
        return FileResponse(
            str(artifact_path),
            media_type="application/octet-stream",
            filename=artifact_path.name,
        )

    temp_dir = Path(tempfile.mkdtemp(prefix=f"algoforge-dataset-{dataset_id}-"))
    archive_base = temp_dir / f"dataset-{dataset_id}"
    archive_path = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=str(artifact_path.parent),
        base_dir=artifact_path.name,
    )

    def _cleanup(path: str, directory: str) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=f"dataset-{dataset.id}.zip",
        background=BackgroundTask(_cleanup, archive_path, str(temp_dir)),
    )


# ── Collection Jobs ────────────────────────────────────────────────────────────

@router.get("/collection-jobs", response_model=DataResponse[list[CollectionJobRead]])
async def list_collection_jobs(
    datasource_id: int | None = None,
    pagination: Pagination = Depends(),
    db: AsyncSession = Depends(get_db),
):
    items, total = await data_service.list_collection_jobs(db, datasource_id=datasource_id, offset=pagination.offset, limit=pagination.page_size)
    return DataResponse(data=items, meta=Meta(total=total, page=pagination.page, page_size=pagination.page_size))


@router.post("/collection-jobs", response_model=DataResponse[CollectionJobRead], status_code=201)
async def create_collection_job(body: CollectionJobCreate, db: AsyncSession = Depends(get_db)):
    item = await data_service.create_collection_job(db, body)
    return DataResponse(data=item)


@router.get("/collection-jobs/{job_id}", response_model=DataResponse[CollectionJobRead])
async def get_collection_job(job_id: int, db: AsyncSession = Depends(get_db)):
    item = await data_service.get_collection_job(db, job_id)
    return DataResponse(data=item)


@router.post("/collection-jobs/{job_id}/reset", response_model=DataResponse[CollectionJobRead])
async def reset_collection_job(job_id: int, db: AsyncSession = Depends(get_db)):
    """Force-reset a stuck job from 'running' back to 'idle'. Use after an ungraceful worker restart."""
    from sqlalchemy import update
    from data.models import CollectionJob
    from data.service import _revoke_collection_task
    _revoke_collection_task(job_id)
    await db.execute(
        update(CollectionJob).where(CollectionJob.id == job_id).values(status="idle", last_error=None)
    )
    await db.commit()
    item = await data_service.get_collection_job(db, job_id)
    return DataResponse(data=item)


@router.post("/collection-jobs/{job_id}/run", response_model=DataResponse[CollectionJobRead], status_code=202)
async def run_collection_job(
    job_id: int,
    backfill_days: int | None = Query(None, description="Number of past business days to backfill"),
    is_initial: bool = Query(False, description="Whether to treat as initial collection"),
    db: AsyncSession = Depends(get_db),
):
    """Trigger an immediate collection run."""
    from fastapi import HTTPException
    from celery_app import AlreadyRunningError
    overrides: dict = {}
    if backfill_days is not None:
        overrides["backfill_days"] = backfill_days
    if is_initial:
        overrides["is_initial"] = True
    try:
        item = await data_service.trigger_collection(db, job_id, overrides=overrides or None)
    except AlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail={"code": "ALREADY_RUNNING", "message": str(exc)})
    return DataResponse(data=item)


@router.patch("/collection-jobs/{job_id}", response_model=DataResponse[CollectionJobRead])
async def update_collection_job(job_id: int, body: CollectionJobUpdate, db: AsyncSession = Depends(get_db)):
    item = await data_service.update_collection_job(db, job_id, body)
    return DataResponse(data=item)


@router.delete("/collection-jobs/{job_id}", status_code=204)
async def delete_collection_job(job_id: int, db: AsyncSession = Depends(get_db)):
    await data_service.delete_collection_job(db, job_id)


@router.get("/collection-jobs/{job_id}/runs")
async def get_job_run_history(job_id: int, db: AsyncSession = Depends(get_db)):
    runs = await data_service.list_job_runs(db, job_id)
    return DataResponse(data=[CollectionJobRunRead.model_validate(r) for r in runs], meta=Meta(total=len(runs)))


@router.get("/analyses", response_model=DataResponse[list[str]])
async def list_analyses():
    """List all registered characteristic analysis names."""
    from data.characteristics import CHARACTERISTIC_REGISTRY
    return DataResponse(data=list(CHARACTERISTIC_REGISTRY.keys()))


@router.get("/dataset-config/analyses")
async def list_analyses_config():
    """List all registered characteristic analysis names with descriptions."""
    from data.characteristics import CHARACTERISTIC_REGISTRY
    return DataResponse(data=[
        {"name": name, "description": getattr(fn, "__doc__", "") or "", "output_metrics": []}
        for name, fn in CHARACTERISTIC_REGISTRY.items()
    ])


@router.get("/reports/forecasts")
async def get_report_forecast(
    symbol: str = Query("USDJPY", description="Currency pair or symbol"),
    target_date: str | None = Query(None, description="ISO date (YYYY-MM-DD). Defaults to today"),
    source: str | None = Query(None, description="Source filter (e.g. SMBC信託, みんなのFX, ソニーFG)"),
    horizon: str = Query("Daily", description="Forecast horizon (Daily or Weekly)"),
):
    """Retrieve report forecast with autonomous fallback (DB -> Disk -> On-demand fetch)."""
    from datetime import date as dt_date
    t_date = dt_date.fromisoformat(target_date) if target_date else dt_date.today()
    fc = await data_service.get_or_fetch_report_forecast(
        symbol=symbol,
        target_date=t_date,
        source=source,
        horizon=horizon,
    )
    if fc is None:
        raise HTTPException(status_code=404, detail="No forecast available for given parameters")
    return DataResponse(data=fc)


@router.post("/reports/forecasts/sync")
async def sync_report_forecasts():
    """Bulk sync historical report forecasts from reports store into DB."""
    import os
    from pathlib import Path
    from data.extractors.report_forecast import extract_forecast_from_pdf, save_report_forecast
    from database import SessionLocal

    custom_root = os.getenv("WEB_REPORTS_STORE_PATH")
    reports_root = Path(custom_root) if custom_root else (Path("artifacts") / "web_reports")

    synced_count = 0
    with SessionLocal() as db:
        for folder in ["SMBC", "みんなのFX", "SonyFinance"]:
            f_dir = reports_root / folder
            if not f_dir.exists():
                continue
            for pdf_file in f_dir.glob("*.pdf"):
                fc = extract_forecast_from_pdf(pdf_file)
                if fc:
                    save_report_forecast(db, fc)
                    synced_count += 1
        db.commit()

    return DataResponse(data={"synced_records": synced_count})


# ── Swing Trading: Margin & Large Holding Data ─────────────────────────────────

@router.get("/margin/daily")
async def get_daily_margin_records(
    code: str | None = Query(None, description="Symbol code filter (e.g. 3042, 3656)"),
    order_by: str = Query("buy_ratio", description="Order field: buy_ratio, sell_ratio, ratio"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Query latest JPX daily margin trading records with sorting and filters."""
    import sqlalchemy as sa
    from data.models import DailyMarginRecord

    # Find the latest as_of_date
    latest_date_q = sa.select(sa.func.max(DailyMarginRecord.as_of_date))
    latest_date = (await db.execute(latest_date_q)).scalar_one_or_none()
    if not latest_date:
        return DataResponse(data=[])

    q = sa.select(DailyMarginRecord).where(DailyMarginRecord.as_of_date == latest_date)
    if code:
        q = q.where(DailyMarginRecord.code == code)

    if order_by == "sell_ratio":
        q = q.order_by(DailyMarginRecord.sell_ratio.desc())
    elif order_by == "ratio":
        q = q.order_by(DailyMarginRecord.ratio.desc())
    else:
        q = q.order_by(DailyMarginRecord.buy_ratio.desc())

    q = q.limit(limit)
    rows = (await db.execute(q)).scalars().all()

    return DataResponse(
        data=[
            {
                "as_of_date": r.as_of_date.isoformat(),
                "code": r.code,
                "name": r.name,
                "market": r.market,
                "issue_type": r.issue_type,
                "restrictions": r.restrictions,
                "sell_shares": r.sell_shares,
                "sell_change": r.sell_change,
                "sell_ratio": r.sell_ratio,
                "buy_shares": r.buy_shares,
                "buy_change": r.buy_change,
                "buy_ratio": r.buy_ratio,
                "ratio": r.ratio,
            }
            for r in rows
        ],
        meta=Meta(total=len(rows)),
    )


@router.get("/margin/regulations")
async def get_margin_regulations(
    event_type: str | None = Query(None, description="Filter by 'designated' or 'lifted'"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Query daily margin publication events (designation & lifting announcements)."""
    import sqlalchemy as sa
    from data.models import MarginRegulationRecord

    q = sa.select(MarginRegulationRecord)
    if event_type:
        q = q.where(MarginRegulationRecord.event_type == event_type)
    q = q.order_by(MarginRegulationRecord.event_date.desc()).limit(limit)
    rows = (await db.execute(q)).scalars().all()

    return DataResponse(
        data=[
            {
                "code": r.code,
                "name": r.name,
                "event_date": r.event_date.isoformat(),
                "event_type": r.event_type,
            }
            for r in rows
        ],
        meta=Meta(total=len(rows)),
    )


@router.post("/margin/sync")
async def sync_daily_margin_data():
    """Trigger on-demand fetch and sync of JPX daily margin balances & announcements."""
    import asyncio
    from data.swing_data_service import run_daily_margin_collection

    res = await asyncio.get_event_loop().run_in_executor(None, run_daily_margin_collection)
    return DataResponse(data=res)


@router.get("/large-holding/reports")
async def get_large_holding_reports(
    issuer_code: str | None = Query(None, description="Issuer symbol code filter (e.g. 4381)"),
    filer_name: str | None = Query(None, description="Filer name substring"),
    min_ratio_change: float | None = Query(None, description="Filter reports with absolute ratio_change >= value"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Query EDINET Large Volume Holding Reports (5% rule) with joint holders & change delta."""
    import sqlalchemy as sa
    from data.models import LargeHoldingReportRecord

    q = sa.select(LargeHoldingReportRecord)
    if issuer_code:
        q = q.where(LargeHoldingReportRecord.issuer_code == issuer_code)
    if filer_name:
        q = q.where(LargeHoldingReportRecord.filer_name.contains(filer_name))
    if min_ratio_change is not None:
        q = q.where(sa.func.abs(LargeHoldingReportRecord.ratio_change) >= min_ratio_change)

    q = q.order_by(LargeHoldingReportRecord.base_date.desc(), LargeHoldingReportRecord.extracted_at.desc()).limit(limit)
    rows = (await db.execute(q)).scalars().all()

    return DataResponse(
        data=[
            {
                "id": r.id,
                "doc_id": r.doc_id,
                "issuer_code": r.issuer_code,
                "issuer_name": r.issuer_name,
                "filer_name": r.filer_name,
                "base_date": r.base_date.isoformat() if r.base_date else None,
                "holding_ratio": r.holding_ratio,
                "prev_holding_ratio": r.prev_holding_ratio,
                "ratio_change": r.ratio_change,
                "total_shares_held": r.total_shares_held,
                "total_outstanding_shares": r.total_outstanding_shares,
                "purpose_of_holding": r.purpose_of_holding,
                "joint_holders": r.joint_holders_json,
                "trades_count": len(r.trades),
            }
            for r in rows
        ],
        meta=Meta(total=len(rows)),
    )


@router.post("/large-holding/sync")
async def sync_large_holding_data(
    target_date: str | None = Query(None, description="Target date YYYY-MM-DD. Defaults to today"),
    max_documents: int = Query(30, ge=1, le=100),
):
    """Trigger on-demand fetch and sync of EDINET Large Volume Holding Reports."""
    import asyncio
    from datetime import date as dt_date
    from data.swing_data_service import run_large_holding_collection

    t_dt = dt_date.fromisoformat(target_date) if target_date else dt_date.today()
    res = await asyncio.get_event_loop().run_in_executor(
        None, run_large_holding_collection, t_dt, max_documents
    )
    return DataResponse(data=res)


@router.get("/swing/signals")
async def get_swing_signals(
    target_date: str | None = Query(None, description="ISO date (YYYY-MM-DD). Defaults to latest date"),
    signal_type: str | None = Query(None, description="Filter: squeeze_alert, daily_margin_lifted, institutional_buy, supply_relief, supply_overhang"),
    direction: str | None = Query(None, description="Filter: LONG, SHORT, AVOID"),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Scan and generate actionable swing trading signals combining margin and holding data."""
    from datetime import date as dt_date
    from database import SessionLocal
    from strategy.engine.swing_signals import screen_swing_signals

    t_dt = dt_date.fromisoformat(target_date) if target_date else None

    # Run screening in sync session
    with SessionLocal() as sync_db:
        signals = screen_swing_signals(sync_db, target_date=t_dt, min_confidence=min_confidence)

    if signal_type:
        signals = [s for s in signals if s.signal_type == signal_type]
    if direction:
        signals = [s for s in signals if s.direction == direction]

    signals = signals[:limit]

    return DataResponse(
        data=[
            {
                "code": s.code,
                "name": s.name,
                "signal_type": s.signal_type,
                "direction": s.direction,
                "confidence_score": s.confidence_score,
                "detected_date": s.detected_date,
                "title": s.title,
                "reasons": s.reasons,
                "metrics": s.metrics,
            }
            for s in signals
        ],
        meta=Meta(total=len(signals)),
    )
