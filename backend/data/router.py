"""Data Management layer — HTTP endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, Form, UploadFile
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
async def create_datasource(body: DatasourceCreate, db: AsyncSession = Depends(get_db)):
    item = await data_service.create_datasource(db, body)
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
async def trigger_collection(datasource_id: int, db: AsyncSession = Depends(get_db)):
    from fastapi import HTTPException
    from celery_app import AlreadyRunningError
    try:
        job = await data_service.trigger_datasource_collection(db, datasource_id)
    except AlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail={"code": "ALREADY_RUNNING", "message": str(exc)})
    return DataResponse(data={"job_id": job.id, "status": job.status})


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
    pagination: Pagination = Depends(),
    db: AsyncSession = Depends(get_db),
):
    items, total = await data_service.list_datasets(db, symbol=symbol, timeframe=timeframe, datasource_id=datasource_id, offset=pagination.offset, limit=pagination.page_size)
    return DataResponse(data=items, meta=Meta(total=total, page=pagination.page, page_size=pagination.page_size))


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
    file: UploadFile = File(...),
    datasource_id: int | None = Form(None),
    symbol: str | None = Form(None),
    timeframe: str | None = Form(None),
    close_col: str | None = Form(None),
    open_col: str | None = Form(None),
    high_col: str | None = Form(None),
    low_col: str | None = Form(None),
    volume_col: str | None = Form(None),
    datetime_col: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    col_map = {k: v for k, v in {
        "close": close_col, "open": open_col, "high": high_col,
        "low": low_col, "volume": volume_col, "datetime": datetime_col,
    }.items() if v}
    dataset = await data_service.create_dataset_from_upload(
        db, file, datasource_id=datasource_id, symbol=symbol, timeframe=timeframe, col_map=col_map or None,
    )
    return DataResponse(data={"dataset_id": dataset.id, "status": dataset.status})


@router.post("/datasets/{dataset_id}/analyze", status_code=202)
async def analyze_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    item = await data_service.trigger_analysis(db, dataset_id)
    return DataResponse(data={"status": item.status})


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    await data_service.delete_dataset(db, dataset_id)


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: int, timeframe: str | None = None, db: AsyncSession = Depends(get_db)):
    import os
    import io
    from pathlib import Path
    import pandas as pd
    from fastapi.responses import StreamingResponse as _SR
    from data.service import _resample_ticks

    ds = await data_service.get_dataset(db, dataset_id)
    if not ds.artifact_path:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail={"code": "DATASET_FILE_NOT_FOUND", "message": "Dataset file not found"})
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    df = pd.read_parquet(store / ds.artifact_path)

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


@router.post("/collection-jobs/{job_id}/run", response_model=DataResponse[CollectionJobRead], status_code=202)
async def run_collection_job(job_id: int, db: AsyncSession = Depends(get_db)):
    """Trigger an immediate collection run."""
    from fastapi import HTTPException
    from celery_app import AlreadyRunningError
    try:
        item = await data_service.trigger_collection(db, job_id)
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
