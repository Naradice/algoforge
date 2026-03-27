"""Data Management layer — HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from data.models import (
    DatasourceCreate, DatasourceUpdate, DatasourceRead,
    DatasetRead, CollectionJobCreate, CollectionJobRead,
    DataCharacteristicsRead,
)
from data.service import data_service

router = APIRouter(prefix="/data", tags=["data"])


# ── Datasources ────────────────────────────────────────────────────────────────

@router.get("/datasources", response_model=list[DatasourceRead])
async def list_datasources(db: AsyncSession = Depends(get_db)):
    return await data_service.list_datasources(db)


@router.post("/datasources", response_model=DatasourceRead, status_code=201)
async def create_datasource(body: DatasourceCreate, db: AsyncSession = Depends(get_db)):
    return await data_service.create_datasource(db, body)


@router.get("/datasources/{datasource_id}", response_model=DatasourceRead)
async def get_datasource(datasource_id: int, db: AsyncSession = Depends(get_db)):
    return await data_service.get_datasource(db, datasource_id)


@router.patch("/datasources/{datasource_id}", response_model=DatasourceRead)
async def update_datasource(datasource_id: int, body: DatasourceUpdate, db: AsyncSession = Depends(get_db)):
    return await data_service.update_datasource(db, datasource_id, body)


@router.delete("/datasources/{datasource_id}", status_code=204)
async def delete_datasource(datasource_id: int, db: AsyncSession = Depends(get_db)):
    await data_service.delete_datasource(db, datasource_id)


# ── Datasets ───────────────────────────────────────────────────────────────────

@router.get("/datasets", response_model=list[DatasetRead])
async def list_datasets(symbol: str | None = None, timeframe: str | None = None, db: AsyncSession = Depends(get_db)):
    return await data_service.list_datasets(db, symbol=symbol, timeframe=timeframe)


@router.get("/datasets/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    return await data_service.get_dataset(db, dataset_id)


@router.get("/datasets/{dataset_id}/characteristics", response_model=DataCharacteristicsRead | None)
async def get_characteristics(dataset_id: int, db: AsyncSession = Depends(get_db)):
    return await data_service.get_characteristics(db, dataset_id)


# ── Collection Jobs ────────────────────────────────────────────────────────────

@router.get("/collection-jobs", response_model=list[CollectionJobRead])
async def list_collection_jobs(datasource_id: int | None = None, db: AsyncSession = Depends(get_db)):
    return await data_service.list_collection_jobs(db, datasource_id=datasource_id)


@router.post("/collection-jobs", response_model=CollectionJobRead, status_code=202)
async def create_collection_job(body: CollectionJobCreate, db: AsyncSession = Depends(get_db)):
    job = await data_service.create_collection_job(db, body)
    # TODO Phase 1: enqueue run_collection_job arq job
    return job


@router.get("/collection-jobs/{job_id}", response_model=CollectionJobRead)
async def get_collection_job(job_id: int, db: AsyncSession = Depends(get_db)):
    return await data_service.get_collection_job(db, job_id)
