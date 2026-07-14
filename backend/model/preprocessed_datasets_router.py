"""Preprocessed Datasets — named, reusable preprocessing recipes for training runs."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from pagination import Pagination
from schemas import DataResponse, Meta
from model.models import PreprocessedDatasetCreate, PreprocessedDatasetUpdate, PreprocessedDatasetRead
from model.service import model_service
from celery_app import AlreadyRunningError

pd_router = APIRouter(prefix="/preprocessed-datasets", tags=["preprocessed-datasets"])


@pd_router.get("", response_model=DataResponse[list[PreprocessedDatasetRead]])
async def list_preprocessed_datasets(
    dataset_id: int | None = None,
    pagination: Pagination = Depends(),
    db: AsyncSession = Depends(get_db),
):
    items, total = await model_service.list_preprocessed_datasets(
        db, dataset_id=dataset_id, offset=pagination.offset, limit=pagination.page_size
    )
    return DataResponse(data=items, meta=Meta(total=total, page=pagination.page, page_size=pagination.page_size))


@pd_router.post("", response_model=DataResponse[PreprocessedDatasetRead], status_code=202)
async def create_preprocessed_dataset(body: PreprocessedDatasetCreate, db: AsyncSession = Depends(get_db)):
    item = await model_service.create_preprocessed_dataset(db, body)
    return DataResponse(data=item)


@pd_router.get("/{preprocessed_dataset_id}", response_model=DataResponse[PreprocessedDatasetRead])
async def get_preprocessed_dataset(preprocessed_dataset_id: int, db: AsyncSession = Depends(get_db)):
    item = await model_service.get_preprocessed_dataset(db, preprocessed_dataset_id)
    return DataResponse(data=item)


@pd_router.patch("/{preprocessed_dataset_id}", response_model=DataResponse[PreprocessedDatasetRead])
async def update_preprocessed_dataset(preprocessed_dataset_id: int, body: PreprocessedDatasetUpdate, db: AsyncSession = Depends(get_db)):
    item = await model_service.update_preprocessed_dataset(db, preprocessed_dataset_id, body)
    return DataResponse(data=item)


@pd_router.delete("/{preprocessed_dataset_id}", status_code=204)
async def delete_preprocessed_dataset(preprocessed_dataset_id: int, db: AsyncSession = Depends(get_db)):
    await model_service.delete_preprocessed_dataset(db, preprocessed_dataset_id)


@pd_router.post("/{preprocessed_dataset_id}/characteristics/compute", response_model=DataResponse[PreprocessedDatasetRead], status_code=202)
async def compute_preprocessed_characteristics(preprocessed_dataset_id: int, db: AsyncSession = Depends(get_db)):
    try:
        item = await model_service.recompute_preprocessed_characteristics(db, preprocessed_dataset_id)
    except AlreadyRunningError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return DataResponse(data=item)
