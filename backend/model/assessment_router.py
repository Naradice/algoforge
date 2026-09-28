"""/target-assessments -- see model/assessment_service.py."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from celery_app import enqueue
from database import get_db
from model import assessment_service
from model.models import TargetAssessmentCreate
from schemas import DataResponse

assessment_router = APIRouter(prefix="/target-assessments", tags=["target-assessments"])


@assessment_router.post("", status_code=202)
async def create_target_assessment(body: TargetAssessmentCreate, db: AsyncSession = Depends(get_db)):
    a = await assessment_service.create_assessment(db, body)
    await db.commit()
    await enqueue("assess_target", a.id)
    return DataResponse(data=assessment_service.to_dict(a))


@assessment_router.get("/{assessment_id}")
async def get_target_assessment(assessment_id: int, db: AsyncSession = Depends(get_db)):
    return DataResponse(data=assessment_service.to_dict(await assessment_service.get_assessment(db, assessment_id)))
