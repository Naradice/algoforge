"""Target assessments: queue model_core.analysis.assess_target on a dataset and read the result.

The analysis takes ~2 minutes on a million rows, so it runs as a Celery job (`assess_target`, on the
training queue like validate_model) and is polled / pushed via webhook (`assessment.completed` /
`assessment.error`), never inside a request.
"""
from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from model.models import TargetAssessment, TargetAssessmentCreate

MAX_HORIZON = 240
MAX_OBS = 240
MAX_ROWS = 2_000_000
ALLOWED_MODELS = ("hgb", "mlp", "knn")


def _validate(body: TargetAssessmentCreate) -> None:
    from model_core.analysis.targets import TARGETS

    problems = []
    if body.target not in TARGETS:
        problems.append(f"target must be one of {sorted(TARGETS)}")
    if not 1 <= body.horizon <= MAX_HORIZON:
        problems.append(f"horizon must be in 1..{MAX_HORIZON}")
    if not 5 <= body.obs <= MAX_OBS:
        problems.append(f"obs must be in 5..{MAX_OBS}")
    if not 1_000 <= body.max_rows <= MAX_ROWS:
        problems.append(f"max_rows must be in 1000..{MAX_ROWS}")
    bad = set(body.models) - set(ALLOWED_MODELS)
    if bad or not body.models:
        problems.append(f"models must be a non-empty subset of {list(ALLOWED_MODELS)}")
    if problems:
        raise HTTPException(status_code=422, detail={"code": "INVALID_ASSESSMENT", "message": "; ".join(problems)})


def to_dict(a: TargetAssessment) -> dict:
    return {
        "id": a.id, "dataset_id": a.dataset_id, "params": a.params, "status": a.status,
        "verdict": a.verdict, "reason": (a.result or {}).get("reason"), "result": a.result,
        "error_message": a.error_message,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "started_at": a.started_at.isoformat() if a.started_at else None,
        "ended_at": a.ended_at.isoformat() if a.ended_at else None,
    }


async def create_assessment(db: AsyncSession, body: TargetAssessmentCreate) -> TargetAssessment:
    from data.models import Dataset

    _validate(body)
    ds = (await db.execute(select(Dataset).where(Dataset.id == body.dataset_id))).scalar_one_or_none()
    if ds is None or not ds.artifact_path:
        raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND",
                                                     "message": f"dataset {body.dataset_id} not found or has no artifact"})
    a = TargetAssessment(dataset_id=body.dataset_id, params=body.model_dump(exclude={"dataset_id"}))
    db.add(a)
    await db.flush()
    await db.refresh(a)
    return a


async def get_assessment(db: AsyncSession, assessment_id: int) -> TargetAssessment:
    a = (await db.execute(select(TargetAssessment).where(TargetAssessment.id == assessment_id))).scalar_one_or_none()
    if a is None:
        raise HTTPException(status_code=404, detail={"code": "ASSESSMENT_NOT_FOUND",
                                                     "message": f"target assessment {assessment_id} not found"})
    return a
