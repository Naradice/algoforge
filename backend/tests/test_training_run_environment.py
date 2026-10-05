"""celery_worker._resolve_training_context records run_environment when a run starts, and the
REST read schema exposes it."""
from __future__ import annotations

from sqlalchemy import select

from data.models import Dataset
from model.models import MLModel, TrainingRun


async def _make_run(db_session) -> tuple[int, int]:
    async with db_session() as db:
        model = MLModel(name="env model", architecture="decoder_only", config={})
        ds = Dataset(name="env ds", artifact_path="datasets/x.parquet", status="ready")
        db.add_all([model, ds])
        await db.flush()
        run = TrainingRun(model_id=model.id, dataset_id=ds.id, hyperparams={"seed": 1}, status="pending")
        db.add(run)
        await db.commit()
        return model.id, run.id


async def test_run_start_records_run_environment(db_session, client, monkeypatch):
    import celery_worker
    from model import run_environment

    fake = {"git_commit": "f" * 40, "git_dirty": False, "python": "3.11.9", "packages": ["torch==2.5.1"]}
    monkeypatch.setattr(run_environment, "capture", lambda: fake)

    model_id, run_id = await _make_run(db_session)
    await celery_worker._resolve_training_context(db_session, run_id)

    async with db_session() as db:
        run = (await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))).scalar_one()
    assert run.status == "running"
    assert run.run_environment == fake
    assert run.hyperparams == {"seed": 1}  # the fingerprint lives in its own column, not hyperparams

    r = await client.get(f"/api/v1/models/{model_id}/training-runs")
    assert r.status_code == 200, r.text
    row = next(x for x in r.json()["data"] if x["id"] == run_id)
    assert row["run_environment"]["git_commit"] == "f" * 40


async def test_runs_without_a_fingerprint_read_as_null(db_session, client):
    model_id, run_id = await _make_run(db_session)
    r = await client.get(f"/api/v1/models/{model_id}/training-runs")
    row = next(x for x in r.json()["data"] if x["id"] == run_id)
    assert row["run_environment"] is None
