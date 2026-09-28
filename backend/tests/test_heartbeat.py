"""R-13: training-run heartbeat, stale-run reaper, and force stop (model/heartbeat.py)."""
import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from model.models import MLModel, TrainingRun


@pytest.fixture(autouse=True)
def _no_redis_locks(monkeypatch):
    # _clear_run_locks talks to the real Redis; test run ids could collide with real runs' keys.
    import model.heartbeat as hb
    monkeypatch.setattr(hb, "_clear_run_locks", lambda run_id: None)


def _ago(**kw):
    return datetime.now(timezone.utc) - timedelta(**kw)


async def _make_runs(db_session, specs: dict) -> dict:
    async with db_session() as db:
        model = MLModel(name="hb model", architecture="decoder_only")
        db.add(model)
        await db.flush()
        runs = {}
        for name, (status, hb, started) in specs.items():
            run = TrainingRun(model_id=model.id, dataset_id=1, status=status,
                              last_heartbeat_at=hb, started_at=started)
            db.add(run)
            await db.flush()
            runs[name] = run.id
        await db.commit()
    return runs


async def _get(db_session, run_id):
    async with db_session() as db:
        return (await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))).scalar_one()


async def test_reaper_only_reaps_running_runs_with_a_stale_heartbeat(db_session):
    from model.heartbeat import reap_stale_training_runs

    runs = await _make_runs(db_session, {
        "stale": ("running", _ago(minutes=30), _ago(hours=1)),
        "fresh": ("running", _ago(minutes=1), _ago(hours=1)),
        "legacy": ("running", None, _ago(days=3)),          # no heartbeat: never judged
        "done": ("completed", _ago(minutes=30), _ago(hours=1)),
    })
    async with db_session() as db:
        reaped = await reap_stale_training_runs(db, stale_after=600)
        await db.commit()
    assert runs["stale"] in reaped
    assert not {runs["fresh"], runs["legacy"], runs["done"]} & set(reaped)

    stale = await _get(db_session, runs["stale"])
    assert stale.status == "error" and stale.ended_at is not None
    assert stale.error_message.startswith("worker lost")
    for name in ("fresh", "legacy"):
        assert (await _get(db_session, runs[name])).status == "running"
    assert (await _get(db_session, runs["done"])).status == "completed"

    async with db_session() as db:   # idempotent: a second pass finds nothing new
        assert runs["stale"] not in await reap_stale_training_runs(db, stale_after=600)


async def test_status_reports_heartbeat_staleness(db_session):
    from model.service import model_service

    # started_at left NULL: get_training_progress's pre-existing elapsed-time math subtracts it
    # from an aware now(), which SQLite's naive timestamps can't do (Postgres returns aware ones)
    runs = await _make_runs(db_session, {
        "stale": ("running", _ago(minutes=30), None),
        "fresh": ("running", _ago(seconds=5), None),
        "legacy": ("running", None, None),
    })
    async with db_session() as db:
        stale = await model_service.get_training_progress(db, runs["stale"])
        fresh = await model_service.get_training_progress(db, runs["fresh"])
        legacy = await model_service.get_training_progress(db, runs["legacy"])
    assert stale["heartbeat_stale"] is True and stale["last_heartbeat_at"]
    assert fresh["heartbeat_stale"] is False
    assert legacy["heartbeat_stale"] is None and legacy["last_heartbeat_at"] is None


async def test_force_stop_only_ends_worker_lost_runs(client, db_session):
    runs = await _make_runs(db_session, {
        "fresh": ("running", _ago(seconds=5), _ago(hours=1)),
        "stale": ("running", _ago(minutes=30), _ago(hours=1)),
        "legacy_old": ("running", None, _ago(hours=2)),
        "legacy_new": ("running", None, _ago(seconds=30)),
    })
    r = await client.post(f"/api/v1/training-runs/{runs['fresh']}/stop?force=true")
    assert r.status_code == 409
    r = await client.post(f"/api/v1/training-runs/{runs['legacy_new']}/stop?force=true")
    assert r.status_code == 409

    for name in ("stale", "legacy_old"):
        r = await client.post(f"/api/v1/training-runs/{runs[name]}/stop?force=true")
        assert r.status_code == 200, r.text
        body = r.json()["data"]
        assert body["status"] == "error"
        assert body["error_message"].startswith("force-stopped: worker lost")

    # graceful stop is unchanged: only sets the flag
    r = await client.post(f"/api/v1/training-runs/{runs['fresh']}/stop")
    assert r.status_code == 200
    assert r.json()["data"] == {**r.json()["data"], "status": "running", "stop_requested": True}


async def test_heartbeat_thread_writes_until_stopped(db_session):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from model.heartbeat import HeartbeatThread
    from tests.conftest import TEST_DB_URL

    runs = await _make_runs(db_session, {"live": ("running", None, _ago(seconds=1))})

    def make_db():
        # NullPool, like celery_worker._make_db: a pooled aiosqlite connection left open here
        # keeps a non-daemon aiosqlite thread alive and hangs interpreter exit after the session
        from sqlalchemy.pool import NullPool
        eng = create_async_engine(TEST_DB_URL, poolclass=NullPool, connect_args={"check_same_thread": False})
        return async_sessionmaker(eng, expire_on_commit=False), eng

    beat = HeartbeatThread(runs["live"], make_db, interval=0.05).start()
    try:
        deadline = time.monotonic() + 10
        first = None
        while time.monotonic() < deadline:
            first = (await _get(db_session, runs["live"])).last_heartbeat_at
            if first is not None:
                break
            await asyncio.sleep(0.05)
        assert first is not None
        await asyncio.sleep(0.3)
        second = (await _get(db_session, runs["live"])).last_heartbeat_at
        assert second > first
    finally:
        beat.stop()
    assert not beat._thread.is_alive()


async def test_batch_status_reports_each_run_and_unknown_ids(client, db_session):
    runs = await _make_runs(db_session, {
        "stale": ("running", _ago(minutes=30), None),
        "fresh": ("running", _ago(seconds=5), None),
        "done": ("completed", _ago(minutes=30), None),
    })
    ids = [runs["stale"], runs["fresh"], runs["done"], 999999]
    r = await client.get(f"/api/v1/training-runs/status?run_ids={','.join(map(str, ids))}")
    assert r.status_code == 200
    by_id = {row["run_id"]: row for row in r.json()["data"]}
    assert [row["run_id"] for row in r.json()["data"]] == ids          # request order kept
    assert by_id[runs["stale"]]["heartbeat_stale"] is True and not by_id[runs["stale"]]["terminal"]
    assert by_id[runs["fresh"]]["heartbeat_stale"] is False
    assert by_id[runs["done"]]["terminal"] and by_id[runs["done"]]["heartbeat_stale"] is None
    assert by_id[999999] == {"run_id": 999999, "status": "not_found", "terminal": True}
