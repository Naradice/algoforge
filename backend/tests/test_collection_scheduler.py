"""Tests for data/scheduler.py: next_run calculation, scheduling tick, heartbeat, and stale-job reaper."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from data.models import CollectionJob, Datasource
from data.scheduler import (
    compute_next_run,
    reap_stale_collection_jobs,
    retry_run_at,
    scheduler_tick,
)


def _ago(**kw):
    return datetime.now(timezone.utc) - timedelta(**kw)


def _future(**kw):
    return datetime.now(timezone.utc) + timedelta(**kw)


def test_compute_next_run_web_report():
    now = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    # interval_days = 1
    next_run = compute_next_run("web_report", {"interval_days": 1}, None, now=now)
    assert next_run == now + timedelta(days=1)

    # interval_days = 2 with download_time
    next_run_time = compute_next_run(
        "web_report",
        {"interval_days": 2, "download_time": "15:00"},
        None,
        now=now,
    )
    assert next_run_time == datetime(2026, 11, 7, 15, 0, 0, tzinfo=timezone.utc)

    # download_time already passed today -> step forward by interval_days
    next_run_passed = compute_next_run(
        "web_report",
        {"interval_days": 1, "download_time": "08:00"},
        None,
        now=now,
    )
    assert next_run_passed == datetime(2026, 11, 8, 8, 0, 0, tzinfo=timezone.utc)

    # one-off (no interval_days)
    assert compute_next_run("web_report", {}, None, now=now) is None


def test_compute_next_run_cron():
    now = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    # Cron at 12:00 every day
    next_run = compute_next_run("ohlc_download", {}, "0 12 * * *", now=now)
    assert next_run == datetime(2026, 11, 7, 12, 0, 0, tzinfo=timezone.utc)


def test_retry_run_at():
    now = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    # regular is 24h away -> retry in 30 minutes
    retry_time = retry_run_at("web_report", {"interval_days": 1}, None, now=now, retry_minutes=30)
    assert retry_time == now + timedelta(minutes=30)

    # one-off has no retry
    assert retry_run_at("web_report", {}, None, now=now) is None


async def _setup_datasource_and_jobs(db_session, job_specs: dict) -> dict:
    async with db_session() as db:
        source = Datasource(
            name="test_web_reports",
            type="web_report",
            config={"url": "https://example.com/reports", "interval_days": 1},
        )
        db.add(source)
        await db.flush()

        jobs = {}
        for name, spec in job_specs.items():
            job = CollectionJob(
                datasource_id=source.id,
                status=spec.get("status", "idle"),
                enabled=spec.get("enabled", True),
                last_run_at=spec.get("last_run_at"),
                next_run_at=spec.get("next_run_at"),
                last_heartbeat_at=spec.get("last_heartbeat_at"),
            )
            db.add(job)
            await db.flush()
            jobs[name] = job.id
        await db.commit()
    return jobs


@pytest.mark.asyncio
async def test_scheduler_tick_enqueues_due_jobs(db_session):
    jobs = await _setup_datasource_and_jobs(
        db_session,
        {
            "due": {"next_run_at": _ago(minutes=5), "last_run_at": _ago(days=1)},
            "future": {"next_run_at": _future(hours=2)},
            "running": {"status": "running", "next_run_at": _ago(minutes=10)},
            "new_unassigned": {"next_run_at": None, "last_run_at": None},
        },
    )

    enqueued_ids = []

    async def mock_enqueue(job_id: int):
        enqueued_ids.append(job_id)

    async def mock_lock_held(job_id: int):
        return False

    async with db_session() as db:
        result = await scheduler_tick(
            db,
            enqueue=mock_enqueue,
            lock_held=mock_lock_held,
        )
        await db.commit()

    # "due" and "new_unassigned" should be enqueued
    assert jobs["due"] in result
    assert jobs["new_unassigned"] in result
    assert jobs["future"] not in result
    assert jobs["running"] not in result
    assert set(enqueued_ids) == set(result)

    # Check updated next_run_at
    async with db_session() as db:
        due_job = (await db.execute(select(CollectionJob).where(CollectionJob.id == jobs["due"]))).scalar_one()
        # next_run_at should have been advanced into the future
        assert due_job.next_run_at is not None
        from data.scheduler import as_utc
        assert as_utc(due_job.next_run_at) > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_reaper_cleans_stale_collection_jobs(db_session):
    jobs = await _setup_datasource_and_jobs(
        db_session,
        {
            "stale_hb": {"status": "running", "last_heartbeat_at": _ago(minutes=30)},
            "fresh_hb": {"status": "running", "last_heartbeat_at": _ago(minutes=1)},
            "idle_with_old_hb": {"status": "idle", "last_heartbeat_at": _ago(hours=2)},
        },
    )

    released_locks = []

    def mock_release(job_id: int):
        released_locks.append(job_id)

    async def mock_lock_held(job_id: int):
        return False

    async with db_session() as db:
        reaped = await reap_stale_collection_jobs(
            db,
            stale_after=600,
            lock_held=mock_lock_held,
            release_lock=mock_release,
        )
        await db.commit()

    assert jobs["stale_hb"] in reaped
    assert jobs["fresh_hb"] not in reaped
    assert jobs["idle_with_old_hb"] not in reaped
    assert jobs["stale_hb"] in released_locks

    async with db_session() as db:
        stale_job = (await db.execute(select(CollectionJob).where(CollectionJob.id == jobs["stale_hb"]))).scalar_one()
        assert stale_job.status == "error"
        assert "worker lost" in (stale_job.last_error or "")
        # next_run_at should be scheduled for a retry
        assert stale_job.next_run_at is not None
