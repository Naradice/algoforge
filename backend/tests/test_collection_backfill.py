"""Tests for backfill and initialization features in web_report and collection triggers."""
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from data.collectors.web_report import (
    _expand_backfill_dates,
    _extract_date_from_url,
    _render_date_template,
    _resolve_filename,
)
from data.models import CollectionJob, Datasource
from data.service import data_service


def test_extract_date_from_url():
    # 8-digit YYYYMMDD
    assert _extract_date_from_url("https://min-fx.jp/market/report/20261106/") == date(2026, 11, 6)
    assert _extract_date_from_url("https://www.sonyfg.co.jp/ja/market_report/daily/20261105.html") == date(2026, 11, 5)
    assert _extract_date_from_url("/pdf/2026-10-31_report.pdf") == date(2026, 10, 31)

    # 6-digit YYMMDD
    assert _extract_date_from_url("https://example.com/reports/261106_daily.pdf") == date(2026, 11, 6)

    # No date in url
    assert _extract_date_from_url("https://example.com/static/report.pdf") is None


def test_resolve_filename_with_url_date():
    # Resolves {YYMMDD} and {YYYYMMDD} from URL date rather than today's date
    url = "https://min-fx.jp/market/report/20261104/"
    fn = _resolve_filename("daily_report_{YYMMDD}.pdf", url, "pdf")
    assert fn == "daily_report_261104.pdf"

    fn_long = _resolve_filename("d_{YYYYMMDD}.pdf", url, "pdf")
    assert fn_long == "d_20261104.pdf"

    # Preserves extension
    fn_no_ext = _resolve_filename("daily_report_{YYMMDD}", url, "pdf")
    assert fn_no_ext == "daily_report_261104.pdf"


def test_expand_backfill_dates():
    # 5 business days: none should be Saturday (5) or Sunday (6)
    dates = _expand_backfill_dates(5, weekdays_only=True)
    assert len(dates) == 5
    for d in dates:
        assert d.weekday() < 5
    # Should be strictly descending
    for i in range(len(dates) - 1):
        assert dates[i] > dates[i + 1]


def test_render_date_template():
    d = date(2026, 11, 6)
    tpl = "https://example.com/daily/{YYYYMMDD}/d_{YYMMDD}.pdf"
    rendered = _render_date_template(tpl, d)
    assert rendered == "https://example.com/daily/20261106/d_261106.pdf"


@pytest.mark.asyncio
async def test_trigger_datasource_collection_passes_overrides(db_session):
    async with db_session() as db:
        ds = Datasource(name="Test Web Source", type="web_report", config={"url": "https://example.com"})
        db.add(ds)
        await db.commit()
        await db.refresh(ds)

        job = CollectionJob(datasource_id=ds.id)
        db.add(job)
        await db.commit()
        await db.refresh(job)

        with patch("celery_app.enqueue", new_callable=AsyncMock) as mock_enqueue:
            # Trigger with overrides
            await data_service.trigger_datasource_collection(
                db, ds.id, overrides={"is_initial": True, "backfill_days": 10}
            )
            mock_enqueue.assert_called_once_with(
                "run_collection_job", job.id, {"is_initial": True, "backfill_days": 10}
            )


@pytest.mark.asyncio
async def test_run_collection_job_defaults_initial_when_never_run(db_session):
    from celery_worker import _run_collection_job

    async with db_session() as db:
        ds = Datasource(name="Web DS", type="web_report", config={"url": "https://example.com/{YYYYMMDD}.pdf"})
        db.add(ds)
        await db.commit()
        await db.refresh(ds)

        job = CollectionJob(datasource_id=ds.id, last_run_at=None)
        db.add(job)
        await db.commit()
        await db.refresh(job)
        job_id = job.id

    captured_config = {}

    def _fake_run_collector(dtype, did, config, incremental):
        nonlocal captured_config
        captured_config = dict(config)
        from data.collectors.web_report import CollectResult
        now = datetime.now(timezone.utc)
        return CollectResult(artifact_path="web_reports/test", row_count=1, from_ts=now, to_ts=now)

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()

    with patch("celery_worker._make_db", return_value=(db_session, mock_engine)), \
         patch("celery_worker._run_collector", side_effect=_fake_run_collector), \
         patch("data.scheduler.CollectionHeartbeat.start", return_value=MagicMock()), \
         patch("celery_worker.dispatch", new_callable=AsyncMock):

        await _run_collection_job(job_id)

    # Since job.last_run_at was None, is_initial should automatically have been set to True
    assert captured_config.get("is_initial") is True


def test_polite_delay_behavior():
    from unittest.mock import patch
    from data.collectors.web_report import _polite_delay
    import os

    # In pytest without FORCE_POLITE_DELAY, should return immediately without sleeping
    with patch('time.sleep') as mock_sleep:
        _polite_delay(2.0)
        mock_sleep.assert_not_called()

    # With FORCE_POLITE_DELAY=1, it sleeps with jitter
    with patch.dict(os.environ, {'FORCE_POLITE_DELAY': '1'}), patch('time.sleep') as mock_sleep:
        _polite_delay(2.0)
        mock_sleep.assert_called_once()
        called_delay = mock_sleep.call_args[0][0]
        # Should be between 2.0 - 20% (1.6) and 2.0 + 20% (2.4)
        assert 1.59 <= called_delay <= 2.41


@pytest.mark.asyncio
async def test_resolve_dataset_artifact_with_web_reports_store(db_session, tmp_path):
    import os
    from unittest.mock import patch
    from data.models import Dataset

    fake_external_store = tmp_path / "finance_data"
    fake_external_store.mkdir()
    sub_dir = fake_external_store / "SonyFinance"
    sub_dir.mkdir()
    sample_file = sub_dir / "d_20261106.pdf"
    sample_file.write_bytes(b"%PDF-1.4 sample")

    async with db_session() as db:
        ds = Dataset(
            name="Sony Daily",
            artifact_path="web_reports/SonyFinance",
            status="ready",
            row_count=1,
        )
        db.add(ds)
        await db.commit()
        await db.refresh(ds)
        ds_id = ds.id

    with patch.dict(os.environ, {"WEB_REPORTS_STORE_PATH": str(fake_external_store)}):
        async with db_session() as db:
            resolved_ds, resolved_path = await data_service.resolve_dataset_artifact(db, ds_id)
            assert resolved_path == sub_dir
            assert (resolved_path / "d_20261106.pdf").exists()
