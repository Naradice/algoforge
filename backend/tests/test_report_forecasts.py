"""Unit and integration tests for report forecast extraction, autonomous on-demand fetching, and OHLCV enrichment."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from data.extractors.report_forecast import (
    extract_forecast_from_pdf,
    save_report_forecast,
)
from data.models import Datasource, ReportForecast
from data.service import data_service
from database import SessionLocal


def test_save_and_upsert_report_forecast():
    """Test saving and upserting forecast records into DB."""
    test_date = date(2026, 12, 1)
    fc_data = {
        "source": "テスト証券",
        "symbol": "USDJPY",
        "horizon": "Daily",
        "report_date": test_date,
        "analyst": "テストアナリスト",
        "direction": "Bullish",
        "expected_low": 150.00,
        "expected_high": 152.00,
        "title": "テスト見通し",
        "raw_artifact_path": "test.pdf",
    }

    with SessionLocal() as db:
        # 1. Insert
        rf = save_report_forecast(db, fc_data)
        db.commit()
        assert rf is not None
        assert rf.id is not None
        assert rf.expected_low == 150.00

        # 2. Upsert (update expected_high)
        fc_data["expected_high"] = 153.00
        rf_updated = save_report_forecast(db, fc_data)
        db.commit()
        assert rf_updated.id == rf.id
        assert rf_updated.expected_high == 153.00

        # Cleanup
        db.delete(rf_updated)
        db.commit()


@pytest.mark.asyncio
async def test_get_or_fetch_from_db_cache():
    """Test Tier-1 autonomous fetch: retrieves from DB cache directly."""
    test_date = date(2026, 12, 2)
    with SessionLocal() as db:
        rf = ReportForecast(
            source="テスト信託",
            symbol="USDJPY",
            horizon="Daily",
            report_date=test_date,
            analyst="テスト氏",
            expected_low=155.00,
            expected_high=157.00,
            direction="Neutral/Range",
            title="キャッシュテスト",
        )
        db.add(rf)
        db.commit()
        rf_id = rf.id

    try:
        res = await data_service.get_or_fetch_report_forecast(
            symbol="USDJPY",
            target_date=test_date,
            source="テスト信託",
            horizon="Daily",
        )
        assert res is not None
        assert res["id"] == rf_id
        assert res["expected_low"] == 155.00
        assert res["expected_high"] == 157.00
    finally:
        with SessionLocal() as db:
            obj = db.get(ReportForecast, rf_id)
            if obj:
                db.delete(obj)
                db.commit()


@pytest.mark.asyncio
async def test_get_or_fetch_disk_fallback(tmp_path: Path):
    """Test Tier-2 autonomous fetch: extracts from disk when missing in DB."""
    test_date = date(2026, 11, 20)
    subfolder = "SMBC"
    dest_dir = tmp_path / subfolder
    dest_dir.mkdir(parents=True)
    pdf_file = dest_dir / f"daily_{test_date.strftime('%Y%m%d')}.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy")

    mock_forecast = {
        "source": "SMBC信託",
        "symbol": "USDJPY",
        "horizon": "Daily",
        "report_date": test_date,
        "analyst": "祖父江",
        "expected_low": 154.50,
        "expected_high": 156.50,
        "direction": "Neutral/Range",
        "title": "オンデマンド抽出テスト",
        "raw_artifact_path": pdf_file.name,
    }

    with patch.dict(os.environ, {"WEB_REPORTS_STORE_PATH": str(tmp_path)}):
        with patch("data.extractors.report_forecast.extract_forecast_from_pdf", return_value=mock_forecast):
            res = await data_service.get_or_fetch_report_forecast(
                symbol="USDJPY",
                target_date=test_date,
                source="SMBC信託",
                horizon="Daily",
            )
            assert res is not None
            assert res["expected_low"] == 154.50
            assert res["expected_high"] == 156.50

            # Verify it was persisted in DB
            with SessionLocal() as db:
                stmt = sa.select(ReportForecast).where(
                    ReportForecast.source == "SMBC信託",
                    ReportForecast.report_date == test_date,
                )
                saved = db.execute(stmt).scalar_one_or_none()
                assert saved is not None
                db.delete(saved)
                db.commit()


def test_enrich_dataframe_with_forecasts():
    """Test merging forecast features into OHLCV price series."""
    d1 = date(2026, 12, 5)
    with SessionLocal() as db:
        rf = ReportForecast(
            source="SMBC信託",
            symbol="USDJPY",
            horizon="Daily",
            report_date=d1,
            analyst="祖父江",
            expected_low=155.00,
            expected_high=157.00,
            direction="Neutral/Range",
            title="特徴量テスト",
        )
        db.add(rf)
        db.commit()
        rf_id = rf.id

    try:
        df = pd.DataFrame(
            {
                "Open": [155.50, 156.00],
                "High": [156.50, 158.00],
                "Low": [155.20, 154.50],
                "Close": [156.20, 157.80],
            },
            index=pd.to_datetime(["2026-12-05", "2026-12-06"]),
        )

        enriched = data_service.enrich_dataframe_with_forecasts(
            df, symbol="USDJPY", horizon="Daily", source="SMBC信託"
        )

        assert "fc_low" in enriched.columns
        assert "fc_high" in enriched.columns
        assert "is_in_range" in enriched.columns
        assert "is_high_break" in enriched.columns
        assert "is_low_break" in enriched.columns

        row1 = enriched.loc["2026-12-05"]
        assert row1["fc_low"] == 155.00
        assert row1["fc_high"] == 157.00
        assert row1["is_in_range"] == True
        assert row1["is_high_break"] == False
    finally:
        with SessionLocal() as db:
            obj = db.get(ReportForecast, rf_id)
            if obj:
                db.delete(obj)
                db.commit()


def test_collector_on_file_saved_hook(tmp_path: Path):
    """Test collector hook runs without error on non-PDF or empty files."""
    from data.collectors.web_report import _on_file_saved

    non_pdf = tmp_path / "sample.html"
    non_pdf.write_text("dummy")
    _on_file_saved(non_pdf)
