"""Unit tests for PDF Data Service (Storage, ORM models, persistence, title detection)."""

from __future__ import annotations

import datetime
from pathlib import Path
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from database import Base
from data.extractors.pdf.models import ExtractedPdfTable
from data.models import PdfDocumentRecord, PdfExtractedTableRecord
from data.pdf_data_service import (
    save_pdf_file,
    resolve_pdf_path,
    upsert_pdf_document,
    save_extracted_tables,
    detect_table_title,
    ingest_pdf_document,
)
import pandas as pd


@pytest.fixture
def test_db_session():
    """In-memory SQLite session with Base tables created."""
    engine = sa.create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_save_pdf_file_and_resolve(tmp_path, monkeypatch):
    """Test saving PDF file to disk and path resolution."""
    import data.pdf_data_service as pds
    monkeypatch.setattr(pds, "PDF_STORAGE_ROOT", tmp_path)

    sample_content = b"%PDF-1.4 test content for pdf data service"
    doc_id = "TEST_DOC_001"
    filing_date = datetime.date(2026, 11, 23)

    rel_path, sha256, size, abs_path = save_pdf_file(
        doc_id=doc_id,
        category="prospectus",
        filing_date=filing_date,
        content=sample_content,
    )

    assert rel_path == "prospectus/2026/TEST_DOC_001.pdf"
    assert size == len(sample_content)
    assert abs_path.exists()
    assert abs_path.read_bytes() == sample_content

    # Check resolve_pdf_path
    resolved = resolve_pdf_path(rel_path)
    assert resolved == abs_path


def test_upsert_pdf_document(test_db_session):
    """Test creating and updating PdfDocumentRecord in database."""
    doc = upsert_pdf_document(
        session=test_db_session,
        doc_id="S100TEST",
        sec_code="12340",
        filer_name="テスト株式会社",
        doc_type="有価証券届出書",
        doc_type_code="010",
        category="prospectus",
        filing_date=datetime.date(2026, 11, 20),
        file_path="prospectus/2026/S100TEST.pdf",
        file_size=1024,
        sha256="abc123hash",
        num_pages=15,
        extraction_status="pending",
    )
    test_db_session.commit()

    assert doc.id is not None
    assert doc.doc_id == "S100TEST"
    assert doc.sec_code == "12340"
    assert doc.extraction_status == "pending"

    # Update document status to completed
    updated = upsert_pdf_document(
        session=test_db_session,
        doc_id="S100TEST",
        extraction_status="completed",
        extracted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    test_db_session.commit()

    assert updated.id == doc.id
    assert updated.extraction_status == "completed"
    assert updated.extracted_at is not None


def test_save_extracted_tables(test_db_session):
    """Test persisting ExtractedPdfTable objects into DB."""
    doc = upsert_pdf_document(
        session=test_db_session,
        doc_id="S100TABLE",
        filer_name="テーブルテスト株式会社",
        category="prospectus",
    )
    test_db_session.commit()

    # Create dummy ExtractedPdfTable
    df1 = pd.DataFrame({"項目": ["発行数", "払込金額"], "内容": ["10,000株", "1,000円"]})
    t1 = ExtractedPdfTable(
        page_number=2,
        table_index=0,
        bbox=(50.0, 100.0, 500.0, 200.0),
        df=df1,
        html="<table><tr><td>発行数</td><td>10,000株</td></tr></table>",
        row_count=2,
        col_count=2,
    )

    df2 = pd.DataFrame({"資金使途": ["設備投資", "運転資金"], "金額": ["100百万円", "50百万円"]})
    t2 = ExtractedPdfTable(
        page_number=16,
        table_index=0,
        bbox=(50.0, 300.0, 500.0, 450.0),
        df=df2,
        html="<table><tr><td>設備投資</td><td>100百万円</td></tr></table>",
        row_count=2,
        col_count=2,
    )

    records = save_extracted_tables(
        session=test_db_session,
        document_id=doc.id,
        doc_id="S100TABLE",
        tables=[t1, t2],
    )
    test_db_session.commit()

    assert len(records) == 2
    assert records[0].page_number == 2
    assert records[0].table_title == "募集・売出の条件"
    assert records[0].row_count == 2
    assert records[0].col_count == 2
    assert records[0].headers_json == ["項目", "内容"]

    assert records[1].page_number == 16
    assert records[1].table_title == "手取金の使途"


def test_detect_table_title():
    """Test heuristic title detection based on content."""
    df_use = pd.DataFrame({"資金使途の内訳": ["R&D"], "支出予定時期": ["2027年"]})
    t_use = ExtractedPdfTable(
        page_number=16, table_index=0, bbox=(0, 0, 1, 1), df=df_use, html="", row_count=1, col_count=2
    )
    assert detect_table_title(t_use) == "手取金の使途"

    df_other = pd.DataFrame({"ColA": [1], "ColB": [2]})
    t_other = ExtractedPdfTable(
        page_number=5, table_index=2, bbox=(0, 0, 1, 1), df=df_other, html="", row_count=1, col_count=2
    )
    assert detect_table_title(t_other) == "表 (P.5 #3)"
