"""Unit tests for Foreign Company and Foreign Investment Trust PDF table extraction (Issue #2)."""

from __future__ import annotations

import datetime
from pathlib import Path
import pytest
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from database import Base
from data.extractors.pdf import PdfTableExtractor
from data.extractors.pdf.models import ExtractedPdfTable
from data.models import PdfDocumentRecord, PdfExtractedTableRecord
from data.pdf_data_service import (
    detect_table_title,
    upsert_pdf_document,
    save_extracted_tables,
    FOREIGN_DOC_TYPES,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


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


def test_foreign_doc_types_registered():
    """Verify standard foreign doc type codes are mapped."""
    assert "120" in FOREIGN_DOC_TYPES
    assert "160" in FOREIGN_DOC_TYPES
    assert "180" in FOREIGN_DOC_TYPES
    assert "030" in FOREIGN_DOC_TYPES


def test_detect_table_title_foreign_keywords():
    """Verify heuristic title detection for foreign financial and fund statements."""
    # 1. Balance Sheet
    df_bs = pd.DataFrame({"Consolidated Balance Sheets": ["Current assets", "Total assets"], "2024": [100, 200]})
    t_bs = ExtractedPdfTable(page_number=58, table_index=0, bbox=(0, 0, 1, 1), df=df_bs, html="", row_count=2, col_count=2)
    assert detect_table_title(t_bs) == "貸借対照表 (Balance Sheet)"

    # 2. Statement of Operations (Income Statement)
    df_is = pd.DataFrame({"Consolidated Statements of Operations": ["Total revenue", "Net income"], "2024": [500, 50]})
    t_is = ExtractedPdfTable(page_number=59, table_index=0, bbox=(0, 0, 1, 1), df=df_is, html="", row_count=2, col_count=2)
    assert detect_table_title(t_is) == "損益計算書 (Statement of Operations)"

    # 3. Schedule of Investments / Portfolio
    df_port = pd.DataFrame({"銘柄": ["アドバンテスト", "東京エレクトロン"], "時価合計": [1000, 2000], "投資比率": [10.2, 8.2]})
    t_port = ExtractedPdfTable(page_number=35, table_index=1, bbox=(0, 0, 1, 1), df=df_port, html="", row_count=2, col_count=3)
    assert detect_table_title(t_port) == "組入有価証券・資産構成表 (Schedule of Investments)"


def test_foreign_cisco_fs_extraction_if_sample_exists():
    """Verify extraction of Balance Sheet & Income Statement from Cisco 10-K slice."""
    sample_path = BACKEND_DIR / "artifacts" / "pdf_storage" / "foreign" / "2026" / "cisco_fs_sliced.pdf"
    if not sample_path.exists():
        pytest.skip("cisco_fs_sliced.pdf not present, skipping physical PDF extraction check")

    extractor = PdfTableExtractor()
    tables = extractor.extract_tables(sample_path, page_numbers=[1, 2])
    assert len(tables) >= 2, f"Expected at least 2 financial tables, got {len(tables)}"

    bs_table = tables[0]
    is_table = tables[1]

    # Verify Balance Sheet table
    assert bs_table.row_count >= 8
    assert bs_table.col_count >= 3
    bs_text = bs_table.df.to_string()
    assert "ASSETS" in bs_text or "assets" in bs_text.lower()

    # Verify Operations table
    assert is_table.row_count >= 8
    assert is_table.col_count >= 3
    is_text = is_table.df.to_string()
    assert "revenue" in is_text.lower() or "cost of sales" in is_text.lower()


def test_save_foreign_tables_in_db(test_db_session):
    """Verify saving foreign company document and extracted tables into DB."""
    doc = upsert_pdf_document(
        session=test_db_session,
        doc_id="S100ZI6C",
        filer_name="シスコ・システムズ・インク (Cisco Systems, Inc.)",
        doc_type="外国会社報告書",
        doc_type_code="120",
        category="foreign",
        filing_date=datetime.date(2026, 11, 17),
        file_path="foreign/2026/S100ZI6C_10k.pdf",
        file_size=2500487,
        num_pages=285,
        extraction_status="completed",
    )
    test_db_session.commit()

    df_bs = pd.DataFrame({
        "Item": ["Cash and cash equivalents", "Total assets", "Total liabilities"],
        "2024": ["$8,346", "$122,291", "$75,448"],
    })
    t_bs = ExtractedPdfTable(
        page_number=58,
        table_index=0,
        bbox=(41.5, 100.0, 526.0, 750.0),
        df=df_bs,
        html='<table border="1"><tr><th>Item</th><th>2024</th></tr></table>',
        row_count=3,
        col_count=2,
    )

    saved = save_extracted_tables(
        session=test_db_session,
        document_id=doc.id,
        doc_id="S100ZI6C",
        tables=[t_bs],
    )
    test_db_session.commit()

    assert len(saved) == 1
    assert saved[0].table_title == "貸借対照表 (Balance Sheet)"
    assert saved[0].row_count == 3
    assert saved[0].col_count == 2
    assert saved[0].data_json[0]["Item"] == "Cash and cash equivalents"
