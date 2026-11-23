"""Integration and route tests for PDF document and table extraction endpoints."""

from __future__ import annotations

import datetime
from pathlib import Path
import pytest
from httpx import AsyncClient

from data.models import PdfDocumentRecord, PdfExtractedTableRecord


@pytest.mark.asyncio
async def test_pdf_document_and_table_endpoints(client: AsyncClient, db_session, tmp_path, monkeypatch):
    """Test GET /api/v1/pdf/documents, /api/v1/pdf/documents/{doc_id}, and /api/v1/pdf/tables."""
    import data.pdf_data_service as pds
    monkeypatch.setattr(pds, "PDF_STORAGE_ROOT", tmp_path)

    # Create dummy PDF file on disk
    dummy_pdf = tmp_path / "prospectus" / "2026" / "S100TESTDOC.pdf"
    dummy_pdf.parent.mkdir(parents=True, exist_ok=True)
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy prospectus content")

    async with db_session() as session:
        doc = PdfDocumentRecord(
            doc_id="S100TESTDOC",
            sec_code="48820",
            filer_name="テスト製薬株式会社",
            doc_type="有価証券届出書",
            doc_type_code="010",
            category="prospectus",
            filing_date=datetime.date(2026, 11, 20),
            file_path="prospectus/2026/S100TESTDOC.pdf",
            file_size=len(dummy_pdf.read_bytes()),
            sha256="abc123456789",
            num_pages=30,
            extraction_status="completed",
        )
        session.add(doc)
        await session.flush()

        # Add tables
        t1 = PdfExtractedTableRecord(
            document_id=doc.id,
            doc_id="S100TESTDOC",
            page_number=2,
            table_index=0,
            table_title="募集・売出の条件",
            row_count=13,
            col_count=2,
            headers_json=["項目", "内容"],
            data_json=[{"項目": "発行数", "内容": "46,000個"}],
            html='<table class="pdf-extracted-table"><tr><th>項目</th><th>内容</th></tr></table>',
        )
        t2 = PdfExtractedTableRecord(
            document_id=doc.id,
            doc_id="S100TESTDOC",
            page_number=16,
            table_index=0,
            table_title="手取金の使途",
            row_count=2,
            col_count=3,
            headers_json=["払込金額", "諸費用", "差引手取概算額"],
            data_json=[{"払込金額": "500,000,000円", "諸費用": "10,000,000円", "差引手取概算額": "490,000,000円"}],
            html='<table class="pdf-extracted-table"><tr><th>払込金額</th></tr></table>',
        )
        session.add(t1)
        session.add(t2)
        await session.commit()

    # 1. Test GET /api/v1/pdf/documents
    resp = await client.get("/api/v1/pdf/documents?category=prospectus")
    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["total"] >= 1
    docs = body["data"]
    match = [d for d in docs if d["doc_id"] == "S100TESTDOC"]
    assert len(match) == 1
    assert match[0]["filer_name"] == "テスト製薬株式会社"
    assert match[0]["tables_count"] == 2

    # 2. Test GET /api/v1/pdf/documents/{doc_id}
    resp = await client.get("/api/v1/pdf/documents/S100TESTDOC")
    assert resp.status_code == 200
    body = resp.json()
    detail = body["data"]
    assert detail["doc_id"] == "S100TESTDOC"
    assert len(detail["tables"]) == 2
    titles = [t["table_title"] for t in detail["tables"]]
    assert "募集・売出の条件" in titles
    assert "手取金の使途" in titles

    # 3. Test GET /api/v1/pdf/documents/{doc_id}/file
    resp = await client.get("/api/v1/pdf/documents/S100TESTDOC/file")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content == b"%PDF-1.4 dummy prospectus content"

    # 4. Test GET /api/v1/pdf/tables
    resp = await client.get("/api/v1/pdf/tables?doc_id=S100TESTDOC&title=使途")
    assert resp.status_code == 200
    body = resp.json()
    tables = body["data"]
    assert len(tables) == 1
    assert tables[0]["table_title"] == "手取金の使途"
    assert tables[0]["headers"] == ["払込金額", "諸費用", "差引手取概算額"]


@pytest.mark.asyncio
async def test_pdf_document_not_found(client: AsyncClient):
    """Test 404 behavior for non-existent document."""
    resp = await client.get("/api/v1/pdf/documents/NON_EXISTENT_DOC")
    assert resp.status_code == 404
