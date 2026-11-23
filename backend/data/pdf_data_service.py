"""PDF Data Service: Storage, Database Persistence, and Automated Ingestion for EDINET PDFs.

Manages:
1. Storage of raw PDF documents (Category/Year/doc_id.pdf) with SHA-256 deduplication
2. Database persistence of PDF metadata (PdfDocumentRecord) and extracted tables (PdfExtractedTableRecord)
3. EDINET API PDF download (type=2) and table extraction via PdfTableExtractor
4. Batch collection of offering and prospectus documents (DocTypeCode: 010, 020, 040, 060, 080, 090)
"""

from __future__ import annotations

import datetime
from datetime import date as dt_date
import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any
import requests
import sqlalchemy as sa
from sqlalchemy.orm import Session

from database import SessionLocal
from data.extractors.pdf import PdfTableExtractor, ExtractedPdfTable
from data.models import PdfDocumentRecord, PdfExtractedTableRecord
from scripts.verify_edinet_html_parser import fetch_document_list, get_edinet_api_key

logger = logging.getLogger(__name__)

# Base storage path for PDF files
BACKEND_DIR = Path(__file__).resolve().parent.parent
PDF_STORAGE_ROOT = Path(
    os.getenv("PDF_STORAGE_PATH", str(BACKEND_DIR / "artifacts" / "pdf_storage"))
).resolve()

PROSPECTUS_DOC_TYPES = {
    "010": "有価証券届出書",
    "020": "訂正有価証券届出書",
    "030": "届出の取回書",
    "040": "発行登録書",
    "050": "訂正発行登録書",
    "060": "発行登録追補書類",
    "070": "発行登録取下書",
    "080": "目論見書",
    "090": "訂正目論見書",
}


def get_storage_root() -> Path:
    """Return and ensure the root directory for PDF storage."""
    PDF_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return PDF_STORAGE_ROOT


def _to_date(val: Any) -> dt_date | None:
    """Convert string YYYY-MM-DD or date/datetime object to datetime.date."""
    if val is None:
        return None
    if isinstance(val, dt_date):
        return val
    if isinstance(val, datetime.datetime):
        return val.date()
    try:
        parts = str(val).split("-")
        return dt_date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None


def save_pdf_file(
    doc_id: str,
    category: str,
    filing_date: dt_date | None,
    content: bytes,
) -> tuple[str, str, int, Path]:
    """
    Save raw PDF bytes to disk partitioned by category and year.
    Returns: (relative_path, sha256_hash, size_bytes, absolute_path)
    """
    root = get_storage_root()
    year = str(filing_date.year) if filing_date else datetime.datetime.now().strftime("%Y")
    rel_dir = Path(category) / year
    dest_dir = root / rel_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / f"{doc_id}.pdf"
    dest_file.write_bytes(content)

    sha256 = hashlib.sha256(content).hexdigest()
    size = len(content)
    rel_path = str(rel_dir / f"{doc_id}.pdf").replace("\\", "/")

    return rel_path, sha256, size, dest_file


def resolve_pdf_path(rel_or_abs_path: str | None) -> Path | None:
    """Resolve a relative storage path or absolute path to a concrete Path."""
    if not rel_or_abs_path:
        return None
    p = Path(rel_or_abs_path)
    if p.is_absolute() and p.exists():
        return p
    root = get_storage_root()
    cand = (root / rel_or_abs_path).resolve()
    if cand.exists():
        return cand
    # Fallback to artifacts/pdf_samples if present
    sample_path = (BACKEND_DIR / "artifacts" / "pdf_samples" / rel_or_abs_path).resolve()
    if sample_path.exists():
        return sample_path
    return cand


def download_edinet_pdf(doc_id: str) -> bytes | None:
    """Download PDF binary directly from EDINET API (type=2)."""
    api_key = get_edinet_api_key()
    url = f"https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}?type=2"
    if api_key:
        url += f"&Subscription-Key={api_key}"

    logger.info("Downloading PDF for doc_id=%s from EDINET API...", doc_id)
    try:
        resp = requests.get(url, timeout=45)
        if resp.status_code == 200 and resp.content.startswith(b"%PDF"):
            logger.info("Successfully downloaded PDF for doc_id=%s (%d bytes)", doc_id, len(resp.content))
            return resp.content
        logger.warning(
            "EDINET API returned non-PDF or error for doc_id=%s: status=%d, prefix=%r",
            doc_id,
            resp.status_code,
            resp.content[:40],
        )
        return None
    except Exception as exc:
        logger.exception("Failed to download PDF for doc_id=%s: %s", doc_id, exc)
        return None


def detect_table_title(table: ExtractedPdfTable, pdf_path: Path | None = None) -> str | None:
    """Attempt to detect the table title from column headers or content."""
    # Check column header keywords if available
    headers_str = " ".join(str(c) for c in table.df.columns if c)
    data_str = " ".join(str(v) for v in table.df.values.flatten()[:20] if v)
    combined = f"{headers_str} {data_str}"

    if "手取金の使途" in combined or "手取概算額" in combined or "資金使途" in combined:
        return "手取金の使途"
    if "新株予約権" in combined or "募集の条件" in combined or "発行数" in combined:
        return "募集・売出の条件"
    if "大株主" in combined or "主要株主" in combined:
        return "大株主の状況"
    if "役員" in combined:
        return "役員の状況"
    if "社債" in combined:
        return "社債発行要項"
    return f"表 (P.{table.page_number} #{table.table_index + 1})"


def upsert_pdf_document(
    session: Session,
    doc_id: str,
    sec_code: str | None = None,
    filer_name: str | None = None,
    doc_type: str | None = None,
    doc_type_code: str | None = None,
    category: str = "prospectus",
    filing_date: dt_date | None = None,
    file_path: str | None = None,
    file_size: int | None = None,
    sha256: str | None = None,
    num_pages: int | None = None,
    extraction_status: str = "pending",
    error_message: str | None = None,
    extracted_at: datetime.datetime | None = None,
) -> PdfDocumentRecord:
    """Create or update a PdfDocumentRecord."""
    stmt = sa.select(PdfDocumentRecord).where(PdfDocumentRecord.doc_id == doc_id)
    doc = session.execute(stmt).scalar_one_or_none()

    if doc is None:
        doc = PdfDocumentRecord(
            doc_id=doc_id,
            sec_code=sec_code,
            filer_name=filer_name,
            doc_type=doc_type,
            doc_type_code=doc_type_code,
            category=category,
            filing_date=filing_date,
            file_path=file_path,
            file_size=file_size,
            sha256=sha256,
            num_pages=num_pages,
            extraction_status=extraction_status,
            error_message=error_message,
            extracted_at=extracted_at,
        )
        session.add(doc)
    else:
        if sec_code:
            doc.sec_code = sec_code
        if filer_name:
            doc.filer_name = filer_name
        if doc_type:
            doc.doc_type = doc_type
        if doc_type_code:
            doc.doc_type_code = doc_type_code
        if category:
            doc.category = category
        if filing_date:
            doc.filing_date = filing_date
        if file_path:
            doc.file_path = file_path
        if file_size is not None:
            doc.file_size = file_size
        if sha256:
            doc.sha256 = sha256
        if num_pages is not None:
            doc.num_pages = num_pages
        doc.extraction_status = extraction_status
        doc.error_message = error_message
        if extracted_at is not None:
            doc.extracted_at = extracted_at

    session.flush()
    return doc


def save_extracted_tables(
    session: Session,
    document_id: int,
    doc_id: str,
    tables: list[ExtractedPdfTable],
    pdf_path: Path | None = None,
) -> list[PdfExtractedTableRecord]:
    """Persist extracted tables into pdf_extracted_tables."""
    # Delete existing extracted tables for this document to ensure idempotence
    session.execute(
        sa.delete(PdfExtractedTableRecord).where(PdfExtractedTableRecord.document_id == document_id)
    )

    records: list[PdfExtractedTableRecord] = []
    for t in tables:
        title = detect_table_title(t, pdf_path)
        headers = [str(c) for c in t.df.columns]
        data = t.df.to_dict(orient="records")

        rec = PdfExtractedTableRecord(
            document_id=document_id,
            doc_id=doc_id,
            page_number=t.page_number,
            table_index=t.table_index,
            table_title=title,
            row_count=t.row_count,
            col_count=t.col_count,
            headers_json=headers,
            data_json=data,
            html=t.html,
            bbox_json=list(t.bbox),
            meta_json={},
        )
        session.add(rec)
        records.append(rec)

    session.flush()
    return records


def ingest_pdf_document(
    doc_id: str,
    meta: dict[str, Any],
    session: Session,
    force: bool = False,
    page_numbers: list[int] | None = None,
    max_pages: int | None = 25,
) -> dict[str, Any]:
    """
    Download, save, and extract tables from an EDINET PDF document.
    """
    filing_date = _to_date(meta.get("submitDateTime") or meta.get("filing_date"))
    sec_code = meta.get("secCode") or meta.get("issuer_code")
    filer_name = meta.get("filerName") or meta.get("filer_name")
    doc_type = meta.get("docDescription") or meta.get("doc_type")
    doc_type_code = meta.get("docTypeCode")
    category = meta.get("category", "prospectus")

    # Check if already processed
    existing = session.execute(
        sa.select(PdfDocumentRecord).where(PdfDocumentRecord.doc_id == doc_id)
    ).scalar_one_or_none()

    if existing and existing.extraction_status == "completed" and not force:
        logger.info("Document %s already processed (status=completed). Skipping.", doc_id)
        return {
            "doc_id": doc_id,
            "status": "skipped",
            "message": "Already processed",
            "tables_extracted": len(existing.tables),
        }

    # Record or update initial state
    doc_rec = upsert_pdf_document(
        session=session,
        doc_id=doc_id,
        sec_code=sec_code,
        filer_name=filer_name,
        doc_type=doc_type,
        doc_type_code=doc_type_code,
        category=category,
        filing_date=filing_date,
        extraction_status="pending",
    )

    # 1. Obtain PDF bytes: check local disk first or download from EDINET API
    pdf_bytes: bytes | None = None
    if doc_rec.file_path:
        local_path = resolve_pdf_path(doc_rec.file_path)
        if local_path and local_path.exists():
            pdf_bytes = local_path.read_bytes()

    if not pdf_bytes:
        pdf_bytes = download_edinet_pdf(doc_id)

    if not pdf_bytes:
        err = "Failed to download PDF binary from EDINET API"
        doc_rec.extraction_status = "error"
        doc_rec.error_message = err
        session.commit()
        return {"doc_id": doc_id, "status": "error", "error": err}

    # 2. Save PDF file to storage directory
    rel_path, sha256, size, abs_path = save_pdf_file(
        doc_id=doc_id,
        category=category,
        filing_date=filing_date,
        content=pdf_bytes,
    )
    doc_rec.file_path = rel_path
    doc_rec.sha256 = sha256
    doc_rec.file_size = size

    # 3. Extract tables
    extractor = PdfTableExtractor()
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(abs_path))
        total_pdf_pages = len(reader.pages)
        doc_rec.num_pages = total_pdf_pages

        # Target pages for table extraction (default: up to max_pages)
        if page_numbers is None:
            limit = min(total_pdf_pages, max_pages) if max_pages else total_pdf_pages
            target_pages = list(range(1, limit + 1))
        else:
            target_pages = page_numbers

        tables = extractor.extract_tables(abs_path, page_numbers=target_pages)
        saved_tables = save_extracted_tables(
            session=session,
            document_id=doc_rec.id,
            doc_id=doc_id,
            tables=tables,
            pdf_path=abs_path,
        )

        doc_rec.extraction_status = "completed"
        doc_rec.error_message = None
        doc_rec.extracted_at = datetime.datetime.now(datetime.timezone.utc)
        session.commit()

        logger.info(
            "Document %s ingested successfully: %d tables extracted from %d pages",
            doc_id,
            len(saved_tables),
            len(target_pages),
        )
        return {
            "doc_id": doc_id,
            "status": "completed",
            "pages_scanned": len(target_pages),
            "total_pages": total_pdf_pages,
            "tables_extracted": len(saved_tables),
            "file_path": rel_path,
        }
    except Exception as exc:
        logger.exception("Error extracting tables from %s: %s", abs_path, exc)
        doc_rec.extraction_status = "error"
        doc_rec.error_message = str(exc)
        session.commit()
        return {"doc_id": doc_id, "status": "error", "error": str(exc)}


def run_pdf_prospectus_collection(
    target_date: dt_date | None = None,
    max_documents: int = 20,
    session: Session | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Scan EDINET documents for a given date, filter for prospectus/offering filings,
    download their PDF and extract structured tables into the database.
    """
    t_date = target_date or dt_date.today()
    date_str = t_date.isoformat()
    logger.info("Starting prospectus PDF collection for date=%s (max=%d)", date_str, max_documents)

    api_key = get_edinet_api_key()
    doc_list = fetch_document_list(date_str, api_key)
    if not doc_list:
        logger.info("No documents found on EDINET for %s", date_str)
        return {
            "date": date_str,
            "total_candidates": 0,
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "documents": [],
        }

    # Filter for offering and prospectus document type codes
    candidates = []
    for d in doc_list:
        doc_type_code = d.get("docTypeCode")
        if doc_type_code in PROSPECTUS_DOC_TYPES:
            candidates.append(d)

    logger.info("Found %d prospectus/offering candidates for %s", len(candidates), date_str)

    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    results = []
    try:
        for item in candidates[:max_documents]:
            doc_id = item.get("docID")
            if not doc_id:
                continue

            meta = {
                "submitDateTime": item.get("submitDateTime"),
                "secCode": item.get("secCode"),
                "filerName": item.get("filerName"),
                "docDescription": item.get("docDescription"),
                "docTypeCode": item.get("docTypeCode"),
                "category": "prospectus",
            }

            res = ingest_pdf_document(
                doc_id=doc_id,
                meta=meta,
                session=session,
                force=force,
                max_pages=25,
            )
            results.append(res)
    finally:
        if close_session:
            session.close()

    succeeded = sum(1 for r in results if r.get("status") in ("completed", "skipped"))
    failed = sum(1 for r in results if r.get("status") == "error")

    return {
        "date": date_str,
        "total_candidates": len(candidates),
        "processed": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }
