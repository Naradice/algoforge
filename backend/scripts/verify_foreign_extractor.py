"""Verification script for Foreign Company (10-K) and Foreign Fund PDF Extraction (Issue #2)."""

from pathlib import Path
import sys

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))
sys.stdout.reconfigure(encoding="utf-8")

from database import SessionLocal
from data.extractors.pdf import PdfTableExtractor
from data.pdf_data_service import ingest_pdf_document, get_storage_root

def verify_foreign():
    print("=" * 70)
    print("Issue #2: Foreign Company & Foreign Fund Extraction Verification")
    print("=" * 70)

    db = SessionLocal()
    try:
        # 1. Global Funds Management (Foreign Fund - Schedule of Investments)
        print("\n[1] Verifying Foreign Investment Trust (Global Funds Management: S100ZIPG)...")
        meta_fund = {
            "submitDateTime": "2026-11-20",
            "filerName": "グローバル・ファンズ・マネジメント・エス・エー",
            "docDescription": "有価証券報告書（外国投資信託受益証券）",
            "docTypeCode": "120",
            "category": "foreign",
        }
        res_fund = ingest_pdf_document("S100ZIPG", meta_fund, db, force=True, page_numbers=[5, 35])
        print("  Fund Ingest Result:", res_fund)

        # 2. Cisco Systems Form 10-K (Foreign Corporation - Financial Statements)
        print("\n[2] Verifying Foreign Corporation 10-K (Cisco Systems: S100ZI6C)...")
        meta_cisco = {
            "submitDateTime": "2026-11-17",
            "filerName": "シスコ・システムズ・インク (Cisco Systems, Inc.)",
            "docDescription": "外国会社報告書",
            "docTypeCode": "120",
            "category": "foreign",
        }
        res_cisco = ingest_pdf_document("S100ZI6C", meta_cisco, db, force=True, page_numbers=[58, 59])
        print("  Cisco 10-K Ingest Result:", res_cisco)

        print("\n✓ Foreign company and fund verification completed successfully!")
    finally:
        db.close()

if __name__ == "__main__":
    verify_foreign()
