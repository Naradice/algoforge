"""Download real Prospectus/Offering PDF from EDINET API and extract tables."""

from pathlib import Path
import sys
import requests
import time

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.verify_edinet_html_parser import get_edinet_api_key
from data.extractors.pdf import PdfTableExtractor


def download_edinet_pdf(doc_id: str, save_path: Path) -> bool:
    """Download PDF (type=2) directly from EDINET API."""
    api_key = get_edinet_api_key()
    url = f"https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}?type=2&Subscription-Key={api_key}"
    print(f"Downloading PDF for docID={doc_id} from EDINET API...")
    
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200 and resp.content.startswith(b"%PDF"):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(resp.content)
        print(f"  ✓ Saved PDF ({len(resp.content):,} bytes) to {save_path}")
        return True
    else:
        print(f"  ✗ Failed: HTTP {resp.status_code}, content prefix: {resp.content[:30]}")
        return False


def test_real_prospectus_extraction():
    target_doc_id = "S100ZJBD"  # 株式会社ペルセウスプロテオミクス 有価証券届出書
    pdf_dir = backend_root / "artifacts" / "pdf_samples"
    pdf_path = pdf_dir / f"{target_doc_id}_prospectus.pdf"

    if not pdf_path.exists():
        ok = download_edinet_pdf(target_doc_id, pdf_path)
        if not ok:
            print("Download failed, aborting.")
            return

    extractor = PdfTableExtractor()
    # Test first 5 pages of the offering document
    print(f"\nExtracting tables from {pdf_path.name} (pages 1 to 5)...")
    tables = extractor.extract_tables(pdf_path, page_numbers=[1, 2, 3, 4, 5])
    print(f"\nExtracted {len(tables)} tables from pages 1-5:")

    for i, t in enumerate(tables, start=1):
        print(f"\n--- Table #{i} (Page {t.page_number}, Index {t.table_index}) ---")
        print(f"Shape: {t.row_count} rows x {t.col_count} columns")
        print("Data Preview (first 5 rows):")
        print(t.df.head(5))
        print("HTML Preview:")
        print(t.html[:150].replace("\n", " "))
        print("-" * 60)


if __name__ == "__main__":
    test_real_prospectus_extraction()
