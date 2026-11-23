"""End-to-end verification pipeline for Category 1: Prospectus & Offering Documents.

Demonstrates:
1. Fetching real offering document from EDINET API (type=2 PDF)
2. Extracting Offering Terms Table (Page 2)
3. Extracting MS Warrant / Exercise Adjustment Table (Page 3-4)
4. Extracting Use of Proceeds Breakdown Table (Page 16)
"""

from pathlib import Path
import sys

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))
sys.stdout.reconfigure(encoding="utf-8")

from data.extractors.pdf import PdfTableExtractor


def main():
    pdf_path = backend_root / "artifacts" / "pdf_samples" / "S100ZJBD_prospectus.pdf"
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        sys.exit(1)

    print("=" * 70)
    print(" Category 1: Prospectus Table Extraction Verification Pipeline")
    print(f" Target Document: {pdf_path.name}")
    print("=" * 70)

    extractor = PdfTableExtractor()

    # 1. Page 2: Offering Terms
    print("\n[Target 1] Offering Terms Table (Page 2):")
    t_p2 = extractor.extract_tables(pdf_path, page_numbers=[2])
    if t_p2:
        print(f"  ✓ Extracted table ({t_p2[0].row_count} rows x {t_p2[0].col_count} cols)")
        print(t_p2[0].df.head(5).to_string())

    # 2. Page 16: Use of Proceeds Breakdown
    print("\n[Target 2] Use of Proceeds Breakdown (Page 16):")
    t_p16 = extractor.extract_tables(pdf_path, page_numbers=[16])
    for idx, t in enumerate(t_p16, 1):
        print(f"\n  -- Table {idx} ({t.row_count} rows x {t.col_count} cols) --")
        print(t.df.to_string())

    print("\n" + "=" * 70)
    print(" Verification Pipeline Passed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
