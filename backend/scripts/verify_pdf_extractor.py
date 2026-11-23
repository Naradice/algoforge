"""Interactive verification script for PdfTableExtractor.

Usage:
    python scripts/verify_pdf_extractor.py [--pdf path/to/file.pdf] [--pages 2,13] [--save]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure backend root is on sys.path
backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))
sys.stdout.reconfigure(encoding="utf-8")

from data.extractors.pdf import PdfTableExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract structured tables from PDF documents.")
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(backend_root / "tests" / "samples" / "S100S37P_2.pdf"),
        help="Path to the target PDF file.",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default="2,13",
        help="Comma-separated list of page numbers (1-indexed), e.g. '2,13'. Omit to process all pages.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save extracted tables to artifacts/pdf_extracted/ as CSV and HTML.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    page_numbers = [int(p.strip()) for p in args.pages.split(",")] if args.pages else None

    print("=" * 70)
    print(" AlgoForge PDF Table Extractor Verification")
    print("=" * 70)
    print(f"Target PDF : {pdf_path.name}")
    print(f"Pages      : {page_numbers if page_numbers else 'All'}")

    extractor = PdfTableExtractor()
    tables = extractor.extract_tables(pdf_path, page_numbers=page_numbers)

    print(f"\n[Result] Extracted {len(tables)} tables.\n")

    out_dir = backend_root / "artifacts" / "pdf_extracted"
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(tables, start=1):
        print(f"--- Table #{i} (Page {t.page_number}, Index {t.table_index}) ---")
        print(f"Shape : {t.row_count} rows x {t.col_count} columns")
        print(f"BBox  : x=[{t.bbox[0]:.1f}, {t.bbox[2]:.1f}], y=[{t.bbox[1]:.1f}, {t.bbox[3]:.1f}]")
        print("\nDataFrame Preview:")
        print(t.df.head(6))
        print("\nHTML Preview (First 200 chars):")
        print(t.html[:200].replace("\n", " "))
        print("\n" + "-" * 70)

        if args.save:
            csv_path = out_dir / f"{pdf_path.stem}_p{t.page_number}_t{t.table_index}.csv"
            html_path = out_dir / f"{pdf_path.stem}_p{t.page_number}_t{t.table_index}.html"
            t.df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            html_path.write_text(t.html, encoding="utf-8")
            print(f"Saved: {csv_path.name} & {html_path.name}")

    print("Verification completed successfully.")


if __name__ == "__main__":
    main()
