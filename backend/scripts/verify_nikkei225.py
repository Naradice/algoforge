"""Nikkei 225 Constituents Verification Runner for EDINET HTML Parser.

Downloads annual securities reports (有価証券報告書) for Nikkei 225 component companies
and verifies robust extraction of primary financial metric tables across diverse accounting
standards (IFRS, Japanese GAAP, US GAAP) and sectors (Banks, Insurance, Autos, Tech, Retail, etc.).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import re
import sys
import time
import zipfile
from typing import Any

import lxml.html
import pandas as pd
import requests

# Ensure algoforge backend is on sys.path
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from data.extractors.edinet import (
    extract_financial_metrics_from_html,
    parse_financial_metrics_table,
)
from scripts.verify_edinet_html_parser import (
    decode_html_bytes,
    download_document_zip,
    fetch_document_list,
    get_edinet_api_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("nikkei225_verifier")


def load_nikkei225_codes() -> set[str]:
    """Load Nikkei 225 security codes (4-digit string)."""
    codes_path = backend_root / "scripts" / "nikkei225_codes.json"
    if not codes_path.exists():
        # Fallback to local file
        codes_path = Path("nikkei225_codes.json")
    with open(codes_path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def extract_primary_financial_table_from_zip(
    zip_bytes: bytes,
) -> tuple[pd.DataFrame | None, str | None, Any]:
    """Extract primary financial metrics DataFrame from document ZIP."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        candidate_files: list[str] = []
        for name in z.namelist():
            if name.endswith((".htm", ".html")) and "PublicDoc" in name:
                candidate_files.append(name)

        # Sort candidate files to prioritize main business summary (0101... or honbun)
        candidate_files.sort(
            key=lambda fn: (
                0 if "0101" in fn else 1 if "honbun" in fn else 2
            )
        )

        for fname in candidate_files:
            try:
                raw_bytes = z.read(fname)
                content = decode_html_bytes(raw_bytes)
                if not any(kw in content for kw in ("主要な経営指標等の推移", "提出会社の状況")):
                    if not any(kw in content for kw in ("売上高", "売上収益", "経常収益", "営業収益")):
                        continue

                # Strip XML declaration if present
                if "<?xml" in content[:120]:
                    content = re.sub(r"<\?xml.*?\?>", "", content, count=1).strip()

                tree = lxml.html.fromstring(content)
                tables = tree.xpath(".//table")
                for table in tables:
                    txt = table.text_content()
                    if any(kw in txt for kw in ("売上高", "売上収益", "経常収益", "営業収益", "当期純利益", "当期利益")):
                        df = parse_financial_metrics_table(table)
                        # Check if it has multiple metrics and multiple periods
                        if not df.empty and len(df.columns) >= 2 and len(df.index) >= 3:
                            return df, fname, table
            except Exception as exc:
                logger.debug("Error in %s: %s", fname, exc)

    return None, None, None


def run_nikkei225_verification(
    dates: list[str],
    target_count: int = 15,
) -> None:
    """Run verification specifically for Nikkei 225 companies."""
    api_key = get_edinet_api_key()
    nikkei_codes = load_nikkei225_codes()
    logger.info("Loaded %d Nikkei 225 component security codes", len(nikkei_codes))

    fixture_candidates = backend_root / "tests" / "fixtures" / "edinet_html" / "candidates"

    matched_docs: dict[str, dict[str, Any]] = {}

    for d in dates:
        logger.info("Scanning EDINET documents on date: %s", d)
        docs = fetch_document_list(d, api_key)
        for doc in docs:
            raw_sec = doc.get("secCode")
            doc_type = doc.get("docTypeCode")
            if raw_sec and doc_type == "120":
                sec_str = str(raw_sec).strip()
                code4 = sec_str[:4]
                if code4 in nikkei_codes and code4 not in matched_docs:
                    matched_docs[code4] = doc
                    logger.info(
                        "   Found Nikkei 225 component: %s (%s) - %s",
                        doc.get("filerName"), code4, doc.get("docID")
                    )

        if len(matched_docs) >= target_count:
            break

    logger.info("Found %d distinct Nikkei 225 companies in disclosure dates", len(matched_docs))

    results_summary: list[dict[str, Any]] = []

    for idx, (code4, doc) in enumerate(list(matched_docs.items())[:target_count], 1):
        doc_id = doc.get("docID")
        filer_name = doc.get("filerName", "")
        sec_code = str(doc.get("secCode", "")).strip()

        logger.info(
            "[%d/%d] Testing Nikkei 225: %s (%s) - DocID: %s",
            idx, min(len(matched_docs), target_count), filer_name, code4, doc_id
        )

        zip_bytes = download_document_zip(doc_id, api_key)
        if not zip_bytes:
            logger.warning("   Failed downloading ZIP for %s", doc_id)
            results_summary.append({
                "code": code4,
                "name": filer_name,
                "status": "download_failed",
            })
            continue

        df, matched_file, raw_table = extract_primary_financial_table_from_zip(zip_bytes)

        if df is not None and not df.empty:
            logger.info(
                "   [SUCCESS] Metrics (%d): %s | Periods (%d): %s",
                len(df.index), list(df.index)[:5], len(df.columns), list(df.columns)
            )
            results_summary.append({
                "code": code4,
                "name": filer_name,
                "doc_id": doc_id,
                "status": "success",
                "metric_count": len(df.index),
                "period_count": len(df.columns),
                "metrics": list(df.index)[:5],
                "periods": [str(c) for c in df.columns],
            })
        else:
            logger.warning("   [FAILED] Financial table not successfully extracted for %s (%s)", filer_name, code4)
            results_summary.append({
                "code": code4,
                "name": filer_name,
                "doc_id": doc_id,
                "status": "table_not_extracted",
            })

        time.sleep(0.5)

    # Print final summary
    logger.info("=" * 70)
    logger.info("NIKKEI 225 VERIFICATION REPORT:")
    success_count = sum(1 for r in results_summary if r["status"] == "success")
    logger.info("Total Nikkei 225 Companies Tested: %d", len(results_summary))
    logger.info(
        "Successfully Extracted Primary Metrics: %d (%.1f%%)",
        success_count,
        (success_count / len(results_summary) * 100) if results_summary else 0.0,
    )
    logger.info("-" * 70)
    for r in results_summary:
        st = "OK" if r["status"] == "success" else "FAIL"
        logger.info(
            " [%s] %-5s %-25s | Metrics: %-2s | Periods: %-2s",
            st,
            r["code"],
            r["name"][:20],
            r.get("metric_count", "-"),
            r.get("period_count", "-"),
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    peak_dates = [
        "2026-06-26", "2026-06-25", "2026-06-24", "2026-06-23", "2026-06-22",
        "2026-06-19", "2026-06-18", "2026-03-27", "2026-03-26", "2026-03-25",
    ]
    run_nikkei225_verification(peak_dates, target_count=20)
