"""EDINET Live Downloader and Table Parser Verification Runner.

Fetches real disclosure documents (有価証券報告書 / 四半期報告書) from EDINET API,
extracts financial tables, and checks whether the extractor parses them correctly.
Saves problematic or novel HTML layouts as test fixtures for regression testing.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import zipfile
import winreg
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
    html_tables_to_df,
    parse_financial_metrics_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("edinet_verifier")


def get_edinet_api_key() -> str:
    """Retrieve EDI_KEY from environment or Windows registry."""
    key = os.environ.get("EDI_KEY")
    if key:
        return key

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as reg_key:
            val, _ = winreg.QueryValueEx(reg_key, "EDI_KEY")
            if val:
                return str(val).strip()
    except Exception:
        pass

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ) as reg_key:
            val, _ = winreg.QueryValueEx(reg_key, "EDI_KEY")
            if val:
                return str(val).strip()
    except Exception:
        pass

    raise RuntimeError(
        "EDI_KEY not found in environment or registry. Please set EDI_KEY."
    )


def fetch_document_list(date_str: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch EDINET document list for a specific date (YYYY-MM-DD)."""
    url = (
        f"https://api.edinet-fsa.go.jp/api/v2/documents.json"
        f"?date={date_str}&type=2&Subscription-Key={api_key}"
    )
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        logger.error("Failed to fetch document list for %s: HTTP %s", date_str, resp.status_code)
        return []

    data = resp.json()
    return data.get("results", [])


def download_document_zip(doc_id: str, api_key: str) -> bytes | None:
    """Download disclosure document ZIP bytes from EDINET API."""
    url = (
        f"https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}"
        f"?type=1&Subscription-Key={api_key}"
    )
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.content
            logger.warning("[%s] Download failed: HTTP %s (attempt %d)", doc_id, resp.status_code, attempt + 1)
        except Exception as exc:
            logger.warning("[%s] Download error: %s (attempt %d)", doc_id, exc, attempt + 1)
        time.sleep(2 * (attempt + 1))
    return None


def decode_html_bytes(raw_bytes: bytes) -> str:
    """Safely decode HTML bytes trying utf-8, cp932 (Shift-JIS), then fallback."""
    for enc in ("utf-8", "cp932", "shift_jis", "euc-jp"):
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def extract_html_documents_from_zip(zip_bytes: bytes) -> list[tuple[str, str]]:
    """Extract HTML documents containing financial reports from ZIP bytes.

    Returns a list of (filename, html_content_str).
    """
    html_docs: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            # Filter for public disclosure HTML/HTM documents
            if name.endswith((".htm", ".html")) and "PublicDoc" in name:
                try:
                    content = decode_html_bytes(z.read(name))
                    html_docs.append((name, content))
                except Exception as exc:
                    logger.debug("Failed reading %s: %s", name, exc)
    return html_docs


def verify_html_parsing(
    doc_id: str,
    sec_code: str,
    filer_name: str,
    html_name: str,
    html_content: str,
    fixture_dir: Path,
) -> dict[str, Any]:
    """Test extracting metrics from an HTML document."""
    tree = lxml.html.fromstring(html_content.encode("utf-8", errors="ignore"))
    tables = tree.xpath(".//table")

    if not tables:
        return {"status": "no_tables"}

    # Find candidate financial tables
    target_tables: list[Any] = []
    for t in tables:
        txt = t.text_content()
        if any(kw in txt for kw in ("売上高", "営業収益", "経常利益", "当期純利益", "純資産額")):
            target_tables.append(t)

    if not target_tables:
        return {"status": "no_financial_tables"}

    results: list[dict[str, Any]] = []
    for idx, table in enumerate(target_tables):
        try:
            df = parse_financial_metrics_table(table)
            if df.empty or len(df.columns) == 0:
                status = "empty_df"
            else:
                status = "success"

            metric_count = len(df.index) if not df.empty else 0
            period_count = len(df.columns) if not df.empty else 0

            # Save suspicious / problematic HTML for inspection
            if status != "success" or metric_count < 2 or period_count < 2:
                fixture_dir.mkdir(parents=True, exist_ok=True)
                sample_file = fixture_dir / f"{doc_id}_{sec_code}_t{idx}.html"
                sample_file.write_text(lxml.html.tostring(table, encoding="unicode"), encoding="utf-8")
                logger.warning(
                    "[%s - %s (%s)] Table %d has questionable output (metrics: %d, periods: %d). Saved to %s",
                    doc_id, filer_name, sec_code, idx, metric_count, period_count, sample_file.name
                )

            results.append({
                "table_idx": idx,
                "status": status,
                "metrics": list(df.index)[:5] if not df.empty else [],
                "columns": [str(c) for c in df.columns] if not df.empty else [],
                "metric_count": metric_count,
                "period_count": period_count,
            })
        except Exception as exc:
            logger.error("[%s - %s] Error parsing table %d: %s", doc_id, filer_name, idx, exc, exc_info=True)
            # Save failing table HTML
            fixture_dir.mkdir(parents=True, exist_ok=True)
            err_file = fixture_dir / f"ERROR_{doc_id}_{sec_code}_t{idx}.html"
            err_file.write_text(lxml.html.tostring(table, encoding="unicode"), encoding="utf-8")
            results.append({
                "table_idx": idx,
                "status": f"error: {exc}",
                "error_file": str(err_file),
            })

    return {"status": "checked", "tables": results}


def run_verification(dates: list[str], max_docs: int = 30) -> None:
    """Run live verification against EDINET API across diverse listed companies."""
    api_key = get_edinet_api_key()
    logger.info("Retrieved EDINET API key (length: %d)", len(api_key))

    fixture_candidates = backend_root / "tests" / "fixtures" / "edinet_html" / "candidates"

    seen_companies: set[str] = set()
    candidate_docs: list[dict[str, Any]] = []

    for d in dates:
        logger.info("Fetching document list for date: %s", d)
        docs = fetch_document_list(d, api_key)
        # Filter for unique listed companies with annual securities reports (有価証券報告書: 120)
        for doc in docs:
            raw_sec = doc.get("secCode")
            doc_type = doc.get("docTypeCode")
            if raw_sec and str(raw_sec).strip() and str(raw_sec).strip() != "None" and doc_type == "120":
                sec_code = str(raw_sec).strip()
                edinet_code = doc.get("edinetCode") or sec_code
                if edinet_code not in seen_companies:
                    seen_companies.add(edinet_code)
                    candidate_docs.append(doc)
                    if len(candidate_docs) >= max_docs:
                        break
        if len(candidate_docs) >= max_docs:
            break

    logger.info("Selected %d unique companies for diverse testing", len(candidate_docs))

    summary_records: list[dict[str, Any]] = []

    for idx, doc in enumerate(candidate_docs, 1):
        doc_id = doc.get("docID")
        sec_code = str(doc.get("secCode", "")).strip()
        filer_name = doc.get("filerName", "")
        doc_desc = doc.get("docDescription", "")

        logger.info(
            "[%d/%d] Testing: %s (%s) - DocID: %s",
            idx, len(candidate_docs), filer_name, sec_code, doc_id
        )

        zip_bytes = download_document_zip(doc_id, api_key)
        if not zip_bytes:
            logger.warning("   Failed to download ZIP for %s", doc_id)
            summary_records.append({
                "sec_code": sec_code,
                "filer_name": filer_name,
                "status": "download_failed",
            })
            continue

        html_docs = extract_html_documents_from_zip(zip_bytes)
        main_table_found = False
        parsed_metric_list: list[str] = []
        parsed_periods_list: list[str] = []

        for html_name, html_content in html_docs:
            # Look for main financial section
            if not any(kw in html_content for kw in ("主要な経営指標等の推移", "提出会社の状況")):
                if not any(kw in html_content for kw in ("売上高", "営業収益", "経常利益", "当期純利益")):
                    continue

            # Strip XML declaration if present
            if "<?xml" in html_content[:120]:
                html_content = re.sub(r"<\?xml.*?\?>", "", html_content, count=1).strip()

            try:
                tree = lxml.html.fromstring(html_content)
            except Exception as exc:
                logger.debug("Failed parsing tree for %s: %s", html_name, exc)
                continue
            tables = tree.xpath(".//table")
            for t_idx, table in enumerate(tables):
                txt = table.text_content()
                if any(kw in txt for kw in ("売上高", "営業収益", "経常利益", "当期純利益", "純資産額")):
                    try:
                        df = parse_financial_metrics_table(table)
                        if not df.empty and len(df.columns) >= 2 and len(df.index) >= 3:
                            main_table_found = True
                            parsed_metric_list = list(df.index)
                            parsed_periods_list = [str(c) for c in df.columns]
                            logger.info(
                                "   [SUCCESS] Metrics (%d): %s | Periods (%d): %s",
                                len(df.index), list(df.index)[:4], len(df.columns), list(df.columns)
                            )
                            break
                    except Exception as exc:
                        logger.error("   [ERROR] Table %d failed: %s", t_idx, exc)
                        fixture_candidates.mkdir(parents=True, exist_ok=True)
                        err_path = fixture_candidates / f"ERROR_{doc_id}_{sec_code}_t{t_idx}.html"
                        err_path.write_text(lxml.html.tostring(table, encoding="unicode"), encoding="utf-8")
            if main_table_found:
                break

        summary_records.append({
            "sec_code": sec_code,
            "filer_name": filer_name,
            "doc_id": doc_id,
            "status": "success" if main_table_found else "table_not_found",
            "metric_count": len(parsed_metric_list),
            "period_count": len(parsed_periods_list),
            "sample_metrics": parsed_metric_list[:5],
            "sample_periods": parsed_periods_list,
        })

        # Rate limit friendly sleep
        time.sleep(0.5)

    # Print Report
    logger.info("=" * 70)
    logger.info("DIVERSE COMPANY VERIFICATION REPORT:")
    success_count = sum(1 for r in summary_records if r["status"] == "success")
    logger.info("Total Unique Companies Tested: %d", len(summary_records))
    logger.info("Successfully Parsed Main Financial Table: %d (%.1f%%)",
                success_count, (success_count / len(summary_records) * 100) if summary_records else 0.0)

    failed = [r for r in summary_records if r["status"] != "success"]
    if failed:
        logger.warning("Unsuccessful / Problematic Companies (%d):", len(failed))
        for r in failed:
            logger.warning(" - %s (%s): %s", r["filer_name"], r["sec_code"], r["status"])
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify EDINET HTML parser with live data.")
    parser.add_argument("--dates", nargs="+", default=["2026-11-06", "2026-10-30", "2026-06-26"], help="Dates to check")
    parser.add_argument("--max-docs", type=int, default=5, help="Maximum documents to process")
    args = parser.parse_args()

    run_verification(args.dates, max_docs=args.max_docs)
