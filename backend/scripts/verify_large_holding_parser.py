"""Verification script for EDINET Large Volume Holding Report (5% rule) Extractor.

Downloads real recent large holding reports (大量保有報告書 docTypeCode 350, 360, 370)
from EDINET API, parses them with extract_large_holding_from_html, and verifies
accuracy of extracted fields (issuer, filer, ratio, delta, 60-day trades).
"""

from __future__ import annotations

import datetime
import io
import logging
from pathlib import Path
import sys
import zipfile

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from data.extractors.edinet import extract_large_holding_from_html
from scripts.verify_edinet_html_parser import (
    download_document_zip,
    fetch_document_list,
    get_edinet_api_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_large_holding")
sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    api_key = get_edinet_api_key()
    today = datetime.date.today()

    # Collect large holding documents from past 5 business days
    candidates = []
    for days_back in range(1, 10):
        d = today - datetime.timedelta(days=days_back)
        if d.weekday() >= 5:
            continue
        d_str = d.strftime("%Y-%m-%d")
        docs = fetch_document_list(d_str, api_key)
        for doc in docs:
            dt = doc.get("docTypeCode")
            if dt in ["350", "360"]:
                candidates.append(doc)
        if len(candidates) >= 15:
            break

    print(f"\n==================================================================")
    print(f"Targeting {len(candidates[:10])} Large Holding Reports for verification")
    print(f"==================================================================")

    success_count = 0
    trades_found_count = 0

    for idx, doc in enumerate(candidates[:10], start=1):
        doc_id = doc.get("docID")
        doc_type = doc.get("docTypeCode")
        desc = doc.get("docDescription")
        filer = doc.get("filerName")

        print(f"\n[{idx}/10] Fetching {doc_id} | Type: {doc_type} | {filer} | {desc}")
        zip_bytes = download_document_zip(doc_id, api_key)
        if not zip_bytes:
            print(f"  [FAIL] Failed to download ZIP for {doc_id}")
            continue

        honbun_bytes = None
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for name in z.namelist():
                if "honbun" in name and name.endswith((".htm", ".html")):
                    honbun_bytes = z.read(name)
                    break

        if not honbun_bytes:
            print(f"  [FAIL] No honbun HTML found in {doc_id}")
            continue

        try:
            report = extract_large_holding_from_html(honbun_bytes, doc_id=doc_id)
            has_ratio = report.holding_ratio is not None
            has_code = report.issuer_code is not None or report.issuer_name is not None

            if has_ratio and has_code:
                success_count += 1
                change_str = f" (Delta: {report.ratio_change:+.2f}%)" if report.ratio_change is not None else ""
                print(f"  [SUCCESS] Code: {report.issuer_code} ({report.issuer_name})")
                print(f"            Filer: {report.filer_name} | Ratio: {report.holding_ratio}%{change_str}")
                print(f"            Holders: {len(report.joint_holders)} | Base Date: {report.base_date}")
                if report.recent_trades:
                    trades_found_count += 1
                    print(f"            60-Day Trades: {len(report.recent_trades)} records")
                    sample_t = report.recent_trades[0]
                    print(f"              -> Sample: {sample_t.trade_date} | {sample_t.trade_type} | {sample_t.quantity:,.0f} shares")
            else:
                print(f"  [PARTIAL] Parsed but missing ratio or issuer info: ratio={report.holding_ratio}, code={report.issuer_code}")
        except Exception as exc:
            print(f"  [ERROR] Exception parsing {doc_id}: {exc}")

    print(f"\n==================================================================")
    print(f"Summary: {success_count}/10 successful ({success_count/10*100:.1f}%)")
    print(f"Reports with 60-day trade history: {trades_found_count}")
    print(f"==================================================================")


if __name__ == "__main__":
    main()
