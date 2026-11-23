"""Search and download prospectus and offering documents (有価証券届出書・発行登録追補書類)."""

import datetime
from pathlib import Path
import sys
import requests

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.verify_edinet_html_parser import get_edinet_api_key, fetch_document_list


def search_prospectus_documents(days_back: int = 15):
    api_key = get_edinet_api_key()
    today = datetime.date.today()
    
    target_doc_types = {
        "010": "有価証券届出書",
        "020": "発行登録書",
        "030": "発行登録追補書類",
        "050": "訂正有価証券届出書",
        "070": "訂正発行登録追補書類",
    }
    
    found_docs = []
    print(f"Scanning EDINET documents over past {days_back} days...")
    
    for offset in range(1, days_back + 1):
        target_date = today - datetime.timedelta(days=offset)
        # Skip weekends if possible, but EDINET API handles them gracefully
        d_str = target_date.strftime("%Y-%m-%d")
        docs = fetch_document_list(d_str, api_key)
        
        for doc in docs:
            dt = doc.get("docTypeCode")
            if dt in target_doc_types:
                found_docs.append({
                    "date": d_str,
                    "docId": doc.get("docId"),
                    "docTypeCode": dt,
                    "docTypeName": target_doc_types.get(dt, ""),
                    "secCode": doc.get("secCode"),
                    "filerName": doc.get("filerName"),
                    "docDescription": doc.get("docDescription"),
                    "pdfFlag": doc.get("pdfFlag"),  # 1 = PDF available
                })
        if len(found_docs) >= 5:
            break

    print(f"\nFound {len(found_docs)} offering / prospectus documents:")
    for d in found_docs:
        print(f"  [{d['date']}] {d['docId']} | {d['docTypeName']} ({d['docTypeCode']}) | "
              f"Code: {d['secCode']} | {d['filerName'][:20]} | PDF: {d['pdfFlag']}")
        if d['docDescription']:
            print(f"      Description: {d['docDescription']}")

    return found_docs


if __name__ == "__main__":
    search_prospectus_documents()
