import io
import re
import sys
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import openpyxl
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
headers = {"User-Agent": "Mozilla/5.0"}

# 1. Inspect Daily Margin Excel (日々公表銘柄残高)
daily_excel_url = "https://www.jpx.co.jp/markets/statistics-equities/margin/tvdivq0000001r92-att/20261106_mtdaily.xlsx"
print(f"--- 1. Inspecting Daily Margin Excel: {daily_excel_url} ---")
r_daily = requests.get(daily_excel_url, headers=headers)
if r_daily.status_code == 200:
    wb = openpyxl.load_workbook(io.BytesIO(r_daily.content), read_only=True)
    print("Sheets:", wb.sheetnames)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    print(f"Total rows: {len(rows)}")
    for i, r in enumerate(rows[:15]):
        print(f"Row {i}: {r[:10]}")
    # Sample data row
    for i, r in enumerate(rows[10:25], start=10):
        if any(r):
            print(f"Data Row {i}: {r[:10]}")

# 2. Inspect Weekly Margin page (銘柄別信用取引残高 01.html)
weekly_01_url = "https://www.jpx.co.jp/markets/statistics-equities/margin/01.html"
print(f"\n--- 2. Inspecting Weekly Margin by Symbol: {weekly_01_url} ---")
r_01 = requests.get(weekly_01_url, headers=headers)
if r_01.status_code == 200:
    soup = BeautifulSoup(r_01.content, "html.parser")
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if ".xlsx" in href or ".csv" in href or ".xls" in href:
            full = urljoin(weekly_01_url, href)
            print("Found weekly symbol file:", full)
            # Inspect first weekly file
            r_w = requests.get(full, headers=headers)
            if r_w.status_code == 200:
                wb_w = openpyxl.load_workbook(io.BytesIO(r_w.content), read_only=True)
                print("  Sheets:", wb_w.sheetnames)
                ws_w = wb_w.active
                w_rows = list(ws_w.iter_rows(values_only=True))
                print(f"  Total rows: {len(w_rows)}")
                for i, r in enumerate(w_rows[:12]):
                    print(f"  Row {i}: {r[:10]}")
            break
