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

# 1. Daily Excel rows 20 to 40
daily_excel_url = "https://www.jpx.co.jp/markets/statistics-equities/margin/tvdivq0000001r92-att/20261106_mtdaily.xlsx"
r_daily = requests.get(daily_excel_url, headers=headers)
wb = openpyxl.load_workbook(io.BytesIO(r_daily.content), read_only=True)
ws = wb.active
rows = list(ws.iter_rows(values_only=True))

print("--- Daily Table Header & Samples ---")
for i in range(23, 40):
    if i < len(rows):
        print(f"Row {i:2d}: {rows[i][:15]}")

# 2. Check 01.html links
weekly_01_url = "https://www.jpx.co.jp/markets/statistics-equities/margin/01.html"
r_01 = requests.get(weekly_01_url, headers=headers)
soup = BeautifulSoup(r_01.content, "html.parser")
print("\n--- Weekly 01.html links ---")
for a in soup.find_all("a"):
    h = a.get("href", "")
    t = a.text.strip().replace("\n", " ")
    if any(ext in h for ext in [".xls", ".xlsx", ".csv", ".zip"]):
        print(f"Weekly link: {t} -> {urljoin(weekly_01_url, h)}")
