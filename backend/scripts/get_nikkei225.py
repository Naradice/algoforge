import json
import urllib.request
import lxml.html

url = "https://ja.wikipedia.org/wiki/%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87%E6%A0%AA%E4%BE%A1%E3%81%AE%E6%A7%8B%E6%88%90%E9%8A%98%E6%9F%84%E4%B8%80%E8%A6%A7"
req = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
)

with urllib.request.urlopen(req, timeout=15) as resp:
    tree = lxml.html.parse(resp)

tables = tree.xpath("//table")
nikkei_codes = []

for t in tables:
    for r in t.xpath(".//tr"):
        cells = r.xpath("./th|./td")
        for c in cells:
            val = c.text_content().strip()
            if len(val) == 4 and val.isdigit():
                nikkei_codes.append(val)

nikkei_codes = sorted(list(set(nikkei_codes)))
print(f"Extracted Nikkei 225 constituent codes: {len(nikkei_codes)}")
print("Sample codes:", nikkei_codes[:15])

if len(nikkei_codes) >= 200:
    with open("nikkei225_codes.json", "w", encoding="utf-8") as f:
        json.dump(nikkei_codes, f, indent=2)
    print("Saved to nikkei225_codes.json")
