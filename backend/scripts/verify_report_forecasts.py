"""Report Forecast Extractor & Ground Truth Verification Engine.

Extracts analyst predictions (direction, support/resistance, expected range) from:
1. みんなのFX (Daily USD/JPY)
2. Sony Financial Group (Weekly USD/JPY)
3. SMBC Trust Bank (Weekly USD/JPY)

Evaluates accuracy against actual USD/JPY market prices from Yahoo Finance.
"""

from __future__ import annotations

import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pypdf
import yfinance as yf

# UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"L:\data\finance")


# ----------------------------------------------------------------------
# 1. Min-FX (みんなのFX - Daily) Extractor
# ----------------------------------------------------------------------
def extract_minfx(pdf_path: Path) -> dict | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception:
        return None

    full_text = "".join([page.extract_text() or "" for page in reader.pages])
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    if not lines:
        return None

    # Date
    report_date = None
    m = re.search(r"(20\d{2})[/-]([01]\d)[/-]([0-3]\d)", lines[0])
    if m:
        report_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        fn_match = re.search(r"(\d{2})([01]\d)([0-3]\d)", pdf_path.stem)
        if fn_match:
            report_date = date(2000 + int(fn_match.group(1)), int(fn_match.group(2)), int(fn_match.group(3)))

    if not report_date:
        return None

    # Title
    title = ""
    for l in lines:
        if l.startswith("-") and not l.startswith("-前営業"):
            title = l.strip("-").strip()
            break

    # Paragraphs (excluding copyright / previous day summary)
    paras = full_text.split(" ")
    outlook_paras = []
    for p in paras:
        if any(w in p for w in ["本日のイベント", "今後", "想定", "可能性", "上値", "下値", "目処", "レンジ", "警戒", "焦点", "試す"]):
            clean = " ".join(p.split())
            if not clean.startswith("Copyright") and not clean.startswith("http") and not clean.startswith("-前営業") and len(clean) > 30:
                outlook_paras.append(clean)

    outlook_text = " ".join(outlook_paras)

    # Price levels
    levels = []
    for m in re.finditer(r"(\d{2,3}(?:\.\d{1,2})?)\s*円(?:台|中盤|前半|後半)?", outlook_text):
        try:
            val = float(m.group(1))
            if 140.0 <= val <= 170.0:
                snippet = outlook_text[m.start():m.end() + 10]
                if "中盤" in snippet:
                    val += 0.50
                elif "後半" in snippet:
                    val += 0.80
                elif "前半" in snippet:
                    val += 0.20
                levels.append(val)
        except ValueError:
            pass

    levels = sorted(list(set(levels)))
    expected_low = levels[0] if levels else None
    expected_high = levels[-1] if len(levels) >= 2 else (expected_low + 1.0 if expected_low else None)

    # Direction
    last_p = outlook_paras[-1] if outlook_paras else ""
    if any(w in last_p for w in ["方向感を見極め", "見極めたい", "慎重に", "様子見", "方向感を判断", "もみ合い", "レンジ"]):
        direction = "Neutral/Range (保ち合い・見極め)"
    else:
        bullish = sum(outlook_text.count(w) for w in ["上値を試す", "上昇基調", "買い戻し", "一段高", "底堅く"])
        bearish = sum(outlook_text.count(w) for w in ["下値を試す", "下落基調", "上値を抑える", "一段安", "売り圧力", "急落"])
        if bullish > bearish:
            direction = "Bullish (上昇)"
        elif bearish > bullish:
            direction = "Bearish (下落)"
        else:
            direction = "Neutral/Range (保ち合い)"

    return {
        "source": "みんなのFX",
        "horizon": "Daily (日次)",
        "report_date": report_date,
        "analyst": "為替ディーラー",
        "title": title[:40] if title else pdf_path.stem,
        "direction": direction,
        "expected_low": expected_low,
        "expected_high": expected_high,
    }


# ----------------------------------------------------------------------
# 2. Sony Finance (Weekly) Extractor
# ----------------------------------------------------------------------
def extract_sony(pdf_path: Path) -> dict | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception:
        return None

    p0 = reader.pages[0].extract_text() or ""

    # Date
    date_m = re.search(r"issue date\s+(\d{4}/\d{2}/\d{2})", p0)
    if not date_m:
        date_m = re.search(r"(20\d{2})[/-]([01]\d)[/-]([0-3]\d)", p0)
    if not date_m:
        return None

    parts = [int(x) for x in re.findall(r"\d+", date_m.group(0)) if len(x) in (2, 4)]
    if len(parts) >= 3:
        report_date = date(parts[0], parts[1], parts[2])
    else:
        return None

    # Expected Range (e.g. 155.00-158.80円)
    range_m = re.search(r"(\d{3}\.\d{2})\s*[-~～―]\s*(\d{3}\.\d{2})\s*円", p0)
    expected_low, expected_high = None, None
    if range_m:
        expected_low = float(range_m.group(1))
        expected_high = float(range_m.group(2))

    # Analyst
    analyst_m = re.search(r"(石川\s*久美子|尾河\s*眞樹|渡辺\s*浩志|井上\s*哲也)", p0)
    analyst = analyst_m.group(1).replace(" ", "") if analyst_m else "SFGアナリスト"

    # Title / Outlook tone
    title_m = re.search(r"米ドル[：:]([^\n]+)", p0)
    title = title_m.group(1).strip() if title_m else "週刊ドル円見通し"

    if any(w in p0 for w in ["荒い値動き", "見守る", "見極め", "乱高下", "レンジ"]):
        direction = "Neutral/Range (保ち合い・波乱)"
    elif any(w in p0 for w in ["上値追い", "堅調", "上昇余地", "一段高"]):
        direction = "Bullish (上昇)"
    elif any(w in p0 for w in ["下値リスク", "軟調", "調整色", "一段安"]):
        direction = "Bearish (下落)"
    else:
        direction = "Neutral/Range (保ち合い)"

    return {
        "source": "ソニーFG",
        "horizon": "Weekly (週次)",
        "report_date": report_date,
        "analyst": analyst,
        "title": title[:40],
        "direction": direction,
        "expected_low": expected_low,
        "expected_high": expected_high,
    }


# ----------------------------------------------------------------------
# 3. SMBC Trust Bank (Weekly) Extractor
# ----------------------------------------------------------------------
def extract_smbc(pdf_path: Path) -> dict | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception:
        return None

    full_text = "".join([p.extract_text() or "" for p in reader.pages[:8]])

    # Date
    date_m = re.search(r"為替見通し[（\(]米ドル[、,].*?[）\)]\s*(20\d{2})年([01]?\d)月([0-3]?\d)日", full_text)
    if not date_m:
        date_m = re.search(r"(20\d{2})年([01]?\d)月([0-3]?\d)日", full_text)
    if not date_m:
        return None

    report_date = date(int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3)))

    # Range: 今週の予想レンジ ドル円： 156.00 -159.00
    range_m = re.search(r"今週の予想レンジ\s*ドル円[：:]\s*(\d{2,3}(?:\.\d{1,2})?)\s*[-~～―]\s*(\d{2,3}(?:\.\d{1,2})?)", full_text)
    expected_low, expected_high = None, None
    if range_m:
        expected_low = float(range_m.group(1))
        expected_high = float(range_m.group(2))

    # Headline & Tone
    title_m = re.search(r"米ドル[≫>]([^\n]+)", full_text)
    title = title_m.group(1).strip() if title_m else "為替見通し"

    analyst_m = re.search(r"（(二宮|四十内|祖父江|米良)）", full_text)
    analyst = analyst_m.group(1) if analyst_m else "SMBCアナリスト"

    if any(w in title for w in ["レンジ相場", "もみ合い", "見極め", "一服"]):
        direction = "Neutral/Range (保ち合い)"
    elif any(w in title for w in ["上昇", "底堅い", "ドル高"]):
        direction = "Bullish (上昇)"
    elif any(w in title for w in ["下落", "上値重い", "ドル安", "警戒"]):
        direction = "Bearish (軟調)"
    else:
        direction = "Neutral/Range (保ち合い)"

    return {
        "source": "SMBC信託",
        "horizon": "Weekly (週次)",
        "report_date": report_date,
        "analyst": analyst,
        "title": title[:40],
        "direction": direction,
        "expected_low": expected_low,
        "expected_high": expected_high,
    }


# ----------------------------------------------------------------------
# Main Verification Engine
# ----------------------------------------------------------------------
def main():
    print("=" * 90)
    print("      MULTI-INSTITUTION REPORT FORECAST & GROUND TRUTH VERIFICATION ENGINE")
    print("=" * 90)

    all_forecasts = []

    # 1. Min-FX (Last 10 daily reports)
    minfx_dir = BASE_DIR / "みんなのFX"
    if minfx_dir.exists():
        for f in sorted(minfx_dir.glob("daily_report_*.pdf"))[-10:]:
            fc = extract_minfx(f)
            if fc and fc.get("expected_low"):
                all_forecasts.append(fc)

    # 2. Sony Finance (Last 5 weekly reports)
    sony_dir = BASE_DIR / "SonyFinance"
    if sony_dir.exists():
        for f in sorted(sony_dir.glob("w_*.pdf"))[-6:]:
            fc = extract_sony(f)
            if fc and fc.get("expected_low"):
                all_forecasts.append(fc)

    # 3. SMBC (Last 5 weekly reports)
    smbc_dir = BASE_DIR / "SMBC"
    if smbc_dir.exists():
        for f in sorted(smbc_dir.glob("weekly_*.pdf"))[-6:]:
            fc = extract_smbc(f)
            if fc and fc.get("expected_low"):
                all_forecasts.append(fc)

    if not all_forecasts:
        print("No forecasts extracted.")
        return

    df_fc = pd.DataFrame(all_forecasts).sort_values("report_date")
    min_date = df_fc["report_date"].min() - timedelta(days=3)
    max_date = df_fc["report_date"].max() + timedelta(days=7)

    # Fetch market data
    print(f"\n[1] Fetching USD/JPY OHLC market data ({min_date} to {max_date}) from Yahoo Finance...")
    try:
        yf_df = yf.download("USDJPY=X", start=min_date.isoformat(), end=max_date.isoformat(), progress=False)
        if isinstance(yf_df.columns, pd.MultiIndex):
            yf_df.columns = yf_df.columns.get_level_values(0)
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return

    print(f"[2] Evaluating forecast accuracy against ground-truth performance...\n")

    results = []
    for _, row in df_fc.iterrows():
        r_date = row["report_date"]
        horizon = row["horizon"]
        ts_date = pd.Timestamp(r_date)

        if "Daily" in horizon:
            # Single session evaluation
            avail = yf_df[yf_df.index >= ts_date]
            if avail.empty:
                continue
            actual = avail.iloc[0]
            open_p = float(actual["Open"])
            high_p = float(actual["High"])
            low_p = float(actual["Low"])
            close_p = float(actual["Close"])
        else:
            # Weekly evaluation (5 trading days window)
            avail = yf_df[(yf_df.index >= ts_date) & (yf_df.index <= ts_date + pd.Timedelta(days=7))]
            if avail.empty:
                continue
            open_p = float(avail.iloc[0]["Open"])
            high_p = float(avail["High"].max())
            low_p = float(avail["Low"].min())
            close_p = float(avail.iloc[-1]["Close"])

        change = close_p - open_p
        actual_trend = "Up (上昇)" if change > 0.15 else ("Down (下落)" if change < -0.15 else "Range (保ち合い)")

        # Direction Hit
        pred_dir = row["direction"]
        if ("Bullish" in pred_dir and "Up" in actual_trend) or \
           ("Bearish" in pred_dir and "Down" in actual_trend) or \
           ("Neutral" in pred_dir and "Range" in actual_trend):
            dir_eval = "HIT (的中)"
        elif "Neutral" in pred_dir and abs(change) < 0.5:
            dir_eval = "NEAR (ほぼ的中)"
        else:
            dir_eval = "MISS (外れ)"

        # Range Containment
        exp_low = row["expected_low"]
        exp_high = row["expected_high"]
        tolerance = 0.20
        if low_p >= exp_low - tolerance and high_p <= exp_high + tolerance:
            range_eval = "IN RANGE (収束)"
        elif high_p > exp_high + tolerance:
            range_eval = f"HIGH BREAK (+{high_p - exp_high:.2f})"
        elif low_p < exp_low - tolerance:
            range_eval = f"LOW BREAK (-{exp_low - low_p:.2f})"
        else:
            range_eval = "IN RANGE"

        results.append({
            "機関": row["source"],
            "期間": "日次" if "Daily" in horizon else "週次",
            "日付": r_date.strftime("%Y-%m-%d"),
            "執筆者": row["analyst"],
            "予想方向": pred_dir.split(" ")[0],
            "予想レンジ": f"{exp_low:.2f}〜{exp_high:.2f}",
            "実績高安": f"{low_p:.2f}〜{high_p:.2f}",
            "方向判定": dir_eval,
            "レンジ判定": range_eval,
            "テーマ": row["title"],
        })

    res_df = pd.DataFrame(results)

    cols = ["機関", "期間", "日付", "執筆者", "予想方向", "方向判定", "予想レンジ", "実績高安", "レンジ判定"]
    sub_df = res_df[cols]

    # Markdown Table Output
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    print(header)
    print(sep)
    for _, r in sub_df.iterrows():
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    # Scorecard by Institution
    print("\n" + "=" * 90)
    print("               INSTITUTION / HORIZON SCORECARD")
    print("=" * 90)
    for inst, group in res_df.groupby("機関"):
        tot = len(group)
        hits = len(group[group["方向判定"].str.contains("HIT|NEAR")])
        in_rng = len(group[group["レンジ判定"].str.contains("IN RANGE")])
        print(f"[{inst}] 件数: {tot}件 | 方向的中率: {hits}/{tot} ({hits/tot*100:.1f}%) | レンジ収束率: {in_rng}/{tot} ({in_rng/tot*100:.1f}%)")

    tot_all = len(res_df)
    hits_all = len(res_df[res_df["方向判定"].str.contains("HIT|NEAR")])
    in_rng_all = len(res_df[res_df["レンジ判定"].str.contains("IN RANGE")])
    print("-" * 90)
    print(f"【全体総計】 検証サンプル数: {tot_all} 件")
    print(f" 方向性的中率 (HIT+NEAR): {hits_all} / {tot_all} ({hits_all / tot_all * 100:.1f}%)")
    print(f" レンジ収束率 (IN RANGE): {in_rng_all} / {tot_all} ({in_rng_all / tot_all * 100:.1f}%)")
    print("=" * 90)


if __name__ == "__main__":
    main()
