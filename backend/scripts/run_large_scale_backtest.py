"""Large-scale Analyst Report Forecast Backtesting & Strategy Simulation Engine.

Parses over 1,000 historical analyst reports from:
- SMBC Trust Bank Daily (~435 reports)
- SMBC Trust Bank Weekly (~346 reports)
- Min-FX Daily (~289 reports)
- Sony Financial Group Weekly (~46 reports)

Fetches ground-truth USD/JPY historical prices from Yahoo Finance and evaluates:
1. Containment Rate (In-Range vs High/Low Break)
2. Directional Hit Rate (Bullish / Bearish / Neutral)
3. Analyst-by-Analyst Scorecard & Ranking
4. Algorithmic Strategy Backtest:
   - Strategy 1: Range Mean-Reversion (Fade boundaries)
   - Strategy 2: Range Breakout Momentum (Follow breaks)
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf
import yfinance as yf

# UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"L:\data\finance")
OUTPUT_DIR = Path(r"c:\Users\ksato\workspace\finance\algoforge\backend\data\reports_cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Parsers for each institution
# ======================================================================

def parse_smbc_daily(pdf_path: Path) -> dict | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        text = reader.pages[0].extract_text() or ""
    except Exception:
        return None

    # Date
    date_m = re.search(r"(20\d{2})[/年.-]([01]?\d)[/月.-]([0-3]?\d)", text)
    if not date_m:
        fn_m = re.search(r"(\d{4})(\d{2})(\d{2})", pdf_path.stem)
        if fn_m:
            report_date = date(int(fn_m.group(1)), int(fn_m.group(2)), int(fn_m.group(3)))
        else:
            return None
    else:
        report_date = date(int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3)))

    # USD/JPY Range
    range_m = re.search(
        r"USD/JPY\s*[\n\r\s]*(\d{2,3}(?:\.\d{1,2})?)\s*[-~～―]\s*(\d{2,3}(?:\.\d{1,2})?)", text
    )
    if not range_m:
        return None

    try:
        low = float(range_m.group(1))
        high = float(range_m.group(2))
    except ValueError:
        return None

    if low > high or low < 80 or high > 250:
        return None

    # Analyst
    analyst_m = re.search(r"[（\(]\s*(祖父江|二宮|四十内|米良)\s*[）\)]", text)
    analyst = analyst_m.group(1) if analyst_m else "SMBCアナリスト"

    # Tone
    direction = "Neutral/Range"
    if "上昇" in text[:500] and "下落" not in text[:500]:
        direction = "Bullish"
    elif "下落" in text[:500] and "上昇" not in text[:500]:
        direction = "Bearish"

    return {
        "file": pdf_path.name,
        "source": "SMBC信託",
        "horizon": "Daily",
        "date": report_date,
        "analyst": analyst,
        "expected_low": low,
        "expected_high": high,
        "direction": direction,
    }


def parse_smbc_weekly(pdf_path: Path) -> dict | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        # Usually on page 6 or 7
        text = ""
        for p in reader.pages[:8]:
            text += (p.extract_text() or "") + "\n"
    except Exception:
        return None

    # Date
    date_m = re.search(r"為替見通し[（\(]米ドル[、,].*?[）\)]\s*(20\d{2})年([01]?\d)月([0-3]?\d)日", text)
    if not date_m:
        date_m = re.search(r"(20\d{2})年([01]?\d)月([0-3]?\d)日", text)
    if not date_m:
        return None

    report_date = date(int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3)))

    # Range: 今週の予想レンジ ドル円： 156.00 -159.00
    range_m = re.search(
        r"今週の予想レンジ\s*ドル円[：:]\s*(\d{2,3}(?:\.\d{1,2})?)\s*[-~～―]\s*(\d{2,3}(?:\.\d{1,2})?)",
        text,
    )
    if not range_m:
        return None

    try:
        low = float(range_m.group(1))
        high = float(range_m.group(2))
    except ValueError:
        return None

    if low > high or low < 80 or high > 250:
        return None

    analyst_m = re.search(r"（(二宮|四十内|祖父江|米良)）", text)
    analyst = analyst_m.group(1) if analyst_m else "SMBCアナリスト"

    return {
        "file": pdf_path.name,
        "source": "SMBC信託",
        "horizon": "Weekly",
        "date": report_date,
        "analyst": analyst,
        "expected_low": low,
        "expected_high": high,
        "direction": "Neutral/Range",
    }


def parse_minfx(pdf_path: Path) -> dict | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        text = "".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        return None

    # Date
    report_date = None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        m = re.search(r"(20\d{2})[/-]([01]\d)[/-]([0-3]\d)", lines[0])
        if m:
            report_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    if not report_date:
        fn_m = re.search(r"(\d{2})([01]\d)([0-3]\d)", pdf_path.stem)
        if fn_m:
            report_date = date(2000 + int(fn_m.group(1)), int(fn_m.group(2)), int(fn_m.group(3)))

    if not report_date:
        return None

    paras = text.split(" ")
    clean_paras = []
    for p in paras:
        if any(w in p for w in ["想定", "上値", "下値", "目処", "レンジ", "試す"]):
            c = " ".join(p.split())
            if not c.startswith("Copyright") and not c.startswith("-前営業") and len(c) > 30:
                clean_paras.append(c)

    outlook = " ".join(clean_paras)
    levels = []
    for m in re.finditer(r"(\d{2,3}(?:\.\d{1,2})?)\s*円(?:台|中盤|前半|後半)?", outlook):
        try:
            val = float(m.group(1))
            if 100.0 <= val <= 180.0:
                snippet = outlook[m.start():m.end() + 10]
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
    if not levels:
        return None

    low = levels[0]
    high = levels[-1] if len(levels) >= 2 else low + 1.0

    return {
        "file": pdf_path.name,
        "source": "みんなのFX",
        "horizon": "Daily",
        "date": report_date,
        "analyst": "為替ディーラー",
        "expected_low": low,
        "expected_high": high,
        "direction": "Neutral/Range",
    }


def parse_sony_weekly(pdf_path: Path) -> dict | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        p0 = reader.pages[0].extract_text() or ""
    except Exception:
        return None

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

    range_m = re.search(r"(\d{3}\.\d{2})\s*[-~～―]\s*(\d{3}\.\d{2})\s*円", p0)
    if not range_m:
        return None

    low = float(range_m.group(1))
    high = float(range_m.group(2))

    analyst_m = re.search(r"(石川\s*久美子|尾河\s*眞樹|渡辺\s*浩志|井上\s*哲也)", p0)
    analyst = analyst_m.group(1).replace(" ", "") if analyst_m else "SFGアナリスト"

    return {
        "file": pdf_path.name,
        "source": "ソニーFG",
        "horizon": "Weekly",
        "date": report_date,
        "analyst": analyst,
        "expected_low": low,
        "expected_high": high,
        "direction": "Neutral/Range",
    }


# ======================================================================
# Batch Extractor
# ======================================================================

def extract_all_reports() -> pd.DataFrame:
    tasks = []
    # SMBC daily
    smbc_dir = BASE_DIR / "SMBC"
    if smbc_dir.exists():
        for f in smbc_dir.glob("daily_*.pdf"):
            tasks.append((parse_smbc_daily, f))
        for f in smbc_dir.glob("weekly_*.pdf"):
            tasks.append((parse_smbc_weekly, f))

    # Min-FX
    minfx_dir = BASE_DIR / "みんなのFX"
    if minfx_dir.exists():
        for f in minfx_dir.glob("daily_report_*.pdf"):
            tasks.append((parse_minfx, f))

    # SonyFinance weekly
    sony_dir = BASE_DIR / "SonyFinance"
    if sony_dir.exists():
        for f in sony_dir.glob("w_*.pdf"):
            tasks.append((parse_sony_weekly, f))

    print(f"[Phase 1] Scanning and parsing {len(tasks)} candidate PDF reports with thread pool...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(func, path): (func, path) for func, path in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    df = pd.DataFrame(results)
    if not df.empty:
        # Deduplicate identical (source, horizon, date)
        df = df.drop_duplicates(subset=["source", "horizon", "date"]).sort_values("date").reset_index(drop=True)

    print(f"-> Successfully extracted {len(df)} validated forecast records!")
    return df


# ======================================================================
# Ground Truth Matching & Backtest Engine
# ======================================================================

def run_backtest(df_fc: pd.DataFrame):
    if df_fc.empty:
        print("No forecast data.")
        return

    min_date = df_fc["date"].min() - timedelta(days=7)
    max_date = df_fc["date"].max() + timedelta(days=10)

    print(f"\n[Phase 2] Fetching historical USD/JPY OHLC data ({min_date} to {max_date}) from Yahoo Finance...")
    yf_df = yf.download("USDJPY=X", start=min_date.isoformat(), end=max_date.isoformat(), progress=False)
    if isinstance(yf_df.columns, pd.MultiIndex):
        yf_df.columns = yf_df.columns.get_level_values(0)

    print(f"-> Acquired {len(yf_df)} trading days of OHLC market data.")

    print("\n[Phase 3] Matching forecasts with actual price movements & simulating strategies...")

    matched_rows = []
    for _, row in df_fc.iterrows():
        r_date = row["date"]
        horizon = row["horizon"]
        ts_date = pd.Timestamp(r_date)

        exp_low = row["expected_low"]
        exp_high = row["expected_high"]
        range_width = exp_high - exp_low

        if horizon == "Daily":
            avail = yf_df[yf_df.index >= ts_date]
            if avail.empty:
                continue
            act = avail.iloc[0]
            act_date = avail.index[0].date()
            # Only accept if matched day is within 4 days (handling weekends)
            if (act_date - r_date).days > 4:
                continue

            open_p = float(act["Open"])
            high_p = float(act["High"])
            low_p = float(act["Low"])
            close_p = float(act["Close"])
        else:
            # Weekly: Next 5-7 days window
            window = yf_df[(yf_df.index >= ts_date) & (yf_df.index <= ts_date + pd.Timedelta(days=7))]
            if window.empty:
                continue
            open_p = float(window.iloc[0]["Open"])
            high_p = float(window["High"].max())
            low_p = float(window["Low"].min())
            close_p = float(window.iloc[-1]["Close"])

        # Range classification (Tolerance 0.10 JPY)
        tol = 0.10
        high_break = max(0.0, high_p - (exp_high + tol))
        low_break = max(0.0, (exp_low - tol) - low_p)

        if high_break == 0.0 and low_break == 0.0:
            status = "IN_RANGE"
        elif high_break > 0 and low_break == 0:
            status = "HIGH_BREAK"
        elif low_break > 0 and high_break == 0:
            status = "LOW_BREAK"
        else:
            status = "BOTH_BREAK"

        # -------------------------------------------------------------
        # Strategy Simulation (in JPY pips / 100 JPY per standard lot)
        # 1. Mean Reversion (Fade Range Boundaries):
        #    - Buy if Low <= Exp_Low, exit at Close or Exp_High
        #    - Sell if High >= Exp_High, exit at Close or Exp_Low
        # -------------------------------------------------------------
        mr_pnl = 0.0
        mr_trades = 0
        if low_p <= exp_low:
            # Bought at exp_low
            mr_trades += 1
            exit_p = min(exp_high, close_p)
            mr_pnl += (exit_p - exp_low)
        if high_p >= exp_high:
            # Sold at exp_high
            mr_trades += 1
            exit_p = max(exp_low, close_p)
            mr_pnl += (exp_high - exit_p)

        # -------------------------------------------------------------
        # 2. Breakout Momentum:
        #    - If High > Exp_High, Buy at Exp_High, exit at Close
        #    - If Low < Exp_Low, Sell at Exp_Low, exit at Close
        # -------------------------------------------------------------
        bo_pnl = 0.0
        bo_trades = 0
        if high_p > exp_high:
            bo_trades += 1
            bo_pnl += (close_p - exp_high)
        if low_p < exp_low:
            bo_trades += 1
            bo_pnl += (exp_low - close_p)

        matched_rows.append({
            "source": row["source"],
            "horizon": row["horizon"],
            "date": r_date,
            "analyst": row["analyst"],
            "expected_low": exp_low,
            "expected_high": exp_high,
            "range_width": range_width,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "status": status,
            "high_break": high_break,
            "low_break": low_break,
            "mr_pnl": mr_pnl,
            "mr_trades": mr_trades,
            "bo_pnl": bo_pnl,
            "bo_trades": bo_trades,
        })

    res = pd.DataFrame(matched_rows)
    print(f"-> Successfully analyzed {len(res)} verified trading events.\n")

    # =================================================================
    # Print Comprehensive Backtest Statistics
    # =================================================================
    print("=" * 95)
    print("                       LARGE-SCALE BACKTEST RESULTS")
    print("=" * 95)
    print(f"検証全サンプル数: {len(res)} 件")
    print(f"期間: {res['date'].min()} 〜 {res['date'].max()}")
    print("-" * 95)

    # 1. Performance by Category (Source x Horizon)
    cat_summary = []
    for (src, hor), g in res.groupby(["source", "horizon"]):
        n = len(g)
        in_rng = (g["status"] == "IN_RANGE").sum()
        hi_brk = (g["status"] == "HIGH_BREAK").sum()
        lo_brk = (g["status"] == "LOW_BREAK").sum()
        both_brk = (g["status"] == "BOTH_BREAK").sum()
        avg_w = g["range_width"].mean()
        tot_mr_pnl = g["mr_pnl"].sum()
        tot_bo_pnl = g["bo_pnl"].sum()

        cat_summary.append({
            "機関": src,
            "期間": hor,
            "件数": n,
            "平均予想幅(円)": f"{avg_w:.2f}",
            "収束率": f"{in_rng / n * 100:.1f}%",
            "上抜け率": f"{hi_brk / n * 100:.1f}%",
            "下抜け率": f"{lo_brk / n * 100:.1f}%",
            "両抜け率": f"{both_brk / n * 100:.1f}%",
            "逆張り累計(円)": f"{tot_mr_pnl:+.2f}",
            "順張り累計(円)": f"{tot_bo_pnl:+.2f}",
        })

    cat_df = pd.DataFrame(cat_summary)
    print("【1. 機関・期間別 レンジ精度 & 戦略収益性】")
    _print_table(cat_df)

    # 2. Performance by Analyst
    analyst_summary = []
    for analyst, g in res.groupby("analyst"):
        if len(g) < 5:
            continue
        n = len(g)
        in_rng = (g["status"] == "IN_RANGE").sum()
        avg_w = g["range_width"].mean()
        tot_mr = g["mr_pnl"].sum()
        tot_bo = g["bo_pnl"].sum()
        analyst_summary.append({
            "アナリスト": analyst,
            "件数": n,
            "平均予想幅": f"{avg_w:.2f}",
            "収束率": f"{in_rng / n * 100:.1f}%",
            "逆張り損益": f"{tot_mr:+.2f} 円",
            "順張り損益": f"{tot_bo:+.2f} 円",
        })

    an_df = pd.DataFrame(analyst_summary).sort_values("件数", ascending=False)
    print("\n【2. アナリスト別 レンジ信頼度ランキング (5件以上)】")
    _print_table(an_df)

    # 3. Overall Strategy Simulation Statistics
    total_mr = res["mr_pnl"].sum()
    total_bo = res["bo_pnl"].sum()
    mr_wins = (res["mr_pnl"] > 0).sum()
    mr_trades = (res["mr_trades"] > 0).sum()
    bo_wins = (res["bo_pnl"] > 0).sum()
    bo_trades = (res["bo_trades"] > 0).sum()

    print("\n【3. 全体戦略シミュレーション比較】")
    print(f"■ 戦略A (レンジ境界 逆張り/ミーンリバージョン):")
    print(f"   トレード成立日数: {mr_trades} / {len(res)} 日")
    print(f"   勝率: {mr_wins / mr_trades * 100:.1f}%" if mr_trades else "   勝率: N/A")
    print(f"   累計損益: {total_mr:+.2f} 円 (1ロットあたり換算)")
    print(f"■ 戦略B (レンジ上限・下限ブレイク 順張り/モメンタム):")
    print(f"   ブレイク発生日数: {bo_trades} / {len(res)} 日")
    print(f"   勝率: {bo_wins / bo_trades * 100:.1f}%" if bo_trades else "   勝率: N/A")
    print(f"   累計損益: {total_bo:+.2f} 円 (1ロットあたり換算)")
    print("=" * 95)

    # Save detailed CSV for reporting
    res.to_csv(OUTPUT_DIR / "backtest_detailed_results.csv", index=False, encoding="utf-8-sig")
    cat_df.to_csv(OUTPUT_DIR / "category_summary.csv", index=False, encoding="utf-8-sig")
    an_df.to_csv(OUTPUT_DIR / "analyst_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved detailed analysis to {OUTPUT_DIR}")


def _print_table(df: pd.DataFrame):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    print(header)
    print(sep)
    for _, r in df.iterrows():
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    df_forecasts = extract_all_reports()
    run_backtest(df_forecasts)
