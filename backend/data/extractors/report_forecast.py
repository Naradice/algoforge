"""Report forecast extraction module.

Parses analyst forecasts (direction, support/resistance, expected price ranges)
from downloaded financial report PDFs (SMBC Trust Bank, Min-FX, Sony Financial Group, etc.)
and persists them into the `report_forecasts` database table.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pypdf
import sqlalchemy as sa
from sqlalchemy.orm import Session

from data.models import ReportForecast

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Individual Parsers
# ----------------------------------------------------------------------

def parse_smbc_daily(pdf_path: Path) -> dict[str, Any] | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        text = reader.pages[0].extract_text() or ""
    except Exception as exc:
        logger.debug("[extractor] failed reading %s: %s", pdf_path.name, exc)
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

    # USD/JPY Range: USD/JPY \n 150.80-152.40 円
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

    analyst_m = re.search(r"[（\(]\s*(祖父江|二宮|四十内|米良)\s*[）\)]", text)
    analyst = analyst_m.group(1) if analyst_m else "SMBCアナリスト"

    direction = "Neutral/Range"
    if "上昇" in text[:500] and "下落" not in text[:500]:
        direction = "Bullish"
    elif "下落" in text[:500] and "上昇" not in text[:500]:
        direction = "Bearish"

    title_m = re.search(r"トップ\s*マーケット\s*情報\s*([^\n\r]+)", text)
    title = title_m.group(1).strip() if title_m else "デイリーマーケットレポート"

    return {
        "source": "SMBC信託",
        "symbol": "USDJPY",
        "horizon": "Daily",
        "report_date": report_date,
        "analyst": analyst,
        "direction": direction,
        "expected_low": low,
        "expected_high": high,
        "title": title[:100],
        "raw_artifact_path": pdf_path.name,
    }


def parse_smbc_weekly(pdf_path: Path) -> dict[str, Any] | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        text = ""
        for p in reader.pages[:8]:
            text += (p.extract_text() or "") + "\n"
    except Exception as exc:
        logger.debug("[extractor] failed reading %s: %s", pdf_path.name, exc)
        return None

    date_m = re.search(r"為替見通し[（\(]米ドル[、,].*?[）\)]\s*(20\d{2})年([01]?\d)月([0-3]?\d)日", text)
    if not date_m:
        date_m = re.search(r"(20\d{2})年([01]?\d)月([0-3]?\d)日", text)
    if not date_m:
        return None

    report_date = date(int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3)))

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

    title_m = re.search(r"米ドル[≫>]([^\n]+)", text)
    title = title_m.group(1).strip() if title_m else "為替見通し"

    direction = "Neutral/Range"
    if any(w in title for w in ["上昇", "ドル高"]):
        direction = "Bullish"
    elif any(w in title for w in ["下落", "上値重い", "ドル安"]):
        direction = "Bearish"

    return {
        "source": "SMBC信託",
        "symbol": "USDJPY",
        "horizon": "Weekly",
        "report_date": report_date,
        "analyst": analyst,
        "direction": direction,
        "expected_low": low,
        "expected_high": high,
        "title": title[:100],
        "raw_artifact_path": pdf_path.name,
    }


def parse_minfx(pdf_path: Path) -> dict[str, Any] | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        text = "".join([p.extract_text() or "" for p in reader.pages])
    except Exception as exc:
        logger.debug("[extractor] failed reading %s: %s", pdf_path.name, exc)
        return None

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    report_date = None
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

    # Title
    title = ""
    for l in lines:
        if l.startswith("-") and not l.startswith("-前営業"):
            title = l.strip("-").strip()
            break

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

    last_p = clean_paras[-1] if clean_paras else ""
    if any(w in last_p for w in ["方向感を見極め", "見極めたい", "慎重に", "様子見", "方向感を判断", "もみ合い", "レンジ"]):
        direction = "Neutral/Range"
    else:
        bullish = sum(outlook.count(w) for w in ["上値を試す", "上昇基調", "買い戻し", "一段高", "底堅く"])
        bearish = sum(outlook.count(w) for w in ["下値を試す", "下落基調", "上値を抑える", "一段安", "売り圧力", "急落"])
        if bullish > bearish:
            direction = "Bullish"
        elif bearish > bullish:
            direction = "Bearish"
        else:
            direction = "Neutral/Range"

    return {
        "source": "みんなのFX",
        "symbol": "USDJPY",
        "horizon": "Daily",
        "report_date": report_date,
        "analyst": "為替ディーラー",
        "direction": direction,
        "expected_low": low,
        "expected_high": high,
        "title": title[:100] if title else pdf_path.stem,
        "raw_artifact_path": pdf_path.name,
    }


def parse_sony_weekly(pdf_path: Path) -> dict[str, Any] | None:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        p0 = reader.pages[0].extract_text() or ""
    except Exception as exc:
        logger.debug("[extractor] failed reading %s: %s", pdf_path.name, exc)
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

    title_m = re.search(r"米ドル[：:]([^\n]+)", p0)
    title = title_m.group(1).strip() if title_m else "週刊ドル円見通し"

    direction = "Neutral/Range"
    if any(w in p0 for w in ["上値追い", "堅調", "上昇余地", "一段高"]):
        direction = "Bullish"
    elif any(w in p0 for w in ["下値リスク", "軟調", "調整色", "一段安"]):
        direction = "Bearish"

    return {
        "source": "ソニーFG",
        "symbol": "USDJPY",
        "horizon": "Weekly",
        "report_date": report_date,
        "analyst": analyst,
        "direction": direction,
        "expected_low": low,
        "expected_high": high,
        "title": title[:100],
        "raw_artifact_path": pdf_path.name,
    }


def parse_uob(pdf_path: Path) -> dict[str, Any] | None:
    """Parse UOB (United Overseas Bank) 'FX Insights' daily report for USD/JPY."""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as exc:
        logger.debug("[extractor] failed reading %s: %s", pdf_path.name, exc)
        return None

    # Date from filename FXI_YYMMDD.pdf or content
    fn_m = re.search(r"FXI_(\d{2})(\d{2})(\d{2})", pdf_path.name, re.I)
    report_date = None
    if fn_m:
        report_date = date(2000 + int(fn_m.group(1)), int(fn_m.group(2)), int(fn_m.group(3)))

    jpy_page = None
    for page in reader.pages:
        txt = page.extract_text() or ""
        if re.search(r"USD/JPY\s*:\s*\d{2,3}\.\d{2}", txt):
            jpy_page = txt
            break

    if not jpy_page:
        return None

    if not report_date:
        # Fallback date parse: Tuesday, 09 September 2025
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        }
        date_m = re.search(r"(?:Monday|Tuesday|Wednesday|Thursday|Friday),\s*([0-3]?\d)\s+([A-Za-z]+)\s+(20\d{2})", jpy_page)
        if date_m:
            d_val = int(date_m.group(1))
            m_val = month_map.get(date_m.group(2).lower())
            y_val = int(date_m.group(3))
            if m_val:
                report_date = date(y_val, m_val, d_val)

    if not report_date:
        return None

    # Support / Resistance levels
    s1_m = re.search(r"S1:\s*([0-9\.]+)", jpy_page)
    s2_m = re.search(r"S2:\s*([0-9\.]+)", jpy_page)
    r1_m = re.search(r"R1:\s*([0-9\.]+)", jpy_page)
    r2_m = re.search(r"R2:\s*([0-9\.]+)", jpy_page)

    s1 = float(s1_m.group(1)) if s1_m else None
    s2 = float(s2_m.group(1)) if s2_m else None
    r1 = float(r1_m.group(1)) if r1_m else None
    r2 = float(r2_m.group(1)) if r2_m else None

    # Expected range
    low = s1 or s2
    high = r1 or r2

    # Check 24-HOUR VIEW text for explicit expected range
    view_24h = ""
    if "24-HOUR VIEW" in jpy_page:
        view_part = jpy_page.split("24-HOUR VIEW")[1]
        if "1-3 WEEKS VIEW" in view_part:
            view_24h = view_part.split("1-3 WEEKS VIEW")[0].strip()
        else:
            view_24h = view_part[:400].strip()

    range_m = re.search(
        r"(?:range\s+(?:between|of)?\s*)(\d{2,3}\.\d{2})\s*(?:and|[-/～~])\s*(\d{2,3}\.\d{2})",
        view_24h,
        re.I,
    )
    if range_m:
        m_low, m_high = float(range_m.group(1)), float(range_m.group(2))
        if m_low < m_high and 80 < m_low < 250:
            low, high = m_low, m_high

    if not low or not high or low >= high or low < 80 or high > 250:
        return None

    # Direction bias
    direction = "Neutral/Range"
    view_lower = view_24h.lower()
    if any(k in view_lower for k in ["upside", "upward bias", "bullish", "advance", "rebound"]):
        direction = "Bullish"
    elif any(k in view_lower for k in ["downside", "downward bias", "bearish", "decline", "pullback"]):
        direction = "Bearish"

    # Title extraction
    title = "UOB FX Insights - USD/JPY 24-Hour View"
    if view_24h:
        first_sentence = view_24h.split(".")[0].strip()
        if len(first_sentence) > 10:
            title = first_sentence[:100]

    return {
        "source": "UOB",
        "symbol": "USDJPY",
        "horizon": "Daily",
        "report_date": report_date,
        "analyst": "Quek Ser Leang",
        "direction": direction,
        "expected_low": low,
        "expected_high": high,
        "title": title,
        "raw_artifact_path": pdf_path.name,
    }


def parse_fx168(file_path: Path) -> list[dict[str, Any]]:
    """Parse FX168 daily pivot table (HTML or JSON) for USD/JPY and major pairs."""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    pivot_json = None

    if content.strip().startswith("{") and "pivotData" in content:
        try:
            import json
            pivot_json = json.loads(content)
        except Exception:
            pass

    if not pivot_json:
        m = re.search(r'\{"props":\{.*?"pivotData":\{.*\}\}\}', content)
        if m:
            try:
                import json
                pivot_json = json.loads(m.group(0))
            except Exception:
                pass

    if not pivot_json:
        try:
            from bs4 import BeautifulSoup
            import json
            soup = BeautifulSoup(content, "html.parser")
            for s in soup.find_all("script"):
                if s.string and "pivotData" in s.string:
                    try:
                        pivot_json = json.loads(s.string)
                        break
                    except Exception:
                        pass
        except Exception:
            pass

    if not pivot_json:
        return []

    data_block = pivot_json.get("props", {}).get("pageProps", {}).get("data", {})
    pivot_data = data_block.get("pivotData", {})
    if not pivot_data and "pivotData" in pivot_json:
        pivot_data = pivot_json["pivotData"]

    fn_m = re.search(r"(\d{4})(\d{2})(\d{2})", file_path.name)
    if fn_m:
        report_date = date(int(fn_m.group(1)), int(fn_m.group(2)), int(fn_m.group(3)))
    else:
        report_date = date.today()

    symbol_map = {
        "FESJPY": "USDJPY",
        "FESEUR": "EURUSD",
        "FESGBP": "GBPUSD",
        "FESAUD": "AUDUSD",
        "FESCAD": "USDCAD",
        "FESCHF": "USDCHF",
    }

    results = []
    for code, sym in symbol_map.items():
        sym_pivots = pivot_data.get(code)
        if not sym_pivots:
            continue

        classic = sym_pivots.get("classic", {})
        try:
            p_val = float(classic.get("p", 0))
            s1_val = float(classic.get("s1", 0))
            r1_val = float(classic.get("r1", 0))
        except (ValueError, TypeError):
            continue

        if s1_val <= 0 or r1_val <= 0 or s1_val >= r1_val:
            continue

        results.append({
            "source": "FX168",
            "symbol": sym,
            "horizon": "Daily",
            "report_date": report_date,
            "analyst": "FX168 Pivot (Classic)",
            "direction": "Neutral/Range",
            "expected_low": s1_val,
            "expected_high": r1_val,
            "title": f"FX168 Pivot Classic: P={p_val}, S1={s1_val}, R1={r1_val}",
            "raw_artifact_path": file_path.name,
        })

    return results


# ----------------------------------------------------------------------
# Universal Extractor Router
# ----------------------------------------------------------------------

def extract_forecast_from_pdf(pdf_path: Path) -> dict[str, Any] | None:
    """Detect institution from path/filename and extract forecast metadata."""
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return None

    parent_name = pdf_path.parent.name
    fn = pdf_path.name

    if "SMBC" in parent_name or fn.startswith("daily_") or fn.startswith("weekly_"):
        if fn.startswith("daily_"):
            return parse_smbc_daily(pdf_path)
        elif fn.startswith("weekly_"):
            return parse_smbc_weekly(pdf_path)

    if "みんなのFX" in parent_name or "daily_report_" in fn:
        return parse_minfx(pdf_path)

    if "SonyFinance" in parent_name or fn.startswith("w_"):
        if fn.startswith("w_"):
            return parse_sony_weekly(pdf_path)

    if "UOB" in parent_name or fn.startswith("FXI_"):
        return parse_uob(pdf_path)

    return None


def extract_forecast_from_file(file_path: Path) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Universal dispatcher accepting PDF, HTML, or JSON files."""
    if not file_path.is_file():
        return None

    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return extract_forecast_from_pdf(file_path)

    parent_name = file_path.parent.name
    fn = file_path.name

    if "FX168" in parent_name or "pivot" in fn.lower():
        return parse_fx168(file_path)

    return None


# ----------------------------------------------------------------------
# DB Persistence & Sync
# ----------------------------------------------------------------------

def save_report_forecast(
    session: Session,
    forecast: dict[str, Any] | list[dict[str, Any]],
    datasource_id: int | None = None,
) -> ReportForecast | list[ReportForecast] | None:
    """Upsert an extracted report forecast (or list of forecasts) into `report_forecasts` table."""
    if isinstance(forecast, list):
        saved = []
        for item in forecast:
            res = save_report_forecast(session, item, datasource_id=datasource_id)
            if res:
                saved.append(res)
        return saved

    source = forecast.get("source")
    symbol = forecast.get("symbol", "USDJPY")
    horizon = forecast.get("horizon", "Daily")
    report_date = forecast.get("report_date")

    if not source or not report_date:
        return None

    stmt = sa.select(ReportForecast).where(
        ReportForecast.source == source,
        ReportForecast.symbol == symbol,
        ReportForecast.horizon == horizon,
        ReportForecast.report_date == report_date,
    )
    existing = session.execute(stmt).scalar_one_or_none()

    if existing is not None:
        # Update existing record
        existing.datasource_id = datasource_id or existing.datasource_id
        existing.analyst = forecast.get("analyst", existing.analyst)
        existing.direction = forecast.get("direction", existing.direction)
        existing.expected_low = forecast.get("expected_low", existing.expected_low)
        existing.expected_high = forecast.get("expected_high", existing.expected_high)
        existing.title = forecast.get("title", existing.title)
        existing.raw_artifact_path = forecast.get("raw_artifact_path", existing.raw_artifact_path)
        session.flush()
        return existing

    rf = ReportForecast(
        datasource_id=datasource_id,
        source=source,
        symbol=symbol,
        horizon=horizon,
        report_date=report_date,
        analyst=forecast.get("analyst"),
        direction=forecast.get("direction"),
        expected_low=forecast.get("expected_low"),
        expected_high=forecast.get("expected_high"),
        title=forecast.get("title"),
        raw_artifact_path=forecast.get("raw_artifact_path"),
    )
    session.add(rf)
    session.flush()
    return rf
