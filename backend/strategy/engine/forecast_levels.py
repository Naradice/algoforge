"""
Forecast Levels and Pivot Provider for Strategy Engine.
Retrieves and caches analyst forecasts and classic pivot levels from DB/Parquet.
"""
from __future__ import annotations

from datetime import time
from pathlib import Path
import pandas as pd
from database import SessionLocal
from data.models import ReportForecast


ANALYST_CLOSE_UTC = time(23, 30)

def get_forecast_records(symbol: str = "USDJPY", source: str | None = None) -> pd.DataFrame:
    """Retrieve forecast records from DB (or cached detailed results as fallback)."""
    forecast_rows = []

    # 1. From DB ReportForecast
    try:
        with SessionLocal() as session:
            q = session.query(ReportForecast).filter(
                ReportForecast.symbol == symbol,
                ReportForecast.expected_low.isnot(None),
                ReportForecast.expected_high.isnot(None),
            )
            if source and source.lower() != "all":
                q = q.filter(ReportForecast.source.ilike(f"%{source}%"))
            q = q.order_by(ReportForecast.report_date.asc(), ReportForecast.id.asc())

            for rf in q.all():
                forecast_rows.append({
                    "date": rf.report_date,
                    "expected_low": float(rf.expected_low),
                    "expected_high": float(rf.expected_high),
                    "source": rf.source,
                    "horizon": rf.horizon or "Daily",
                })
    except Exception as e:
        pass

    # 2. From cached CSV if DB was empty
    if not forecast_rows:
        csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "reports_cache" / "backtest_detailed_results.csv"
        if csv_path.exists():
            df_csv = pd.read_csv(csv_path)
            df_csv["date"] = pd.to_datetime(df_csv["date"]).dt.date
            if source and source.lower() != "all":
                df_csv = df_csv[df_csv["source"].str.contains(source, case=False, na=False)]
            for _, r in df_csv.iterrows():
                forecast_rows.append({
                    "date": r["date"],
                    "expected_low": float(r["expected_low"]),
                    "expected_high": float(r["expected_high"]),
                    "source": r.get("source", "SMBC"),
                    "horizon": r.get("horizon", "Daily"),
                })

    if not forecast_rows:
        return pd.DataFrame()

    df_fc = pd.DataFrame(forecast_rows)
    df_fc = df_fc.sort_values("date").drop_duplicates(subset=["date", "source"], keep="last")
    return df_fc


def apply_forecast_indicators(
    df: pd.DataFrame,
    symbol: str = "USDJPY",
    source: str = "FX168",
    tolerance_pips: float = 10.0,
) -> pd.DataFrame:
    """
    Appends 'expected_low', 'expected_high', 'buy_boundary', 'sell_boundary',
    and 'has_forecast' columns to an OHLC DataFrame.
    Matches daily forecast records by date.
    """
    df = df.copy()
    fc_df = get_forecast_records(symbol=symbol, source=source)
    pip_unit = 0.01 if "JPY" in symbol.upper() else 0.0001
    tol_val = float(tolerance_pips) * pip_unit

    if fc_df.empty:
        df["expected_low"] = float("nan")
        df["expected_high"] = float("nan")
        df["buy_boundary"] = float("nan")
        df["sell_boundary"] = float("nan")
        df["has_forecast"] = 0
        return df

    # Map by date
    fc_map = {}
    for _, row in fc_df.iterrows():
        fc_map[row["date"]] = (float(row["expected_low"]), float(row["expected_high"]), str(row["source"]))

    timestamps = pd.to_datetime(df.index, utc=True)
    exp_lows = []
    exp_highs = []
    buy_bounds = []
    sell_bounds = []
    has_fcs = []

    for timestamp in timestamps:
        forecast = fc_map.get(timestamp.date())
        source_name = forecast[2].lower() if forecast else ""
        publication = time(0, 0) if "minna" in source_name or "みんな" in source_name else time(0, 30)
        if forecast and publication <= timestamp.time() < ANALYST_CLOSE_UTC:
            low, high, _ = forecast
            exp_lows.append(low)
            exp_highs.append(high)
            buy_bounds.append(low + tol_val)
            sell_bounds.append(high - tol_val)
            has_fcs.append(1)
        else:
            exp_lows.append(float("nan"))
            exp_highs.append(float("nan"))
            buy_bounds.append(float("nan"))
            sell_bounds.append(float("nan"))
            has_fcs.append(0)

    df["expected_low"] = exp_lows
    df["expected_high"] = exp_highs
    df["buy_boundary"] = buy_bounds
    df["sell_boundary"] = sell_bounds
    df["has_forecast"] = has_fcs
    return df

