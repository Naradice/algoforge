"""
Verify whether an ML model can autonomously discover the reversal boundaries
(Support / Resistance levels) without being given any pivot formula.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

backend_dir = Path(r"c:\Users\ksato\workspace\finance\algoforge\backend")
m30_path = backend_dir / "artifacts" / "datasets" / "USDJPY_M30_latest.parquet"
output_dir = backend_dir / "artifacts" / "models" / "autonomous_sr_discovery"
output_dir.mkdir(parents=True, exist_ok=True)


def build_daily_features() -> pd.DataFrame:
    raw_df = pd.read_parquet(m30_path)
    if not isinstance(raw_df.index, pd.DatetimeIndex) and "time" in raw_df.columns:
        raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
        raw_df.set_index("time", inplace=True)
    raw_df.rename(columns={c: str(c).lower().strip() for c in raw_df.columns}, inplace=True)

    raw_df["date_key"] = raw_df.index.date
    daily = raw_df.groupby("date_key").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).sort_index()

    # Targets for day T:
    # How far does price go down from Day Open to Day Low (Support Drop in pips)
    # How far does price go up from Day Open to Day High (Resistance Rise in pips)
    # And Classic Pivot S1/R1 distance for comparison
    daily["target_low_drop_pips"] = (daily["open"] - daily["low"]) / 0.01
    daily["target_high_rise_pips"] = (daily["high"] - daily["open"]) / 0.01

    # Classic S1 / R1 distances from today's Open
    prev_H = daily["high"].shift(1)
    prev_L = daily["low"].shift(1)
    prev_C = daily["close"].shift(1)
    prev_P = (prev_H + prev_L + prev_C) / 3.0
    s1 = 2 * prev_P - prev_H
    r1 = 2 * prev_P - prev_L
    daily["classic_s1_drop_pips"] = (daily["open"] - s1) / 0.01
    daily["classic_r1_rise_pips"] = (r1 - daily["open"]) / 0.01

    # Input Features available at 00:00 UTC (Yesterday's and historical data ONLY)
    # We do NOT pass P, S1, or R1! Only raw past price action.
    daily["prev_open"] = daily["open"].shift(1)
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily["prev_close"] = daily["close"].shift(1)
    daily["prev_range_pips"] = (daily["prev_high"] - daily["prev_low"]) / 0.01
    daily["prev_body_pips"] = (daily["prev_close"] - daily["prev_open"]) / 0.01
    daily["prev_upper_wick_pips"] = (daily["prev_high"] - np.maximum(daily["prev_open"], daily["prev_close"])) / 0.01
    daily["prev_lower_wick_pips"] = (np.minimum(daily["prev_open"], daily["prev_close"]) - daily["prev_low"]) / 0.01

    # Rolling ranges & ATR
    daily["range_ma3"] = daily["prev_range_pips"].rolling(3).mean()
    daily["range_ma5"] = daily["prev_range_pips"].rolling(5).mean()
    daily["range_ma10"] = daily["prev_range_pips"].rolling(10).mean()
    daily["range_ma20"] = daily["prev_range_pips"].rolling(20).mean()
    daily["range_std10"] = daily["prev_range_pips"].rolling(10).std()

    # Distance between today's open and yesterday's close/high/low
    daily["dist_open_prev_close"] = (daily["open"] - daily["prev_close"]) / 0.01
    daily["dist_open_prev_high"] = (daily["open"] - daily["prev_high"]) / 0.01
    daily["dist_open_prev_low"] = (daily["open"] - daily["prev_low"]) / 0.01

    daily["dayofweek"] = pd.to_datetime(daily.index).dayofweek
    daily["month"] = pd.to_datetime(daily.index).month

    df_clean = daily.dropna().copy()
    return df_clean


def train_extrema_models():
    print("Building dataset for ML Support/Resistance discovery...")
    df = build_daily_features()
    print(f"Total days: {len(df)} (from {df.index.min()} to {df.index.max()})")

    features = [
        "prev_range_pips", "prev_body_pips", "prev_upper_wick_pips", "prev_lower_wick_pips",
        "range_ma3", "range_ma5", "range_ma10", "range_ma20", "range_std10",
        "dist_open_prev_close", "dist_open_prev_high", "dist_open_prev_low",
        "dayofweek", "month"
    ]

    X = df[features]
    y_low = df["target_low_drop_pips"]
    y_high = df["target_high_rise_pips"]
    classic_s1 = df["classic_s1_drop_pips"].values
    classic_r1 = df["classic_r1_rise_pips"].values

    print(f"Features: {len(features)}")

    # TimeSeriesSplit cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    oof_pred_s = np.zeros(len(df))
    oof_pred_r = np.zeros(len(df))
    oof_indices = []

    importances_s = np.zeros(len(features))
    importances_r = np.zeros(len(features))

    print("\n--- Training LightGBM Models to Predict Day's Support Drop and Resistance Rise ---")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

        # Model for Support (Low drop)
        model_s = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.03, max_depth=3, num_leaves=7,
            min_child_samples=10, random_state=42 + fold, verbose=-1
        )
        model_s.fit(X_train, y_low.iloc[train_idx])
        oof_pred_s[val_idx] = model_s.predict(X_val)
        importances_s += model_s.feature_importances_ / 5.0

        # Model for Resistance (High rise)
        model_r = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.03, max_depth=3, num_leaves=7,
            min_child_samples=10, random_state=42 + fold, verbose=-1
        )
        model_r.fit(X_train, y_high.iloc[train_idx])
        oof_pred_r[val_idx] = model_r.predict(X_val)
        importances_r += model_r.feature_importances_ / 5.0

        oof_indices.extend(val_idx)

    oof_idx_arr = np.array(oof_indices)
    eval_df = df.iloc[oof_idx_arr].copy()
    eval_df["ml_pred_s1_drop"] = oof_pred_s[oof_idx_arr]
    eval_df["ml_pred_r1_rise"] = oof_pred_r[oof_idx_arr]

    # Evaluate correlation between ML predicted S/R and Classic Pivot S1/R1
    corr_s = np.corrcoef(eval_df["ml_pred_s1_drop"], eval_df["classic_s1_drop_pips"])[0, 1]
    corr_r = np.corrcoef(eval_df["ml_pred_r1_rise"], eval_df["classic_r1_rise_pips"])[0, 1]

    mae_ml_s = mean_absolute_error(eval_df["target_low_drop_pips"], eval_df["ml_pred_s1_drop"])
    mae_classic_s = mean_absolute_error(eval_df["target_low_drop_pips"], eval_df["classic_s1_drop_pips"])

    mae_ml_r = mean_absolute_error(eval_df["target_high_rise_pips"], eval_df["ml_pred_r1_rise"])
    mae_classic_r = mean_absolute_error(eval_df["target_high_rise_pips"], eval_df["classic_r1_rise_pips"])

    print("\n==========================================================================================")
    print("DOES ML AUTONOMOUSLY DISCOVER THE PIVOT BOUNDARIES?")
    print("==========================================================================================")
    print(f"Correlation between ML Support and Classic S1:    {corr_s:.4f} ({corr_s*100:.1f}%)")
    print(f"Correlation between ML Resistance and Classic R1: {corr_r:.4f} ({corr_r*100:.1f}%)")
    print("-" * 75)
    print(f"Support Error vs Actual Low:    ML Model MAE = {mae_ml_s:.2f} pips | Classic S1 MAE = {mae_classic_s:.2f} pips")
    print(f"Resistance Error vs Actual High: ML Model MAE = {mae_ml_r:.2f} pips | Classic R1 MAE = {mae_classic_r:.2f} pips")
    print("==========================================================================================")

    # Top features discovered by ML
    df_imp_s = pd.DataFrame({"feature": features, "importance": importances_s}).sort_values("importance", ascending=False)
    print("\nTop Features Discovered by ML for Support (S1):")
    print(df_imp_s.head(5).to_string(index=False))

    df_imp_r = pd.DataFrame({"feature": features, "importance": importances_r}).sort_values("importance", ascending=False)
    print("\nTop Features Discovered by ML for Resistance (R1):")
    print(df_imp_r.head(5).to_string(index=False))

    # Save results
    results = {
        "corr_support_with_classic": float(corr_s),
        "corr_resistance_with_classic": float(corr_r),
        "mae_ml_support": float(mae_ml_s),
        "mae_classic_support": float(mae_classic_s),
        "mae_ml_resistance": float(mae_ml_r),
        "mae_classic_resistance": float(mae_classic_r),
        "top_support_features": df_imp_s.to_dict(orient="records"),
        "top_resistance_features": df_imp_r.to_dict(orient="records"),
    }
    with open(output_dir / "autonomous_sr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    train_extrema_models()
