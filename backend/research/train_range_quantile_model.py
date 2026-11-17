"""
Train LightGBM Quantile Regressor (alpha=0.80) to predict Daily High-Low Range in pips.
Applies dynamic SL/TP scaling to Analyst Range Reversion strategy and evaluates metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

backend_dir = Path(__file__).resolve().parent.parent
data_path = backend_dir / "artifacts" / "datasets" / "USDJPY_M30_latest.parquet"
output_dir = backend_dir / "artifacts" / "models" / "range_quantile"
output_dir.mkdir(parents=True, exist_ok=True)


def build_daily_features() -> pd.DataFrame:
    raw_df = pd.read_parquet(data_path)
    if not isinstance(raw_df.index, pd.DatetimeIndex) and "time" in raw_df.columns:
        raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
        raw_df.set_index("time", inplace=True)
    raw_df.rename(columns={c: str(c).lower().strip() for c in raw_df.columns}, inplace=True)

    raw_df["date_key"] = raw_df.index.date
    raw_df["hour"] = raw_df.index.hour

    # Group by day
    days = []
    for d, g in raw_df.groupby("date_key"):
        if len(g) < 20:  # Skip incomplete days
            continue
        d_open = float(g.iloc[0]["open"])
        d_high = float(g["high"].max())
        d_low = float(g["low"].min())
        d_close = float(g.iloc[-1]["close"])
        d_range = (d_high - d_low) / 0.01

        # Asian session range (00:00 - 06:00 UTC)
        asia = g[g["hour"] < 6]
        if not asia.empty:
            asia_range = (float(asia["high"].max()) - float(asia["low"].min())) / 0.01
        else:
            asia_range = d_range * 0.3

        days.append({
            "date": d,
            "open": d_open,
            "high": d_high,
            "low": d_low,
            "close": d_close,
            "day_range_pips": d_range,
            "asia_range_pips": asia_range,
            "dayofweek": pd.to_datetime(d).dayofweek,
            "month": pd.to_datetime(d).month,
        })

    df_d = pd.DataFrame(days).sort_values("date").reset_index(drop=True)

    # Rolling historical features (strictly shift(1) to avoid lookahead)
    r = df_d["day_range_pips"]
    df_d["range_lag1"] = r.shift(1)
    df_d["range_lag2"] = r.shift(2)
    df_d["range_ma3"] = r.shift(1).rolling(3).mean()
    df_d["range_ma5"] = r.shift(1).rolling(5).mean()
    df_d["range_ma10"] = r.shift(1).rolling(10).mean()
    df_d["range_ma20"] = r.shift(1).rolling(20).mean()
    df_d["range_std10"] = r.shift(1).rolling(10).std()

    # Returns
    c = df_d["close"]
    df_d["ret_1d"] = c.shift(1).pct_change(1)
    df_d["ret_5d"] = c.shift(1).pct_change(5)

    df_d = df_d.dropna().reset_index(drop=True)
    return df_d


def train_and_backtest():
    print("Building daily range dataset...")
    df = build_daily_features()
    print(f"Total days: {len(df)}, Date range: {df['date'].min()} to {df['date'].max()}")

    feature_cols = [
        "asia_range_pips",
        "range_lag1", "range_lag2",
        "range_ma3", "range_ma5", "range_ma10", "range_ma20",
        "range_std10", "ret_1d", "ret_5d",
        "dayofweek", "month"
    ]

    X = df[feature_cols].copy()
    y = df["day_range_pips"].values

    # Train LightGBM Quantile Regressor (alpha=0.80: 80th percentile)
    alpha = 0.80
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(df))
    oof_indices = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=10,
            random_state=42 + fold,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        oof_indices.extend(val_idx)

        cov = np.mean(y_val <= preds)
        mae = np.mean(np.abs(y_val - preds))
        print(f"Fold {fold+1}: Val size={len(val_idx)}, 80% Coverage={cov*100:.1f}%, MAE={mae:.2f} pips")

    oof_idx_arr = np.array(oof_indices)
    y_eval = y[oof_idx_arr]
    preds_eval = oof_preds[oof_idx_arr]
    total_coverage = np.mean(y_eval <= preds_eval)
    avg_pred_range = np.mean(preds_eval)
    avg_actual_range = np.mean(y_eval)

    print("\n==========================================")
    print("OOF Quantile Model Evaluation:")
    print(f"  Target Alpha:        {alpha*100:.0f}%")
    print(f"  Empirical Coverage:  {total_coverage*100:.1f}% (Actual range <= Predicted range)")
    print(f"  Avg Predicted Range: {avg_pred_range:.2f} pips")
    print(f"  Avg Actual Range:    {avg_actual_range:.2f} pips")
    print("==========================================")

    # Final model on full dataset
    final_model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
    )
    final_model.fit(X, y)
    model_file = output_dir / "lgbm_range_quantile_model.txt"
    final_model.booster_.save_model(str(model_file))
    print(f"Saved model to {model_file}")

    metadata = {
        "model_name": "DailyRangeQuantileRegressor_LGBM",
        "alpha": alpha,
        "feature_cols": feature_cols,
        "empirical_coverage": float(total_coverage),
        "avg_pred_range_pips": float(avg_pred_range),
        "avg_actual_range_pips": float(avg_actual_range),
    }
    meta_file = output_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {meta_file}")

    return metadata


if __name__ == "__main__":
    train_and_backtest()
