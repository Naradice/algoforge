"""
Backtest Dynamic SL/TP scaling based on LightGBM Quantile Range Prediction.
Compares:
1. FX168 Fixed SL 15 / TP 40 (Baseline)
2. FX168 Dynamic SL/TP
3. FX168 Dynamic SL/TP + ML Rebound Filter (prob >= 0.50)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

import sys
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

data_path = backend_dir / "artifacts" / "datasets" / "rebound_events.parquet"
range_model_path = backend_dir / "artifacts" / "models" / "range_quantile" / "lgbm_range_quantile_model.txt"
range_meta_path = backend_dir / "artifacts" / "models" / "range_quantile" / "metadata.json"
rebound_model_path = backend_dir / "artifacts" / "models" / "rebound_classifier" / "lgbm_rebound_model.txt"
rebound_meta_path = backend_dir / "artifacts" / "models" / "rebound_classifier" / "metadata.json"

from research.train_range_quantile_model import build_daily_features


def evaluate_dynamic_sltp():
    # 1. Load Daily Range Predictions
    df_days = build_daily_features()
    with open(range_meta_path, "r", encoding="utf-8") as f:
        meta_range = json.load(f)
    range_booster = lgb.Booster(model_file=str(range_model_path))
    df_days["pred_day_range"] = range_booster.predict(df_days[meta_range["feature_cols"]])
    day_range_map = dict(zip(df_days["date"], df_days["pred_day_range"]))

    # 2. Load Rebound Events
    df_events = pd.read_parquet(data_path)
    df_events["time"] = pd.to_datetime(df_events["time"])
    df_events = df_events.sort_values("time").reset_index(drop=True)
    df_events["is_buy"] = (df_events["direction"] == "buy").astype(int)

    # 3. Load Rebound Classifier
    with open(rebound_meta_path, "r", encoding="utf-8") as f:
        meta_rebound = json.load(f)
    rebound_booster = lgb.Booster(model_file=str(rebound_model_path))
    df_events["pred_rebound_prob"] = rebound_booster.predict(df_events[meta_rebound["feature_cols"]])

    # Map predicted range to events
    df_events["pred_day_range"] = df_events["date"].map(day_range_map).fillna(130.0)

    # Filter FX168
    fx_df = df_events[df_events["source"].str.contains("FX168", case=False, na=False)].copy()

    # Dynamic scaling:
    # Typical predicted range ~ 150 pips.
    # Fixed SL 15 = 10% of 150.
    # Fixed TP 40 = 26.7% of 150.
    # Dynamic: sl_pips = pred_range * 0.10, tp_pips = pred_range * 0.27
    fx_df["dyn_sl_pips"] = np.clip(fx_df["pred_day_range"] * 0.10, 12.0, 25.0)
    fx_df["dyn_tp_pips"] = np.clip(fx_df["pred_day_range"] * 0.27, 30.0, 60.0)

    print(f"FX168 events: {len(fx_df)}")
    print(f"Dynamic SL range: {fx_df['dyn_sl_pips'].min():.1f} to {fx_df['dyn_sl_pips'].max():.1f} pips (mean {fx_df['dyn_sl_pips'].mean():.1f})")
    print(f"Dynamic TP range: {fx_df['dyn_tp_pips'].min():.1f} to {fx_df['dyn_tp_pips'].max():.1f} pips (mean {fx_df['dyn_tp_pips'].mean():.1f})")

    # Metrics calculator
    starting_capital = 1_000_000.0
    def calc_metrics(sub_df: pd.DataFrame, label: str):
        pnl = sub_df["pnl"].values
        trades_count = len(pnl)
        if trades_count == 0:
            return {"name": label, "trades": 0, "win_rate": 0, "pf": 0, "total_pnl": 0, "max_dd_pct": 0, "sharpe": 0}
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = len(wins) / trades_count
        gp = float(wins.sum()) if len(wins) > 0 else 0
        gl = float(abs(losses.sum())) if len(losses) > 0 else 0
        pf = (gp / gl) if gl > 0 else 99.0

        cum_equity = starting_capital + np.cumsum(pnl)
        peak = np.maximum.accumulate(cum_equity)
        dd = (peak - cum_equity) / peak
        max_dd = float(np.max(dd)) * 100 if len(dd) > 0 else 0.0

        returns = pnl / starting_capital
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) if len(returns) > 1 else 0.0
        sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0

        return {
            "name": label,
            "trades": trades_count,
            "win_rate": round(win_rate * 100, 1),
            "pf": round(pf, 2),
            "total_pnl": round(float(pnl.sum())),
            "max_dd_pct": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
        }

    results = []
    # 1. Base (Fixed SL 15 / TP 40)
    results.append(calc_metrics(fx_df, "1. FX168 Fixed SL15/TP40 (Baseline)"))

    # 2. Rebound ML Filter only (Fixed SL 15 / TP 40)
    sub_ml = fx_df[fx_df["pred_rebound_prob"] >= 0.50]
    results.append(calc_metrics(sub_ml, "2. FX168 ML Rebound Filter (prob >= 0.50)"))

    # 3. Dynamic SL/TP scaling (simulated adjustment)
    # On high volatility days, SL wider prevents whipsaw losses; TP wider captures bigger moves
    # We estimate adjusted PnL based on dynamic SL/TP ratios:
    # Win trades gain (dyn_tp / 40.0), Loss trades lose (dyn_sl / 15.0)
    fx_df["pnl_dyn"] = np.where(
        fx_df["pnl"] > 0,
        fx_df["pnl"] * (fx_df["dyn_tp_pips"] / 40.0),
        fx_df["pnl"] * (fx_df["dyn_sl_pips"] / 15.0)
    )

    fx_dyn_df = fx_df.copy()
    fx_dyn_df["pnl"] = fx_dyn_df["pnl_dyn"]
    results.append(calc_metrics(fx_dyn_df, "3. FX168 Dynamic SL/TP Scaling (Quantile)"))

    # 4. Dynamic SL/TP + ML Rebound Filter
    fx_dyn_ml_df = fx_dyn_df[fx_dyn_df["pred_rebound_prob"] >= 0.50]
    results.append(calc_metrics(fx_dyn_ml_df, "4. FX168 Dynamic SL/TP + ML Rebound Filter (Combined)"))

    res_df = pd.DataFrame(results)
    print("\n==========================================================================================")
    print("BACKTEST COMPARISON: FIXED vs DYNAMIC SL/TP vs ML COMBINED")
    print("==========================================================================================")
    print(res_df.to_string(index=False))
    print("==========================================================================================")

    output_dir = backend_dir / "artifacts" / "models" / "range_quantile"
    output_dir.mkdir(parents=True, exist_ok=True)
    res_path = output_dir / "dynamic_sltp_comparison.json"
    res_df.to_json(res_path, orient="records", indent=2, force_ascii=False)
    print(f"Saved comparison to {res_path}")

    return res_df


if __name__ == "__main__":
    evaluate_dynamic_sltp()
