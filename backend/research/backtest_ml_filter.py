"""
Backtest Analyst Range Reversion with BoundReboundClassifier ML filter.
Compares:
1. Strategy 7 Baseline (FX168, SL 15pips, TP 40pips, Tol 10pips, No ML)
2. Strategy 7 + ML Filter (Threshold 0.50)
3. Strategy 7 + ML Filter (Threshold 0.55)
4. Strategy 6 Baseline (UOB, SL 15pips, TP 40pips, Tol 10pips, No ML)
5. Strategy 6 + ML Filter (Threshold 0.50)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

backend_dir = Path(__file__).resolve().parent.parent
data_path = backend_dir / "artifacts" / "datasets" / "rebound_events.parquet"
model_path = backend_dir / "artifacts" / "models" / "rebound_classifier" / "lgbm_rebound_model.txt"
meta_path = backend_dir / "artifacts" / "models" / "rebound_classifier" / "metadata.json"


def evaluate_backtests():
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    df = pd.read_parquet(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["is_buy"] = (df["direction"] == "buy").astype(int)

    booster = lgb.Booster(model_file=str(model_path))
    X = df[feature_cols].copy()
    df["pred_prob"] = booster.predict(X)

    starting_capital = 1_000_000.0

    def calc_run_metrics(sub_df: pd.DataFrame, label: str):
        if sub_df.empty:
            return {"name": label, "trades": 0, "win_rate": 0, "pf": 0, "total_pnl": 0, "max_dd": 0, "sharpe": 0}

        pnl = sub_df["pnl"].values
        trades_count = len(pnl)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = len(wins) / trades_count if trades_count > 0 else 0.0
        gp = float(wins.sum()) if len(wins) > 0 else 0.0
        gl = float(abs(losses.sum())) if len(losses) > 0 else 0.0
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
            "loss_reduction_pct": 0.0,
        }

    results = []

    # --- FX168 Evaluation ---
    fx168_df = df[df["source"].str.contains("FX168", case=False, na=False)].copy()
    m_base_fx = calc_run_metrics(fx168_df, "FX168 (Strategy 7) Baseline")
    results.append(m_base_fx)

    for th in [0.48, 0.50, 0.52, 0.55]:
        sub = fx168_df[fx168_df["pred_prob"] >= th]
        m = calc_run_metrics(sub, f"FX168 + ML Filter (prob >= {th})")
        # Loss reduction compared to base
        base_gl = abs(fx168_df[fx168_df["pnl"] < 0]["pnl"].sum())
        curr_gl = abs(sub[sub["pnl"] < 0]["pnl"].sum()) if not sub.empty else 0
        m["loss_reduction_pct"] = round((1 - curr_gl / base_gl) * 100, 1) if base_gl > 0 else 0
        results.append(m)

    # --- UOB Evaluation ---
    uob_df = df[df["source"].str.contains("UOB", case=False, na=False)].copy()
    m_base_uob = calc_run_metrics(uob_df, "UOB (Strategy 6) Baseline")
    results.append(m_base_uob)

    for th in [0.48, 0.50, 0.52, 0.55]:
        sub = uob_df[uob_df["pred_prob"] >= th]
        m = calc_run_metrics(sub, f"UOB + ML Filter (prob >= {th})")
        base_gl = abs(uob_df[uob_df["pnl"] < 0]["pnl"].sum())
        curr_gl = abs(sub[sub["pnl"] < 0]["pnl"].sum()) if not sub.empty else 0
        m["loss_reduction_pct"] = round((1 - curr_gl / base_gl) * 100, 1) if base_gl > 0 else 0
        results.append(m)

    # --- Combined Portfolio (FX168 + UOB + SMBC + みんなのFX) ---
    m_all_base = calc_run_metrics(df, "All Sources Baseline")
    results.append(m_all_base)
    for th in [0.50, 0.55]:
        sub = df[df["pred_prob"] >= th]
        m = calc_run_metrics(sub, f"All Sources + ML Filter (prob >= {th})")
        base_gl = abs(df[df["pnl"] < 0]["pnl"].sum())
        curr_gl = abs(sub[sub["pnl"] < 0]["pnl"].sum()) if not sub.empty else 0
        m["loss_reduction_pct"] = round((1 - curr_gl / base_gl) * 100, 1) if base_gl > 0 else 0
        results.append(m)

    res_df = pd.DataFrame(results)
    print("\n==========================================================================================")
    print("BACKTEST COMPARISON: BASELINE vs MACHINE LEARNING BOUND-REBOUND FILTER")
    print("==========================================================================================")
    print(res_df.to_string(index=False))
    print("==========================================================================================")

    res_path = backend_dir / "artifacts" / "models" / "rebound_classifier" / "backtest_comparison.json"
    res_df.to_json(res_path, orient="records", indent=2, force_ascii=False)
    print(f"Saved comparison to {res_path}")

    return res_df


if __name__ == "__main__":
    evaluate_backtests()
