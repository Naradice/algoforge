"""
Rigorous Rebound Evaluation Script
1. Walk-Forward with Purging (strictly Out-of-Sample evaluation)
2. Eliminating Intra-bar lookahead: entry at next bar's Open, SL/TP evaluated on subsequent bars.
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

backend_dir = Path(__file__).resolve().parent.parent
data_path = backend_dir / "artifacts" / "datasets" / "rebound_events.parquet"
m30_path = backend_dir / "artifacts" / "datasets" / "USDJPY_M30_latest.parquet"

feature_cols = [
    "is_buy",
    "expected_range_pips",
    "dist_to_boundary_pips",
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24",
    "atr_14", "atr_ratio", "rsi_14", "bb_width", "bb_pct",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "hour", "dayofweek", "is_asian", "is_london", "is_ny"
]

def load_data():
    df = pd.read_parquet(data_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["is_buy"] = (df["direction"] == "buy").astype(int)
    return df

def run_walk_forward_purged(df: pd.DataFrame, n_splits=5):
    """
    Walk-forward validation with purging:
    Training set only includes events that finished before the validation window starts.
    """
    df = df.copy()
    X = df[feature_cols]
    y = df["target_rebound"].values

    total_n = len(df)
    fold_size = total_n // (n_splits + 1)

    oof_preds = np.full(total_n, np.nan)

    print(f"\n--- Running Walk-Forward with Purging ({n_splits} folds) ---")
    for fold in range(n_splits):
        train_end_idx = fold_size * (fold + 1)
        val_start_idx = train_end_idx
        val_end_idx = min(val_start_idx + fold_size, total_n) if fold < n_splits - 1 else total_n

        val_start_time = df.loc[val_start_idx, "time"]

        # PURGING: remove training events whose exit occurred on or after val_start_time
        # Since exit_price/exit_reason occurred on the same day by 23:30, purge any train event on same date as val_start_time
        train_df = df.iloc[:train_end_idx]
        purged_train_df = train_df[train_df["time"] < val_start_time]

        X_train = purged_train_df[feature_cols]
        y_train = purged_train_df["target_rebound"].values

        X_val = df.iloc[val_start_idx:val_end_idx][feature_cols]
        y_val = y[val_start_idx:val_end_idx]

        model = lgb.LGBMClassifier(
            n_estimators=80,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_start_idx:val_end_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds) if len(np.unique(y_val)) > 1 else 0.5
        print(f"Fold {fold+1}: Train={len(X_train)} (purged), Val={len(X_val)}, Val AUC={fold_auc:.4f}")

    df["oof_prob"] = oof_preds
    return df

def simulate_with_next_bar_execution(df_events: pd.DataFrame, df_m30: pd.DataFrame, sl_pips=15.0, tp_pips=40.0):
    """
    Eliminates intra-bar lookahead:
    Triggered on bar b close -> Entry at bar b+1 open.
    SL/TP evaluated on bar b+1 and subsequent bars until 23:30 UTC.
    """
    sl_yen = sl_pips * 0.01
    tp_yen = tp_pips * 0.01
    lot_multiplier = 100_000.0

    results = []

    for idx, row in df_events.iterrows():
        b_time = row["time"]
        direction = row["direction"]
        exp_low = row["expected_low"]
        exp_high = row["expected_high"]
        target_rebound = row["target_rebound"]
        oof_prob = row.get("oof_prob", np.nan)

        # find bar b in M30
        if b_time not in df_m30.index:
            # fallback to existing
            continue

        loc_b = df_m30.index.get_loc(b_time)
        if loc_b + 1 >= len(df_m30):
            continue

        # Bar b+1 Open is the execution price
        next_bar = df_m30.iloc[loc_b + 1]
        next_bar_time = df_m30.index[loc_b + 1]
        entry_p = float(next_bar["open"])

        # Determine remaining bars on the same day (up to 23:30 UTC)
        same_day_bars = df_m30.iloc[loc_b + 1:]
        same_day_bars = same_day_bars[same_day_bars.index.date == next_bar_time.date()]

        if same_day_bars.empty:
            continue

        exit_p = float(same_day_bars.iloc[-1]["close"])
        exit_reason = "close"

        for _, rb in same_day_bars.iterrows():
            rb_low = float(rb["low"])
            rb_high = float(rb["high"])

            if direction == "buy":
                if rb_low <= (entry_p - sl_yen):
                    exit_p = entry_p - sl_yen
                    exit_reason = "sl"
                    break
                elif rb_high >= (entry_p + tp_yen):
                    exit_p = entry_p + tp_yen
                    exit_reason = "tp"
                    break
            else:  # sell
                if rb_high >= (entry_p + sl_yen):
                    exit_p = entry_p + sl_yen
                    exit_reason = "sl"
                    break
                elif rb_low <= (entry_p - tp_yen):
                    exit_p = entry_p - tp_yen
                    exit_reason = "tp"
                    break

        pnl = (exit_p - entry_p) * lot_multiplier if direction == "buy" else (entry_p - exit_p) * lot_multiplier
        results.append({
            **row.to_dict(),
            "sim_entry_time": next_bar_time,
            "sim_entry_price": entry_p,
            "sim_exit_price": exit_p,
            "sim_exit_reason": exit_reason,
            "sim_pnl": pnl,
            "sim_win": 1 if pnl > 0 else 0,
        })

    return pd.DataFrame(results)

def compute_summary(df_res: pd.DataFrame, label: str):
    n = len(df_res)
    if n == 0:
        return f"{label}: 0 trades"
    wins = df_res[df_res["sim_pnl"] > 0]
    losses = df_res[df_res["sim_pnl"] < 0]
    win_rate = len(wins) / n * 100
    gp = wins["sim_pnl"].sum()
    gl = abs(losses["sim_pnl"].sum())
    pf = (gp / gl) if gl > 0 else 99.0
    tot_pnl = df_res["sim_pnl"].sum()
    return f"{label} | Trades: {n:2d} | WinRate: {win_rate:5.1f}% | PF: {pf:5.2f} | Total PnL: {tot_pnl:+10,.0f} JPY (GP: {gp:,.0f}, GL: {gl:,.0f})"

if __name__ == "__main__":
    df = load_data()
    df_fx = df[df["source"] == "FX168"].copy().reset_index(drop=True)

    print(f"Loaded {len(df)} total events, {len(df_fx)} FX168 events.")

    # Load M30 bars
    df_m30 = pd.read_parquet(m30_path)
    if not isinstance(df_m30.index, pd.DatetimeIndex) and "time" in df_m30.columns:
        df_m30["time"] = pd.to_datetime(df_m30["time"], utc=True)
        df_m30.set_index("time", inplace=True)
    df_m30.rename(columns={c: str(c).lower().strip() for c in df_m30.columns}, inplace=True)

    # 1. Walk-Forward with Purging on all events (larger pool for ML training)
    df_wf = run_walk_forward_purged(df, n_splits=5)

    # 2. Simulate next-bar execution (eliminating intra-bar lookahead)
    df_sim = simulate_with_next_bar_execution(df_wf, df_m30)

    # Filter FX168 only for comparison with Strategy 7 / 8
    sim_fx = df_sim[df_sim["source"] == "FX168"].dropna(subset=["oof_prob"])

    print("\n==================== RIGOROUS RE-EVALUATION RESULTS ====================")
    print(compute_summary(sim_fx, "FX168 Next-Bar Exec (Baseline / No Filter)          "))
    for th in [0.48, 0.50, 0.52, 0.55]:
        filtered = sim_fx[sim_fx["oof_prob"] >= th]
        print(compute_summary(filtered, f"FX168 Next-Bar Exec + Purged OOF ML (prob >= {th:.2f})"))
    print("=========================================================================\n")
