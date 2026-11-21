"""
Train LightGBM Classifier to predict Rebound vs Breakout at boundary touches.
Evaluates model using TimeSeriesSplit and backtests filtered trade outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss, log_loss

backend_dir = Path(__file__).resolve().parent.parent
data_path = backend_dir / "artifacts" / "datasets" / "rebound_events.parquet"
output_dir = backend_dir / "artifacts" / "models" / "rebound_classifier"
output_dir.mkdir(parents=True, exist_ok=True)


def train_and_evaluate():
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Encode categorical features
    df["is_buy"] = (df["direction"] == "buy").astype(int)

    feature_cols = [
        "is_buy",
        "expected_range_pips",
        "dist_to_boundary_pips",
        "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24",
        "atr_14", "atr_ratio", "rsi_14", "bb_width", "bb_pct",
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "hour", "dayofweek", "is_asian", "is_london", "is_ny"
    ]

    X = df[feature_cols].copy()
    y = df["target_rebound"].values
    pnl = df["pnl"].values

    print(f"Features: {len(feature_cols)}, Total Samples: {len(X)}")

    # TimeSeriesSplit (5 splits)
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_preds = np.zeros(len(df))
    oof_indices = []

    models = []
    feature_importances = np.zeros(len(feature_cols))

    print("\n--- Running TimeSeriesSplit Cross Validation ---")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        oof_indices.extend(val_idx)

        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold+1}: Train size={len(train_idx)}, Val size={len(val_idx)}, AUC={fold_auc:.4f}")

        feature_importances += model.feature_importances_ / n_splits
        models.append(model)

    # Evaluate OOF on the evaluated portion (all validation splits)
    oof_idx_arr = np.array(oof_indices)
    y_eval = y[oof_idx_arr]
    preds_eval = oof_preds[oof_idx_arr]
    pnl_eval = pnl[oof_idx_arr]
    df_eval = df.iloc[oof_idx_arr].copy()
    df_eval["pred_prob"] = preds_eval

    overall_auc = roc_auc_score(y_eval, preds_eval)
    brier = brier_score_loss(y_eval, preds_eval)
    loss = log_loss(y_eval, preds_eval)

    print(f"\n==========================================")
    print(f"Overall OOF Evaluation ({len(y_eval)} samples):")
    print(f"  AUC-ROC:     {overall_auc:.4f}")
    print(f"  Log Loss:    {loss:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"==========================================")

    # Feature Importance
    df_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importances
    }).sort_values("importance", ascending=False)
    print("\nTop 10 Important Features:")
    print(df_importance.head(10).to_string(index=False))

    # Threshold Simulation
    print("\n--- Trade Performance by Probability Threshold ---")
    print(f"{'Threshold':>10} | {'Trades':>6} | {'Selected%':>9} | {'Win Rate':>8} | {'Gross Profit':>12} | {'Gross Loss':>12} | {'Total PnL':>12} | {'PF':>6}")
    print("-" * 88)

    threshold_results = []
    # Baseline (all trades in eval period)
    base_wins = pnl_eval[pnl_eval > 0]
    base_losses = pnl_eval[pnl_eval < 0]
    base_gp = base_wins.sum() if len(base_wins) > 0 else 0
    base_gl = abs(base_losses.sum()) if len(base_losses) > 0 else 0
    base_pf = (base_gp / base_gl) if base_gl > 0 else 99.0
    print(f"{'None (Base)':>10} | {len(pnl_eval):>6} | {'100.0%':>9} | {(y_eval.mean()*100):>7.1f}% | {base_gp:>12,.0f} | {base_gl:>12,.0f} | {pnl_eval.sum():>12,.0f} | {base_pf:>6.2f}")

    for th in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
        mask = preds_eval >= th
        sel_count = int(mask.sum())
        if sel_count == 0:
            continue
        sel_y = y_eval[mask]
        sel_pnl = pnl_eval[mask]
        win_rate = sel_y.mean()
        wins = sel_pnl[sel_pnl > 0]
        losses = sel_pnl[sel_pnl < 0]
        gp = wins.sum() if len(wins) > 0 else 0
        gl = abs(losses.sum()) if len(losses) > 0 else 0
        pf = (gp / gl) if gl > 0 else 99.0
        tot_pnl = sel_pnl.sum()
        pct = (sel_count / len(pnl_eval)) * 100

        print(f"{th:>10.2f} | {sel_count:>6} | {pct:>8.1f}% | {(win_rate*100):>7.1f}% | {gp:>12,.0f} | {gl:>12,.0f} | {tot_pnl:>12,.0f} | {pf:>6.2f}")
        threshold_results.append({
            "threshold": th,
            "trades": sel_count,
            "selected_pct": pct,
            "win_rate": float(win_rate),
            "profit_factor": float(pf),
            "total_pnl": float(tot_pnl),
            "gross_profit": float(gp),
            "gross_loss": float(gl)
        })

    # Train final model on full dataset
    final_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    final_model.fit(X, y)

    # Save artifacts
    model_file = output_dir / "lgbm_rebound_model.txt"
    final_model.booster_.save_model(str(model_file))
    print(f"\nSaved final model to {model_file}")

    metadata = {
        "model_name": "BoundReboundClassifier_LGBM",
        "feature_cols": feature_cols,
        "n_samples": len(df),
        "oof_auc": float(overall_auc),
        "oof_log_loss": float(loss),
        "oof_brier_score": float(brier),
        "top_features": df_importance.to_dict(orient="records"),
        "threshold_performance": threshold_results,
    }
    meta_file = output_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {meta_file}")

    return metadata


if __name__ == "__main__":
    train_and_evaluate()
