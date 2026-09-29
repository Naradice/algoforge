"""
Phase 11: scratch neural baselines on activity forecasting -- linear vs tuned HistGB vs tuned MLP
vs Transformer, same inputs / target / split / metric, no pretraining.

Phase 10D/E: the mean log tick count over the next 20 minutes is the only target where a nonlinear
model (HistGB) beats a strong linear baseline by a wide margin (+8-10 %), while an untuned MLP was
worse than linear. Before any pretraining study the question is whether a neural sequence model can
represent what the trees find:
  A  Transformer ~ HistGB  -> synthetic pretraining becomes worth testing
  B  Transformer < HistGB  -> pretraining could at most catch a neural net up to trees
  C  Transformer > HistGB  -> probe / CKA the representation first

Data: dataset 165 (USDJPY Dukascopy ticks -> 1-minute bars by tick_aggregate), the most recent
ROWS rows. Per-bar channels (all models see the same information):
  log_return, log_tick_count, log_spread, imbalance, slog_ofi (sign * log1p|OFI|),
  time of day (4 harmonics) and weekday (Mon-Fri one-hot) of each bar.
Target: mean log tick count over the 20 bars after the anchor (preprocessing future_mean).
Split: whole days, drawn once with SPLIT_SEED -- 70 % train / 10 % val / 20 % test; a window is used
only if all its rows (60 inputs .. 20-bar target) lie on days of one role (OHLCWindowDataset
split_mode="days"; the baselines apply the same rule to the same day lists). Val is used for
early stopping and hyperparameter selection only; every model is scored once on test.

Models:
  linear       ridge on the Phase-10 strong baseline features (multi-scale RMS, every log|r|, signed
               sums, squares/interactions, clock, hour x RMS) + each aux series' 1/5/20/60-bar means
               + hour x aux terms; alpha chosen on val
  hgb          HistGradientBoosting on the tree features (+ aux means); learning rate / leaves /
               min leaf size and the number of iterations chosen on val
  mlp          sklearn MLP on the tree features + the raw 60 x 5 per-bar channels; width / L2 chosen
               on val, then 3 seeds of the chosen setting
  transformer  AlgoForge decoder_only (the Phase 6-9 config) trained as TrainingRuns on the per-bar
               channels, early-stopped on val, 3 seeds

Usage:
  python phase11_activity_scratch.py split      # writes phase11_split_days.json
  python phase11_activity_scratch.py submit     # creates the MLModel, submits 3 Transformer runs
  python phase11_activity_scratch.py evaluate <run_id> <run_id> <run_id>   # writes phase11_activity_scratch.json
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = 165
ROWS = 500_000
OBS, H = 60, 20
SPLIT_SEED = 0
SPLIT_PATH = Path("phase11_split_days.json")
OUT_PATH = Path("phase11_activity_scratch.json")
SEEDS = (0, 1, 2)

CHANNELS = ["log_return", "log_tick_count", "log_spread", "imbalance", "slog_ofi"]
TIME_CHANNELS = [f"tod_{f}{k}" for k in range(1, 5) for f in ("sin", "cos")] + [f"dow_{d}" for d in range(5)]
TARGET = f"future_mean_{H}_log_tick_count"
PREPROCESSING = {"indicators": [
    {"type": "log_return", "column": "close"},
    {"type": "log", "column": "tick_count"},
    {"type": "log", "column": "spread"},
    {"type": "signed_log1p", "column": "ofi"},
    {"type": "time_features", "harmonics": 4},
    {"type": "future_mean", "column": "log_tick_count", "period": H},
]}
MODEL_CONFIG = {"layernorm_mode": "pre", "layerscale_init": 1e-2, "pooling": "last", "d_model": 64,
                "nhead": 4, "num_layers": 4, "dim_feedforward": 256, "dropout": 0.1}


def transformer_hp(seed: int) -> dict:
    return {
        "obs_len": OBS, "pred_len": 1, "feature_cols": CHANNELS + TIME_CHANNELS, "tgt_feature_cols": [TARGET],
        "preprocessing": PREPROCESSING, "normalize": "zscore", "src_normalize": "zscore",
        "normalize_scope": "valid_windows", "require_contiguous": True, "target_lookahead": H - 1,
        "src_lookback": 1, "split_mode": "days", "split_days": json.loads(SPLIT_PATH.read_text()),
        "max_rows": ROWS, "batch_size": 256, "disable_lr_scheduler": True, "seed": seed,
        "max_steps": 30_000, "val_every_steps": 1_000, "early_stop_patience_checks": 5,
    }


# ── data ──────────────────────────────────────────────────────────────────────

def load_frame() -> pd.DataFrame:
    """The dataset exactly as OHLCWindowDataset sees it: sorted, NaN-free, indicators added, then the
    most recent ROWS rows."""
    from model_core.trainers.dataset import OHLCWindowDataset
    from model_core.trainers.preprocessing import apply_preprocessing
    from probe_representations import _dataset_artifact_path

    path = asyncio.run(_dataset_artifact_path(DATASET_ID))
    df, _, _ = OHLCWindowDataset._load_preprocessed_df(path, CHANNELS + TIME_CHANNELS + [TARGET],
                                                       PREPROCESSING, ROWS)
    return df, path


def split_days() -> None:
    df, _ = load_frame()
    days = np.array(sorted({str(d.date()) for d in pd.DatetimeIndex(df.index).normalize()}))
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(days))
    n_test, n_val = int(len(days) * 0.2), int(len(days) * 0.1)
    split = {"test": sorted(days[perm[:n_test]].tolist()), "val": sorted(days[perm[n_test:n_test + n_val]].tolist()),
             "train": sorted(days[perm[n_test + n_val:]].tolist())}
    SPLIT_PATH.write_text(json.dumps(split, indent=0))
    print({k: len(v) for k, v in split.items()}, df.index[0], df.index[-1])


# ── Transformer runs through AlgoForge ─────────────────────────────────────────

async def _submit_all() -> None:
    import database
    from celery_app import enqueue
    from model.models import MLModel
    from model.repository import model_repo

    async with database.async_session_factory() as db:
        model = MLModel(name="decoder_only (phase 11 activity scratch)", architecture="decoder_only", config=MODEL_CONFIG)
        db.add(model)
        await db.commit()
        await db.refresh(model)
        run_ids = []
        for seed in SEEDS:
            run = await model_repo.create_training_run(db, model_id=model.id, dataset_id=DATASET_ID,
                                                       preprocessed_dataset_id=None,
                                                       hyperparams=transformer_hp(seed), execution_target="local")
            await db.commit()
            run_ids.append(run.id)
    for rid in run_ids:
        await enqueue("train_model", rid)
    print(f"MLModel {model.id}, TrainingRuns {run_ids}")


# ── evaluation ────────────────────────────────────────────────────────────────

def _roles(index: pd.DatetimeIndex, split: dict) -> dict[pd.Timestamp, str]:
    return {pd.Timestamp(d): role for role, ds in split.items() for d in ds}


def baseline_windows(df: pd.DataFrame, split: dict):
    """model_core.analysis windows on the same rows, with the dataset's same-role-days rule."""
    from model_core.analysis import build_return_windows
    from model_core.analysis.features import future_aux_mean

    close = df["close"]
    aux = df[["log_tick_count", "log_spread", "imbalance", "slog_ofi"]]
    keep = aux.notna().all(axis=1).to_numpy() & np.isfinite(close.to_numpy())
    close, aux = close[keep], aux[keep]
    w = build_return_windows(close, obs=OBS, horizon=H, aux=aux)
    idx = pd.DatetimeIndex(close.index)
    day = pd.Series(idx.normalize()).map(_roles(idx, split)).to_numpy()
    first, anchor, last = day[w.t - OBS + 1], day[w.t + 1], day[w.t + H + 1]   # rows of r_{t-59} .. bar t+1+H
    role = np.where((first == anchor) & (anchor == last), anchor, None)
    y = future_aux_mean(w, H)                                  # aux column 0 = log_tick_count
    return w, y, role


def features(w):
    from model_core.analysis.features import (
        _window_means, aux_features, aux_hour_interactions, linear_extras, time_features, vol_memory_features,
    )
    base = vol_memory_features(w.X)
    clock = time_features(w.anchor_ts)
    A = aux_features(w)
    F_lin = np.c_[base, linear_extras(w.X, w.anchor_ts), clock, A, aux_hour_interactions(w)]
    F_tree = np.c_[base, clock, A]
    # raw per-bar channels for the MLP: returns and each aux series over the 60 input bars
    k = np.arange(-OBS + 1, 1)
    raw = np.concatenate([w.X] + [w.aux_v[w.t[:, None] + k[None, :], c] for c in range(w.aux_v.shape[1])], axis=1)
    return F_lin, F_tree, raw


def _std(tr, *others):
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return [((a - mu) / sd).astype(np.float32) for a in (tr, *others)]


def transformer_test_predictions(run_ids: list[int], hp_seed0: dict):
    """Best-on-val checkpoint of each run, scored on the dataset's test windows (target units)."""
    import torch
    from model_core.architectures import build_model
    from model.models import TrainingRun
    from model_core.trainers.dataset import OHLCWindowDataset
    import database
    from sqlalchemy import select

    async def runs():
        async with database.async_session_factory() as db:
            rows = (await db.execute(select(TrainingRun).where(TrainingRun.id.in_(run_ids)))).scalars().all()
            return {r.id: (r.status, r.artifact_path, r.hyperparams) for r in rows}

    info = asyncio.run(runs())
    _, path = load_frame()
    hp = hp_seed0
    ds = OHLCWindowDataset(path, obs_len=hp["obs_len"], pred_len=hp["pred_len"], feature_cols=hp["feature_cols"],
                           normalize="zscore", val_split=0.2, preprocessing=hp["preprocessing"], max_rows=hp["max_rows"],
                           tgt_feature_cols=hp["tgt_feature_cols"], src_normalize="zscore", split_mode="days",
                           require_contiguous=True, normalize_scope="valid_windows", target_lookahead=hp["target_lookahead"],
                           src_lookback=1, split_days=hp["split_days"])
    src, tgt, anchor = ds.test_arrays()
    y = tgt[:, 1, 0] * ds.tgt_std[0] + ds.tgt_mean[0]
    store = Path(__import__("os").getenv("ARTIFACT_STORE_PATH", "../artifacts"))
    preds = {}
    for rid in run_ids:
        status, art, rhp = info[rid]
        if status != "completed":
            raise SystemExit(f"TrainingRun {rid} is {status}")
        cfg = {**MODEL_CONFIG, "input_dim": ds.n_features, "output_dim": ds.n_tgt_features,
               "obs_len": OBS, "pred_len": 1, "seq_len": ds.effective_seq_len}
        model = build_model("decoder_only", cfg, device="cpu")
        model.load_state_dict(torch.load(store / art, map_location="cpu")["model_state"])
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(src), 4096):
                s = torch.tensor(src[i:i + 4096])
                t = torch.tensor(tgt[i:i + 4096, :1, :])
                out.append(model(s, t)[:, 0, 0].numpy())
        preds[f"transformer_s{rhp['seed']}"] = np.concatenate(out) * ds.tgt_std[0] + ds.tgt_mean[0]
    return pd.DatetimeIndex(anchor), y, preds


def evaluate(run_ids: list[int]) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from model_core.analysis import paired_block_bootstrap_ci

    t0 = time.time()
    split = json.loads(SPLIT_PATH.read_text())
    df, _ = load_frame()
    w, y, role = baseline_windows(df, split)
    F_lin, F_tree, raw = features(w)
    tr, va, te = [np.flatnonzero(role == r) for r in ("train", "val", "test")]
    print(f"windows train {len(tr)} val {len(va)} test {len(te)} ({time.time() - t0:.0f}s)", flush=True)

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    preds, choice = {}, {}
    # linear
    L_tr, L_va, L_te = _std(F_lin[tr], F_lin[va], F_lin[te])
    best = min(((mse(Ridge(alpha=a).fit(L_tr, y[tr]).predict(L_va), y[va]), a) for a in (0.1, 1, 10, 100, 1000)))
    choice["linear"] = {"alpha": best[1], "val_mse": best[0]}
    preds["linear"] = Ridge(alpha=best[1]).fit(L_tr, y[tr]).predict(L_te)
    print("linear", choice["linear"], flush=True)
    # HistGB: grid, number of iterations picked on val via staged_predict
    grid = []
    for lr in (0.05, 0.1):
        for leaves in (31, 127):
            for msl in (20, 200):
                m = HistGradientBoostingRegressor(max_iter=600, learning_rate=lr, max_leaf_nodes=leaves,
                                                  min_samples_leaf=msl, early_stopping=False, random_state=0)
                m.fit(F_tree[tr], y[tr])
                curve = [mse(p, y[va]) for p in m.staged_predict(F_tree[va])]
                it = int(np.argmin(curve)) + 1
                grid.append((curve[it - 1], lr, leaves, msl, it))
                print("hgb", grid[-1], flush=True)
    v, lr, leaves, msl, it = min(grid)
    choice["hgb"] = {"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": msl, "max_iter": it, "val_mse": v}
    preds["hgb"] = HistGradientBoostingRegressor(max_iter=it, learning_rate=lr, max_leaf_nodes=leaves,
                                                 min_samples_leaf=msl, early_stopping=False, random_state=0) \
        .fit(F_tree[tr], y[tr]).predict(F_tree[te])
    # MLP: grid on seed 0, then 3 seeds of the chosen setting
    M_tr, M_va, M_te = _std(np.c_[F_tree[tr], raw[tr]], np.c_[F_tree[va], raw[va]], np.c_[F_tree[te], raw[te]])
    grid = []
    for hidden in ((256, 128), (512, 256), (512, 256, 128)):
        for alpha in (1e-4, 1e-3):
            m = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=1e-3, batch_size=512,
                             early_stopping=True, max_iter=100, random_state=0).fit(M_tr, y[tr])
            grid.append((mse(m.predict(M_va), y[va]), hidden, alpha))
            print("mlp", grid[-1], flush=True)
    v, hidden, alpha = min(grid)
    choice["mlp"] = {"hidden": list(hidden), "alpha": alpha, "val_mse": v}
    for s in SEEDS:
        preds[f"mlp_s{s}"] = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=1e-3,
                                          batch_size=512, early_stopping=True, max_iter=100, random_state=s) \
            .fit(M_tr, y[tr]).predict(M_te)

    # Transformer test predictions, aligned to the baseline test windows by anchor timestamp
    anchor_t, y_t, tpreds = transformer_test_predictions(run_ids, transformer_hp(SEEDS[0]))
    base_anchor = pd.DatetimeIndex(w.anchor_ts[te])
    tz = anchor_t.tz
    if tz is not None and base_anchor.tz is None:
        anchor_t = anchor_t.tz_convert("UTC").tz_localize(None)
    common = base_anchor.intersection(anchor_t)
    bi = base_anchor.get_indexer(common)
    ti = anchor_t.get_indexer(common)
    yb = y[te][bi]
    gap = float(np.max(np.abs(yb - y_t[ti])))
    print(f"test windows: baselines {len(te)}, transformer {len(anchor_t)}, common {len(common)}; "
          f"max target mismatch {gap:.2e}", flush=True)
    all_preds = {k: p[bi] for k, p in preds.items()} | {k: p[ti] for k, p in tpreds.items()}
    groups = np.asarray(common.normalize().asi8)
    var = float(yb.var())
    loss = {k: (yb - p) ** 2 for k, p in all_preds.items()}
    for fam in ("mlp", "transformer"):                     # seed-mean prediction as an ensemble row
        ks = [k for k in all_preds if k.startswith(fam + "_s")]
        loss[fam + "_mean3"] = (yb - np.mean([all_preds[k] for k in ks], axis=0)) ** 2
    metrics = {}
    for k, l in loss.items():
        metrics[k] = {"mse": float(l.mean()), "r2": 1 - float(l.mean()) / var}
        for ref in ("linear", "hgb"):
            if k != ref:
                lo, hi = paired_block_bootstrap_ci(loss[ref], l, groups, n_boot=500, seed=1)
                metrics[k][f"gain_vs_{ref}"] = float((loss[ref].mean() - l.mean()) / loss[ref].mean())
                metrics[k][f"ci95_vs_{ref}"] = [lo, hi]
    out = {"dataset_id": DATASET_ID, "rows": [str(df.index[0]), str(df.index[-1])], "n_test_common": int(len(common)),
           "target_mismatch_max": gap, "choices": choice, "transformer_runs": run_ids, "metrics": metrics,
           "elapsed_seconds": round(time.time() - t0)}
    OUT_PATH.write_text(json.dumps(out, indent=1))
    for k, m in metrics.items():
        print(f"{k:18s} R2 {m['r2']:+.4f}  vs linear {m.get('gain_vs_linear', 0)*100:+.1f}%  "
              f"vs hgb {m.get('gain_vs_hgb', 0)*100:+.1f}% {m.get('ci95_vs_hgb', '')}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "split":
        split_days()
    elif mode == "submit":
        asyncio.run(_submit_all())
    elif mode == "evaluate":
        evaluate([int(x) for x in sys.argv[2:]])
    else:
        raise SystemExit(__doc__)
