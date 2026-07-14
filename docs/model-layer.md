# ML Model Layer

## Concepts

```
PreprocessedDataset (recipe) ─┐
                               ↓
MLModel  →  TrainingRun  →  (celery worker)  →  Checkpoint  →  Deploy  →  Predict
                                                    ↓
                                           ModelValidation
```

An **MLModel** defines an architecture and its configuration.
A **PreprocessedDataset** is a named, reusable preprocessing recipe (indicators/clustering,
feature columns, normalization) built on a raw dataset — see "Preprocessed Datasets" below.
A **TrainingRun** trains the model on a dataset (directly, or via a `PreprocessedDataset`
recipe) with given hyperparameters.
A **TrainingCheckpoint** stores the model weights at each epoch.
**Deployment** copies the best checkpoint to the model's `artifact_path`, enabling inference.

---

## Architectures

| Architecture | Description | Best for |
|---|---|---|
| `lstm` | LSTM autoencoder/predictor | Short-term price forecasting |
| `seq2seq_transformer` | Transformer encoder-decoder | Multi-step sequence prediction |
| `timegan` | TimeGAN generative model | Synthetic data generation |
| `rl_agent` | Reinforcement learning agent | Adaptive policy learning |

Configuration schemas: `GET /model-config/architectures/{architecture}`

---

## Training Hyperparameters

Common hyperparameters (all architectures):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obs_len` | 60 | Input window length (number of bars) |
| `pred_len` | 10 | Output prediction horizon |
| `epochs` | 50 | Training epochs |
| `batch_size` | 32 | Mini-batch size |
| `lr` | 0.001 | Learning rate |
| `val_split` | 0.2 | Fraction of data held out for validation |
| `feature_cols` | `["close"]` | Columns to use as features — only meaningful for ad-hoc runs; ignored (overridden) when `preprocessed_dataset_id` is set |
| `normalize` | `"returns"` | Normalisation: `returns`, `zscore`, `minmax`, `robust`, `none` — same override rule as `feature_cols` |
| `preprocessing` | `null` | `{ indicators: [...], clustering: {...} }` — same override rule as `feature_cols`. See `backend/model/trainers/preprocessing.py` for the indicator types (`sma`, `ema`, `rsi`, `macd`, `bbands`, `atr`, `returns`, `volatility`) and their output column-naming convention. |

Architecture-specific params are passed in the same `hyperparams` dict.

`feature_cols`/`normalize`/`preprocessing` can still be set inline for one-off/ad-hoc runs (e.g.
via the MCP `start_training_run` tool or `POST /training-runs/search`), but the `/model/{id}`
"New Training Run" UI only offers picking a saved `PreprocessedDataset` recipe — see below.

Row cap: `OHLCWindowDataset` keeps only the most recent `_MAX_OHLC_ROWS` (50,000) rows after
preprocessing, so a training run on a larger dataset only ever sees a recent slice of it.

---

## Preprocessed Datasets

A **PreprocessedDataset** is a named, reusable preprocessing recipe — save it once, then pick
it from a list every time you start a training run instead of re-specifying indicators/
clustering/feature columns/normalization inline. It also gives you a browsable record of what
kind of data each training run actually used.

```
POST   /preprocessed-datasets                              → 202, status: pending
GET    /preprocessed-datasets?dataset_id={id}               → list (optionally filtered)
GET    /preprocessed-datasets/{id}
PATCH  /preprocessed-datasets/{id}    { "name": "..." }      → rename only, immutable otherwise
DELETE /preprocessed-datasets/{id}                           → 409 if referenced by a TrainingRun
POST   /preprocessed-datasets/{id}/characteristics/compute  → 202, recompute
```

```json
POST /preprocessed-datasets
{
  "name": "USDJPY + RSI/MACD zscore",
  "dataset_id": 3,
  "preprocessing": { "indicators": [{"type": "rsi", "period": 14}], "clustering": {"enabled": false} },
  "feature_cols": ["close", "rsi_14"],
  "normalize": "zscore"
}
```

Creating one enqueues `compute_preprocessed_characteristics` (`characteristics` queue) — the same
5 "Structure" analyses as dataset/training-run characteristics (see below), computed once on this
recipe's resulting series and cached (`status: ready` or `error`). To use it:

```json
POST /models/{id}/training-runs
{ "preprocessed_dataset_id": 7, "hyperparams": { "obs_len": 60, "epochs": 50, "lr": 0.001 } }
```

`dataset_id` on the `TrainingRun` is derived server-side from the recipe — you don't send it. A
recipe is immutable after creation (only `name` can change) so its cached characteristics always
match what it actually produces; to change indicators, create a new recipe.

Recipes are **config-only** — no extra parquet is written. The actual transform still runs
on-the-fly at training time via the existing `apply_preprocessing`/`OHLCWindowDataset` pipeline,
so a recipe never goes stale even if the base dataset later gets more data via incremental
collection.

Browse recipes and their characteristics at `/data/preprocessed` (list) and
`/data/preprocessed/{id}` (detail) in the UI.

---

## Training Lifecycle

```
POST /models/{id}/training-runs  →  status: pending
       ↓ celery enqueues train_model
status: running
  ↓ per epoch:  current_epoch, val_loss, eta_seconds updated in DB
  ↓             TrainingRunMetric written
  ↓             TrainingCheckpoint written (best kept)
  ↓             SSE event published: { type: "epoch_completed", epoch, train_loss, val_loss }
status: completed | error | stopped
```

Stop a running job: `POST /training-runs/{id}/stop` — sets `stop_requested=true`, worker exits after current epoch.

Right after the model is built (before the epoch loop), the worker also records two things on
the `TrainingRun`:
- `num_params` — trainable parameter count (`null` for `rl_agent`, trained outside this worker).
- `preprocessed_characteristics` — the same 5 "Structure" analyses as the dataset characteristics
  (`long_range_dependence`, `spectral_periodicity`, `multiscale_wavelet`,
  `complexity_nonlinearity`, `regime_changes` — see `docs/data-layer.md`), but computed on the
  data this run actually trains on: after `preprocessing` (indicators/clustering) and the row
  cap, on the primary (`feature_cols[0]`) column, before `normalize` (whose output — differenced
  or z-scored/min-maxed values — breaks the log-return math these analyses rely on). When the run
  references a `PreprocessedDataset` recipe, its already-computed `characteristics` are reused
  as-is instead of recomputing (falling back to a fresh
  `model/trainers/dataset.py:compute_effective_characteristics` call if the recipe's own
  background job hasn't finished yet). Best-effort either way — a computation failure is stored
  per-analysis as `{"error": ...}` and never aborts training. Shown on the training run detail
  page, and used by `/model/compare`'s "Data × Model Analysis" chart in place of the raw
  dataset's characteristics whenever a run has it.

When a run references a `PreprocessedDataset`, the worker also snapshots the recipe's resolved
`preprocessing`/`feature_cols`/`normalize` (plus `preprocessed_dataset_id`/`_name`) back into
`TrainingRun.hyperparams` at the same "status → running" update — so every run stays fully
self-describing (what data it actually trained on) even if its recipe is later renamed or
deleted.

---

## Deployment

Deploy a completed training run to make it available for inference:

```
POST /models/{id}/deploy?training_run_id={run_id}
```

This sets `MLModel.artifact_path = TrainingRun.artifact_path` and `MLModel.status = "deployed"`.

Only one run can be deployed at a time per model. Deploying a new run replaces the previous deployment.

---

## Inference

```
POST /models/{id}/predict
{
  "features": [[1.1, 1.2, ...], [1.0, 1.1, ...], ...],  // shape: [obs_len, n_features]
  "feature_names": ["close", "volume"]
}
```

Returns: `{ "predictions": [[...], ...] }` — shape `[pred_len, n_outputs]`.

The model must be deployed (`status: deployed`) before inference is available.

---

## Validation

After training, validate on a held-out dataset to get unbiased performance metrics:

```
POST /models/{id}/validations
{
  "training_run_id": 5,
  "dataset_id": 3
}
```

Validation metrics:

| Metric | Description |
|--------|-------------|
| `directional_accuracy` | % of predictions where direction is correct |
| `mae` | Mean absolute error |
| `rmse` | Root mean squared error |
| `sharpe_proxy` | Sharpe ratio if trading on predictions |
| `acf_match` | For GANs: how well autocorrelation structure is preserved |

---

## Hyperparameter Search

Run a grid search over multiple hyperparameter combinations:

```
POST /training-runs/search
{
  "model_id": 1,
  "dataset_id": 2,
  "search_grid": {
    "lr": [0.001, 0.0001],
    "batch_size": [32, 64],
    "obs_len": [30, 60]
  }
}
```

Creates one `TrainingRun` per combination and enqueues all. Compare results via `GET /training-runs/compare?run_ids=1,2,3,4`.

Each entry in the compare response also includes `architecture`, `model_name`, `num_params`
(trainable parameter count, recorded once the model is built at the start of training — `null`
for `rl_agent` runs, which train outside this worker), `preprocessed_characteristics`, and
`validation` (the latest `ModelValidation.metrics` for that run, or `null` if none has been run
yet). The `/model/compare` page uses these fields to plot training-data characteristics (Hurst,
periodicity strength, entropy, regime changes, etc.) against model size and performance across
runs — preferring each run's own `preprocessed_characteristics` when present, and falling back
to the raw dataset's characteristics (fetched client-side via `GET /datasets/{id}/characteristics`
per run's `dataset_id`) for older runs that predate that field. See `docs/data-layer.md` for what
each characteristic measures.

---

## Using a Model in a Strategy

Reference a deployed model in a strategy condition:

```json
{
  "handler": "ml_signal",
  "params": {
    "model_id": 3,
    "feature_cols": ["close", "high", "low", "volume"],
    "obs_len": 60,
    "threshold": 0.6
  }
}
```

The executor calls `POST /models/3/predict` with the last 60 bars at each tick event. If the prediction exceeds the threshold, the condition returns `true`.
