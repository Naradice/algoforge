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

Training automatically uses a GPU when one is available (`torch.cuda.is_available()`); falls
back to CPU otherwise. In the Docker dev stack, the `celery-training` service reserves a GPU
device via `deploy.resources.reservations.devices` in `infra/docker-compose.dev.yml` — no image
rebuild needed as long as the installed `torch` wheel has CUDA support. Measured speedup: a
1.85M-param `seq2seq_transformer` went from ~5.5 min/epoch on 4 CPU cores to ~12s/epoch on an
RTX 3060 Ti (~25–30×).

---

## Architectures

| Architecture | Description | Best for |
|---|---|---|
| `lstm` | LSTM autoencoder/predictor | Short-term price forecasting |
| `seq2seq_transformer` | Transformer encoder-decoder | Multi-step sequence prediction |
| `timegan` | TimeGAN generative model | Synthetic data generation |
| `rl_agent` | Reinforcement learning agent | Adaptive policy learning |
| `ar` / `ma` / `arma` | Classical statistical baseline (statsmodels ARIMA) | Sanity-checking whether a neural net beats a naive fit — see "Baseline Models" below |

Configuration schemas: `GET /model-config/architectures/{architecture}`

---

## Baseline Models (`ar` / `ma` / `arma`)

Every other architecture is a `torch.nn.Module` trained by gradient descent over epochs. `ar`,
`ma`, and `arma` are classical statistical models (`statsmodels.tsa.arima.model.ARIMA`) fit by
MLE in a single shot — no epochs, no checkpoints per epoch, no `state_dict`. They still go
through the same `MLModel`/`TrainingRun` tables and the same `/model`, `/model/{id}`, and
`/model/compare` UI as any other model, so you can directly compare a real model's MSE against a
naive baseline's.

**Config** (`MLModel.config`) maps to a statsmodels `(p, d, q)` order:

| Architecture | Order | Config fields |
|---|---|---|
| `ar` | `(p, d, 0)` | `p` (default 2), `d` (default 0) |
| `ma` | `(0, d, q)` | `q` (default 2), `d` (default 0) |
| `arma` | `(p, d, q)` | `p`, `d`, `q` (defaults 2, 0, 2) |

`d` defaults to 0 because classical AR/MA assume an already-stationary series — pick a
`PreprocessedDataset` recipe with `normalize: "returns"` (log-differenced), or raise `d` if you
select a recipe with `normalize: "none"` on a raw, non-stationary price series.

**Training** (`celery_worker.py`'s `_run_arima_training`, dispatched from `_train_model` before
reaching the torch-specific code): fits on the training split (from `OHLCWindowDataset`'s
train/val split, but as a flat series, not sliding windows — see
`model/trainers/arima_trainer.py:load_series_for_arima`), then walk-forward evaluates over the
validation split in non-overlapping `pred_len`-sized blocks, extending state after each block via
`.append(actual, refit=False)` (cheap — no re-optimization). Reported as `best_epoch=1` with one
`TrainingRunMetric` row; `num_params` is `len(results.params)` (AR/MA coefficients + const +
`sigma2`) — directly comparable to a neural net's parameter count on the same "Data × Model
Analysis" chart in `/model/compare`. `TrainingRun.val_loss` is the walk-forward MSE — the same
kind of number `MSELoss()` produces for the neural architectures, so it's directly comparable
across every architecture without any extra validation step.

**Relative MSE**: on `/model/compare`, once you select at least one `ar`/`ma`/`arma` run
alongside others, a "Relative MSE" column appears — `other_run.val_loss / baseline_run.val_loss`
for every non-baseline run. Below 1× means the model beats the baseline; above 1× means it's
worse than a naive statistical fit.

**Only valid when every compared run's recipe uses the same `normalize`.** `val_loss` is MSE of
whatever the recipe's `normalize` step produced — `"returns"` (log-returns, typically ~1e-3
magnitude) and `"zscore"` (unit variance) are different units, so a ratio between runs on
different `normalize` settings is meaningless even though it's the same loss function (MSE) on
both sides. Discovered live: an AR baseline trained on a `"returns"` recipe looked ~1,000,000×
better than LSTMs trained on a `"zscore"` recipe — pure scale artifact, not model skill. The
compare table detects this (comparing each run's `hyperparams.normalize` against the baseline's)
and shows "⚠ mismatched normalize" instead of a ratio. To get a real comparison, retrain every
model you want to compare against the same `PreprocessedDataset` recipe.

**Not supported yet**: live inference (`POST /models/{id}/predict`) and held-out-dataset
validation jobs (`POST /models/{id}/validations`) both raise a clear error for these
architectures — `predict()` guards explicitly rather than crashing on `torch.load()` of a
statsmodels pickle. Use `TrainingRun.val_loss` / Relative MSE for comparison in the meantime.

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
| `early_stop_patience` | `null` (disabled) | If set, training stops once `epochs_since_improvement >= early_stop_patience` — i.e. that many consecutive epochs with no new best `val_loss`. `epochs` still applies as a hard cap. Recommended whenever comparing architectures/sizes: a fixed `epochs` budget under-trains larger models and makes params-vs-loss comparisons meaningless (see the "Baseline Models" scaling-law caveat above). |
| `shuffle` | `false` | If set, training windows are shuffled into a new random order every epoch (`backend/model/trainers/supervised.py`); off by default, so every epoch otherwise walks windows in the same fixed chronological order. Only implemented for the supervised trainer (lstm, seq2seq_transformer, cnn_lstm, tcn, nbeats) — accepted but not yet applied for timegan/vae. Matters most for small or strongly periodic datasets trained many epochs: a fixed batch order repeated every epoch is a perfectly periodic gradient sequence, which can lock Adam's momentum/adaptive-LR state into a plateau instead of converging. |
| `max_rows` | `null` (uses the 50,000-row default cap) | Overrides `OHLCWindowDataset._MAX_OHLC_ROWS` for this run — set it to the dataset's actual row count (or higher) to train on more than the default cap. See "Row cap" below for why this exists and when to raise it. |
| `divergence_factor` | `null` (disabled) | If set, training stops the moment `val_loss > divergence_factor × best_val_loss`, checked every epoch. Distinct from `early_stop_patience`: patience fires on a *plateau* (many epochs with no improvement, however small the gap); this fires the moment a run gets much *worse* than its best, however few epochs that takes — catches a blown-up run (e.g. batch_size/lr mismatch) fast instead of waiting out the full patience window. The two can be set together. |
| `lr_warmup_epochs` | `0` (disabled) | If set to `N`, the learning rate ramps linearly from `lr/N` up to the full `lr` over the first `N` epochs; the `ReduceLROnPlateau` scheduler is held off until warmup finishes so it doesn't fight the ramp. Opt-in only — `lr` and `batch_size` are never auto-adjusted relative to each other or to this setting; a user who wants warmup sets it explicitly. Useful for architectures/batch sizes prone to diverging early (observed live: `seq2seq_transformer` at very large or very small `batch_size` with the platform's default `lr` can fail to train at all — see the "Baseline Models" scaling-law caveat above for the XLarge case). |
| `disable_lr_scheduler` | `false` | If set, `ReduceLROnPlateau` is never created — `lr` stays exactly at its configured value for the whole run. `ReduceLROnPlateau`'s `patience=5` is epoch-denominated, same as `early_stop_patience`, so it's a second confound (independent of early stopping) when comparing runs with different epoch lengths — e.g. holding raw optimizer step count constant between a small and a large dataset only isolates step count if this is also disabled, since otherwise the two runs still get a different number of LR reductions per step. |
| `seed` | `null` (unseeded) | If set, seeds `torch`/`numpy`/`random` before model construction, making weight init (and `shuffle=true` ordering) reproducible. Without it, two "identical" runs draw from whatever ambient RNG state the worker process happens to be in — observed live, this alone produced a 2×+ spread between nominally identical runs. Required for any comparison that needs to distinguish a real effect from init noise (e.g. multi-seed repeats to check whether a result holds up). |
| `optimizer` | `"adam"` | `"adam"`, `"adamw"`, or `"sgd"`. Used to isolate whether an optimizer's own internal state (Adam/AdamW's per-parameter first/second moment estimates) — not just `ReduceLROnPlateau`'s epoch-denominated schedule — explains a result that differs between runs with very different epoch lengths at the same total step count. `sgd` ignores `beta1`/`beta2` and instead reads `momentum` (default `0.0`, i.e. plain SGD). |
| `beta1` / `beta2` | `0.9` / `0.999` | Overrides for Adam/AdamW's momentum (`beta1`) and second-moment (`beta2`) coefficients. Ignored for `optimizer="sgd"`. Setting `beta1=0` removes momentum from Adam/AdamW entirely, isolating the second-moment (adaptive learning rate) term on its own. |
| `momentum` | `0.0` | SGD momentum coefficient. Ignored unless `optimizer="sgd"`. |
| `weight_decay` | `null` (uses each optimizer's own torch default: `0` for `adam`/`sgd`, `0.01` for `adamw`) | Explicit override so cross-optimizer comparisons aren't silently skewed by torch's differing per-optimizer defaults — e.g. comparing `optimizer="adam"` against `optimizer="adamw"` without setting this compares "no weight decay" against "0.01 decoupled decay" as much as it compares the optimizers themselves. Set it explicitly (e.g. `0`) on both sides to isolate the optimizer's own update rule. |
| `max_steps` | `null` (uses the epoch-based loop) | Opt-in escape hatch that removes "epoch" from the training loop entirely. When set, training runs an infinite, reshuffled stream of training windows (`train_steps` in `backend/model/trainers/supervised.py`) for exactly `max_steps` gradient updates, with validation/checkpointing/early-stopping keyed to a step counter instead of a pass through the dataset. Only implemented for the supervised trainer (lstm, seq2seq_transformer, cnn_lstm, tcn, nbeats) — raises for timegan/vae. `epochs`, `shuffle`, `lr_warmup_epochs`, and the `ReduceLROnPlateau` scheduler are all epoch-keyed and are ignored in this mode; pair `max_steps` with `disable_lr_scheduler=true`. `current_epoch`/`TrainingCheckpoint.epoch`/`TrainingRunMetric.epoch` store the *validation-check index*, not an epoch — multiply by `val_every_steps` to recover the actual step count (no schema change for what is fundamentally a research-mode loop). |
| `val_every_steps` | `5000` | Only used when `max_steps` is set. How often (in gradient steps) to run a full validation pass, write a checkpoint/metric row, and evaluate early-stopping/divergence — the step-space equivalent of "once per epoch," but with no relationship to how many windows one pass through the dataset takes. |
| `early_stop_patience_checks` | `null` (disabled) | Only used when `max_steps` is set. Step-space equivalent of `early_stop_patience`: stops once this many consecutive validation *checks* (each `val_every_steps` apart) pass with no new best `val_loss`. `divergence_factor` still applies as-is in this mode, checked at the same cadence. |

Architecture-specific params are passed in the same `hyperparams` dict.

`feature_cols`/`normalize`/`preprocessing` can still be set inline for one-off/ad-hoc runs (e.g.
via the MCP `start_training_run` tool or `POST /training-runs/search`), but the `/model/{id}`
"New Training Run" UI only offers picking a saved `PreprocessedDataset` recipe — see below.

Row cap: `OHLCWindowDataset` keeps only the most recent `_MAX_OHLC_ROWS` (50,000) rows after
preprocessing by default, so a training run on a larger dataset only ever sees a recent slice of
it unless `max_rows` is set. The default exists to bound window-array memory (~1 GB) on typical
datasets, not as a hard ceiling — raising it is a deliberate per-run opt-in, since building
windows for a multi-million-row dataset costs real time and memory up front (once, at dataset
load, not per epoch).

### Comparing training runs — methodology

Distilled from a multi-week investigation into why a data-volume sweep on a synthetic,
perfectly-periodic sine wave showed a large, real-looking improvement from more rows even though
the data past the first cycle carries no new information. The eventual answer (see below) was a
confound in how runs were being compared, not a property of any model or optimizer — five
plausible-looking mechanisms were investigated and ruled out along the way (more gradient steps,
batch-order resonance, model capacity, a validation-split artifact, batch size/gradient noise,
the LR scheduler's own epoch-denomination, and finally Adam/AdamW momentum) before the real cause
surfaced. Apply these whenever comparing architectures, sizes, or datasets against each other —
not just for data-volume sweeps:

1. **Always seed, and always replicate across a few seeds before trusting a result.** This
   pipeline has no seed control unless `seed` is set — two "identical" runs can land 2×+ apart on
   nothing but weight-init/shuffle-order luck. The single most repeated failure mode in the
   investigation above was treating a clean-looking single-run result (a monotonic trend, a
   dramatic ablation, a mechanistically satisfying story) as established. Every one of those
   specific claims — a 5.3× optimizer-driven gap, a non-monotonic "hump" in momentum strength, a
   tiled dataset outright beating real data — evaporated or reversed under 3-seed replication.
   Clean and monotonic is not evidence; it just means the noise happened to line up once.
2. **Match total optimizer steps, not epoch count, when comparing runs whose datasets differ in
   size.** A fixed `epochs` budget gives a small dataset far fewer gradient updates than a large
   one, or vice versa depending which side of the comparison you're on — compute each side's
   `epochs` from the row count and `batch_size` to hit a common step-count target instead.
3. **Equalize validation/checkpoint frequency, not just step count.** This was the one confound
   the investigation above almost missed. Ordinary epoch-based training validates once per epoch,
   so a run with short epochs (small dataset) gets far more validation checks — and therefore far
   more chances to record a lucky low `val_loss` — than a run with long epochs (large dataset) at
   the *same total step count*. This is a best-of-N selection effect, unrelated to the models or
   optimizers being compared, and it was large enough on its own to fully explain a gap that had
   survived every other control. Use `max_steps` + `val_every_steps` (see above) to give every
   compared run the same number of validation checks regardless of how many rows its dataset has.
4. **Keep the LR scheduler and early stopping step-denominated too, or disable them.**
   `ReduceLROnPlateau` and `early_stop_patience` both act per-epoch — the same asymmetry as point
   3 applies to them independently. Set `disable_lr_scheduler=true` for step-matched comparisons,
   and use `early_stop_patience_checks` (step-space) rather than `early_stop_patience` (epoch-space)
   whenever `max_steps` is in play.
5. **A result that looks especially clean is a reason for more scrutiny, not less.** In order,
   the investigation's most "decisive"-looking findings — a 54× improvement from 100× more data,
   a tiled dataset beating real data outright, a 5.3× gap explained by Adam's momentum, a smooth
   non-monotonic dose-response curve in β1 — were also, in order, the ones that turned out to be
   partly or wholly artifacts once checked against seeds and, finally, validation-frequency
   parity. Treat "this fits a satisfying narrative" as orthogonal to "this is true."

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
