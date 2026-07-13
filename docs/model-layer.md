# ML Model Layer

## Concepts

```
MLModel  →  TrainingRun  →  (arq worker)  →  Checkpoint  →  Deploy  →  Predict
                                                    ↓
                                           ModelValidation
```

An **MLModel** defines an architecture and its configuration.
A **TrainingRun** trains the model on a specific dataset with given hyperparameters.
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
| `feature_cols` | `["close"]` | Columns to use as features |
| `normalize` | `"returns"` | Normalisation: `returns`, `zscore`, `minmax`, `none` |
| `val_split` | 0.2 | Fraction of data held out for validation |

Architecture-specific params are passed in the same `hyperparams` dict.

---

## Training Lifecycle

```
POST /models/{id}/training-runs  →  status: pending
       ↓ arq enqueues train_model
status: running
  ↓ per epoch:  current_epoch, val_loss, eta_seconds updated in DB
  ↓             TrainingRunMetric written
  ↓             TrainingCheckpoint written (best kept)
  ↓             SSE event published: { type: "epoch_completed", epoch, train_loss, val_loss }
status: completed | error | stopped
```

Stop a running job: `POST /training-runs/{id}/stop` — sets `stop_requested=true`, worker exits after current epoch.

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
for `rl_agent` runs, which train outside this worker) and `validation` (the latest
`ModelValidation.metrics` for that run, or `null` if none has been run yet). The `/model/compare`
page uses these fields, joined client-side with `GET /datasets/{id}/characteristics` per run's
`dataset_id`, to plot training-data characteristics (Hurst, periodicity strength, entropy, regime
changes, etc. — see `docs/data-layer.md`) against model size and performance across runs.

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
