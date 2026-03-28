# AlgoForge ML Worker

Python 3.8 arq worker for RL (Reinforcement Learning) model training.

## Why a separate worker?

The main backend runs Python 3.12. PFRL and older gym environments require Python 3.8 and have incompatible dependencies. This separate worker process bridges the gap: the backend enqueues RL training jobs on the shared Redis queue, and this worker picks them up in the Python 3.8 environment.

All other model training (Seq2Seq Transformer, LSTM, TimeGAN) runs directly in the backend.

## Requirements

- Python 3.8
- Redis 7 (shared with backend)
- PostgreSQL 16 (shared with backend)
- GPU optional (CUDA supported via PyTorch)

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python worker.py
```

Or via Docker (see `../infra/docker-compose.yml`).

## Job functions

| Job | Status | Description |
|-----|--------|-------------|
| `train_rl_model` | Stub | Train an RL agent (PFRL-based). Triggered by backend when `architecture == "rl_agent"` |

The `train_rl_model` job is currently a stub (`NotImplementedError`). To implement, port from `stocknet/trainer/rltrainer.py`:

1. Load `TrainingRun` config from DB via `asyncpg`
2. Build environment and PFRL agent from `hyperparams`
3. Run episode loop, writing `current_epoch` and `val_loss` (mean reward) to DB each episode
4. Save checkpoint to `ARTIFACT_STORE_PATH/models/{model_id}/rl_{run_id}/best.pt`
5. On completion, update `TrainingRun.status = "completed"` and `MLModel.status = "trained"`

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Shared Redis URL | `redis://localhost:6379/0` |
| `DATABASE_URL` | PostgreSQL URL (asyncpg) | `postgresql://algoforge:algoforge@localhost:5432/algoforge` |
| `ARTIFACT_STORE_PATH` | Model checkpoint directory | `../artifacts` |

## Architecture notes

- Uses `asyncpg` directly (not SQLAlchemy) to keep the dependency footprint small
- Shares the `artifacts` Docker volume with the backend for checkpoint files
- `max_jobs = 2` — RL training is GPU-intensive; concurrency is kept low
- `job_timeout = 6 hours` — long RL runs are expected
