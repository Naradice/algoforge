"""
AlgoForge ML Worker — Python 3.8 arq worker for RL model training.

Start with:
    python worker.py

Job functions registered here are enqueued by the main backend (Python 3.12)
via Redis and executed here in the Python 3.8 environment where PFRL/gym are available.

All other training (Seq2SeqTransformer, LSTM, GAN) runs directly in the backend.
"""

import asyncio
import logging
import os

import asyncpg
from arq import create_pool, cron
from arq.connections import RedisSettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("ml_worker")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://algoforge:algoforge@localhost:5432/algoforge")
ARTIFACT_STORE_PATH = os.getenv("ARTIFACT_STORE_PATH", "../artifacts")


async def train_rl_model(ctx, training_run_id: int) -> dict:
    """
    arq job: train an RL model.

    Enqueued by backend/model/service.py when architecture == "rl_agent".
    Updates training_runs.status, current_epoch, val_loss in DB as training progresses.

    TODO Phase 2: implement RL training by copying from stocknet/trainer/rltrainer.py
    """
    logger.info(f"RL training run {training_run_id} started")
    # TODO Phase 2: load hyperparams from DB, run rltrainer, write checkpoints
    raise NotImplementedError("RL training not yet implemented — Phase 2")


class WorkerSettings:
    functions = [train_rl_model]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    max_jobs = 2  # RL training is GPU-intensive; limit concurrency
    job_timeout = 3600 * 6  # 6 hours max per training run


if __name__ == "__main__":
    import arq.worker
    arq.worker.run_worker(WorkerSettings)
