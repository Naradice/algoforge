"""
Orchestrates a Colab-executed training run for a TrainingRun with execution_target="colab":
export a dataset snapshot to Google Drive, generate a self-contained notebook
(model/notebook_export.py), run it via the colab-cli container (model/colab_runner.py), pull the
result back, and update the same TrainingRun row a "local" run would have updated itself.

See docs/colab-workflow.md for the manual, step-by-step CLI version of this same pipeline (the
one a human runs by hand) -- this module automates exactly that sequence. Entry point is
celery_worker.py's `colab_train_model` task, routed to the `colab` queue (see celery_app.py's
task_routes) rather than `training`, so a long Colab run (which spends nearly all its wall time
blocked on a subprocess call, not doing local work) never occupies a slot the local GPU/CPU
trainer needs.

Not yet handled (flag before relying on this unattended):
  - No duplicate-execution Redis lock the way _train_model has (celery_worker.py's
    exec_lock_key) -- a Colab run is expected to finish well within Redis's broker
    visibility_timeout (12h, see celery_app.py), so redelivery-triggered duplicate execution is
    unlikely but not impossible for a run given a very large colab_timeout_seconds.
  - Progress (epoch-by-epoch val_loss) is only visible after the whole run finishes -- no
    TrainingRunMetric rows are written while colab_runner.exec_notebook is blocked, unlike
    _train_model's per-epoch writes. get_training_status/current_epoch stays at 0 until then.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import select, update

from model.notebook_export import build_notebook
from model_core.architectures import NON_GRADIENT_ARCHITECTURES

logger = logging.getLogger("colab_trainer")

# Passed to `colab exec --timeout` when the run's hyperparams don't override it via
# "colab_timeout_seconds" -- generous, since this path is meant for runs that would take a
# while on a CPU. Popped out of hyperparams before they're handed to build_notebook (it isn't a
# model hyperparameter) but kept in the TrainingRun's stored hyperparams for the record.
_DEFAULT_TIMEOUT_SECONDS = 3600.0

# OHLCWindowDataset's own accepted token_level values (see model_core/trainers/dataset.py) --
# validated here (before a Colab runtime is even provisioned) purely so a typo gets a clear
# 422 instead of failing deep inside the notebook. "cluster" additionally needs scikit-learn,
# which notebook_export.py installs automatically only when this value is set.
_VALID_TOKEN_LEVELS = {"diff", "quantize_diff", "cluster", "digits", "sax"}


def check_colab_supported(architecture: str, preprocessed_dataset_id: int | None, hyperparams: dict) -> None:
    """Raise HTTPException(422) if the requested combination isn't something
    model/notebook_export.py's generator can produce. Checked at TrainingRun creation time (see
    model/service.py:create_training_run) so an unsupported request fails immediately with a
    clear reason instead of failing deep into the pipeline after a Colab runtime is already
    provisioned."""
    if architecture in NON_GRADIENT_ARCHITECTURES:
        raise HTTPException(status_code=422, detail={
            "code": "COLAB_UNSUPPORTED_ARCHITECTURE",
            "message": (
                f"execution_target='colab' doesn't support architecture={architecture!r} -- "
                "it's not torch.nn.Module-based (see model_core.architectures.NON_GRADIENT_ARCHITECTURES)"
            ),
        })
    if preprocessed_dataset_id is not None:
        raise HTTPException(status_code=422, detail={
            "code": "COLAB_UNSUPPORTED_PREPROCESSED_DATASET",
            "message": "execution_target='colab' doesn't support preprocessed_dataset_id yet -- use inline hyperparams instead",
        })
    token_level = hyperparams.get("token_level")
    if token_level is not None and token_level not in _VALID_TOKEN_LEVELS:
        raise HTTPException(status_code=422, detail={
            "code": "COLAB_INVALID_TOKEN_LEVEL",
            "message": f"token_level must be one of {sorted(_VALID_TOKEN_LEVELS)} or omitted, got {token_level!r}",
        })
    split_mode = hyperparams.get("split_mode", "chronological")
    if split_mode not in ("chronological", "regime_controlled"):
        raise HTTPException(status_code=422, detail={
            "code": "COLAB_UNSUPPORTED_SPLIT_MODE",
            "message": f"split_mode must be 'chronological' or 'regime_controlled' (OHLCWindowDataset's only two), got {split_mode!r}",
        })


async def run_colab_training(training_run_id: int) -> dict:
    """Entry point called by celery_worker.py's colab_train_model task. Mirrors _train_model's
    shape (own DB engine, catch-all error handling that always leaves the TrainingRun in a
    terminal status, webhook dispatch on completion/error) so a Colab run behaves the same as a
    local one from the outside -- get_training_status / a training.completed webhook works the
    same regardless of execution_target."""
    import asyncio

    from celery_worker import _make_db, _release_lock
    from data import snapshot_service
    from data.models import Dataset
    from model import colab_runner
    from model.models import MLModel, TrainingCheckpoint, TrainingRun, TrainingRunMetric
    from webhooks.dispatcher import dispatch

    factory, engine = _make_db()
    session_name = f"run-{training_run_id}"
    try:
        async with factory() as db:
            run = (await db.execute(select(TrainingRun).where(TrainingRun.id == training_run_id))).scalar_one_or_none()
            if run is None:
                return {"error": "TRAINING_RUN_NOT_FOUND"}
            model_rec = (await db.execute(select(MLModel).where(MLModel.id == run.model_id))).scalar_one_or_none()
            if model_rec is None:
                return {"error": "MODEL_NOT_FOUND"}
            dataset_rec = (await db.execute(select(Dataset).where(Dataset.id == run.dataset_id))).scalar_one_or_none()
            if dataset_rec is None:
                return {"error": "DATASET_NOT_FOUND"}

            architecture = model_rec.architecture
            model_config = dict(model_rec.config)
            model_id = run.model_id
            dataset_id = run.dataset_id
            hyperparams = dict(run.hyperparams)

            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                status="running", started_at=datetime.now(timezone.utc),
            ))
            await db.commit()

        loop = asyncio.get_event_loop()

        # --- 1. Snapshot the dataset and upload it to Drive ------------------------------------
        async with factory() as db:
            snapshot = await snapshot_service.create_snapshot(db, dataset_id)
            await db.commit()
        async with factory() as db:
            snapshot = await snapshot_service.upload_snapshot(db, snapshot.id, provider="gdrive")
            await db.commit()
        if snapshot.status != "uploaded" or not snapshot.export_ref or "url" not in snapshot.export_ref:
            raise RuntimeError(f"snapshot {snapshot.id} did not upload successfully: {snapshot.export_ref}")

        # --- 2. Generate the notebook ------------------------------------------------------------
        timeout_seconds = float(hyperparams.get("colab_timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
        notebook_hyperparams = {k: v for k, v in hyperparams.items() if k != "colab_timeout_seconds"}

        notebooks_dir = Path(os.getenv("ALGOFORGE_NOTEBOOKS_DIR", "../notebooks")).resolve()
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = notebooks_dir / f"{session_name}.ipynb"

        import nbformat as nbf
        nb = build_notebook(
            architecture=architecture,
            model_name=session_name,
            model_config=model_config,
            dataset_id=dataset_id,
            snapshot_id=snapshot.id,
            snapshot_url=snapshot.export_ref["url"],
            snapshot_sha256=snapshot.sha256,
            hyperparams=notebook_hyperparams,
        )
        nbf.write(nb, str(notebook_path))
        logger.info(f"colab training run {training_run_id}: generated {notebook_path}")

        # --- 3. Run it on Colab, via colab-cli -----------------------------------------------
        container_notebook_path = f"/notebooks/{session_name}.ipynb"

        await loop.run_in_executor(None, colab_runner.new_session, session_name)
        try:
            exec_stdout = await loop.run_in_executor(
                None, colab_runner.exec_notebook, session_name, container_notebook_path, timeout_seconds,
            )
            # colab exec's own exit code reliability for an in-notebook failure is unconfirmed
            # (infra/colab-cli/README.md's known-unknowns) -- log the notebook's own print
            # output unconditionally so a silent in-notebook failure (e.g. the sha256 assertion)
            # is still visible here even when colab exec itself reports success.
            logger.info(f"colab training run {training_run_id}: colab exec output:\n{exec_stdout}")
            best_path = notebooks_dir / f"{session_name}_best.pt"
            metrics_path = notebooks_dir / f"{session_name}_metrics.json"
            await loop.run_in_executor(None, colab_runner.download, session_name, "best.pt", best_path)
            await loop.run_in_executor(None, colab_runner.download, session_name, "metrics.json", metrics_path)
        finally:
            try:
                await loop.run_in_executor(None, colab_runner.stop_session, session_name)
            except Exception:
                logger.warning(f"colab training run {training_run_id}: failed to stop session {session_name}", exc_info=True)

        # --- 4. Pull the result back into this TrainingRun --------------------------------------
        metadata = json.loads(metrics_path.read_text())

        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts")).resolve()
        checkpoint_dir = store / "models" / str(model_id) / f"training_{training_run_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dest = checkpoint_dir / "best.pt"
        dest.write_bytes(best_path.read_bytes())
        artifact_rel = dest.relative_to(store).as_posix()

        best_epoch = metadata["best_epoch"]
        val_loss = metadata["val_loss"]
        external_ref = {
            "platform": "colab",
            "dataset_snapshot_id": snapshot.id,
            "dataset_snapshot_sha256": snapshot.sha256,
            **metadata.get("external_ref", {}),
        }

        async with factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                status="completed", ended_at=datetime.now(timezone.utc),
                best_epoch=best_epoch, val_loss=val_loss,
                num_params=metadata.get("num_params"),
                artifact_path=artifact_rel,
                hyperparams={**hyperparams, "_external_ref": external_ref},
            ))
            db.add(TrainingCheckpoint(
                training_run_id=training_run_id, epoch=best_epoch,
                metrics={"val_loss": val_loss, "source": "colab"},
                artifact_path=artifact_rel,
            ))
            for m in metadata.get("epoch_metrics", []):
                db.add(TrainingRunMetric(
                    training_run_id=training_run_id, epoch=m["epoch"],
                    train_loss=m.get("train_loss"), val_loss=m.get("val_loss"), lr=m.get("lr"),
                ))
            await db.execute(update(MLModel).where(MLModel.id == model_id).values(status="trained"))
            await dispatch(db, "training.completed", {
                "training_run_id": training_run_id, "model_id": model_id,
                "val_loss": val_loss, "best_epoch": best_epoch,
            })
            await db.commit()

        logger.info(f"colab training run {training_run_id} completed. best_epoch={best_epoch} val_loss={val_loss}")
        return {"best_epoch": best_epoch, "val_loss": val_loss, "artifact_path": artifact_rel}

    except Exception as e:
        # Same rationale as _train_model's catch-all: without this, a failure anywhere in the
        # pipeline above (snapshot export, notebook generation, colab-cli, or a malformed
        # metrics.json) leaves the TrainingRun stuck at status "running" forever.
        logger.exception(f"colab training run {training_run_id} failed")
        async with factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                status="error", ended_at=datetime.now(timezone.utc),
            ))
            await dispatch(db, "training.error", {
                "training_run_id": training_run_id, "error": str(e),
            })
            await db.commit()
        return {"error": str(e)}
    finally:
        await engine.dispose()
        _release_lock("colab_train_model", training_run_id)
