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

Progress and stop_requested (see _poll_and_maybe_stop) are handled by polling the notebook's
own progress.json (written every epoch -- see notebook_export.py) via `colab download` while
colab_runner.exec_notebook is otherwise blocked -- confirmed live that `colab status`/`colab
ls`/`colab download` all work normally against a session mid-`colab exec`, and that `colab
stop`-ing a session mid-exec makes the blocked `colab exec` process exit non-zero
(RuntimeError: Connection was lost.) promptly, which is what lets _poll_and_maybe_stop's
stop_requested branch actually interrupt a run instead of just marking it for later.

Not yet handled (flag before relying on this unattended):
  - No duplicate-execution Redis lock the way _train_model has (celery_worker.py's
    exec_lock_key) -- a Colab run is expected to finish well within Redis's broker
    visibility_timeout (12h, see celery_app.py), so redelivery-triggered duplicate execution is
    unlikely but not impossible for a run given a very large colab_timeout_seconds.
  - A run stopped early ends up with zero TrainingRunMetric rows (only current_epoch/val_loss
    on TrainingRun itself are updated during polling -- see _poll_and_maybe_stop's docstring for
    why) -- accurate but sparser than a completed run's full metrics.json import.
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

# How often _poll_and_maybe_stop checks progress.json and stop_requested. Each check costs a
# `colab download` (a docker exec round-trip, a couple seconds) plus two small DB round-trips,
# so this trades "how current is current_epoch / how promptly does stop actually stop" against
# not hammering colab-cli with requests while it's already busy running the notebook.
_POLL_INTERVAL_SECONDS = 20.0


def check_colab_supported(architecture: str, preprocessed_dataset_id: int | None, hyperparams: dict) -> None:
    """Raise HTTPException(422) if the requested combination isn't something
    model/notebook_export.py's generator can produce. Checked at TrainingRun creation time (see
    model/service.py:create_training_run) so an unsupported request fails immediately with a
    clear reason instead of failing deep into the pipeline after a Colab runtime is already
    provisioned.

    *preprocessed_dataset_id* itself needs no validation here -- run_colab_training resolves it
    the same way celery_worker.py's _resolve_training_context does (the referenced
    PreprocessedDataset's own preprocessing/feature_cols/normalize become this run's), and
    whatever it resolves to is still built from the same components (token_level, preprocessing)
    validated below. Kept as a parameter so this signature doesn't need to change if that ever
    stops being true (e.g. a future recipe field this generator can't handle).
    """
    if architecture in NON_GRADIENT_ARCHITECTURES:
        raise HTTPException(status_code=422, detail={
            "code": "COLAB_UNSUPPORTED_ARCHITECTURE",
            "message": (
                f"execution_target='colab' doesn't support architecture={architecture!r} -- "
                "it's not torch.nn.Module-based (see model_core.architectures.NON_GRADIENT_ARCHITECTURES)"
            ),
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


async def _poll_and_maybe_stop(
    training_run_id: int, session_name: str, notebooks_dir: Path, factory,
) -> dict | None:
    """Runs concurrently with colab_runner.exec_notebook's blocking call (the caller cancels
    this once that returns on its own -- normal completion). Every _POLL_INTERVAL_SECONDS:
      1. Downloads the notebook's own progress.json (written every epoch -- see
         notebook_export.py) and reflects epoch/val_loss into TrainingRun, so
         get_training_status shows live progress instead of being stuck at 0 until the whole
         run finishes. Deliberately does NOT write TrainingRunMetric rows here -- the normal
         completion path imports metrics.json's full epoch_metrics in one batch, and doing
         that here too would double them up. A run that gets stopped early (see below)
         therefore ends up with zero TrainingRunMetric rows, which is an accurate reflection
         of what happened, not a bug.
      2. Checks TrainingRun.stop_requested. If set, best-effort grabs the latest best.pt (the
         notebook only writes a fresh one when val_loss improves, so this may be a few epochs
         behind progress.json) before calling colab_runner.stop_session -- confirmed live that
         this makes the concurrently-blocked exec_notebook call exit non-zero
         (RuntimeError: Connection was lost.) promptly, which the caller relies on to know the
         run actually stopped rather than just being marked to stop later.

    Returns a dict with "progress" (the last progress.json seen, or None) and "best_path" (the
    downloaded checkpoint Path, or None if none was ever available) if it stopped the session
    itself; returns None if cancelled by the caller instead (the normal-completion case, where
    the caller has metrics.json to work from and doesn't need this function's result).
    """
    import asyncio

    from model import colab_runner
    from model.models import TrainingRun

    loop = asyncio.get_event_loop()
    progress_path = notebooks_dir / f"{session_name}_progress.json"
    last_epoch_seen: int | None = None
    last_progress: dict | None = None

    while True:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        try:
            await loop.run_in_executor(None, colab_runner.download, session_name, "progress.json", progress_path)
            data = json.loads(progress_path.read_text())
            last_progress = data
            epoch = data.get("epoch")
            if epoch is not None and epoch != last_epoch_seen:
                last_epoch_seen = epoch
                async with factory() as db:
                    await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                        current_epoch=epoch, val_loss=data.get("val_loss"),
                    ))
                    await db.commit()
        except Exception:
            # Most likely progress.json doesn't exist yet (still installing/downloading the
            # dataset snapshot before the first epoch) -- not worth failing the run over.
            logger.debug(f"colab training run {training_run_id}: progress poll found nothing (yet?)", exc_info=True)

        try:
            async with factory() as db:
                stop_requested = (await db.execute(
                    select(TrainingRun.stop_requested).where(TrainingRun.id == training_run_id)
                )).scalar()
        except Exception:
            logger.warning(f"colab training run {training_run_id}: stop_requested check failed", exc_info=True)
            continue

        if stop_requested:
            logger.info(f"colab training run {training_run_id}: stop_requested -- stopping session {session_name}")
            best_path = notebooks_dir / f"{session_name}_best.pt"
            try:
                await loop.run_in_executor(None, colab_runner.download, session_name, "best.pt", best_path)
            except Exception:
                logger.warning(f"colab training run {training_run_id}: no checkpoint available yet at stop time", exc_info=True)
                best_path = None
            try:
                await loop.run_in_executor(None, colab_runner.stop_session, session_name)
            except Exception:
                logger.warning(f"colab training run {training_run_id}: failed to stop session {session_name}", exc_info=True)
            return {"progress": last_progress, "best_path": best_path}


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

            # A preprocessed-dataset recipe is the single source of truth for preprocessing/
            # feature_cols/normalize when referenced, overriding any same-named inline
            # hyperparams -- same resolution celery_worker.py's _resolve_training_context does
            # for a local run, including snapshotting the resolved values back onto
            # TrainingRun.hyperparams so this run stays self-describing even if its recipe is
            # later renamed or deleted.
            run_update_values = {"status": "running", "started_at": datetime.now(timezone.utc)}
            if run.preprocessed_dataset_id is not None:
                from model.models import PreprocessedDataset
                pd_rec = (await db.execute(
                    select(PreprocessedDataset).where(PreprocessedDataset.id == run.preprocessed_dataset_id)
                )).scalar_one_or_none()
                if pd_rec is not None:
                    # Mutate the same `hyperparams` dict Step 4 later writes back wholesale
                    # (`hyperparams={**hyperparams, "_external_ref": ...}`) -- putting these
                    # keys only in a copy here, as an earlier version of this code did, meant
                    # Step 4's write silently dropped preprocessed_dataset_id/_name again.
                    hyperparams["preprocessing"] = pd_rec.preprocessing
                    hyperparams["feature_cols"] = pd_rec.feature_cols
                    hyperparams["normalize"] = pd_rec.normalize
                    hyperparams["preprocessed_dataset_id"] = pd_rec.id
                    hyperparams["preprocessed_dataset_name"] = pd_rec.name
                    run_update_values["hyperparams"] = dict(hyperparams)

            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(**run_update_values))
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

        # --- 3. Run it on Colab, via colab-cli -- concurrently polled for progress/stop --------
        container_notebook_path = f"/notebooks/{session_name}.ipynb"

        await loop.run_in_executor(None, colab_runner.new_session, session_name)
        poll_task = asyncio.ensure_future(_poll_and_maybe_stop(training_run_id, session_name, notebooks_dir, factory))
        exec_task = asyncio.ensure_future(loop.run_in_executor(
            None, colab_runner.exec_notebook, session_name, container_notebook_path, timeout_seconds,
        ))
        stopped_early: dict | None = None
        try:
            done, _pending = await asyncio.wait({exec_task, poll_task}, return_when=asyncio.FIRST_COMPLETED)

            if poll_task in done:
                # stop_requested fired: poll_task already called colab_runner.stop_session, which
                # (confirmed live) makes the still-blocked exec_task raise ColabCliError shortly
                # after (the session it's talking to just got torn out from under it) -- expected,
                # not a real failure, so consume it rather than letting the outer except treat
                # this run as an error.
                stopped_early = poll_task.result()
                try:
                    await exec_task
                except colab_runner.ColabCliError:
                    logger.info(f"colab training run {training_run_id}: colab exec ended after stop_requested (expected)")
            else:
                poll_task.cancel()
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass
                exec_stdout = await exec_task
                # colab exec's own exit code reliability for an in-notebook failure is
                # unconfirmed (infra/colab-cli/README.md's known-unknowns) -- log the notebook's
                # own print output unconditionally so a silent in-notebook failure (e.g. the
                # sha256 assertion) is still visible here even when colab exec itself reports
                # success.
                logger.info(f"colab training run {training_run_id}: colab exec output:\n{exec_stdout}")

            best_path = notebooks_dir / f"{session_name}_best.pt"
            metrics_path = notebooks_dir / f"{session_name}_metrics.json"
            if stopped_early is None:
                await loop.run_in_executor(None, colab_runner.download, session_name, "best.pt", best_path)
                await loop.run_in_executor(None, colab_runner.download, session_name, "metrics.json", metrics_path)
        finally:
            if stopped_early is None:
                try:
                    await loop.run_in_executor(None, colab_runner.stop_session, session_name)
                except Exception:
                    logger.warning(f"colab training run {training_run_id}: failed to stop session {session_name}", exc_info=True)

        # --- 4. Pull the result back into this TrainingRun --------------------------------------
        if stopped_early is not None:
            # Same terminal status ("completed") a stop_requested local run ends up with --
            # celery_worker.py's _train_model breaks out of its epoch loop on stop_requested and
            # falls through to the same completion path every other run takes, rather than a
            # distinct "stopped" status; matching that here keeps /model/compare and any status
            # filter behaving the same regardless of execution_target.
            progress = stopped_early.get("progress")
            stopped_best_path = stopped_early.get("best_path")
            if progress is None or stopped_best_path is None or not stopped_best_path.is_file():
                raise RuntimeError(
                    "training run was stopped before any checkpoint/progress was recorded -- nothing to register"
                )
            best_epoch = progress["best_epoch"]
            val_loss = progress["best_val_loss"]
            num_params = None
            best_path = stopped_best_path
            external_ref_extra = {"stopped_early": True}
            epoch_metrics: list[dict] = []
        else:
            metadata = json.loads(metrics_path.read_text())
            best_epoch = metadata["best_epoch"]
            val_loss = metadata["val_loss"]
            num_params = metadata.get("num_params")
            external_ref_extra = metadata.get("external_ref", {})
            epoch_metrics = metadata.get("epoch_metrics", [])

        store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts")).resolve()
        checkpoint_dir = store / "models" / str(model_id) / f"training_{training_run_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dest = checkpoint_dir / "best.pt"
        dest.write_bytes(best_path.read_bytes())
        artifact_rel = dest.relative_to(store).as_posix()

        external_ref = {
            "platform": "colab",
            "dataset_snapshot_id": snapshot.id,
            "dataset_snapshot_sha256": snapshot.sha256,
            **external_ref_extra,
        }

        async with factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                status="completed", ended_at=datetime.now(timezone.utc),
                best_epoch=best_epoch, val_loss=val_loss,
                num_params=num_params,
                artifact_path=artifact_rel,
                hyperparams={**hyperparams, "_external_ref": external_ref},
            ))
            db.add(TrainingCheckpoint(
                training_run_id=training_run_id, epoch=best_epoch,
                metrics={"val_loss": val_loss, "source": "colab"},
                artifact_path=artifact_rel,
            ))
            # Only the normal-completion path has a full epoch-by-epoch record to import --
            # _poll_and_maybe_stop deliberately doesn't write TrainingRunMetric rows itself (see
            # its docstring), so a stopped-early run ends up with none, not duplicates.
            for m in epoch_metrics:
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
