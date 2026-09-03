"""MCP tools — ML model inspection."""

from __future__ import annotations

import sqlalchemy as sa

from mcp_server import mcp


@mcp.tool()
async def list_models() -> list[dict]:
    """
    List all ML models with their architecture and deployment status.
    Use this to find model IDs before calling other model tools.
    """
    from database import async_session_factory
    from model.models import MLModel

    async with async_session_factory() as db:
        rows = (
            await db.execute(sa.select(MLModel).order_by(MLModel.created_at.desc()))
        ).scalars().all()

    return [
        {
            "id": m.id,
            "name": m.name,
            "architecture": m.architecture,
            "status": m.status,
            "artifact_path": m.artifact_path,
            "created_at": m.created_at.isoformat(),
        }
        for m in rows
    ]


@mcp.tool()
async def get_model_training_runs(model_id: int) -> list[dict]:
    """
    Get training run history for a model, ordered newest first.
    Shows training status, best epoch, validation loss, and hyperparameters.

    Args:
        model_id: ID of the ML model.
    """
    from database import async_session_factory
    from model.models import TrainingRun

    async with async_session_factory() as db:
        rows = (
            await db.execute(
                sa.select(TrainingRun)
                .where(TrainingRun.model_id == model_id)
                .order_by(TrainingRun.created_at.desc())
            )
        ).scalars().all()

    return [
        {
            "id": r.id,
            "status": r.status,
            "dataset_id": r.dataset_id,
            "hyperparams": r.hyperparams,
            "current_epoch": r.current_epoch,
            "best_epoch": r.best_epoch,
            "val_loss": r.val_loss,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "ended_at": r.ended_at.isoformat() if r.ended_at else None,
            "artifact_path": r.artifact_path,
        }
        for r in rows
    ]


@mcp.tool()
async def get_model_validations(model_id: int) -> list[dict]:
    """
    Get validation metrics for a model (directional accuracy, MAE, RMSE, Sharpe proxy, etc.).
    For GAN models: ACF match, Hurst exponent difference, kurtosis comparison.

    Args:
        model_id: ID of the ML model.
    """
    from database import async_session_factory
    from model.models import ModelValidation

    async with async_session_factory() as db:
        rows = (
            await db.execute(
                sa.select(ModelValidation)
                .where(ModelValidation.model_id == model_id)
                .order_by(ModelValidation.computed_at.desc())
            )
        ).scalars().all()

    if not rows:
        return [{"message": f"No validations found for model {model_id}. Run a validation job first."}]

    return [
        {
            "id": v.id,
            "training_run_id": v.training_run_id,
            "dataset_id": v.dataset_id,
            "computed_at": v.computed_at.isoformat(),
            "metrics": v.metrics,
            "interpretation": _interpret_validation(v.metrics),
        }
        for v in rows
    ]


@mcp.tool()
async def compare_model_runs(model_id: int) -> dict:
    """
    Compare all training runs for a model to identify the best performing configuration.
    Returns a ranked list with val_loss and directional accuracy if available.

    Args:
        model_id: ID of the ML model.
    """
    from database import async_session_factory
    from model.models import TrainingRun, ModelValidation

    async with async_session_factory() as db:
        runs = (
            await db.execute(
                sa.select(TrainingRun)
                .where(TrainingRun.model_id == model_id, TrainingRun.status == "completed")
                .order_by(TrainingRun.val_loss)
            )
        ).scalars().all()

        if not runs:
            return {"message": f"No completed training runs for model {model_id}"}

        # Get validations keyed by training_run_id
        val_rows = (
            await db.execute(
                sa.select(ModelValidation).where(ModelValidation.model_id == model_id)
            )
        ).scalars().all()

    val_map: dict[int, dict] = {v.training_run_id: v.metrics for v in val_rows if v.training_run_id}

    ranked = []
    for r in runs:
        val = val_map.get(r.id, {})
        ranked.append({
            "run_id": r.id,
            "best_epoch": r.best_epoch,
            "val_loss": r.val_loss,
            "directional_accuracy": val.get("directional_accuracy"),
            "sharpe_proxy": val.get("sharpe_proxy"),
            "hyperparams": r.hyperparams,
        })

    return {
        "model_id": model_id,
        "total_runs": len(ranked),
        "best_run_id": ranked[0]["run_id"] if ranked else None,
        "ranked": ranked,
    }


def _interpret_validation(metrics: dict) -> str:
    parts = []
    if "directional_accuracy" in metrics:
        da = metrics["directional_accuracy"] * 100
        parts.append(f"Directional accuracy {da:.1f}% ({'above' if da > 50 else 'below'} random)")
    if "mae" in metrics:
        parts.append(f"MAE {metrics['mae']:.6f}")
    if "sharpe_proxy" in metrics:
        sp = metrics["sharpe_proxy"]
        parts.append(f"Sharpe proxy {sp:.3f}")
    if "acf_match" in metrics:
        parts.append(f"ACF match {metrics['acf_match']:.3f} (1.0 = perfect)")
    if "hurst_diff" in metrics:
        parts.append(f"Hurst diff {metrics['hurst_diff']:.3f} (0.0 = identical)")
    return "; ".join(parts) if parts else "Raw metrics available."


@mcp.tool()
async def create_model(name: str, architecture: str, config: dict) -> dict:
    """
    Create a new ML model.

    Args:
        name:         Model name.
        architecture: Architecture type: "seq2seq_transformer", "lstm", "timegan", "rl_agent".
        config:       Architecture-specific configuration dict.
    """
    from database import async_session_factory
    from model.service import model_service
    from model.models import MLModelCreate

    async with async_session_factory() as db:
        body = MLModelCreate(name=name, architecture=architecture, config=config)
        m = await model_service.create_model(db, body)
        return {"id": m.id, "name": m.name, "architecture": m.architecture, "status": m.status}


@mcp.tool()
async def start_training_run(
    model_id: int,
    hyperparams: dict,
    dataset_id: int | None = None,
    preprocessed_dataset_id: int | None = None,
    execution_target: str = "local",
) -> dict:
    """
    Start a training run for a model.

    Either dataset_id or preprocessed_dataset_id must be given. Prefer
    preprocessed_dataset_id when a saved preprocessing recipe already exists
    (see list_preprocessed_datasets) — it determines dataset_id, feature
    columns, and normalization automatically, and its structure
    characteristics are reused instead of recomputed.

    Args:
        model_id:                 Model to train.
        hyperparams:               Training hyperparameters (epochs, lr, batch_size, obs_len,
                                    pred_len, etc.). Omit feature_cols/preprocessing/normalize
                                    when preprocessed_dataset_id is set — they come from the recipe.
        dataset_id:                Raw dataset to train on (ad-hoc, no saved recipe).
        preprocessed_dataset_id:   A saved preprocessing recipe (see list_preprocessed_datasets).
        execution_target:          "local" (default): this backend's own training queue/GPU-CPU
                                    worker. "colab": run on a Google Colab CPU runtime instead —
                                    this backend automatically exports a dataset snapshot to
                                    Drive, generates a notebook, runs it via colab-cli, and
                                    imports the result, all before returning control (see
                                    docs/colab-workflow.md). Only architecture="lstm" with no
                                    preprocessed_dataset_id/token_level/preprocessing recipe and
                                    split_mode="chronological" is supported for "colab" today —
                                    an unsupported combination raises immediately rather than
                                    starting a run that would fail partway through.
    """
    from database import async_session_factory
    from model.service import model_service
    from model.models import TrainingRunCreate
    from celery_app import enqueue

    async with async_session_factory() as db:
        body = TrainingRunCreate(
            dataset_id=dataset_id, preprocessed_dataset_id=preprocessed_dataset_id,
            hyperparams=hyperparams, execution_target=execution_target,
        )
        run = await model_service.create_training_run(db, model_id, body)
        task_name = "colab_train_model" if run.execution_target == "colab" else "train_model"
        await enqueue(task_name, run.id)
        return {"run_id": run.id, "model_id": model_id, "status": run.status, "execution_target": run.execution_target}


@mcp.tool()
async def list_preprocessed_datasets(dataset_id: int | None = None) -> list[dict]:
    """
    List saved preprocessing recipes (named indicator/clustering/feature_cols/normalize
    configs that can be reused across training runs instead of specifying preprocessing
    inline every time). Use this before start_training_run to find an existing recipe,
    or to see whether one needs to be created first.

    Args:
        dataset_id: Optional — only recipes built on this raw dataset.
    """
    from database import async_session_factory
    from model.service import model_service

    async with async_session_factory() as db:
        items, _ = await model_service.list_preprocessed_datasets(db, dataset_id=dataset_id, limit=200)

    return [
        {
            "id": p.id,
            "name": p.name,
            "dataset_id": p.dataset_id,
            "preprocessing": p.preprocessing,
            "feature_cols": p.feature_cols,
            "normalize": p.normalize,
            "status": p.status,
            "created_at": p.created_at.isoformat(),
        }
        for p in items
    ]


@mcp.tool()
async def get_preprocessed_dataset(preprocessed_dataset_id: int) -> dict:
    """
    Get a saved preprocessing recipe's config and computed structure characteristics
    (Hurst exponent, spectral periodicity, wavelet multiscale structure,
    entropy/nonlinearity, regime changes) — the same analyses as get_dataset_characteristics,
    but computed on the actual preprocessed series this recipe produces.

    Args:
        preprocessed_dataset_id: ID of the recipe (see list_preprocessed_datasets).
    """
    from database import async_session_factory
    from model.service import model_service

    async with async_session_factory() as db:
        p = await model_service.get_preprocessed_dataset(db, preprocessed_dataset_id)

    return {
        "id": p.id,
        "name": p.name,
        "dataset_id": p.dataset_id,
        "preprocessing": p.preprocessing,
        "feature_cols": p.feature_cols,
        "normalize": p.normalize,
        "status": p.status,
        "characteristics": p.characteristics,
        "created_at": p.created_at.isoformat(),
    }


@mcp.tool()
async def get_training_status(training_run_id: int) -> dict:
    """
    Get the current status of a training run.
    Call repeatedly until status is "completed" or "error".

    Args:
        training_run_id: ID of the training run.
    """
    from database import async_session_factory
    from model.service import model_service

    async with async_session_factory() as db:
        run = await model_service.get_training_run_by_id(db, training_run_id)
    return {
        "status": run.status,
        "current_epoch": run.current_epoch,
        "best_epoch": run.best_epoch,
        "val_loss": run.val_loss,
        "stop_requested": run.stop_requested,
    }


@mcp.tool()
async def stop_training_run(training_run_id: int) -> dict:
    """
    Request graceful stop of a training run.
    The trainer will complete the current epoch then stop.

    Args:
        training_run_id: ID of the training run to stop.
    """
    from database import async_session_factory
    from model.service import model_service

    async with async_session_factory() as db:
        run = await model_service.stop_training_run(db, training_run_id)
    return {"run_id": run.id, "status": run.status, "stop_requested": run.stop_requested}


@mcp.tool()
async def deploy_model(model_id: int, training_run_id: int) -> dict:
    """
    Deploy a trained model, making it available for inference and strategy backtests.

    Args:
        model_id:        Model to deploy.
        training_run_id: Completed training run whose artifact to use.
    """
    from database import async_session_factory
    from model.service import model_service

    async with async_session_factory() as db:
        m = await model_service.deploy_model(db, model_id, training_run_id)
    return {"id": m.id, "name": m.name, "status": m.status, "artifact_path": m.artifact_path}


@mcp.tool()
async def predict(model_id: int, features: list[list[float]], feature_names: list[str]) -> dict:
    """
    Run inference on a deployed model.

    Args:
        model_id:       ID of the deployed model.
        features:       2D array of feature values (rows = samples, cols = features).
        feature_names:  Names matching the column order.
    """
    import asyncio
    from database import async_session_factory
    from model.service import model_service
    from model.inference import predict as run_predict

    async with async_session_factory() as db:
        model_rec = await model_service.get_model(db, model_id)
        training_runs, _ = await model_service.list_training_runs(db, model_id, limit=100)
        deployed_run = next((r for r in training_runs if r.artifact_path == model_rec.artifact_path), None)
        hyperparams = deployed_run.hyperparams if deployed_run else {}

    loop = asyncio.get_event_loop()
    preds = await loop.run_in_executor(
        None, run_predict,
        model_id, model_rec.architecture, model_rec.config,
        model_rec.artifact_path, hyperparams, features, feature_names,
    )
    return {"predictions": preds}


@mcp.tool()
async def start_hyperparameter_search(model_id: int, dataset_id: int, search_grid: dict) -> dict:
    """
    Start a hyperparameter search, creating one training run per grid combination.

    Args:
        model_id:    Model to search hyperparams for.
        dataset_id:  Dataset to train on.
        search_grid: Dict mapping param names to lists of values.
                     Example: {"lr": [0.001, 0.0001], "batch_size": [32, 64]}
    """
    from database import async_session_factory
    from model.service import model_service
    from model.models import HyperparamSearchCreate
    from celery_app import enqueue

    async with async_session_factory() as db:
        body = HyperparamSearchCreate(model_id=model_id, dataset_id=dataset_id, search_grid=search_grid)
        run_ids = await model_service.create_search_runs(db, body)
        for run_id in run_ids:
            await enqueue("train_model", run_id)
    return {"run_ids": run_ids, "count": len(run_ids)}
