"""Unit tests for ModelService — repository mocked out."""
import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from fastapi import HTTPException

from model.service import ModelService
from model.models import (
    MLModel, TrainingRun, ModelValidation, MLModelUpdate,
    TrainingRunCreate, PreprocessedDataset, PreprocessedDatasetCreate, PreprocessedDatasetUpdate,
)


PATCH = "model.service.model_repo"


def _db_with_scalar(value):
    """AsyncMock db whose db.execute(...) awaits to an object with a *synchronous*
    scalar_one_or_none() — matches real SQLAlchemy Result, unlike a bare AsyncMock() where
    every attribute defaults to another AsyncMock and calling it returns an un-awaited coroutine."""
    db = AsyncMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none = MagicMock(return_value=value)
    db.execute = AsyncMock(return_value=exec_result)
    return db


# ── get_model ──────────────────────────────────────────────────────────────────

class TestGetModel:
    @pytest.mark.asyncio
    async def test_raises_404_when_not_found(self):
        svc = ModelService()
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.get_model(AsyncMock(), 999)
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_model_when_found(self):
        svc = ModelService()
        model = MagicMock(spec=MLModel, id=1)
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=model)
            result = await svc.get_model(AsyncMock(), 1)
        assert result is model

    @pytest.mark.asyncio
    async def test_calls_repo_with_correct_id(self):
        svc = ModelService()
        db = AsyncMock()
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=MagicMock(spec=MLModel))
            await svc.get_model(db, 7)
            repo.get_by_id.assert_called_once_with(db, 7)


# ── get_training_run_by_id ─────────────────────────────────────────────────────

class TestGetTrainingRunById:
    @pytest.mark.asyncio
    async def test_raises_404_when_not_found(self):
        svc = ModelService()
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.get_training_run_by_id(AsyncMock(), 99)
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_run_when_found(self):
        svc = ModelService()
        run = MagicMock(spec=TrainingRun, id=5)
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=run)
            result = await svc.get_training_run_by_id(AsyncMock(), 5)
        assert result is run


# ── stop_training_run ──────────────────────────────────────────────────────────

class TestStopTrainingRun:
    @pytest.mark.asyncio
    async def test_sets_stop_requested_for_pending_run(self):
        svc = ModelService()
        run = MagicMock(spec=TrainingRun, id=1, status="pending")
        updated = MagicMock(spec=TrainingRun, stop_requested=True)
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=run)
            repo.update_training_run = AsyncMock(return_value=updated)

            result = await svc.stop_training_run(AsyncMock(), 1)

            repo.update_training_run.assert_called_once_with(ANY, 1, stop_requested=True)
            assert result.stop_requested is True

    @pytest.mark.asyncio
    async def test_sets_stop_requested_for_running_run(self):
        svc = ModelService()
        run = MagicMock(spec=TrainingRun, id=2, status="running")
        updated = MagicMock(spec=TrainingRun, stop_requested=True)
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=run)
            repo.update_training_run = AsyncMock(return_value=updated)

            await svc.stop_training_run(AsyncMock(), 2)
            repo.update_training_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_422_when_already_completed(self):
        svc = ModelService()
        run = MagicMock(spec=TrainingRun, id=1, status="completed")
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=run)

            with pytest.raises(HTTPException) as exc:
                await svc.stop_training_run(AsyncMock(), 1)
            assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_raises_422_when_already_stopped(self):
        svc = ModelService()
        run = MagicMock(spec=TrainingRun, id=1, status="stopped")
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=run)

            with pytest.raises(HTTPException) as exc:
                await svc.stop_training_run(AsyncMock(), 1)
            assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_raises_404_when_run_not_found(self):
        svc = ModelService()
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc:
                await svc.stop_training_run(AsyncMock(), 999)
            assert exc.value.status_code == 404


# ── deploy_model ───────────────────────────────────────────────────────────────

class TestDeployModel:
    @pytest.mark.asyncio
    async def test_raises_422_when_training_run_not_completed(self):
        svc = ModelService()
        model = MagicMock(spec=MLModel, id=1)
        run = MagicMock(spec=TrainingRun, id=1, model_id=1, status="running", artifact_path=None)
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=model)
            repo.get_training_run = AsyncMock(return_value=run)

            with pytest.raises(HTTPException) as exc:
                await svc.deploy_model(AsyncMock(), 1, 1)
            assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_raises_404_when_training_run_missing(self):
        svc = ModelService()
        model = MagicMock(spec=MLModel, id=1)
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=model)
            repo.get_training_run = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc:
                await svc.deploy_model(AsyncMock(), 1, 99)
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_raises_404_when_run_belongs_to_different_model(self):
        svc = ModelService()
        model = MagicMock(spec=MLModel, id=1)
        run = MagicMock(spec=TrainingRun, id=1, model_id=99, status="completed")
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=model)
            repo.get_training_run = AsyncMock(return_value=run)

            with pytest.raises(HTTPException) as exc:
                await svc.deploy_model(AsyncMock(), 1, 1)
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_deploys_completed_run_successfully(self):
        svc = ModelService()
        model = MagicMock(spec=MLModel, id=1)
        run = MagicMock(spec=TrainingRun, id=1, model_id=1, status="completed", artifact_path="/path/model.pt")
        deployed = MagicMock(spec=MLModel, status="deployed")
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=model)
            repo.get_training_run = AsyncMock(return_value=run)
            repo.update = AsyncMock(return_value=deployed)

            result = await svc.deploy_model(AsyncMock(), 1, 1)

            repo.update.assert_called_once()
            assert result.status == "deployed"


# ── compare_training_runs ──────────────────────────────────────────────────────

class TestCompareTrainingRuns:
    @pytest.mark.asyncio
    async def test_returns_run_data_for_valid_ids(self):
        svc = ModelService()
        run = MagicMock(spec=TrainingRun, id=1, model_id=1, dataset_id=2,
                        hyperparams={}, status="completed", best_epoch=10,
                        val_loss=0.1, num_params=None, preprocessed_characteristics=None, artifact_path="/path")
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=run)
            repo.get_by_id = AsyncMock(return_value=None)
            repo.get_latest_validation_for_run = AsyncMock(return_value=None)

            result = await svc.compare_training_runs(AsyncMock(), [1])

        assert result[0]["run_id"] == 1
        assert result[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_returns_error_entry_for_missing_run(self):
        svc = ModelService()
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=None)

            result = await svc.compare_training_runs(AsyncMock(), [999])

        assert result[0]["run_id"] == 999
        assert result[0]["error"] == "not found"

    @pytest.mark.asyncio
    async def test_handles_mixed_valid_and_missing_runs(self):
        svc = ModelService()
        run = MagicMock(spec=TrainingRun, id=1, model_id=1, dataset_id=1,
                        hyperparams={}, status="completed", best_epoch=5,
                        val_loss=0.2, num_params=None, artifact_path=None)

        def side_effect(db, run_id):
            if run_id == 1:
                return run
            return None

        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(side_effect=side_effect)
            repo.get_by_id = AsyncMock(return_value=None)
            repo.get_latest_validation_for_run = AsyncMock(return_value=None)

            result = await svc.compare_training_runs(AsyncMock(), [1, 999])

        assert result[0]["run_id"] == 1
        assert "error" not in result[0]
        assert result[1]["error"] == "not found"

    @pytest.mark.asyncio
    async def test_includes_model_size_and_architecture_when_model_resolves(self):
        svc = ModelService()
        run = MagicMock(spec=TrainingRun, id=1, model_id=7, dataset_id=2, preprocessed_dataset_id=9,
                        hyperparams={}, status="completed", best_epoch=10,
                        val_loss=0.1, num_params=184320, artifact_path="/path")
        model = MagicMock(spec=MLModel, id=7, architecture="lstm")
        model.name = "my-lstm"
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=run)
            repo.get_by_id = AsyncMock(return_value=model)
            repo.get_latest_validation_for_run = AsyncMock(return_value=None)

            result = await svc.compare_training_runs(AsyncMock(), [1])

        assert result[0]["num_params"] == 184320
        assert result[0]["architecture"] == "lstm"
        assert result[0]["model_name"] == "my-lstm"
        assert result[0]["validation"] is None
        assert result[0]["dataset_id"] == 2
        assert result[0]["preprocessed_dataset_id"] == 9

    @pytest.mark.asyncio
    async def test_includes_latest_validation_metrics_when_present(self):
        svc = ModelService()
        run = MagicMock(spec=TrainingRun, id=1, model_id=7, dataset_id=2,
                        hyperparams={}, status="completed", best_epoch=10,
                        val_loss=0.1, num_params=None, artifact_path="/path")
        validation = MagicMock(spec=ModelValidation, metrics={"directional_accuracy": 0.55, "sharpe_proxy": 1.2})
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=run)
            repo.get_by_id = AsyncMock(return_value=None)
            repo.get_latest_validation_for_run = AsyncMock(return_value=validation)

            result = await svc.compare_training_runs(AsyncMock(), [1])

        assert result[0]["validation"] == {"directional_accuracy": 0.55, "sharpe_proxy": 1.2}

    @pytest.mark.asyncio
    async def test_includes_preprocessed_characteristics_when_present(self):
        svc = ModelService()
        chars = {"long_range_dependence": {"hurst": 0.58}, "regime_changes": {"n_changepoints": 4}}
        run = MagicMock(spec=TrainingRun, id=1, model_id=7, dataset_id=2,
                        hyperparams={}, status="completed", best_epoch=10,
                        val_loss=0.1, num_params=None, preprocessed_characteristics=chars, artifact_path="/path")
        with patch(PATCH) as repo:
            repo.get_training_run_by_id = AsyncMock(return_value=run)
            repo.get_by_id = AsyncMock(return_value=None)
            repo.get_latest_validation_for_run = AsyncMock(return_value=None)

            result = await svc.compare_training_runs(AsyncMock(), [1])

        assert result[0]["preprocessed_characteristics"] == chars


# ── create_training_run ─────────────────────────────────────────────────────────

class TestCreateTrainingRun:
    @pytest.mark.asyncio
    async def test_derives_dataset_id_from_preprocessed_dataset(self):
        svc = ModelService()
        model = MagicMock(spec=MLModel, id=1)
        pd = MagicMock(spec=PreprocessedDataset, id=5, dataset_id=42)
        created_run = MagicMock(spec=TrainingRun, id=10)
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=model)
            repo.get_preprocessed_dataset_by_id = AsyncMock(return_value=pd)
            repo.create_training_run = AsyncMock(return_value=created_run)

            body = TrainingRunCreate(preprocessed_dataset_id=5, hyperparams={"epochs": 1})
            result = await svc.create_training_run(AsyncMock(), 1, body)

            repo.create_training_run.assert_called_once_with(
                ANY, model_id=1, dataset_id=42, preprocessed_dataset_id=5, hyperparams={"epochs": 1},
                execution_target="local",
            )
        assert result is created_run

    @pytest.mark.asyncio
    async def test_uses_explicit_dataset_id_when_no_recipe(self):
        svc = ModelService()
        model = MagicMock(spec=MLModel, id=1)
        created_run = MagicMock(spec=TrainingRun, id=11)
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=model)
            repo.create_training_run = AsyncMock(return_value=created_run)

            body = TrainingRunCreate(dataset_id=7, hyperparams={})
            await svc.create_training_run(AsyncMock(), 1, body)

            repo.create_training_run.assert_called_once_with(
                ANY, model_id=1, dataset_id=7, preprocessed_dataset_id=None, hyperparams={},
                execution_target="local",
            )

    @pytest.mark.asyncio
    async def test_raises_422_when_neither_dataset_id_nor_recipe_given(self):
        svc = ModelService()
        model = MagicMock(spec=MLModel, id=1)
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=model)

            body = TrainingRunCreate(hyperparams={})
            with pytest.raises(HTTPException) as exc:
                await svc.create_training_run(AsyncMock(), 1, body)
            assert exc.value.status_code == 422


# ── preprocessed datasets ───────────────────────────────────────────────────────

class TestListPreprocessedDatasets:
    @pytest.mark.asyncio
    async def test_delegates_to_repo_with_dataset_filter(self):
        svc = ModelService()
        with patch(PATCH) as repo:
            repo.get_preprocessed_datasets = AsyncMock(return_value=([], 0))
            await svc.list_preprocessed_datasets(AsyncMock(), dataset_id=3, offset=0, limit=20)
            repo.get_preprocessed_datasets.assert_called_once_with(ANY, dataset_id=3, offset=0, limit=20)


class TestCreatePreprocessedDataset:
    @pytest.mark.asyncio
    async def test_creates_and_enqueues_when_dataset_exists(self):
        svc = ModelService()
        dataset_mock = MagicMock(id=3)
        db = _db_with_scalar(dataset_mock)
        created = MagicMock(spec=PreprocessedDataset, id=9)
        with patch(PATCH) as repo, patch("celery_app.enqueue", new=AsyncMock()) as enqueue_mock:
            repo.create_preprocessed_dataset = AsyncMock(return_value=created)

            body = PreprocessedDatasetCreate(name="r1", dataset_id=3, preprocessing={}, feature_cols=["close"], normalize="zscore")
            result = await svc.create_preprocessed_dataset(db, body)

            repo.create_preprocessed_dataset.assert_called_once_with(
                db, name="r1", dataset_id=3, preprocessing={}, feature_cols=["close"], normalize="zscore", status="pending",
            )
            enqueue_mock.assert_called_once_with("compute_preprocessed_characteristics", 9)
        assert result is created

    @pytest.mark.asyncio
    async def test_raises_404_when_base_dataset_missing(self):
        svc = ModelService()
        db = _db_with_scalar(None)
        with patch(PATCH) as repo:
            repo.create_preprocessed_dataset = AsyncMock()

            body = PreprocessedDatasetCreate(name="r1", dataset_id=999)
            with pytest.raises(HTTPException) as exc:
                await svc.create_preprocessed_dataset(db, body)
            assert exc.value.status_code == 404
            repo.create_preprocessed_dataset.assert_not_called()


class TestUpdatePreprocessedDataset:
    @pytest.mark.asyncio
    async def test_updates_name_only(self):
        svc = ModelService()
        pd = MagicMock(spec=PreprocessedDataset, id=1)
        updated = MagicMock(spec=PreprocessedDataset, id=1)
        updated.name = "renamed"
        with patch(PATCH) as repo:
            repo.get_preprocessed_dataset_by_id = AsyncMock(return_value=pd)
            repo.update_preprocessed_dataset = AsyncMock(return_value=updated)

            result = await svc.update_preprocessed_dataset(AsyncMock(), 1, PreprocessedDatasetUpdate(name="renamed"))

            repo.update_preprocessed_dataset.assert_called_once_with(ANY, 1, name="renamed")
        assert result.name == "renamed"


class TestDeletePreprocessedDataset:
    @pytest.mark.asyncio
    async def test_deletes_when_unreferenced(self):
        svc = ModelService()
        pd = MagicMock(spec=PreprocessedDataset, id=1)
        db = _db_with_scalar(None)  # no referencing TrainingRun
        with patch(PATCH) as repo:
            repo.get_preprocessed_dataset_by_id = AsyncMock(return_value=pd)
            repo.delete_preprocessed_dataset = AsyncMock()

            await svc.delete_preprocessed_dataset(db, 1)
            repo.delete_preprocessed_dataset.assert_called_once_with(db, 1)

    @pytest.mark.asyncio
    async def test_raises_409_when_referenced_by_training_run(self):
        svc = ModelService()
        pd = MagicMock(spec=PreprocessedDataset, id=1)
        ref_run = MagicMock(spec=TrainingRun, id=77)
        db = _db_with_scalar(ref_run)
        with patch(PATCH) as repo:
            repo.get_preprocessed_dataset_by_id = AsyncMock(return_value=pd)
            repo.delete_preprocessed_dataset = AsyncMock()

            with pytest.raises(HTTPException) as exc:
                await svc.delete_preprocessed_dataset(db, 1)
            assert exc.value.status_code == 409
            repo.delete_preprocessed_dataset.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_404_when_not_found(self):
        svc = ModelService()
        with patch(PATCH) as repo:
            repo.get_preprocessed_dataset_by_id = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.delete_preprocessed_dataset(AsyncMock(), 999)
            assert exc.value.status_code == 404


class TestRecomputePreprocessedCharacteristics:
    @pytest.mark.asyncio
    async def test_enqueues_recompute(self):
        svc = ModelService()
        pd = MagicMock(spec=PreprocessedDataset, id=1)
        with patch(PATCH) as repo, patch("celery_app.enqueue", new=AsyncMock()) as enqueue_mock:
            repo.get_preprocessed_dataset_by_id = AsyncMock(return_value=pd)

            result = await svc.recompute_preprocessed_characteristics(AsyncMock(), 1)

            enqueue_mock.assert_called_once_with("compute_preprocessed_characteristics", 1)
        assert result is pd
