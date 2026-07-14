"""Unit tests for ModelService — repository mocked out."""
import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from fastapi import HTTPException

from model.service import ModelService
from model.models import MLModel, TrainingRun, ModelValidation, MLModelUpdate


PATCH = "model.service.model_repo"


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
        run = MagicMock(spec=TrainingRun, id=1, model_id=7, dataset_id=2,
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
