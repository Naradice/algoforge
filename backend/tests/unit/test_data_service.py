"""Unit tests for DataService — repository mocked out."""
import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from fastapi import HTTPException

from data.service import DataService
from data.models import Datasource, Dataset, CollectionJobUpdate


PATCH_REPO = "data.service.data_repo"


# ── get_datasource ─────────────────────────────────────────────────────────────

class TestGetDatasource:
    @pytest.mark.asyncio
    async def test_raises_404_when_not_found(self):
        svc = DataService()
        with patch(PATCH_REPO) as repo:
            repo.get_datasource = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.get_datasource(AsyncMock(), 999)
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_datasource_when_found(self):
        svc = DataService()
        ds = MagicMock(spec=Datasource, id=1)
        with patch(PATCH_REPO) as repo:
            repo.get_datasource = AsyncMock(return_value=ds)
            result = await svc.get_datasource(AsyncMock(), 1)
        assert result is ds


# ── get_dataset ────────────────────────────────────────────────────────────────

class TestGetDataset:
    @pytest.mark.asyncio
    async def test_raises_404_when_not_found(self):
        svc = DataService()
        with patch(PATCH_REPO) as repo:
            repo.get_dataset = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.get_dataset(AsyncMock(), 999)
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_dataset_when_found(self):
        svc = DataService()
        ds = MagicMock(spec=Dataset, id=5)
        with patch(PATCH_REPO) as repo:
            repo.get_dataset = AsyncMock(return_value=ds)
            result = await svc.get_dataset(AsyncMock(), 5)
        assert result is ds


# ── get_collection_job ─────────────────────────────────────────────────────────

class TestGetCollectionJob:
    @pytest.mark.asyncio
    async def test_raises_404_when_not_found(self):
        svc = DataService()
        with patch(PATCH_REPO) as repo:
            repo.get_collection_job = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.get_collection_job(AsyncMock(), 99)
            assert exc.value.status_code == 404


# ── create_collection_job ──────────────────────────────────────────────────────

class TestCreateCollectionJob:
    @pytest.mark.asyncio
    async def test_raises_404_when_datasource_missing(self):
        svc = DataService()
        from data.models import CollectionJobCreate
        with patch(PATCH_REPO) as repo:
            repo.get_datasource = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.create_collection_job(AsyncMock(), CollectionJobCreate(datasource_id=99))
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_creates_job_when_datasource_exists(self):
        svc = DataService()
        from data.models import CollectionJobCreate, CollectionJob
        ds = MagicMock(spec=Datasource, id=1)
        job = MagicMock(spec=CollectionJob, id=1, datasource_id=1)
        with patch(PATCH_REPO) as repo:
            repo.get_datasource = AsyncMock(return_value=ds)
            repo.create_collection_job = AsyncMock(return_value=job)
            result = await svc.create_collection_job(
                AsyncMock(), CollectionJobCreate(datasource_id=1, schedule_cron="0 * * * *")
            )
        assert result is job
        repo.create_collection_job.assert_called_once()


# ── update_datasource ──────────────────────────────────────────────────────────

class TestUpdateDatasource:
    @pytest.mark.asyncio
    async def test_raises_404_when_datasource_missing(self):
        svc = DataService()
        from data.models import DatasourceUpdate
        with patch(PATCH_REPO) as repo:
            repo.get_datasource = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.update_datasource(AsyncMock(), 99, DatasourceUpdate(name="x"))
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_calls_update_repo_with_non_none_fields(self):
        svc = DataService()
        from data.models import DatasourceUpdate
        ds = MagicMock(spec=Datasource, id=1)
        updated = MagicMock(spec=Datasource, id=1)
        updated.name = "Updated"   # 'name' is reserved in MagicMock(); set as attr after creation
        db = AsyncMock()
        with patch(PATCH_REPO) as repo:
            repo.get_datasource = AsyncMock(return_value=ds)
            repo.update_datasource = AsyncMock(return_value=updated)

            result = await svc.update_datasource(db, 1, DatasourceUpdate(name="Updated"))

            repo.update_datasource.assert_called_once_with(ANY, 1, name="Updated")
            assert result.name == "Updated"


# ── trigger_analysis ───────────────────────────────────────────────────────────

class TestTriggerAnalysis:
    @pytest.mark.asyncio
    async def test_raises_422_when_dataset_not_ready(self):
        svc = DataService()
        ds = MagicMock(spec=Dataset, id=1, status="pending")
        with patch(PATCH_REPO) as repo:
            repo.get_dataset = AsyncMock(return_value=ds)
            with pytest.raises(HTTPException) as exc:
                await svc.trigger_analysis(AsyncMock(), 1)
            assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_enqueues_task_when_ready(self):
        svc = DataService()
        ds = MagicMock(spec=Dataset, id=1, status="ready")
        # enqueue is imported inside the function body, so patch at arq_pool module level
        with patch(PATCH_REPO) as repo, patch("arq_pool.enqueue", new=AsyncMock()) as mock_enqueue:
            repo.get_dataset = AsyncMock(return_value=ds)
            result = await svc.trigger_analysis(AsyncMock(), 1)
            mock_enqueue.assert_called_once_with("compute_characteristics", 1)
            assert result is ds

    @pytest.mark.asyncio
    async def test_raises_404_when_dataset_not_found(self):
        svc = DataService()
        with patch(PATCH_REPO) as repo:
            repo.get_dataset = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.trigger_analysis(AsyncMock(), 999)
            assert exc.value.status_code == 404
