"""Unit tests for DataService — repository mocked out."""
import os
import tempfile
from pathlib import Path
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
        # enqueue is imported inside the function body, so patch at celery_app module level
        with patch(PATCH_REPO) as repo, patch("celery_app.enqueue", new=AsyncMock()) as mock_enqueue:
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


# ── get_dataset_preview ────────────────────────────────────────────────────────

class TestGetDatasetPreview:
    """get_dataset_preview() should work for both 'ready' and 'running' datasets."""

    def _make_parquet(self, tmpdir: str, status: str = "ready") -> tuple:
        """Create a minimal parquet file and return (db_mock, dataset_mock)."""
        import pandas as pd
        import numpy as np

        store = Path(tmpdir)
        artifact_path = "test_ohlc.parquet"
        full = store / artifact_path

        # Minimal OHLC data
        idx = pd.date_range("2000-01-03", periods=10, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "open": np.ones(10) * 100,
            "high": np.ones(10) * 101,
            "low": np.ones(10) * 99,
            "close": np.ones(10) * 100,
            "volume": np.ones(10, dtype=int) * 30,
        }, index=idx)
        df.index.name = "datetime"
        df.to_parquet(full)

        ds = MagicMock(spec=Dataset)
        ds.id = 1
        ds.artifact_path = artifact_path
        ds.status = status
        ds.timeframe = "M1"
        return ds

    @pytest.mark.asyncio
    async def test_preview_works_for_ready_dataset(self):
        svc = DataService()
        with tempfile.TemporaryDirectory() as tmpdir:
            import data.service as svc_mod
            orig = os.environ.get("ARTIFACT_STORE_PATH")
            os.environ["ARTIFACT_STORE_PATH"] = tmpdir
            try:
                ds = self._make_parquet(tmpdir, status="ready")
                with patch(PATCH_REPO) as repo:
                    repo.get_dataset = AsyncMock(return_value=ds)
                    rows = await svc.get_dataset_preview(AsyncMock(), 1, rows=5)
                assert len(rows) == 5
                assert "close" in rows[0]
            finally:
                if orig is None:
                    os.environ.pop("ARTIFACT_STORE_PATH", None)
                else:
                    os.environ["ARTIFACT_STORE_PATH"] = orig

    @pytest.mark.asyncio
    async def test_preview_works_for_running_dataset(self):
        """status='running' must NOT raise 422 — live preview is supported."""
        svc = DataService()
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.environ.get("ARTIFACT_STORE_PATH")
            os.environ["ARTIFACT_STORE_PATH"] = tmpdir
            try:
                ds = self._make_parquet(tmpdir, status="running")
                with patch(PATCH_REPO) as repo:
                    repo.get_dataset = AsyncMock(return_value=ds)
                    rows = await svc.get_dataset_preview(AsyncMock(), 1, rows=5)
                assert len(rows) == 5
            finally:
                if orig is None:
                    os.environ.pop("ARTIFACT_STORE_PATH", None)
                else:
                    os.environ["ARTIFACT_STORE_PATH"] = orig

    @pytest.mark.asyncio
    async def test_preview_raises_422_for_pending_dataset(self):
        """status='pending' (no artifact yet) must raise 422."""
        svc = DataService()
        ds = MagicMock(spec=Dataset)
        ds.id = 1
        ds.artifact_path = None
        ds.status = "pending"
        with patch(PATCH_REPO) as repo:
            repo.get_dataset = AsyncMock(return_value=ds)
            with pytest.raises(HTTPException) as exc:
                await svc.get_dataset_preview(AsyncMock(), 1)
            assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_preview_raises_404_when_dataset_not_found(self):
        svc = DataService()
        with patch(PATCH_REPO) as repo:
            repo.get_dataset = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.get_dataset_preview(AsyncMock(), 999)
            assert exc.value.status_code == 404


class TestGetDatasetDownloadInfo:
    @pytest.mark.asyncio
    async def test_returns_file_info_for_file_artifact(self):
        svc = DataService()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_name = "dataset.parquet"
            artifact_path = Path(tmpdir) / artifact_name
            artifact_path.write_bytes(b"parquet")
            ds = MagicMock(spec=Dataset)
            ds.id = 7
            ds.name = "Test Dataset"
            ds.artifact_path = artifact_name
            ds.status = "ready"

            orig = os.environ.get("ARTIFACT_STORE_PATH")
            os.environ["ARTIFACT_STORE_PATH"] = tmpdir
            try:
                with patch(PATCH_REPO) as repo:
                    repo.get_dataset = AsyncMock(return_value=ds)
                    info = await svc.get_dataset_download_info(AsyncMock(), 7)
            finally:
                if orig is None:
                    os.environ.pop("ARTIFACT_STORE_PATH", None)
                else:
                    os.environ["ARTIFACT_STORE_PATH"] = orig

        assert info["artifact_kind"] == "file"
        assert info["download_filename"] == "dataset.parquet"
        assert info["download_format"] == "parquet"

    @pytest.mark.asyncio
    async def test_returns_zip_info_for_directory_artifact(self):
        svc = DataService()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_name = os.path.join("datasets", "src_5", "ddm_ticks")
            artifact_path = Path(tmpdir) / artifact_name
            artifact_path.mkdir(parents=True)
            (artifact_path / "_meta.json").write_text("{}", encoding="utf-8")

            ds = MagicMock(spec=Dataset)
            ds.id = 5
            ds.name = "DDM Dataset"
            ds.artifact_path = artifact_name
            ds.status = "ready"

            orig = os.environ.get("ARTIFACT_STORE_PATH")
            os.environ["ARTIFACT_STORE_PATH"] = tmpdir
            try:
                with patch(PATCH_REPO) as repo:
                    repo.get_dataset = AsyncMock(return_value=ds)
                    info = await svc.get_dataset_download_info(AsyncMock(), 5)
            finally:
                if orig is None:
                    os.environ.pop("ARTIFACT_STORE_PATH", None)
                else:
                    os.environ["ARTIFACT_STORE_PATH"] = orig

        assert info["artifact_kind"] == "directory"
        assert info["download_filename"] == "dataset-5.zip"
        assert info["download_format"] == "zip"
