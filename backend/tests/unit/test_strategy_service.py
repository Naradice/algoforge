"""Unit tests for StrategyService — repository mocked out."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from strategy.service import StrategyService
from strategy.models import Strategy, StrategyRun, StrategyUpdate, StrategyRunCreate


PATCH = "strategy.service.strategy_repo"


# ── get_strategy ───────────────────────────────────────────────────────────────

class TestGetStrategy:
    @pytest.mark.asyncio
    async def test_raises_404_when_not_found(self):
        svc = StrategyService()
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.get_strategy(AsyncMock(), 999)
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_strategy_when_found(self):
        svc = StrategyService()
        strat = MagicMock(spec=Strategy, id=1)
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=strat)
            result = await svc.get_strategy(AsyncMock(), 1)
        assert result is strat

    @pytest.mark.asyncio
    async def test_calls_repo_with_correct_id(self):
        svc = StrategyService()
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=MagicMock(spec=Strategy))
            db = AsyncMock()
            await svc.get_strategy(db, 42)
            repo.get_by_id.assert_called_once_with(db, 42)


# ── delete_strategy ────────────────────────────────────────────────────────────

class TestDeleteStrategy:
    @pytest.mark.asyncio
    async def test_raises_404_when_not_found(self):
        svc = StrategyService()
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc:
                await svc.delete_strategy(AsyncMock(), 1)
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_deletes_strategy_from_db(self):
        svc = StrategyService()
        strat = MagicMock(spec=Strategy)
        db = AsyncMock()
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=strat)
            await svc.delete_strategy(db, 1)
        db.delete.assert_called_once_with(strat)


# ── update_strategy_with_version ───────────────────────────────────────────────

class TestUpdateStrategyWithVersion:
    @pytest.mark.asyncio
    async def test_saves_version_snapshot_when_definition_changes(self):
        svc = StrategyService()
        existing = MagicMock(spec=Strategy, id=1, definition={"v": 1})
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=existing)
            repo.get_version_count = AsyncMock(return_value=2)
            repo.create_version = AsyncMock()
            repo.update = AsyncMock(return_value=existing)

            await svc.update_strategy_with_version(AsyncMock(), 1, StrategyUpdate(definition={"v": 2}))

            repo.create_version.assert_called_once()
            args = repo.create_version.call_args
            # version number = count + 1 = 3
            assert args[0][2] == 3
            # snapshot is the OLD definition
            assert args[0][3] == {"v": 1}

    @pytest.mark.asyncio
    async def test_no_version_when_only_name_changes(self):
        svc = StrategyService()
        existing = MagicMock(spec=Strategy, id=1, definition={"v": 1})
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=existing)
            repo.update = AsyncMock(return_value=existing)
            repo.create_version = AsyncMock()

            await svc.update_strategy_with_version(AsyncMock(), 1, StrategyUpdate(name="New"))

            repo.create_version.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_version_when_only_status_changes(self):
        svc = StrategyService()
        existing = MagicMock(spec=Strategy, id=1, definition={})
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=existing)
            repo.update = AsyncMock(return_value=existing)
            repo.create_version = AsyncMock()

            await svc.update_strategy_with_version(AsyncMock(), 1, StrategyUpdate(status="archived"))

            repo.create_version.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_existing_when_no_updates(self):
        svc = StrategyService()
        existing = MagicMock(spec=Strategy, id=1, definition={})
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=existing)

            result = await svc.update_strategy_with_version(AsyncMock(), 1, StrategyUpdate())

            assert result is existing

    @pytest.mark.asyncio
    async def test_version_count_increments_correctly(self):
        svc = StrategyService()
        existing = MagicMock(spec=Strategy, id=1, definition={"a": 1})
        with patch(PATCH) as repo:
            repo.get_by_id = AsyncMock(return_value=existing)
            repo.get_version_count = AsyncMock(return_value=0)
            repo.create_version = AsyncMock()
            repo.update = AsyncMock(return_value=existing)

            await svc.update_strategy_with_version(AsyncMock(), 1, StrategyUpdate(definition={"a": 2}))

            # First version → number = 0 + 1 = 1
            assert repo.create_version.call_args[0][2] == 1


# ── stop_run ───────────────────────────────────────────────────────────────────

class TestStopRun:
    @pytest.mark.asyncio
    async def test_raises_422_for_completed_run(self):
        svc = StrategyService()
        run = MagicMock(spec=StrategyRun, strategy_id=1, status="completed")
        with patch(PATCH) as repo:
            repo.get_run = AsyncMock(return_value=run)
            repo.get_by_id = AsyncMock(return_value=MagicMock(spec=Strategy))

            with pytest.raises(HTTPException) as exc:
                await svc.stop_run(AsyncMock(), 1, 1)
            assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_raises_422_for_stopped_run(self):
        svc = StrategyService()
        run = MagicMock(spec=StrategyRun, strategy_id=1, status="stopped")
        with patch(PATCH) as repo:
            repo.get_run = AsyncMock(return_value=run)
            repo.get_by_id = AsyncMock(return_value=MagicMock(spec=Strategy))

            with pytest.raises(HTTPException) as exc:
                await svc.stop_run(AsyncMock(), 1, 1)
            assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_stops_pending_run(self):
        svc = StrategyService()
        run = MagicMock(spec=StrategyRun, strategy_id=1, status="pending")
        stopped = MagicMock(spec=StrategyRun, status="stopped")
        with patch(PATCH) as repo:
            repo.get_run = AsyncMock(return_value=run)
            repo.get_by_id = AsyncMock(return_value=MagicMock(spec=Strategy))
            repo.update_run = AsyncMock(return_value=stopped)

            result = await svc.stop_run(AsyncMock(), 1, 1)

            repo.update_run.assert_called_once()
            assert result.status == "stopped"

    @pytest.mark.asyncio
    async def test_stops_running_run(self):
        svc = StrategyService()
        run = MagicMock(spec=StrategyRun, strategy_id=1, status="running")
        stopped = MagicMock(spec=StrategyRun, status="stopped")
        with patch(PATCH) as repo:
            repo.get_run = AsyncMock(return_value=run)
            repo.get_by_id = AsyncMock(return_value=MagicMock(spec=Strategy))
            repo.update_run = AsyncMock(return_value=stopped)

            result = await svc.stop_run(AsyncMock(), 1, 1)
            assert result.status == "stopped"

    @pytest.mark.asyncio
    async def test_raises_404_when_run_not_found(self):
        svc = StrategyService()
        with patch(PATCH) as repo:
            repo.get_run = AsyncMock(return_value=None)
            repo.get_by_id = AsyncMock(return_value=MagicMock(spec=Strategy))

            with pytest.raises(HTTPException) as exc:
                await svc.stop_run(AsyncMock(), 1, 999)
            assert exc.value.status_code == 404


# ── get_run (ownership check) ──────────────────────────────────────────────────

class TestGetRun:
    @pytest.mark.asyncio
    async def test_raises_404_when_run_belongs_to_different_strategy(self):
        svc = StrategyService()
        run = MagicMock(spec=StrategyRun, strategy_id=99)  # belongs to strategy 99
        with patch(PATCH) as repo:
            repo.get_run = AsyncMock(return_value=run)

            with pytest.raises(HTTPException) as exc:
                await svc.get_run(AsyncMock(), strategy_id=1, run_id=5)
            assert exc.value.status_code == 404
