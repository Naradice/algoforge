"""
Pytest configuration — in-memory SQLite DB + TestClient.

Run tests without Docker:
    cd algoforge/backend
    pip install -r requirements.txt pytest pytest-asyncio httpx
    pytest tests/ -v
"""
from __future__ import annotations

import os
import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# ── Point at SQLite before any app import ──────────────────────────────────────
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("ARTIFACT_STORE_PATH", "/tmp/algoforge_test_artifacts")
os.environ.setdefault("ALGOFORGE_NO_AUTH", "1")

# Stub celery_app.enqueue to a no-op so tests don't need Redis
import unittest.mock as mock
import sys

# Stub celery_app before app loads
celery_app_stub = mock.MagicMock()
celery_app_stub.enqueue = mock.AsyncMock(return_value=None)
sys.modules.setdefault("celery_app", celery_app_stub)

# ── Make PostgreSQL-specific types compile on SQLite ───────────────────────────
# JSONB has no SQLite fallback; register a passthrough that renders as JSON TEXT.
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"

from database import Base, get_db  # noqa: E402
from main import app               # noqa: E402


TEST_DB_URL = "sqlite+aiosqlite:///./test.db"

engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSession = async_sessionmaker(engine, expire_on_commit=False)


async def override_get_db():
    async with TestingSession() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
    import os as _os
    try:
        _os.remove("test.db")
    except OSError:
        pass


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
