import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://algoforge:algoforge@localhost:5432/algoforge",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Same commit-on-success/rollback-on-exception contract as get_db(), as a plain async
    context manager instead of a FastAPI dependency generator -- for callers that aren't FastAPI
    routes and can't use Depends(get_db), chiefly the mcp_server/tools/*.py MCP tool functions.

    Confirmed live (2026-09-18): every MCP tool was using bare `async with async_session_factory()
    as db:` directly, which never commits on its own (unlike get_db(), whose whole point is the
    explicit `await session.commit()` after yield) -- so every write-performing MCP tool
    (create_model, start_training_run, create_datasource, deploy_model, create_strategy, ...) has
    silently rolled back its own writes since this MCP layer was built. This went undetected
    because /mcp itself was separately unreachable (see main.py's mount fix, same date) for as
    long as this pattern has existed, so no real MCP client had ever exercised any of these tools
    live before now. Use this for every mcp_server/tools/*.py function that writes; read-only
    tools can keep using async_session_factory() directly (nothing to commit), though there's no
    harm in using this here too for consistency.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
