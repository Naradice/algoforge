"""
ORM model for the logs table. Kept separate from log_writer.py to avoid
circular imports (log_writer imports async_session_factory from database;
anything that imports Base from database can import this freely).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Log(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), index=True)
    level: Mapped[str] = mapped_column(sa.Text, nullable=False, index=True)
    source: Mapped[str] = mapped_column(sa.Text, nullable=False, index=True)
    message: Mapped[str] = mapped_column(sa.Text, nullable=False)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Soft foreign keys — no constraint, nullable
    strategy_run_id: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, index=True)
    training_run_id: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, index=True)
    collection_job_id: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, index=True)
    event_id: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True, index=True)
