"""
SQLAlchemy ORM models and Pydantic schemas for the Strategy layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    description: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="")
    definition: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="inactive")  # active | inactive | archived
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now())

    runs: Mapped[list[StrategyRun]] = relationship("StrategyRun", back_populates="strategy", lazy="noload")


class StrategyRun(Base):
    __tablename__ = "strategy_runs"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    strategy_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(sa.Text, nullable=False)  # backtest | paper | live
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="pending")  # pending | running | completed | error | stopped
    progress_pct: Mapped[float] = mapped_column(sa.Float, nullable=False, server_default="0")
    message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    dataset_id: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)  # soft FK → datasets.id
    broker_client: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    walk_forward_ratio: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    risk_override: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    window_size: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    starting_capital: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    from_ts: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    to_ts: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    equity_curve: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    strategy: Mapped[Strategy] = relationship("Strategy", back_populates="runs")
    trades: Mapped[list[Trade]] = relationship("Trade", back_populates="run", lazy="noload")
    metrics: Mapped[list[RunMetric]] = relationship("RunMetric", back_populates="run", lazy="noload")
    chats: Mapped[list[StrategyRunChat]] = relationship("StrategyRunChat", back_populates="run", lazy="noload")


class StrategyEvent(Base):
    __tablename__ = "strategy_events"

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True)
    run_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("strategy_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy_id: Mapped[int] = mapped_column(sa.Integer, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    occurred_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, index=True)


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True)
    run_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("strategy_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(sa.Text, nullable=False)
    direction: Mapped[str] = mapped_column(sa.Text, nullable=False)  # buy | sell
    entry_price: Mapped[float] = mapped_column(sa.Float, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    volume: Mapped[float] = mapped_column(sa.Float, nullable=False)
    sl_price: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    tp_price: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    profit: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(sa.Text, nullable=True)  # signal | sl | tp | end_of_data
    phase: Mapped[str | None] = mapped_column(sa.Text, nullable=True)        # is | oos
    mae: Mapped[float | None] = mapped_column(sa.Float, nullable=True)       # max adverse excursion
    mfe: Mapped[float | None] = mapped_column(sa.Float, nullable=True)       # max favorable excursion

    run: Mapped[StrategyRun] = relationship("StrategyRun", back_populates="trades")


class RunMetric(Base):
    __tablename__ = "run_metrics"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("strategy_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    key: Mapped[str] = mapped_column(sa.Text, nullable=False)
    value: Mapped[float] = mapped_column(sa.Float, nullable=False)

    run: Mapped[StrategyRun] = relationship("StrategyRun", back_populates="metrics")


class StrategyVersion(Base):
    __tablename__ = "strategy_versions"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    strategy_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False, index=True)
    version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    definition: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


class StrategyRunChat(Base):
    __tablename__ = "strategy_run_chats"

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True)
    run_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("strategy_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(sa.Text, nullable=False)  # user | engine | agent | system
    message: Mapped[str] = mapped_column(sa.Text, nullable=False)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    run: Mapped[StrategyRun] = relationship("StrategyRun", back_populates="chats")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class StrategyCreate(BaseModel):
    name: str
    description: str = ""
    definition: dict[str, Any] = {}


class StrategyUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    definition: dict[str, Any] | None = None
    status: str | None = None


class StrategyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    definition: dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime


class StrategyRunCreate(BaseModel):
    mode: str  # backtest | paper | live
    dataset_id: int | None = None
    broker_client: str | None = None
    walk_forward_ratio: float | None = None  # 0.7 → 70% in-sample, 30% OOS
    risk_override: dict | None = None
    window_size: int | None = None  # None/0 = full history; N = rolling N-bar window
    starting_capital: float | None = None  # None/1.0 = normalised; N = scale PnL to this capital
    from_ts: datetime | None = None
    to_ts: datetime | None = None


class StrategyRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    strategy_id: int
    mode: str
    status: str
    progress_pct: float
    message: str | None
    dataset_id: int | None
    broker_client: str | None
    walk_forward_ratio: float | None
    risk_override: dict | None
    window_size: int | None
    starting_capital: float | None
    from_ts: datetime | None
    to_ts: datetime | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime


class StrategyVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    strategy_id: int
    version: int
    definition: dict[str, Any]
    created_at: datetime


class TradeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float | None = None
    volume: float
    sl_price: float | None = None
    tp_price: float | None = None
    profit: float | None = None
    opened_at: datetime
    closed_at: datetime | None = None
    exit_reason: str | None = None
    phase: str | None = None
    mae: float | None = None
    mfe: float | None = None


class ChatMessageCreate(BaseModel):
    message: str


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    role: str
    message: str
    context: dict[str, Any] | None
    created_at: datetime
