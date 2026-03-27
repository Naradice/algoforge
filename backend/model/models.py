"""
SQLAlchemy ORM models and Pydantic schemas for the ML Model layer.
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


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    architecture: Mapped[str] = mapped_column(sa.Text, nullable=False)  # seq2seq_transformer | lstm | timegan | rl_agent
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="created")  # created | training | trained | deployed | archived
    artifact_path: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now())

    training_runs: Mapped[list[TrainingRun]] = relationship("TrainingRun", back_populates="model", lazy="noload")
    validations: Mapped[list[ModelValidation]] = relationship("ModelValidation", back_populates="model", lazy="noload")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[int] = mapped_column(sa.Integer, nullable=False)  # soft FK → datasets.id
    hyperparams: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="pending")  # pending | running | completed | error | stopped
    current_epoch: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default="0")
    best_epoch: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    val_loss: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    eta_seconds: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    model: Mapped[MLModel] = relationship("MLModel", back_populates="training_runs")
    checkpoints: Mapped[list[TrainingCheckpoint]] = relationship("TrainingCheckpoint", back_populates="training_run", lazy="noload")


class TrainingCheckpoint(Base):
    __tablename__ = "training_checkpoints"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    epoch: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    artifact_path: Mapped[str] = mapped_column(sa.Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    training_run: Mapped[TrainingRun] = relationship("TrainingRun", back_populates="checkpoints")


class ModelValidation(Base):
    __tablename__ = "model_validations"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True)
    training_run_id: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    dataset_id: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    computed_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    model: Mapped[MLModel] = relationship("MLModel", back_populates="validations")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class MLModelCreate(BaseModel):
    name: str
    architecture: str
    config: dict[str, Any] = {}


class MLModelUpdate(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None
    status: str | None = None


class MLModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    architecture: str
    config: dict[str, Any]
    status: str
    artifact_path: str | None
    created_at: datetime
    updated_at: datetime


class TrainingRunCreate(BaseModel):
    dataset_id: int
    hyperparams: dict[str, Any] = {}


class TrainingRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_id: int
    dataset_id: int
    hyperparams: dict[str, Any]
    status: str
    current_epoch: int
    best_epoch: int | None
    val_loss: float | None
    eta_seconds: int | None
    artifact_path: str | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime


class ModelValidationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_id: int
    training_run_id: int | None
    dataset_id: int
    metrics: dict[str, Any]
    computed_at: datetime


class PredictRequest(BaseModel):
    features: list[list[float]]
    feature_names: list[str]


class PredictResponse(BaseModel):
    predictions: list[dict[str, Any]]
