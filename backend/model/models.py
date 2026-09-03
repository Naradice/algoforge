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


class PreprocessedDataset(Base):
    """A named, reusable preprocessing recipe for a base dataset — indicators/clustering,
    feature_cols, and normalize saved once and referenced by TrainingRun instead of being
    re-specified inline every time. Immutable after creation (only `name` may be updated);
    to change the recipe, create a new one."""

    __tablename__ = "preprocessed_datasets"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    dataset_id: Mapped[int] = mapped_column(sa.Integer, nullable=False, index=True)  # soft FK → datasets.id
    preprocessing: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    feature_cols: Mapped[list] = mapped_column(JSONB, nullable=False, server_default='["close"]')
    normalize: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="zscore")
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="pending")  # pending | ready | error
    characteristics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now())


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[int] = mapped_column(sa.Integer, nullable=False)  # soft FK → datasets.id
    preprocessed_dataset_id: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)  # soft FK → preprocessed_datasets.id
    hyperparams: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="pending")  # pending | running | completed | error | stopped
    # internal: trained by this backend's own celery worker (the default, all pre-existing rows).
    # external: trained elsewhere (e.g. a Colab notebook run by hand) and registered after the
    # fact via POST /models/{id}/training-runs/import — see
    # model/service.py:import_external_training_run.
    source: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="internal")
    # Where a source="internal" run's training actually executes: "local" (this backend's own
    # `training` queue, the pre-existing behaviour) or "colab" (this backend orchestrates a
    # Colab run automatically — export a dataset snapshot, generate a notebook, run it via
    # colab-cli, pull the result back — see model/colab_trainer.py). Meaningless for
    # source="external" rows (those were never orchestrated by this backend in the first
    # place); left at the default "local" for them.
    execution_target: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="local")
    current_epoch: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default="0")
    best_epoch: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    val_loss: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    eta_seconds: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    num_params: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    preprocessed_characteristics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # What the dataset loader actually used, captured at load time from the real data rather
    # than trusted bookkeeping: source_rows, effective_rows, max_rows (resolved cap),
    # first_timestamp, last_timestamp, sampling_stride_seconds. See
    # OHLCWindowDataset._load_preprocessed_df -- this exists specifically so a silent row-cap
    # truncation (a real bug that went undetected through an entire DDM data-volume investigation
    # phase) shows up immediately instead of requiring after-the-fact numerical detective work.
    data_provenance: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    stop_requested: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default="false")
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


class TrainingRunMetric(Base):
    __tablename__ = "training_run_metrics"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    epoch: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    train_loss: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    val_loss: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    lr: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


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
    dataset_id: int | None = None
    preprocessed_dataset_id: int | None = None
    hyperparams: dict[str, Any] = {}
    # "local" (default): this backend's own `training` queue, as before. "colab": this backend
    # orchestrates a Colab CPU run automatically instead — see model/colab_trainer.py. Only
    # architectures/options colab_trainer.py's notebook generator supports can use "colab";
    # requesting it for an unsupported combination fails at creation time, not mid-run.
    execution_target: str = "local"


class TrainingRunImportMetric(BaseModel):
    epoch: int
    train_loss: float | None = None
    val_loss: float | None = None
    lr: float | None = None


class TrainingRunImportCreate(BaseModel):
    """Body (as a `metadata` form field, JSON-encoded) for POST /models/{id}/training-runs/import.

    Registers a training run that happened outside this backend (e.g. a Colab notebook) as a
    first-class TrainingRun so it shows up in /model/compare and the model detail page like any
    other run. `dataset_id` must reference the algoforge dataset the external run actually
    trained on (the snapshot exported for the notebook), so provenance stays traceable even
    though the training itself ran elsewhere.
    """

    dataset_id: int
    preprocessed_dataset_id: int | None = None
    hyperparams: dict[str, Any] = {}
    epoch_metrics: list[TrainingRunImportMetric] = []
    best_epoch: int
    val_loss: float
    num_params: int | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    notes: str | None = None
    # Free-form provenance for reproducing the run — e.g. {"notebook_url": "https://colab.research.google.com/github/...",
    # "git_commit": "...", "dataset_snapshot_sha256": "...", "platform": "colab"}. Merged into
    # the created TrainingRun's hyperparams under "_external_ref" rather than given dedicated
    # columns, so no further schema changes are needed as what's tracked here evolves.
    external_ref: dict[str, Any] = {}


class TrainingRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_id: int
    dataset_id: int
    preprocessed_dataset_id: int | None
    hyperparams: dict[str, Any]
    status: str
    source: str
    execution_target: str
    current_epoch: int
    best_epoch: int | None
    val_loss: float | None
    eta_seconds: int | None
    num_params: int | None
    preprocessed_characteristics: dict[str, Any] | None
    data_provenance: dict[str, Any] | None
    artifact_path: str | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime


class PreprocessedDatasetCreate(BaseModel):
    name: str
    dataset_id: int
    preprocessing: dict[str, Any] = {}
    feature_cols: list[str] = ["close"]
    normalize: str = "zscore"


class PreprocessedDatasetUpdate(BaseModel):
    name: str


class PreprocessedDatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    dataset_id: int
    preprocessing: dict[str, Any]
    feature_cols: list[str]
    normalize: str
    status: str
    characteristics: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


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


class TrainingRunMetricRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    training_run_id: int
    epoch: int
    train_loss: float | None
    val_loss: float | None
    lr: float | None
    recorded_at: datetime


class ValidationCreate(BaseModel):
    training_run_id: int
    dataset_id: int


class HyperparamSearchCreate(BaseModel):
    model_id: int
    dataset_id: int
    search_grid: dict[str, list]  # e.g. {"lr": [0.001, 0.0001], "batch_size": [32, 64]}
    # Same meaning as TrainingRunCreate.execution_target -- applies to every run the grid
    # expands into. Each run still goes through create_training_run's own check_colab_supported
    # validation individually when execution_target="colab", so an unsupported combination
    # fails the whole search at creation time rather than partway through.
    execution_target: str = "local"
