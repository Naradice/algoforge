"""
SQLAlchemy ORM models and Pydantic schemas for the Data Management layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Datasource(Base):
    __tablename__ = "datasources"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    type: Mapped[str] = mapped_column(sa.Text, nullable=False)  # ohlc_download | web_report | ddm_simulation | manual_upload | economic_calendar
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now())

    datasets: Mapped[list[Dataset]] = relationship("Dataset", back_populates="datasource", lazy="noload")
    collection_jobs: Mapped[list[CollectionJob]] = relationship("CollectionJob", back_populates="datasource", lazy="noload")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    datasource_id: Mapped[int | None] = mapped_column(sa.Integer, sa.ForeignKey("datasources.id", ondelete="SET NULL"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    symbol: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    timeframe: Mapped[str | None] = mapped_column(sa.Text, nullable=True)  # M1 | M5 | H1 | D1 | etc.
    from_ts: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    to_ts: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    row_count: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="pending")  # pending | ready | error
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    datasource: Mapped[Datasource | None] = relationship("Datasource", back_populates="datasets")
    characteristics: Mapped[list[DataCharacteristics]] = relationship("DataCharacteristics", back_populates="dataset", lazy="noload")


class CollectionJob(Base):
    __tablename__ = "collection_jobs"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    datasource_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("datasources.id", ondelete="CASCADE"), nullable=False, index=True)
    schedule_cron: Mapped[str | None] = mapped_column(sa.Text, nullable=True)  # None = one-off
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="idle")  # idle | running | error
    last_run_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    datasource: Mapped[Datasource] = relationship("Datasource", back_populates="collection_jobs")


class CollectionJobRun(Base):
    __tablename__ = "collection_job_runs"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("collection_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="running")  # running | completed | error
    datasets_produced: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default="0")
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)


class DataCharacteristics(Base):
    __tablename__ = "data_characteristics"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    computed_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="characteristics")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class DatasourceCreate(BaseModel):
    name: str
    type: str
    config: dict[str, Any] = {}


class DatasourceUpdate(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None


class DatasourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    type: str
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    datasource_id: int | None
    name: str
    symbol: str | None
    timeframe: str | None
    from_ts: datetime | None
    to_ts: datetime | None
    row_count: int | None
    status: str
    created_at: datetime


class DatasetUpdate(BaseModel):
    name: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    from_ts: datetime | None = None
    to_ts: datetime | None = None
    row_count: int | None = None
    artifact_path: str | None = None
    status: str | None = None


class CollectionJobCreate(BaseModel):
    datasource_id: int
    schedule_cron: str | None = None


class CollectionJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    datasource_id: int
    schedule_cron: str | None
    status: str
    last_run_at: datetime | None
    next_run_at: datetime | None
    last_error: str | None
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _derive_status(cls, data: object) -> object:
        """Return 'scheduled' when idle with a future next_run_at so the UI
        can distinguish a planned daily download from a never-run job."""
        from datetime import timezone

        # Support both ORM objects (from_attributes) and plain dicts
        if isinstance(data, dict):
            status = data.get("status")
            next_run_at = data.get("next_run_at")
        else:
            status = getattr(data, "status", None)
            next_run_at = getattr(data, "next_run_at", None)

        if status == "idle" and next_run_at is not None:
            # Normalise to aware datetime for comparison
            if isinstance(next_run_at, datetime):
                nra = next_run_at if next_run_at.tzinfo else next_run_at.replace(tzinfo=timezone.utc)
                if nra > datetime.now(tz=timezone.utc):
                    if isinstance(data, dict):
                        data = {**data, "status": "scheduled"}
                    else:
                        # Wrap in dict so Pydantic can build the model cleanly
                        data = {
                            field: getattr(data, field)
                            for field in ("id", "datasource_id", "schedule_cron", "status",
                                         "last_run_at", "next_run_at", "last_error", "created_at")
                        }
                        data["status"] = "scheduled"
        return data


class DataCharacteristicsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    metrics: dict[str, Any]
    computed_at: datetime


class CollectionJobUpdate(BaseModel):
    schedule_cron: str | None = None
    enabled: bool | None = None
    status: str | None = None


class CollectionJobRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: int
    started_at: datetime | None
    finished_at: datetime | None
    status: str
    datasets_produced: int
    error_message: str | None


class CollectRequest(BaseModel):
    from_ts: str | None = None
    to_ts: str | None = None


class AnalyzeRequest(BaseModel):
    analyses: list[str] | None = None
