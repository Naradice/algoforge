"""
SQLAlchemy ORM models and Pydantic schemas for the Data Management layer.
"""

from __future__ import annotations

from datetime import date, datetime
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
    # ohlc_download | web_report | ddm_simulation | manual_upload | economic_calendar | synthetic_function
    type: Mapped[str] = mapped_column(sa.Text, nullable=False)
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
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
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


class DatasetSnapshot(Base):
    """An immutable, single-file, hash-stamped copy of a dataset's contents at one point in
    time — for handing to an external, disconnected execution environment (e.g. a Colab
    notebook) where the live dataset (which incremental collection keeps mutating) can't be
    reached directly. See scripts/export_dataset_snapshot.py.
    """

    __tablename__ = "dataset_snapshots"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    artifact_path: Mapped[str] = mapped_column(sa.Text, nullable=False)  # relative to ARTIFACT_STORE_PATH, single parquet file
    sha256: Mapped[str] = mapped_column(sa.Text, nullable=False)
    row_count: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="local")  # local | uploaded | error
    export_provider: Mapped[str | None] = mapped_column(sa.Text, nullable=True)  # e.g. "gdrive"
    # e.g. {"file_id": "...", "url": "...", "uploaded_at": "..."} — free-form per provider,
    # so adding a new export target never needs a schema change.
    export_ref: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    dataset: Mapped[Dataset] = relationship("Dataset")


class ReportForecast(Base):
    __tablename__ = "report_forecasts"
    __table_args__ = (
        sa.UniqueConstraint("source", "symbol", "horizon", "report_date", name="uq_report_forecasts_source_sym_hor_date"),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    datasource_id: Mapped[int | None] = mapped_column(sa.Integer, sa.ForeignKey("datasources.id", ondelete="SET NULL"), nullable=True, index=True)
    source: Mapped[str] = mapped_column(sa.Text, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="USDJPY", index=True)
    horizon: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="Daily")
    report_date: Mapped[date] = mapped_column(sa.Date, nullable=False, index=True)
    analyst: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    direction: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    expected_low: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    expected_high: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    title: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    raw_artifact_path: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


class LargeHoldingReportRecord(Base):
    __tablename__ = "large_holding_reports"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(sa.Text, unique=True, index=True, nullable=False)
    issuer_code: Mapped[str | None] = mapped_column(sa.Text, index=True, nullable=True)
    issuer_name: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    filer_name: Mapped[str | None] = mapped_column(sa.Text, index=True, nullable=True)
    base_date: Mapped[date | None] = mapped_column(sa.Date, index=True, nullable=True)
    submit_date: Mapped[date | None] = mapped_column(sa.Date, index=True, nullable=True)
    doc_type: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    holding_ratio: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    prev_holding_ratio: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    ratio_change: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    total_shares_held: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    total_outstanding_shares: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    purpose_of_holding: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    joint_holders_json: Mapped[list | None] = mapped_column(sa.JSON, nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    trades: Mapped[list[LargeHoldingTradeRecord]] = relationship(
        "LargeHoldingTradeRecord", back_populates="report", cascade="all, delete-orphan", lazy="selectin"
    )


class LargeHoldingTradeRecord(Base):
    __tablename__ = "large_holding_trades"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("large_holding_reports.id", ondelete="CASCADE"), nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(sa.Text, index=True, nullable=False)
    issuer_code: Mapped[str | None] = mapped_column(sa.Text, index=True, nullable=True)
    trade_date: Mapped[date | None] = mapped_column(sa.Date, index=True, nullable=True)
    security_type: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    quantity: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    ratio: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    market_type: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    trade_type: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    unit_price: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)

    report: Mapped[LargeHoldingReportRecord] = relationship("LargeHoldingReportRecord", back_populates="trades")


class DailyMarginRecord(Base):
    __tablename__ = "daily_margin_records"
    __table_args__ = (
        sa.UniqueConstraint("as_of_date", "code", name="uq_daily_margin_date_code"),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    as_of_date: Mapped[date] = mapped_column(sa.Date, index=True, nullable=False)
    code: Mapped[str] = mapped_column(sa.Text, index=True, nullable=False)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    market: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    issue_type: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    restrictions: Mapped[list | None] = mapped_column(sa.JSON, nullable=True)
    isin: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    sell_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    sell_change: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    sell_ratio: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    buy_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    buy_change: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    buy_ratio: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    ratio: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    negotiable_sell_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    negotiable_sell_change: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    standardized_sell_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    standardized_sell_change: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    negotiable_buy_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    negotiable_buy_change: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    standardized_buy_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    standardized_buy_change: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


class MarginRegulationRecord(Base):
    __tablename__ = "margin_regulation_records"
    __table_args__ = (
        sa.UniqueConstraint("code", "event_date", "event_type", name="uq_margin_reg_code_date_type"),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    code: Mapped[str] = mapped_column(sa.Text, index=True, nullable=False)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    event_date: Mapped[date] = mapped_column(sa.Date, index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(sa.Text, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


class MarketMarginWeeklyRecord(Base):
    __tablename__ = "market_margin_weekly_records"
    __table_args__ = (
        sa.UniqueConstraint("date", name="uq_market_margin_weekly_date"),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    date: Mapped[date] = mapped_column(sa.Date, unique=True, index=True, nullable=False)
    standardized_sell_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    standardized_buy_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    negotiable_sell_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    negotiable_buy_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    total_sell_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    total_buy_shares: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    margin_ratio: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


class PdfDocumentRecord(Base):
    __tablename__ = "pdf_documents"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(sa.Text, unique=True, index=True, nullable=False)
    sec_code: Mapped[str | None] = mapped_column(sa.Text, index=True, nullable=True)
    filer_name: Mapped[str | None] = mapped_column(sa.Text, index=True, nullable=True)
    doc_type: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    doc_type_code: Mapped[str | None] = mapped_column(sa.Text, index=True, nullable=True)
    category: Mapped[str] = mapped_column(sa.Text, index=True, nullable=False, default="prospectus")
    filing_date: Mapped[date | None] = mapped_column(sa.Date, index=True, nullable=True)
    file_path: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    file_size: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    sha256: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    num_pages: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    extraction_status: Mapped[str] = mapped_column(sa.Text, nullable=False, server_default="pending")  # pending | completed | error | skipped
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    extracted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now())

    tables: Mapped[list[PdfExtractedTableRecord]] = relationship(
        "PdfExtractedTableRecord", back_populates="document", cascade="all, delete-orphan", lazy="selectin"
    )


class PdfExtractedTableRecord(Base):
    __tablename__ = "pdf_extracted_tables"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey("pdf_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(sa.Text, index=True, nullable=False)
    page_number: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    table_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    table_title: Mapped[str | None] = mapped_column(sa.Text, index=True, nullable=True)
    row_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    col_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    headers_json: Mapped[list | None] = mapped_column(sa.JSON, nullable=True)
    data_json: Mapped[list | None] = mapped_column(sa.JSON, nullable=True)
    html: Mapped[str] = mapped_column(sa.Text, nullable=False, default="")
    bbox_json: Mapped[list | None] = mapped_column(sa.JSON, nullable=True)
    meta_json: Mapped[dict | None] = mapped_column(sa.JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())

    document: Mapped[PdfDocumentRecord] = relationship("PdfDocumentRecord", back_populates="tables")


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


class DatasetSnapshotRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    artifact_path: str
    sha256: str
    row_count: int | None
    size_bytes: int | None
    status: str
    export_provider: str | None
    export_ref: dict[str, Any] | None
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


class ReportForecastRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    datasource_id: int | None = None
    source: str
    symbol: str
    horizon: str
    report_date: date
    analyst: str | None = None
    direction: str | None = None
    expected_low: float | None = None
    expected_high: float | None = None
    title: str | None = None
    raw_artifact_path: str | None = None
    extracted_at: datetime


class PdfExtractedTableRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    doc_id: str
    page_number: int
    table_index: int
    table_title: str | None = None
    row_count: int
    col_count: int
    headers: list | None = None
    data: list | None = None
    html: str
    bbox: list | None = None
    meta: dict | None = None
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _map_json_fields(cls, data: object) -> object:
        if isinstance(data, dict):
            return data
        # Map ORM columns *_json to schema fields
        return {
            "id": getattr(data, "id"),
            "document_id": getattr(data, "document_id"),
            "doc_id": getattr(data, "doc_id"),
            "page_number": getattr(data, "page_number"),
            "table_index": getattr(data, "table_index"),
            "table_title": getattr(data, "table_title"),
            "row_count": getattr(data, "row_count"),
            "col_count": getattr(data, "col_count"),
            "headers": getattr(data, "headers_json", None),
            "data": getattr(data, "data_json", None),
            "html": getattr(data, "html", ""),
            "bbox": getattr(data, "bbox_json", None),
            "meta": getattr(data, "meta_json", None),
            "created_at": getattr(data, "created_at"),
        }


class PdfDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    doc_id: str
    sec_code: str | None = None
    filer_name: str | None = None
    doc_type: str | None = None
    doc_type_code: str | None = None
    category: str
    filing_date: date | None = None
    file_path: str | None = None
    file_size: int | None = None
    sha256: str | None = None
    num_pages: int | None = None
    extraction_status: str
    error_message: str | None = None
    extracted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    tables_count: int = 0

    @model_validator(mode="before")
    @classmethod
    def _map_tables_count(cls, data: object) -> object:
        if isinstance(data, dict):
            return data
        tables = getattr(data, "tables", None)
        count = len(tables) if tables is not None else 0
        return {
            "id": getattr(data, "id"),
            "doc_id": getattr(data, "doc_id"),
            "sec_code": getattr(data, "sec_code"),
            "filer_name": getattr(data, "filer_name"),
            "doc_type": getattr(data, "doc_type"),
            "doc_type_code": getattr(data, "doc_type_code"),
            "category": getattr(data, "category"),
            "filing_date": getattr(data, "filing_date"),
            "file_path": getattr(data, "file_path"),
            "file_size": getattr(data, "file_size"),
            "sha256": getattr(data, "sha256"),
            "num_pages": getattr(data, "num_pages"),
            "extraction_status": getattr(data, "extraction_status"),
            "error_message": getattr(data, "error_message"),
            "extracted_at": getattr(data, "extracted_at"),
            "created_at": getattr(data, "created_at"),
            "updated_at": getattr(data, "updated_at"),
            "tables_count": count,
        }


class PdfDocumentDetailRead(PdfDocumentRead):
    tables: list[PdfExtractedTableRead] = []

    @model_validator(mode="before")
    @classmethod
    def _map_tables_detail(cls, data: object) -> object:
        if isinstance(data, dict):
            return data
        base = PdfDocumentRead._map_tables_count(data)
        tables = getattr(data, "tables", []) or []
        base["tables"] = tables
        return base


class PdfCollectRequest(BaseModel):
    target_date: str | None = None
    category: str = "prospectus"
    max_documents: int = 20
    force: bool = False

