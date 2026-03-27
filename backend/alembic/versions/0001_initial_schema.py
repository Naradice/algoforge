"""Initial schema — all tables

Revision ID: 0001
Revises:
Create Date: 2026-03-27
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── auth ─────────────────────────────────────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("key_hash", sa.Text, nullable=False, unique=True),
        sa.Column("scopes", ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"])

    op.create_table(
        "webhook_registrations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("events", ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("secret_hash", sa.Text, nullable=True),
        sa.Column("active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # ── strategy layer ────────────────────────────────────────────────────────
    op.create_table(
        "strategies",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("definition", JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.Text, nullable=False, server_default="inactive"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "strategy_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("strategy_id", sa.Integer, sa.ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("mode", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("progress_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("message", sa.Text, nullable=True),
        sa.Column("dataset_id", sa.Integer, nullable=True),
        sa.Column("broker_client", sa.Text, nullable=True),
        sa.Column("from_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("to_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_strategy_runs_strategy_id", "strategy_runs", ["strategy_id"])

    op.create_table(
        "strategy_events",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("strategy_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("strategy_id", sa.Integer, nullable=False),
        sa.Column("event_type", sa.Text, nullable=False),
        sa.Column("payload", JSONB, nullable=False, server_default="{}"),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_strategy_events_run_id", "strategy_events", ["run_id"])
    op.create_index("ix_strategy_events_occurred_at", "strategy_events", ["occurred_at"])

    op.create_table(
        "trades",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("strategy_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("symbol", sa.Text, nullable=False),
        sa.Column("direction", sa.Text, nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("exit_price", sa.Float, nullable=True),
        sa.Column("volume", sa.Float, nullable=False),
        sa.Column("sl_price", sa.Float, nullable=True),
        sa.Column("tp_price", sa.Float, nullable=True),
        sa.Column("profit", sa.Float, nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_trades_run_id", "trades", ["run_id"])

    op.create_table(
        "run_metrics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("strategy_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.Text, nullable=False),
        sa.Column("value", sa.Float, nullable=False),
    )
    op.create_index("ix_run_metrics_run_id", "run_metrics", ["run_id"])

    op.create_table(
        "strategy_run_chats",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("strategy_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("context", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_strategy_run_chats_run_id", "strategy_run_chats", ["run_id"])

    # ── model layer ───────────────────────────────────────────────────────────
    op.create_table(
        "ml_models",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("architecture", sa.Text, nullable=False),
        sa.Column("config", JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.Text, nullable=False, server_default="created"),
        sa.Column("artifact_path", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_id", sa.Integer, sa.ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_id", sa.Integer, nullable=False),
        sa.Column("hyperparams", JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("current_epoch", sa.Integer, nullable=False, server_default="0"),
        sa.Column("best_epoch", sa.Integer, nullable=True),
        sa.Column("val_loss", sa.Float, nullable=True),
        sa.Column("eta_seconds", sa.Integer, nullable=True),
        sa.Column("artifact_path", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_training_runs_model_id", "training_runs", ["model_id"])

    op.create_table(
        "training_checkpoints",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("training_run_id", sa.Integer, sa.ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("epoch", sa.Integer, nullable=False),
        sa.Column("metrics", JSONB, nullable=False, server_default="{}"),
        sa.Column("artifact_path", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_training_checkpoints_training_run_id", "training_checkpoints", ["training_run_id"])

    op.create_table(
        "model_validations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_id", sa.Integer, sa.ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("training_run_id", sa.Integer, nullable=True),
        sa.Column("dataset_id", sa.Integer, nullable=False),
        sa.Column("metrics", JSONB, nullable=False, server_default="{}"),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_model_validations_model_id", "model_validations", ["model_id"])

    # ── data layer ────────────────────────────────────────────────────────────
    op.create_table(
        "datasources",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("config", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("datasource_id", sa.Integer, sa.ForeignKey("datasources.id", ondelete="SET NULL"), nullable=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("symbol", sa.Text, nullable=True),
        sa.Column("timeframe", sa.Text, nullable=True),
        sa.Column("from_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("to_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("row_count", sa.Integer, nullable=True),
        sa.Column("artifact_path", sa.Text, nullable=True),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_datasets_datasource_id", "datasets", ["datasource_id"])

    op.create_table(
        "collection_jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("datasource_id", sa.Integer, sa.ForeignKey("datasources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("schedule_cron", sa.Text, nullable=True),
        sa.Column("status", sa.Text, nullable=False, server_default="idle"),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_collection_jobs_datasource_id", "collection_jobs", ["datasource_id"])

    op.create_table(
        "data_characteristics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset_id", sa.Integer, sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("metrics", JSONB, nullable=False, server_default="{}"),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_data_characteristics_dataset_id", "data_characteristics", ["dataset_id"])

    # ── logging ───────────────────────────────────────────────────────────────
    op.create_table(
        "logs",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("level", sa.Text, nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("context", JSONB, nullable=True),
        sa.Column("strategy_run_id", sa.Integer, nullable=True),
        sa.Column("training_run_id", sa.Integer, nullable=True),
        sa.Column("collection_job_id", sa.Integer, nullable=True),
        sa.Column("event_id", sa.BigInteger, nullable=True),
    )
    op.create_index("ix_logs_created_at", "logs", ["created_at"])
    op.create_index("ix_logs_level", "logs", ["level", "created_at"])
    op.create_index("ix_logs_source", "logs", ["source", "created_at"])
    op.create_index("ix_logs_strategy_run_id", "logs", ["strategy_run_id", "created_at"])
    op.create_index("ix_logs_training_run_id", "logs", ["training_run_id", "created_at"])
    op.create_index("ix_logs_collection_job_id", "logs", ["collection_job_id", "created_at"])
    op.create_index("ix_logs_event_id", "logs", ["event_id"])
    op.execute("CREATE INDEX ix_logs_context ON logs USING GIN (context)")


def downgrade() -> None:
    op.drop_table("logs")
    op.drop_table("data_characteristics")
    op.drop_table("collection_jobs")
    op.drop_table("datasets")
    op.drop_table("datasources")
    op.drop_table("model_validations")
    op.drop_table("training_checkpoints")
    op.drop_table("training_runs")
    op.drop_table("ml_models")
    op.drop_table("strategy_run_chats")
    op.drop_table("run_metrics")
    op.drop_table("trades")
    op.drop_table("strategy_events")
    op.drop_table("strategy_runs")
    op.drop_table("strategies")
    op.drop_table("webhook_registrations")
    op.drop_table("api_keys")
