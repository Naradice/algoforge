"""Add missing columns and tables omitted from initial migration

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # collection_jobs.enabled
    op.add_column(
        "collection_jobs",
        sa.Column("enabled", sa.Boolean, nullable=False, server_default="true"),
    )

    # collection_job_runs table
    op.create_table(
        "collection_job_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "job_id",
            sa.Integer,
            sa.ForeignKey("collection_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text, nullable=False, server_default="running"),
        sa.Column("datasets_produced", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_index("ix_collection_job_runs_job_id", "collection_job_runs", ["job_id"])

    # training_run_metrics.recorded_at
    op.add_column(
        "training_run_metrics",
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # strategy_runs.equity_curve
    op.add_column(
        "strategy_runs",
        sa.Column("equity_curve", JSONB, nullable=True),
    )

    # strategy_versions table
    op.create_table(
        "strategy_versions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "strategy_id",
            sa.Integer,
            sa.ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("definition", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_strategy_versions_strategy_id", "strategy_versions", ["strategy_id"]
    )

    # webhook_registrations missing columns
    op.add_column(
        "webhook_registrations",
        sa.Column("secret", sa.Text, nullable=False, server_default=""),
    )
    op.add_column(
        "webhook_registrations",
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "webhook_registrations",
        sa.Column("last_status", sa.Integer, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("webhook_registrations", "last_status")
    op.drop_column("webhook_registrations", "last_fired_at")
    op.drop_column("webhook_registrations", "secret")
    op.drop_index("ix_strategy_versions_strategy_id", table_name="strategy_versions")
    op.drop_table("strategy_versions")
    op.drop_column("strategy_runs", "equity_curve")
    op.drop_column("training_run_metrics", "recorded_at")
    op.drop_index("ix_collection_job_runs_job_id", table_name="collection_job_runs")
    op.drop_table("collection_job_runs")
    op.drop_column("collection_jobs", "enabled")
