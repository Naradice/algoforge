"""Add stop_requested to training_runs and training_run_metrics table

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text(
        "ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS "
        "stop_requested BOOLEAN NOT NULL DEFAULT false"
    ))

    op.create_table(
        "training_run_metrics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "training_run_id",
            sa.Integer,
            sa.ForeignKey("training_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("epoch", sa.Integer, nullable=False),
        sa.Column("train_loss", sa.Float, nullable=True),
        sa.Column("val_loss", sa.Float, nullable=True),
        sa.Column("lr", sa.Float, nullable=True),
        sa.Column("metrics", JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        if_not_exists=True,
    )
    op.create_index(
        "ix_training_run_metrics_run_id", "training_run_metrics", ["training_run_id"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_training_run_metrics_run_id", table_name="training_run_metrics")
    op.drop_table("training_run_metrics")
    op.drop_column("training_runs", "stop_requested")
