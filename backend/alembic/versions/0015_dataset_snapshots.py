"""add dataset_snapshots table

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A brand-new table — no lock is taken on any existing table (datasets, training_runs, ...),
    # safe to run while collection/training jobs are active.
    op.create_table(
        "dataset_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("row_count", sa.BigInteger(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="local"),
        sa.Column("export_provider", sa.Text(), nullable=True),
        sa.Column("export_ref", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_dataset_snapshots_dataset_id", "dataset_snapshots", ["dataset_id"])


def downgrade() -> None:
    op.drop_index("ix_dataset_snapshots_dataset_id", table_name="dataset_snapshots")
    op.drop_table("dataset_snapshots")
