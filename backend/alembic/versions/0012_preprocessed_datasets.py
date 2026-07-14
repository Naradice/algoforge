"""add preprocessed_datasets table + training_runs.preprocessed_dataset_id

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "preprocessed_datasets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("dataset_id", sa.Integer, nullable=False),  # soft FK → datasets.id
        sa.Column("preprocessing", JSONB, nullable=False, server_default="{}"),
        sa.Column("feature_cols", JSONB, nullable=False, server_default='["close"]'),
        sa.Column("normalize", sa.Text, nullable=False, server_default="zscore"),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),  # pending | ready | error
        sa.Column("characteristics", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_preprocessed_datasets_dataset_id", "preprocessed_datasets", ["dataset_id"])

    op.add_column("training_runs", sa.Column("preprocessed_dataset_id", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_runs", "preprocessed_dataset_id")
    op.drop_index("ix_preprocessed_datasets_dataset_id", table_name="preprocessed_datasets")
    op.drop_table("preprocessed_datasets")
