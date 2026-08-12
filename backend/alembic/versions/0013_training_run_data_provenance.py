"""add data_provenance to training_runs

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-12
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("training_runs", sa.Column("data_provenance", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("training_runs", "data_provenance")
