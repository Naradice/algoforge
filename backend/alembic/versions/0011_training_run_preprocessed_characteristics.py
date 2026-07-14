"""add preprocessed_characteristics to training_runs

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("training_runs", sa.Column("preprocessed_characteristics", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("training_runs", "preprocessed_characteristics")
