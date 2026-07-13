"""add num_params to training_runs

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-14
"""
from alembic import op
import sqlalchemy as sa

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("training_runs", sa.Column("num_params", sa.BigInteger(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_runs", "num_params")
