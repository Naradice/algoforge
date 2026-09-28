"""add starting_capital to strategy_runs

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-03
"""
from alembic import op
import sqlalchemy as sa

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("strategy_runs", sa.Column("starting_capital", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("strategy_runs", "starting_capital")
