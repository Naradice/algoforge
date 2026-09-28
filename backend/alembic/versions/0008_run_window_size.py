"""Add window_size column to strategy_runs

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-01
"""

from alembic import op
import sqlalchemy as sa


revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("strategy_runs", sa.Column("window_size", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("strategy_runs", "window_size")
