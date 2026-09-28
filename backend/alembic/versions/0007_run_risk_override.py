"""Add risk_override JSONB column to strategy_runs

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("strategy_runs", sa.Column("risk_override", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("strategy_runs", "risk_override")
