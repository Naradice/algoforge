"""Add exit_reason column to trades table

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-10
"""

from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("trades", sa.Column("exit_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("trades", "exit_reason")
