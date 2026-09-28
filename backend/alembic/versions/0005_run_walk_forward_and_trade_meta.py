"""Add walk_forward_ratio to strategy_runs and mae/mfe/phase to trades

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-10
"""

from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("strategy_runs", sa.Column("walk_forward_ratio", sa.Float(), nullable=True))
    op.add_column("trades", sa.Column("phase", sa.Text(), nullable=True))   # "is" | "oos"
    op.add_column("trades", sa.Column("mae",   sa.Float(), nullable=True))
    op.add_column("trades", sa.Column("mfe",   sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("strategy_runs", "walk_forward_ratio")
    op.drop_column("trades", "phase")
    op.drop_column("trades", "mae")
    op.drop_column("trades", "mfe")
